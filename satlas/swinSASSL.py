import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import random
from torchvision.models import vit_b_16, ViT_B_16_Weights
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from satlasswin import SatlasSwin
import time

class RandomRotation90(nn.Module):
    """Randomly rotate image by 0, 90, 180, or 270 degrees."""
    def forward(self, img):
        angle = torch.randint(0, 4, (1,)).item() * 90
        return TF.rotate(img, angle)

class AddGaussianNoise(nn.Module):
    """Add Gaussian noise to the image tensor."""
    def __init__(self, mean=0.0, std=0.1):
        super().__init__()
        self.mean = mean
        self.std = std
    
    def forward(self, img):
        noise = torch.randn(img.shape, device=img.device, dtype=img.dtype) * self.std + self.mean
        return img + noise

class RandomCropTransform(nn.Module):
    """Random crop with specified output size."""
    def __init__(self, size):
        super().__init__()
        self.crop = T.RandomCrop(size)
    
    def forward(self, img):
        return self.crop(img)
    
class DropRandomChannels(nn.Module):
    """Randomly drop (zero out) channels from the tensor."""
    def __init__(self, p=0.4, min_keep=1):
        super().__init__()
        self.p = p
        self.min_keep = min_keep
    
    def forward(self, img):
        # img shape: [C, H, W] or [B, C, H, W]
        if img.dim() == 3:
            C = img.shape[0]
            n_drop = min(max(int(C * self.p), 0), C - self.min_keep)
            if n_drop > 0:
                channels_to_drop = torch.randperm(C, device=img.device)[:n_drop]
                mask = torch.ones(C, device=img.device, dtype=img.dtype)
                mask[channels_to_drop] = 0
                img = img * mask.view(-1, 1, 1)
        elif img.dim() == 4:
            B, C = img.shape[:2]
            n_drop = min(max(int(C * self.p), 0), C - self.min_keep)
            if n_drop > 0:
                for b in range(B):
                    # ✅ FIX: Create on same device as img
                    channels_to_drop = torch.randperm(C, device=img.device)[:n_drop]
                    mask = torch.ones(C, device=img.device, dtype=img.dtype)
                    mask[channels_to_drop] = 0
                    img[b] = img[b] * mask.view(-1, 1, 1)
        return img

class SwinSASSL(nn.Module):
    def __init__(self,random_crop_size = (32,32),drop_probability = 0.4 ,swin_in_channels = 9,*args, **kwargs,):
        super().__init__(*args, **kwargs)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.teacher = SatlasSwin(channels=swin_in_channels)
        self.student = SatlasSwin(channels=swin_in_channels)

        self.teacher_stack = nn.Sequential(
            nn.Conv2d(1024,256,kernel_size=1,stride=1),
            nn.ReLU(),
            nn.Conv2d(256,32,kernel_size=1,stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8,256)
        )

        self.student_stack = nn.Sequential(
            nn.Conv2d(1024,256,kernel_size=1,stride=1),
            nn.ReLU(),
            nn.Conv2d(256,32,kernel_size=1,stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8,256)
        )

        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

        for param in self.teacher_stack.parameters():
            param.requires_grad = False
        self.teacher_stack.eval()

        self.teacher = self.teacher.to(self.device)
        self.student = self.student.to(self.device)
        self.teacher_stack = self.teacher_stack.to(self.device)
        self.student_stack = self.student_stack.to(self.device)

        self.n_local = 4
        

        self.regular_augment = T.Compose([
            RandomRotation90(),
            AddGaussianNoise(mean=0.0, std=0.1),
            T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)),
        ])

        self.random_crop = T.Compose([RandomCropTransform(size=random_crop_size),T.Resize((256, 256))])
        self.drop_random_channels = T.Compose([DropRandomChannels( p= drop_probability, min_keep=1)])
        self.resize_img = T.Resize((256, 256))

        self.flatten = nn.Flatten()

    def generate_views(self, image):
        # Global view (original size)

        if image.device != self.device:
            image = image.to(self.device)
    
        image = self.resize_img(image)
        image = self.regular_augment(image)
        global_view = image
    
        # Generate all local views at once (more efficient)
        local_views = [self.random_crop(image) for _ in range(self.n_local)]
    
        spectral_view = self.drop_random_channels(image)
    
        return global_view, local_views, spectral_view


    
    def forward(self, image ):

        # t0 = time.time()
        global_view, local_views, spectral_view = self.generate_views(image)
        # print(f"  generate_views: {time.time()-t0:.3f}s")
            
        # Teacher processes only global view
        # t0 = time.time()
        with torch.no_grad(): 
            teacher_output = self.teacher(global_view)[3]
            teacher_output = self.teacher_stack(teacher_output)
        # print(f"  teacher: {time.time()-t0:.3f}s")

        # Student processes local and spectral views
        # t0 = time.time()
        student_outputs = []
        for local_view in local_views:
            student_outputs.append(self.student_stack(self.student(local_view)[3]))
        student_outputs.append(self.student_stack(self.student(spectral_view)[3]))    
        # print(f"  student_views: {time.time()-t0:.3f}s")
        
        return teacher_output, student_outputs
    
    @torch.no_grad()
    def update_teacher(self, momentum=0.996):
        """Update teacher parameters with EMA of student parameters."""
        for teacher_param, student_param in zip(
            self.teacher.parameters(),
            self.student.parameters()
        ):
            # In-place operations (more efficient and ensures device consistency)
            teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1 - momentum)

        for teacher_param, student_param in zip(
        self.teacher_stack.parameters(),
        self.student_stack.parameters()
        ):
            teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1 - momentum)


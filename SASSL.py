import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
import torchvision.transforms.functional as TF
import torchvision.transforms as T

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
                # ❌ PROBLEM: randperm and mask created on CPU by default
                # ✅ FIX: Create on same device as img
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

class SASSL(nn.Module):
    def __init__(self,random_crop_size = (32,32),drop_probability = 0.4 ,*args, **kwargs,):
        super().__init__(*args, **kwargs)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        pretrained_model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

        # Create your models with 12 channels
        self.teacher = vit_b_16(weights=None)
        self.student = vit_b_16(weights=None)

        # 1. Modify first layer FIRST
        self.teacher.conv_proj = nn.Conv2d(12, 768, kernel_size=(16, 16), stride=(16, 16))
        self.student.conv_proj = nn.Conv2d(12, 768, kernel_size=(16, 16), stride=(16, 16))

        # 2. Remove classification heads BEFORE loading weights
        self.teacher.heads = nn.Identity()
        self.student.heads = nn.Identity()

        # 3. NOW copy pretrained weights (excluding conv_proj and heads)
        pretrained_dict = pretrained_model.state_dict()
        teacher_dict = self.teacher.state_dict()

        # Filter out conv_proj AND heads weights
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                           if k in teacher_dict and 'conv_proj' not in k and 'heads' not in k}
        
        self.teacher.load_state_dict(pretrained_dict, strict=False)
        self.student.load_state_dict(pretrained_dict, strict=False)

        self.teacher = self.teacher.to(self.device)
        self.student = self.student.to(self.device)

        self.n_local = 4
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

        self.regular_augment = T.Compose([
            RandomRotation90(),
            AddGaussianNoise(mean=0.0, std=0.1),
            T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)),
        ])

        self.random_crop = T.Compose([RandomCropTransform(size=random_crop_size),T.Resize((224, 224))])
        self.drop_random_channels = T.Compose([DropRandomChannels( p= drop_probability, min_keep=1)])
        self.resize_img = T.Resize((224, 224))

    def generate_views(self, image):
        # Global view (original size)

        image = image.to(self.device)
        image = self.resize_img(image)
        image = self.regular_augment(image)
        global_view = image
        
        # Local views (smaller crops)
        local_views = [self.random_crop(image) for _ in range(self.n_local)]

        # Spectral-aware view
        spectral_view = self.drop_random_channels(image)
        
        return global_view, local_views, spectral_view
    
    def forward(self, image):
        global_view, local_views, spectral_view = self.generate_views(image)
            
        # Teacher processes only global view
        with torch.no_grad(): 
            teacher_output = self.teacher(global_view)
            
        # Student processes local and spectral views
        student_outputs = []
        for local_view in local_views:
            student_outputs.append(self.student(local_view))
        student_outputs.append(self.student(spectral_view))    
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


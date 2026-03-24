import os
import cv2
import glob
import tifffile
import numpy as np
from tqdm import tqdm
from scipy.ndimage import zoom

import random
from torch.utils.data import Dataset, DataLoader, random_split

class ClassificationDataset(Dataset):
    def __init__(self, root_dir, target_size=None):
        self.root_dir = root_dir
        self.target_size = target_size
       
        # Band names in order (12 bands total)
        self.bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
        
        self.band_resolutions = {
            'B1': 60, 'B2': 10, 'B3': 10, 'B4': 10,
            'B5': 20, 'B6': 20, 'B7': 20, 'B8': 10,
            'B8A': 20, 'B9': 60, 'B11': 20, 'B12': 20
        }

        idx_from_label = {'RPH':0, 'Blast':1, 'Rust':2, 'Aphid':3}
        
        print("Loading dataset into memory...")

        self.samples = []
        self.samples_by_class = {'RPH':[], 'Blast':[], 'Rust':[], 'Aphid':[]}

        for label in tqdm(os.listdir(root_dir)):
            # skip evaluation set
            if label == 'evaluation':
                continue

            region_folder = os.path.join(root_dir, label)
            for timestamp_folder in glob.glob(os.path.join(region_folder, '*')):
                # Check if all bands exist
                band_paths = {band: os.path.join(timestamp_folder, f'{band}.tif') 
                                for band in self.bands}
                if all(os.path.exists(p) for p in band_paths.values()):
                    img = self.load_multispectral_image(band_paths)
                    img = self.normalize_satlas(img)
                    self.samples.append([img, idx_from_label[label]])
                    self.samples_by_class[label].append([img, idx_from_label[label]])
        
        print(f"Found {len(self.samples)} samples with all 12 bands")
    
    def __len__(self):
        return len(self.samples)
    
    def resize_band(self, band_data, target_shape):
        if band_data.shape == target_shape:
            return band_data
        
        resized = cv2.resize(
            band_data, 
            (target_shape[1], target_shape[0]),  # OpenCV uses (W, H)
            interpolation=cv2.INTER_LINEAR
        )
        return resized
    
    def load_multispectral_image(self, band_paths):
        bands_data = []
        shapes = []
        
        loaded_bands = {}
        for band in self.bands:
            img = tifffile.imread(band_paths[band])
            loaded_bands[band] = img
            shapes.append(img.shape)
        
        if self.target_size is not None:
            target_shape = self.target_size
        else:
            reference_band = loaded_bands['B4']  # 10m band
            target_shape = reference_band.shape
        
        # Second pass: resize all bands to target shape
        for band in self.bands:
            img = loaded_bands[band]
            
            # Resize if needed
            if img.shape != target_shape:
                img = self.resize_band(img, target_shape)
            
            bands_data.append(img)
        
        # Stack along channel dimension: (12, H, W)
        multi_band = np.stack(bands_data, axis=0)
        return multi_band 
    
    def normalize_sentinel2(self, img):
        # Clip extreme values and normalize
        img = np.clip(img, 0, 10000)
        img = img.astype(np.float32) / 10000.0
        return img  
    
    def normalize_satlas(self,img):
        img = np.clip(img, 0, 8160)
        img = img.astype(np.float32) / 8160
        return img
    
    def __getitem__(self, idx):
        img, label = self.samples[idx]
        c9 = img[(3,2,1,4,5,6,7,10,11),:,:]
        c12 = img[(3,2,1,4,5,6,7,10,11,0,8,9),:,:]
        return {
            'c9': c9,      
            'c12': c12,
            'label': label
        }
    
    def __len__(self):
        return len(self.samples)

class ClassificationTrainDataset(Dataset):
    def __init__(self, backing_dataset, train_samples_per_class, val_samples_per_class, seed):
        self.backing_dataset = backing_dataset
        self.min_train_samples_per_class = train_samples_per_class
        self.min_val_samples_per_class = val_samples_per_class

        self.samples = []
        self.samples_by_class = {'RPH':[], 'Blast':[], 'Rust':[], 'Aphid':[]}

        # see if the backing dataset has enough samples
        for label, class_samples in self.backing_dataset.samples_by_class.items():
            if len(class_samples) < val_samples_per_class:
                raise ValueError(f"Backing dataset has insufficient samples for class {label}")

            # random partition for train set
            random.seed(seed)
            # make a copy to avoid modifying the backing dataset
            class_samples = class_samples.copy()
            random.shuffle(class_samples)
            class_samples = class_samples[val_samples_per_class:][:train_samples_per_class]
            
            if len(class_samples) < train_samples_per_class:
                # must augment train set for this class
                print(f"Augmenting train set for class {label}")
                num_to_augment = train_samples_per_class - len(class_samples)
                augmented_samples = []
                for _ in range(num_to_augment):
                    sample = random.choice(class_samples)
                    augmented_samples.append([self.augment_sample(sample[0]), sample[1]])
                class_samples = class_samples + augmented_samples
            
            self.samples.extend(class_samples)
            self.samples_by_class[label] = class_samples
        
    def augment_sample(self, sample):
        """Apply random augmentation: rotation (0, 90, 180, 270) and/or flip"""
        # sample shape: (C, H, W)
        aug_sample = sample.copy()
        
        # Random rotation (0, 90, 180, 270 degrees)
        k = np.random.randint(0, 4)  # 0, 1, 2, or 3 rotations of 90 degrees
        if k > 0:
            # Rotate in the spatial dimensions (H, W)
            aug_sample = np.rot90(aug_sample, k=k, axes=(1, 2))
        
        # Random horizontal flip
        if np.random.rand() > 0.5:
            aug_sample = np.flip(aug_sample, axis=2)  # Flip along width
        
        # Random vertical flip
        if np.random.rand() > 0.5:
            aug_sample = np.flip(aug_sample, axis=1)  # Flip along height
        
        return aug_sample.copy()  # Ensure contiguous array

    def __getitem__(self, idx):
        img, label = self.samples[idx]
        c9 = img[(3,2,1,4,5,6,7,10,11),:,:]
        c12 = img[(3,2,1,4,5,6,7,10,11,0,8,9),:,:]
        return {
            'c9': c9,
            'c12': c12,
            'label': label
        }
    
    def __len__(self):
        return len(self.samples)

class ClassificationValDataset(Dataset):
    def __init__(self, backing_dataset, val_samples_per_class, seed):
        self.backing_dataset = backing_dataset
        self.min_val_samples_per_class = val_samples_per_class

        self.samples = []
        self.samples_by_class = {'RPH':[], 'Blast':[], 'Rust':[], 'Aphid':[]}

        # see if the backing dataset has enough samples
        for label, class_samples in self.backing_dataset.samples_by_class.items():
            if len(class_samples) < val_samples_per_class:
                raise ValueError(f"Backing dataset has insufficient samples for class {label}")

            # random partition for val set
            random.seed(seed)
            # make a copy to avoid modifying the backing dataset
            class_samples = class_samples.copy()
            random.shuffle(class_samples)
            class_samples = class_samples[:val_samples_per_class]
            
            self.samples.extend(class_samples)
            self.samples_by_class[label] = class_samples
        
    def __getitem__(self, idx):
        img, label = self.samples[idx]
        c9 = img[(3,2,1,4,5,6,7,10,11),:,:]
        c12 = img[(3,2,1,4,5,6,7,10,11,0,8,9),:,:]
        return {
            'c9': c9,
            'c12': c12,
            'label': label
        }
    
    def __len__(self):
        return len(self.samples)
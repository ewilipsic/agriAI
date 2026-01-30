# dataset.py (UPDATED)
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile
import json
from scipy.ndimage import zoom
import cv2
from tqdm import tqdm

class Sentinel2InpaintingDataset(Dataset):
    """Dataset for Sentinel-2 multi-spectral inpainting"""
    
    def __init__(self, root_dir, mask_type='random', augment=False, target_size=None,limit_samples = None):
        """
        Args:
            root_dir: Path to s2a folder (e.g., 'D:/s2a')
            mask_type: 'random', 'center', or 'irregular'
            augment: Apply random flips/rotations
            target_size: (H, W) to resize all images, or None to use 10m resolution
        """
        self.root_dir = root_dir
        self.mask_type = mask_type
        self.augment = augment
        self.target_size = target_size
        
        # Band names in order (12 bands total)
        self.bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
        
        # Band resolutions (in meters)
        self.band_resolutions = {
            'B1': 60, 'B2': 10, 'B3': 10, 'B4': 10,
            'B5': 20, 'B6': 20, 'B7': 20, 'B8': 10,
            'B8A': 20, 'B9': 60, 'B11': 20, 'B12': 20
        }
        
        # Find all timestamped folders
        sample_count = 0
        self.samples = []
        flag = 0
        for region_folder in tqdm(glob.glob(os.path.join(root_dir, '*'))):
            if flag: break
            if os.path.isdir(region_folder):
                for timestamp_folder in glob.glob(os.path.join(region_folder, '*')):
                    if flag: break
                    if os.path.isdir(timestamp_folder):
                        # Check if all bands exist
                        band_paths = {band: os.path.join(timestamp_folder, f'{band}.tif') 
                                     for band in self.bands}
                        if all(os.path.exists(p) for p in band_paths.values()):
                            sample_count += 1
                            if(limit_samples != None and sample_count >= limit_samples): flag = 1
                            self.samples.append(band_paths)
        
        print(f"Found {len(self.samples)} samples with all 12 bands")
     
    def __len__(self):
        return len(self.samples)
    
    def resize_band(self, band_data, target_shape):
        """Resize a band to target shape using bilinear interpolation"""
        if band_data.shape == target_shape:
            return band_data
        
        # Use OpenCV for faster resizing
        resized = cv2.resize(
            band_data, 
            (target_shape[1], target_shape[0]),  # OpenCV uses (W, H)
            interpolation=cv2.INTER_LINEAR
        )
        return resized
    
    def load_multispectral_image(self, band_paths):
        """Load all 12 bands and stack them with consistent resolution"""
        bands_data = []
        shapes = []
        
        # First pass: load all bands and record shapes
        loaded_bands = {}
        for band in self.bands:
            img = tifffile.imread(band_paths[band])
            loaded_bands[band] = img
            shapes.append(img.shape)
        
        # Determine target shape
        if self.target_size is not None:
            target_shape = self.target_size
        else:
            # Use the shape of 10m bands (B2, B3, B4, B8) as reference
            # These are typically the largest
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
        """
        Normalize Sentinel-2 data to [0, 1]
        Sentinel-2 L1C has typical range 0-10000 (reflectance * 10000)
        """
        # Clip extreme values and normalize
        img = np.clip(img, 0, 10000)
        img = img.astype(np.float32) / 10000.0
        return img
    
    def create_random_mask(self, shape, num_masks=None):
        """Create random rectangular mask(s)"""
        C, H, W = shape
        mask = np.ones((C, H, W), dtype=np.float32)
        
        if num_masks is None:
            num_masks = np.random.randint(1, 4)
        
        for _ in range(num_masks):
            mask_h = np.random.randint(32, min(128, H // 2))
            mask_w = np.random.randint(32, min(128, W // 2))
            
            top = np.random.randint(0, H - mask_h)
            left = np.random.randint(0, W - mask_w)
            
            # Mask all channels
            mask[:, top:top+mask_h, left:left+mask_w] = 0
        
        return mask
    
    def create_center_mask(self, shape, mask_ratio=0.25):
        """Create centered square mask"""
        C, H, W = shape
        mask = np.ones((C, H, W), dtype=np.float32)
        
        mask_size = int(min(H, W) * mask_ratio)
        top = (H - mask_size) // 2
        left = (W - mask_size) // 2
        
        mask[:, top:top+mask_size, left:left+mask_size] = 0
        return mask
    
    def augment_data(self, img):
        """Random flips and rotations"""
        # Random horizontal flip
        if np.random.random() > 0.5:
            img = np.flip(img, axis=2).copy()
        
        # Random vertical flip
        if np.random.random() > 0.5:
            img = np.flip(img, axis=1).copy()
        
        # Random 90 degree rotations
        k = np.random.randint(0, 4)
        if k > 0:
            img = np.rot90(img, k, axes=(1, 2)).copy()
        
        return img
    
    def __getitem__(self, idx):
        # Load all 12 bands
        band_paths = self.samples[idx]
        img = self.load_multispectral_image(band_paths)
        
        # Normalize
        img = self.normalize_sentinel2(img)
        
        # Augmentation
        if self.augment:
            img = self.augment_data(img)
        
        # Create mask
        if self.mask_type == 'random':
            mask = self.create_random_mask(img.shape)
        elif self.mask_type == 'center':
            mask = self.create_center_mask(img.shape)
        else:  # irregular
            mask = self.create_center_mask(img.shape)  # fallback
        
        # Apply mask
        masked_img = img * mask
        
        # Convert to torch tensors
        img = torch.from_numpy(img)
        masked_img = torch.from_numpy(masked_img)
        mask = torch.from_numpy(mask)
        
        return {
            'original': img,          # (12, H, W)
            'masked': masked_img,     # (12, H, W)
            'mask': mask              # (12, H, W)
        }

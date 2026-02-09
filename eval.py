from torch.utils.data import Dataset
import os
import cv2
import tifffile
import numpy as np

class EvalDataset(Dataset):
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
        
        self.samples = []
        eval_dir = os.path.join(root_dir, 'evaluation')
        if os.path.exists(eval_dir):
            self.timestamps = os.listdir(eval_dir)
            for folder_name in self.timestamps:
                timestamp_folder = os.path.join(eval_dir, folder_name)
                if os.path.isdir(timestamp_folder):
                    # Check if all bands exist
                    band_paths = {band: os.path.join(timestamp_folder, f'{band}.tif') 
                                 for band in self.bands}
                    if all(os.path.exists(p) for p in band_paths.values()):
                        self.samples.append(band_paths)
        
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
        """
        Normalize Sentinel-2 data to [0, 1]
        Sentinel-2 L1C has typical range 0-10000 (reflectance * 10000)
        """
        # Clip extreme values and normalize
        img = np.clip(img, 0, 10000)
        img = img.astype(np.float32) / 10000.0
        return img  
    
    def normalize_satlas(self,img):
        img = np.clip(img, 0, 8160)
        img = img.astype(np.float32) / 8160
        return img
    
    def __getitem__(self, idx):
        band_paths = self.samples[idx]
        img = self.load_multispectral_image(band_paths)
        img = self.normalize_satlas(img)
        self.bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
        c9 = img[(3, 2, 1, 4, 5, 6, 7, 10, 11), :, :]
        c12 = img[(3, 2, 1, 4, 5, 6, 7, 10, 11, 0, 8, 9), :, :]
        return {
            'c9': c9,      
            'c12': c12,
        }

    def write_csv(self, predictions, output_dir):
        """
        Write submission.csv based on predictions.
        
        Args:
            predictions: List/Array of predicted class indices (0-3) or labels.
            output_dir: Directory to save submission.csv
        """
        class_map = {0: 'RPH', 1: 'Blast', 2: 'Rust', 3: 'Aphid'}
        output_path = os.path.join(output_dir, 'submission.csv')
        
        print(f"Writing submission to {output_path}...")
        
        with open(output_path, 'w') as f:
            f.write("Id,Category\n")
            
            for timestamp, pred in zip(self.timestamps, predictions):
                f.write(f"{timestamp},{class_map[pred]}\n")

        print(f"Successfully wrote {len(predictions)} predictions.")

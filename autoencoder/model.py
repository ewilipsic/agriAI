# model.py (FULLY FIXED VERSION)
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvAutoEncoder(nn.Module):
    """Convolutional AutoEncoder with guaranteed dimension preservation"""
    
    def __init__(self, latent_dim=256, input_size=(512, 512)):
        super(ConvAutoEncoder, self).__init__()
        self.input_size = input_size
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=3, stride=2, padding=1),  # H/2, W/2
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # H/4, W/4
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # H/8, W/8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # H/16, W/16
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.enc5 = nn.Sequential(
            nn.Conv2d(512, latent_dim, kernel_size=3, stride=2, padding=1),  # H/32, W/32
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True),
        )
        
        # Decoder with output_padding to ensure exact size matching
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64, 12, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
     
    def encode(self, x):
        """Get latent representation"""
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        z = self.enc5(x)
        return z
    
    def decode(self, z):
        """Reconstruct from latent"""
        x = self.dec5(z)
        x = self.dec4(x)
        x = self.dec3(x)
        x = self.dec2(x)
        x = self.dec1(x)
        return x
    
    def forward(self, x):
        input_size = x.shape[2:]
        z = self.encode(x)
        x_recon = self.decode(z)
        
        # Force exact size match with input (in case of rounding issues)
        if x_recon.shape[2:] != input_size:
            x_recon = F.interpolate(x_recon, size=input_size, mode='bilinear', align_corners=False)
        
        return x_recon, z


class UNetAutoEncoder(nn.Module):
    """U-Net style AutoEncoder with skip connections - FULLY FIXED"""
    
    def __init__(self, latent_dim=256):
        super(UNetAutoEncoder, self).__init__()
        
        # Encoder with skip outputs
        self.enc1 = self._conv_block(12, 64)
        self.pool1 = nn.MaxPool2d(2, return_indices=False)
        
        self.enc2 = self._conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2, return_indices=False)
        
        self.enc3 = self._conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2, return_indices=False)
        
        self.enc4 = self._conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(2, return_indices=False)
        
        # Bottleneck
        self.bottleneck = self._conv_block(512, latent_dim)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(latent_dim, 512, 2, stride=2)
        self.dec4 = self._conv_block(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self._conv_block(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self._conv_block(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self._conv_block(128, 64)
        
        self.final = nn.Conv2d(64, 12, 1)
        self.sigmoid = nn.Sigmoid()
    
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def _match_size_and_concat(self, upsampled, skip):
        """Match dimensions and concatenate"""
        # Get target size from upsampled
        target_h, target_w = upsampled.shape[2], upsampled.shape[3]
        
        # Resize skip connection to match
        if skip.shape[2] != target_h or skip.shape[3] != target_w:
            skip = F.interpolate(skip, size=(target_h, target_w), mode='bilinear', align_corners=False)
        
        return torch.cat([upsampled, skip], dim=1)
    
    def encode(self, x):
        """Get latent representation"""
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        x4 = self.enc4(self.pool3(x3))
        z = self.bottleneck(self.pool4(x4))
        return z
    
    def forward(self, x):
        input_size = x.shape[2:]
        
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        x4 = self.enc4(self.pool3(x3))
        z = self.bottleneck(self.pool4(x4))
        
        # Decoder with skip connections
        x = self.up4(z)
        x = self._match_size_and_concat(x, x4)
        x = self.dec4(x)
        
        x = self.up3(x)
        x = self._match_size_and_concat(x, x3)
        x = self.dec3(x)
        
        x = self.up2(x)
        x = self._match_size_and_concat(x, x2)
        x = self.dec2(x)
        
        x = self.up1(x)
        x = self._match_size_and_concat(x, x1)
        x = self.dec1(x)
        
        x_recon = self.sigmoid(self.final(x))
        
        # Force exact size match with input
        if x_recon.shape[2:] != input_size:
            x_recon = F.interpolate(x_recon, size=input_size, mode='bilinear', align_corners=False)
        
        return x_recon, z

import torch
import torch.nn as nn
import satlaspretrain_models
import torch

class SatlasSwin(nn.Module):
    def __init__(self, channels = 9):
        super().__init__()
        def load_sentinel2_model(checkpoint_path, device='cuda'):
            from satlaspretrain_models import Model

            checkpoint = torch.load(checkpoint_path, map_location=device , weights_only=False)
            arch = checkpoint['architecture']

            model = Model(
                num_channels=arch['num_channels'],
                multi_image=arch['multi_image'],
                backbone=arch['backbone'],
                fpn=False,
                head=None,
                num_categories=None,
                weights=None
            )

            model.load_state_dict(checkpoint['state_dict'])
            return model
        
        model = load_sentinel2_model('sentinel2_swinb_portable.pth', device= 'cuda' if torch.cuda.is_available() else "cpu")
        
        if channels == 12:
            first_conv = model.backbone.backbone.features[0][0]

            new_conv = nn.Conv2d(
                in_channels=12,
                out_channels=first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride
            )

            with torch.no_grad():
                # Copy first 9 channels as-is
                new_conv.weight[:, :9, :, :] = first_conv.weight.clone()

                # Initialize last 3 channels by averaging all original channels
                avg_weight = first_conv.weight.mean(dim=1, keepdim=True)
                new_conv.weight[:, 9:, :, :] = avg_weight.repeat(1, 3, 1, 1)

                if first_conv.bias is not None:
                    new_conv.bias.copy_(first_conv.bias)

            model.backbone.backbone.features[0][0] = new_conv

        self.enc = model
       

    def forward(self,x):
        return self.enc(x)
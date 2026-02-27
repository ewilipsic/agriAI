"""Model definition for pseudo-labelling and confidence analysis.

Contains the 4-class divider (ClassificationModel) extracted from the
ensemble trained in satlas/classification_satlas_baseline_9c.ipynb.

The full ensemble is too large for backprop (each step runs through
the Swin encoder multiple times), so we only use the divider.
The encoder is fully unfrozen so all weights are updated during training.

Import path used by confidence_analysis.py:
    from model_spec import build_model
"""

import sys
import os

# Add the satlas directory to the path so SatlasSwin can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'satlas'))

import torch
import torch.nn as nn
from satlasswin import SatlasSwin


# ---------------------------------------------------------------------------
# 4-class divider (the only model we need)
# ---------------------------------------------------------------------------

class ClassificationModel(nn.Module):
    """4-class classifier (RPH / Blast / Rust / Aphid).

    Architecture: SatlasSwin encoder (fully trainable) → Conv2d 1024→256
    → Linear head → 4 logits.
    """

    def __init__(self):
        super().__init__()
        self.encoder = SatlasSwin(channels=9)


        self.stack = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 256 * 8),
            nn.LeakyReLU(),
            nn.Linear(256 * 8, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 4),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x[3]
        return self.stack(x)


# ---------------------------------------------------------------------------
# Factory used by confidence_analysis.py / main.py
# ---------------------------------------------------------------------------

def build_model(checkpoint_path: str, device: str = 'auto') -> nn.Module:
    """Instantiate a ClassificationModel and load the ``div`` sub-state-dict
    from an *ensemble* checkpoint.

    The ensemble checkpoint stores keys like ``div.encoder.…``,
    ``div.stack.…``, ``m1.…``, ``m2.…``, etc.  We strip the ``div.``
    prefix and load only those weights into a standalone
    ClassificationModel.

    Args:
        checkpoint_path: Path to the ensemble checkpoint (.pth).
                         Must contain ``'model_state_dict'``.
        device:          ``'cuda'``, ``'cpu'``, or ``'auto'``.

    Returns:
        The loaded ClassificationModel in eval mode.
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = ClassificationModel()

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ensemble_sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt

    # Extract only the div.* keys and strip the prefix
    prefix = 'div.'
    div_sd = {
        k[len(prefix):]: v
        for k, v in ensemble_sd.items()
        if k.startswith(prefix)
    }

    if not div_sd:
        raise RuntimeError(
            f'No keys starting with "{prefix}" found in checkpoint. '
            f'Available key prefixes: '
            f'{sorted(set(k.split(".")[0] for k in ensemble_sd))}'
        )

    model.load_state_dict(div_sd)
    model.to(device)
    model.eval()
    return model

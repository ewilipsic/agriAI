"""Analyse class-wise confidence distributions on unlabelled data.

Usage (example):
    python confidence_analysis.py \
        --unlabelled_dir ../s2a \
        --checkpoint ../classification_checkpoints/best_classifier.pth \
        --label_minus 0 \
        --batch_size 16 \
        --target_size 256 \
        --out_dir confidence_report

Produces:
  - A histogram of confidence scores per predicted class.
  - Printed percentile statistics so you can pick a good threshold.
"""

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Adjust these imports to match your project layout
from s2a import Sentinel2InpaintingDataset


# ------------------------------------------------------------------
# Core analysis
# ------------------------------------------------------------------

def collect_confidences(model, dataloader, label_minus=0):
    """Run inference and return per-sample (predicted_class, confidence).

    Returns:
        predictions:  np.ndarray of shape (N,) — predicted class indices
                      in model-output space (i.e. *before* adding label_minus back).
        confidences:  np.ndarray of shape (N,) — softmax confidence for the
                      predicted class.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    model.to(device)

    all_preds = []
    all_confs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Collecting confidences'):
            images = batch['c9'].to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            confidence, predicted = probs.max(dim=1)

            all_preds.extend(predicted.cpu().numpy())
            all_confs.extend(confidence.cpu().numpy())

    return np.array(all_preds), np.array(all_confs)


def print_statistics(predictions, confidences, num_classes, class_names=None):
    """Print per-class confidence percentiles to stdout."""
    if class_names is None:
        class_names = [f'Class {i}' for i in range(num_classes)]

    percentiles = [10, 25, 50, 75, 90, 95, 99]

    print(f'\n{"="*70}')
    print('Per-class confidence distribution (predicted classes)')
    print(f'{"="*70}')
    print(f'Total unlabelled samples: {len(predictions)}')

    header = f'{"Class":<12} {"Count":>6}  ' + '  '.join(f'P{p:02d}' for p in percentiles)
    print(f'\n{header}')
    print('-' * len(header))

    for c in range(num_classes):
        mask = predictions == c
        count = mask.sum()
        if count == 0:
            vals = '  '.join('  - ' for _ in percentiles)
            print(f'{class_names[c]:<12} {count:>6}  {vals}')
            continue

        confs = confidences[mask]
        pvals = np.percentile(confs, percentiles)
        vals = '  '.join(f'{v:.2f}' for v in pvals)
        print(f'{class_names[c]:<12} {count:>6}  {vals}')

    # Overall
    pvals = np.percentile(confidences, percentiles)
    vals = '  '.join(f'{v:.2f}' for v in pvals)
    print('-' * len(header))
    print(f'{"Overall":<12} {len(predictions):>6}  {vals}')
    print()


def plot_histograms(predictions, confidences, num_classes, class_names=None,
                    out_path=None):
    """Save a figure with per-class confidence histograms."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not available — skipping histogram plot.')
        return

    if class_names is None:
        class_names = [f'Class {i}' for i in range(num_classes)]

    fig, axes = plt.subplots(1, num_classes + 1, figsize=(5 * (num_classes + 1), 4),
                             sharey=True)

    for c in range(num_classes):
        ax = axes[c]
        mask = predictions == c
        confs = confidences[mask]
        ax.hist(confs, bins=50, range=(0, 1), edgecolor='black', alpha=0.7)
        ax.set_title(f'{class_names[c]} (n={mask.sum()})')
        ax.set_xlabel('Confidence')
        if c == 0:
            ax.set_ylabel('Count')
        ax.axvline(np.median(confs) if len(confs) > 0 else 0.5,
                   color='red', linestyle='--', label='median')
        ax.legend(fontsize=8)

    # Overall
    ax = axes[-1]
    ax.hist(confidences, bins=50, range=(0, 1), edgecolor='black', alpha=0.7,
            color='gray')
    ax.set_title(f'Overall (n={len(confidences)})')
    ax.set_xlabel('Confidence')
    ax.axvline(np.median(confidences), color='red', linestyle='--', label='median')
    ax.legend(fontsize=8)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f'Histogram saved to {out_path}')
    else:
        plt.show()
    plt.close()


# ------------------------------------------------------------------
# CLI entry-point
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Analyse class-wise confidence on unlabelled data.')
    parser.add_argument('--unlabelled_dir', type=str, required=True,
                        help='Root directory for unlabelled Sentinel-2 data (s2a format)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--label_minus', type=int, default=0)
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--class_names', nargs='+', default=None,
                        help='Optional class names, e.g. --class_names RPH Blast Rust Aphid')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--target_size', type=int, default=256,
                        help='Spatial size to resize images to (square)')
    parser.add_argument('--limit_samples', type=int, default=None,
                        help='Limit number of unlabelled samples (for quick testing)')
    parser.add_argument('--out_dir', type=str, default='confidence_report',
                        help='Directory to save histogram image')
    args = parser.parse_args()

    # ---- dataset ----
    target = (args.target_size, args.target_size) if args.target_size else None
    dataset = Sentinel2InpaintingDataset(
        root_dir=args.unlabelled_dir,
        format='satlas',
        target_size=target,
        limit_samples=args.limit_samples,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ---- model ----
    from model_spec import build_model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_model(args.checkpoint, device=device)

    # ---- analyse ----
    predictions, confidences = collect_confidences(model, loader, label_minus=args.label_minus)

    num_classes = args.num_classes - args.label_minus
    names = args.class_names
    if names is None:
        all_names = ['RPH', 'Blast', 'Rust', 'Aphid']
        names = all_names[args.label_minus:args.label_minus + num_classes]

    print_statistics(predictions, confidences, num_classes, class_names=names)

    os.makedirs(args.out_dir, exist_ok=True)
    plot_histograms(predictions, confidences, num_classes, class_names=names,
                    out_path=os.path.join(args.out_dir, 'confidence_histograms.png'))


if __name__ == '__main__':
    main()

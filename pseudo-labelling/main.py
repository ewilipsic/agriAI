# Pseudo-labelling / self-training pipeline
#
# 1. Evaluate a base model on the labelled data to find its class-bias.
# 2. Use the model to predict labels for unlabelled data.
# 3. Keep predictions above a confidence threshold,
#    sampling proportionally more from the classes with low recall.
# 4. Retrain the model on labelled + pseudo-labelled data.
# 5. Iterate until num_iterations is reached.
# 6. Return the model and per-iteration evaluation metrics.

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm
from classification_datasets import ClassificationDataset
from s2a import Sentinel2InpaintingDataset
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Lightweight dataset for pseudo-labelled samples
# ---------------------------------------------------------------------------

class PseudoLabelledDataset(Dataset):
    """Wraps a list of (image_tensor, label) pairs as a PyTorch Dataset."""

    def __init__(self, images, labels):
        """
        Args:
            images: list/array of numpy images, each shaped (C, H, W).
            labels: list/array of integer class labels (already offset-corrected).
        """
        assert len(images) == len(labels)
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]                     # numpy (C, H, W)
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img).float()
        return {
            'c9': img,
            'label': int(self.labels[idx]),
        }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, dataloader, label_minus=0):
    """Run the model on *dataloader* and return a confusion matrix.

    The labels in the dataloader are shifted by ``label_minus`` so that
    they match the model's output range (e.g. if the model only
    distinguishes the last 2 classes, ``label_minus=2``).
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.eval()
    model.to(device)

    predictions = []
    ground_truth_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            images = batch['c9'].to(device)
            labels = batch['label'].to(device)
            labels = labels - label_minus

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            predictions.extend(predicted.cpu().numpy())
            ground_truth_labels.extend(labels.cpu().numpy())

    predictions = np.array(predictions)
    ground_truth_labels = np.array(ground_truth_labels)

    return confusion_matrix(ground_truth_labels, predictions)


# ---------------------------------------------------------------------------
# Pseudo-label generation
# ---------------------------------------------------------------------------

def generate_pseudo_labels(model, unlabelled_loader, label_minus=0):
    """Predict on unlabelled data and return per-sample predictions + confidences.

    Returns:
        all_images:       list of numpy arrays (C, H, W) – the c9 images.
        all_predictions:  np.ndarray of predicted class indices (already offset
                          back by ``label_minus`` so they live in the *original*
                          label space – handy for combining with labelled data).
        all_confidences:  np.ndarray of softmax confidence for the predicted class.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.eval()
    model.to(device)

    all_images = []
    all_predictions = []
    all_confidences = []

    with torch.no_grad():
        for batch in tqdm(unlabelled_loader, desc='Generating pseudo-labels'):
            images = batch['c9'].to(device)

            outputs = model(images)                       # raw logits
            probs = torch.softmax(outputs, dim=1)         # (B, num_classes)
            confidence, predicted = torch.max(probs, 1)   # (B,), (B,)

            # Store images as numpy on CPU so we can build a dataset later
            all_images.extend(images.cpu().numpy())
            # Shift predictions back to the original label space
            all_predictions.extend((predicted + label_minus).cpu().numpy())
            all_confidences.extend(confidence.cpu().numpy())

    return all_images, np.array(all_predictions), np.array(all_confidences)


# ---------------------------------------------------------------------------
# Single-epoch training helper
# ---------------------------------------------------------------------------

def _train_one_epoch(model, dataloader, criterion, optimizer, device, label_minus=0):
    """Train the model for one epoch and return (loss, accuracy)."""
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        images = batch['c9'].to(device)
        labels = batch['label'].to(device) - label_minus

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix(loss=f'{loss.item():.4f}',
                         acc=f'{100.0 * correct / total:.2f}%')

    epoch_loss = running_loss / max(len(dataloader), 1)
    epoch_acc = 100.0 * correct / max(total, 1)
    return epoch_loss, epoch_acc


# ---------------------------------------------------------------------------
# Main pseudo-labelling loop
# ---------------------------------------------------------------------------

def train_pseudo_labelled_model(
    base_model,
    labelled_loader,
    unlabelled_loader,
    confidence_threshold,
    num_iterations,
    num_epochs_per_iteration,
    label_minus=0,
    learning_rate=1e-3,
    batch_size=16,
):
    """Iterative self-training with pseudo-labelling.

    Args:
        base_model:       A classification model (encoder + head, no softmax).
                          Only the classification-head parameters that have
                          ``requires_grad=True`` will be optimised.
        labelled_loader:  DataLoader over the *labelled* training set.
                          Each batch must contain ``'c9'`` images and ``'label'``.
        unlabelled_loader:
                          DataLoader over *unlabelled* data.
                          Each batch must contain at least ``'c9'`` images.
        confidence_threshold:
                          Minimum softmax confidence to accept a pseudo-label.
        num_iterations:   Number of self-training rounds.
        num_epochs_per_iteration:
                          Number of training epochs in each round.
        label_minus:      Offset subtracted from dataset labels so they match
                          the model's output classes (e.g. 2 when the model only
                          distinguishes the last 2 of 4 classes).
        learning_rate:    Learning rate for the Adam optimiser.
        batch_size:       Batch size for the combined (labelled + pseudo) loader.

    Returns:
        model:            The trained model (modified in-place).
        iteration_cms:    List of confusion matrices, one per iteration,
                          evaluated on the labelled loader after training.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = base_model.to(device)

    iteration_cms = []

    for iteration in range(num_iterations):
        print(f'\n{"="*60}')
        print(f'Pseudo-labelling iteration {iteration + 1}/{num_iterations}')
        print(f'{"="*60}')

        # ------------------------------------------------------------------
        # 1. Evaluate current model on labelled data → class-bias
        # ------------------------------------------------------------------
        cm = evaluate_model(model, labelled_loader, label_minus=label_minus)
        print('Confusion matrix on labelled set:')
        print(cm)

        # Per-class recall; add eps to avoid division by zero
        row_sums = cm.sum(axis=1).astype(float)
        row_sums[row_sums == 0] = 1.0
        recalls = cm.diagonal().astype(float) / row_sums
        print(f'Per-class recall: {recalls}')

        # Classes with low recall should receive *more* pseudo-labels
        sample_ratios = 1.0 - recalls
        ratio_sum = sample_ratios.sum()
        if ratio_sum > 0:
            sample_ratios = sample_ratios / ratio_sum
        else:
            # Model is perfect on all classes – sample uniformly
            sample_ratios = np.ones_like(sample_ratios) / len(sample_ratios)
        print(f'Sampling ratios (toward weak classes): {sample_ratios}')

        # ------------------------------------------------------------------
        # 2. Predict on unlabelled data
        # ------------------------------------------------------------------
        images_ul, preds_ul, confs_ul = generate_pseudo_labels(
            model, unlabelled_loader, label_minus=label_minus
        )

        # ------------------------------------------------------------------
        # 3. Filter by confidence threshold
        # ------------------------------------------------------------------
        confident_mask = confs_ul >= confidence_threshold
        num_confident = confident_mask.sum()
        print(f'Confident samples (>= {confidence_threshold}): '
              f'{num_confident}/{len(confs_ul)}')

        if num_confident == 0:
            print('No samples above confidence threshold – skipping iteration.')
            iteration_cms.append(cm)
            continue

        # Map predictions back to model-output space for grouping
        preds_model_space = preds_ul - label_minus
        num_classes = cm.shape[0]

        # ------------------------------------------------------------------
        # 4. Stratified sampling biased toward weak classes
        # ------------------------------------------------------------------
        # Determine the *limiting class* – the class with the fewest
        # confident pseudo-labels among those that should be sampled.
        per_class_confident_counts = np.array([
            np.sum(confident_mask & (preds_model_space == c))
            for c in range(num_classes)
        ])
        print(f'Per-class confident counts: {per_class_confident_counts}')

        # Total pseudo-labels to select = number of confident samples
        # (we take all of them but re-weight across classes).
        selected_images = []
        selected_labels = []

        for c in range(num_classes):
            class_mask = confident_mask & (preds_model_space == c)
            class_indices = np.where(class_mask)[0]

            if len(class_indices) == 0:
                continue

            # Number to sample for this class, proportional to sample_ratios
            n_to_sample = max(1, int(sample_ratios[c] * num_confident))
            n_to_sample = min(n_to_sample, len(class_indices))

            # Sort by confidence (highest first) and take the top-n
            class_confs = confs_ul[class_indices]
            top_indices = class_indices[np.argsort(-class_confs)[:n_to_sample]]

            for idx in top_indices:
                selected_images.append(images_ul[idx])
                selected_labels.append(preds_ul[idx])  # original label space

        print(f'Selected {len(selected_images)} pseudo-labelled samples')

        if len(selected_images) == 0:
            print('No samples selected after stratified sampling – skipping.')
            iteration_cms.append(cm)
            continue

        # ------------------------------------------------------------------
        # 5. Build combined dataset and retrain
        # ------------------------------------------------------------------
        pseudo_dataset = PseudoLabelledDataset(selected_images, selected_labels)
        combined_dataset = ConcatDataset([labelled_loader.dataset, pseudo_dataset])
        combined_loader = DataLoader(
            combined_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
        )

        for epoch in range(num_epochs_per_iteration):
            loss, acc = _train_one_epoch(
                model, combined_loader, criterion, optimizer,
                device, label_minus=label_minus,
            )
            print(f'  Epoch {epoch + 1}/{num_epochs_per_iteration} — '
                  f'loss: {loss:.4f}, acc: {acc:.2f}%')

        # ------------------------------------------------------------------
        # 6. Re-evaluate after training
        # ------------------------------------------------------------------
        cm_after = evaluate_model(model, labelled_loader, label_minus=label_minus)
        iteration_cms.append(cm_after)
        print('Confusion matrix after retraining:')
        print(cm_after)

    return model, iteration_cms

if __name__ == '__main__':
    # load a model (satlas/satlas_ensemble_classification_checkpoints9/old_good.pth) and actually run the pseudo-labelling algorithm
    model = torch.load('../satlas/satlas_ensemble_classification_checkpoints9/old_good.pth')
    
    # load the labelled and unlabelled datasets
    labelled_dataset = ClassificationDataset('../kaggle')
    unlabelled_dataset = Sentinel2InpaintingDataset(
        '../s2a',
        format='satlas',
        limit_samples=2000,
    )
    
    # create dataloaders
    labelled_loader = DataLoader(
        labelled_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0, # already loaded in memory
    )
    unlabelled_loader = DataLoader(
        unlabelled_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=1,
    )
    
    # run the pseudo-labelling algorithm
    model, iteration_cms = pseudo_labelling(
        model,
        labelled_loader,
        unlabelled_loader,
        num_iterations=10,
        confidence_threshold=0.7,
        stratify_by_recall=True,
        num_epochs_per_iteration=5,
        learning_rate=0.001,
    )
    
    # save the model
    torch.save(model, 'pseudo_labelled_model.pth')
    
    # save the iteration confusion matrices
    np.save('iteration_cms.npy', iteration_cms)
    
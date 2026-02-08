import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def evaluate_and_plot_k(val_outputs, val_labels, k_range):
    """
    Run KNN evaluation over a range of k values, plot accuracies, and show confusion matrix for best k.

    Args:
        val_outputs: Feature array, shape (n_samples, n_features)
        val_labels: True labels, shape (n_samples,)
        k_range: Iterable of k values to test (e.g., range(1, 20))

    Returns:
        best_k: The k value that achieved the highest accuracy
        best_accuracy: The highest accuracy value
        best_confusion: Confusion matrix for the best k
    """
    n_samples = len(val_outputs)
    
    # Calculate distance matrix using broadcasting
    # Shape: (n_samples, n_samples)
    diff = val_outputs[:, np.newaxis] - val_outputs[np.newaxis, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
    
    # Don't include the point itself (distance = 0)
    np.fill_diagonal(dist_matrix, np.inf)

    accuracies = []
    confusion_matrices = []
    
    for k in k_range:
        predictions = np.zeros(n_samples, dtype=val_labels.dtype)
        
        # For each point
        for i in range(n_samples):
            # Get distances
            distances = dist_matrix[i]
            
            # Find k nearest neighbors
            nearest_indices = np.argsort(distances)[:k]
            
            # Get labels
            neighbor_labels = val_labels[nearest_indices]
            
            # Vote
            vote_counts = Counter(neighbor_labels)
            predictions[i] = vote_counts.most_common(1)[0][0]
            
        # Calculate accuracy
        correct = np.sum(predictions == val_labels)
        accuracy = correct / n_samples
        accuracies.append(accuracy)
        
        # Build confusion matrix
        unique_labels = np.unique(val_labels)
        n_classes = len(unique_labels)
        confusion = np.zeros((n_classes, n_classes), dtype=int)
        
        for true_label, pred_label in zip(val_labels, predictions):
            true_idx = np.where(unique_labels == true_label)[0][0]
            pred_idx = np.where(unique_labels == pred_label)[0][0]
            confusion[true_idx, pred_idx] += 1
            
        confusion_matrices.append(confusion)

    # Find best k
    best_idx = np.argmax(accuracies)
    best_k = list(k_range)[best_idx]
    best_accuracy = accuracies[best_idx]
    best_confusion = confusion_matrices[best_idx]
    
    # Plot accuracy vs k
    plt.figure(figsize=(10, 6))
    plt.plot(list(k_range), accuracies, marker='o', linestyle='-', color='b')
    plt.title('KNN Accuracy vs. k')
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.xticks(list(k_range))
    plt.show()

    # Plot confusion matrix for best k
    plt.figure(figsize=(8, 8))
    plt.imshow(best_confusion, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix (Best k={best_k}, Acc={best_accuracy:.4f})')
    plt.colorbar()
    
    tick_marks = np.arange(len(unique_labels))
    plt.xticks(tick_marks, unique_labels, rotation=45)
    plt.yticks(tick_marks, unique_labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Add counts to confusion matrix plot
    thresh = best_confusion.max() / 2.
    for i in range(best_confusion.shape[0]):
        for j in range(best_confusion.shape[1]):
            plt.text(j, i, format(best_confusion[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if best_confusion[i, j] > thresh else "black")
                     
    plt.tight_layout()
    plt.show()

    return best_k, best_accuracy, best_confusion
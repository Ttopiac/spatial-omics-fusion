"""Evaluation metrics for spatial domain detection."""
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, accuracy_score


def compute_metrics(y_true, y_pred):
    """
    Compute clustering/classification metrics.

    ARI (Adjusted Rand Index): Main metric. Measures agreement between predicted
    and true clusters, adjusted for chance. Range [-1, 1], higher is better.

    NMI (Normalized Mutual Information): Secondary metric. Measures shared
    information between predictions and truth. Range [0, 1], higher is better.
    """
    return {
        "ari": adjusted_rand_score(y_true, y_pred),
        "nmi": normalized_mutual_info_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def compute_extended_metrics(y_true, y_pred, y_probs, edge_index):
    """
    Compute extended metrics that account for boundary ambiguity.

    Args:
        y_true: ground truth labels (N,)
        y_pred: predicted labels (N,)
        y_probs: softmax probabilities (N, C)
        edge_index: graph edges (2, E) numpy array

    Returns:
        dict with standard + extended metrics
    """
    metrics = compute_metrics(y_true, y_pred)

    # --- Top-2 accuracy ---
    sorted_classes = np.argsort(y_probs, axis=1)[:, ::-1]
    top2_correct = (sorted_classes[:, 0] == y_true) | (sorted_classes[:, 1] == y_true)
    metrics["top2_accuracy"] = top2_correct.mean()

    # --- Boundary vs interior accuracy ---
    src, tgt = edge_index[0], edge_index[1]
    n = len(y_true)
    is_boundary = np.zeros(n, dtype=bool)
    for spot in range(n):
        neighbors = src[tgt == spot]
        if len(neighbors) > 0 and len(set(y_true[neighbors])) > 1:
            is_boundary[spot] = True

    n_boundary = is_boundary.sum()
    n_interior = n - n_boundary
    if n_interior > 0:
        metrics["interior_accuracy"] = (y_pred[~is_boundary] == y_true[~is_boundary]).mean()
    else:
        metrics["interior_accuracy"] = float("nan")
    if n_boundary > 0:
        metrics["boundary_accuracy"] = (y_pred[is_boundary] == y_true[is_boundary]).mean()
    else:
        metrics["boundary_accuracy"] = float("nan")
    metrics["n_boundary"] = int(n_boundary)
    metrics["n_interior"] = int(n_interior)

    # --- Cross-entropy (log-loss) ---
    # Clamp to avoid log(0)
    eps = 1e-7
    probs_clamped = np.clip(y_probs, eps, 1 - eps)
    log_loss = -np.log(probs_clamped[np.arange(n), y_true]).mean()
    metrics["log_loss"] = log_loss

    return metrics

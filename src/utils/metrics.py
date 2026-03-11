"""Evaluation metrics for spatial domain detection."""
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

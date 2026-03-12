import torch
import numpy as np


def compute_sample_metrics(
    pred_exists: torch.Tensor,
    gt_counts: list[int],
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute sample-level track count metrics.

    Args:
        pred_exists: (B, Q, 1) exists probabilities
        gt_counts: list of length B with ground truth track counts
        threshold: exists threshold

    Returns:
        dict with count_mae, null_day_accuracy
    """
    pred_counts = (pred_exists.squeeze(-1) > threshold).sum(dim=1).cpu().numpy()
    gt_arr = np.array(gt_counts)

    count_mae = float(np.abs(pred_counts - gt_arr).mean())

    # Null day accuracy
    gt_null = gt_arr == 0
    pred_null = pred_counts == 0
    if len(gt_arr) > 0:
        null_correct = ((gt_null & pred_null) | (~gt_null & ~pred_null)).sum()
        null_day_accuracy = float(null_correct / len(gt_arr))
    else:
        null_day_accuracy = 1.0

    return {
        "count_mae": count_mae,
        "null_day_accuracy": null_day_accuracy,
    }

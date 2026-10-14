"""Evaluation engine calculating Temporal IoU and flaw classification accuracy."""
import numpy as np


def compute_temporal_iou(pred_interval: tuple[float, float], gt_interval: tuple[float, float]) -> float:
    """Computes Intersection over Union (IoU) between two 1D temporal intervals."""
    p_start, p_end = pred_interval
    g_start, g_end = gt_interval

    inter_start = max(p_start, g_start)
    inter_end = min(p_end, g_end)
    intersection = max(0.0, inter_end - inter_start)

    union = (p_end - p_start) + (g_end - g_start) - intersection
    if union <= 0.0:
        return 0.0
    return float(intersection / union)


def evaluate_flaw_predictions(predictions: list[dict], ground_truth: list[dict], iou_threshold: float = 0.25) -> dict:
    """Evaluates predicted flaws against ground truth annotations."""
    tp, fp, fn = 0, 0, 0
    ious = []

    matched_gt = set()

    for pred in predictions:
        p_int = (pred["start_time"], pred["end_time"])
        p_cat = pred["category"]
        best_iou = 0.0
        best_gt_idx = -1

        for idx, gt in enumerate(ground_truth):
            if idx in matched_gt:
                continue
            if gt["category"] == p_cat:
                iou = compute_temporal_iou(p_int, (gt["start_time"], gt["end_time"]))
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

        if best_iou >= iou_threshold and best_gt_idx != -1:
            tp += 1
            matched_gt.add(best_gt_idx)
            ious.append(best_iou)
        else:
            fp += 1

    fn = len(ground_truth) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = float(np.mean(ious)) if len(ious) > 0 else 0.0

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1_score": round(f1, 3),
        "mean_matched_iou": round(mean_iou, 3)
    }
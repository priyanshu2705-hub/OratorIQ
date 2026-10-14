"""Automated benchmark runner: assesses precision, recall, and IoU against ground truth."""
import json
import pandas as pd
from src.audio_io import load_and_validate_audio
from src.feature_extraction import extract_acoustic_features
from src.alignment import align_audio_sequences
from src.comparison import compute_normalized_differentials
from src.flaw_detection import detect_delivery_flaws
from src.evaluation import evaluate_flaw_predictions


def main():
    manifest_path = "dataset/manifest.csv"
    with open("config/thresholds.json") as f:
        thresholds = json.load(f)

    manifest = pd.read_csv(manifest_path)
    total_evals = []

    print("======================================================================")
    print("OratorIQ Contrastive Temporal Flaw Grounding Evaluation Benchmark")
    print("======================================================================\n")

    for _, row in manifest.iterrows():
        sample_id = row["sample_id"]
        baseline_path = row["baseline_path"]
        part_path = row["participant_path"]

        # Load ground truth
        annotation_file = f"dataset/annotations/{part_path.split('/')[-1].replace('.wav', '.json')}"
        with open(annotation_file) as f:
            gt_annotations = json.load(f)

        # Run Pipeline
        ref_y, sr = load_and_validate_audio(baseline_path, thresholds["sample_rate"])
        part_y, _ = load_and_validate_audio(part_path, thresholds["sample_rate"])

        ref_feats = extract_acoustic_features(ref_y, sr)
        part_feats = extract_acoustic_features(part_y, sr)

        wp_ref, wp_part, _ = align_audio_sequences(ref_feats["mfcc"], part_feats["mfcc"])
        diffs = compute_normalized_differentials(ref_feats, part_feats, wp_ref, wp_part)
        detected_flaws = detect_delivery_flaws(diffs, part_feats, thresholds)

        metrics = evaluate_flaw_predictions(detected_flaws, gt_annotations, iou_threshold=0.20)
        metrics["sample_id"] = sample_id
        metrics["detected_count"] = len(detected_flaws)
        metrics["ground_truth_count"] = len(gt_annotations)
        total_evals.append(metrics)

        print(f"Sample: {sample_id:<22} | Precision: {metrics['precision']:.2f} | "
              f"Recall: {metrics['recall']:.2f} | F1: {metrics['f1_score']:.2f} | "
              f"Mean IoU: {metrics['mean_matched_iou']:.2f}")

    eval_df = pd.DataFrame(total_evals)
    print("\n----------------------------------------------------------------------")
    print(f"Overall Mean Precision : {eval_df['precision'].mean():.3f}")
    print(f"Overall Mean Recall    : {eval_df['recall'].mean():.3f}")
    print(f"Overall Mean F1 Score  : {eval_df['f1_score'].mean():.3f}")
    print(f"Overall Mean IoU       : {eval_df['mean_matched_iou'].mean():.3f}")
    print("======================================================================")


if __name__ == "__main__":
    main()
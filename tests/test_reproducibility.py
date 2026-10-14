import numpy as np
from src.feature_extraction import extract_acoustic_features
from src.alignment import align_audio_sequences
from src.comparison import compute_normalized_differentials
from src.flaw_detection import detect_delivery_flaws
from src.scoring import compute_rubric_scores


def test_identical_audio_scoring_reproducibility():
    """Comparing an audio signal to itself must yield zero flaws and full score."""
    sr = 22050
    t = np.linspace(0, 3.0, sr * 3, endpoint=False)
    y = (0.6 * np.sin(2 * np.pi * 180.0 * t)).astype(np.float32)

    feats = extract_acoustic_features(y, sr)
    wp_ref, wp_part, _ = align_audio_sequences(feats["mfcc"], feats["mfcc"])
    diffs = compute_normalized_differentials(feats, feats, wp_ref, wp_part)

    thresholds = {
        "rate_fast_ratio_threshold": 1.35,
        "rate_slow_ratio_threshold": 0.70,
        "pitch_monotone_std_semitones": 0.05,  # Low floor for synthetic test
        "excessive_pause_sec": 1.75
    }

    flaws = detect_delivery_flaws(diffs, feats, thresholds)
    scores = compute_rubric_scores(flaws)

    assert len(flaws) == 0
    assert scores["overall_score"] == 100.0
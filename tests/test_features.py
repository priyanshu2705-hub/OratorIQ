import numpy as np
from src.feature_extraction import extract_acoustic_features


def test_feature_extraction_structure():
    sr = 22050
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 220.0 * t).astype(np.float32)
    
    feats = extract_acoustic_features(y, sr=sr)
    assert "rms_db" in feats
    assert "f0_semitones" in feats
    assert "spectral_centroid" in feats
    assert len(feats["times"]) == len(feats["rms_db"])
    assert feats["total_duration_sec"] == pytest.approx(2.0, rel=1e-2)
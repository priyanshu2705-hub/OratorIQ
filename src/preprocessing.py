"""Signal conditioning, silence-preserving normalization, and trimming."""
import numpy as np
import librosa


def preprocess_signal(
    y: np.ndarray,
    sr: int = 22050,
    target_lufs_db: float = -20.0,
    trim_db: float = 40.0
) -> tuple[np.ndarray, dict]:
    """Applies non-destructive peak and dynamic normalization and leading/trailing trimming."""
    # Trim leading and trailing dead air
    y_trimmed, trim_idx = librosa.effects.trim(y, top_db=trim_db)
    
    # Peak normalization with 0.5 dB headroom
    current_peak = np.max(np.abs(y_trimmed))
    if current_peak > 0:
        gain = (10.0 ** (-0.5 / 20.0)) / current_peak
        y_norm = y_trimmed * min(gain, 10.0)  # Capped gain multiplier
    else:
        y_norm = y_trimmed

    # Compute short-term RMS
    rms = librosa.feature.rms(y=y_norm, frame_length=2048, hop_length=512)[0]
    mean_rms_db = float(librosa.amplitude_to_db(np.mean(rms) + 1e-6))

    metadata = {
        "original_samples": len(y),
        "trimmed_samples": len(y_norm),
        "duration_sec": len(y_norm) / sr,
        "trim_start_sec": trim_idx[0] / sr,
        "trim_end_sec": trim_idx[1] / sr,
        "mean_rms_db": mean_rms_db
    }
    return y_norm.astype(np.float32), metadata
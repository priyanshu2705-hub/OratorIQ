"""Speaker-agnostic feature normalization and relative differential calculations."""
import numpy as np


def compute_normalized_differentials(ref_feats: dict, part_feats: dict, wp_ref: np.ndarray, wp_part: np.ndarray) -> dict:
    """Calculates speaker-agnostic acoustic differentials along the DTW alignment path."""
    # Slice features along warping path
    ref_rms_path = ref_feats["rms_db"][wp_ref]
    part_rms_path = part_feats["rms_db"][wp_part]

    ref_f0_path = ref_feats["f0_semitones"][wp_ref]
    part_f0_path = part_feats["f0_semitones"][wp_part]

    part_times_path = part_feats["times"][wp_part]

    # Normalize energy to 0-mean, 1-std across valid voiced/unvoiced speech
    ref_rms_z = (ref_rms_path - np.nanmean(ref_rms_path)) / (np.nanstd(ref_rms_path) + 1e-6)
    part_rms_z = (part_rms_path - np.nanmean(part_rms_path)) / (np.nanstd(part_rms_path) + 1e-6)
    energy_delta_z = part_rms_z - ref_rms_z

    # Pitch normalization: mean-center relative pitch contours (speaker-invariant intonation)
    ref_f0_centered = ref_f0_path - np.nanmean(ref_f0_path)
    part_f0_centered = part_f0_path - np.nanmean(part_f0_path)
    pitch_delta_semitones = part_f0_centered - ref_f0_centered

    # Temporal warping velocity: derivative of participant steps vs. reference steps
    # d(t_part) / d(t_ref) > 1 indicates participant is speaking slower; < 1 indicates faster
    window_sz = 35
    wp_part_smooth = np.convolve(wp_part, np.ones(window_sz)/window_sz, mode="valid")
    wp_ref_smooth = np.convolve(wp_ref, np.ones(window_sz)/window_sz, mode="valid")
    
    d_part = np.diff(wp_part_smooth)
    d_ref = np.diff(wp_ref_smooth) + 1e-5
    local_rate_ratio = d_part / d_ref  # > 1: participant extended time (slow), < 1: compressed time (fast)

    # Pad velocity to match warped path length
    pad_len = len(wp_part) - len(local_rate_ratio)
    local_rate_ratio = np.pad(local_rate_ratio, (0, pad_len), mode="edge")

    return {
        "part_times_path": part_times_path,
        "energy_delta_z": energy_delta_z,
        "pitch_delta_semitones": pitch_delta_semitones,
        "local_rate_ratio": local_rate_ratio,
        "ref_rms_z": ref_rms_z,
        "part_rms_z": part_rms_z,
        "ref_f0_centered": ref_f0_centered,
        "part_f0_centered": part_f0_centered
    }
"""Acoustic alignment using Dynamic Time Warping (DTW) with energy-vad fallback."""
import numpy as np
import librosa
from scipy.spatial.distance import cdist


def align_audio_sequences(ref_mfcc: np.ndarray, part_mfcc: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aligns reference and participant feature trajectories using subsequence-aware DTW.
    
    Returns:
        ref_indices: aligned frame indices for reference.
        part_indices: aligned frame indices for participant.
        cost_matrix: accumulated cost matrix.
    """
    # Z-score normalize MFCCs across time for speaker timbre invariance
    r_norm = (ref_mfcc - np.mean(ref_mfcc, axis=1, keepdims=True)) / (np.std(ref_mfcc, axis=1, keepdims=True) + 1e-6)
    p_norm = (part_mfcc - np.mean(part_mfcc, axis=1, keepdims=True)) / (np.std(part_mfcc, axis=1, keepdims=True) + 1e-6)

    # Compute Euclidean distance matrix
    dist_matrix = cdist(r_norm.T, p_norm.T, metric="cosine")

    # Dynamic Time Warping via librosa
    cost, wp = librosa.sequence.dtw(C=dist_matrix, subseq=False)
    wp = wp[::-1]  # Order from start to finish

    ref_indices = wp[:, 0]
    part_indices = wp[:, 1]
    return ref_indices, part_indices, cost


def map_participant_time_to_ref(time_part: float, part_times: np.ndarray, wp_part: np.ndarray, wp_ref: np.ndarray, ref_times: np.ndarray) -> float:
    """Maps a given participant timestamp to the corresponding reference timestamp via the DTW warping path."""
    closest_part_frame = np.argmin(np.abs(part_times - time_part))
    path_matches = np.where(wp_part == closest_part_frame)[0]
    if len(path_matches) == 0:
        return time_part
    matched_ref_frame = wp_ref[path_matches[0]]
    return float(ref_times[min(matched_ref_frame, len(ref_times) - 1)])
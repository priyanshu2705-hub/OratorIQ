"""Time-domain, spectral, and pitch acoustic feature extraction."""
import numpy as np
import librosa


def extract_acoustic_features(y: np.ndarray, sr: int = 22050, hop_length: int = 512, n_fft: int = 2048) -> dict:
    """Computes frame-level and global acoustic features."""
    times = librosa.frames_to_time(np.arange(0, int(np.ceil(len(y) / hop_length))), sr=sr, hop_length=hop_length)

    # 1. Time-Domain: RMS Energy
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms + 1e-7, ref=np.max)

    # 2. Spectral Features: Centroid, Roll-off, Bandwidth
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, roll_percent=0.85)[0]
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]

    # 3. MFCCs (13 coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)

    # 4. Fundamental Frequency (F0) tracking via probabilistic YIN (pYIN)
    # Pitch bounds: 65 Hz (C2) to 400 Hz (G4)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('G4'),
        sr=sr,
        frame_length=n_fft,
        hop_length=hop_length,
        fill_na=np.nan
    )

    # Convert F0 in Hz to relative semitones (reference: 110 Hz = A2)
    with np.errstate(divide='ignore', invalid='ignore'):
        f0_semitones = 12.0 * np.log2(f0 / 110.0)

    # Pause intervals (energy below threshold)
    silence_mask = rms_db < -36.0
    pause_frames = np.sum(silence_mask)
    pause_ratio = float(pause_frames / max(len(rms_db), 1))

    # Clean pitch vector for statistics
    voiced_semitones = f0_semitones[~np.isnan(f0_semitones)]
    pitch_mean = float(np.mean(voiced_semitones)) if len(voiced_semitones) > 0 else 0.0
    pitch_std = float(np.std(voiced_semitones)) if len(voiced_semitones) > 0 else 0.0

    return {
        "times": times[:len(rms)],
        "rms": rms,
        "rms_db": rms_db,
        "spectral_centroid": spec_centroid,
        "spectral_rolloff": spec_rolloff,
        "spectral_bandwidth": spec_bandwidth,
        "mfcc": mfcc,
        "f0_hz": f0,
        "f0_semitones": f0_semitones,
        "voiced_flag": voiced_flag,
        "pause_mask": silence_mask,
        "pause_ratio": pause_ratio,
        "pitch_mean_semitones": pitch_mean,
        "pitch_std_semitones": pitch_std,
        "total_duration_sec": float(len(y) / sr)
    }
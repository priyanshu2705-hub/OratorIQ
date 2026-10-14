"""Audio validation, format conversion, and uniform loading."""
import io
import os
import soundfile as sf
import numpy as np
import librosa


def load_and_validate_audio(file_input, target_sr: int = 22050) -> tuple[np.ndarray, int]:
    """Loads WAV, MP3, or raw bytes, converts to mono, and resamples to target_sr."""
    if isinstance(file_input, (str, os.PathLike)):
        if not os.path.exists(file_input):
            raise FileNotFoundError(f"Audio file not found: {file_input}")
        y, sr = librosa.load(file_input, sr=target_sr, mono=True)
    elif hasattr(file_input, "read"):
        # Streamlit UploadedFile or BytesIO
        audio_bytes = file_input.read()
        file_input.seek(0)
        buffer = io.BytesIO(audio_bytes)
        y, sr = sf.read(buffer, dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
    else:
        raise ValueError("Unsupported audio input format.")

    if len(y) == 0:
        raise ValueError("Audio signal is empty.")

    # Guard against pure digital silence or NaN values
    if np.isnan(y).any() or np.isinf(y).any():
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    max_amp = np.max(np.abs(y))
    if max_amp < 1e-4:
        raise ValueError("Audio is pure silence or amplitude is below operational threshold.")

    return y.astype(np.float32), sr


def save_audio(path: str, y: np.ndarray, sr: int = 22050):
    """Saves floating-point audio array to disk in 16-bit WAV format."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sf.write(path, y, sr, subtype="PCM_16")
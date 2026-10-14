import numpy as np
import pytest
from src.audio_io import load_and_validate_audio, save_audio


def test_audio_save_and_load(tmp_path):
    sr = 22050
    test_signal = np.sin(2 * np.pi * 440.0 * np.linspace(0, 1.0, sr)).astype(np.float32)
    path = tmp_path / "test.wav"
    save_audio(str(path), test_signal, sr)

    loaded_y, loaded_sr = load_and_validate_audio(str(path), target_sr=sr)
    assert loaded_sr == sr
    assert len(loaded_y) == len(test_signal)
    assert np.max(np.abs(loaded_y)) > 0.5


def test_empty_audio_rejection(tmp_path):
    path = tmp_path / "empty.wav"
    save_audio(str(path), np.zeros(100, dtype=np.float32), 22050)
    with pytest.raises(ValueError, match="Audio is pure silence"):
        load_and_validate_audio(str(path))
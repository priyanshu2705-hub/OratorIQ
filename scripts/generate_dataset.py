"""Generates reproducible baseline audio and synthetic contrastive flaw variants."""
import os
import json
import numpy as np
import soundfile as sf
import librosa

SAMPLE_RATE = 22050
DURATION = 8.0  # seconds

# Sample Speech: Gettysburg Address excerpt
TRANSCRIPT = "Four score and seven years ago our fathers brought forth on this continent a new nation conceived in liberty."


def generate_harmonic_speech_carrier(duration: float, sr: int, f0_base: float = 140.0) -> np.ndarray:
    """Generates synthetic baseline harmonic speech signal with dynamic formants."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Pitch curve: natural rhetorical rise and fall
    pitch_contour = f0_base + 18.0 * np.sin(2 * np.pi * 0.45 * t) - 10.0 * (t / duration)
    phase = 2 * np.pi * np.cumsum(pitch_contour) / sr
    
    # Harmonics
    carrier = (
        0.55 * np.sin(phase) +
        0.28 * np.sin(2 * phase) +
        0.14 * np.sin(3 * phase) +
        0.06 * np.sin(4 * phase)
    )
    # Formant envelope modulation simulating speech syllables
    syllable_env = 0.5 * (1.0 + np.sin(2 * np.pi * 3.8 * t)) ** 1.4
    signal = carrier * syllable_env
    # Soft onset/offset
    fade = int(0.08 * sr)
    signal[:fade] *= np.linspace(0, 1, fade)
    signal[-fade:] *= np.linspace(1, 0, fade)
    return (signal / np.max(np.abs(signal)) * 0.85).astype(np.float32)


def main():
    os.makedirs("dataset/audio", exist_ok=True)
    os.makedirs("dataset/transcripts", exist_ok=True)
    os.makedirs("dataset/annotations", exist_ok=True)

    # 1. Baseline Audio
    baseline_audio = generate_harmonic_speech_carrier(DURATION, SAMPLE_RATE, f0_base=135.0)
    baseline_path = "dataset/audio/baseline_speech.wav"
    sf.write(baseline_path, baseline_audio, SAMPLE_RATE)

    with open("dataset/transcripts/sample_speech.txt", "w") as f:
        f.write(TRANSCRIPT)

    manifest_rows = []

    # Flaw 1: Excessive Speed (Time-compressed segment between 2.0s and 4.5s)
    flaw_fast = np.copy(baseline_audio)
    s_idx, e_idx = int(2.0 * SAMPLE_RATE), int(4.5 * SAMPLE_RATE)
    speed_factor = 1.65
    chunk = flaw_fast[s_idx:e_idx]
    compressed_chunk = librosa.effects.time_stretch(chunk, rate=speed_factor)
    # Insert compressed chunk and pad tail
    flaw_fast_signal = np.concatenate([
        flaw_fast[:s_idx],
        compressed_chunk,
        flaw_fast[e_idx:],
        np.zeros(len(chunk) - len(compressed_chunk), dtype=np.float32)
    ])[:len(baseline_audio)]
    sf.write("dataset/audio/flaw_rushed_delivery.wav", flaw_fast_signal, SAMPLE_RATE)

    gt_fast = [{
        "start_time": 2.0,
        "end_time": 3.5,
        "category": "pacing_fluency",
        "flaw_name": "Excessive Speech Velocity (Rushed Delivery)",
        "severity": "Severe"
    }]
    with open("dataset/annotations/flaw_rushed_delivery.json", "w") as f:
        json.dump(gt_fast, f, indent=2)
    manifest_rows.append(["flaw_01_rushed", baseline_path, "dataset/audio/flaw_rushed_delivery.wav", "pacing_fluency", "Severe", 2.0, 3.5])

    # Flaw 2: Unnatural Pause / Dead Air (Mute 3.0s to 5.2s)
    flaw_pause = np.copy(baseline_audio)
    p_start, p_end = int(3.0 * SAMPLE_RATE), int(5.2 * SAMPLE_RATE)
    flaw_pause[p_start:p_end] = 0.0
    sf.write("dataset/audio/flaw_unnatural_pause.wav", flaw_pause, SAMPLE_RATE)

    gt_pause = [{
        "start_time": 3.0,
        "end_time": 5.2,
        "category": "pause_management",
        "flaw_name": "Unnatural Silence / Extended Dead-Air",
        "severity": "Severe"
    }]
    with open("dataset/annotations/flaw_unnatural_pause.json", "w") as f:
        json.dump(gt_pause, f, indent=2)
    manifest_rows.append(["flaw_02_pause", baseline_path, "dataset/audio/flaw_unnatural_pause.wav", "pause_management", "Severe", 3.0, 5.2])

    # Flaw 3: Monotone Pitch Delivery (Zero-variance pitch across entire utterance)
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    monotone_carrier = 0.6 * np.sin(2 * np.pi * 135.0 * t) + 0.3 * np.sin(2 * np.pi * 270.0 * t)
    syllable_env = 0.5 * (1.0 + np.sin(2 * np.pi * 3.8 * t)) ** 1.4
    flaw_monotone = (monotone_carrier * syllable_env).astype(np.float32)
    sf.write("dataset/audio/flaw_monotone_delivery.wav", flaw_monotone, SAMPLE_RATE)

    gt_monotone = [{
        "start_time": 0.5,
        "end_time": 7.5,
        "category": "pitch_expressiveness",
        "flaw_name": "Monotonous Pitch Contour (Flat Delivery)",
        "severity": "Severe"
    }]
    with open("dataset/annotations/flaw_monotone_delivery.json", "w") as f:
        json.dump(gt_monotone, f, indent=2)
    manifest_rows.append(["flaw_03_monotone", baseline_path, "dataset/audio/flaw_monotone_delivery.wav", "pitch_expressiveness", "Severe", 0.5, 7.5])

    # Flaw 4: Vocal Energy Drop (Muffle/attenuate 4.0s to 7.0s)
    flaw_energy = np.copy(baseline_audio)
    e_s, e_e = int(4.0 * SAMPLE_RATE), int(7.0 * SAMPLE_RATE)
    flaw_energy[e_s:e_e] *= 0.15
    sf.write("dataset/audio/flaw_energy_drop.wav", flaw_energy, SAMPLE_RATE)

    gt_energy = [{
        "start_time": 4.0,
        "end_time": 7.0,
        "category": "vocal_energy_clarity",
        "flaw_name": "Subdued Vocal Energy / Lost Projection",
        "severity": "Severe"
    }]
    with open("dataset/annotations/flaw_energy_drop.json", "w") as f:
        json.dump(gt_energy, f, indent=2)
    manifest_rows.append(["flaw_04_energy_drop", baseline_path, "dataset/audio/flaw_energy_drop.wav", "vocal_energy_clarity", "Severe", 4.0, 7.0])

    # Save manifest CSV
    manifest_header = "sample_id,baseline_path,participant_path,flaw_category,severity,gt_start_time,gt_end_time\n"
    with open("dataset/manifest.csv", "w") as f:
        f.write(manifest_header)
        for row in manifest_rows:
            f.write(",".join(map(str, row)) + "\n")

    print("[SUCCESS] Contrastive speech dataset generated in dataset/.")


if __name__ == "__main__":
    main()
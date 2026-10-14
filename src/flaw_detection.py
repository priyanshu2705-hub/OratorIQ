"""Temporal grounding and heuristic flaw detection based on acoustic deviations."""
import numpy as np


def detect_delivery_flaws(diffs: dict, part_feats: dict, thresholds: dict) -> list[dict]:
    """Identifies temporal flaw regions using acoustic thresholds and returns grounded segments."""
    flaws = []
    times = diffs["part_times_path"]
    n_frames = len(times)
    
    # 1. Excessive Speed Flaws (Rate ratio < 0.65 for sustained duration)
    fast_mask = diffs["local_rate_ratio"] < (1.0 / thresholds["rate_fast_ratio_threshold"])
    flaws.extend(_extract_contiguous_flaws(
        mask=fast_mask,
        times=times,
        category="pacing_fluency",
        flaw_name="Excessive Speech Velocity (Rushed Delivery)",
        threshold_val=thresholds["rate_fast_ratio_threshold"],
        diff_array=1.0 / (diffs["local_rate_ratio"] + 1e-4),
        min_duration=0.6,
        is_ratio=True
    ))

    # 2. Sluggish Pacing Flaws (Rate ratio > 1.45 for sustained duration)
    slow_mask = diffs["local_rate_ratio"] > (1.0 / thresholds["rate_slow_ratio_threshold"])
    flaws.extend(_extract_contiguous_flaws(
        mask=slow_mask,
        times=times,
        category="pacing_fluency",
        flaw_name="Sluggish Delivery & Drawled Articulation",
        threshold_val=thresholds["rate_slow_ratio_threshold"],
        diff_array=diffs["local_rate_ratio"],
        min_duration=0.8,
        is_ratio=True
    ))

    # 3. Unnatural Pauses & Dead Air (Participant silent while reference active)
    pause_mask = part_feats["pause_mask"]
    p_times = part_feats["times"]
    raw_pauses = _extract_contiguous_flaws(
        mask=pause_mask,
        times=p_times,
        category="pause_management",
        flaw_name="Unnatural Silence / Extended Dead-Air",
        threshold_val=thresholds["excessive_pause_sec"],
        diff_array=np.ones_like(p_times),
        min_duration=thresholds["excessive_pause_sec"],
        is_ratio=False
    )
    flaws.extend(raw_pauses)

    # 4. Monotone Intonation Flaw (Pitch variance across sliding window is below threshold)
    f0_c = diffs["part_f0_centered"]
    win = 45
    if len(f0_c) > win:
        rolling_f0_std = np.array([
            np.nanstd(f0_c[max(0, i - win):i + 1]) if np.sum(~np.isnan(f0_c[max(0, i - win):i + 1])) > 10 else 1.5
            for i in range(len(f0_c))
        ])
        monotone_mask = rolling_f0_std < thresholds["pitch_monotone_std_semitones"]
        flaws.extend(_extract_contiguous_flaws(
            mask=monotone_mask,
            times=times,
            category="pitch_expressiveness",
            flaw_name="Monotonous Pitch Contour (Flat Delivery)",
            threshold_val=thresholds["pitch_monotone_std_semitones"],
            diff_array=rolling_f0_std,
            min_duration=1.2,
            is_ratio=False,
            inverse_severity=True
        ))

    # 5. Vocal Energy Drop / Fading Projection
    energy_drop_mask = diffs["energy_delta_z"] < -1.8
    flaws.extend(_extract_contiguous_flaws(
        mask=energy_drop_mask,
        times=times,
        category="vocal_energy_clarity",
        flaw_name="Subdued Vocal Energy / Lost Projection",
        threshold_val=-1.8,
        diff_array=diffs["energy_delta_z"],
        min_duration=0.7,
        is_ratio=False
    ))

    # Deduplicate and sort chronologically
    flaws = sorted(flaws, key=lambda x: x["start_time"])
    return flaws


def _extract_contiguous_flaws(
    mask: np.ndarray,
    times: np.ndarray,
    category: str,
    flaw_name: str,
    threshold_val: float,
    diff_array: np.ndarray,
    min_duration: float,
    is_ratio: bool,
    inverse_severity: bool = False
) -> list[dict]:
    """Helper to group boolean masks into start/end intervals with calculated statistics."""
    flaws = []
    in_flaw = False
    start_idx = 0

    for i, flag in enumerate(mask):
        if flag and not in_flaw:
            in_flaw = True
            start_idx = i
        elif not flag and in_flaw:
            in_flaw = False
            duration = times[i - 1] - times[start_idx]
            if duration >= min_duration:
                segment_diff = diff_array[start_idx:i]
                val = float(np.mean(segment_diff))
                
                # Severity heuristic based on deviation scale
                if inverse_severity:
                    sev = "Severe" if val < (threshold_val * 0.6) else "Moderate"
                else:
                    sev = "Severe" if val > (threshold_val * 1.5) else "Moderate"

                flaws.append({
                    "start_time": round(float(times[start_idx]), 2),
                    "end_time": round(float(times[i - 1]), 2),
                    "duration_sec": round(float(duration), 2),
                    "category": category,
                    "flaw_name": flaw_name,
                    "severity": sev,
                    "measured_val": round(val, 2),
                    "threshold_val": round(threshold_val, 2),
                    "is_ratio": is_ratio
                })

    if in_flaw:
        duration = times[-1] - times[start_idx]
        if duration >= min_duration:
            val = float(np.mean(diff_array[start_idx:]))
            flaws.append({
                "start_time": round(float(times[start_idx]), 2),
                "end_time": round(float(times[-1]), 2),
                "duration_sec": round(float(duration), 2),
                "category": category,
                "flaw_name": flaw_name,
                "severity": "Moderate",
                "measured_val": round(val, 2),
                "threshold_val": round(threshold_val, 2),
                "is_ratio": is_ratio
            })
    return flaws
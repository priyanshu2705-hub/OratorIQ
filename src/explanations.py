"""Evidence-based causal explainability generator grounding mathematical deltas."""


def generate_causal_explanations(flaws: list[dict]) -> list[dict]:
    """Constructs structured, auditable causal feedback for each detected temporal flaw."""
    explained = []
    for f in flaws:
        st, et = f["start_time"], f["end_time"]
        cat = f["category"]
        m_val, t_val = f["measured_val"], f["threshold_val"]
        
        if cat == "pacing_fluency":
            if "Rushed" in f["flaw_name"]:
                obs = f"Acoustic delivery accelerated significantly across window {st}s–{et}s."
                diff_str = f"Estimated speaking rate ratio was {m_val}x normal reference velocity (exceeded limit {t_val}x)."
                rule = "Local dynamic time-warping frame progression d(t_part)/d(t_ref) fell below upper speed tolerance."
                action = "Slow down articulation. Emphasize syllable endings and introduce deliberate pauses before major nouns."
            else:
                obs = f"Speech pacing dragged abnormally between {st}s and {et}s."
                diff_str = f"Local rate ratio measured {m_val}x slower than baseline cadence (exceeded limit {t_val}x)."
                rule = "DTW trajectory slope exceeded sluggishness threshold for > 0.8 continuous seconds."
                action = "Maintain rhythmic forward momentum; avoid drawling vowels or lingering between clauses."

        elif cat == "pause_management":
            obs = f"Unnatural dead air detected from {st}s to {et}s ({f['duration_sec']}s total)."
            diff_str = f"Silence interval exceeded allowable pause limit of {t_val}s."
            rule = "RMS energy dropped below -36 dB threshold during a transcript phrase segment."
            action = "Eliminate hesitant pauses; use controlled breath inhalation instead of halting vocalization."

        elif cat == "pitch_expressiveness":
            obs = f"Vocal tone displayed severe monotony from {st}s to {et}s."
            diff_str = f"Sliding pitch standard deviation collapsed to {m_val} semitones (baseline minimum: {t_val} semitones)."
            rule = "pYIN fundamental frequency variation dropped below conversational inflection floor."
            action = "Inflect upward on key thematic terms and vary pitch pitch across sentence transitions."

        elif cat == "vocal_energy_clarity":
            obs = f"Audible drop in vocal projection between {st}s and {et}s."
            diff_str = f"Relative RMS z-score delta dropped to {m_val} relative to reference baseline."
            rule = "Speaker-normalized RMS energy fell below dynamic threshold -1.8."
            action = "Engage diaphragmatic breath support; sustain consistent decibel projection through phrase endings."

        else:
            obs = f"Acoustic irregularity detected from {st}s to {et}s."
            diff_str = f"Measured delta was {m_val} vs expected threshold {t_val}."
            rule = "Spectral energy deviated from normalized baseline distribution."
            action = "Re-align stress and vocal emphasis with baseline delivery."

        explained.append({
            "observation": obs,
            "timestamp_range": f"{st:.2f}s – {et:.2f}s",
            "start_time": st,
            "end_time": et,
            "category": cat,
            "severity": f["severity"],
            "penalty": f.get("applied_penalty", 0.0),
            "mathematical_evidence": diff_str,
            "rule_triggered": rule,
            "recommended_action": action,
            "limitation": "Acoustic divergence indicates delivery deviation but cannot discern subjective speaker intent."
        })
    return explained
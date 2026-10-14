"""Rubric scoring engine implementing deterministic point deductions."""
import json
import os


def compute_rubric_scores(flaws: list[dict], rubric_config_path: str = "config/rubric.json") -> dict:
    """Computes a 100-point reproducible speech score by deducting points per flaw."""
    if os.path.exists(rubric_config_path):
        with open(rubric_config_path, "r") as f:
            rubric = json.load(f)
    else:
        # Fallback default configuration
        rubric = {
            "categories": {
                "pacing_fluency": {"weight": 25, "max_penalty_per_event": 6.0},
                "pause_management": {"weight": 20, "max_penalty_per_event": 5.0},
                "vocal_energy_clarity": {"weight": 20, "max_penalty_per_event": 5.0},
                "pitch_expressiveness": {"weight": 20, "max_penalty_per_event": 5.0},
                "emphasis_consistency": {"weight": 15, "max_penalty_per_event": 4.0}
            }
        }

    category_scores = {}
    deductions_by_category = {cat: 0.0 for cat in rubric["categories"]}

    for flaw in flaws:
        cat = flaw["category"]
        if cat in deductions_by_category:
            max_pen = rubric["categories"][cat]["max_penalty_per_event"]
            multiplier = 1.0 if flaw["severity"] == "Severe" else 0.55
            # Scale deduction proportionally by duration (0.5x to 1.5x)
            dur_scale = np_clip = min(max(flaw["duration_sec"] / 1.0, 0.5), 1.5)
            penalty = round(min(max_pen * multiplier * dur_scale, max_pen), 2)
            flaw["applied_penalty"] = penalty
            deductions_by_category[cat] += penalty

    # Compute category scores and final score
    total_score = 0.0
    strengths = []
    weaknesses = []

    for cat, cfg in rubric["categories"].items():
        base_weight = cfg["weight"]
        actual_deduction = deductions_by_category[cat]
        final_cat_score = max(0.0, round(base_weight - actual_deduction, 1))
        category_scores[cat] = {
            "score": final_cat_score,
            "weight": base_weight,
            "deductions": round(actual_deduction, 1)
        }
        total_score += final_cat_score

        pct = (final_cat_score / base_weight) * 100.0
        if pct >= 88.0:
            strengths.append(f"Strong performance in {cat.replace('_', ' ').title()} ({final_cat_score}/{base_weight})")
        elif pct < 70.0:
            weaknesses.append(f"Needs improvement in {cat.replace('_', ' ').title()} ({final_cat_score}/{base_weight})")

    return {
        "overall_score": round(total_score, 1),
        "category_scores": category_scores,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "total_flaws_detected": len(flaws)
    }
"""Serializes analysis output to JSON and flat CSV formats."""
import json
import pandas as pd


def export_analysis_to_json(analysis_results: dict, output_path: str):
    """Exports comprehensive analysis results and metadata to JSON."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(analysis_results, f, indent=2)


def export_flaws_to_csv(flaws_explained: list[dict], output_path: str):
    """Exports structured flaw breakdown and causal explanations to CSV."""
    df = pd.DataFrame(flaws_explained)
    df.to_csv(output_path, index=False)
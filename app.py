"""OratorIQ — Contrastive Speech Analytics & Temporal Flaw Grounding Interactive Dashboard."""
import json
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.audio_io import load_and_validate_audio
from src.preprocessing import preprocess_signal
from src.feature_extraction import extract_acoustic_features
from src.alignment import align_audio_sequences
from src.comparison import compute_normalized_differentials
from src.flaw_detection import detect_delivery_flaws
from src.scoring import compute_rubric_scores
from src.explanations import generate_causal_explanations
from src.report_export import export_analysis_to_json, export_flaws_to_csv

# Streamlit Page Config
st.set_page_config(
    page_title="OratorIQ — Contrastive Speech Analytics",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #1f77b4;
    }
    .flaw-box {
        background-color: #fff3cd;
        border-radius: 6px;
        padding: 12px;
        margin-bottom: 10px;
        border: 1px solid #ffeeba;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_configs():
    with open("config/rubric.json") as f:
        rubric = json.load(f)
    with open("config/thresholds.json") as f:
        thresholds = json.load(f)
    return rubric, thresholds


def main():
    rubric_cfg, thresh_cfg = load_configs()

    st.sidebar.title("🎙️ OratorIQ Engine")
    menu = st.sidebar.radio(
        "Navigation",
        ["Home & Overview", "Interactive Analysis", "Dataset Spectrum Explorer", "System Methodology & Rubric"]
    )

    if menu == "Home & Overview":
        st.title("OratorIQ: Contrastive Speech Analytics")
        st.subheader("Objective Temporal Grounding & Causal Delivery Evaluation")
        
        st.markdown("""
        Evaluating public speaking performance is historically subjective. **OratorIQ** resolves this by contrasting 
        a participant's spoken delivery against an ideal reference delivery of the identical transcript.
        
        ### Core Processing Pipeline:
        1. **Dual Audio Signal Conditioning**: 22.05 kHz resampling, dynamic headroom preservation, and mono mixing.
        2. **Subsequence Dynamic Time Warping (DTW)**: Non-linear temporal warping over spectral MFCC representations.
        3. **Speaker-Agnostic Feature Extraction**: Relative pitch contours (pYIN semitone differential) and dynamic z-score RMS energy.
        4. **Temporal Grounding & Causal Explainability**: Precise start/stop bounds on pacing anomalies, dead-air blocks, and monotonic deliveries.
        5. **Deterministic 100-Point Rubric**: Rule-based penalties providing auditable, bias-free evaluative scoring.
        """)
        
        col1, col2, col3 = st.columns(3)
        col1.info("**Dataset Grounding**\nPaired contrastive recordings spanning near-perfect to severely flawed delivery.")
        col2.info("**Speaker-Agnostic**\nNormalized against pitch register and absolute biological volume differences.")
        col3.info("**Causal Explanations**\nDirect mathematical delta and rule triggers for every detected defect.")

    elif menu == "Interactive Analysis":
        st.title("Speech Analysis & Flaw Grounding")
        
        col_up1, col_up2 = st.columns(2)
        with col_up1:
            ref_file = st.file_uploader("Upload Reference Audio (Ideal)", type=["wav", "mp3"])
        with col_up2:
            part_file = st.file_uploader("Upload Participant Audio (Candidate)", type=["wav", "mp3"])

        transcript_input = st.text_area(
            "Speech Transcript",
            "Four score and seven years ago our fathers brought forth on this continent a new nation conceived in liberty."
        )

        use_demo = st.checkbox("Load Preset Demo (Dataset Sample 01: Rushed Pacing Flaw)")
        if use_demo:
            ref_file = "dataset/audio/baseline_speech.wav"
            part_file = "dataset/audio/flaw_rushed_delivery.wav"

        if st.button("Run Contrastive Speech Evaluation", type="primary"):
            if not ref_file or not part_file:
                st.error("Please provide both a Reference and Participant audio recording.")
                return

            with st.spinner("Executing acoustic feature extraction, DTW alignment, and temporal flaw classification..."):
                try:
                    # 1. Audio Ingestion
                    y_ref_raw, sr = load_and_validate_audio(ref_file, thresh_cfg["sample_rate"])
                    y_part_raw, _ = load_and_validate_audio(part_file, thresh_cfg["sample_rate"])

                    # 2. Preprocessing
                    y_ref, meta_ref = preprocess_signal(y_ref_raw, sr)
                    y_part, meta_part = preprocess_signal(y_part_raw, sr)

                    # 3. Acoustic Feature Extraction
                    ref_feats = extract_acoustic_features(y_ref, sr)
                    part_feats = extract_acoustic_features(y_part, sr)

                    # 4. Alignment
                    wp_ref, wp_part, _ = align_audio_sequences(ref_feats["mfcc"], part_feats["mfcc"])

                    # 5. Speaker-Agnostic Differentials
                    diffs = compute_normalized_differentials(ref_feats, part_feats, wp_ref, wp_part)

                    # 6. Flaw Detection & Scoring
                    flaws = detect_delivery_flaws(diffs, part_feats, thresh_cfg)
                    scoring_res = compute_rubric_scores(flaws, "config/rubric.json")
                    explanations = generate_causal_explanations(flaws)

                    st.session_state["analysis"] = {
                        "ref_feats": ref_feats,
                        "part_feats": part_feats,
                        "diffs": diffs,
                        "flaws": flaws,
                        "scoring": scoring_res,
                        "explanations": explanations
                    }
                    st.success("Analysis complete!")
                except Exception as e:
                    st.error(f"Processing error: {str(e)}")
                    return

        # Render Results
        if "analysis" in st.session_state:
            res = st.session_state["analysis"]
            score_data = res["scoring"]
            flaws = res["flaws"]
            explanations = res["explanations"]

            st.markdown("---")
            st.header("Executive Evaluation Summary")

            # Score Cards
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Overall Delivery Score", f"{score_data['overall_score']} / 100")
            m2.metric("Total Flaws Grounded", f"{score_data['total_flaws_detected']}")
            m3.metric("Pacing & Fluency", f"{score_data['category_scores']['pacing_fluency']['score']} / 25")
            m4.metric("Pitch Expressiveness", f"{score_data['category_scores']['pitch_expressiveness']['score']} / 20")

            # Interactive Time-Series Charts
            st.subheader("Temporal Feature Contours & Grounded Flaws")

            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                subplot_titles=("Speaker-Normalized Energy Contours (Z-Score)", "Pitch Intonation Contours (Centered Semitones)")
            )

            # Energy Plot
            times = res["part_feats"]["times"]
            fig.add_trace(
                go.Scatter(x=res["diffs"]["part_times_path"], y=res["diffs"]["ref_rms_z"],
                           mode="lines", name="Reference Energy", line=dict(color="#2ca02c", dash="dot")),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=res["diffs"]["part_times_path"], y=res["diffs"]["part_rms_z"],
                           mode="lines", name="Participant Energy", line=dict(color="#1f77b4")),
                row=1, col=1
            )

            # Pitch Plot
            fig.add_trace(
                go.Scatter(x=res["diffs"]["part_times_path"], y=res["diffs"]["ref_f0_centered"],
                           mode="lines", name="Reference Pitch", line=dict(color="#2ca02c", dash="dot")),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=res["diffs"]["part_times_path"], y=res["diffs"]["part_f0_centered"],
                           mode="lines", name="Participant Pitch", line=dict(color="#ff7f0e")),
                row=2, col=1
            )

            # Highlight Grounded Flaw Regions
            for f in flaws:
                fig.add_vrect(
                    x0=f["start_time"], x1=f["end_time"],
                    fillcolor="red", opacity=0.18, line_width=1,
                    annotation_text=f["severity"], annotation_position="top left",
                    row=1, col=1
                )
                fig.add_vrect(
                    x0=f["start_time"], x1=f["end_time"],
                    fillcolor="red", opacity=0.18, line_width=1,
                    row=2, col=1
                )

            fig.update_layout(height=520, margin=dict(l=20, r=20, t=40, b=20), hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            # Detailed Causal Explanations
            st.subheader("Causal Explanations & Evidence-Based Feedback")
            if not explanations:
                st.info("No significant delivery flaws detected. Delivery closely tracks baseline.")
            else:
                for exp in explanations:
                    with st.expander(f"⚠️ [{exp['timestamp_range']}] {exp['category'].replace('_', ' ').title()} — {exp['severity']} Severity (-{exp['penalty']} pts)"):
                        st.markdown(f"**Observation:** {exp['observation']}")
                        st.markdown(f"**Mathematical Evidence:** `{exp['mathematical_evidence']}`")
                        st.markdown(f"**Rule Triggered:** {exp['rule_triggered']}")
                        st.markdown(f"**Actionable Correction:** {exp['recommended_action']}")
                        st.caption(f"*System Boundary:* {exp['limitation']}")

            # Export
            st.subheader("Export & Reproducibility")
            export_col1, export_col2 = st.columns(2)
            with export_col1:
                json_data = json.dumps(res["scoring"], indent=2)
                st.download_button("Download JSON Score Report", data=json_data, file_name="oratoriq_evaluation.json", mime="application/json")
            with export_col2:
                import pandas as pd
                csv_data = pd.DataFrame(explanations).to_csv(index=False)
                st.download_button("Download Flaws CSV", data=csv_data, file_name="grounded_flaws.csv", mime="text/csv")

    elif menu == "Dataset Spectrum Explorer":
        st.title("Contrastive Dataset Explorer")
        st.markdown("Inspect paired samples spanning the synthetic flaw spectrum.")
        
        try:
            import pandas as pd
            df = pd.read_csv("dataset/manifest.csv")
            st.dataframe(df, use_container_width=True)
            
            selected_sample = st.selectbox("Select Sample to Preview", df["sample_id"].tolist())
            row = df[df["sample_id"] == selected_sample].iloc[0]
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Baseline Reference Audio**")
                st.audio(row["baseline_path"])
            with col_b:
                st.markdown(f"**Flawed Variant ({row['flaw_category']})**")
                st.audio(row["participant_path"])
        except Exception:
            st.warning("Dataset manifest not generated yet. Run `python scripts/generate_dataset.py`.")

    elif menu == "System Methodology & Rubric":
        st.title("Evaluation Rubric & Mathematical Methodology")
        st.json(rubric_cfg)


if __name__ == "__main__":
    main()
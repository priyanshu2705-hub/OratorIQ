# OratorIQ — Contrastive Speech Analytics & Temporal Flaw Grounding
**Multimodal AI Hackathon 2026 — Track C Deliverable**

OratorIQ is a digital signal processing and speech evaluation pipeline designed to evaluate public speaking delivery objectively. By contrasting a participant's spoken audio against an ideal reference delivery of the exact same speech transcript, OratorIQ eliminates judge subjectivity, bounds the exact timestamps of vocal flaws, and generates causal mathematical feedback.

---

## 1. Key Architectural Features

- **Contrastive Signal Alignment**: Employs Subsequence Dynamic Time Warping (DTW) over speaker-normalized Mel-Frequency Cepstral Coefficients (MFCCs).
- **Speaker-Agnostic Normalization**:
  - Converts Fundamental Frequency ($F_0$) to semitones ($12 \times \log_2(F_0 / 110)$) and mean-centers across voiced intervals to accommodate different vocal registers.
  - Normalizes RMS energy contours via dynamic $z$-scoring to eliminate microphone gain and room distance discrepancies.
- **Temporal Flaw Grounding**: Detects and bounds five core delivery flaws:
  1. *Excessive Speech Velocity (Rushed Delivery)*
  2. *Sluggish Delivery & Drawled Articulation*
  3. *Unnatural Silence & Extended Dead-Air*
  4. *Monotonous Pitch Contours (Flat Delivery)*
  5. *Subdued Vocal Energy & Lost Projection*
- **Deductive 100-Point Rubric**: Computes deterministic category and cumulative scores with full mathematical provenance.
- **Interactive Streamlit Dashboard**: Provides synchronized time-series plots with Plotly, interactive flaw boundaries, and JSON/CSV export.

---

## 2. Environment Setup & Installation

### Prerequisites
- Python 3.11
- Virtual environment tool (`venv` or `conda`)
- OS: Linux, macOS, or Windows

### Step 1: Clone Repository & Create Environment
```bash
git clone [https://github.com/your-username/oratoriq.git](https://github.com/your-username/oratoriq.git)
cd oratoriq

python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
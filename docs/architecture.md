# OratorIQ — System Architecture & Dataflow

## 1. High-Level Architectural Diagram

```text
+-----------------------+       +--------------------------+
| Reference Audio (WAV) |       | Participant Audio (WAV)  |
+-----------+-----------+       +------------+-------------+
            |                                |
            v                                v
+----------------------------------------------------------+
|                  src/audio_io.py                         |
|  - Decoding, Format Validation, Mono Mixing, Resampling  |
+---------------------------+------------------------------+
                            |
                            v
+----------------------------------------------------------+
|                 src/preprocessing.py                     |
|  - Boundary Silence Trimming & Peak Headroom Capping     |
+---------------------------+------------------------------+
                            |
                            v
+----------------------------------------------------------+
|               src/feature_extraction.py                  |
|  - Time Domain: RMS Energy, Silence/Pause Masks          |
|  - Frequency Domain: Centroid, Roll-off, Bandwidth       |
|  - Timbre & Pitch: 13 MFCCs, pYIN F0 Tracking (Hz -> st) |
+---------------------------+------------------------------+
                            |
            +---------------+---------------+
            |                               |
            v                               v
+-----------------------+       +--------------------------+
|   Normalized MFCCs    |       |     Normalized MFCCs     |
+-----------+-----------+       +------------+-------------+
            \                               /
             \                             /
              v                           v
+----------------------------------------------------------+
|                   src/alignment.py                       |
|  - Subsequence Dynamic Time Warping (Cosine Distance)    |
|  - Optimal Warping Path Construction (wp_ref, wp_part)   |
+---------------------------+------------------------------+
                            |
                            v
+----------------------------------------------------------+
|                   src/comparison.py                      |
|  - Z-Score Energy Normalization                          |
|  - Mean-Centered Relative Pitch Differentials (st)       |
|  - Warping Path Derivative: Local Rate Velocity Ratio    |
+---------------------------+------------------------------+
                            |
                            v
+----------------------------------------------------------+
|                 src/flaw_detection.py                    |
|  - Sliding-Window Heuristic Temporal Flaw Grounding      |
|  - Severity Assignment & Duration Filtering              |
+---------------------------+------------------------------+
                            |
            +---------------+---------------+
            |                               |
            v                               v
+-----------------------+       +--------------------------+
|    src/scoring.py     |       |   src/explanations.py    |
| - 100-Point Rubric    |       | - Mathematical Evidence  |
| - Deductive Engine    |       | - Actionable Corrections |
+-----------+-----------+       +------------+-------------+
            \                               /
             \                             /
              v                           v
+----------------------------------------------------------+
|                        app.py                            |
|  - Streamlit UI, Plotly Time-Series Overlay, JSON Export |
+----------------------------------------------------------+
# OratorIQ — System Limitations, Assumptions & Boundary Conditions

## 1. Digital Signal Processing Boundaries

1. **Acoustic Environment Sensitivity**: Highly reverberant rooms or background acoustic noise ($\text{SNR} < 12\text{ dB}$) can degrade pYIN pitch tracking accuracy and produce spurious unvoiced frames, impacting pitch expressiveness scores.
2. **Dynamic Time Warping Monotonicity**: DTW assumes chronological monotonicity. If a participant skips entire paragraphs or delivers sentences out of order relative to the reference recording, global path alignment will degrade.
3. **Phonetic Timbre Differences**: While MFCC z-scoring minimizes speaker timbre discrepancies, extreme foreign accents or severe vocal pathology can reduce cosine alignment precision across phonetic transitions.

## 2. Evaluation Boundaries

1. **Content Quality Decoupling**: OratorIQ exclusively evaluates acoustic and rhetorical delivery dynamics (prosody, cadence, projection, pitch modulation). It does not grade rhetorical textual structure, logical coherence, or argumentative validity.
2. **Stylistic Variability**: Public speaking admits stylistic variation. An intentional, dramatic pause that exceeds 1.75 seconds may be flagged as dead air by rule-based heuristic thresholds unless configured prior to analysis.
3. **Absence of Psychological Profiling**: Acoustic features such as pitch shifts and energy drops reflect signal deviations; they do not substantiate claims regarding speaker nervousness, anxiety, or psychological state.
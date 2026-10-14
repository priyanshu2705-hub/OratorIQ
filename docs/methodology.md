# OratorIQ — Mathematical Methodology & Scoring Rubric

## 1. Acoustic Signal Processing

### 1.1 Root Mean Square (RMS) Energy & Dynamic Scaling
To decouple vocal projection from hardware variations (microphone gain, recording distance), short-time RMS energy is computed:

$$\text{RMS}(m) = \sqrt{\frac{1}{N} \sum_{n=0}^{N-1} x^2(m \cdot H + n)}$$

Where $N = 2048$ (window length) and $H = 512$ (hop size). RMS decibels are converted:

$$\text{RMS}_{\text{dB}}(m) = 20 \log_{10}(\text{RMS}(m) + \epsilon)$$

Speaker-agnostic energy is achieved through z-score standardization across all speech frames along the warping path:

$$Z_{\text{RMS}}(m) = \frac{\text{RMS}_{\text{dB}}(m) - \mu_{\text{RMS}}}{\sigma_{\text{RMS}}}$$

$$\Delta_{\text{Energy}}(m) = Z_{\text{RMS, part}}(m) - Z_{\text{RMS, ref}}(m)$$

A flaw of **Subdued Vocal Energy** is flagged when $\Delta_{\text{Energy}}(m) < -1.8$ for $\ge 0.70$ contiguous seconds.

---

### 1.2 Fundamental Frequency ($F_0$) & Relative Semitone Scale
Pitch is extracted using the probabilistic YIN (pYIN) algorithm between 65 Hz ($C_2$) and 400 Hz ($G_4$). Absolute pitch in Hz is converted to a logarithmic musical semitone scale relative to $f_{\text{ref}} = 110.0\text{ Hz}$ ($A_2$):

$$\text{Pitch}_{\text{semitones}}(m) = 12 \cdot \log_2\left(\frac{F_0(m)}{110.0}\right)$$

To remove natural biological pitch register differences (e.g., baritone vs. soprano), intonation is evaluated via mean-centered contours:

$$\widetilde{\text{Pitch}}(m) = \text{Pitch}_{\text{semitones}}(m) - \frac{1}{|\mathcal{V}|} \sum_{k \in \mathcal{V}} \text{Pitch}_{\text{semitones}}(k)$$

Where $\mathcal{V}$ denotes the set of voiced frames. **Pitch Monotony** is triggered when the rolling standard deviation of $\widetilde{\text{Pitch}}$ drops below $1.20\text{ semitones}$ for $\ge 1.20$ seconds.

---

### 1.3 Sequence Alignment & Instantaneous Speaking Velocity
Alignment is achieved using Dynamic Time Warping (DTW) over z-normalized 13-dimensional MFCC vectors:

$$D(i, j) = 1 - \frac{\mathbf{c}_{\text{ref}}(i) \cdot \mathbf{c}_{\text{part}}(j)}{\|\mathbf{c}_{\text{ref}}(i)\| \|\mathbf{c}_{\text{part}}(j)\|}$$

The cumulative cost matrix $\mathbf{\Gamma}$ determines the optimal warping path $\mathcal{W} = \{(w_{\text{ref}}(k), w_{\text{part}}(k))\}_{k=1}^K$.

Instantaneous speaking rate ratio $R(k)$ is the derivative of participant path progression with respect to reference path progression, smoothed over a 31-frame boxcar filter:

$$R(k) = \frac{\Delta w_{\text{part}}(k)}{\Delta w_{\text{ref}}(k)}$$

- **Excessive Speed (Rushed Delivery)**: $R(k) < \frac{1}{1.35} \approx 0.74$ sustained for $\ge 0.60\text{ s}$.
- **Sluggish Delivery**: $R(k) > \frac{1}{0.70} \approx 1.43$ sustained for $\ge 0.80\text{ s}$.

---

## 2. Deductive Scoring Rubric (100 Points)

| Rubric Category | Weight | Deductive Trigger Criteria | Max Penalty / Event |
| :--- | :--- | :--- | :--- |
| **Pacing & Fluency** | 25 pts | Local rate ratio $< 0.74\times$ or $> 1.43\times$ | 6.0 pts |
| **Pause Management** | 20 pts | RMS energy $< -36\text{ dB}$ for $> 1.75\text{ s}$ | 5.0 pts |
| **Vocal Energy & Clarity** | 20 pts | Aligned energy $z$-score delta $< -1.8$ | 5.0 pts |
| **Pitch Expressiveness** | 20 pts | Voiced pitch rolling standard deviation $< 1.2\text{ semitones}$ | 5.0 pts |
| **Emphasis Consistency** | 15 pts | Discrepancies in spectral dynamic range at rhetorical emphasis peaks | 4.0 pts |

### Penalty Scaling Function
For each detected flaw $i$:

$$\text{Penalty}_i = \min\left(P_{\max} \cdot M_{\text{severity}} \cdot \text{clamp}\left(\frac{\text{Duration}_i}{1.0}, 0.5, 1.5\right), P_{\max}\right)$$

Where $M_{\text{severity}} = 1.0$ for Severe flaws, and $M_{\text{severity}} = 0.55$ for Moderate flaws.
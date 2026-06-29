# Figure Specifications

> Specification for the two framework figures used in the paper.
> All text, labels, captions, and legends are in English.
> Both figures follow the IEEE conference paper visual style:
> monochrome-safe palette, vector output (PDF/SVG), 300 DPI for raster.

---

## Figure 1 — Research Workflow

**Filename:** `fig1_research_workflow.pdf` (vector) and `fig1_research_workflow.png` (300 DPI, ≥ 3000 px wide).

**Caption (English):**
> Fig. 1. End-to-end research workflow of the proposed AI-enabled 5G link optimization system, covering channel modeling, AI-based signal processing, intelligent resource scheduling, and performance evaluation.

**Layout:** Top-to-bottom flow with 4 horizontal swim-lanes (senders / channel / AI receiver / control & evaluation). Use rounded rectangles for processes, parallelograms for data, diamonds for decisions, solid arrows for data flow, dashed arrows for control flow.

**Lane 1 — Transmitter**
- Bit stream generation
- Channel coding (LDPC, 5G NR)
- Modulation mapping (BPSK / QPSK / 16QAM / 64QAM / 256QAM)
- OFDM modulation (IFFT, 2048-pt, CP = 512)

**Lane 2 — Wireless Channel (3GPP 38.901)**
- Path loss (UMa / UMi / InH)
- Shadow fading (log-normal, σ = 8 dB)
- Multipath fading (6–12 taps, delay spread 300 ns)
- AWGN, SNR ∈ [−10, 30] dB
- Carrier frequency offset and sampling frequency offset

**Lane 3 — AI Receiver (the three AI modules)**
- AI-ChannelNet (Transformer, 4 layers, 8 heads) — produces CSI
- CNN-LSTM Detector (64 → 128 → 256 filters, Bi-LSTM 128) — symbol detection
- CNN Modulation Recognizer (kernel 7 / 5 / 3) — modulation classification
- Channel decoding (output bits)

**Lane 4 — Intelligent Control & Evaluation**
- PPO scheduler (state / action / reward) — RB and power allocation
- Feedback link to transmitter (closed loop)
- Metric collection: NMSE, BER, SER, accuracy, throughput, Jain fairness, energy efficiency

**Connecting arrows (must be labeled):**
- Solid: data flow
- Dashed: control / feedback
- Color code: blue = data, orange = control, gray = evaluation

**Typography:**
- Title: 10 pt bold, sentence case
- Box labels: 9 pt regular
- Arrows: 8 pt italic
- Caption: 8 pt, two lines max

---

## Figure 2 — Method Architecture

**Filename:** `fig2_method_architecture.pdf` (vector) and `fig2_method_architecture.png` (300 DPI, ≥ 3000 px wide).

**Caption (English):**
> Fig. 2. Method architecture of the proposed system: (a) AI-ChannelNet with multi-head self-attention; (b) CNN-LSTM signal detector; (c) CNN-based modulation recognizer; (d) PPO-based intelligent resource scheduler.

**Layout:** 2×2 grid of sub-figures, each enclosed in a dashed bounding box with a sub-caption (a), (b), (c), (d). All sub-figures share the same input convention (left side) and output convention (right side).

**Sub-figure (a) — AI-ChannelNet (Transformer)**
- Input: received signal y, pilot signal x_p (complex-valued)
- Feature extraction: real, imaginary, amplitude, phase
- Linear embedding (d = 128) + sinusoidal positional encoding
- 4 × Transformer encoder block:
  - Multi-Head Self-Attention (h = 8, d_k = 16)
  - Add & LayerNorm
  - Feed-Forward Network (d → 4d → d, GELU)
  - Add & LayerNorm
- Output: estimated channel matrix Ĥ (real + imaginary)

**Sub-figure (b) — CNN-LSTM Detector**
- Input: received IQ signal, length 128
- Conv1D block 1: 64 filters, k = 3, BN, ReLU
- Conv1D block 2: 128 filters, k = 3, BN, ReLU
- Conv1D block 3: 256 filters, k = 3, BN, ReLU
- Bi-LSTM: 2 layers, 128 hidden units, dropout 0.3
- Global average pooling → Fully connected → Sigmoid
- Output: detected symbol probability

**Sub-figure (c) — CNN Modulation Recognizer**
- Input: IQ signal, 128 samples
- Conv1D: 64 filters, k = 7 → 128 filters, k = 5 → 256 filters, k = 3
- Each block: Conv + BN + ReLU + MaxPool
- Global average pooling → Fully connected (256) → Softmax (5 classes)
- Output: {BPSK, QPSK, 16QAM, 64QAM, 256QAM}

**Sub-figure (d) — PPO Scheduler**
- State s_t: [CQI₁…ₙ, queue₁…ₙ, priority₁…ₙ, resource usage]
- Shared feature extractor: 2 × FC (128, ReLU)
- Policy head π(a|s): FC → Softmax over (users × RBs)
- Value head V(s): FC → scalar
- Action a_t: RB-to-user assignment + power level
- Environment: 3GPP UMa channel, 20 users, 100 RBs
- Reward r_t: w₁·throughput + w₂·Jain fairness + w₃·energy efficiency
  (w₁ = 1.0, w₂ = 0.3, w₃ = 0.2)

**Visual conventions (apply to both figures):**
- Stroke: 1 pt for boxes, 0.75 pt for arrows
- Fill: white boxes with 1 pt colored border; pale tint for "data" objects
- Font: Times New Roman or equivalent serif
- Color palette (color-blind safe):
  - Primary: #1F4E79 (deep blue)
  - Accent: #C0504D (red-brown)
  - Secondary: #4F81BD (mid blue)
  - Highlight: #9BBB59 (olive green)
  - Neutral: #7F7F7F (gray)
- Size: each figure must fit one column (≤ 3.5 in) or two columns (≤ 7.16 in) of IEEE template
- Resolution: 300 DPI minimum; vector PDF preferred

**Terminology consistency (must match paper body):**
| Term used in figure | Same as in paper body? |
|---------------------|------------------------|
| AI-ChannelNet        | ✓ |
| CNN-LSTM Detector    | ✓ |
| Modulation Recognizer | ✓ |
| PPO Scheduler        | ✓ |
| NMSE / BER / SER     | ✓ |
| Resource Block (RB)  | ✓ |
| Channel Quality Indicator (CQI) | ✓ |
| Jain Fairness Index  | ✓ |
| 3GPP UMa / UMi / InH | ✓ |
| 5G NR                | ✓ |

# Figure 1 — Research Workflow Information

> Caption: "Fig. 1. End-to-end research workflow of the proposed AI-enabled 5G link optimization system, covering channel modeling, AI-based signal processing, intelligent resource scheduling, and performance evaluation."

## Layout

Top-to-bottom flow with 4 horizontal swim-lanes. Rounded rectangles for processes, parallelograms for channel effects, solid arrows for data flow, dashed arrows for control/feedback.

## Lane 1 — Transmitter

| Block | Details |
|---|---|
| Bit Stream Generation | Random binary data |
| Channel Coding | LDPC, 5G NR |
| Modulation Mapping | BPSK / QPSK / 16QAM / 64QAM / 256QAM |
| OFDM Modulation | IFFT 2048-pt, CP = 512 |

## Lane 2 — Wireless Channel (3GPP 38.901)

| Block | Parameter |
|---|---|
| Path Loss | UMa / UMi / InH models |
| Shadow Fading | Log-normal, σ = 8 dB |
| Multipath Fading | 6–12 taps, delay spread 300 ns |
| AWGN | SNR ∈ [−10, 30] dB |
| Carrier Frequency Offset + Sampling Frequency Offset | CFO + SFO |

## Lane 3 — AI Receiver

| Module | Architecture | Input | Output |
|---|---|---|---|
| OFDM Demodulator | FFT 2048-pt, CP removal | Rx time-domain signal | Frequency-domain symbols |
| AI-ChannelNet | Transformer, 4 layers, 8 heads, d_model=128, GELU, dropout=0.1 | Pilots + received signal (real/imag) | Estimated channel matrix Ĥ |
| CNN-LSTM Detector | Conv1D 64→128→256 (k=3), Bi-LSTM 128 (2 layers), FC→Sigmoid | Equalized IQ signal (128 samples) | Detected symbol probability |
| CNN Mod. Recognizer | Conv1D (k=7/5/3, filters 64/128/256), MaxPool, GAP, FC 256→128→5, Softmax | IQ signal (128 samples) | Modulation class {BPSK, QPSK, 16QAM, 64QAM, 256QAM} |
| Channel Decoding | LDPC decoder | Detected bits | Recovered bit stream |

## Lane 4 — Intelligent Control & Evaluation

| Component | Details |
|---|---|
| PPO Scheduler | State: [CQI₁…ₙ, queue₁…ₙ, priority₁…ₙ, resource usage]. Action: RB-to-user + power level. Env: 20 users, 100 RBs, 33 dBm max power |
| Resource Allocation | RB & Power allocation |
| Metrics Collection | NMSE, MSE, correlation (channel); accuracy, confusion matrix, per-class accuracy (modulation); BER, SER, EVM (detection); Jain fairness, spectral efficiency, energy efficiency, throughput (scheduling) |
| Closed-loop Feedback | Dashed arrow from PPO Scheduler back to Transmitter |

## Arrows & Connections

| Arrow | Style | Color | Label |
|---|---|---|---|
| Bit Stream → Channel Coding → Modulation → OFDM Mod | Solid | Blue (#1F4E79) | — |
| Tx signal (Transmitter → Channel) | Solid | Blue | "Tx signal" |
| Channel → AI Receiver | Solid | Red-brown (#C0504D) | "Rx signal + pilots" |
| AI-ChannelNet CSI → equalization | Dashed | Blue | "CSI" |
| Receiver → Control | Dotted | Mid-blue (#4F81BD) | "Link metrics" |
| PPO Scheduler → Transmitter (feedback) | Dashed | Olive (#9BBB59) | "Closed-loop feedback" |
| Receiver output | Solid | Blue | "Output bits" |

## Legend

- Blue solid = Data flow
- Green dashed = Control / Feedback
- Blue dotted = Evaluation metrics

## Paper Terminology (must match)

| Term | Verified in paper |
|---|---|
| AI-ChannelNet | ✓ (0-intro.tex L7, 2-methodology.tex L14) |
| CNN-LSTM Detector | ✓ (2-methodology.tex §2.3) |
| CNN Modulation Recognizer | ✓ (2-methodology.tex §2.3, 3-experiments-results.tex L5) |
| PPO Scheduler / RL-Scheduler | ✓ (2-methodology.tex §2.4) |
| NMSE / BER / SER | ✓ |
| Resource Block (RB) | ✓ |
| Channel Quality Indicator (CQI) | ✓ |
| Jain Fairness Index | ✓ |
| 3GPP UMa / UMi / InH | ✓ |
| 5G NR | ✓ |

## OFDM Simulation Parameters (from config.py + paper)

| Parameter | Value |
|---|---|
| Carrier frequency | 3.5 GHz |
| System bandwidth | 100 MHz |
| FFT size | 2048 |
| Cyclic prefix | 512 |
| Data subcarriers | 1200 |
| Channel model | 5G UMa |
| Doppler frequency | 100 Hz |
| Delay spread | 300 ns |
| SNR range | −10 to 30 dB |
| Training samples | 10000 |
| Random seed | 42 |

## PPO Scheduler Parameters (from config.py + agent.py + paper)

| Parameter | Value |
|---|---|
| Algorithm | PPO |
| γ (discount) | 0.99 |
| λ (GAE) | 0.95 |
| ε (clip) | 0.2 |
| Entropy coefficient | 0.01 |
| Learning rate | 3 × 10⁻⁴ |
| Batch size | 256 |
| Users | 20 |
| Resource Blocks | 100 |
| Max power | 33 dBm |
| Throughput weight w₁ | 1.0 |
| Fairness weight w₂ | 0.3 |
| Energy efficiency weight w₃ | 0.2 |
| Time slots | 10 |

## Visual Style (from FIGURES_SPEC.md)

- Font: Times New Roman, serif
- Stroke: 1 pt boxes, 0.75 pt arrows
- Fill: white boxes with colored border
- Data objects: pale tint fill
- Title: 10 pt bold, sentence case
- Box labels: 9 pt regular, 5.5–6 pt for sub-labels
- Arrows: 8 pt italic labels
- Color palette (color-blind safe):
  - Primary: #1F4E79 (deep blue)
  - Accent: #C0504D (red-brown)
  - Secondary: #4F81BD (mid blue)
  - Highlight: #9BBB59 (olive green)
  - Neutral: #7F7F7F (gray)
- Width: ≤ 7.16 in (two columns IEEE)
- Resolution: 300 DPI min, vector PDF preferred

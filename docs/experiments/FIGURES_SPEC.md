# Figure Specifications

> Specification for the two framework figures used in the IEEE conference paper.
> All figure text, labels, legends, and captions must be in English.
> Primary submission artwork should be vector PDF/EPS when possible; PNG is only a preview or fallback.

## IEEE-Oriented Artwork Rules

These rules follow common IEEE Author Center graphics guidance: use publication-ready vector artwork when possible, keep labels readable after column scaling, avoid color-only encoding, and use sufficient raster resolution for line art and mixed graphics.

| Item | Requirement |
|---|---|
| Preferred file type | Vector PDF or EPS for diagrams; keep editable source file separately. |
| Preview / fallback raster | PNG allowed for checking; export at final print size. |
| Resolution | At least 600 DPI for line-art diagrams; 300 DPI is acceptable only for continuous-tone images. Use 600 DPI for these architecture/workflow figures if exported as raster. |
| Width | Single column: about 3.5 in max; double column: about 7.16 in max. These two figures should normally be double-column figures. |
| Font | Times New Roman or IEEE-compatible serif/sans-serif; keep one font family throughout each figure. |
| Final label size | About 8-10 pt after scaling; avoid labels below 6 pt. |
| Line width | About 0.75-1.0 pt after scaling; keep arrows and box borders visible in grayscale. |
| Color | Color-blind safe and grayscale distinguishable; do not use color as the only meaning carrier. |
| Captions | Captions are written in LaTeX, not embedded as figure titles. The figure image itself should not contain "Fig. 1" or full caption text. |
| Subfigure labels | Use `(a)`, `(b)`, etc. inside the figure or as LaTeX subcaptions, consistently. |
| Background | White background; no shadows, gradients, decorative effects, or busy textures. |
| Text style | Sentence case, concise labels, consistent capitalization. |

## Shared Visual Style

| Element | Style |
|---|---|
| Process boxes | White fill, colored border, square or lightly rounded corners. |
| Data/state objects | Very light tint fill; ensure grayscale contrast remains clear. |
| Data-flow arrows | Solid line, deep blue or black. |
| Feedback/control arrows | Dashed line, olive green or dark gray. |
| Metric/evaluation arrows | Dotted or thin solid line, neutral gray. |
| Color palette | Deep blue `#1F4E79`, mid blue `#4F81BD`, red-brown `#C0504D`, olive `#9BBB59`, neutral gray `#7F7F7F`. |
| Mathematical symbols | Use LaTeX-style notation where possible, e.g. `H_hat`, `s_t`, `r_t`, `pi(a|s)`. |

## Figure 1 — Research Workflow

**Recommended files**

- `fig1_research_workflow.pdf` or `.eps`: publication figure.
- `fig1_research_workflow.png`: 600 DPI preview/fallback, exported at final double-column width.

**Recommended caption in LaTeX**

> Fig. 1. Research workflow of the proposed AI-enabled 5G link optimization framework, from OFDM link simulation and wireless channel generation to AI-based channel estimation, signal recognition, PPO scheduling, and metric evaluation.

**Purpose**

Figure 1 is a paper-level workflow figure. It should show how the simulated link, AI receiver modules, scheduler, and metrics connect. It should not be a complete 5G NR protocol-stack figure.

**Layout**

Use four horizontal swim-lanes or a clean left-to-right flow:

| Lane | Include |
|---|---|
| OFDM simulation input | Random bit/symbol generation, modulation mapping, OFDM modulation, pilot setup. |
| Wireless channel | 3.5 GHz, 100 MHz, 5G UMa software channel, AWGN, SNR -10 to 30 dB, Doppler 100 Hz, delay spread 300 ns. |
| AI receiver | OFDM preprocessing, AI-ChannelNet, equalization/receiver evidence, CNN-BiLSTM detector, CNN modulation recognizer. |
| Scheduling and evaluation | Link metrics/state, PPO scheduler, RB allocation, reward, NMSE/detection/recognition/scheduling metrics, feedback arrow. |

**Required labels**

- OFDM simulation
- 5G UMa channel
- Received pilots + IQ samples
- AI-ChannelNet
- CNN-BiLSTM detector
- CNN modulation recognizer
- PPO scheduler
- Metric evaluation
- Closed-loop feedback

**Do not draw**

| Do not include | Reason |
|---|---|
| LDPC encoder / decoder | Not implemented or reported in the current paper experiment path. |
| UMi / InH branches | Current reported channel scenario is 5G UMa only. |
| CFO / SFO correction | Not part of the reported model or metrics. |
| Explicit power-control action head | Current scheduler action is RB-to-user allocation; max power is an environment parameter. |
| End-to-end joint neural training arrow | Current evidence supports a modular closed-loop pipeline, not one jointly trained differentiable model. |

## Figure 2 — Method Architecture

**Recommended files**

- `fig2_method_architecture.pdf` or `.eps`: publication figure.
- `fig2_method_architecture.png`: 600 DPI preview/fallback, exported at final double-column width.

**Recommended caption in LaTeX**

> Fig. 2. Method architecture of the proposed framework: (a) Transformer-based AI-ChannelNet for channel estimation; (b) CNN-BiLSTM detector for binary signal detection; (c) CNN modulation recognizer for five-class modulation recognition; (d) PPO scheduler for resource-block allocation.

**Layout**

Use a 2 x 2 grid of subfigures with consistent left-to-right flow:

| Panel | Module | Include |
|---|---|---|
| (a) | AI-ChannelNet | Real/imag input, linear embedding 2 -> 128, positional encoding, 4 Transformer encoder layers, 8 heads, output projection 128 -> 2. |
| (b) | CNN-BiLSTM detector | Conv1D 64/128/256 with k=3, adaptive pool length 64, BiLSTM hidden 128, FC -> sigmoid. |
| (c) | CNN modulation recognizer | Conv1D kernels 7/5/3 with filters 64/128/256, adaptive pool length 32, dense path, softmax over five classes. |
| (d) | PPO scheduler | State with normalized channel gains and queues, actor/value policy concept, RB allocation action, reward and next state. |

**Important implementation alignment**

| Topic | Correct drawing instruction |
|---|---|
| Transformer activation | Do not label a specific non-default activation. Use "feed-forward activation" or omit activation text. |
| Modulation recognizer filters | Conv1 uses 64 filters, not 65. |
| Dense units | Current best experiment uses enlarged dense capacity: 512 -> 256 -> 5; code default remains configurable. |
| PPO action | Show RB-to-user allocation. Do not draw a separate power-level action head. |
| PPO policy head | Do not draw a single `Linear(128 -> 100)` output as the publication architecture. The environment action is 100 RB decisions, each selecting one of 20 users. |
| Modulation result | Show 76.91% overall accuracy; do not claim 95% modulation recognition. |

## Data Values Allowed In Figures

Use only a few metrics in the diagram. Keep detailed values in tables or text.

| Metric | Value |
|---|---|
| AI-ChannelNet overall NMSE | -20.42 dB |
| Detection accuracy / F1 | 98.08% / 98.04% |
| Modulation overall accuracy | 76.91% |
| 64QAM / 256QAM accuracy | 52.53% / 82.80% |
| PPO reward / Jain fairness | 2.393 / 0.8717 |

## LaTeX Integration Guidance

Use figure captions in the LaTeX source, not inside the image.

```latex
\begin{figure*}[!t]
  \centering
  \includegraphics[width=\textwidth]{fig1_research_workflow.pdf}
  \caption{Research workflow of the proposed AI-enabled 5G link optimization framework.}
  \label{fig:research_workflow}
\end{figure*}
```

For one-column placement, use `figure` and `width=\columnwidth`. For these two figures, `figure*` is recommended because the architecture labels are dense.

## Final Pre-Submission Checklist

- All labels readable at final print size.
- No label below 6 pt after scaling.
- Vector PDF/EPS available; PNG fallback exported at 600 DPI.
- Figure remains understandable in grayscale.
- No color-only distinctions.
- No embedded title or full caption in the artwork.
- Subfigure labels are consistent.
- Figure text matches paper terminology exactly.
- No unsupported modules: LDPC, UMi/InH, CFO/SFO, explicit power action, joint end-to-end training.
- Metrics match `docs/experiments/real/summary.json` and `docs/experiments/real/data/*.csv`.

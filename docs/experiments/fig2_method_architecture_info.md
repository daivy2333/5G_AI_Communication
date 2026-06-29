# Figure 2 — Method Architecture Information

> Caption: "Fig. 2. Method architecture of the proposed system: (a) AI-ChannelNet with multi-head self-attention; (b) CNN-LSTM signal detector; (c) CNN-based modulation recognizer; (d) PPO-based intelligent resource scheduler."

## Layout

2×2 grid of sub-figures, each in a dashed bounding box with sub-caption (a), (b), (c), (d). All share left-side → right-side input/output convention.

## Sub-figure (a) — AI-ChannelNet (Transformer Channel Estimator)

Source: `channel_estimation/models.py` class `TransformerChannelEstimatorModel`

```
Input: y ∈ ℂ^(K×T) (received pilots)
  │
  ▼
┌─────────────────────────────────────┐
│ Feature Extraction                  │
│ x_i = [Re(Y_i), Im(Y_i)] ∈ ℝ²      │  (Eq. 15: complex→real)
└─────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────┐
│ Linear Embedding                    │
│ W_e ∈ ℝ^(128×2), b_e ∈ ℝ^128       │  (Eq. 4: embedding)
│ → z_i⁽⁰⁾ ∈ ℝ^128                   │
└─────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────┐
│ Sinusoidal Positional Encoding      │
│ PE(pos, 2j)   = sin(pos/10000^(2j/d)) │
│ PE(pos, 2j+1) = cos(pos/10000^(2j/d)) │
│ d = 128, max_len = 5000             │
└─────────────────────────────────────┘
  │
  ▼
╔═════════════════════════════════════╗ ×4 layers
║ Transformer Encoder Block           ║
║ ┌───────────────────────────────┐   ║
║ │ Multi-Head Self-Attention     │   ║
║ │ h = 8 heads, d_k = d_v = 16  │   ║
║ │ d_model = 128                 │   ║
║ │ Q,K,V = Linear(128→128) × 3  │   ║
║ │ Attn(Q,K,V) = softmax(QKᵀ/√d_k)·V │
║ └───────────────────────────────┘   ║
║   │  + (Residual)                   ║
║ ┌───────────────────────────────┐   ║
║ │ Layer Normalization           │   ║
║ └───────────────────────────────┘   ║
║   │                                 ║
║ ┌───────────────────────────────┐   ║
║ │ Feed-Forward Network          │   ║
║ │ Linear(128 → 512) → GELU     │   ║
║ │ Linear(512 → 128)             │   ║
║ │ Dropout = 0.1                 │   ║
║ └───────────────────────────────┘   ║
║   │  + (Residual)                   ║
║ ┌───────────────────────────────┐   ║
║ │ Layer Normalization           │   ║
║ └───────────────────────────────┘   ║
╚═════════════════════════════════════╝
  │
  ▼
┌─────────────────────────────────────┐
│ Output Projection                   │
│ Linear(128 → 2)                     │
└─────────────────────────────────────┘
  │
  ▼
Output: Ĥ ∈ ℂ^(K×T) (estimated channel, real+imag)
```

### Key Parameters

| Parameter | Value |
|---|---|
| Input dim | 2 (real, imag) |
| Embedding dim (d_model) | 128 |
| Attention heads (h) | 8 |
| Head dim (d_k) | 16 |
| Transformer layers | 4 |
| FFN hidden dim | 512 (4 × d_model) |
| Activation | GELU |
| Dropout | 0.1 |
| Positional encoding | Sinusoidal |
| Output dim | 2 (real, imag) |
| Loss | MSE: L_H = (1/KT) Σ|Ĥ - H|₂² |
| Batch size | 64 |
| Learning rate | 1 × 10⁻⁴ |
| Optimizer | Adam |
| Training epochs | 100 |

---

## Sub-figure (b) — CNN-LSTM Detector (SignalNet)

Source: `signal_detection/detector.py` class `CNNLSTMNet`

```
Input: IQ signal ∈ ℝ^(T×2) (e.g., 128 samples)
  │
  ▼ transpose → (batch, 2, T)
  │
┌─────────────────────────────────────┐
│ Conv1D Block 1                      │
│ Conv1D(2 → 64, kernel=3, pad=1)    │
│ BatchNorm1d(64)                     │
│ ReLU                                │
└─────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────┐
│ Conv1D Block 2                      │
│ Conv1D(64 → 128, kernel=3, pad=1)  │
│ BatchNorm1d(128)                    │
│ ReLU                                │
└─────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────┐
│ Conv1D Block 3                      │
│ Conv1D(128 → 256, kernel=3, pad=1) │
│ BatchNorm1d(256)                    │
│ ReLU                                │
└─────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────┐
│ Adaptive Average Pooling            │
│ → output_size = 64                  │
└─────────────────────────────────────┘
  │
  ▼ transpose → (batch, 64, 256)
  │
┌─────────────────────────────────────┐
│ Bi-LSTM (2 layers)                  │
│ input_size = 256                    │
│ hidden_size = 128                   │
│ bidirectional = True                │
│ → output: (batch, 64, 256)          │
│   h_t = [→h_t; ←h_t]  (Eq. 8)      │
└─────────────────────────────────────┘
  │
  ▼ take last timestep → (batch, 256)
  │
┌─────────────────────────────────────┐
│ Dropout(0.3)                        │
│ Linear(256 → 128) → ReLU            │
│ Dropout(0.3)                        │
│ Linear(128 → 1) → Sigmoid           │
└─────────────────────────────────────┘
  │
  ▼
Output: p ∈ [0,1] (detection probability)
```

### Key Parameters

| Parameter | Value |
|---|---|
| CNN filters | [64, 128, 256] |
| CNN kernel size | 3 |
| CNN padding | 1 (same) |
| Adaptive pool output | 64 |
| Bi-LSTM layers | 2 |
| Bi-LSTM hidden | 128 per direction |
| Bi-LSTM output | 256 (concatenated) |
| FC hidden | 128 |
| Final output | 1 (sigmoid) |
| Dropout | 0.3 |
| Loss | Binary Cross-Entropy |
| Batch size | 128 |
| Learning rate | 1 × 10⁻³ |
| Epochs | 50 |

---

## Sub-figure (c) — CNN Modulation Recognizer

Source: `signal_detection/recognizer.py` class `CNNModulationNet`

```
Input: IQ signal ∈ ℝ^(T×2) (128 samples)
  │
  ▼ transpose → (batch, 2, 128)
  │
┌─────────────────────────────────────┐
│ Conv1D Block 1                      │
│ Conv1D(2 → 64, kernel=7, pad=3)    │
│ BatchNorm1d(64)                     │
│ ReLU                                │
│ MaxPool1d(kernel=2)                 │
└─────────────────────────────────────┘
  │ → (batch, 64, 64)
  ▼
┌─────────────────────────────────────┐
│ Conv1D Block 2                      │
│ Conv1D(64 → 128, kernel=5, pad=2)  │
│ BatchNorm1d(128)                    │
│ ReLU                                │
│ MaxPool1d(kernel=2)                 │
└─────────────────────────────────────┘
  │ → (batch, 128, 32)
  ▼
┌─────────────────────────────────────┐
│ Conv1D Block 3                      │
│ Conv1D(128 → 256, kernel=3, pad=1) │
│ BatchNorm1d(256)                    │
│ ReLU                                │
│ MaxPool1d(kernel=2)                 │
└─────────────────────────────────────┘
  │ → (batch, 256, 16)
  ▼
┌─────────────────────────────────────┐
│ AdaptiveAvgPool1d(32)               │
└─────────────────────────────────────┘
  │ → flatten: (batch, 256×32 = 8192)
  ▼
┌─────────────────────────────────────┐
│ Dropout(0.4)                        │
│ Linear(8192 → 256) → ReLU           │
│ Dropout(0.4)                        │
│ Linear(256 → 128) → ReLU            │
│ Dropout(0.4)                        │
│ Linear(128 → 5)                     │
└─────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────┐
│ Softmax                             │
│ p(c|x) = exp(o_c)/Σexp(o_c')        │
└─────────────────────────────────────┘
  │
  ▼
Output: {BPSK, QPSK, 16QAM, 64QAM, 256QAM}
```

### Key Parameters

| Parameter | Value |
|---|---|
| Conv1 kernel | 7 (65 filters) |
| Conv2 kernel | 5 (128 filters) |
| Conv3 kernel | 3 (256 filters) |
| MaxPool | 2 |
| AdaptiveAvgPool output | 32 |
| Dense units | 256 → 128 |
| Output classes | 5 |
| Dropout | 0.4 |
| Loss | Cross-Entropy |
| Batch size | 128 |
| Learning rate | 1 × 10⁻³ |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| Epochs | 50 |

---

## Sub-figure (d) — PPO Scheduler (RL-Scheduler)

Source: `resource_scheduling/agent.py` class `PPOScheduler`, `PPONetwork`

```
State s_t ∈ ℝ^40
  │
  ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐
  │ CQI₁…₂₀   │ Queue₁…₂₀           │
  │ (norm.     │ (norm. queue         │
  │  channel   │  length in bits)     │
  │  gains)    │                      │
  └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
  │
  ▼
╔══════════════════════════════════════╗
║ Shared Feature Extractor             ║
║ Linear(40 → 128) → ReLU             ║
║ Linear(128 → 128) → ReLU            ║
╚══════════════════════════════════════╝
  │
  ├────────────────────┐
  ▼                    ▼
┌──────────────┐  ┌──────────────┐
│ Policy Head  │  │ Value Head   │
│ (Actor)      │  │ (Critic)     │
│              │  │              │
│ Linear       │  │ Linear       │
│ (128 → |A|)  │  │ (128 → 1)    │
│              │  │              │
│ Softmax      │  │ → V(s) scalar│
│              │  │              │
│ → π(a|s)     │  │              │
│  distribution│  │              │
└──────────────┘  └──────────────┘
  │                    │
  ▼                    ▼
Action a_t                Value V(s_t)
(RB allocation            (state value
 per RB to user)          for advantage)
  │
  ▼
┌──────────────────────────────────────┐
│ Environment (3GPP UMa)               │
│  • 20 users, 100 RBs                 │
│  • Max power: 33 dBm                 │
│  • Time slots: 10 per episode        │
│                                      │
│ Reward r_t:                          │
│   r_t = 1.0 × Throughput             │
│       + 0.3 × Jain Fairness          │
│       + 0.2 × Energy Efficiency      │
│                                      │
│ Throughput: T_t = Σ transmitted bits │
│ Fairness: J = (ΣR_u)²/(N·ΣR_u²)     │
│ Energy eff: E = ΣR_u / P_total       │
└──────────────────────────────────────┘
  │
  ▼
Next state s_{t+1}, reward r_t, done
```

### PPO Update

```
For each training batch (size=256):
  1. Collect trajectories {s, a, r, log_prob, value}
  2. Compute advantages A_t via GAE(λ=0.95)
  3. Compute returns R_t (discounted, γ=0.99)
  4. For K epochs:
     a. Compute new log_probs, entropy, values
     b. Ratio ρ = exp(log_prob_new - log_prob_old)
     c. L_PPO = -min(ρ·A, clip(ρ, 0.8, 1.2)·A)
     d. L_value = MSE(V_new, R)
     e. L_total = L_PPO + 0.5·L_value + 0.01·entropy
     f. Gradient step (clip grad norm 0.5)
```

### Key Parameters

| Parameter | Value |
|---|---|
| State dim | 40 (20 CQI + 20 Queue) |
| Shared FC | 128 → 128 (ReLU) |
| Policy head | Linear(128 → 100) → Softmax |
| Value head | Linear(128 → 1) |
| γ (discount) | 0.99 |
| λ (GAE) | 0.95 |
| ε (clip) | 0.2 |
| Entropy coeff | 0.01 |
| Learning rate | 3 × 10⁻⁴ |
| Batch size | 256 |
| PPO epochs | 4 |
| Users | 20 |
| RBs | 100 |
| Max power | 33 dBm |
| Reward weights | w_TP=1.0, w_fair=0.3, w_energy=0.2 |

---

## Visual Conventions (from FIGURES_SPEC.md)

- Stroke: 1 pt boxes, 0.75 pt arrows
- Fill: white boxes with 1 pt colored border
- Font: Times New Roman, serif
- Sub-captions: (a), (b), (c), (d) in bold
- Color palette (color-blind safe):
  - Primary: #1F4E79 (deep blue)
  - Accent: #C0504D (red-brown)
  - Secondary: #4F81BD (mid blue)
  - Highlight: #9BBB59 (olive green)
  - Neutral: #7F7F7F (gray)
- Size per sub-figure: ~3.5 in wide each (half-column)
- Overall figure: ≤ 7.16 in × ~5 in (two columns)
- Resolution: 300 DPI min, vector PDF preferred
- All text in English, sentence case

## Terminology Consistency (must match paper)

| Figure Term | Paper Term | Verified |
|---|---|---|
| AI-ChannelNet | AI-ChannelNet | ✓ |
| Multi-Head Self-Attention | multi-head self-attention | ✓ |
| Feed-Forward Network | Feed-Forward Network | ✓ |
| Layer Normalization | LayerNorm | ✓ |
| CNN-LSTM Detector | SignalNet / CNN-LSTM detector | ✓ (paper uses both) |
| Bi-LSTM | BiLSTM | ✓ |
| Modulation Recognizer | Modulation Recognizer / recognition branch | ✓ |
| PPO Scheduler | RL-Scheduler / PPO scheduler | ✓ |
| Policy Head / Value Head | Policy head / Value head | ✓ (code only) |
| GAE | Generalized Advantage Estimation | ✓ |

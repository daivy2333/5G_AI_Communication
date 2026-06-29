# 5G无线通信链路+AI创新设计项目

本项目是针对"赛项一: 5G+软件无线电创新设计"中"无线通信链路+AI设计方向"的完整实现。项目完全基于软件仿真，无需硬件设备，专注于AI算法在5G通信系统中的应用。
组长jqq，组员lw，whj
---

## 核心技术模块

| 模块 | 技术方案 | 性能提升 |
|------|---------|----------|
| AI信道估计 | Transformer架构 | NMSE -20.42 dB |
| 信号检测与调制识别 | CNN-LSTM + CNN | 检测F1 98.04%，调制识别76.91% |
| 智能资源调度 | PPO强化学习 | 平均奖励2.393，Jain公平性0.8717 |
| 可视化仿真平台 | Flask Web平台 | 实时监控 |

---

## 目录结构

```
5G_AI_Communication/
├── docs/                      # 📚 文档目录
│   ├── README.md              # 文档入口
│   └── ALGORITHMS.md          # 算法架构（含Mermaid图表）
├── channel_estimation/        # AI信道估计模块
│   ├── models.py              # Transformer/CNN模型
│   ├── trainer.py             # 训练器
│   └── data_generator.py      # 数据生成
├── signal_detection/          # 信号检测模块
│   ├── detector.py            # CNN-LSTM检测器
│   ├── recognizer.py          # 调制识别网络
│   └── modulator.py           # 调制信号生成
├── resource_scheduling/       # 资源调度模块
│   ├── agent.py               # PPO/DQN智能体
│   └── environment.py         # 仿真环境
├── simulation/                 # 链路仿真模块
│   └── transceivers.py        # OFDM收发机
├── utils/                      # 性能评估
│   └── metrics.py             # 指标计算
├── config.py                  # 配置文件
└── requirements.txt           # 依赖列表
```

---

## 快速开始

### 1. 环境配置

```bash
pip install -r requirements.txt
```

### 2. 运行完整仿真

```bash
python main.py --mode full
```

### 3. 运行单个模块

```bash
# AI信道估计训练
python -m channel_estimation.trainer --epochs 100

# 信号检测与识别测试
python -m signal_detection.detector --mode test

# 资源调度训练 (PPO)
python -m resource_scheduling.agent --algorithm ppo --timesteps 50000
```

### 4. 启动Web可视化平台

```bash
# 启动Dashboard
python3 visualization/dashboard.py

# 或作为模块运行
python3 -m visualization.dashboard
```

访问地址: **http://127.0.0.1:5000**

**功能特性**:
- 实时星座图可视化（支持BPSK/QPSK/16QAM/64QAM）
- OFDM频谱动态显示
- 信道响应展示
- 性能指标实时监控（NMSE、准确率、吞吐量、BER）
- SNR和调制方式在线配置

**关闭Dashboard**:
```bash
# 前台运行: Ctrl+C
# 后台运行: kill -9 $(lsof -t -i:5000)
```

---

## 📚 文档导航

详细技术文档请参阅 `docs/` 目录：

| 文档 | 说明 |
|------|------|
| [docs/SOLUTION.md](./docs/SOLUTION.md) | 技术方案 - 项目概述、算法设计、系统架构 |
| [docs/ALGORITHMS.md](./docs/ALGORITHMS.md) | 算法架构 - Mermaid流程图、数据流设计 |
| [docs/PERFORMANCE.md](./docs/PERFORMANCE.md) | 性能参数 - 测试结果、NMSE对比、吞吐量 |

---

## 算法说明

详细算法架构请参阅 **[docs/ALGORITHMS.md](./docs/ALGORITHMS.md)**，包含完整Mermaid图表。

### 1. AI信道估计 (AI-ChannelNet)

基于Transformer架构的端到端信道估计网络：

- **网络**: 4层Transformer，8头注意力，GELU激活
- **输入**: 导频信号 + 接收信号
- **输出**: 信道估计矩阵
- **优势**: 高速移动场景优于传统LS/MMSE

### 2. 信号检测与调制识别 (SignalNet)

CNN-LSTM混合网络架构：

- **CNN**: 三级卷积 (64/128/256)，特征提取
- **LSTM**: 2层Bi-LSTM，时序建模
- **支持**: BPSK, QPSK, 16QAM, 64QAM, 256QAM
- **当前结果**: 信号检测F1 98.04%，五类调制识别总体准确率76.91%

### 3. 智能资源调度 (RL-Scheduler)

PPO强化学习算法：

- **状态**: 用户CQI, 队列长度, 业务优先级
- **动作**: RB分配, 功率等级
- **奖励**: 吞吐量 + 公平性 + 能效
- **参数**: γ=0.99, λ=0.95, ε=0.2

---

## 性能评估

详细性能参数请参阅 **[docs/PERFORMANCE.md](./docs/PERFORMANCE.md)**

| 模块 | 指标 | 基线/补充指标 | AI方法 | 说明 |
|------|------|---------|--------|------|
| 信道估计 | NMSE | LMMSE -0.70dB | -20.42dB | AI-ChannelNet |
| 信号检测 | F1 Score | - | 98.04% | CNN-BiLSTM |
| 调制识别 | 准确率 | 64QAM 52.53% | 76.91% | 高阶QAM仍需优化 |
| 资源调度 | 平均奖励/公平性 | Jain 0.8717 | 2.393 | PPO算法 |

### 评估指标

| 模块 | 指标函数 |
|------|---------|
| 信道估计 | NMSE, MSE, 相关性 |
| 调制识别 | 准确率, 混淆矩阵, 各类别准确率 |
| 信号检测 | BER, SER, EVM |
| 资源调度 | Jain公平指数, 频谱效率, 能效 |

---

## 技术栈

- **Python**: 3.8+
- **深度学习**: PyTorch (必需)
- **科学计算**: NumPy, SciPy (基础运算)
- **可视化**: Matplotlib
- **Web**: Flask
- **强化学习**: stable-baselines3


---

## 5G系统参数

| 参数 | 数值 | 符合标准 |
|------|------|----------|
| 载波频率 | 3.5 GHz | C波段5G |
| 系统带宽 | 100 MHz | 3GPP规范 |
| FFT大小 | 2048 | OFDM标准 |
| 循环前缀 | 512 | 抗多径 |
| 数据子载波 | 1200 | 有效带宽 |

---

## 许可证

MIT License

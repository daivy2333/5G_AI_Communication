# SNAPSHOT.md — 项目状态快照

> 最后更新: 2026-06-29
> 当前阶段: 实验完成 — 全部 4 模块真实训练数据已收集

---

## 项目结构树

```
5G_AI_Communication/
├── .claude/                    # AI 命令与技能
│   ├── commands/opsx/          # OpenSpec 命令
│   └── skills/                 # OpenSpec 技能
├── .claude/docs/               # 项目状态文档
│   ├── SNAPSHOT.md             # 本文件
│   └── tasks.md                # 任务追踪
├── .codegraph/                 # CodeGraph 代码索引
├── channel_estimation/         # AI 信道估计模块
│   ├── __init__.py
│   ├── models.py               # Transformer/CNN 模型
│   ├── trainer.py              # 训练器 (已修复: pilot 注入 + 标准化)
│   └── data_generator.py       # 数据生成 (已修复: FFT 维度)
├── signal_detection/           # 信号检测与调制识别
│   ├── __init__.py
│   ├── detector.py             # CNN-LSTM 检测器
│   ├── recognizer.py           # 调制识别网络 (已修复: shuffle split)
│   └── modulator.py            # 调制信号生成 (已修复: 移除 seed=idx)
├── resource_scheduling/        # 智能资源调度
│   ├── __init__.py
│   ├── agent.py                # PPO/DQN 智能体
│   └── environment.py          # 仿真环境
├── simulation/                 # 链路仿真
│   ├── __init__.py
│   └── transceivers.py        # OFDM 收发机
├── utils/                      # 性能评估
│   └── metrics.py              # 指标计算
├── visualization/              # 可视化平台
│   ├── __init__.py
│   └── dashboard.py            # Flask Web 仪表盘
├── docs/                       # 项目文档
│   ├── SOLUTION.md
│   ├── ALGORITHMS.md
│   ├── PERFORMANCE.md
│   └── experiments/            # 实验数据
│       ├── collect_experiment_data.py  # 真实数据收集脚本
│       ├── real/               # 真实训练数据
│       │   ├── data/           # CSV 数据文件
│       │   ├── figures/        # 训练曲线图 (300 DPI)
│       │   ├── logs/           # 训练日志
│       │   └── summary.json    # 汇总指标
│       ├── fig1_research_workflow_info.md   # 研究流程图描述
│       └── fig2_method_architecture_info.md # 架构图描述
├── openspec/                   # OpenSpec 规范管理
│   ├── config.yaml
│   ├── specs/
│   └── changes/
├── main.py                     # 主入口
├── config.py                   # 全局配置
├── requirements.txt            # 依赖列表
├── README.md                   # 项目说明
└── CLAUDE.md                   # AI 规则入口
```

---

## 技术栈

| 类别 | 技术 | 版本 |
|------|------|------|
| 语言 | Python | 3.10.12 |
| 深度学习 | PyTorch | >=1.12.0 |
| 科学计算 | NumPy, SciPy | >=1.21.0, >=1.7.0 |
| 机器学习 | scikit-learn | >=1.0.0 |
| 强化学习 | stable-baselines3 | >=1.5.0 |
| Web 框架 | Flask, flask-socketio | >=2.0.0 |
| 可视化 | Matplotlib, seaborn | >=3.5.0, >=0.11.0 |
| 代码索引 | CodeGraph | 0.9.9 |
| 规范管理 | OpenSpec | 1.4.0 |

---

## Git 状态

| 项目 | 信息 |
|------|------|
| 当前分支 | main |
| 仓库状态 | 已修改: trainer.py (patience可配置), data_generator.py (FFT修复), recognizer.py (shuffle+FocalLoss+Grouped), modulator.py (seed修复), detector.py (seed修复) |

---

## 关键文件

| 文件 | 用途 |
|------|------|
| `main.py` | 主入口，全链路仿真编排 |
| `config.py` | 全局配置参数 |
| `docs/experiments/collect_experiment_data.py` | 真实数据收集脚本（v3 过采样方案） |
| `docs/experiments/real/` | 真实训练数据输出目录 |
| `docs/experiments/fig1_research_workflow_info.md` | 研究流程图描述 |
| `docs/experiments/fig2_method_architecture_info.md` | 架构图描述 |
| `requirements.txt` | Python 依赖 |
| `CLAUDE.md` | AI 规则入口 |
| `README.md` | 项目说明 |

---

## 当前工作

### 已完成

1. **Bug 修复 — 5 个关键缺陷** (learned/spec.md L01-L05)
2. **真实实验数据 — 全部 4 模块** (v3 最优方案):

| 模块 | 指标 | 数值 | vs 基线 |
|------|------|------|---------|
| 信道估计 | NMSE | **-20.42 dB** | LMMSE: -0.70 dB |
| 信号检测 | F1 Score | **98.04%** | — |
| 调制识别 | Overall Acc | **76.91%** | 256QAM 82.80%, 64QAM 52.53% |
| 资源调度 | Avg Reward | **2.39** | PPO 收敛稳定 |

3. **调制识别优化尝试**：
   - v1 CE 均衡 → v2 3000/类 → **v3 FC=512+过采样 (76.91%)** ✅
   - v4 Focal Loss → 失败 (64QAM 崩)
   - v5 Grouped 多出口 → 失败 (QAM 头全 0%)
4. **文档体系更新** — SNAPSHOT, PERFORMANCE(V2), tasks, learned

### 待办

- 论文 Fig1/Fig2 绘制
- 论文 LaTeX 集成真实数据

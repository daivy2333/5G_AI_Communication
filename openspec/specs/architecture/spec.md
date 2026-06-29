## Purpose

定义项目的架构决策和设计原则，指导开发过程中的技术选型和系统设计。

## Requirements

### Requirement: 架构决策记录

所有重要的架构决策 SHALL 以 ADR（Architecture Decision Record）形式记录，包含决策内容、原因、影响和替代方案。

#### Scenario: 记录新决策

- **WHEN** 开发者做出影响系统架构的决策（如选择 Transformer 替代 LS/MMSE 做信道估计、设计模块边界、确定通信协议）
- **THEN** 必须创建新的 ADR 条目，包含：决策标题、决策内容、决策原因、影响范围、替代方案

#### Scenario: 查询已有决策

- **WHEN** 开发者需要了解某个架构选择的原因
- **THEN** 可以通过阅读本文件快速定位相关 ADR

### Requirement: 架构原则遵循

所有架构设计 SHALL 遵循项目定义的架构原则。

#### Scenario: 评估设计方案

- **WHEN** 开发者提出新的设计方案
- **THEN** 方案必须符合 SOLID、DRY、关注点分离等原则，不符合时需说明理由

### Requirement: 模块边界清晰

系统模块之间 SHALL 有清晰的边界和接口定义。

#### Scenario: 新增模块依赖

- **WHEN** 模块 A 需要依赖模块 B
- **THEN** 必须通过明确定义的接口交互，禁止直接访问内部实现

### Requirement: 现有架构决策

项目当前架构决策 SHALL 记录，确保设计可追溯。

#### Scenario: 模块化设计

- **WHEN** 项目架构设计
- **THEN** 采用模块化架构：channel_estimation（信道估计）、signal_detection（信号检测）、resource_scheduling（资源调度）、simulation（链路仿真）、utils（工具）、visualization（可视化），各模块独立可测试

#### Scenario: AI 框架选择

- **WHEN** 选择深度学习框架
- **THEN** 使用 PyTorch 作为统一深度学习框架，Transformer 用于信道估计，CNN-LSTM 用于信号检测，PPO 用于资源调度

#### Scenario: 仿真优先策略

- **WHEN** 项目开发与验证
- **THEN** 完全基于软件仿真（无硬件依赖），使用 NumPy/SciPy 生成合成信道数据和信号数据

---

<!-- A01 --> ### 2026-06-29 - 实验产物以真实训练脚本为论文数据边界

**决策**: T06/T07/T09 的论文数据应以 `docs/experiments/collect_experiment_data.py` 生成的 `docs/experiments/real/` 产物为边界，而不是直接引用模块 demo 输出或旧的公式生成脚本。

**原因**: 该脚本统一调用项目内置训练器，集中导出 CSV、日志、图和 summary，能把信道估计、调制识别、资源调度的实验口径固定下来。

**影响**: 后续论文集成应优先读取 `docs/experiments/real/data/*.csv` 和 `summary.json`；资源调度必须补齐 `modules.resource_scheduling` 后再进入论文结果表。

**替代方案**: 直接运行 `main.py` 或各模块 demo。该方案适合演示，但输出结构不稳定，不适合作为论文数据源。

<!-- A02 --> ### 2026-06-29 - 高阶 QAM 优化以类别指标驱动

**决策**: T07 的改进目标以 64QAM/256QAM 类别准确率和 SNR 分层准确率为主，不以 overall accuracy 单独验收。

**原因**: 当前 BPSK/QPSK 已接近可用，主要缺口集中在 64QAM/256QAM；overall accuracy 会被低阶类别稀释，不能反映任务真实风险。

**影响**: 后续训练脚本应保留类别准确率输出，并补充 SNR 分层落盘；数据增强和样本扩充优先服务高阶 QAM。

**替代方案**: 只提高整体准确率。该方案可能通过低阶类别收益掩盖高阶 QAM 失败，不适合 T07。

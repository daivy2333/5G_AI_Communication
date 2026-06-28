## Purpose

记录项目开发过程中学到的知识，避免重复探索，加速问题解决。

## Requirements

### Requirement: API 路径记录

项目中使用的关键 API 路径 SHALL 记录，包含用途和使用示例。

#### Scenario: 发现新 API

- **WHEN** 开发者发现或使用了新的 API 端点
- **THEN** 必须记录到本文件，包含：API 路径、用途、请求/响应格式

### Requirement: 踩坑经验记录

遇到的技术陷阱和解决方案 SHALL 记录，防止重复踩坑。

#### Scenario: 解决棘手问题

- **WHEN** 开发者花费大量时间解决了一个技术问题
- **THEN** 必须记录踩坑档案，包含：症状、根因、解决方案、预防措施

### Requirement: 技巧模式记录

有效的开发技巧和模式 SHALL 记录，促进知识共享。

#### Scenario: 发现高效做法

- **WHEN** 开发者发现了一种高效的开发技巧或模式
- **THEN** 必须记录到技巧模式区，包含：技巧名称、适用场景、使用方法

### Requirement: 文件速查表

关键文件和目录的位置 SHALL 记录，加速代码导航。

#### Scenario: 定位关键文件

- **WHEN** 开发者频繁访问某些文件或目录
- **THEN** 必须记录到文件速查表，包含：文件路径、用途、关键内容

---

## 文件速查表

| 文件 | 用途 |
|------|------|
| `main.py` | 主入口，全链路仿真编排 |
| `config.py` | 全局配置（系统参数、模型超参、路径） |
| `channel_estimation/models.py` | Transformer/CNN 信道估计模型 |
| `channel_estimation/trainer.py` | 信道估计训练器 |
| `channel_estimation/data_generator.py` | 信道数据生成 |
| `signal_detection/detector.py` | CNN-LSTM 信号检测器 |
| `signal_detection/recognizer.py` | 调制识别网络 |
| `signal_detection/modulator.py` | 调制信号生成 |
| `resource_scheduling/agent.py` | PPO/DQN 强化学习智能体 |
| `resource_scheduling/environment.py` | 资源调度仿真环境 |
| `simulation/transceivers.py` | OFDM 收发机 |
| `utils/metrics.py` | 性能指标计算 |
| `visualization/dashboard.py` | Flask Web 可视化平台 |

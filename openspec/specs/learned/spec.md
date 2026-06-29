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
| `channel_estimation/trainer.py` | 信道估计训练器 (已修复: CE-1, CE-3) |
| `channel_estimation/data_generator.py` | 信道数据生成 (已修复: CE-2) |
| `signal_detection/detector.py` | CNN-LSTM 信号检测器 (已修复: MR-2) |
| `signal_detection/recognizer.py` | 调制识别网络 (已修复: MR-1) |
| `signal_detection/modulator.py` | 调制信号生成 (已修复: MR-2) |
| `resource_scheduling/agent.py` | PPO/DQN 强化学习智能体 |
| `resource_scheduling/environment.py` | 资源调度仿真环境 |
| `simulation/transceivers.py` | OFDM 收发机 |
| `utils/metrics.py` | 性能指标计算 |
| `docs/experiments/collect_experiment_data.py` | 真实数据收集脚本 |

---

## 踩坑记录

### <!-- L01 --> CE-1: 导频未传入 Transformer（trainer.py:279-280）

**症状**: AI-ChannelNet 训练 loss 完全平坦（4.31→4.26，26 epoch 几乎不动），NMSE=0.08 dB 弱于 LS 基线。

**根因**: `_prepare_pytorch_input()` 中 Transformer 分支只传入 received signal [real, imag]，但未传入 transmitted pilot。接收信号 = H × X + N，模型看到 Y 不知道 X，无法分离信道和调制。

**解决**: 在数据准备层用 `received × conj(pilot)` 消去 QPSK 调制，使输入变为纯信道+噪声信号。不需改模型架构。

**预防**: 检查神经网络输入是否包含任务所需的所有必要信息。信道估计任务必须有导频信息。

### <!-- L02 --> CE-2: FFT 维度错误（data_generator.py:275-278）

**症状**: 同上，进一步恶化信道估计效果。

**根因**: `np.fft.fft(channel_true, axis=1)` 对 16 抽头 CIR 做 16-pt FFT，然后零填充到 150。FFT 频率 bin 与导频子载波位置不对应（16-pt bin 在 subcarrier 0,128,256...，导频在 0,8,16...）。

**解决**: 先将 CIR 零填充到目标长度，再做 FFT：`np.fft.fft(h_time_padded, axis=1)`，使频率响应与导频位置对齐。

### <!-- L03 --> CE-3: 无输入归一化（trainer.py:279-281）

**症状**: SNR 跨 40 dB（-10~30dB），输入量级波动 10000 倍，Transformer 难以学习。

**解决**: 用 `received × conj(pilot)` 预处理后，天然将输入归一化到信道+噪声的量级范围。导频功率归一化（QPSK/√2）保证输入量级稳定。

### <!-- L04 --> MR-1: 未 shuffle 就切分训练/验证集（recognizer.py:480-484）

**症状**: 调制识别 val accuracy = 0%，val loss 从 1.3 爆炸到 18+。

**根因**: `generate_dataset()` 按类别顺序生成（BPSK→QPSK→16QAM→64QAM→256QAM），80/20 切分后验证集全是 256QAM。模型训练 4 类，验证 1 类未见过的类。

**解决**: 切分前用 `np.random.permutation(len(train_data))` 打乱索引。

### <!-- L05 --> MR-2: seed=idx 全局 RNG 重置导致数据泄漏（modulator.py:240-241）

**症状**: BPSK 测试准确率虚高 100%（199/200 测试样本与训练样本完全相同）。

**根因**: `generate_iq_samples(seed=idx)` 对每个样本调用 `np.random.seed(idx)` 重置全局 RNG。训练和测试从 idx=0 开始，相同 idx 产生相同噪声/signal。

**解决**: 移除 `seed` 参数，依赖全局 RNG 自然状态。

### <!-- L06 --> T06: PPO 调度验证输出缺失的定位路径

**发现**: `docs/experiments/real/summary.json` 当前没有 `modules.resource_scheduling`，`docs/experiments/real/data/` 也没有 `resource_scheduling_reward.csv`。T06 应通过 `docs/experiments/collect_experiment_data.py::collect_resource_scheduling()` 补跑。

**关键路径**: `collect_resource_scheduling()` -> `PPOScheduler.train()` -> `scheduler.training_stats` -> `resource_scheduling_reward.csv` / `resource_scheduling_train.log` / `real_ppo_*.png`。

**预防**: 验收调度训练时看输出文件和 `summary.json`，不要只看命令退出码。

### <!-- L07 --> T07: 高阶 QAM 准确率必须按类别验收

**发现**: 当前调制识别 overall accuracy 为 66.64%，但 64QAM 仅 16.8%，256QAM 为 46.0%。整体准确率会掩盖高阶 QAM 的主要问题。

**关键路径**: `SignalGenerator.generate_dataset()` 生成 5 类 IQ 样本，`RecognitionTrainer.evaluate()` 输出 `class_accuracy`，实验脚本写入 `modulation_recognition_results.csv`。

**预防**: T07 的验收指标必须包含 `class_64QAM` 和 `class_256QAM`，并建议新增 SNR 分层 CSV。

### <!-- L08 --> T09: 100 epoch 配置会被 trainer 早停截断

**发现**: `collect_channel_estimation(quick=False)` 设置 `epochs=100`，但 `ChannelEstimationTrainer.train_pytorch_model()` 内置 `patience=10`。当前真实日志实际训练到 37 epoch，best val loss 为 0.031107。

**关键路径**: `collect_channel_estimation()` -> `ChannelEstimationTrainer.train_pytorch_model()` -> early stopping -> `channel_estimation_loss.csv` / `channel_estimation_train.log`。

**预防**: 若 T09 要“完整 100 epoch 曲线”，需要显式关闭或提高早停 patience；若目标是“优化收敛”，当前结果已按早停恢复最佳权重。

### <!-- L09 --> Focal Loss 不适合多调制分类

**症状**: Focal Loss γ=2.0 替换 CrossEntropy, 64QAM 71.9%→18.4%, 整体 68.9%→69.2%。

**根因**: (1-p_t)^γ 无差别压低已确信样本梯度。64QAM 是困难样本, 被 Focal Loss 误杀。无法区分"低 SNR 难"和"高阶调制难"。

**解决**: 回退 CrossEntropy + 过采样。

### <!-- L10 --> 分组多出口架构 QAM 头失效

**症状**: PSK 头完美 (BPSK 99.7%, QPSK 97.9%), QAM 头全 0%, 整体 39.6%。

**根因**: 共享 CNN 特征提取器被 PSK 简单类梯度主导, QAM 头无有效梯度更新。

**解决**: 放弃分组架构, 回退单头 + 过采样 (76.91%)。

### <!-- L11 --> patience 变量引用遗漏

**症状**: trainer.py:263 引用局部 patience → NameError。

**解决**: 改为 self.patience。

### <!-- L12 --> 最终方案: FC=512 + 过采样

**方案**: FC 容量翻倍 (256→512) + 256QAM×2 过采样 + 64QAM×1.5 过采样。Overall 76.91%, 256QAM 82.8%, 64QAM 52.5%。

## Purpose

记录项目依赖和外部参考资源，确保依赖可追溯，资源可获取。

## Requirements

### Requirement: 依赖版本锁定

所有项目依赖 SHALL 记录版本信息，确保构建可重现。

#### Scenario: 添加新依赖

- **WHEN** 开发者引入新的外部依赖
- **THEN** 必须记录到本文件，包含：依赖名称、版本、官方链接、用途说明

#### Scenario: 更新依赖版本

- **WHEN** 开发者升级或降级依赖版本
- **THEN** 必须更新本文件中的版本记录，标注更新原因

### Requirement: 外部资源记录

项目使用的外部资源和文档 SHALL 记录，方便查阅。

#### Scenario: 参考外部文档

- **WHEN** 开发者参考了重要的外部文档或资源
- **THEN** 必须记录到本文件，包含：资源名称、链接、关键内容摘要

### Requirement: 项目分析文档索引

深度分析文档 SHALL 建立索引，方便查找。

#### Scenario: 生成分析文档

- **WHEN** openspec-explorer 生成了项目分析文档
- **THEN** 必须在 references/spec.md 中注册索引条目

---

## 核心依赖

| 依赖 | 版本 | 用途 |
|------|------|------|
| Python | 3.10+ | 运行环境 |
| PyTorch | >=1.12.0 | 深度学习框架 |
| torchvision | >=0.13.0 | 视觉相关（辅助） |
| NumPy | >=1.21.0 | 科学计算 |
| SciPy | >=1.7.0 | 信号处理 |
| scikit-learn | >=1.0.0 | 机器学习工具 |
| Matplotlib | >=3.5.0 | 数据可视化 |
| seaborn | >=0.11.0 | 高级可视化 |
| Flask | >=2.0.0 | Web 框架 |
| flask-socketio | >=5.0.0 | WebSocket 支持 |
| tqdm | >=4.62.0 | 进度条 |
| PyYAML | >=6.0 | YAML 配置解析 |
| Pillow | >=9.0.0 | 图像处理 |
| gym | >=0.21.0 | 强化学习环境 |
| stable-baselines3 | >=1.5.0 | 强化学习算法库 |

## 项目文档

| 文档 | 路径 | 说明 |
|------|------|------|
| 技术方案 | docs/SOLUTION.md | 项目概述、算法设计、系统架构 |
| 算法架构 | docs/ALGORITHMS.md | Mermaid 流程图、数据流设计 |
| 性能参数 | docs/PERFORMANCE.md | 测试结果、NMSE 对比、吞吐量 |

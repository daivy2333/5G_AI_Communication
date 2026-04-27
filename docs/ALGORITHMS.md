# 算法架构文档

本文档详细说明5G无线通信链路智能优化系统中各AI模块的算法架构、数据流和处理流程。

---

## 1. 系统整体架构

### 1.1 模块关系图

```mermaid
flowchart TB
    subgraph 发送端["发送端处理"]
        A1[信源数据]
        A2[信道编码]
        A3[调制映射]
        A4[OFDM调制]
    end
    
    subgraph 无线信道["无线信道"]
        B1[路径损耗]
        B2[阴影衰落]
        B3[多径衰落]
        B4[加性噪声]
    end
    
    subgraph 接收端["接收端AI处理"]
        C1[OFDM解调]
        C2[AI信道估计]
        C3[AI信号检测]
        C4[调制识别]
        C5[信道译码]
    end
    
    subgraph 控制层["智能控制层"]
        D1[资源调度]
        D2[功率分配]
    end
    
    A1 --> A2 --> A3 --> A4
    A4 --> B1 --> B2 --> B3 --> B4
    B4 --> C1 --> C2 --> C3 --> C4 --> C5
    C2 -.-> D1
    C3 -.-> D2
    D1 -.-> A4
    D2 -.-> A3
```

### 1.2 OFDM完整链路流程

```mermaid
graph LR
    A[比特流] --> B[信道编码]
    B --> C[调制映射]
    C --> D[串并转换]
    D --> E[子载波映射]
    E --> F[IDFT]
    F --> G[添加CP]
    G --> H[无线信道]
    H --> I[去除CP]
    I --> J[DFT]
    J --> K[子载波解映射]
    K --> L[AI信道估计]
    L --> M[AI信号检测]
    M --> N[并串转换]
    N --> O[信道译码]
    O --> P[比特流]
```

---

## 2. AI信道估计 (AI-ChannelNet)

### 2.1 算法架构

基于Transformer架构的端到端信道估计网络，核心特性：

| 参数 | 配置 | 说明 |
|------|------|------|
| Transformer层数 | 4层 | 深度学习网络层数 |
| 注意力头数 | 8头 | 多头自注意力机制 |
| 嵌入维度 | 128 | 特征空间维度 |
| 激活函数 | GELU | 平滑激活函数 |
| Dropout率 | 0.1 | 防止过拟合 |

### 2.2 数据处理流程

```mermaid
graph LR
    subgraph 输入层["输入处理"]
        A1[导频信号<br/>复数形式]
        A2[接收信号<br/>复数形式]
        A3[特征提取<br/>实部/虚部/幅度/相位]
    end
    
    subgraph 编码层["编码器"]
        B1[线性嵌入<br/>维度128]
        B2[位置编码<br/>Sinusoidal]
        B3[多头注意力<br/>8头并行]
    end
    
    subgraph 处理层["Transformer层"]
        C1[自注意力<br/>捕捉时频相关性]
        C2[前馈网络<br/>维度512]
        C3[残差连接<br/>+层归一化]
    end
    
    subgraph 输出层["解码器"]
        D1[线性输出<br/>实部/虚部]
        D2[信道估计<br/>矩阵]
    end
    
    A1 --> A3
    A2 --> A3
    A3 --> B1 --> B2 --> B3 --> C1 --> C2 --> C3 --> D1 --> D2
```

### 2.3 Transformer编码器详细架构

```mermaid
flowchart TB
    subgraph TransformerBlock["Transformer编码器层 (重复4次)"]
        subgraph AttentionBlock["多头自注意力模块"]
            A1[输入 X]
            A2[Query投影]
            A3[Key投影]
            A4[Value投影]
            A5[分数计算<br/>Q·K^T / sqrt d]
            A6[Softmax]
            A7[注意力加权<br/>Attn·V]
            A8[多头合并]
        end
        
        subgraph FFNBlock["前馈网络"]
            F1[线性层1<br/>d→4d]
            F2[GELU激活]
            F3[线性层2<br/>4d→d]
        end
        
        A1 --> A2 --> A5
        A1 --> A3 --> A5
        A1 --> A4 --> A7
        A5 --> A6 --> A7 --> A8
        
        A8 --> Add1[残差连接+]
        Add1 --> LN1[层归一化]
        
        LN1 --> F1 --> F2 --> F3
        F3 --> Add2[残差连接+]
        Add2 --> LN2[层归一化]
    end
```

### 2.4 与传统算法对比

```mermaid
graph LR
    subgraph 传统方法["传统方法"]
        T1[LS估计<br/>简单快速<br/>噪声敏感]
        T2[MMSE估计<br/>统计依赖<br/>计算复杂]
    end
    
    subgraph AI方法["AI方法"]
        AI1[AI-ChannelNet<br/>数据驱动<br/>复杂信道优势]
    end
    
    subgraph 性能对比["性能对比"]
        P1[LS: MSE -15dB]
        P2[MMSE: MSE -18dB]
        P3[AI: MSE -22dB<br/>+7dB提升]
    end
    
    T1 --> P1
    T2 --> P2
    AI1 --> P3
```

---

## 3. AI信号检测与调制识别 (SignalNet)

### 3.1 CNN-LSTM混合架构

采用三级卷积网络进行空间特征提取，双向LSTM进行时序建模：

| 参数 | 配置 | 说明 |
|------|------|------|
| CNN滤波器 | 64→128→256 | 逐层抽象特征 |
| 卷积核大小 | 3×1 | 局部特征提取 |
| LSTM层数 | 2层 | 双向时序建模 |
| LSTM单元 | 128 | 隐藏状态维度 |
| Dropout率 | 0.3 | 防止过拟合 |

### 3.2 信号检测数据流

```mermaid
graph LR
    A[接收信号<br/>IQ两路] --> B[三级CNN<br/>特征提取]
    B --> C[Bi-LSTM<br/>时序建模]
    C --> D[全局池化]
    D --> E[全连接层]
    E --> F[检测概率<br/>sigmoid输出]
    
    subgraph CNN层["CNN特征提取"]
        B1[Conv64<br/>kernel 3×1]
        B2[Conv128<br/>kernel 3×1]
        B3[Conv256<br/>kernel 3×1]
    end
    
    subgraph LSTM层["时序建模"]
        C1[前向LSTM]
        C2[后向LSTM]
    end
    
    B1 --> B2 --> B3 --> C
    C --> C1 --> C2
```

### 3.3 调制识别网络架构

```mermaid
flowchart TB
    subgraph 输入["输入数据"]
        A[IQ信号<br/>128采样点]
    end
    
    subgraph CNN层["卷积特征提取"]
        B1[Conv1: 64滤波器<br/>kernel 7×1]
        B2[Conv2: 128滤波器<br/>kernel 5×1]
        B3[Conv3: 256滤波器<br/>kernel 3×1]
        B4[最大池化]
    end
    
    subgraph 分类["分类输出"]
        C1[全局平均池化]
        C2[全连接层]
        C3[Softmax<br/>5类输出]
    end
    
    subgraph 调制类型["调制类型"]
        D1[BPSK]
        D2[QPSK]
        D3[16QAM]
        D4[64QAM]
        D5[256QAM]
    end
    
    A --> B1 --> B2 --> B3 --> B4 --> C1 --> C2 --> C3
    C3 --> D1
    C3 --> D2
    C3 --> D3
    C3 --> D4
    C3 --> D5
```

### 3.4 处理流程序列图

```mermaid
sequenceDiagram
    participant 发送端
    participant 信道
    participant 接收机
    participant CNN模块
    participant LSTM模块
    participant 分类器
    
    发送端->>信道: 发送调制信号
    信道->>接收机: 接收到含噪声信号
    接收机->>CNN模块: IQ两路信号
    CNN模块->>CNN模块: Conv1层提取
    CNN模块->>CNN模块: Conv2层抽象
    CNN模块->>CNN模块: Conv3层高级特征
    CNN模块->>LSTM模块: 特征序列
    LSTM模块->>LSTM模块: 前向时序建模
    LSTM模块->>LSTM模块: 后向时序建模
    LSTM模块->>分类器: 时序特征
    分类器->>分类器: Softmax分类
    分类器->>接收机: 调制类型识别结果
```

---

## 4. 智能资源调度 (RL-Scheduler)

### 4.1 强化学习框架

基于PPO (Proximal Policy Optimization) 算法的资源调度：

| 参数 | 配置 | 说明 |
|------|------|------|
| 折扣因子 γ | 0.99 | 长期收益权重 |
| GAE参数 λ | 0.95 | 优势函数估计 |
| 裁剪范围 ε | 0.2 | 策略更新限制 |
| 熵系数 | 0.01 | 探索鼓励 |
| 学习率 | 3e-4 | Adam优化器 |

### 4.2 状态-动作-奖励设计

```mermaid
flowchart LR
    subgraph 状态空间["状态空间 S"]
        S1[用户CQI]
        S2[队列长度]
        S3[业务优先级]
        S4[资源使用率]
    end
    
    subgraph 动作空间["动作空间 A"]
        A1[RB分配]
        A2[功率等级]
    end
    
    subgraph 奖励函数["奖励函数 R"]
        R1[吞吐量<br/>权重1.0]
        R2[公平性Jain<br/>权重0.3]
        R3[能效<br/>权重0.2]
    end
    
    S1 & S2 & S3 & S4 --> Agent[智能体]
    Agent --> A1 & A2
    A1 & A2 --> 环境[环境执行]
    环境 --> R1 & R2 & R3 --> Agent
```

### 4.3 PPO算法流程

```mermaid
sequenceDiagram
    participant 环境
    participant 智能体
    participant 策略网络
    participant 价值网络
    
    loop 训练循环
        环境->>智能体: 状态 s
        智能体->>策略网络: 状态输入
        策略网络->>智能体: 动作概率分布
        智能体->>环境: 执行动作 a
        环境->>智能体: 新状态 s', 奖励 r
        智能体->>智能体: 存储轨迹 (s,a,r,s')
    end
    
    loop PPO更新
        智能体->>价值网络: 计算价值估计
        价值网络->>智能体: 优势函数 A
        智能体->>策略网络: 计算概率比率
        策略网络->>策略网络: PPO裁剪更新
        策略网络->>智能体: 熵正则化
        智能体->>价值网络: MSE损失更新
    end
```

### 4.4 资源调度详细流程

```mermaid
flowchart TB
    subgraph 观察阶段["状态观察"]
        O1[收集各用户CQI]
        O2[统计队列状态]
        O3[评估信道质量]
        O4[形成状态向量]
    end
    
    subgraph 决策阶段["策略决策"]
        D1[策略网络推理]
        D2[动作概率分布]
        D3[资源块分配]
        D4[功率等级分配]
    end
    
    subgraph 执行阶段["环境交互"]
        E1[分配资源块]
        E2[设定发射功率]
        E3[传输数据]
        E4[统计性能]
    end
    
    subgraph 评估阶段["奖励计算"]
        R1[吞吐量计算]
        R2[Jain公平指数]
        R3[能效计算]
        R4[加权奖励]
    end
    
    subgraph 学习阶段["策略更新"]
        L1[收集经验轨迹]
        L2[计算优势函数]
        L3[PPO裁剪更新]
        L4[价值网络更新]
    end
    
    O1 --> O2 --> O3 --> O4 --> D1 --> D2 --> D3 --> D4
    D4 --> E1 --> E2 --> E3 --> E4 --> R1 --> R2 --> R3 --> R4
    R4 --> L1 --> L2 --> L3 --> L4 --> D1
```

---

## 5. 性能评估体系

### 5.1 指标体系架构

```mermaid
graph TB
    subgraph 信道估计指标["信道估计"]
        CE1[NMSE]
        CE2[MSE]
        CE3[信道相关性]
    end
    
    subgraph 调制识别指标["调制识别"]
        MR1[总体准确率]
        MR2[混淆矩阵]
        MR3[各类别准确率]
    end
    
    subgraph 信号检测指标["信号检测"]
        SD1[BER误码率]
        SD2[SER误符号率]
        SD3[EVM误差矢量幅度]
    end
    
    subgraph 资源调度指标["资源调度"]
        RS1[Jain公平指数]
        RS2[频谱效率]
        RS3[能效]
        RS4[吞吐量]
    end
    
    CE1 & CE2 & CE3 --> 性能评估[综合评估]
    MR1 & MR2 & MR3 --> 性能评估
    SD1 & SD2 & SD3 --> 性能评估
    RS1 & RS2 & RS3 & RS4 --> 性能评估
```

### 5.2 技术指标要求

| 模块 | 指标 | 要求 | 验证条件 |
|------|------|------|----------|
| AI信道估计 | NMSE | < -20dB | SNR ≥ 10dB |
| AI信道估计 | 性能提升 | ≥ 7dB | 相比LS估计 |
| 调制识别 | 总体准确率 | ≥ 95% | SNR ≥ 10dB |
| 调制识别 | 各类别准确率 | ≥ 90% | SNR ≥ 10dB |
| 资源调度 | 吞吐量提升 | ≥ 20% | 相比轮询调度 |
| 资源调度 | Jain公平指数 | ≥ 0.85 | 长期统计 |

---

## 6. 数据生成与增强

### 6.1 训练数据生成流程

```mermaid
flowchart LR
    subgraph 参数采样["参数采样"]
        A1[信道模型<br/>UMa/UMi/InH]
        A2[SNR范围<br/>-10dB~30dB]
        A3[多径数目<br/>6~12]
        A4[多普勒频率<br/>随机]
    end
    
    subgraph 信号生成["信号生成"]
        B1[随机比特流]
        B2[调制映射]
        B3[OFDM处理]
        B4[导频插入]
    end
    
    subgraph 信道施加["信道效应"]
        C1[多径衰落]
        C2[相位噪声]
        C3[频率偏移]
        C4[加性噪声]
    end
    
    subgraph 标签生成["标签生成"]
        D1[真实信道矩阵]
        D2[调制类型标签]
        D3[符号标签]
    end
    
    A1 & A2 & A3 & A4 --> B1 --> B2 --> B3 --> B4 --> C1 --> C2 --> C3 --> C4 --> D1 & D2 & D3
```

### 6.2 数据增强策略

```mermaid
graph TB
    A[原始信号] --> B[增强策略]
    
    subgraph 增强策略["数据增强"]
        E1[相位旋转<br/>模拟CFO]
        E2[幅度缩放<br/>模拟路径损耗]
        E3[时间偏移<br/>模拟定时误差]
        E4[噪声注入<br/>不同SNR]
    end
    
    B --> E1 & E2 & E3 & E4 --> C[增强后信号]
```

---

## 附录：5G系统参数

### 系统配置参数

| 参数 | 数值 | 符合标准 |
|------|------|----------|
| 载波频率 | 3.5 GHz | C波段5G |
| 系统带宽 | 100 MHz | 3GPP规范 |
| FFT大小 | 2048 | OFDM标准 |
| 循环前缀 | 512 | 抗多径 |
| 数据子载波 | 1200 | 有效带宽 |

---

**文档版本**: V1.0  
**更新日期**: 2026年4月
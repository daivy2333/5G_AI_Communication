"""
5G无线通信链路+AI创新设计项目
配置文件 - 集中管理所有系统参数
"""

# ============================================================================
# 系统基础配置
# ============================================================================
class SystemConfig:
    """系统基础参数"""
    # 仿真参数
    NUM_SAMPLES = 10000          # 训练样本数量
    SNR_RANGE = (-10, 30)       # 信噪比范围(dB)
    CARRIER_FREQ = 3.5e9         # 载波频率(Hz) - C波段5G
    BANDWIDTH = 100e6            # 系统带宽(Hz) - 100MHz
    FFT_SIZE = 2048              # FFT大小
    CYCLIC_PREFIX = 512          # 循环前缀长度
    NUM_SUBCARRIERS = 1200       # 数据子载波数

    # 仿真场景
    CHANNEL_MODEL = '5G_UMa'    # 5G城市宏蜂窝信道
    DOPPLER_FREQ = 100           # 多普勒频率(Hz)
    DELAY_SPREAD = 300e-9        # 时延扩展(s)

    # 随机种子
    RANDOM_SEED = 42


# ============================================================================
# AI信道估计配置
# ============================================================================
class ChannelEstimationConfig:
    """信道估计模块配置"""
    # 网络架构参数
    EMBEDDING_DIM = 128          # 嵌入维度
    NUM_HEADS = 8                # 注意力头数
    NUM_LAYERS = 4               # Transformer层数
    DROPOUT_RATE = 0.1           # Dropout比率

    # 训练参数
    BATCH_SIZE = 64
    EPOCHS = 100
    LEARNING_RATE = 1e-4
    OPTIMIZER = 'adam'

    # 信道模型
    CHANNEL_TAPS = 16            # 信道抽头数
    PILOT_SPACING = 8            # 导频间隔

    # 性能基准对比算法
    BASELINES = ['LS', 'MMSE', 'LMMSE']


# ============================================================================
# 信号检测与调制识别配置
# ============================================================================
class SignalDetectionConfig:
    """信号检测与调制识别配置"""
    # 支持的调制方式
    MODULATION_TYPES = ['BPSK', 'QPSK', '16QAM', '64QAM', '256QAM']

    # 信号参数
    SYMBOL_RATE = 15.36e6        # 符号速率
    SAMPLES_PER_SYMBOL = 4       # 每符号采样数

    # CNN-RNN网络参数
    CNN_FILTERS = [64, 128, 256]
    CNN_KERNEL_SIZE = 3
    LSTM_UNITS = 128
    DROPOUT_RATE = 0.3

    # 训练参数
    BATCH_SIZE = 128
    EPOCHS = 50
    LEARNING_RATE = 1e-3

    # 数据增强
    AUGMENTATION = True
    NOISE_LEVELS = [-10, 0, 10, 20, 30]


# ============================================================================
# 资源调度配置
# ============================================================================
class ResourceSchedulingConfig:
    """智能资源调度配置"""
    # 仿真环境参数
    NUM_USERS = 20               # 用户数量
    NUM_RESOURCE_BLOCKS = 100    # 资源块数量
    MAX_POWER = 33               # 最大发射功率(dBm)

    # 强化学习参数
    ALGORITHM = 'PPO'            # PPO算法
    GAMMA = 0.99                 # 折扣因子
    LAMDA = 0.95                 # GAE参数
    CLIP_RANGE = 0.2             # PPO裁剪范围
    ENTROPY_COEF = 0.01          # 熵系数

    # 训练参数
    TIMESTEPS = 10000           # 总训练步数
    BATCH_SIZE = 256
    LEARNING_RATE = 3e-4

    # 奖励函数权重
    REWARD_WEIGHTS = {
        'throughput': 1.0,        # 吞吐量权重
        'fairness': 0.3,          # 公平性权重
        'energy': 0.2             # 能效权重
    }


# ============================================================================
# 可视化配置
# ============================================================================
class VisualizationConfig:
    """可视化配置"""
    # Web服务配置
    HOST = '0.0.0.0'
    PORT = 5000
    DEBUG = True

    # 图表样式
    STYLE = 'seaborn'
    FIGURE_SIZE = (12, 8)
    DPI = 100

    # 实时更新间隔(秒)
    UPDATE_INTERVAL = 1.0

    # 颜色方案
    COLORS = {
        'primary': '#2196F3',     # 主色-蓝
        'secondary': '#4CAF50',  # 副色-绿
        'accent': '#FF9800',     # 强调色-橙
        'error': '#F44336',       # 错误色-红
        'warning': '#FFC107'     # 警告色-黄
    }


# ============================================================================
# 导出统一配置
# ============================================================================
class CompleteConfig:
    """完整配置汇总"""
    system = SystemConfig()
    channel_estimation = ChannelEstimationConfig()
    signal_detection = SignalDetectionConfig()
    resource_scheduling = ResourceSchedulingConfig()
    visualization = VisualizationConfig()


# 全局配置实例
config = CompleteConfig()


def get_config(module_name=None):
    """
    获取配置项

    Args:
        module_name: 模块名称，可选值:
                    'system', 'channel_estimation',
                    'signal_detection', 'resource_scheduling',
                    'visualization', None(返回全部)

    Returns:
        配置对象或配置字典
    """
    if module_name is None:
        return config.__dict__
    return getattr(config, module_name, None)


# 便捷函数
def get_system_config():
    """获取系统配置"""
    return SystemConfig()


def get_channel_config():
    """获取信道估计配置"""
    return ChannelEstimationConfig()


def get_signal_config():
    """获取信号检测配置"""
    return SignalDetectionConfig()


def get_scheduling_config():
    """获取资源调度配置"""
    return ResourceSchedulingConfig()


def get_viz_config():
    """获取可视化配置"""
    return VisualizationConfig()

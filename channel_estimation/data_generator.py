"""
AI信道估计模块 - 信道数据生成器
生成5G标准信道数据用于深度学习训练
"""

import numpy as np
from typing import Tuple, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_channel_config


class ChannelDataGenerator:
    """
    5G信道数据生成器

    支持生成多种信道模型的数据：
    - 5G城市宏蜂窝(UMa)
    - 5G城市微蜂窝(UMi)
    - 室内办公环境(InH)
    - 自由空间传播
    """

    def __init__(self, config: Optional[dict] = None):
        """
        初始化信道数据生成器

        Args:
            config: 配置字典，如果为None则使用默认配置
        """
        self.config = config or {}
        self.channel_config = get_channel_config()

        # 系统参数
        self.fft_size = self.config.get("fft_size", 2048)
        self.num_subcarriers = self.config.get("num_subcarriers", 1200)
        self.pilot_spacing = self.config.get("pilot_spacing", 8)
        self.channel_taps = self.config.get("channel_taps", 16)

        # 信道模型参数
        self.channel_models = {
            "5G_UMa": self._generate_uma_params,
            "5G_UMi": self._generate_umi_params,
            "5G_InH": self._generate_inh_params,
            "AWGN": self._generate_awgn_channel,
        }

    def _generate_uma_params(self, num_samples: int) -> dict:
        """
        生成5G城市宏蜂窝(UMa)信道参数

        Args:
            num_samples: 样本数量

        Returns:
            信道参数字典
        """
        # 路径损耗参数(3.5GHz)
        d = np.random.uniform(10, 500, num_samples)  # 距离(m)
        h_bs = 25  # 基站高度(m)
        h_ut = 1.5  # 用户终端高度(m)

        # 阴影衰落标准差(dB)
        sigma_sf = 4

        # 时延扩展(ms)
        delay_spread = np.random.uniform(100, 300, num_samples) * 1e-9

        # 多径数量
        num_paths = np.random.randint(6, 12, num_samples)

        return {
            "distance": d,
            "path_loss_params": {
                "f_c": 3.5e9,
                "h_bs": h_bs,
                "h_ut": h_ut,
                "sigma_sf": sigma_sf,
            },
            "delay_spread": delay_spread,
            "num_paths": num_paths,
            "model_name": "5G_UMa",
        }

    def _generate_umi_params(self, num_samples: int) -> dict:
        """生成5G城市微蜂窝(UMi)信道参数"""
        d = np.random.uniform(10, 200, num_samples)
        h_bs = 10
        h_ut = 1.5
        sigma_sf = 4
        delay_spread = np.random.uniform(50, 200, num_samples) * 1e-9
        num_paths = np.random.randint(5, 10, num_samples)

        return {
            "distance": d,
            "path_loss_params": {
                "f_c": 3.5e9,
                "h_bs": h_bs,
                "h_ut": h_ut,
                "sigma_sf": sigma_sf,
            },
            "delay_spread": delay_spread,
            "num_paths": num_paths,
            "model_name": "5G_UMi",
        }

    def _generate_inh_params(self, num_samples: int) -> dict:
        """生成室内办公环境(InH)信道参数"""
        d = np.random.uniform(3, 100, num_samples)
        h_bs = 3
        h_ut = 1.5
        sigma_sf = 3
        delay_spread = np.random.uniform(20, 100, num_samples) * 1e-9
        num_paths = np.random.randint(3, 8, num_samples)

        return {
            "distance": d,
            "path_loss_params": {
                "f_c": 3.5e9,
                "h_bs": h_bs,
                "h_ut": h_ut,
                "sigma_sf": sigma_sf,
            },
            "delay_spread": delay_spread,
            "num_paths": num_paths,
            "model_name": "5G_InH",
        }

    def _generate_awgn_channel(self, num_samples: int) -> dict:
        """生成加性高斯白噪声信道参数"""
        return {
            "distance": np.full(num_samples, 100.0),
            "path_loss_params": {"sigma_sf": 0},
            "delay_spread": np.zeros(num_samples),
            "num_paths": np.ones(num_samples, dtype=int),
            "model_name": "AWGN",
        }

    def generate_channel_impulse_response(
        self, num_samples: int, model: str = "5G_UMa", seed: Optional[int] = None
    ) -> np.ndarray:
        """
        生成信道脉冲响应(CIR)

        Args:
            num_samples: 样本数量
            model: 信道模型名称
            seed: 随机种子

        Returns:
            信道脉冲响应，形状为(num_samples, channel_taps)
        """
        if seed is not None:
            np.random.seed(seed)

        # 获取信道参数
        if model in self.channel_models:
            params = self.channel_models[model](num_samples)
        else:
            params = self._generate_uma_params(num_samples)

        cir = np.zeros((num_samples, self.channel_taps), dtype=complex)
        num_paths_all = params["num_paths"]
        total_paths = int(np.sum(num_paths_all))
        
        sample_indices = np.repeat(np.arange(num_samples), num_paths_all)
        
        delay_spreads_ns = params["delay_spread"] * 1e9
        delays_flat = np.zeros(total_paths)
        cumsum = 0
        for i in range(num_samples):
            n = num_paths_all[i]
            delays_flat[cumsum:cumsum + n] = np.sort(
                np.random.uniform(0, delay_spreads_ns[i], n)
            )
            cumsum += n
        delays_flat *= 1e-9
        
        amplitudes_flat = np.random.rayleigh(scale=1.0 / np.sqrt(2), size=total_paths)
        phases_flat = np.random.uniform(0, 2 * np.pi, total_paths)
        
        tap_indices = np.minimum(
            (delays_flat * self.fft_size / self.fft_size).astype(int),
            self.channel_taps - 1
        )
        cir_values = amplitudes_flat * np.exp(1j * phases_flat)
        
        for idx in range(total_paths):
            cir[sample_indices[idx], tap_indices[idx]] += cir_values[idx]

        return cir

    def generate_pilot_signal(self, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成导频信号

        Args:
            num_samples: 样本数量

        Returns:
            (发送导频, 导频位置索引)
        """
        num_pilots = self.num_subcarriers // self.pilot_spacing
        pilot_positions = np.arange(0, self.num_subcarriers, self.pilot_spacing)

        # 生成QPSK导频序列
        pilot_data = np.random.choice(
            [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], size=(num_samples, num_pilots)
        )
        pilot_data = pilot_data / np.sqrt(2)  # 归一化功率

        return pilot_data, pilot_positions

    def generate_received_signal(
        self, pilot_tx: np.ndarray, channel: np.ndarray, snr_db: float
    ) -> np.ndarray:
        """
        生成接收信号(含噪声)

        Args:
            pilot_tx: 发送导频信号，形状为(num_samples, num_pilots)
            channel: 信道脉冲响应，形状为(num_samples, channel_taps)
            snr_db: 信噪比(dB)

        Returns:
            接收信号，形状与pilot_tx相同
        """
        # 信道频域响应(补零到导频长度)
        h_freq = np.fft.fft(channel, axis=1)
        target_len = pilot_tx.shape[1]
        h_freq_padded = np.zeros((h_freq.shape[0], target_len), dtype=complex)
        h_freq_padded[:, :min(h_freq.shape[1], target_len)] = h_freq[:, :target_len]
        h_freq = h_freq_padded

        # 通过信道(逐样本相乘)
        received = pilot_tx * h_freq

        # 添加噪声
        signal_power = np.mean(np.abs(received) ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(*received.shape) + 1j * np.random.randn(*received.shape)
        )
        received_noisy = received + noise

        return received_noisy

    def generate_training_dataset(
        self,
        num_samples: int,
        model: str = "5G_UMa",
        snr_range: Tuple[float, float] = (-10, 30),
        seed: int = 42,
    ) -> Tuple[dict, np.ndarray]:
        """
        生成完整的训练数据集

        Args:
            num_samples: 样本数量
            model: 信道模型
            snr_range: 信噪比范围(dB)
            seed: 随机种子

        Returns:
            (输入数据字典, 真实信道估计标签)
        """
        np.random.seed(seed)

        pilot_tx, pilot_positions = self.generate_pilot_signal(num_samples)
        channel_true = self.generate_channel_impulse_response(num_samples, model)
        snr_db = np.random.uniform(snr_range[0], snr_range[1], num_samples)

        h_freq = np.fft.fft(channel_true, axis=1)
        target_len = pilot_tx.shape[1]
        h_freq_padded = np.zeros((h_freq.shape[0], target_len), dtype=complex)
        h_freq_padded[:, :min(h_freq.shape[1], target_len)] = h_freq[:, :target_len]
        
        received_clean = pilot_tx * h_freq_padded
        
        signal_power = np.mean(np.abs(received_clean) ** 2, axis=1, keepdims=True)
        noise_power = signal_power / (10 ** (snr_db[:, np.newaxis] / 10))
        noise_std = np.sqrt(noise_power / 2)
        noise = noise_std * (
            np.random.randn(*received_clean.shape) + 1j * np.random.randn(*received_clean.shape)
        )
        received = received_clean + noise

        # Zero-pad time-domain channel to pilot length before FFT for proper frequency resolution
        channel_time_padded = np.zeros((num_samples, pilot_tx.shape[1]), dtype=complex)
        channel_time_padded[:, :self.channel_taps] = channel_true
        channel_freq_true = np.fft.fft(channel_time_padded, axis=1)

        dataset = {
            "received_signal": received,
            "transmitted_pilot": pilot_tx,
            "pilot_positions": pilot_positions,
            "channel_time": channel_true,
            "snr_db": snr_db,
            "model": model,
        }

        return dataset, channel_freq_true

    def compute_baseline_estimates(
        self, received: np.ndarray, pilot_tx: np.ndarray
    ) -> dict:
        """
        计算传统基准算法估计

        Args:
            received: 接收信号
            pilot_tx: 发送导频

        Returns:
            包含各种算法估计结果的字典
        """
        # LS估计
        h_ls = received / (pilot_tx + 1e-10)

        # 简化MMSE估计(假设已知信道统计)
        snr_approx = np.mean(np.abs(received) ** 2 / np.abs(pilot_tx) ** 2)
        h_mmse = h_ls * (snr_approx / (snr_approx + 1))

        # LMMSE估计
        correlation_matrix = np.eye(h_ls.shape[1]) * 0.5
        noise_variance = 1 / (snr_approx + 1e-10)
        h_lmmse = np.linalg.solve(
            correlation_matrix + noise_variance * np.eye(h_ls.shape[1]),
            correlation_matrix @ h_ls.T,
        ).T

        return {"LS": h_ls, "MMSE": h_mmse, "LMMSE": h_lmmse}


def generate_synthetic_channel(
    num_taps: int = 16, num_samples: int = 1000, snr_db: float = 20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    便捷函数：生成合成信道数据

    Args:
        num_taps: 信道抽头数
        num_samples: 样本数
        snr_db: 信噪比(dB)

    Returns:
        (接收信号, 发送信号, 真实信道)
    """
    generator = ChannelDataGenerator({"channel_taps": num_taps})
    pilot_tx, _ = generator.generate_pilot_signal(num_samples)
    channel_true = generator.generate_channel_impulse_response(num_samples)
    received = generator.generate_received_signal(pilot_tx, channel_true, snr_db)

    return received, pilot_tx, channel_true


if __name__ == "__main__":
    # 测试数据生成器
    print("=" * 60)
    print("AI信道估计模块 - 数据生成器测试")
    print("=" * 60)

    # 创建数据生成器
    generator = ChannelDataGenerator()

    # 生成测试数据
    print("\n1. 生成信道脉冲响应测试:")
    cir = generator.generate_channel_impulse_response(100, model="5G_UMa")
    print(f"   信道脉冲响应形状: {cir.shape}")
    print(f"   平均信道增益: {np.mean(np.abs(cir)):.4f}")

    # 生成导频信号
    print("\n2. 生成导频信号测试:")
    pilot_tx, positions = generator.generate_pilot_signal(100)
    print(f"   导频信号形状: {pilot_tx.shape}")
    print(f"   导频位置: 前10个 = {positions[:10]}")

    # 生成完整数据集
    print("\n3. 生成训练数据集测试:")
    dataset, labels = generator.generate_training_dataset(
        num_samples=1000, model="5G_UMa", snr_range=(-10, 30)
    )
    print(f"   数据集样本数: {len(dataset['received_signal'])}")
    print(
        f"   SNR范围: {dataset['snr_db'].min():.1f} ~ {dataset['snr_db'].max():.1f} dB"
    )

    # 计算基准估计
    print("\n4. 基准算法测试:")
    baselines = generator.compute_baseline_estimates(
        dataset["received_signal"][:10], dataset["transmitted_pilot"][:10]
    )
    for name, estimate in baselines.items():
        print(f"   {name}估计形状: {estimate.shape}")

    print("\n" + "=" * 60)
    print("数据生成器测试完成！")
    print("=" * 60)

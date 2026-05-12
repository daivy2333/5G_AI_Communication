"""
AI信号检测与调制识别模块 - 调制器与信号生成器
实现5G标准调制方式和仿真信号生成
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import get_signal_config


def generate_64qam_constellation() -> np.ndarray:
    """生成64QAM星座图"""
    i_grid, q_grid = np.meshgrid(np.arange(-7, 8, 2), np.arange(-7, 8, 2), indexing='ij')
    constellation = (i_grid + 1j * q_grid).ravel() / np.sqrt(42)
    return constellation


def generate_256qam_constellation() -> np.ndarray:
    """生成256QAM星座图"""
    i_grid, q_grid = np.meshgrid(np.arange(-15, 16, 2), np.arange(-15, 16, 2), indexing='ij')
    constellation = (i_grid + 1j * q_grid).ravel() / np.sqrt(170)
    return constellation


class Modulator:
    """
    数字调制器

    支持多种调制方式：
    - BPSK: 二进制相移键控
    - QPSK: 四进制相移键控
    - 16QAM: 16进制正交幅度调制
    - 64QAM: 64进制正交幅度调制
    - 256QAM: 256进制正交幅度调制
    """

    CONSTELLATIONS = None

    @classmethod
    def _init_constellations(cls):
        if cls.CONSTELLATIONS is None:
            cls.CONSTELLATIONS = {
                "BPSK": np.array([1, -1]),
                "QPSK": np.array([1 + 1j, -1 + 1j, 1 - 1j, -1 - 1j]) / np.sqrt(2),
                "16QAM": np.array(
                    [
                        3 + 3j,
                        1 + 3j,
                        -1 + 3j,
                        -3 + 3j,
                        3 + 1j,
                        1 + 1j,
                        -1 + 1j,
                        -3 + 1j,
                        3 - 1j,
                        1 - 1j,
                        -1 - 1j,
                        -3 - 1j,
                        3 - 3j,
                        1 - 3j,
                        -1 - 3j,
                        -3 - 3j,
                    ]
                )
                / np.sqrt(10),
                "64QAM": generate_64qam_constellation(),
                "256QAM": generate_256qam_constellation(),
            }

    def __init__(self, modulation_type: str = "QPSK"):
        """
        初始化调制器

        Args:
            modulation_type: 调制方式
        """
        self._init_constellations()
        self.modulation_type = modulation_type
        self.constellation = self._get_constellation(modulation_type)
        self.bits_per_symbol = self._get_bits_per_symbol(modulation_type)

    def _get_constellation(self, mod_type: str) -> np.ndarray:
        """获取星座图"""
        if mod_type in self.CONSTELLATIONS:
            return self.CONSTELLATIONS[mod_type]
        else:
            raise ValueError(f"不支持的调制方式: {mod_type}")

    def _get_bits_per_symbol(self, mod_type: str) -> int:
        """获取每符号比特数"""
        mapping = {"BPSK": 1, "QPSK": 2, "16QAM": 4, "64QAM": 6, "256QAM": 8}
        return mapping.get(mod_type, 2)

    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """
        调制比特流

        Args:
            bits: 输入比特流

        Returns:
            复数调制符号序列
        """
        num_symbols = len(bits) // self.bits_per_symbol
        bits = bits[: num_symbols * self.bits_per_symbol]

        bit_matrix = bits.reshape(-1, self.bits_per_symbol)
        symbol_indices = self._bits_to_indices(bit_matrix)
        symbols = self.constellation[symbol_indices]

        return symbols

    def _bits_to_indices(self, bit_matrix: np.ndarray) -> np.ndarray:
        """将比特矩阵转换为符号索引"""
        powers = 2 ** np.arange(bit_matrix.shape[1] - 1, -1, -1)
        indices = bit_matrix.astype(int) @ powers
        return indices

    def demodulate(self, symbols: np.ndarray, method: str = "hard") -> np.ndarray:
        """
        解调符号

        Args:
            symbols: 接收符号
            method: 解调方式 ('hard' 硬判决, 'soft' 软判决)

        Returns:
            解调比特流
        """
        if method == "hard":
            indices = self._find_nearest_constellation(symbols)
            bits = self._indices_to_bits(indices)
        else:
            bits = self._compute_llr(symbols)

        return bits

    def _find_nearest_constellation(self, symbols: np.ndarray) -> np.ndarray:
        """找到最近的星座点"""
        distances = np.abs(symbols[:, np.newaxis] - self.constellation[np.newaxis, :])
        indices = np.argmin(distances, axis=1)
        return indices

    def _indices_to_bits(self, indices: np.ndarray) -> np.ndarray:
        """将符号索引转换回比特"""
        bit_matrix = ((indices[:, np.newaxis] >> np.arange(self.bits_per_symbol - 1, -1, -1)) & 1)
        return bit_matrix.flatten()

    def _compute_llr(self, symbols: np.ndarray) -> np.ndarray:
        """计算对数似然比(LLR) - 简化版硬判决"""
        # 简化：直接使用硬判决
        indices = self._find_nearest_constellation(symbols)
        bits = self._indices_to_bits(indices)
        return bits

    def add_phase_noise(
        self, symbols: np.ndarray, phase_noise_std: float = 0.01
    ) -> np.ndarray:
        """
        添加相位噪声

        Args:
            symbols: 输入符号
            phase_noise_std: 相位噪声标准差(弧度)

        Returns:
            添加相位噪声后的符号
        """
        phase_noise = np.random.normal(0, phase_noise_std, len(symbols))
        return symbols * np.exp(1j * phase_noise)

    def add_frequency_offset(
        self, symbols: np.ndarray, offset: float, sample_idx: np.ndarray
    ) -> np.ndarray:
        """
        添加频率偏移

        Args:
            symbols: 输入符号
            offset: 频率偏移量(弧度/样本)
            sample_idx: 样本索引

        Returns:
            添加频率偏移后的符号
        """
        return symbols * np.exp(1j * offset * sample_idx)


class SignalGenerator:
    """
    多模态信号生成器

    生成用于调制识别训练的合成信号数据集
    """

    def __init__(self, config: Optional[dict] = None):
        """
        初始化信号生成器

        Args:
            config: 配置字典
        """
        self.config = config or {}
        signal_config = get_signal_config()

        self.modulation_types = self.config.get(
            "modulation_types", signal_config.MODULATION_TYPES
        )
        self.samples_per_symbol = self.config.get(
            "samples_per_symbol", signal_config.SAMPLES_PER_SYMBOL
        )
        self.snr_range = self.config.get("snr_range", (-10, 30))

        self.modulators = {}
        for mod_type in self.modulation_types:
            self.modulators[mod_type] = Modulator(mod_type)

    def generate_iq_samples(
        self, mod_type: str, num_symbols: int, snr_db: float, seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成IQ采样数据

        Args:
            mod_type: 调制方式
            num_symbols: 符号数量
            snr_db: 信噪比(dB)
            seed: 随机种子

        Returns:
            (IQ样本, 标签)
        """
        if seed is not None:
            np.random.seed(seed)

        modulator = self.modulators[mod_type]
        num_bits = num_symbols * modulator.bits_per_symbol
        bits = np.random.randint(0, 2, num_bits)

        symbols = modulator.modulate(bits)
        iq_samples = self._pulse_shape(symbols)
        iq_samples = self._add_channel_effects(iq_samples)
        iq_samples = self._add_awgn(iq_samples, snr_db)

        label = self._create_label(mod_type)

        return iq_samples, label

    def _pulse_shape(self, symbols: np.ndarray) -> np.ndarray:
        """脉冲成形"""
        num_samples = len(symbols) * self.samples_per_symbol
        samples = np.zeros(num_samples, dtype=complex)
        samples[::self.samples_per_symbol] = symbols
        return samples

    def _add_channel_effects(self, samples: np.ndarray) -> np.ndarray:
        """添加信道效应"""
        channel = np.array([1.0, 0.3 * np.exp(1j * np.random.uniform(0, 2 * np.pi))])
        faded = np.convolve(samples, channel, mode="same")

        phase_noise = np.random.normal(0, 0.02, len(faded))
        with_noise = faded * np.exp(1j * phase_noise)

        dc_offset = 0.01 * np.random.randn()
        with_dc = with_noise + dc_offset

        return with_dc

    def _add_awgn(self, samples: np.ndarray, snr_db: float) -> np.ndarray:
        """添加高斯白噪声"""
        signal_power = np.mean(np.abs(samples) ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.sqrt(noise_power) * (np.random.randn(len(samples)) + 1j * np.random.randn(len(samples))) / np.sqrt(2)
        return samples + noise

    def _create_label(self, mod_type: str) -> np.ndarray:
        """创建标签"""
        label = np.zeros(len(self.modulation_types))
        idx = self.modulation_types.index(mod_type)
        label[idx] = 1
        return label

    def generate_dataset(
        self, num_samples_per_class: int, snr_distribution: str = "uniform"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成完整数据集

        Args:
            num_samples_per_class: 每类样本数量
            snr_distribution: SNR分布 ('uniform', 'gaussian')

        Returns:
            (IQ样本数组, 标签数组)
        """
        total_samples = len(self.modulation_types) * num_samples_per_class
        sample_length = 128 * self.samples_per_symbol

        iq_data = np.zeros((total_samples, sample_length, 2))
        labels = np.zeros((total_samples, len(self.modulation_types)))

        print(f"生成数据集: {total_samples} 样本, 每类 {num_samples_per_class} 样本")

        idx = 0
        for mod_type in self.modulation_types:
            print(f"  生成 {mod_type} 数据...")

            for i in range(num_samples_per_class):
                if snr_distribution == "uniform":
                    snr_db = np.random.uniform(self.snr_range[0], self.snr_range[1])
                else:
                    snr_db = np.random.normal(10, 10)
                    snr_db = np.clip(snr_db, self.snr_range[0], self.snr_range[1])

                iq_samples, label = self.generate_iq_samples(
                    mod_type, 128, snr_db, seed=idx
                )

                iq_data[idx, :, 0] = iq_samples.real
                iq_data[idx, :, 1] = iq_samples.imag
                labels[idx] = label

                idx += 1

        return iq_data, labels

    def generate_batch(
        self, batch_size: int, shuffle: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成批次数据

        Args:
            batch_size: 批量大小
            shuffle: 是否打乱

        Returns:
            (批量IQ数据, 批量标签)
        """
        samples_per_class = batch_size // len(self.modulation_types)

        iq_batch = []
        label_batch = []

        for mod_type in self.modulation_types:
            modulator = self.modulators[mod_type]

            for _ in range(samples_per_class):
                snr_db = np.random.uniform(self.snr_range[0], self.snr_range[1])
                num_symbols = 128

                bits = np.random.randint(0, 2, num_symbols * modulator.bits_per_symbol)
                symbols = modulator.modulate(bits)
                iq_samples = self._pulse_shape(symbols)
                iq_samples = self._add_channel_effects(iq_samples)
                iq_samples = self._add_awgn(iq_samples, snr_db)

                iq_batch.append(np.stack([iq_samples.real, iq_samples.imag], axis=1))
                label_batch.append(self._create_label(mod_type))

        iq_batch = np.array(iq_batch)
        label_batch = np.array(label_batch)

        if shuffle:
            indices = np.random.permutation(len(iq_batch))
            iq_batch = iq_batch[indices]
            label_batch = label_batch[indices]

        return iq_batch, label_batch


def generate_training_data(
    num_samples_per_class: int = 1000, output_file: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    便捷函数：生成调制识别训练数据

    Args:
        num_samples_per_class: 每类样本数量
        output_file: 输出文件路径

    Returns:
        (IQ数据, 标签, 调制类型列表)
    """
    generator = SignalGenerator()
    iq_data, labels = generator.generate_dataset(num_samples_per_class)

    if output_file:
        np.savez(output_file, iq_data=iq_data, labels=labels)
        print(f"数据已保存至: {output_file}")

    return iq_data, labels, generator.modulation_types


if __name__ == "__main__":
    print("=" * 60)
    print("信号调制模块测试")
    print("=" * 60)

    print("\n1. 调制器测试:")
    for mod_type in ["BPSK", "QPSK", "16QAM", "64QAM"]:
        modulator = Modulator(mod_type)
        bits = np.random.randint(0, 2, 100 * modulator.bits_per_symbol)
        symbols = modulator.modulate(bits)
        print(f"   {mod_type}: {len(bits)} 比特 -> {len(symbols)} 符号")

    print("\n2. 星座图:")
    for mod_type in ["BPSK", "QPSK", "16QAM", "64QAM"]:
        modulator = Modulator(mod_type)
        print(f"   {mod_type}: {len(modulator.constellation)} 星座点")

    print("\n3. 信号生成器测试:")
    generator = SignalGenerator()
    iq_data, labels = generator.generate_dataset(
        num_samples_per_class=100, snr_distribution="uniform"
    )
    print(f"   生成数据集形状: IQ={iq_data.shape}, Labels={labels.shape}")

    print("\n4. 标签验证:")
    for i, mod_type in enumerate(generator.modulation_types):
        count = np.sum(np.argmax(labels, axis=1) == i)
        print(f"   {mod_type}: {count} 样本")

    print("\n" + "=" * 60)
    print("信号调制模块测试完成！")
    print("=" * 60)

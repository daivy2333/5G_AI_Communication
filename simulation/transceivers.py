"""
链路仿真模块 - 收发机模型
实现5G标准的OFDM收发机
"""

import numpy as np
from typing import Tuple, Optional, List
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from signal_detection.modulator import Modulator


class Transmitter:
    """
    数字通信发射机

    实现基带信号生成和调制
    """

    def __init__(self, modulation: str = "QPSK", sample_rate: float = 15.36e6):
        """
        初始化发射机

        Args:
            modulation: 调制方式
            sample_rate: 采样率(Hz)
        """
        self.modulation = modulation
        self.sample_rate = sample_rate
        self.modulator = Modulator(modulation)

    def generate_bits(self, num_bits: int) -> np.ndarray:
        """生成随机比特流"""
        return np.random.randint(0, 2, num_bits)

    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """调制比特流"""
        return self.modulator.modulate(bits)

    def pulse_shape(
        self, symbols: np.ndarray, samples_per_symbol: int = 4
    ) -> np.ndarray:
        """
        脉冲成形

        Args:
            symbols: 调制符号
            samples_per_symbol: 每符号采样数

        Returns:
            成形后的基带信号
        """
        num_samples = len(symbols) * samples_per_symbol
        signal = np.zeros(num_samples, dtype=complex)

        for i, s in enumerate(symbols):
            signal[i * samples_per_symbol] = s

        # 升余弦滤波器(简化)
        alpha = 0.25  # 滚降系数
        num_taps = samples_per_symbol * 8 + 1
        t = np.arange(num_taps) - num_taps // 2
        rc_pulse = (
            np.sinc(t / samples_per_symbol)
            * np.cos(np.pi * alpha * t / samples_per_symbol)
            / (1 - (2 * alpha * t / samples_per_symbol) ** 2 + 1e-10)
        )

        # 卷积
        shaped = np.convolve(signal, rc_pulse, mode="same")
        return shaped

    def upconvert(
        self, baseband: np.ndarray, carrier_freq: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        上变频到射频

        Args:
            baseband: 基带信号
            carrier_freq: 载波频率(Hz)

        Returns:
            (射频信号实部, 射频信号虚部)
        """
        t = np.arange(len(baseband)) / self.sample_rate
        i_signal = np.real(baseband)
        q_signal = np.imag(baseband)

        # IQ调制
        carrier_i = np.cos(2 * np.pi * carrier_freq * t)
        carrier_q = np.sin(2 * np.pi * carrier_freq * t)

        rf_i = i_signal * carrier_i
        rf_q = q_signal * carrier_q

        return rf_i, rf_q

    def transmit(
        self, num_bits: int, carrier_freq: float = 3.5e9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        执行完整发送过程

        Args:
            num_bits: 比特数
            carrier_freq: 载波频率

        Returns:
            (射频I路, 射频Q路, 原始比特)
        """
        # 生成比特
        bits = self.generate_bits(num_bits)

        # 调制
        symbols = self.modulate(bits)

        # 脉冲成形
        baseband = self.pulse_shape(symbols)

        # 上变频
        rf_i, rf_q = self.upconvert(baseband, carrier_freq)

        return rf_i, rf_q, bits


class Receiver:
    """
    数字通信接收机

    实现同步、解调、判决
    """

    def __init__(self, modulation: str = "QPSK", sample_rate: float = 15.36e6):
        """初始化接收机"""
        self.modulation = modulation
        self.sample_rate = sample_rate
        self.modulator = Modulator(modulation)

    def downconvert(
        self, rf_i: np.ndarray, rf_q: np.ndarray, carrier_freq: float
    ) -> np.ndarray:
        """
        下变频到基带

        Args:
            rf_i: 射频I路信号
            rf_q: 射频Q路信号
            carrier_freq: 载波频率

        Returns:
            基带复信号
        """
        t = np.arange(len(rf_i)) / self.sample_rate

        # 混频
        local_i = np.cos(2 * np.pi * carrier_freq * t)
        local_q = np.sin(2 * np.pi * carrier_freq * t)

        baseband_i = rf_i * local_i + rf_q * local_q
        baseband_q = rf_q * local_i - rf_i * local_q

        # 低通滤波(简化)
        baseband = baseband_i + 1j * baseband_q

        # 移动平均滤波
        kernel = np.ones(16) / 16
        baseband = np.convolve(baseband, kernel, mode="same")

        return baseband

    def match_filter(
        self, baseband: np.ndarray, samples_per_symbol: int = 4
    ) -> np.ndarray:
        """匹配滤波"""
        kernel = np.ones(samples_per_symbol)
        filtered = np.convolve(baseband, kernel, mode="same")
        return filtered

    def timing_recovery(
        self, baseband: np.ndarray, samples_per_symbol: int = 4
    ) -> np.ndarray:
        """符号同步(简化版)"""
        # 定时恢复(使用最大幅度位置)
        amplitude = np.abs(baseband)
        max_idx = np.argmax(amplitude)

        # 采样
        symbol_indices = np.arange(max_idx, len(baseband), samples_per_symbol)
        symbols = baseband[symbol_indices]

        return symbols[
            : len(symbol_indices)
            // samples_per_symbol
            * samples_per_symbol : samples_per_symbol
        ]

    def detect(self, symbols: np.ndarray) -> np.ndarray:
        """符号判决"""
        return self.modulator.demodulate(symbols)

    def receive(
        self, rf_i: np.ndarray, rf_q: np.ndarray, carrier_freq: float = 3.5e9
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行完整接收过程

        Args:
            rf_i: 射频I路信号
            rf_q: 射频Q路信号
            carrier_freq: 载波频率

        Returns:
            (检测比特, 基带信号)
        """
        # 下变频
        baseband = self.downconvert(rf_i, rf_q, carrier_freq)

        # 匹配滤波
        filtered = self.match_filter(baseband)

        # 定时恢复
        symbols = self.timing_recovery(filtered)

        # 判决
        detected_bits = self.detect(symbols)

        return detected_bits, baseband


class OFDMTransceiver:
    """
    OFDM收发机

    实现5G标准OFDM调制解调
    """

    def __init__(self, config: Optional[dict] = None):
        """
        初始化OFDM收发机

        Args:
            config: 配置字典
        """
        self.config = config or {}

        # OFDM参数
        self.fft_size = self.config.get("fft_size", 2048)
        self.num_subcarriers = self.config.get("num_subcarriers", 1200)
        self.cp_length = self.config.get("cp_length", 512)
        self.modulation = self.config.get("modulation", "QPSK")
        self.modulator = Modulator(self.modulation)

        # 导频参数
        self.pilot_spacing = self.config.get("pilot_spacing", 8)

    def generate_pilot_subcarriers(self, num_symbols: int) -> np.ndarray:
        """生成导频子载波"""
        num_pilots = self.num_subcarriers // self.pilot_spacing
        pilots = np.random.choice([1, -1], size=(num_symbols, num_pilots))
        return pilots.astype(complex)

    def ofdm_modulate(self, data_symbols: np.ndarray) -> np.ndarray:
        """
        OFDM调制

        Args:
            data_symbols: 数据符号，形状为(num_symbols, num_data_carriers)

        Returns:
            OFDM时域信号
        """
        num_symbols = len(data_symbols)
        ofdm_signal = np.zeros(
            (num_symbols, self.fft_size + self.cp_length), dtype=complex
        )

        symbol_lengths = np.array([len(s) for s in data_symbols])
        start_indices = (self.fft_size - symbol_lengths) // 2
        
        freq_matrix = np.zeros((num_symbols, self.fft_size), dtype=complex)
        for i in range(num_symbols):
            freq_matrix[i, start_indices[i]:start_indices[i] + symbol_lengths[i]] = data_symbols[i]
        
        time_matrix = np.fft.ifft(freq_matrix, axis=1)
        
        ofdm_signal[:, :self.cp_length] = time_matrix[:, -self.cp_length:]
        ofdm_signal[:, self.cp_length:] = time_matrix

        return ofdm_signal.flatten()

    def ofdm_demodulate(
        self, received_signal: np.ndarray, num_symbols: int
    ) -> np.ndarray:
        """
        OFDM解调

        Args:
            received_signal: 接收信号
            num_symbols: OFDM符号数

        Returns:
            频域数据符号
        """
        # 重塑
        signal_per_symbol = self.fft_size + self.cp_length
        received_per_symbol = received_signal[
            : num_symbols * signal_per_symbol
        ].reshape(num_symbols, -1)

        no_cp_matrix = received_per_symbol[:, self.cp_length:]
        
        freq_matrix = np.fft.fft(no_cp_matrix, axis=1)
        
        start_idx = (self.fft_size - self.num_subcarriers) // 2
        freq_data = freq_matrix[:, start_idx:start_idx + self.num_subcarriers]

        return freq_data

    def channel_encode(self, bits: np.ndarray, rate: float = 0.5) -> np.ndarray:
        """
        信道编码(卷积码，简化版)

        Args:
            bits: 输入比特
            rate: 编码速率

        Returns:
            编码后比特
        """
        # 重复编码(简化)
        num_coded = int(len(bits) / rate)
        coded = np.repeat(bits, int(1 / rate))[:num_coded]
        return coded

    def channel_decode(self, soft_bits: np.ndarray, rate: float = 0.5) -> np.ndarray:
        """
        信道译码(简化版)

        Args:
            soft_bits: 软比特
            rate: 编码速率

        Returns:
            译码比特
        """
        # 合并重复编码的比特（多数判决）
        repeat_factor = int(1 / rate)
        num_original_bits = len(soft_bits) // repeat_factor

        reshaped = soft_bits[:num_original_bits * repeat_factor].reshape(num_original_bits, repeat_factor)
        decoded = (reshaped.mean(axis=1) > 0.5).astype(int)
        
        return decoded

    def transmit(self, bits: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        完整OFDM发送过程

        Args:
            bits: 信息比特

        Returns:
            (发送信号, 编码比特, 发送信息)
        """
        # 信道编码
        coded_bits = self.channel_encode(bits, rate=0.5)

        # 调制
        symbols = self.modulator.modulate(coded_bits)

        # 计算每个符号的数据子载波数（除去导频）
        num_pilots = self.num_subcarriers // self.pilot_spacing
        num_data_per_symbol = self.num_subcarriers - num_pilots

        # 计算需要的OFDM符号数
        num_symbols = len(symbols) // num_data_per_symbol
        if num_symbols == 0:
            num_symbols = 1

        # 创建数据矩阵（只包含数据位置）
        data_symbols = symbols[: num_symbols * num_data_per_symbol]
        if len(data_symbols) < num_symbols * num_data_per_symbol:
            # 填充零
            data_symbols = np.concatenate(
                [
                    data_symbols,
                    np.zeros(
                        num_symbols * num_data_per_symbol - len(data_symbols),
                        dtype=complex,
                    ),
                ]
            )

        # 构建完整的传输矩阵（数据+导频）
        pilot_positions = np.arange(0, self.num_subcarriers, self.pilot_spacing)[
            :num_pilots
        ]
        data_positions = np.setdiff1d(np.arange(self.num_subcarriers), pilot_positions)

        tx_matrix = np.zeros((num_symbols, self.num_subcarriers), dtype=complex)
        pilots = self.generate_pilot_subcarriers(num_symbols)
        
        tx_matrix[:, pilot_positions] = pilots
        
        data_symbols_matrix = data_symbols[:num_symbols * num_data_per_symbol].reshape(num_symbols, num_data_per_symbol)
        tx_matrix[:, data_positions] = data_symbols_matrix

        # OFDM调制
        tx_signal = self.ofdm_modulate(tx_matrix)

        info = {
            "num_symbols": num_symbols,
            "coded_bits": coded_bits,
            "data_symbols": tx_matrix,
            "pilots": pilots,
            "pilot_positions": pilot_positions,
            "data_positions": data_positions,
            "num_data_per_symbol": num_data_per_symbol,
        }

        return tx_signal, coded_bits, info

    def receive(
        self, rx_signal: np.ndarray, info: dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        完整OFDM接收过程

        Args:
            rx_signal: 接收信号
            info: 发送时保存的信息

        Returns:
            (译码比特, 估计信道)
        """
        num_symbols = info["num_symbols"]

        # OFDM解调
        freq_data = self.ofdm_demodulate(rx_signal, num_symbols)

        # 信道估计(简化LS)
        pilots = info["pilots"]
        pilot_positions = info["pilot_positions"]
        data_positions = info["data_positions"]

        channel_estimates = np.zeros_like(freq_data)
        
        h_ls = freq_data[:, pilot_positions] / (pilots + 1e-10)
        channel_estimates[:, pilot_positions] = h_ls
        
        nearest_pilot_indices = np.array([np.argmin(np.abs(pilot_positions - pos)) for pos in data_positions])
        channel_estimates[:, data_positions] = h_ls[:, nearest_pilot_indices]

        # 信道均衡
        equalized = freq_data / (channel_estimates + 1e-10)

        # 提取数据子载波
        data_symbols = equalized[:, data_positions].flatten()

        # 解调
        soft_bits = self.modulator.demodulate(
            data_symbols[: len(info["coded_bits"])], method="soft"
        )

        # 信道译码
        decoded_bits = self.channel_decode(soft_bits)

        return decoded_bits[: len(info["coded_bits"]) // 2], channel_estimates


def ofdm_link_simulation(
    snr_db: float = 20, num_bits: int = 10000, modulation: str = "QPSK"
) -> dict:
    """
    便捷函数：OFDM链路仿真

    Args:
        snr_db: 信噪比(dB)
        num_bits: 比特数
        modulation: 调制方式

    Returns:
        仿真结果字典
    """
    # 创建收发机
    config = {
        "fft_size": 2048,
        "num_subcarriers": 1200,
        "cp_length": 512,
        "modulation": modulation,
    }
    transceiver = OFDMTransceiver(config)

    # 发送
    original_bits = np.random.randint(0, 2, num_bits)
    tx_signal, coded_bits, info = transceiver.transmit(original_bits)

    # 信道(添加噪声)
    signal_power = np.mean(np.abs(tx_signal) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(len(tx_signal)) + 1j * np.random.randn(len(tx_signal))
    )
    rx_signal = tx_signal + noise

    # 接收
    decoded_bits, channel = transceiver.receive(rx_signal, info)

    # 计算BER (比较原始比特和解码比特)
    min_len = min(len(original_bits), len(decoded_bits))
    num_errors = np.sum(original_bits[:min_len] != decoded_bits[:min_len])
    ber = num_errors / min_len if min_len > 0 else 0

    # 计算吞吐量
    throughput = len(decoded_bits) / len(tx_signal)

    return {
        "ber": ber,
        "num_errors": num_errors,
        "num_bits": len(original_bits),
        "snr_db": snr_db,
        "throughput": throughput,
        "channel_estimate": channel,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("OFDM收发机测试")
    print("=" * 60)

    # 测试基础收发机
    print("\n1. 基础收发机测试:")
    tx = Transmitter(modulation="QPSK")
    rx = Receiver(modulation="QPSK")

    rf_i, rf_q, bits = tx.transmit(1000)
    print(f"   发送比特数: {len(bits)}")
    print(f"   射频信号长度: {len(rf_i)}")

    detected, baseband = rx.receive(rf_i, rf_q)
    print(f"   检测比特数: {len(detected)}")

    # 测试OFDM收发机
    print("\n2. OFDM收发机测试:")
    config = {
        "fft_size": 256,
        "num_subcarriers": 180,
        "cp_length": 64,
        "modulation": "QPSK",
    }
    ofdm = OFDMTransceiver(config)

    # 发送
    original_bits = np.random.randint(0, 2, 1000)
    tx_signal, coded_bits, info = ofdm.transmit(original_bits)
    print(f"   OFDM符号数: {info['num_symbols']}")
    print(f"   发送信号长度: {len(tx_signal)}")

    # 添加噪声
    snr_db = 20
    signal_power = np.mean(np.abs(tx_signal) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(len(tx_signal)) + 1j * np.random.randn(len(tx_signal))
    )
    rx_signal = tx_signal + noise

    # 接收
    decoded, channel = ofdm.receive(rx_signal, info)
    print(f"   译码比特数: {len(decoded)}")

    # BER (比较原始比特和解码比特)
    min_len = min(len(original_bits), len(decoded))
    errors = np.sum(original_bits[:min_len] != decoded[:min_len])
    ber = errors / min_len
    print(f"   误码率: {ber:.6f}")

    # SNR扫描
    print("\n3. SNR-BER曲线测试:")
    snr_range = range(-5, 30, 5)
    results = []

    for snr in snr_range:
        result = ofdm_link_simulation(snr_db=snr, num_bits=5000, modulation="QPSK")
        results.append(result)
        print(f"   SNR={snr:3d}dB: BER={result['ber']:.6f}")

    print("\n" + "=" * 60)
    print("OFDM收发机测试完成！")
    print("=" * 60)

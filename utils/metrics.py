"""
性能评估模块
实现5G通信系统各模块的性能指标计算
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class ChannelEstimationMetrics:
    """信道估计性能指标"""
    
    @staticmethod
    def compute_nmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        计算归一化均方误差(NMSE)
        
        Args:
            y_true: 真实信道
            y_pred: 估计信道
            
        Returns:
            NMSE值(dB)
        """
        signal_power = np.mean(np.abs(y_true) ** 2)
        error_power = np.mean(np.abs(y_true - y_pred) ** 2)
        if signal_power == 0:
            return float('inf')
        nmse_db = 10 * np.log10(error_power / signal_power + 1e-10)
        return nmse_db
    
    @staticmethod
    def compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算均方误差(MSE)"""
        return np.mean(np.abs(y_true - y_pred) ** 2)
    
    @staticmethod
    def compute_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算信道相关性"""
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        correlation = np.abs(np.corrcoef(y_true_flat, y_pred_flat)[0, 1])
        return correlation
    
    @staticmethod
    def evaluate_estimator(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        全面评估信道估计器
        
        Args:
            y_true: 真实信道
            y_pred: 估计信道
            
        Returns:
            性能指标字典
        """
        return {
            'nmse_db': ChannelEstimationMetrics.compute_nmse(y_true, y_pred),
            'mse': ChannelEstimationMetrics.compute_mse(y_true, y_pred),
            'correlation': ChannelEstimationMetrics.compute_correlation(y_true, y_pred)
        }


class ModulationRecognitionMetrics:
    """调制识别性能指标"""
    
    @staticmethod
    def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        计算识别准确率
        
        Args:
            y_true: 真实标签 (num_samples,) 或 (num_samples, num_classes)
            y_pred: 预测标签 (num_samples,) 或 (num_samples, num_classes)
            
        Returns:
            准确率
        """
        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)
        if len(y_pred.shape) > 1:
            y_pred = np.argmax(y_pred, axis=1)
        return np.mean(y_true == y_pred)
    
    @staticmethod
    def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                                 num_classes: int) -> np.ndarray:
        """
        计算混淆矩阵
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            num_classes: 类别数量
            
        Returns:
            混淆矩阵 (num_classes, num_classes)
        """
        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)
        if len(y_pred.shape) > 1:
            y_pred = np.argmax(y_pred, axis=1)
            
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        return cm
    
    @staticmethod
    def compute_per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[int, float]:
        """
        计算各类别准确率
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            
        Returns:
            各类别准确率字典
        """
        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)
        if len(y_pred.shape) > 1:
            y_pred = np.argmax(y_pred, axis=1)
            
        unique_classes = np.unique(y_true)
        per_class_acc = {}
        for cls in unique_classes:
            mask = y_true == cls
            per_class_acc[int(cls)] = np.mean(y_pred[mask] == y_true[mask])
        return per_class_acc
    
    @staticmethod
    def evaluate_recognizer(y_true: np.ndarray, y_pred: np.ndarray, 
                           class_names: Optional[List[str]] = None) -> Dict:
        """
        全面评估调制识别器
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            class_names: 类别名称列表
            
        Returns:
            性能指标字典
        """
        if len(y_true.shape) > 1:
            num_classes = y_true.shape[1]
        else:
            num_classes = len(np.unique(y_true))
            
        accuracy = ModulationRecognitionMetrics.compute_accuracy(y_true, y_pred)
        per_class_acc = ModulationRecognitionMetrics.compute_per_class_accuracy(y_true, y_pred)
        confusion_matrix = ModulationRecognitionMetrics.compute_confusion_matrix(
            y_true, y_pred, num_classes
        )
        
        if class_names:
            per_class_acc = {class_names[k]: v for k, v in per_class_acc.items()}
        
        return {
            'overall_accuracy': accuracy,
            'per_class_accuracy': per_class_acc,
            'confusion_matrix': confusion_matrix
        }


class SignalDetectionMetrics:
    """信号检测性能指标"""
    
    @staticmethod
    def compute_ber(original_bits: np.ndarray, detected_bits: np.ndarray) -> float:
        """
        计算误码率(BER)
        
        Args:
            original_bits: 原始比特
            detected_bits: 检测比特
            
        Returns:
            误码率
        """
        min_len = min(len(original_bits), len(detected_bits))
        errors = np.sum(original_bits[:min_len] != detected_bits[:min_len])
        return errors / min_len if min_len > 0 else 0.0
    
    @staticmethod
    def compute_ser(original_symbols: np.ndarray, detected_symbols: np.ndarray) -> float:
        """
        计算误符号率(SER)
        
        Args:
            original_symbols: 原始符号
            detected_symbols: 检测符号
            
        Returns:
            误符号率
        """
        min_len = min(len(original_symbols), len(detected_symbols))
        errors = np.sum(original_symbols[:min_len] != detected_symbols[:min_len])
        return errors / min_len if min_len > 0 else 0.0
    
    @staticmethod
    def compute_evm(reference_symbols: np.ndarray, measured_symbols: np.ndarray) -> float:
        """
        计算误差矢量幅度(EVM)
        
        Args:
            reference_symbols: 参考符号
            measured_symbols: 测量符号
            
        Returns:
            EVM值(%)
        """
        min_len = min(len(reference_symbols), len(measured_symbols))
        ref = reference_symbols[:min_len]
        meas = measured_symbols[:min_len]
        
        error = np.abs(ref - meas) ** 2
        reference_power = np.mean(np.abs(ref) ** 2)
        
        evm = np.sqrt(np.mean(error) / reference_power) * 100
        return evm
    
    @staticmethod
    def evaluate_detector(original_bits: np.ndarray, detected_bits: np.ndarray,
                         original_symbols: Optional[np.ndarray] = None,
                         detected_symbols: Optional[np.ndarray] = None) -> Dict:
        """
        全面评估信号检测器
        
        Args:
            original_bits: 原始比特
            detected_bits: 检测比特
            original_symbols: 原始符号
            detected_symbols: 检测符号
            
        Returns:
            性能指标字典
        """
        results = {
            'ber': SignalDetectionMetrics.compute_ber(original_bits, detected_bits)
        }
        
        if original_symbols is not None and detected_symbols is not None:
            results['ser'] = SignalDetectionMetrics.compute_ser(
                original_symbols, detected_symbols
            )
            results['evm'] = SignalDetectionMetrics.compute_evm(
                original_symbols, detected_symbols
            )
        
        return results


class ResourceSchedulingMetrics:
    """资源调度性能指标"""
    
    @staticmethod
    def compute_jain_fairness(allocated_resources: np.ndarray) -> float:
        """
        计算Jain公平指数
        
        Args:
            allocated_resources: 各用户分配的资源
            
        Returns:
            Jain公平指数 (0-1, 1表示完全公平)
        """
        n = len(allocated_resources)
        if n == 0:
            return 0.0
        
        numerator = np.sum(allocated_resources) ** 2
        denominator = n * np.sum(allocated_resources ** 2)
        
        return numerator / denominator if denominator > 0 else 0.0
    
    @staticmethod
    def compute_spectral_efficiency(throughput: float, bandwidth: float) -> float:
        """
        计算频谱效率
        
        Args:
            throughput: 吞吐量 (bps)
            bandwidth: 带宽 (Hz)
            
        Returns:
            频谱效率 (bps/Hz)
        """
        return throughput / bandwidth if bandwidth > 0 else 0.0
    
    @staticmethod
    def compute_energy_efficiency(throughput: float, power: float) -> float:
        """
        计算能效
        
        Args:
            throughput: 吞吐量 (bps)
            power: 功率 (W)
            
        Returns:
            能效 (bps/W)
        """
        return throughput / power if power > 0 else 0.0
    
    @staticmethod
    def compute_outage_probability(snr_values: np.ndarray, 
                                   snr_threshold: float) -> float:
        """
        计算中断概率
        
        Args:
            snr_values: SNR值数组
            snr_threshold: SNR阈值(dB)
            
        Returns:
            中断概率
        """
        return np.mean(snr_values < snr_threshold)
    
    @staticmethod
    def evaluate_scheduler(throughput: float, fairness: float,
                          bandwidth: float, power: float,
                          user_rates: Optional[np.ndarray] = None) -> Dict:
        """
        全面评估资源调度器
        
        Args:
            throughput: 总吞吐量
            fairness: 公平指数
            bandwidth: 带宽
            power: 功率
            user_rates: 各用户速率
            
        Returns:
            性能指标字典
        """
        results = {
            'throughput': throughput,
            'fairness': fairness,
            'spectral_efficiency': ResourceSchedulingMetrics.compute_spectral_efficiency(
                throughput, bandwidth
            ),
            'energy_efficiency': ResourceSchedulingMetrics.compute_energy_efficiency(
                throughput, power
            )
        }
        
        if user_rates is not None:
            results['jain_index'] = ResourceSchedulingMetrics.compute_jain_fairness(
                user_rates
            )
        
        return results


def compare_algorithms(results_dict: Dict[str, Dict], metric_name: str) -> Dict:
    """
    对比不同算法的性能
    
    Args:
        results_dict: 算法结果字典 {算法名: 结果字典}
        metric_name: 指标名称
        
    Returns:
        对比结果字典
    """
    comparison = {}
    for algo_name, results in results_dict.items():
        if metric_name in results:
            comparison[algo_name] = results[metric_name]
    
    sorted_comparison = dict(sorted(comparison.items(), key=lambda x: x[1], reverse=True))
    return sorted_comparison


if __name__ == '__main__':
    print("=" * 60)
    print("性能评估模块测试")
    print("=" * 60)
    
    print("\n1. 信道估计指标测试:")
    y_true = np.random.randn(100, 16) + 1j * np.random.randn(100, 16)
    noise = (np.random.randn(100, 16) + 1j * np.random.randn(100, 16)) * 0.1
    y_pred = y_true + noise
    
    ce_metrics = ChannelEstimationMetrics.evaluate_estimator(y_true, y_pred)
    print(f"   NMSE: {ce_metrics['nmse_db']:.2f} dB")
    print(f"   MSE: {ce_metrics['mse']:.6f}")
    print(f"   相关系数: {ce_metrics['correlation']:.4f}")
    
    print("\n2. 调制识别指标测试:")
    y_true_cls = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    y_pred_cls = np.array([0, 1, 2, 0, 2, 2, 0, 1, 1, 0])
    
    mr_metrics = ModulationRecognitionMetrics.evaluate_recognizer(
        y_true_cls, y_pred_cls, class_names=['BPSK', 'QPSK', '16QAM']
    )
    print(f"   总体准确率: {mr_metrics['overall_accuracy']:.2%}")
    print(f"   各类别准确率: {mr_metrics['per_class_accuracy']}")
    
    print("\n3. 信号检测指标测试:")
    bits_true = np.random.randint(0, 2, 1000)
    bits_pred = bits_true.copy()
    bits_pred[:50] = 1 - bits_pred[:50]
    
    sd_metrics = SignalDetectionMetrics.evaluate_detector(bits_true, bits_pred)
    print(f"   BER: {sd_metrics['ber']:.4f}")
    
    print("\n4. 资源调度指标测试:")
    user_rates = np.array([10, 12, 8, 15, 10])
    
    rs_metrics = ResourceSchedulingMetrics.evaluate_scheduler(
        throughput=np.sum(user_rates) * 1e6,
        fairness=0.85,
        bandwidth=100e6,
        power=1.0,
        user_rates=user_rates
    )
    print(f"   Jain公平指数: {rs_metrics['jain_index']:.4f}")
    print(f"   频谱效率: {rs_metrics['spectral_efficiency']:.2f} bps/Hz")
    print(f"   能效: {rs_metrics['energy_efficiency']:.2e} bps/W")
    
    print("\n" + "=" * 60)
    print("性能评估模块测试完成！")
    print("=" * 60)
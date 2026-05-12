"""
AI信号检测与调制识别模块 - 信号检测器
实现基于深度学习的信号检测算法
"""

import numpy as np
from typing import Tuple, Optional, Dict, List, TYPE_CHECKING
import sys
import os

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from tqdm import tqdm

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("警告: PyTorch未安装，信号检测功能将无法使用")

if os.path.dirname(__file__):
    sys.path.insert(0, os.path.dirname(__file__))
try:
    from modulator import SignalGenerator, Modulator
except ImportError:
    from .modulator import SignalGenerator, Modulator


class SignalDetectorBase:
    """
    信号检测器基类

    定义信号检测的通用接口
    """

    def __init__(self, config: Optional[dict] = None):
        """初始化检测器"""
        self.config = config or {}
        self.model = None
        self.is_trained = False

    def train(
        self,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        val_data: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None,
    ):
        """训练模型"""
        raise NotImplementedError

    def detect(self, signals: np.ndarray) -> np.ndarray:
        """检测信号"""
        raise NotImplementedError

    def evaluate(self, test_data: np.ndarray, test_labels: np.ndarray) -> Dict:
        """评估性能"""
        raise NotImplementedError


class CNNLSTMDetector(SignalDetectorBase):
    """
    基于CNN-LSTM的信号检测器

    使用CNN进行空间特征提取，LSTM进行时序建模
    """

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)

        # 设备设置
        if HAS_TORCH:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"使用设备: {self.device}")
        else:
            self.device = None

        # 网络参数
        self.cnn_filters = self.config.get("cnn_filters", [64, 128, 256])
        self.lstm_units = self.config.get("lstm_units", 128)
        self.dropout_rate = self.config.get("dropout_rate", 0.3)

        # 训练参数
        self.batch_size = self.config.get("batch_size", 128)
        self.epochs = self.config.get("epochs", 50)
        self.learning_rate = self.config.get("learning_rate", 1e-3)

        # 训练历史
        self.history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

    def _build_torch_model(self, input_shape: Tuple[int, int]):
        """构建PyTorch模型"""

        class CNNLSTMNet(nn.Module):
            def __init__(self, cnn_filters, lstm_units, dropout_rate):
                super().__init__()

                # CNN特征提取
                self.conv1 = nn.Conv1d(2, cnn_filters[0], kernel_size=3, padding=1)
                self.bn1 = nn.BatchNorm1d(cnn_filters[0])
                self.conv2 = nn.Conv1d(
                    cnn_filters[0], cnn_filters[1], kernel_size=3, padding=1
                )
                self.bn2 = nn.BatchNorm1d(cnn_filters[1])
                self.conv3 = nn.Conv1d(
                    cnn_filters[1], cnn_filters[2], kernel_size=3, padding=1
                )
                self.bn3 = nn.BatchNorm1d(cnn_filters[2])
                self.pool = nn.AdaptiveAvgPool1d(64)

                # Bi-LSTM
                self.lstm = nn.LSTM(
                    input_size=cnn_filters[2],
                    hidden_size=lstm_units,
                    num_layers=2,
                    batch_first=True,
                    bidirectional=True,
                )

                # 全连接层
                self.fc1 = nn.Linear(lstm_units * 2, 128)
                self.fc2 = nn.Linear(128, 1)
                self.dropout = nn.Dropout(dropout_rate)

            def forward(self, x):
                # x: (batch, seq_len, 2) -> (batch, 2, seq_len)
                x = x.transpose(1, 2)

                # CNN
                x = F.relu(self.bn1(self.conv1(x)))
                x = F.relu(self.bn2(self.conv2(x)))
                x = F.relu(self.bn3(self.conv3(x)))
                x = self.pool(x)

                # -> (batch, seq_len, features)
                x = x.transpose(1, 2)

                # LSTM
                x, _ = self.lstm(x)

                # 取最后时刻
                x = x[:, -1, :]

                # 全连接
                x = self.dropout(x)
                x = F.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)

                return torch.sigmoid(x)

        return CNNLSTMNet(self.cnn_filters, self.lstm_units, self.dropout_rate)

    def train(
        self,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        val_data: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None,
    ):
        """
        训练检测器

        Args:
            train_data: 训练数据，形状为 (num_samples, seq_len, 2)
            train_labels: 训练标签，形状为 (num_samples,)
            val_data: 验证数据
            val_labels: 验证标签
        """
        print(f"训练CNN-LSTM信号检测器...")
        print(f"训练数据形状: {train_data.shape}")

        if HAS_TORCH:
            # PyTorch实现
            self.model = self._build_torch_model(train_data.shape[1:]).to(self.device)
            train_tensor = TensorDataset(
                torch.FloatTensor(train_data), torch.FloatTensor(train_labels)
            )
            train_loader = DataLoader(
                train_tensor, batch_size=self.batch_size, shuffle=True
            )

            criterion = nn.BCELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

            for epoch in range(self.epochs):
                self.model.train()
                epoch_loss = 0
                epoch_acc = 0
                num_batches = 0

                pbar = tqdm(
                    train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}", leave=True
                )
                for batch_data, batch_labels in pbar:
                    batch_data = batch_data.to(self.device)
                    batch_labels = batch_labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(batch_data).squeeze()
                    loss = criterion(outputs, batch_labels)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    batch_acc = (
                        ((outputs > 0.5).float() == batch_labels).float().mean().item()
                    )
                    epoch_acc += batch_acc
                    num_batches += 1

                    pbar.set_postfix(
                        {"loss": f"{loss.item():.4f}", "acc": f"{batch_acc:.4f}"}
                    )

                avg_loss = epoch_loss / num_batches
                avg_acc = epoch_acc / num_batches
                self.history["loss"].append(avg_loss)
                self.history["accuracy"].append(avg_acc)

                # 验证
                if val_data is not None and val_labels is not None:
                    self.model.eval()
                    with torch.no_grad():
                        val_tensor = TensorDataset(
                            torch.FloatTensor(val_data), torch.FloatTensor(val_labels)
                        )
                        val_loader = DataLoader(val_tensor, batch_size=self.batch_size)
                        val_loss = 0
                        val_acc = 0
                        val_batches = 0

                        for val_batch_data, val_batch_labels in val_loader:
                            val_batch_data = val_batch_data.to(self.device)
                            val_batch_labels = val_batch_labels.to(self.device)
                            val_outputs = self.model(val_batch_data).squeeze()
                            val_loss += criterion(val_outputs, val_batch_labels).item()
                            val_acc += (
                                ((val_outputs > 0.5).float() == val_batch_labels)
                                .float()
                                .mean()
                                .item()
                            )
                            val_batches += 1

                        self.history["val_loss"].append(val_loss / val_batches)
                        self.history["val_accuracy"].append(val_acc / val_batches)

                if (epoch + 1) % 10 == 0:
                    val_str = ""
                    if val_data is not None:
                        val_str = f" - Val Loss: {self.history['val_loss'][-1]:.4f} - Val Acc: {self.history['val_accuracy'][-1]:.4f}"
                    print(
                        f"Epoch {epoch + 1}/{self.epochs} - Loss: {avg_loss:.4f} - Acc: {avg_acc:.4f}{val_str}"
                    )

        else:
            print("警告: 未安装PyTorch，使用简化实现")
            self._train_simplified(train_data, train_labels)

        self.is_trained = True
        print("训练完成！")

    def _train_simplified(self, train_data: np.ndarray, train_labels: np.ndarray):
        """简化训练(无深度学习框架)"""
        # 使用简单的统计方法作为占位
        self.mean_signal = np.mean(train_data, axis=0)
        self.std_signal = np.std(train_data, axis=0)
        self.is_trained = True

    def detect(self, signals: np.ndarray) -> np.ndarray:
        """
        检测信号是否存在

        Args:
            signals: 输入信号，形状为 (num_samples, seq_len, 2)

        Returns:
            检测结果概率
        """
        if not self.is_trained:
            raise ValueError("模型未训练")

        if HAS_TORCH and isinstance(self.model, nn.Module):
            self.model.eval()
            with torch.no_grad():
                inputs = torch.FloatTensor(signals).to(self.device)
                outputs = self.model(inputs).squeeze().cpu().numpy()
            return outputs
        else:
            # 简化检测
            signal_power = np.mean(
                np.abs(signals[:, :, 0] + 1j * signals[:, :, 1]) ** 2, axis=1
            )
            threshold = np.mean(signal_power) * 0.5
            return (signal_power > threshold).astype(float)

    def detect_modulation(self, signals: np.ndarray) -> np.ndarray:
        """
        检测信号调制方式

        Args:
            signals: 输入信号

        Returns:
            调制方式预测概率
        """
        # 使用简化特征
        signal_complex = signals[:, :, 0] + 1j * signals[:, :, 1]

        # 提取特征
        features = []
        for sig in signal_complex:
            feat = [
                np.std(np.abs(sig)),  # 幅度标准差
                np.mean(np.abs(sig)),  # 平均幅度
                np.std(np.angle(sig)),  # 相位标准差
                np.mean(np.abs(np.diff(np.angle(sig)))),  # 瞬时频率
                np.max(np.abs(sig)) / (np.mean(np.abs(sig)) + 1e-10),  # 峰均比
            ]
            features.append(feat)

        features = np.array(features)
        return features


class SignalDetector:
    """
    信号检测器(兼容接口)

    支持传统算法和AI算法
    """

    def __init__(self, method: str = "ai", config: Optional[dict] = None):
        """
        初始化信号检测器

        Args:
            method: 检测方法 ('ai', 'energy', 'matched_filter')
            config: 配置字典
        """
        self.method = method
        self.config = config or {}

        if method == "ai":
            self.detector = CNNLSTMDetector(config)
        else:
            self.detector = None

    def train(
        self,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        val_data: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None,
    ):
        """训练检测器"""
        if self.detector:
            self.detector.train(train_data, train_labels, val_data, val_labels)

    def detect(self, signals: np.ndarray) -> np.ndarray:
        """检测信号"""
        if self.method == "energy":
            return self._energy_detection(signals)
        elif self.method == "matched_filter":
            return self._matched_filter_detection(signals)
        elif self.detector:
            return self.detector.detect(signals)
        else:
            raise ValueError("未指定检测方法")

    def _energy_detection(
        self, signals: np.ndarray, threshold: Optional[float] = None
    ) -> np.ndarray:
        """能量检测"""
        signal_power = np.mean(
            np.abs(signals[:, :, 0] + 1j * signals[:, :, 1]) ** 2, axis=1
        )

        if threshold is None:
            threshold = np.mean(signal_power) * 0.1

        return (signal_power > threshold).astype(float)

    def _matched_filter_detection(self, signals: np.ndarray) -> np.ndarray:
        """匹配滤波器检测"""
        # 简化的匹配滤波器
        template = np.mean(signals[:, :, 0] + 1j * signals[:, :, 1], axis=0)
        template = template / np.max(np.abs(template))

        detections = []
        for sig in signals:
            sig_complex = sig[:, 0] + 1j * sig[:, 1]
            matched = np.correlate(sig_complex, template, mode="valid")
            detections.append(np.max(np.abs(matched)))

        return np.array(detections) / np.max(detections)


class DetectorTrainer:
    """
    信号检测器训练器

    提供完整的训练流程和评估功能
    """

    def __init__(self, config: Optional[dict] = None):
        """初始化训练器"""
        self.config = config or {}
        self.detector = SignalDetector(method="ai", config=config)
        self.signal_generator = SignalGenerator(config)

    def generate_training_data(
        self, num_samples: int = 5000, signal_ratio: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成训练数据

        Args:
            num_samples: 总样本数
            signal_ratio: 信号样本比例

        Returns:
            (数据, 标签)
        """
        num_signal = int(num_samples * signal_ratio)
        num_noise = num_samples - num_signal

        data_list = []
        labels = []

        # 生成信号样本
        print(f"生成 {num_signal} 个信号样本...")
        for i in tqdm(range(num_signal), desc="生成信号"):
            mod_type = np.random.choice(self.signal_generator.modulation_types)
            snr_db = np.random.uniform(-10, 30)

            iq_samples, _ = self.signal_generator.generate_iq_samples(
                mod_type, 128, snr_db, seed=i
            )

            data_list.append(np.stack([iq_samples.real, iq_samples.imag], axis=1))
            labels.append(1)

        # 生成噪声样本
        print(f"生成 {num_noise} 个噪声样本...")
        for i in tqdm(range(num_noise), desc="生成噪声"):
            snr_db = -20  # 极低SNR
            num_symbols = 128
            samples_per_symbol = self.signal_generator.samples_per_symbol
            noise = np.random.randn(
                num_symbols * samples_per_symbol
            ) + 1j * np.random.randn(num_symbols * samples_per_symbol)

            data_list.append(np.stack([noise.real, noise.imag], axis=1))
            labels.append(0)

        data = np.array(data_list)
        labels = np.array(labels)

        # 打乱
        indices = np.random.permutation(len(data))
        data = data[indices]
        labels = labels[indices]

        return data, labels

    def train(self, num_samples: int = 5000):
        """训练检测器"""
        print("=" * 60)
        print("信号检测器训练")
        print("=" * 60)

        # 生成数据
        train_data, train_labels = self.generate_training_data(num_samples)

        # 划分训练集和验证集
        split_idx = int(len(train_data) * 0.8)
        val_data = train_data[split_idx:]
        val_labels = train_labels[split_idx:]
        train_data = train_data[:split_idx]
        train_labels = train_labels[:split_idx]

        # 训练
        self.detector.train(train_data, train_labels, val_data, val_labels)

        return train_data, train_labels

    def evaluate(self, num_samples: int = 1000) -> Dict:
        """评估检测器性能"""
        print("\n评估检测器性能...")

        test_data, test_labels = self.generate_training_data(num_samples)
        predictions = self.detector.detect(test_data)

        # 计算指标
        pred_binary = (predictions > 0.5).astype(int)
        accuracy = np.mean(pred_binary == test_labels)

        # 混淆矩阵元素
        tp = np.sum((pred_binary == 1) & (test_labels == 1))
        tn = np.sum((pred_binary == 0) & (test_labels == 0))
        fp = np.sum((pred_binary == 1) & (test_labels == 0))
        fn = np.sum((pred_binary == 0) & (test_labels == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": {
                "TP": int(tp),
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
            },
        }

        print(f"\n检测性能:")
        print(f"  准确率: {accuracy:.4f}")
        print(f"  精确率: {precision:.4f}")
        print(f"  召回率: {recall:.4f}")
        print(f"  F1分数: {f1:.4f}")

        return results


def main():
    """主函数"""
    print("=" * 60)
    print("信号检测器训练与评估")
    print("=" * 60)

    trainer = DetectorTrainer()
    trainer.train(num_samples=2000)
    results = trainer.evaluate(num_samples=500)

    return results


if __name__ == "__main__":
    results = main()

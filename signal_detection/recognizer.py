"""
AI信号检测与调制识别模块 - 调制方式识别器
实现基于深度学习的自动调制识别(AMR)系统
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from pathlib import Path
import sys

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# 尝试导入深度学习框架
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
    print("警告: PyTorch未安装，调制识别功能将无法使用")

from .modulator import SignalGenerator


class ModulationRecognizerBase:
    """调制识别器基类"""

    def __init__(self, num_classes: int, config: Optional[dict] = None):
        self.num_classes = num_classes
        self.config = config or {}
        self.model = None
        self.history = None

    def train(self, train_data, train_labels, val_data=None, val_labels=None):
        raise NotImplementedError

    def predict(self, signals):
        raise NotImplementedError

    def evaluate(self, test_data, test_labels):
        raise NotImplementedError


class CNNModulationRecognizer(ModulationRecognizerBase):
    """
    基于CNN的调制方式识别器

    使用卷积神经网络进行自动调制识别
    """

    def __init__(self, num_classes: int, config: Optional[dict] = None):
        super().__init__(num_classes, config)

        # 设备设置
        if HAS_TORCH:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"使用设备: {self.device}")
        else:
            self.device = None

        # 网络参数
        self.input_shape = self.config.get("input_shape", (128, 2))
        self.cnn_filters = self.config.get("cnn_filters", [64, 128, 256])
        self.kernel_size = self.config.get("kernel_size", 3)
        self.pool_size = self.config.get("pool_size", 2)
        self.dense_units = self.config.get("dense_units", 256)
        self.dropout_rate = self.config.get("dropout_rate", 0.4)

        # 训练参数
        self.batch_size = self.config.get("batch_size", 128)
        self.epochs = self.config.get("epochs", 50)
        self.learning_rate = self.config.get("learning_rate", 1e-3)

        # 调制类型列表
        self.modulation_types = ["BPSK", "QPSK", "16QAM", "64QAM", "256QAM"]

    def _build_torch_model(self) -> nn.Module:
        """构建PyTorch CNN模型"""

        class CNNModulationNet(nn.Module):
            def __init__(self, num_classes, cnn_filters, dense_units, dropout_rate):
                super().__init__()

                # 卷积层
                self.conv1 = nn.Conv1d(2, cnn_filters[0], kernel_size=7, padding=3)
                self.bn1 = nn.BatchNorm1d(cnn_filters[0])
                self.conv2 = nn.Conv1d(
                    cnn_filters[0], cnn_filters[1], kernel_size=5, padding=2
                )
                self.bn2 = nn.BatchNorm1d(cnn_filters[1])
                self.conv3 = nn.Conv1d(
                    cnn_filters[1], cnn_filters[2], kernel_size=3, padding=1
                )
                self.bn3 = nn.BatchNorm1d(cnn_filters[2])

                self.pool = nn.MaxPool1d(2)
                self.adaptive_pool = nn.AdaptiveAvgPool1d(32)

                # 全连接层
                self.fc1 = nn.Linear(cnn_filters[2] * 32, dense_units)
                self.fc2 = nn.Linear(dense_units, dense_units // 2)
                self.fc3 = nn.Linear(dense_units // 2, num_classes)

                self.dropout = nn.Dropout(dropout_rate)

            def forward(self, x):
                # x: (batch, seq_len, 2) -> (batch, 2, seq_len)
                x = x.transpose(1, 2)

                # 卷积块1
                x = self.pool(F.relu(self.bn1(self.conv1(x))))
                # 卷积块2
                x = self.pool(F.relu(self.bn2(self.conv2(x))))
                # 卷积块3
                x = self.pool(F.relu(self.bn3(self.conv3(x))))

                # 自适应池化
                x = self.adaptive_pool(x)

                # 展平
                x = x.view(x.size(0), -1)

                # 全连接
                x = self.dropout(x)
                x = F.relu(self.fc1(x))
                x = self.dropout(x)
                x = F.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.fc3(x)

                return x

        return CNNModulationNet(
            self.num_classes, self.cnn_filters, self.dense_units, self.dropout_rate
        )

    def train(
        self,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        val_data: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None,
    ):
        """
        训练调制识别器

        Args:
            train_data: 训练数据 (num_samples, seq_len, 2)
            train_labels: 训练标签 (num_samples, num_classes)
            val_data: 验证数据
            val_labels: 验证标签
        """
        print("=" * 60)
        print("CNN调制方式识别器训练")
        print("=" * 60)
        print(f"训练数据形状: {train_data.shape}")
        print(f"训练标签形状: {train_labels.shape}")
        print(f"类别数量: {self.num_classes}")

        self.history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

        if HAS_TORCH:
            # PyTorch训练
            self.model = self._build_torch_model().to(self.device)

            train_dataset = TensorDataset(
                torch.FloatTensor(train_data), torch.FloatTensor(train_labels)
            )
            train_loader = DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True
            )

            if val_data is not None:
                val_dataset = TensorDataset(
                    torch.FloatTensor(val_data), torch.FloatTensor(val_labels)
                )
                val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5
            )

            best_val_loss = float("inf")

            for epoch in range(self.epochs):
                # 训练阶段
                self.model.train()
                train_loss = 0
                train_correct = 0
                train_total = 0

                pbar = tqdm(
                    train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}", leave=True
                )
                for batch_data, batch_labels in pbar:
                    batch_data = batch_data.to(self.device)
                    batch_labels = batch_labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(batch_data)
                    loss = criterion(outputs, torch.argmax(batch_labels, dim=1))
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    train_total += batch_labels.size(0)
                    train_correct += (
                        predicted.eq(torch.argmax(batch_labels, dim=1)).sum().item()
                    )

                    batch_acc = train_correct / train_total
                    pbar.set_postfix(
                        {"loss": f"{loss.item():.4f}", "acc": f"{batch_acc:.4f}"}
                    )

                avg_train_loss = train_loss / len(train_loader)
                train_acc = train_correct / train_total

                # 验证阶段
                if val_data is not None:
                    self.model.eval()
                    val_loss = 0
                    val_correct = 0
                    val_total = 0

                    with torch.no_grad():
                        for batch_data, batch_labels in val_loader:
                            batch_data = batch_data.to(self.device)
                            batch_labels = batch_labels.to(self.device)

                            outputs = self.model(batch_data)
                            loss = criterion(outputs, torch.argmax(batch_labels, dim=1))

                            val_loss += loss.item()
                            _, predicted = outputs.max(1)
                            val_total += batch_labels.size(0)
                            val_correct += (
                                predicted.eq(torch.argmax(batch_labels, dim=1))
                                .sum()
                                .item()
                            )

                    avg_val_loss = val_loss / len(val_loader)
                    val_acc = val_correct / val_total

                    scheduler.step(avg_val_loss)

                    self.history["loss"].append(avg_train_loss)
                    self.history["accuracy"].append(train_acc)
                    self.history["val_loss"].append(avg_val_loss)
                    self.history["val_accuracy"].append(val_acc)

                    if (epoch + 1) % 5 == 0:
                        print(
                            f"Epoch {epoch + 1:3d}/{self.epochs} | "
                            f"Loss: {avg_train_loss:.4f} | Acc: {train_acc:.4f} | "
                            f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}"
                        )

                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        self.best_weights = {
                            k: v.cpu().clone()
                            for k, v in self.model.state_dict().items()
                        }
                else:
                    self.history["loss"].append(avg_train_loss)
                    self.history["accuracy"].append(train_acc)

                    if (epoch + 1) % 5 == 0:
                        print(
                            f"Epoch {epoch + 1:3d}/{self.epochs} | Loss: {avg_train_loss:.4f} | Acc: {train_acc:.4f}"
                        )

            # 恢复最佳权重
            if hasattr(self, "best_weights"):
                self.model.load_state_dict(self.best_weights)

        else:
            print("警告: 未安装PyTorch，使用统计学习方法")
            self._train_statistical(train_data, train_labels)

        print("\n训练完成！")

    def _train_statistical(self, train_data: np.ndarray, train_labels: np.ndarray):
        """使用统计学习方法(无深度学习框架时)"""
        # 提取统计特征
        features = self._extract_features(train_data)

        # 简单分类器(最近邻)
        self.stat_model = {
            "features": features,
            "labels": np.argmax(train_labels, axis=1),
        }

    def _extract_features(self, data: np.ndarray) -> np.ndarray:
        """提取统计特征"""
        signal_complex = data[:, :, 0] + 1j * data[:, :, 1]
        features = []

        for sig in signal_complex:
            # 基本统计量
            amp = np.abs(sig)
            phase = np.angle(sig)

            feat = [
                np.mean(amp),
                np.std(amp),
                np.max(amp) - np.min(amp),
                np.mean(phase),
                np.std(phase),
                np.percentile(amp, 25),
                np.percentile(amp, 75),
                np.mean(np.abs(np.diff(sig))),
            ]

            # 高阶统计量
            feat.extend(
                [
                    np.mean((amp - np.mean(amp)) ** 3)
                    / (np.std(amp) ** 3 + 1e-10),  # 偏度
                    np.mean((amp - np.mean(amp)) ** 4)
                    / (np.std(amp) ** 4 + 1e-10),  # 峰度
                ]
            )

            features.append(feat)

        return np.array(features)

    def predict(self, signals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测调制方式

        Args:
            signals: 输入信号 (num_samples, seq_len, 2)

        Returns:
            (预测概率, 预测类别索引)
        """
        if HAS_TORCH and isinstance(self.model, nn.Module):
            self.model.eval()
            with torch.no_grad():
                inputs = torch.FloatTensor(signals).to(self.device)
                outputs = self.model(inputs)
                probs = F.softmax(outputs, dim=1).cpu().numpy()
                predictions = np.argmax(probs, axis=1)
            return probs, predictions

        else:
            # 统计方法
            features = self._extract_features(signals)
            predictions = self._knn_predict(features)
            probs = np.zeros((len(predictions), self.num_classes))
            for i, p in enumerate(predictions):
                probs[i, p] = 1.0
            return probs, predictions

    def _knn_predict(self, features: np.ndarray, k: int = 5) -> np.ndarray:
        """K近邻预测"""
        train_features = self.stat_model["features"]
        train_labels = self.stat_model["labels"]

        predictions = []
        for feat in features:
            distances = np.linalg.norm(train_features - feat, axis=1)
            nearest_indices = np.argsort(distances)[:k]
            nearest_labels = train_labels[nearest_indices]

            # 投票
            votes = np.bincount(nearest_labels, minlength=self.num_classes)
            predictions.append(np.argmax(votes))

        return np.array(predictions)

    def evaluate(
        self,
        test_data: np.ndarray,
        test_labels: np.ndarray,
        by_snr: bool = False,
        snr_values: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        评估识别性能

        Args:
            test_data: 测试数据
            test_labels: 测试标签
            by_snr: 是否按SNR分组评估
            snr_values: SNR值数组(如果by_snr为True)

        Returns:
            评估结果字典
        """
        probs, predictions = self.predict(test_data)
        true_labels = np.argmax(test_labels, axis=1)

        # 总体准确率
        overall_acc = np.mean(predictions == true_labels)

        # 各类别准确率
        class_acc = {}
        for i, mod_type in enumerate(self.modulation_types[: self.num_classes]):
            mask = true_labels == i
            if np.sum(mask) > 0:
                class_acc[mod_type] = np.mean(predictions[mask] == true_labels[mask])

        results = {
            "overall_accuracy": overall_acc,
            "class_accuracy": class_acc,
            "predictions": predictions,
            "true_labels": true_labels,
            "probabilities": probs,
        }

        # 按SNR分组
        if by_snr and snr_values is not None:
            snr_ranges = [(-10, 0), (0, 10), (10, 20), (20, 30)]
            snr_results = {}

            for snr_min, snr_max in snr_ranges:
                mask = (snr_values >= snr_min) & (snr_values < snr_max)
                if np.sum(mask) > 0:
                    snr_acc = np.mean(predictions[mask] == true_labels[mask])
                    snr_results[f"{snr_min}_{snr_max}dB"] = snr_acc

            results["accuracy_by_snr"] = snr_results

        return results


def plot_confusion_matrix(self, results: Dict, save_path: Optional[str] = None):
    """绘制混淆矩阵"""
    if not HAS_MATPLOTLIB:
        print("警告: matplotlib未安装，无法绘制混淆矩阵")
        return

    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(results["true_labels"], results["predictions"])
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=self.modulation_types[: self.num_classes],
        yticklabels=self.modulation_types[: self.num_classes],
        ax=ax,
    )
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title(f"调制方式识别混淆矩阵 (准确率: {results['overall_accuracy']:.2%})")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"混淆矩阵已保存至: {save_path}")

    plt.close()


def plot_accuracy_by_snr(self, results: Dict, save_path: Optional[str] = None):
    """绘制各SNR下的准确率"""
    if not HAS_MATPLOTLIB:
        print("警告: matplotlib未安装，无法绘制SNR性能图")
        return

    if "accuracy_by_snr" not in results:
        print("警告: 没有按SNR分类的结果")
        return

    snr_ranges = list(results["accuracy_by_snr"].keys())
    accuracies = list(results["accuracy_by_snr"].values())

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        snr_ranges, accuracies, color="#2196F3", edgecolor="white", linewidth=2
    )

    # 添加数值标签
    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{acc:.1%}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_xlabel("SNR范围 (dB)")
    ax.set_ylabel("识别准确率")
    ax.set_title("不同SNR条件下的调制识别准确率")
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")

    # 添加基准线
    ax.axhline(y=0.8, color="red", linestyle="--", linewidth=1, label="80%基准")
    ax.legend()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"SNR性能图已保存至: {save_path}")

    plt.close()


class RecognitionTrainer:
    """
    调制识别训练器

    提供完整的训练和评估流程
    """

    def __init__(self, config: Optional[dict] = None):
        """初始化训练器"""
        self.config = config or {}

        # 信号生成器
        self.signal_generator = SignalGenerator(config)

        # 识别器
        num_classes = len(self.signal_generator.modulation_types)
        self.recognizer = CNNModulationRecognizer(num_classes, config)

        # 结果目录
        self.results_dir = Path("signal_detection/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def train(self, num_samples_per_class: int = 1000):
        """
        训练调制识别器

        Args:
            num_samples_per_class: 每类训练样本数
        """
        print("=" * 60)
        print("调制方式识别模型训练")
        print("=" * 60)

        # 生成训练数据
        print("\n生成训练数据...")
        train_data, train_labels = self.signal_generator.generate_dataset(
            num_samples_per_class=num_samples_per_class
        )

        # 划分训练集和验证集
        split_idx = int(len(train_data) * 0.8)
        val_data = train_data[split_idx:]
        val_labels = train_labels[split_idx:]
        train_data = train_data[:split_idx]
        train_labels = train_labels[:split_idx]

        print(f"\n训练集大小: {len(train_data)}")
        print(f"验证集大小: {len(val_data)}")

        # 训练
        self.recognizer.train(train_data, train_labels, val_data, val_labels)

        # 保存模型
        self._save_model()

        return train_data, train_labels

    def evaluate(
        self, num_samples_per_class: int = 500, generate_snr_values: bool = True
    ) -> Dict:
        """
        评估识别器性能

        Args:
            num_samples_per_class: 每类测试样本数
            generate_snr_values: 是否生成带SNR值的数据

        Returns:
            评估结果
        """
        print("\n" + "=" * 60)
        print("调制识别性能评估")
        print("=" * 60)

        # 生成测试数据
        print("\n生成测试数据...")
        test_data, test_labels = self.signal_generator.generate_dataset(
            num_samples_per_class=num_samples_per_class
        )

        # 生成对应的SNR值
        if generate_snr_values:
            snr_values = np.random.uniform(-10, 30, len(test_data))
        else:
            snr_values = None

        # 评估
        results = self.recognizer.evaluate(
            test_data, test_labels, by_snr=True, snr_values=snr_values
        )

        # 打印结果
        print(f"\n总体识别准确率: {results['overall_accuracy']:.2%}")

        print("\n各类别准确率:")
        print("-" * 40)
        for mod_type, acc in results["class_accuracy"].items():
            print(f"  {mod_type:10s}: {acc:.2%}")

        if "accuracy_by_snr" in results:
            print("\n各SNR区间准确率:")
            print("-" * 40)
            for snr_range, acc in results["accuracy_by_snr"].items():
                print(f"  {snr_range:10s}: {acc:.2%}")

        # 绘制图表
        print("\n生成可视化图表...")
        self.recognizer.plot_confusion_matrix(
            results, self.results_dir / "confusion_matrix.png"
        )
        self.recognizer.plot_accuracy_by_snr(
            results, self.results_dir / "accuracy_by_snr.png"
        )

        return results

    def _save_model(self):
        """保存模型"""
        if HAS_TORCH and isinstance(self.recognizer.model, nn.Module):
            torch.save(
                {
                    "model_state_dict": self.recognizer.model.state_dict(),
                    "config": self.recognizer.config,
                    "history": self.recognizer.history,
                    "modulation_types": self.recognizer.modulation_types,
                },
                self.results_dir / "modulation_recognizer.pth",
            )
            print(f"\n模型已保存至: {self.results_dir / 'modulation_recognizer.pth'}")

    def run_full_training(self):
        """运行完整训练流程"""
        self.train(num_samples_per_class=1000)
        results = self.evaluate(num_samples_per_class=500)
        return results


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="调制方式识别训练")
    parser.add_argument("--samples", type=int, default=1000, help="每类样本数")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    args = parser.parse_args()

    config = {"batch_size": 128, "epochs": args.epochs}

    trainer = RecognitionTrainer(config)
    results = trainer.run_full_training()

    return results


if __name__ == "__main__":
    results = main()

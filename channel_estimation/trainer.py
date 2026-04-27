"""
AI信道估计模块 - 训练器
实现信道估计模型的训练、验证和评估功能
使用 PyTorch 模型，支持 GPU 加速
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, List
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .data_generator import ChannelDataGenerator
from .models import TransformerChannelEstimator


class ChannelEstimationTrainer:
    """
    信道估计模型训练器

    提供完整的训练流程，包括：
    - 数据生成与预处理
    - 模型训练与验证
    - 性能评估与可视化
    - 模型保存与加载
    """

    def __init__(self, config: Optional[dict] = None):
        """
        初始化训练器

        Args:
            config: 配置字典
        """
        self.config = config or {}

        self.model_type = self.config.get('model_type', 'transformer')

        self.batch_size = self.config.get('batch_size', 64)
        self.epochs = self.config.get('epochs', 100)
        self.learning_rate = self.config.get('learning_rate', 1e-4)

        self.num_train_samples = self.config.get('num_train_samples', 8000)
        self.num_val_samples = self.config.get('num_val_samples', 2000)
        self.snr_range = self.config.get('snr_range', (-10, 30))

        self.channel_model = self.config.get('channel_model', '5G_UMa')

        self.data_generator = ChannelDataGenerator(self.config)

        self.model = None
        self.optimizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"训练设备: {self.device}")

        self.train_history = {
            'loss': [],
            'val_loss': [],
            'mse': [],
            'val_mse': []
        }

        self.results_dir = Path('channel_estimation/results')
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.best_state_dict = None

    def prepare_data(self) -> Tuple[Dict, Dict]:
        """
        准备训练和验证数据

        Returns:
            (训练数据字典, 验证数据字典)
        """
        print("正在生成训练数据...")

        train_dataset, train_labels = self.data_generator.generate_training_dataset(
            num_samples=self.num_train_samples,
            model=self.channel_model,
            snr_range=self.snr_range
        )

        val_dataset, val_labels = self.data_generator.generate_training_dataset(
            num_samples=self.num_val_samples,
            model=self.channel_model,
            snr_range=self.snr_range,
            seed=123
        )

        train_baselines = self.data_generator.compute_baseline_estimates(
            train_dataset['received_signal'],
            train_dataset['transmitted_pilot']
        )
        val_baselines = self.data_generator.compute_baseline_estimates(
            val_dataset['received_signal'],
            val_dataset['transmitted_pilot']
        )

        train_data = {
            'received': train_dataset['received_signal'],
            'pilot': train_dataset['transmitted_pilot'],
            'labels': train_labels,
            'snr': train_dataset['snr_db'],
            'baselines': train_baselines
        }

        val_data = {
            'received': val_dataset['received_signal'],
            'pilot': val_dataset['transmitted_pilot'],
            'labels': val_labels,
            'snr': val_dataset['snr_db'],
            'baselines': val_baselines
        }

        # 准备PyTorch张量
        train_data['received_tensor'] = torch.from_numpy(train_data['received']).float()
        train_data['pilot_tensor'] = torch.from_numpy(train_data['pilot']).float()
        train_data['labels_tensor'] = torch.from_numpy(train_data['labels']).float()
        val_data['received_tensor'] = torch.from_numpy(val_data['received']).float()
        val_data['pilot_tensor'] = torch.from_numpy(val_data['pilot']).float()
        val_data['labels_tensor'] = torch.from_numpy(val_data['labels']).float()

        print(f"训练样本数: {len(train_labels)}")
        print(f"验证样本数: {len(val_labels)}")

        return train_data, val_data

    def compute_mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        计算均方误差

        Args:
            y_true: 真实值
            y_pred: 预测值

        Returns:
            MSE值
        """
        return np.mean(np.abs(y_true - y_pred) ** 2)

    def compute_nmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        计算归一化均方误差

        Args:
            y_true: 真实值
            y_pred: 预测值

        Returns:
            NMSE值(dB)
        """
        signal_power = np.mean(np.abs(y_true) ** 2)
        error_power = np.mean(np.abs(y_true - y_pred) ** 2)
        nmse_db = 10 * np.log10(error_power / signal_power)
        return nmse_db

    def train_pytorch_model(self, train_data: dict, val_data: dict):
        """
        训练PyTorch实现的模型

        Args:
            train_data: 训练数据字典
            val_data: 验证数据字典
        """
        print(f"\n初始化{self.model_type}模型...")

        if self.model_type == 'transformer':
            model_config = {
                'embedding_dim': self.config.get('embedding_dim', 128),
                'num_heads': self.config.get('num_heads', 8),
                'num_layers': self.config.get('num_layers', 4),
                'dropout_rate': self.config.get('dropout_rate', 0.1)
            }
            estimator = TransformerChannelEstimator(model_config)
            self.model = estimator.model.to(self.device)
        elif self.model_type == 'cnn':
            from .models import CNNChannelEstimator
            input_dim = train_data['received'].shape[1]
            estimator = CNNChannelEstimator(input_dim=input_dim)
            self.model = estimator.model.to(self.device)
        elif self.model_type == 'hybrid':
            from .models import HybridChannelEstimator
            estimator = HybridChannelEstimator()
            self.model = estimator.model.to(self.device)
        else:
            raise ValueError(f"未知的PyTorch模型类型: {self.model_type}")

        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"模型参数数量: {num_params}")

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        X_train = self._prepare_pytorch_input(train_data)
        y_train = self._prepare_pytorch_target(train_data)
        X_val = self._prepare_pytorch_input(val_data)
        y_val = self._prepare_pytorch_target(val_data)

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        print(f"\n开始PyTorch训练，共{self.epochs}个epoch...")

        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0

        for epoch in range(self.epochs):
            start_time = time.time()

            self.model.train()
            epoch_loss = 0
            num_batches = 0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            train_loss = epoch_loss / num_batches

            self.model.eval()
            with torch.no_grad():
                X_val_dev = X_val.to(self.device)
                y_val_dev = y_val.to(self.device)
                val_output = self.model(X_val_dev)
                val_loss = criterion(val_output, y_val_dev).item()

                train_output = self.model(X_train.to(self.device))
                train_mse = self.compute_mse(
                    y_train.cpu().numpy(),
                    train_output.cpu().numpy()
                )
                val_mse = self.compute_mse(
                    y_val.cpu().numpy(),
                    val_output.cpu().numpy()
                )

            self.train_history['loss'].append(train_loss)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['mse'].append(train_mse)
            self.train_history['val_mse'].append(val_mse)

            epoch_time = time.time() - start_time

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{self.epochs} | "
                      f"Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                      f"MSE: {train_mse:.4f} | Time: {epoch_time:.2f}s")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.best_state_dict = self.model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n早停触发！连续{patience}个epoch未改善")
                    break

        if hasattr(self, 'best_state_dict'):
            self.model.load_state_dict(self.best_state_dict)

        print(f"\n训练完成！最佳验证损失: {best_val_loss:.6f}")

    def _prepare_pytorch_input(self, data: dict) -> torch.Tensor:
        """准备PyTorch输入张量"""
        received = data['received']
        pilot = data['pilot']

        received_real = received.real
        received_imag = received.imag

        if self.model_type == 'transformer':
            x = np.stack([received_real, received_imag], axis=-1)
            return torch.from_numpy(x).float()
        elif self.model_type in ['cnn', 'hybrid']:
            x = np.stack([received_real, received_imag], axis=1)
            return torch.from_numpy(x).float()
        else:
            features = np.concatenate([
                received_real, received_imag,
                np.abs(received), np.angle(received),
                pilot.real, pilot.imag
            ], axis=1)
            return torch.from_numpy(features).float()

    def _prepare_pytorch_target(self, data: dict) -> torch.Tensor:
        """准备PyTorch目标张量"""
        labels = data['labels']
        labels_real = labels.real
        labels_imag = labels.imag

        if self.model_type == 'transformer':
            y = np.stack([labels_real, labels_imag], axis=-1)
            return torch.from_numpy(y).float()
        elif self.model_type in ['cnn', 'hybrid']:
            y = np.stack([labels_real, labels_imag], axis=1)
            return torch.from_numpy(y).float()
        else:
            y = np.concatenate([labels_real, labels_imag], axis=1)
            return torch.from_numpy(y).float()

    def evaluate(self, data: dict) -> dict:
        """
        评估模型性能

        Args:
            data: 测试数据字典

        Returns:
            性能指标字典
        """
        if self.model is None:
            raise ValueError("模型未初始化")

        X = self._prepare_pytorch_input(data)
        self.model.eval()
        with torch.no_grad():
            X_dev = X.to(self.device)
            ai_predictions = self.model(X_dev).cpu().numpy()
        y_true = np.concatenate([data['labels'].real, data['labels'].imag], axis=1)

        if self.model_type in ['transformer', 'hybrid']:
            ai_pred_complex = ai_predictions[:, :, 0] + 1j * ai_predictions[:, :, 1]
        elif self.model_type == 'cnn':
            ai_pred_complex = ai_predictions[:, 0, :] + 1j * ai_predictions[:, 1, :]
        else:
            labels_shape = data['labels'].shape[1]
            ai_pred_complex = ai_predictions[:, :labels_shape] + 1j * ai_predictions[:, labels_shape:]

        ai_nmse = self.compute_nmse(data['labels'], ai_pred_complex)

        baseline_nmses = {}
        for name, estimate in data.get('baselines', {}).items():
            baseline_nmses[name] = self.compute_nmse(data['labels'], estimate)

        results = {
            'AI-ChannelNet': ai_nmse,
            **baseline_nmses
        }

        results = dict(sorted(results.items(), key=lambda x: x[1]))

        return results

    def evaluate_by_snr(self, data: dict, snr_ranges: List[Tuple[float, float]] = None) -> dict:
        """
        按SNR区间评估模型性能

        Args:
            data: 测试数据字典
            snr_ranges: SNR区间列表

        Returns:
            各SNR区间的性能字典
        """
        if snr_ranges is None:
            snr_ranges = [(-10, 0), (0, 10), (10, 20), (20, 30)]

        snr_results = {}

        for snr_min, snr_max in snr_ranges:
            mask = (data['snr'] >= snr_min) & (data['snr'] < snr_max)
            if np.sum(mask) == 0:
                continue

            snr_data = {
                'received': data['received'][mask],
                'pilot': data['pilot'][mask],
                'labels': data['labels'][mask],
                'snr': data['snr'][mask],
                'baselines': {
                    name: est[mask]
                    for name, est in data.get('baselines', {}).items()
                }
            }

            snr_results[f'{snr_min}_{snr_max}dB'] = self.evaluate(snr_data)

        return snr_results

    def plot_training_history(self):
        """绘制训练历史曲线"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 损失曲线
        axes[0].plot(self.train_history['loss'], label='训练损失')
        axes[0].plot(self.train_history['val_loss'], label='验证损失')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('训练损失曲线')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # MSE曲线
        axes[1].plot(self.train_history['mse'], label='训练MSE')
        axes[1].plot(self.train_history['val_mse'], label='验证MSE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MSE')
        axes[1].set_title('均方误差曲线')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.results_dir / 'training_history.png', dpi=150)
        plt.close()
        print(f"\n训练历史曲线已保存至: {self.results_dir / 'training_history.png'}")

    def plot_performance_comparison(self, test_data: dict):
        """
        绘制性能对比图

        Args:
            test_data: 测试数据字典
        """
        results = self.evaluate_by_snr(test_data)

        # 提取各SNR区间的NMSE
        snr_labels = list(results.keys())
        algorithms = ['AI-ChannelNet', 'LS', 'MMSE', 'LMMSE']
        colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0']

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(snr_labels))
        width = 0.2

        for i, algo in enumerate(algorithms):
            values = [results[snr].get(algo, float('inf')) for snr in snr_labels]
            bars = ax.bar(x + i * width, values, width, label=algo, color=colors[i])

            # 添加数值标签
            for bar, val in zip(bars, values):
                if val < float('inf'):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           f'{val:.1f}', ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('SNR范围 (dB)')
        ax.set_ylabel('NMSE (dB)')
        ax.set_title('AI-ChannelNet与传统算法性能对比')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(snr_labels)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(self.results_dir / 'performance_comparison.png', dpi=150)
        plt.close()
        print(f"性能对比图已保存至: {self.results_dir / 'performance_comparison.png'}")

    def plot_channel_visualization(self, data: dict, num_examples: int = 3):
        """
        绘制信道估计可视化对比

        Args:
            data: 测试数据字典
            num_examples: 示例数量
        """
        X = self._prepare_pytorch_input(data)
        y_true = data['labels']

        # AI预测
        self.model.eval()
        with torch.no_grad():
            X_dev = X.to(self.device)
            ai_pred = self.model(X_dev).cpu().numpy()

        if self.model_type in ['transformer', 'hybrid']:
            ai_pred_complex = ai_pred[:, :, 0] + 1j * ai_pred[:, :, 1]
        elif self.model_type == 'cnn':
            ai_pred_complex = ai_pred[:, 0, :] + 1j * ai_pred[:, 1, :]
        else:
            labels_shape = y_true.shape[1]
            ai_pred_complex = ai_pred[:, :labels_shape] + 1j * ai_pred[:, labels_shape:]

        # LS基准
        ls_pred = data.get('baselines', {}).get('LS', data['labels'])  # 如果没有LS，使用真实值作为fallback

        num_examples = min(num_examples, len(y_true))

        fig, axes = plt.subplots(num_examples, 3, figsize=(15, 4 * num_examples))

        if num_examples == 1:
            axes = axes.reshape(1, -1)

        for i in range(num_examples):
            # 真实信道
            axes[i, 0].stem(np.abs(y_true[i]), linefmt='b-', markerfmt='bo', basefmt='r-')
            axes[i, 0].set_title(f'示例{i+1}: 真实信道')
            axes[i, 0].set_ylabel('幅度')
            axes[i, 0].grid(True, alpha=0.3)

            # AI估计
            axes[i, 1].stem(np.abs(ai_pred_complex[i]), linefmt='g-', markerfmt='go', basefmt='r-')
            axes[i, 1].set_title('AI-ChannelNet估计')
            axes[i, 1].grid(True, alpha=0.3)

            # LS估计
            axes[i, 2].stem(np.abs(ls_pred[i]), linefmt='r-', markerfmt='ro', basefmt='r-')
            axes[i, 2].set_title('LS估计')
            axes[i, 2].grid(True, alpha=0.3)

            if i == 0:
                axes[i, 0].set_title(f'示例{i+1}: 真实信道')
                axes[i, 1].set_title('AI-ChannelNet估计')
                axes[i, 2].set_title('LS估计')

        plt.tight_layout()
        plt.savefig(self.results_dir / 'channel_visualization.png', dpi=150)
        plt.close()
        print(f"信道可视化图已保存至: {self.results_dir / 'channel_visualization.png'}")

    def save_model(self, filepath: str = 'channel_estimation_model'):
        """
        保存模型

        Args:
            filepath: 保存路径（不含扩展名）
        """
        if self.model is None:
            raise ValueError("模型未初始化")

        torch_path = str(self.results_dir / f"{filepath}.pth")
        torch.save({
            'state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'config': self.config,
            'train_history': self.train_history
        }, torch_path)

        print(f"模型已保存至: {torch_path}")

    def load_model(self, filepath: str = 'channel_estimation_model.pth'):
        """
        加载模型

        Args:
            filepath: 模型路径
        """
        torch_path = str(self.results_dir / filepath) if not str(filepath).endswith('.pth') else str(filepath)
        self._load_pytorch_model(torch_path)
        print(f"模型已加载: {torch_path}")

    def _load_pytorch_model(self, filepath: str):
        """加载PyTorch模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model_type = checkpoint['model_type']
        self.config = checkpoint.get('config', {})

        if self.model_type == 'transformer':
            estimator = TransformerChannelEstimator(self.config)
            self.model = estimator.model.to(self.device)
        elif self.model_type == 'cnn':
            from .models import CNNChannelEstimator
            estimator = CNNChannelEstimator(input_dim=self.config.get('input_dim', 150))
            self.model = estimator.model.to(self.device)
        elif self.model_type == 'hybrid':
            from .models import HybridChannelEstimator
            estimator = HybridChannelEstimator()
            self.model = estimator.model.to(self.device)

        self.model.load_state_dict(checkpoint['state_dict'])
        self.train_history = checkpoint.get('train_history', self.train_history)

    def run_full_training(self):
        """运行完整训练流程"""
        print("=" * 60)
        print(f"AI-ChannelNet 信道估计模型训练 ({self.model_type})")
        print("=" * 60)

        train_data, val_data = self.prepare_data()

        self.train_pytorch_model(train_data, val_data)

        print("\n生成测试数据...")
        test_dataset, test_labels = self.data_generator.generate_training_dataset(
            num_samples=1000,
            model=self.channel_model,
            snr_range=self.snr_range,
            seed=456
        )
        test_baselines = self.data_generator.compute_baseline_estimates(
            test_dataset['received_signal'],
            test_dataset['transmitted_pilot']
        )
        test_data = {
            'received': test_dataset['received_signal'],
            'pilot': test_dataset['transmitted_pilot'],
            'labels': test_labels,
            'snr': test_dataset['snr_db'],
            'baselines': test_baselines
        }

        print("\n评估模型性能...")
        overall_results = self.evaluate(test_data)

        print("\n整体性能对比 (NMSE, dB):")
        print("-" * 40)
        for algo, nmse in overall_results.items():
            print(f"{algo:15s}: {nmse:8.2f} dB")

        snr_results = self.evaluate_by_snr(test_data)
        print("\n各SNR区间性能对比:")
        print("-" * 60)
        for snr_range, results in snr_results.items():
            print(f"\n{snr_range}:")
            for algo, nmse in results.items():
                print(f"  {algo:15s}: {nmse:8.2f} dB")

        print("\n生成可视化图表...")
        self.plot_training_history()
        self.plot_performance_comparison(test_data)
        self.plot_channel_visualization(test_data)

        self.save_model()

        print("\n" + "=" * 60)
        print("训练完成！")
        print("=" * 60)

        return overall_results


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='AI信道估计模型训练')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批量大小')
    parser.add_argument('--num_samples', type=int, default=8000, help='训练样本数')
    parser.add_argument('--snr_min', type=float, default=-10, help='最小SNR(dB)')
    parser.add_argument('--snr_max', type=float, default=30, help='最大SNR(dB)')
    parser.add_argument('--model_type', type=str, default='transformer',
                        choices=['transformer', 'cnn', 'hybrid'],
                        help='模型类型: transformer, cnn, hybrid')
    args = parser.parse_args()

    config = {
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'num_train_samples': args.num_samples,
        'num_val_samples': args.num_samples // 4,
        'snr_range': (args.snr_min, args.snr_max),
        'model_type': args.model_type
    }

    trainer = ChannelEstimationTrainer(config)
    results = trainer.run_full_training()

    return results


if __name__ == '__main__':
    results = main()

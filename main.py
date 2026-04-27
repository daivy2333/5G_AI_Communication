"""
5G无线通信链路+AI创新设计项目 - 统一入口程序

支持多种运行模式：
- full: 运行完整链路仿真
- channel_estimation: 只运行信道估计
- signal_detection: 只运行信号检测
- resource_scheduling: 只运行资源调度
- demo: 运行演示模式（快速验证所有模块）

用法示例：
    python main.py --mode full
    python main.py --mode demo --num_samples 100
    python main.py --mode channel_estimation --snr 20 --epochs 50
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
import json

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

# 导入配置
from config import (
    SystemConfig, ChannelEstimationConfig, SignalDetectionConfig,
    ResourceSchedulingConfig, get_config
)

# 尝试导入各模块
try:
    from channel_estimation.trainer import ChannelEstimationTrainer
    HAS_CHANNEL_ESTIMATION = True
except ImportError as e:
    print(f"警告: 信道估计模块导入失败 - {e}")
    HAS_CHANNEL_ESTIMATION = False

try:
    from signal_detection.detector import DetectorTrainer, SignalDetector
    HAS_SIGNAL_DETECTION = True
except ImportError as e:
    print(f"警告: 信号检测模块导入失败 - {e}")
    HAS_SIGNAL_DETECTION = False

try:
    from resource_scheduling.agent import create_scheduler, train_and_compare
    HAS_RESOURCE_SCHEDULING = True
except ImportError as e:
    print(f"警告: 资源调度模块导入失败 - {e}")
    HAS_RESOURCE_SCHEDULING = False


def setup_results_dir():
    """创建结果目录"""
    results_dir = Path('results')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建子目录
    (results_dir / 'channel_estimation').mkdir(exist_ok=True)
    (results_dir / 'signal_detection').mkdir(exist_ok=True)
    (results_dir / 'resource_scheduling').mkdir(exist_ok=True)
    
    return results_dir


def check_gpu():
    """检查GPU是否可用"""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"检测到GPU: {device_name}")
            return True
        else:
            print("未检测到GPU，将使用CPU运行")
            return False
    except ImportError:
        print("PyTorch未安装，无法检测GPU")
        return False


def run_channel_estimation_demo(num_samples: int, epochs: int, snr: float, 
                                 visualize: bool, use_gpu: bool):
    """
    运行信道估计演示
    
    Args:
        num_samples: 样本数量
        epochs: 训练轮数
        snr: 信噪比
        visualize: 是否生成可视化
        use_gpu: 是否使用GPU
    """
    print("\n" + "=" * 60)
    print("【模块一】AI信道估计演示")
    print("=" * 60)
    
    if not HAS_CHANNEL_ESTIMATION:
        print("错误: 信道估计模块未正确安装")
        return None
    
    # 配置参数（演示模式使用少量样本）
    config = {
        'num_train_samples': num_samples,
        'num_val_samples': num_samples // 4,
        'epochs': min(epochs, 20),  # 演示模式限制epoch数
        'batch_size': 32,
        'snr_range': (snr - 10, snr + 10),
        'learning_rate': 1e-4,
        'channel_model': '5G_UMa'
    }
    
    print(f"训练样本数: {num_samples}")
    print(f"训练轮数: {config['epochs']}")
    print(f"SNR范围: {config['snr_range']} dB")
    
    # 创建训练器
    trainer = ChannelEstimationTrainer(config)
    
    try:
        # 运行训练
        results = trainer.run_full_training()
        
        print("\n信道估计性能结果:")
        print("-" * 40)
        for algo, nmse in results.items():
            improvement = nmse - results.get('LS', 0)
            print(f"{algo:15s}: NMSE = {nmse:.2f} dB")
        
        if visualize:
            trainer.plot_training_history()
            trainer.plot_performance_comparison(trainer.test_data)
        
        return results
    except Exception as e:
        print(f"训练过程中出错: {e}")
        return None


def run_signal_detection_demo(num_samples: int, visualize: bool, use_gpu: bool):
    """
    运行信号检测演示
    
    Args:
        num_samples: 样本数量
        visualize: 是否生成可视化
        use_gpu: 是否使用GPU
    """
    print("\n" + "=" * 60)
    print("【模块二】信号检测与调制识别演示")
    print("=" * 60)
    
    if not HAS_SIGNAL_DETECTION:
        print("错误: 信号检测模块未正确安装")
        return None
    
    config = {
        'epochs': 10,  # 演示模式少量训练
        'batch_size': 64,
        'learning_rate': 1e-3
    }
    
    print(f"样本数量: {num_samples}")
    
    # 创建训练器
    trainer = DetectorTrainer(config)
    
    try:
        # 训练
        trainer.train(num_samples=num_samples)
        
        # 评估
        results = trainer.evaluate(num_samples=num_samples // 2)
        
        print("\n信号检测性能结果:")
        print("-" * 40)
        print(f"准确率: {results['accuracy']:.4f}")
        print(f"精确率: {results['precision']:.4f}")
        print(f"召回率: {results['recall']:.4f}")
        print(f"F1分数: {results['f1_score']:.4f}")
        
        if visualize:
            # 保存混淆矩阵信息
            results_dir = Path('results/signal_detection')
            cm_file = results_dir / 'confusion_matrix.json'
            with open(cm_file, 'w') as f:
                json.dump(results['confusion_matrix'], f, indent=2)
            print(f"混淆矩阵已保存至: {cm_file}")
        
        return results
    except Exception as e:
        print(f"训练过程中出错: {e}")
        return None


def run_resource_scheduling_demo(num_timesteps: int, visualize: bool, use_gpu: bool):
    """
    运行资源调度演示
    
    Args:
        num_timesteps: 训练步数
        visualize: 是否生成可视化
        use_gpu: 是否使用GPU
    """
    print("\n" + "=" * 60)
    print("【模块三】智能资源调度演示")
    print("=" * 60)
    
    if not HAS_RESOURCE_SCHEDULING:
        print("错误: 资源调度模块未正确安装")
        return None
    
    env_config = {
        'num_users': 10,
        'num_resource_blocks': 50,
        'time_slots': 10
    }
    
    agent_config = {
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'batch_size': 64
    }
    
    print(f"用户数: {env_config['num_users']}")
    print(f"资源块数: {env_config['num_resource_blocks']}")
    print(f"训练步数: {num_timesteps}")
    
    try:
        # 使用PPO调度器（更稳定）
        scheduler = create_scheduler('ppo', env_config, agent_config)
        
        # 训练
        scheduler.train(num_timesteps, log_interval=50)
        
        # 保存模型
        results_dir = Path('results/resource_scheduling')
        scheduler.save(str(results_dir / 'ppo_scheduler'))
        
        # 统计结果
        stats = scheduler.training_stats
        avg_reward = np.mean(stats['episode_rewards']) if stats['episode_rewards'] else 0
        avg_throughput = np.mean(stats['throughputs']) if stats['throughputs'] else 0
        
        results = {
            'algorithm': 'PPO',
            'avg_reward': avg_reward,
            'avg_throughput': avg_throughput,
            'episodes': len(stats['episode_rewards'])
        }
        
        print("\n资源调度性能结果:")
        print("-" * 40)
        print(f"算法: PPO")
        print(f"平均奖励: {avg_reward:.3f}")
        print(f"平均吞吐量: {avg_throughput:.2f}")
        
        if visualize:
            stats_file = results_dir / 'training_stats.json'
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"训练统计已保存至: {stats_file}")
        
        return results
    except Exception as e:
        print(f"训练过程中出错: {e}")
        return None


def generate_comparison_report(results_dict: dict, results_dir: Path):
    """
    生成性能对比报告
    
    Args:
        results_dict: 各模块结果字典
        results_dir: 结果目录
    """
    print("\n" + "=" * 60)
    print("【性能对比报告】")
    print("=" * 60)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'modules': {}
    }
    
    # 信道估计结果
    if results_dict.get('channel_estimation'):
        ce_results = results_dict['channel_estimation']
        report['modules']['channel_estimation'] = {
            'AI-ChannelNet NMSE': ce_results.get('AI-ChannelNet'),
            'LS NMSE': ce_results.get('LS'),
            'MMSE NMSE': ce_results.get('MMSE'),
            'improvement_vs_LS': ce_results.get('AI-ChannelNet') - ce_results.get('LS', 0)
        }
        
        print("\n信道估计性能:")
        print(f"  AI-ChannelNet: {ce_results.get('AI-ChannelNet', 'N/A')} dB")
        print(f"  LS基准: {ce_results.get('LS', 'N/A')} dB")
        if ce_results.get('AI-ChannelNet') and ce_results.get('LS'):
            improvement = ce_results['AI-ChannelNet'] - ce_results['LS']
            print(f"  相比LS提升: {improvement:.2f} dB")
    
    # 信号检测结果
    if results_dict.get('signal_detection'):
        sd_results = results_dict['signal_detection']
        report['modules']['signal_detection'] = {
            'accuracy': sd_results.get('accuracy'),
            'precision': sd_results.get('precision'),
            'recall': sd_results.get('recall'),
            'f1_score': sd_results.get('f1_score')
        }
        
        print("\n信号检测性能:")
        print(f"  准确率: {sd_results.get('accuracy', 'N/A'):.2%}")
        print(f"  F1分数: {sd_results.get('f1_score', 'N/A'):.2%}")
    
    # 资源调度结果
    if results_dict.get('resource_scheduling'):
        rs_results = results_dict['resource_scheduling']
        report['modules']['resource_scheduling'] = {
            'algorithm': rs_results.get('algorithm'),
            'avg_reward': rs_results.get('avg_reward'),
            'avg_throughput': rs_results.get('avg_throughput')
        }
        
        print("\n资源调度性能:")
        print(f"  算法: {rs_results.get('algorithm', 'N/A')}")
        print(f"  平均奖励: {rs_results.get('avg_reward', 'N/A'):.3f}")
        print(f"  平均吞吐量: {rs_results.get('avg_throughput', 'N/A'):.2f}")
    
    # 保存报告
    report_file = results_dir / 'comparison_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n完整报告已保存至: {report_file}")
    
    return report


def run_demo_mode(args):
    """
    运行演示模式 - 快速验证所有模块
    
    Args:
        args: 命令行参数
    """
    print("=" * 60)
    print("5G无线通信链路+AI创新设计项目 - 演示模式")
    print("=" * 60)
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 设置结果目录
    results_dir = setup_results_dir()
    
    # 检测GPU
    use_gpu = args.gpu if args.gpu is not None else check_gpu()
    
    # 收集各模块结果
    results_dict = {}
    
    # 1. 运行信道估计演示
    ce_samples = min(args.num_samples, 1000)  # 演示模式限制样本数
    ce_epochs = min(args.epochs, 20)
    results_dict['channel_estimation'] = run_channel_estimation_demo(
        ce_samples, ce_epochs, args.snr, args.visualize, use_gpu
    )
    
    # 2. 运行信号检测演示
    sd_samples = min(args.num_samples, 500)
    results_dict['signal_detection'] = run_signal_detection_demo(
        sd_samples, args.visualize, use_gpu
    )
    
    # 3. 运行资源调度演示
    rs_timesteps = min(args.num_samples * 10, 5000)  # 演示模式限制步数
    results_dict['resource_scheduling'] = run_resource_scheduling_demo(
        rs_timesteps, args.visualize, use_gpu
    )
    
    # 生成对比报告
    generate_comparison_report(results_dict, results_dir)
    
    print("\n" + "=" * 60)
    print("演示模式完成！")
    print("=" * 60)
    
    return results_dict


def run_full_mode(args):
    """
    运行完整链路仿真
    
    Args:
        args: 命令行参数
    """
    print("=" * 60)
    print("5G无线通信链路+AI创新设计项目 - 完整仿真模式")
    print("=" * 60)
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    results_dir = setup_results_dir()
    use_gpu = args.gpu if args.gpu is not None else check_gpu()
    
    results_dict = {}
    
    # 使用完整参数
    results_dict['channel_estimation'] = run_channel_estimation_demo(
        args.num_samples, args.epochs, args.snr, args.visualize, use_gpu
    )
    
    results_dict['signal_detection'] = run_signal_detection_demo(
        args.num_samples, args.visualize, use_gpu
    )
    
    results_dict['resource_scheduling'] = run_resource_scheduling_demo(
        args.num_samples * 20, args.visualize, use_gpu
    )
    
    generate_comparison_report(results_dict, results_dir)
    
    print("\n完整仿真完成！")
    
    return results_dict


def run_single_module(mode: str, args):
    """
    运行单个模块
    
    Args:
        mode: 模块名称
        args: 命令行参数
    """
    print("=" * 60)
    print(f"5G无线通信链路+AI创新设计项目 - {mode}模块")
    print("=" * 60)
    
    results_dir = setup_results_dir()
    use_gpu = args.gpu if args.gpu is not None else check_gpu()
    
    results_dict = {}
    
    if mode == 'channel_estimation':
        results_dict['channel_estimation'] = run_channel_estimation_demo(
            args.num_samples, args.epochs, args.snr, args.visualize, use_gpu
        )
    elif mode == 'signal_detection':
        results_dict['signal_detection'] = run_signal_detection_demo(
            args.num_samples, args.visualize, use_gpu
        )
    elif mode == 'resource_scheduling':
        results_dict['resource_scheduling'] = run_resource_scheduling_demo(
            args.num_samples * 20, args.visualize, use_gpu
        )
    
    generate_comparison_report(results_dict, results_dir)
    
    return results_dict


def main():
    """主入口函数"""
    parser = argparse.ArgumentParser(
        description='5G无线通信链路+AI创新设计项目 - 统一入口程序',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
运行模式说明:
  --mode full             运行完整链路仿真（推荐正式实验使用）
  --mode demo             运行演示模式（快速验证，适合测试）
  --mode channel_estimation  只运行AI信道估计模块
  --mode signal_detection    只运行信号检测模块
  --mode resource_scheduling 只运行资源调度模块

示例用法:
  python main.py --mode demo                    # 快速演示所有模块
  python main.py --mode full --num_samples 5000 # 完整仿真
  python main.py --mode channel_estimation --snr 15 --epochs 100
  python main.py --mode resource_scheduling --gpu
        '''
    )
    
    # 运行模式参数
    parser.add_argument(
        '--mode',
        type=str,
        default='demo',
        choices=['full', 'demo', 'channel_estimation', 'signal_detection', 'resource_scheduling'],
        help='运行模式 (default: demo)'
    )
    
    # 数值参数
    parser.add_argument(
        '--snr',
        type=float,
        default=20,
        help='信噪比 (dB, default: 20)'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=1000,
        help='样本数量 (default: 1000)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='训练轮数 (default: 50)'
    )
    
    # 系统参数
    parser.add_argument(
        '--gpu',
        type=bool,
        default=None,
        help='是否使用GPU (default: 自动检测)'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        default=True,
        help='生成可视化图表 (default: True)'
    )
    
    # 解析参数
    args = parser.parse_args()
    
    # 打印参数信息
    print(f"\n运行参数:")
    print(f"  模式: {args.mode}")
    print(f"  SNR: {args.snr} dB")
    print(f"  样本数: {args.num_samples}")
    print(f"  训练轮数: {args.epochs}")
    print(f"  可视化: {args.visualize}")
    print()
    
    # 根据模式运行
    if args.mode == 'demo':
        results = run_demo_mode(args)
    elif args.mode == 'full':
        results = run_full_mode(args)
    else:
        results = run_single_module(args.mode, args)
    
    return results


if __name__ == '__main__':
    results = main()
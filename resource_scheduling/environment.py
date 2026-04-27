"""
智能资源调度模块 - 仿真环境
实现5G资源调度的强化学习环境
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
import gym
from gym import spaces


class SchedulingEnvironment(gym.Env):
    """
    5G资源调度仿真环境

    基于OpenAI Gym接口实现的资源分配环境
    状态空间: 用户信道质量、队列状态、资源使用情况
    动作空间: 资源块分配、功率控制
    奖励函数: 系统吞吐量 + 公平性 + 能效
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, config: Optional[dict] = None):
        """
        初始化调度环境

        Args:
            config: 配置字典
        """
        super().__init__()

        # 配置参数
        self.config = config or {}
        self.num_users = self.config.get("num_users", 20)
        self.num_resource_blocks = self.config.get("num_resource_blocks", 100)
        self.max_power_dbm = self.config.get("max_power", 33)
        self.time_slots = self.config.get("time_slots", 10)

        # 信道模型参数
        self.channel_model = self.config.get("channel_model", "rayleigh")
        self.doppler_frequency = self.config.get("doppler_frequency", 10)
        self.shadowing_std = self.config.get("shadowing_std", 8)

        # 奖励权重
        self.reward_weights = self.config.get(
            "reward_weights", {"throughput": 1.0, "fairness": 0.3, "energy": 0.2}
        )

        # 定义空间
        self._define_spaces()

        # 状态变量
        self.time_step = 0
        self.channel_gains = None
        self.queue_lengths = None
        self.transmitted_bits = None

    def _define_spaces(self):
        """定义状态空间和动作空间"""
        # 状态空间: 信道质量(归一化) + 队列长度(归一化)
        # 每个用户: 1个信道质量 + 1个队列长度
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.num_users * 2,), dtype=np.float32
        )

        # 动作空间: 每个资源块的分配决策
        # 动作 = 资源块分配(每个RB分配给哪个用户) + 功率分配(每个用户的功率等级)
        # 使用多离散动作空间
        self.action_space = spaces.MultiDiscrete(
            [
                self.num_users  # 每个RB分配给哪个用户
            ]
            * self.num_resource_blocks
        )

        # 简化版: 只考虑资源块分配，功率均分
        self._action_type = "discrete"

    def reset(self) -> np.ndarray:
        """
        重置环境

        Returns:
            初始状态
        """
        self.time_step = 0

        # 初始化信道增益(dB)
        self.channel_gains = self._generate_channel_gains()

        # 初始化数据队列(比特数)
        self.queue_lengths = np.random.randint(1000, 50000, size=self.num_users).astype(
            np.float32
        )

        # 初始化传输统计
        self.transmitted_bits = np.zeros(self.num_users)

        return self._get_observation()

    def _generate_channel_gains(self) -> np.ndarray:
        """生成信道增益"""
        if self.channel_model == "rayleigh":
            # 瑞利衰落
            gains = np.random.rayleigh(
                scale=1.0, size=(self.time_slots, self.num_users)
            )
        elif self.channel_model == "rician":
            # 莱斯衰落
            gains = np.abs(
                np.random.randn(self.time_slots, self.num_users)
                + 1j * np.random.randn(self.time_slots, self.num_users)
                + 2
            ) / np.sqrt(2)
        else:
            gains = np.random.exponential(
                scale=1.0, size=(self.time_slots, self.num_users)
            )

        return gains

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行一步调度

        Args:
            action: 动作，资源块分配决策

        Returns:
            (next_state, reward, done, info)
        """
        # 解析动作
        rb_allocation = action[: self.num_resource_blocks]

        # 计算每个用户的可用资源块
        user_rbs = [[] for _ in range(self.num_users)]
        for rb_idx, user_idx in enumerate(rb_allocation):
            if 0 <= user_idx < self.num_users:
                user_rbs[user_idx].append(rb_idx)

        # 获取当前信道
        current_gains = self.channel_gains[self.time_step]

        # 计算每个用户的数据速率(香农容量近似)
        rates = np.zeros(self.num_users)
        for user_idx in range(self.num_users):
            if len(user_rbs[user_idx]) > 0:
                # 信噪比计算
                rb_gains = current_gains[user_idx]
                snr = rb_gains * 10 ** (
                    (self.max_power_dbm - 30 - 10 * np.log10(self.num_resource_blocks))
                    / 10
                )
                # 速率 = 带宽 * log2(1 + SNR)
                rate = (self.num_resource_blocks / self.time_slots) * np.log2(1 + snr)
                rates[user_idx] = rate

        # 计算传输比特数
        transmitted = np.minimum(
            rates * 1e-3, self.queue_lengths
        )  # 限制传输量不超过队列长度
        self.transmitted_bits = transmitted

        # 更新队列
        new_arrivals = np.random.poisson(lam=1000, size=self.num_users)  # 新到达的数据
        self.queue_lengths = (
            np.maximum(0, self.queue_lengths - transmitted) + new_arrivals
        )

        # 计算奖励
        reward, reward_info = self._calculate_reward(rates, transmitted)

        # 更新时间步
        self.time_step += 1
        done = self.time_step >= self.time_slots

        # 获取下一个状态
        if done:
            next_state = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            next_state = self._get_observation()

        # 信息字典
        info = {
            "rates": rates,
            "transmitted_bits": transmitted,
            "queue_lengths": self.queue_lengths,
            "user_rbs": user_rbs,
            **reward_info,
        }

        return next_state, reward, done, info

    def _get_observation(self) -> np.ndarray:
        """获取当前状态"""
        # 信道质量(归一化)
        current_gains = self.channel_gains[self.time_step]
        normalized_gains = current_gains / (np.max(current_gains) + 1e-10)

        # 队列长度(归一化)
        max_queue = 100000
        normalized_queues = np.clip(self.queue_lengths / max_queue, 0, 1)

        # 拼接状态
        state = np.concatenate([normalized_gains, normalized_queues])
        return state.astype(np.float32)

    def _calculate_reward(
        self, rates: np.ndarray, transmitted: np.ndarray
    ) -> Tuple[float, Dict]:
        """
        计算奖励函数

        Args:
            rates: 数据速率
            transmitted: 传输比特数

        Returns:
            (奖励值, 奖励详细信息)
        """
        # 吞吐量奖励
        total_throughput = np.sum(transmitted)
        throughput_reward = total_throughput / 1e6  # 归一化

        # 公平性奖励(Jain公平指数)
        if np.sum(transmitted) > 0:
            fairness = (np.sum(transmitted) ** 2) / (
                self.num_users * np.sum(transmitted**2) + 1e-10
            )
        else:
            fairness = 0
        fairness_reward = fairness

        # 能效奖励(传输比特/消耗功率)
        total_power = self.num_resource_blocks * self.max_power_dbm
        if total_power > 0:
            energy_efficiency = total_throughput / total_power
        else:
            energy_efficiency = 0
        energy_reward = energy_efficiency

        # 加权总奖励
        total_reward = (
            self.reward_weights["throughput"] * throughput_reward
            + self.reward_weights["fairness"] * fairness_reward
            + self.reward_weights["energy"] * energy_reward
        )

        reward_info = {
            "throughput": total_throughput,
            "fairness": fairness,
            "energy_efficiency": energy_efficiency,
            "throughput_reward": throughput_reward,
            "fairness_reward": fairness_reward,
            "energy_reward": energy_reward,
        }

        return total_reward, reward_info

    def render(self, mode: str = "human"):
        """渲染环境状态"""
        print(f"\n{'=' * 60}")
        print(f"时间步: {self.time_step}/{self.time_slots}")
        print(f"{'=' * 60}")
        print(f"用户数: {self.num_users}")
        print(f"资源块数: {self.num_resource_blocks}")
        print(f"\n信道增益(dB):")
        current_gains = self.channel_gains[self.time_step]
        print(
            f"  范围: {10 * np.log10(np.min(current_gains)):.1f} ~ {10 * np.log10(np.max(current_gains)):.1f} dB"
        )
        print(f"\n队列状态:")
        print(f"  总队列长度: {np.sum(self.queue_lengths):.0f} bits")
        print(f"  平均队列长度: {np.mean(self.queue_lengths):.0f} bits")


class MultiUserEnvironment(SchedulingEnvironment):
    """
    多用户资源调度环境

    扩展版调度环境，支持更多用户场景
    """

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)

        # 额外参数
        self.user_priority = self.config.get("user_priority", None)
        self.qos_requirements = self.config.get("qos_requirements", None)

        if self.user_priority is None:
            # 默认优先级(随机)
            self.user_priority = np.random.rand(self.num_users)

        if self.qos_requirements is None:
            # 默认QoS要求
            self.qos_requirements = np.random.choice(
                ["embb", "urllc", "mmtc"], size=self.num_users
            )

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """扩展的step函数"""
        next_state, reward, done, info = super().step(action)

        # 添加额外信息
        info["user_priority"] = self.user_priority
        info["qos_requirements"] = self.qos_requirements

        # QoS违约惩罚
        qos_violation = self._check_qos_violation(info["transmitted_bits"])
        if qos_violation > 0:
            reward -= qos_violation * 0.5
            info["qos_violation"] = qos_violation

        return next_state, reward, done, info

    def _check_qos_violation(self, transmitted: np.ndarray) -> float:
        """检查QoS违约"""
        violation = 0

        for i, qos in enumerate(self.qos_requirements):
            if qos == "urllc":
                # URLLC需要低延迟，惩罚长队列
                if self.queue_lengths[i] > 10000:
                    violation += 1
            elif qos == "embb":
                # eMBB需要高吞吐
                if transmitted[i] < 500:
                    violation += 0.5

        return violation


class DynamicChannelEnvironment(SchedulingEnvironment):
    """
    动态信道环境

    信道条件随时间变化，包含多径和阴影效应
    """

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)

        # 动态信道参数
        self.large_scale_fading = None
        self.small_scale_fading = None
        self.shadowing_correlation = self.config.get("shadowing_correlation", 0.5)

        # 初始化大尺度衰落
        self.large_scale_fading = np.zeros((self.time_slots, self.num_users))
        self._init_large_scale_fading()

    def _init_large_scale_fading(self):
        """初始化大尺度衰落"""
        # 路径损耗(简化模型)
        distances = np.random.uniform(50, 500, self.num_users)
        path_loss_db = 128.1 + 37.6 * np.log10(distances / 1000)

        # 阴影衰落
        shadowing = np.zeros((self.time_slots, self.num_users))
        for t in range(self.time_slots):
            if t == 0:
                shadowing[t] = np.random.normal(0, self.shadowing_std, self.num_users)
            else:
                # 相关阴影衰落
                shadowing[t] = self.shadowing_correlation * shadowing[t - 1] + np.sqrt(
                    1 - self.shadowing_correlation**2
                ) * np.random.normal(0, self.shadowing_std, self.num_users)

        self.large_scale_fading = -path_loss_db + shadowing

    def _generate_channel_gains(self) -> np.ndarray:
        """生成包含大小尺度效应的信道增益"""
        # 小尺度衰落(瑞利)
        small_scale = np.abs(
            np.random.randn(self.time_slots, self.num_users)
            + 1j * np.random.randn(self.time_slots, self.num_users)
        ) / np.sqrt(2)

        # 转换为线性值
        large_scale_linear = 10 ** (self.large_scale_fading / 10)

        # 总信道增益
        total_gains = small_scale**2 * large_scale_linear

        # 归一化
        return total_gains / np.mean(total_gains)


def create_scheduling_environment(
    env_type: str = "basic", config: Optional[dict] = None
) -> gym.Env:
    """
    创建调度环境工厂函数

    Args:
        env_type: 环境类型 ('basic', 'multi_user', 'dynamic')
        config: 配置字典

    Returns:
        调度环境实例
    """
    if env_type == "basic":
        return SchedulingEnvironment(config)
    elif env_type == "multi_user":
        return MultiUserEnvironment(config)
    elif env_type == "dynamic":
        return DynamicChannelEnvironment(config)
    else:
        raise ValueError(f"未知的环境类型: {env_type}")


if __name__ == "__main__":
    print("=" * 60)
    print("资源调度环境测试")
    print("=" * 60)

    # 测试基础环境
    print("\n1. 基础调度环境测试:")
    env = SchedulingEnvironment(
        {"num_users": 10, "num_resource_blocks": 50, "time_slots": 5}
    )

    state = env.reset()
    print(f"   状态空间: {env.observation_space}")
    print(f"   动作空间: {env.action_space}")
    print(f"   初始状态形状: {state.shape}")

    # 运行几步
    print("\n2. 运行环境模拟:")
    total_reward = 0
    for i in range(5):
        # 随机动作
        action = np.random.randint(0, env.num_users, size=env.num_resource_blocks)
        next_state, reward, done, info = env.step(action)
        total_reward += reward

        print(f"   步骤 {i + 1}: 奖励={reward:.4f}, 吞吐量={info['throughput']:.2f}")

        if done:
            break

    print(f"\n   总奖励: {total_reward:.4f}")

    # 测试多用户环境
    print("\n3. 多用户调度环境测试:")
    env_multi = MultiUserEnvironment(
        {"num_users": 20, "num_resource_blocks": 100, "time_slots": 10}
    )

    state = env_multi.reset()
    action = np.random.randint(
        0, env_multi.num_users, size=env_multi.num_resource_blocks
    )
    _, reward, _, info = env_multi.step(action)

    print(f"   用户优先级: {env_multi.user_priority}")
    print(f"   QoS需求: {env_multi.qos_requirements}")
    print(f"   Jain公平指数: {info['fairness']:.4f}")

    print("\n" + "=" * 60)
    print("环境测试完成！")
    print("=" * 60)

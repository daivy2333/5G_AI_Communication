"""
智能资源调度模块 - 强化学习智能体
实现基于PPO和DQN的智能资源调度算法
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
import sys
from pathlib import Path

# 尝试导入强化学习框架
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.distributions import Categorical
    from tqdm import tqdm

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import gym

    HAS_GYM = True
except ImportError:
    HAS_GYM = False

try:
    from stable_baselines3 import PPO, DQN
    from stable_baselines3.common.vec_env import DummyVecEnv

    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False

from .environment import SchedulingEnvironment, create_scheduling_environment


class RLSchedulerBase:
    """
    强化学习调度器基类

    定义通用接口
    """

    def __init__(self, env_config: dict, agent_config: Optional[dict] = None):
        """
        初始化调度器

        Args:
            env_config: 环境配置
            agent_config: 智能体配置
        """
        self.env_config = env_config
        self.agent_config = agent_config or {}

        # 创建环境
        self.env = create_scheduling_environment(
            env_config.get("env_type", "basic"), env_config
        )

        # 训练统计
        self.training_stats = {
            "episode_rewards": [],
            "episode_lengths": [],
            "throughputs": [],
            "fairness": [],
        }

    def train(self, total_timesteps: int):
        """训练调度器"""
        raise NotImplementedError

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """选择动作"""
        raise NotImplementedError

    def save(self, path: str):
        """保存模型"""
        raise NotImplementedError

    def load(self, path: str):
        """加载模型"""
        raise NotImplementedError


class PPOScheduler(RLSchedulerBase):
    """
    基于PPO算法的资源调度器

    使用Proximal Policy Optimization算法进行资源分配
    """

    def __init__(self, env_config: dict, agent_config: Optional[dict] = None):
        super().__init__(env_config, agent_config)

        # 设备设置
        if HAS_TORCH:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"使用设备: {self.device}")
        else:
            self.device = None

        # PPO参数
        self.gamma = self.agent_config.get("gamma", 0.99)
        self.gae_lambda = self.agent_config.get("gae_lambda", 0.95)
        self.clip_range = self.agent_config.get("clip_range", 0.2)
        self.entropy_coef = self.agent_config.get("entropy_coef", 0.01)
        self.learning_rate = self.agent_config.get("learning_rate", 3e-4)
        self.hidden_dim = self.agent_config.get("hidden_dim", 128)

        # 如果有stable-baselines3，使用它
        if HAS_SB3:
            self._use_sb3 = True
            self.model = PPO(
                "MlpPolicy",
                self.env,
                learning_rate=self.learning_rate,
                gamma=self.gamma,
                clip_range=self.clip_range,
                ent_coef=self.entropy_coef,
                verbose=0,
            )
        elif HAS_TORCH:
            self._use_sb3 = False
            self._build_pytorch_model()
        else:
            raise RuntimeError("需要PyTorch或stable-baselines3")

    def _build_pytorch_model(self):
        """构建PyTorch PPO模型"""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch未安装")

        state_dim = self.env.observation_space.shape[0]
        action_dim = (
            self.env.num_resource_blocks * self.num_users
            if hasattr(self.env, "num_users")
            else 1000
        )

        class PPONetwork(nn.Module):
            def __init__(self, state_dim, action_dim, hidden_dim):
                super().__init__()

                # 策略网络
                self.actor = nn.Sequential(
                    nn.Linear(state_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, action_dim),
                )

                # 价值网络
                self.critic = nn.Sequential(
                    nn.Linear(state_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                )

            def forward(self, x):
                return self.actor(x), self.critic(x)

            def get_action(self, x):
                logits = self.actor(x)
                probs = F.softmax(logits, dim=-1)
                dist = Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                return action, log_prob

            def evaluate(self, x, actions):
                logits = self.actor(x)
                probs = F.softmax(logits, dim=-1)
                dist = Categorical(probs)
                log_prob = dist.log_prob(actions)
                entropy = dist.entropy()
                value = self.critic(x)
                return log_prob, entropy, value

        self.policy = PPONetwork(
            state_dim,
            self.env.action_space.nvec[0]
            if hasattr(self.env.action_space, "nvec")
            else 100,
            self.hidden_dim,
        ).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)

    def train(self, total_timesteps: int, log_interval: int = 100):
        """
        训练PPO调度器

        Args:
            total_timesteps: 总训练步数
            log_interval: 日志输出间隔
        """
        print("=" * 60)
        print("PPO资源调度器训练")
        print("=" * 60)
        print(f"环境: 5G资源调度")
        print(f"用户数: {self.env_config.get('num_users', 20)}")
        print(f"资源块数: {self.env_config.get('num_resource_blocks', 100)}")
        print(f"总训练步数: {total_timesteps}")
        print()

        if self._use_sb3 and HAS_SB3:
            # 使用stable-baselines3
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=None,
                log_interval=log_interval,
            )
            print("训练完成！")
        else:
            # 自实现PPO
            self._train_ppo(total_timesteps, log_interval)

    def _train_ppo(self, total_timesteps: int, log_interval: int):
        """自实现PPO训练"""
        num_episodes = total_timesteps // self.env.time_slots
        batch_size = self.agent_config.get("batch_size", 64)
        epochs = self.agent_config.get("ppo_epochs", 4)

        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_info = []

            # 收集轨迹
            states, actions, rewards, log_probs, values = [], [], [], [], []

            done = False
            while not done:
                state_tensor = torch.FloatTensor(state).to(self.device)

                with torch.no_grad():
                    action, log_prob = self.policy.get_action(state_tensor)
                    value = self.policy.critic(state_tensor)

                # 执行动作
                action_np = (
                    action.cpu().numpy()
                    if hasattr(action, "cpu")
                    else action.numpy()
                    if hasattr(action, "numpy")
                    else action
                )
                next_state, reward, done, info = self.env.step(action_np)

                # 存储
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob)
                values.append(value)

                state = next_state
                episode_reward += reward

            # 存储episode统计
            self.training_stats["episode_rewards"].append(episode_reward)
            self.training_stats["episode_lengths"].append(len(rewards))
            self.training_stats["throughputs"].append(info.get("throughput", 0))
            self.training_stats["fairness"].append(info.get("fairness", 0))

            # PPO更新
            if episode > 0 and episode % batch_size == 0:
                self._update_ppo(states, actions, rewards, log_probs, values, epochs)

            # 日志输出
            if (episode + 1) % log_interval == 0:
                avg_reward = np.mean(
                    self.training_stats["episode_rewards"][-log_interval:]
                )
                avg_throughput = np.mean(
                    self.training_stats["throughputs"][-log_interval:]
                )
                avg_fairness = np.mean(self.training_stats["fairness"][-log_interval:])

                print(
                    f"Episode {episode + 1:4d}/{num_episodes} | "
                    f"Avg Reward: {avg_reward:.3f} | "
                    f"Throughput: {avg_throughput:.2f} | "
                    f"Fairness: {avg_fairness:.3f}"
                )

        print("\n训练完成！")

    def _update_ppo(self, states, actions, rewards, old_log_probs, values, epochs):
        """PPO更新"""
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        old_log_probs = torch.FloatTensor(
            [lp.item() if hasattr(lp, "item") else lp for lp in old_log_probs]
        ).to(self.device)

        # 计算折扣回报
        returns = []
        discounted = 0
        for reward in reversed(rewards):
            discounted = reward + self.gamma * discounted
            returns.insert(0, discounted)
        returns = torch.FloatTensor(returns).to(self.device)

        # 标准化
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        for _ in range(epochs):
            # 计算新策略的概率
            log_probs, entropy, values_new = self.policy.evaluate(states, actions)

            # 计算比率
            ratio = torch.exp(log_probs - old_log_probs)

            # PPO裁剪
            surr1 = ratio * returns
            surr2 = (
                torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * returns
            )
            policy_loss = -torch.min(surr1, surr2).mean()

            # 价值损失
            value_loss = F.mse_loss(values_new.squeeze(), returns)

            # 熵正则化
            entropy_loss = -entropy.mean()

            # 总损失
            loss = policy_loss + 0.5 * value_loss + self.entropy_coef * entropy_loss

            # 更新
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """选择动作"""
        if self._use_sb3 and HAS_SB3:
            action, _ = self.model.predict(state)
            return action
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action, _ = self.policy.get_action(state_tensor)
            return action.cpu().numpy().squeeze()

    def save(self, path: str):
        """保存模型"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self._use_sb3 and HAS_SB3:
            self.model.save(str(path))
        else:
            torch.save(
                {
                    "policy_state_dict": self.policy.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "training_stats": self.training_stats,
                },
                path,
            )

        print(f"模型已保存至: {path}")

    def load(self, path: str):
        """加载模型"""
        path = Path(path)
        if not path.exists():
            print(f"警告: 模型文件不存在: {path}")
            return

        if self._use_sb3 and HAS_SB3:
            self.model = PPO.load(str(path))
        else:
            checkpoint = torch.load(path)
            self.policy.load_state_dict(checkpoint["policy_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.training_stats = checkpoint.get("training_stats", self.training_stats)

        print(f"模型已加载: {path}")


class DQNScheduler(RLSchedulerBase):
    """
    基于DQN算法的资源调度器

    使用Deep Q-Network进行资源分配决策
    """

    def __init__(self, env_config: dict, agent_config: Optional[dict] = None):
        super().__init__(env_config, agent_config)

        # 设备设置
        if HAS_TORCH:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"使用设备: {self.device}")
        else:
            self.device = None

        # DQN参数
        self.gamma = self.agent_config.get("gamma", 0.99)
        self.epsilon = self.agent_config.get("epsilon", 1.0)
        self.epsilon_decay = self.agent_config.get("epsilon_decay", 0.995)
        self.epsilon_min = self.agent_config.get("epsilon_min", 0.01)
        self.learning_rate = self.agent_config.get("learning_rate", 1e-3)
        self.batch_size = self.agent_config.get("batch_size", 64)
        self.memory_size = self.agent_config.get("memory_size", 10000)
        self.target_update = self.agent_config.get("target_update", 100)

        # 经验回放内存
        self.memory = []

        if HAS_SB3:
            self._use_sb3 = True
            self.model = DQN(
                "MlpPolicy",
                self.env,
                learning_rate=self.learning_rate,
                gamma=self.gamma,
                exploration_fraction=0.1,
                verbose=0,
            )
        elif HAS_TORCH:
            self._use_sb3 = False
            self._build_dqn_model()
        else:
            raise RuntimeError("需要PyTorch或stable-baselines3")

    def _build_dqn_model(self):
        """构建DQN网络"""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch未安装")

        state_dim = self.env.observation_space.shape[0]
        action_dim = (
            self.env.action_space.n if hasattr(self.env.action_space, "n") else 100
        )

        class DQNNetwork(nn.Module):
            def __init__(self, state_dim, action_dim):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(state_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, action_dim),
                )

            def forward(self, x):
                return self.network(x)

        self.q_network = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_network = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

    def train(self, total_timesteps: int, log_interval: int = 100):
        """训练DQN调度器"""
        print("=" * 60)
        print("DQN资源调度器训练")
        print("=" * 60)
        print(f"环境: 5G资源调度")
        print(f"用户数: {self.env_config.get('num_users', 20)}")
        print(f"总训练步数: {total_timesteps}")
        print()

        if self._use_sb3 and HAS_SB3:
            self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval)
        else:
            self._train_dqn(total_timesteps, log_interval)

    def _train_dqn(self, total_timesteps: int, log_interval: int):
        """自实现DQN训练"""
        num_episodes = total_timesteps // self.env.time_slots
        step_count = 0

        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0

            done = False
            while not done:
                # Epsilon贪婪选择动作
                if np.random.rand() < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = self.select_action(state)

                # 执行
                next_state, reward, done, info = self.env.step(action)

                # 存储经验
                self._store_transition(state, action, reward, next_state, done)

                # 训练
                if len(self.memory) >= self.batch_size:
                    self._train_step()

                state = next_state
                episode_reward += reward
                step_count += 1

                # 更新目标网络
                if step_count % self.target_update == 0:
                    self.target_network.load_state_dict(self.q_network.state_dict())

            # Epsilon衰减
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # 记录统计
            self.training_stats["episode_rewards"].append(episode_reward)
            self.training_stats["throughputs"].append(info.get("throughput", 0))

            if (episode + 1) % log_interval == 0:
                avg_reward = np.mean(
                    self.training_stats["episode_rewards"][-log_interval:]
                )
                print(
                    f"Episode {episode + 1:4d}/{num_episodes} | "
                    f"Avg Reward: {avg_reward:.3f} | "
                    f"Epsilon: {self.epsilon:.3f}"
                )

        print("\n训练完成！")

    def _store_transition(self, state, action, reward, next_state, done):
        """存储转移经验"""
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

    def _train_step(self):
        """单步训练"""
        # 采样
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        states = torch.FloatTensor([self.memory[i][0] for i in batch]).to(self.device)
        actions = torch.LongTensor([self.memory[i][1] for i in batch]).to(self.device)
        rewards = torch.FloatTensor([self.memory[i][2] for i in batch]).to(self.device)
        next_states = torch.FloatTensor([self.memory[i][3] for i in batch]).to(
            self.device
        )
        dones = torch.FloatTensor([self.memory[i][4] for i in batch]).to(self.device)

        # 计算目标Q值
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # 计算当前Q值
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()

        # 损失和更新
        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """选择动作"""
        if self._use_sb3 and HAS_SB3:
            action, _ = self.model.predict(state)
            return action
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return q_values.argmax(1).item()

    def save(self, path: str):
        """保存模型"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self._use_sb3 and HAS_SB3:
            self.model.save(str(path))
        else:
            torch.save(
                {
                    "q_network": self.q_network.state_dict(),
                    "target_network": self.target_network.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "epsilon": self.epsilon,
                    "training_stats": self.training_stats,
                },
                path,
            )

    def load(self, path: str):
        """加载模型"""
        path = Path(path)
        if not path.exists():
            print(f"警告: 模型文件不存在: {path}")
            return

        if self._use_sb3 and HAS_SB3:
            self.model = DQN.load(str(path))
        else:
            checkpoint = torch.load(path)
            self.q_network.load_state_dict(checkpoint["q_network"])
            self.target_network.load_state_dict(checkpoint["target_network"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.epsilon = checkpoint["epsilon"]
            self.training_stats = checkpoint.get("training_stats", self.training_stats)


def create_scheduler(
    algorithm: str = "ppo",
    env_config: Optional[dict] = None,
    agent_config: Optional[dict] = None,
) -> RLSchedulerBase:
    """
    创建调度器工厂函数

    Args:
        algorithm: 算法类型 ('ppo', 'dqn')
        env_config: 环境配置
        agent_config: 智能体配置

    Returns:
        调度器实例
    """
    if algorithm.lower() == "ppo":
        return PPOScheduler(env_config, agent_config)
    elif algorithm.lower() == "dqn":
        return DQNScheduler(env_config, agent_config)
    else:
        raise ValueError(f"未知的算法类型: {algorithm}")


def train_and_compare(num_timesteps: int = 50000):
    """
    训练并对比PPO和DQN调度器

    Args:
        num_timesteps: 训练步数
    """
    env_config = {"num_users": 10, "num_resource_blocks": 50, "time_slots": 10}

    print("=" * 60)
    print("PPO vs DQN 调度器对比实验")
    print("=" * 60)

    # 训练PPO
    print("\n训练PPO调度器...")
    ppo_scheduler = create_scheduler("ppo", env_config)
    ppo_scheduler.train(num_timesteps // 2, log_interval=50)
    ppo_scheduler.save("resource_scheduling/results/ppo_scheduler")

    # 训练DQN
    print("\n训练DQN调度器...")
    dqn_scheduler = create_scheduler("dqn", env_config)
    dqn_scheduler.train(num_timesteps // 2, log_interval=50)
    dqn_scheduler.save("resource_scheduling/results/dqn_scheduler")

    # 对比结果
    print("\n" + "=" * 60)
    print("训练结果对比")
    print("=" * 60)

    for name, scheduler in [("PPO", ppo_scheduler), ("DQN", dqn_scheduler)]:
        avg_reward = np.mean(scheduler.training_stats["episode_rewards"])
        avg_throughput = np.mean(scheduler.training_stats["throughputs"])
        print(f"\n{name}:")
        print(f"  平均奖励: {avg_reward:.3f}")
        print(f"  平均吞吐量: {avg_throughput:.2f}")

    return ppo_scheduler, dqn_scheduler


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="资源调度器训练")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="ppo",
        choices=["ppo", "dqn"],
        help="强化学习算法",
    )
    parser.add_argument("--timesteps", type=int, default=50000, help="训练步数")
    args = parser.parse_args()

    env_config = {"num_users": 20, "num_resource_blocks": 100, "time_slots": 10}

    scheduler = create_scheduler(args.algorithm, env_config)
    scheduler.train(args.timesteps, log_interval=100)
    scheduler.save(f"resource_scheduling/results/{args.algorithm}_scheduler")

# -*- coding: utf-8 -*-
"""
模块2：DQN网络、经验回放与训练循环
深度Q网络的核心实现
"""

import os
import random
import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

#新增画图部分
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import threading
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def pick_device():
    """选择计算设备，优先使用CUDA"""
    if not torch.cuda.is_available():
        return 'cpu'
    # 尝试做一次极小的 CUDA 前向，失败就回退
    try:
        x = torch.randn(1, device='cuda')
        y = x.relu()  # 触发一次kernel
        _ = y.item()
        return 'cuda'
    except Exception as e:
        print("[Device] CUDA 不可用，回退到 CPU：", repr(e))
        return 'cpu'
@dataclass
class DQNConfig:
    """DQN训练配置"""
    # 核心超参数
    gamma: float = 0.95              # 折扣因子
    lr: float = 1e-3                 # 学习率
    batch_size: int = 64             # 批大小
    
    # 训练控制
    train_start_size: int = 1000     # 开始训练的最小经验数
    target_update_every: int = 500   # 目标网络更新频率
    max_episodes: int = 800          # 最大训练轮数
    
    # 探索策略
    epsilon_start: float = 1.0       # 初始探索率
    epsilon_end: float = 0.05        # 最终探索率
    epsilon_decay_steps: int = 20000 # 探索率衰减步数
    
    # 网络结构
    hidden_sizes: List[int] = None   # 隐藏层大小
    
    # 其他设置
    device: str = pick_device()
    seed: int = 42

    
    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [128, 128]


class QNetwork(nn.Module):
    """近似Q函数的神经网络，Q网络：状态-动作价值函数近似器
    Q 函数表示在某个状态下，执行一个动作能得到的长期回报。
    换句话说，这个网络的输入是“状态”，输出是一组数字，对应每个动作的价值。
    """
    
    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes: List[int] = [128, 128]):
        super().__init__()
        
        layers = []
        input_dim = obs_dim
        
        # 构建隐藏层
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)  # 添加dropout防止过拟合，随机丢弃 10% 的神经元，避免网络过拟合。
            ])
            input_dim = hidden_size
        
        # 输出层
        layers.append(nn.Linear(input_dim, n_actions))
        
        self.net = nn.Sequential(*layers)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重
            遍历所有子模块，如果是线性层：
            用 Xavier 均匀分布初始化权重（适合 ReLU 激活，能让训练更稳定）。
            把偏置初始化为 0。
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.net(x)


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int = 2000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """添加经验"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """采样批量经验"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.FloatTensor(np.stack(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.stack(next_states)),
            torch.FloatTensor(dones)
        )
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    @property
    def is_full(self) -> bool:
        return len(self.buffer) == self.capacity


class DQNAgent:
    """DQN智能体"""
    
    def __init__(self, obs_dim: int, n_actions: int, config: DQNConfig):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.config = config
        
        # 设置随机种子
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)
        
        # 创建Q网络和目标网络
        self.q_network = QNetwork(obs_dim, n_actions, config.hidden_sizes).to(config.device)
        self.target_network = QNetwork(obs_dim, n_actions, config.hidden_sizes).to(config.device)
        
        # 初始化目标网络
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # 目标网络不需要梯度
        
        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.lr)
        
        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer()
        
        # 训练状态
        self.total_steps = 0
        self.training_steps = 0
        self.episode_count = 0
        
        # 性能统计
        self.loss_history = []
        self.q_value_history = []
    
    def get_epsilon(self) -> float:
        """获取当前探索率（线性衰减）"""
        if self.total_steps >= self.config.epsilon_decay_steps:
            return self.config.epsilon_end
        
        decay_ratio = self.total_steps / self.config.epsilon_decay_steps
        return self.config.epsilon_start + (self.config.epsilon_end - self.config.epsilon_start) * decay_ratio
    
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """选择动作（ε-贪心策略）"""
        if not eval_mode and random.random() < self.get_epsilon():
            # 探索：随机动作
            return random.randrange(self.n_actions)
        
        # 利用：选择Q值最大的动作
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.config.device)
            q_values = self.q_network(state_tensor)
            action = q_values.argmax(dim=1).item()
        
        return action
    
    def store_experience(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """存储经验到回放缓冲区"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self) -> Optional[float]:
        """更新Q网络"""
        if len(self.replay_buffer) < self.config.train_start_size:
            return None
        
        # 采样批量经验
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.config.batch_size)
        
        # 移动到设备
        states = states.to(self.config.device)
        actions = actions.to(self.config.device)
        rewards = rewards.to(self.config.device)
        next_states = next_states.to(self.config.device)
        dones = dones.to(self.config.device)
        
        # 计算当前Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.config.gamma * next_q_values * (1 - dones))
        
        # 计算损失
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        # 反向传播和优化
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # 更新统计信息
        self.training_steps += 1
        self.loss_history.append(loss.item())
        self.q_value_history.append(current_q_values.mean().item())
        
        # 定期更新目标网络
        if self.training_steps % self.config.target_update_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """获取状态的Q值"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.config.device)
            q_values = self.q_network(state_tensor)
            return q_values.cpu().numpy()[0]
    
    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'total_steps': self.total_steps,
            'training_steps': self.training_steps,
            'episode_count': self.episode_count,
            'loss_history': self.loss_history,
            'q_value_history': self.q_value_history
        }, filepath)
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.config.device, weights_only=False)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.total_steps = checkpoint.get('total_steps', 0)
        self.training_steps = checkpoint.get('training_steps', 0)
        self.episode_count = checkpoint.get('episode_count', 0)
        self.loss_history = checkpoint.get('loss_history', [])
        self.q_value_history = checkpoint.get('q_value_history', [])
    
    def get_training_stats(self) -> Dict:
        """获取训练统计信息"""
        return {
            'total_steps': self.total_steps,
            'training_steps': self.training_steps,
            'episode_count': self.episode_count,
            'epsilon': self.get_epsilon(),
            'buffer_size': len(self.replay_buffer),
            'recent_loss': np.mean(self.loss_history[-100:]) if self.loss_history else 0.0,
            'recent_q_value': np.mean(self.q_value_history[-100:]) if self.q_value_history else 0.0
        }


class DQNTrainer:
    """DQN训练器"""
    
    def __init__(self, agent: DQNAgent, env, config: DQNConfig):
        self.agent = agent
        self.env = env
        self.config = config
        
        # 新增 画图用
        self.episode_rewards = []
        self.episode_losses = []
        self.episode_f1_scores = []
        self.moving_avg_rewards = []
        self.moving_avg_windows = 50
        # 训练统计,原
        self.episode_lengths = []
        self.episode_f1_scores = []
        self.best_f1 = -float('inf')
        self.best_combo = None
        self.best_model_path = None
    
        # 实时绘图相关
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.ax3 = None
        self.plot_thread = None
        self.plot_lock = threading.Lock()
        self.plot_enabled = True  # 控制是否启用绘图
    #新增 画图用
    def setup_realtime_plot(self):
        """设置实时绘图"""
        plt.ion()  # 开启交互模式
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(12, 10))
        self.fig.suptitle('DQN训练实时监控', fontsize=16)
        
        # 奖励曲线
        self.ax1.set_title('Episode Rewards & Moving Average')
        self.ax1.set_xlabel('Episode')
        self.ax1.set_ylabel('Reward')
        self.ax1.grid(True, alpha=0.3)
        
        # 损失曲线
        self.ax2.set_title('Training Loss')
        self.ax2.set_xlabel('Episode')
        self.ax2.set_ylabel('Loss')
        self.ax2.grid(True, alpha=0.3)
        
        # F1分数曲线
        self.ax3.set_title('Episode F1 Scores')
        self.ax3.set_xlabel('Episode')
        self.ax3.set_ylabel('F1 Score')
        self.ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.01)
    
    def update_plot(self):
        """更新绘图"""
        if not self.plot_enabled or len(self.episode_rewards) == 0:
            return
            
        with self.plot_lock:
            episodes = list(range(1, len(self.episode_rewards) + 1))
            
            # 清空所有子图
            self.ax1.clear()
            self.ax2.clear()
            self.ax3.clear()
            
            # 奖励曲线
            self.ax1.plot(episodes, self.episode_rewards, 'b-', alpha=0.6, label='Episode Reward')
            if len(self.moving_avg_rewards) > 0:
                self.ax1.plot(episodes[-len(self.moving_avg_rewards):], 
                             self.moving_avg_rewards, 'r-', linewidth=2, label=f'{self.moving_avg_window}-Episode MA')
            self.ax1.set_title('Episode Rewards & Moving Average')
            self.ax1.set_xlabel('Episode')
            self.ax1.set_ylabel('Reward')
            self.ax1.legend()
            self.ax1.grid(True, alpha=0.3)

                        # 损失曲线
            if len(self.episode_losses) > 0:
                loss_episodes = list(range(1, len(self.episode_losses) + 1))
                self.ax2.plot(loss_episodes, self.episode_losses, 'g-', alpha=0.7)
                self.ax2.set_title('Training Loss')
                self.ax2.set_xlabel('Episode')
                self.ax2.set_ylabel('Loss')
                self.ax2.grid(True, alpha=0.3)
            
            # F1分数曲线
            if len(self.episode_f1_scores) > 0:
                self.ax3.plot(episodes, self.episode_f1_scores, 'purple', alpha=0.7, label='Episode F1')
                max_f1 = max(self.episode_f1_scores)
                self.ax3.axhline(y=max_f1, color='red', linestyle='--', alpha=0.7, 
                                label=f'Max F1: {max_f1:.4f}')
                self.ax3.set_title('Episode F1 Scores')
                self.ax3.set_xlabel('Episode')
                self.ax3.set_ylabel('F1 Score')
                self.ax3.legend()
                self.ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.draw()
            plt.pause(0.01)
    def calculate_moving_average(self):
      """计算移动平均"""
      if len(self.episode_rewards) >= self.moving_avg_window:
        window_rewards = self.episode_rewards[-self.moving_avg_window:]
        moving_avg = sum(window_rewards) / len(window_rewards)
        self.moving_avg_rewards.append(moving_avg)
      elif len(self.episode_rewards) > 0:
        # 如果数据不足窗口大小，使用所有可用数据
        moving_avg = sum(self.episode_rewards) / len(self.episode_rewards)
        self.moving_avg_rewards.append(moving_avg)
    #新增参数：plot_realtime
    def train(self, save_dir: str = './models',plot_realtime: bool = True) -> Dict:
        """训练DQN智能体"""
        os.makedirs(save_dir, exist_ok=True)
        
        #新增： 画图用
        self.plot_enabled = plot_realtime

        if self.plot_enabled:
            self.setup_realtime_plot()

        print("开始DQN训练...")
        print(f"设备: {self.config.device}")
        print(f"最大训练轮数: {self.config.max_episodes}")

        #新增 参数
        best_avg_reward = float('-inf')
        best_model_path = None
        patience_counter = 0
        start_time = time.time()
        
        for episode in range(self.config.max_episodes + 1):
            # 新增 episode_start
            episode_start = time.time()
            episode_reward, episode_length, final_f1, loss_count, episode_loss = self._train_episode()
            
            # 记录统计信息
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.episode_f1_scores.append(final_f1)
            self.agent.episode_count += 1
            if loss_count > 0:
                avg_episode_loss = episode_loss / loss_count
                self.episode_losses.append(avg_episode_loss)
            
            # 计算移动平均
            self.calculate_moving_average()

            if self.plot_enabled and episode % 5 == 0:
                self.update_plot()
            
            episode_time = time.time() - episode_start
            current_epsilon = self.agent.get_epsilon()

            if episode % 50 == 0 and episode <= 10:
                avg_reward = self.moving_avg_rewards[-1] if self.moving_avg_rewards else episode_reward
                print(f"[训练] Episode {episode:4d} | "
                      f"奖励: {episode_reward:+.4f} | "
                      f"平均奖励: {avg_reward:+.4f} | "
                      f"F1: {final_f1:.4f} | "
                      f"损失: {avg_episode_loss:.4f} | "
                      f"探索率: {current_epsilon:.3f} | "
                      f"时长: {episode_time:.2f}s")
            if len(self.moving_avg_rewards) >= 10:
                current_avg = self.moving_avg_rewards[-1]
                if current_avg > best_avg_reward:
                  best_avg_reward = current_avg
                  patience_counter = 0
                  best_model_path = os.path.join(save_dir, f'best__model_ep{episode}.pth')
                  self.agent.save_model(best_model_path)
                  print(f"新最佳平均奖励 {best_avg_reward:.4f}，模型已保存到 {best_model_path}")
                else:
                  patience_counter += 1
          #新增至此
          
            # 更新最佳模型
            if final_f1 > self.best_f1:
                self.best_f1 = final_f1
                self.best_combo = self.env.current_combo_tuple()
                self.best_model_path = os.path.join(save_dir, 'best_model.pth')
                self.agent.save_model(self.best_model_path)
            
            # 定期输出训练信息
            if (episode + 1) % 20 == 0 or episode == 0:
                self._print_training_progress(episode)
            
            # 定期保存检查点
            if (episode + 1) % 100 == 0:
                checkpoint_path = os.path.join(save_dir, f'checkpoint_ep{episode+1}.pth')
                self.agent.save_model(checkpoint_path)
        
        print(f"\n训练完成！")
        print(f"最佳F1分数: {self.best_f1:.4f}")
        print(f"最佳参数组合: {self.env.param_space.decode_combo(self.best_combo)}")
        print(f"最佳模型路径: {self.best_model_path}")
        
        return self._get_training_summary()
    
    def _train_episode(self) -> Tuple[float, int, float, int, float]:
        """训练一个episode"""
        state = self.env.reset(random_start=True)
        episode_loss = 0.0
        loss_count = 0
        episode_reward = 0.0
        episode_length = 0
        
        while True:
            # 选择动作
            action = self.agent.select_action(state)
            
            # 执行动作
            next_state, reward, done, info = self.env.step(action)
            
            # 存储经验
            self.agent.store_experience(state, action, reward, next_state, done)
            
            # 更新网络
            if len(self.agent.memory) >= self.config.train_start_size:
                loss = self.agent.train()
                if loss is not None:
                  episode_loss += loss
                  loss_count += 1

            
            # 更新状态和统计
            state = next_state
            episode_reward += reward #total_reward
            episode_length += 1 
            self.agent.total_steps += 1 #总步长
            
            if done:
                break
        
        final_f1 = info['f1']
        return episode_reward, episode_length, final_f1, loss_count, episode_loss
    
    def _print_training_progress(self, episode: int):
        """打印训练进度"""
        stats = self.agent.get_training_stats()
        recent_rewards = np.mean(self.episode_rewards[-20:]) if self.episode_rewards else 0.0
        recent_f1 = np.mean(self.episode_f1_scores[-20:]) if self.episode_f1_scores else 0.0
        
        print(f"[训练] Episode {episode+1:4d} | "
              f"平均奖励: {recent_rewards:+.4f} | "
              f"平均F1: {recent_f1:.4f} | "
              f"探索率: {stats['epsilon']:.3f} | "
              f"最佳F1: {self.best_f1:.4f}")
    
    def _get_training_summary(self) -> Dict:
        """获取训练总结"""
        return {
            'best_f1': self.best_f1,
            'best_combo': self.best_combo,
            'best_model_path': self.best_model_path,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_f1_scores': self.episode_f1_scores,
            'training_stats': self.agent.get_training_stats()
        }


if __name__ == "__main__":
    # 测试DQN模块
    print("测试DQN模块...")
    
    # 创建配置
    config = DQNConfig(
        max_episodes=10,  # 测试用小数值
        epsilon_decay_steps=100,
        train_start_size=50
    )
    
    # 创建智能体
    obs_dim = 14  # 环境观测维度
    n_actions = 10  # 动作数量
    agent = DQNAgent(obs_dim, n_actions, config)
    
    print(f"Q网络结构: {agent.q_network}")
    print(f"设备: {config.device}")
    
    # 测试动作选择
    dummy_state = np.random.random(obs_dim)
    action = agent.select_action(dummy_state)
    print(f"测试动作选择: {action}")
    
    # 测试Q值计算
    q_values = agent.get_q_values(dummy_state)
    print(f"Q值: {q_values}")
    
    # 测试经验存储和更新
    for i in range(100):
        state = np.random.random(obs_dim)
        action = random.randint(0, n_actions - 1)
        reward = random.uniform(-1, 1)
        next_state = np.random.random(obs_dim)
        done = random.random() < 0.1
        
        agent.store_experience(state, action, reward, next_state, done)
    
    # 测试网络更新
    loss = agent.update()
    print(f"测试损失: {loss}")
    
    # 测试模型保存和加载
    test_model_path = './test_model.pth'
    agent.save_model(test_model_path)
    print(f"模型已保存到: {test_model_path}")
    
    # 创建新智能体并加载模型
    new_agent = DQNAgent(obs_dim, n_actions, config)
    new_agent.load_model(test_model_path)
    print("模型加载成功")
    
    # 清理测试文件
    if os.path.exists(test_model_path):
        os.remove(test_model_path)
    
    print("\nDQN模块测试完成！")

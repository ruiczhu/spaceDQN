import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        # 增加网络容量以处理更大的动作空间
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),  # 更新输入维度为90
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        # 修正实际的状态空间大小
        # 基础状态: 11 (player_x, player_y, player_vx, player_vy, player_direction_x, 
        #            player_direction_y, lives, can_shoot, is_invulnerable, laser_cooldown, active_lasers)
        # 陨石信息: 10 * 6 = 60 (每个陨石 x, y, dist, vx, vy, speed)
        # 激光信息: 3 * 3 = 9 (每个激光 x, y, dist)
        # 总计: 11 + 60 + 9 = 80
        self.state_size = state_size  # 更新为正确的状态空间大小
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        self.gamma = 0.99
        self.initial_epsilon = 1.0
        self.epsilon = self.initial_epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998
        self.episode_count = 0
        self.learning_rate = 0.0001
        self.target_update_frequency = 4
        self.update_counter = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reward_scale = 0.5
        self.batch_size = 64
        self.min_experiences = 64  # 添加最小经验数量要求

        # 主网络和目标网络 - 使用正确的state_size
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())  # 初始化目标网络
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def remember(self, state, action, reward, next_state, done, training=True):
        """
        存储经验到回放缓冲区
        training: 是否处于训练阶段
        """
        reward = reward * self.reward_scale
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            act_values = self.model(state)
            return torch.argmax(act_values[0]).item()

    def replay(self, batch_size):
        if len(self.memory) < self.min_experiences:
            return None  # 返回None而不是0.0，表示还没有足够的经验进行训练

        # 从经验回放缓冲区采样
        minibatch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([i[0] for i in minibatch]).to(self.device)
        actions = torch.LongTensor([i[1] for i in minibatch]).to(self.device)
        rewards = torch.FloatTensor([i[2] for i in minibatch]).to(self.device)
        next_states = torch.FloatTensor([i[3] for i in minibatch]).to(self.device)
        dones = torch.FloatTensor([i[4] for i in minibatch]).to(self.device)

        # 计算当前Q值和目标Q值
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            # 使用Double DQN方法选择动作
            next_actions = self.model(next_states).max(1)[1]
            next_q_values = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target = rewards + self.gamma * next_q_values * (1 - dones)

        # 使用clip防止过大的loss
        loss = nn.SmoothL1Loss()(current_q_values.squeeze(), target)
        loss_value = loss.item()

        # 优化模型
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)  # 降低梯度裁剪阈值
        self.optimizer.step()

        # 更新目标网络
        self.update_counter += 1
        if self.update_counter % self.target_update_frequency == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        return loss_value  # 返回loss值

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, name):
        torch.save(self.model.state_dict(), name)

    def load(self, name):
        self.model.load_state_dict(torch.load(name, map_location=self.device))
        self.model.eval()

    def update_epsilon(self):
        """仅在 episode 结束时调用此方法"""
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(
                self.epsilon_min,
                self.initial_epsilon * (self.epsilon_decay ** self.episode_count)
            )
        self.episode_count += 1

    def reset_epsilon(self):
        """重置epsilon到初始值和episode计数"""
        self.epsilon = self.initial_epsilon
        self.episode_count = 0

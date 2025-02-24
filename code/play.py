import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        # 增加网络容量以处理更大的状态空间
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size  # 现在应该是35 (5个基础状态 + 10个陨石各3个状态)
        self.action_size = action_size
        self.memory = deque(maxlen=10000)  # 经验回放缓冲区
        self.gamma = 0.99
        
        # 调整初始探索率为 0.3（原为 0.9），让网络更快地通过 Q 值行动
        self.initial_epsilon = 0.3
        
        self.epsilon = self.initial_epsilon
        self.epsilon_min = 0.05      # 维持较低但不为零的探索
        self.epsilon_decay = 0.995   # 调整衰减速度
        
        self.episode_count = 0
        self.learning_rate = 0.0005
        self.target_update_frequency = 10
        self.update_counter = 0
        
        # 将奖励缩放因子从 0.1 调整为 1.0，避免奖励被过度压缩
        self.reward_scale = 1.0
        
        self.batch_size = 64
        self.min_experiences = 64
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 主网络和目标网络
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())  # 初始化目标网络
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def remember(self, state, action, reward, next_state, done, training=True):
        """
        存储经验到回放缓冲区
        training: 是否处于训练阶段（在本例中暂未额外使用此参数）
        """
        # 使用 reward_scale 缩放奖励
        scaled_reward = reward * self.reward_scale
        self.memory.append((state, action, scaled_reward, next_state, done))

    def act(self, state):
        """
        根据当前状态选择动作
        若随机数小于 epsilon，则执行随机动作
        否则使用网络输出的 Q 值取 argmax
        """
        if np.random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)
            return torch.argmax(q_values[0]).item()

    def replay(self, batch_size):
        """
        从经验回放中采样并训练一次
        """
        if len(self.memory) < self.min_experiences:
            return None  # 返回 None 表示还没达到最小经验量，不更新

        # 从经验回放缓冲区采样
        minibatch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([i[0] for i in minibatch]).to(self.device)
        actions = torch.LongTensor([i[1] for i in minibatch]).to(self.device)
        rewards = torch.FloatTensor([i[2] for i in minibatch]).to(self.device)
        next_states = torch.FloatTensor([i[3] for i in minibatch]).to(self.device)
        dones = torch.FloatTensor([i[4] for i in minibatch]).to(self.device)

        # 当前 Q 值
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 目标 Q 值 (Double DQN)
        with torch.no_grad():
            # 在线网络选动作
            next_actions = self.model(next_states).max(1)[1]
            # 目标网络估计下一个状态的 Q 值
            next_q_values = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            # 如果回合结束(done=1)则没有后续回报
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # 计算 Smooth L1 损失
        loss = nn.SmoothL1Loss()(current_q_values, target_q_values)
        loss_value = loss.item()

        # 反向传播更新
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()

        # 更新目标网络
        self.update_counter += 1
        if self.update_counter % self.target_update_frequency == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        return loss_value

    def update_target_model(self):
        """
        手动更新目标网络
        """
        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, name):
        """
        保存模型参数
        """
        torch.save(self.model.state_dict(), name)

    def load(self, name):
        """
        加载模型参数并设置为评估模式
        """
        self.model.load_state_dict(torch.load(name, map_location=self.device))
        self.model.eval()

    def update_epsilon(self):
        """
        在每个 episode 结束时调用 (训练过程)
        让 epsilon 按照 epsilon_decay 进行衰减，但不会低于 epsilon_min
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(
                self.epsilon_min,
                self.initial_epsilon * (self.epsilon_decay ** self.episode_count)
            )
        self.episode_count += 1

    def reset_epsilon(self):
        """
        在开始新一轮训练前或需要重置时调用
        将 epsilon 重置回初始值，并清零 episode 统计
        """
        self.epsilon = self.initial_epsilon
        self.episode_count = 0
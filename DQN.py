import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.bn1 = nn.LayerNorm(128)        # normalization sau lớp fc1
        self.dropout1 = nn.Dropout(p=0.2)       # Dropout sau lớp fc1
        
        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.LayerNorm(128)        # Batch normalization sau lớp fc2
        self.dropout2 = nn.Dropout(p=0.2)       # Dropout sau lớp fc2
        
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        
        return self.fc3(x)


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, similarity_threshold=0.1):
        self.capacity = capacity
        self.alpha = alpha
        self.similarity_threshold = similarity_threshold  # Ngưỡng xác định trạng thái giống nhau
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.buffer else 1.0
        state_array = np.array(state)

        # Kiểm tra trạng thái có giống với trạng thái đã có không
        for idx in range(max(0, len(self.buffer) - 20), len(self.buffer)):
            s, a, r, ns, d = self.buffer[idx]
            if np.linalg.norm(state_array - np.array(s)) < self.similarity_threshold:
                self.priorities[idx] = max(self.priorities[idx], max_priority)
                return  
        # Nếu buffer chưa đầy, thêm trạng thái mới vào
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            return [], [], [], [], []

        priorities = self.priorities[: len(self.buffer)] ** self.alpha
        probs = priorities / priorities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(*samples)

        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
            torch.tensor(weights, dtype=torch.float32),
            indices,
        )

    def update_priorities(self, batch_indices, td_errors, epsilon=1e-5):
        for idx, error in zip(batch_indices, td_errors):
            self.priorities[idx] = (abs(error) + epsilon)

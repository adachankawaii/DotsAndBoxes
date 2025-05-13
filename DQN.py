import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import copy
import math
from collections import defaultdict
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

        is_duplicate = False
        recent_range = range(max(0, len(self.buffer) - 20), len(self.buffer))

        for idx in recent_range:
            old_state, old_action, old_reward, old_next_state, old_done = self.buffer[idx]
            old_state_array = np.array(old_state)

            # So sánh trạng thái hiện tại với trạng thái cũ
            state_similar = np.linalg.norm(state_array - old_state_array) < self.similarity_threshold
            action_same = (action == old_action)

            # Nếu giống nhau cả trạng thái và hành động → cập nhật priority
            if state_similar and action_same:
                self.priorities[idx] = max(self.priorities[idx], max_priority)
                is_duplicate = True
                break  # Chỉ cần 1 là đủ

        if is_duplicate:
            return  # Không thêm bản ghi mới nếu trùng

        # Nếu không trùng → thêm vào buffer như bình thường
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

def train_dqn(env, champion_model, challenger_model, num_episodes=10000):
    champ_optimizer = optim.Adam(champion_model.parameters(), lr=1e-4, weight_decay=1e-5)
    chall_optimizer = optim.Adam(challenger_model.parameters(), lr=1e-4, weight_decay=1e-5)
    champ_memory = PrioritizedReplayBuffer(100000)
    chall_memory = PrioritizedReplayBuffer(100000)
    
    epsilon = 1.0
    epsilon_min = 0.01
    gamma = 0.99
    beta = 0.4
    champ_target = copy.deepcopy(champion_model)
    chall_target = copy.deepcopy(challenger_model)

    target_update_frequency = 1000
    champion_update_frequency = 100

    champion_win_threshold = 0.60

    win_count_champ = 0
    win_count_chall = 0
    sum_loss = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward_champ = 0
        total_reward_chall = 0
        next_state = state
        while not done:
            _, full_moves = env.get_possible_moves()
            valid_action_indices = [i for i, move in enumerate(full_moves) if move is not None]
           
            if env.turn == 'bot':
                model, target_model, optimizer = champion_model, champ_target, champ_optimizer
                shared_memory = champ_memory
            else:
                model, target_model, optimizer = challenger_model, chall_target, chall_optimizer
                shared_memory = chall_memory
            

            rnd = random.random()
            if rnd < epsilon:
                # 1. Random action
                action = random.choice(valid_action_indices)

            else:
                # 3. Softmax sampling from DQN
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    q_values = model(state_tensor).squeeze(0)
                    mask = torch.tensor([m is not None for m in full_moves], dtype=torch.bool)
                    q_values[~mask] = -float('inf')
                    T = max(0.05, epsilon**0.5)  # Ban đầu T ~ 1.0, giảm dần nhẹ hơn epsilon
                    probs = torch.nn.functional.softmax(q_values / T, dim=0)
                    action = torch.multinomial(probs, 1).item()

            turn_indicator = 0 if env.turn == 'bot' else 1
            next_state, reward, done = env.step(action, turn_indicator)
        
            shared_memory.push(state, action, reward, next_state, done)
            
            if env.turn == 'bot':
                total_reward_champ += reward
            else:
                total_reward_chall += reward

            state = next_state
            if len(shared_memory.buffer) > 128:
                batch = shared_memory.sample(128, beta)
                states, actions, rewards, next_states, dones, weights, indices = batch

                q_pred = model(states).gather(1, actions.unsqueeze(1)).squeeze()
                # Q = r + gamma * Q(s', a')
                # Target từ target model
                with torch.no_grad():
                    next_q_values = target_model(next_states)
                    next_actions = torch.argmax(model(next_states), dim=1)
                    next_q = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze()
                    q_target = rewards + gamma * next_q * (1 - dones)

                td_errors = q_pred - q_target
                per_sample_loss = F.mse_loss(q_pred, q_target, reduction='none')
                loss = (per_sample_loss * weights).mean()
                sum_loss.append(loss.item())

                optimizer.zero_grad()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                loss.backward()
                optimizer.step()

                shared_memory.update_priorities(indices, td_errors.abs().detach().numpy())
            
            champ_score = np.sum(env.boxes == 'bot')
            chall_score = np.sum(env.boxes == 'player')
        if champ_score > chall_score:
            win_count_champ += 1
        elif chall_score > champ_score:
            win_count_chall += 1

        epsilon = max(epsilon_min, epsilon_min + (1.0 - epsilon_min) * (1 - episode / (num_episodes*0.8)))

        beta = min(1.0, beta + 0.0001)
        
        soft_update(champ_target, champion_model, tau=1e-3)
        soft_update(chall_target, challenger_model, tau=1e-3)

        if (episode + 1) % champion_update_frequency == 0 :
            win_rate = win_count_chall / (win_count_champ + win_count_chall)
            print(f"Episode {episode}: Epsilon = {epsilon:.4f}, Loss = {np.average(sum_loss):.4f}, Memory = {len(shared_memory.buffer)}")
            print(f"After {episode+1} episodes: Challenger win rate = {win_rate:.2f}: Champ = {win_count_champ}; Chall = {win_count_chall}")
            if win_rate > champion_win_threshold:
                print("Updating champion with challenger weights...")
                soft_update(champion_model, challenger_model, tau=0.3)
                soft_update(champ_target, champion_model, tau=1)

                win_count_champ = win_count_chall = 0
            else:
                win_count_champ = win_count_chall = 0
        sum_loss = []
    
    torch.save(champion_model.state_dict(), "dqn_model.pt")
    return champion_model


def soft_update(target, source, tau=0.1):
    # Retained for backward compatibility but not used in the new update scheme.
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)


def get_best_dqn_move(env, model):
    state_base = env.get_state()
    possible_moves, full_moves = env.get_possible_moves()
    valid_indices = [i for i, m in enumerate(full_moves) if m is not None]

    with torch.no_grad():
        state_tensor = torch.tensor(state_base, dtype=torch.float32)
        q_values = model(state_tensor)
        mask = torch.tensor([m is not None for m in full_moves], dtype=torch.bool)
        q_values[~mask] = -float('inf')

    # Lấy nước đi DQN chọn
    dqn_action = torch.argmax(q_values).item()
    dqn_move = full_moves[dqn_action]

    # Kiểm tra nước DQN chọn có nguy hiểm không
    dqn_state = env.clone()
    extra = dqn_state.apply_move(dqn_move, 'bot')

    dqn_move_is_safe = True
    if not extra:
        opp_moves, _ = dqn_state.get_possible_moves()
        for opp_move in opp_moves:
            opp_clone = dqn_state.clone()
            if opp_clone.apply_move(opp_move, 'player'):
                dqn_move_is_safe = False
                break

    # Nếu DQN chọn nước đi an toàn → dùng luôn
    if dqn_move_is_safe and mask[dqn_action]:
        return dqn_move

    # Nếu DQN chọn nước nguy hiểm → tìm nước đi an toàn bằng tham lam
    best_safe_move = None
    best_score = -float('inf')
    has_safe_move = False

    for idx in valid_indices:
        move = full_moves[idx]
        state_clone = env.clone()
        extra = state_clone.apply_move(move, 'bot')

        # Nếu có thể ăn điểm thì ưu tiên
        if extra:
            return move

        # Kiểm tra có để đối thủ ăn điểm không
        opp_possible_moves, _ = state_clone.get_possible_moves()
        safe = True
        for opp_move in opp_possible_moves:
            opp_clone = state_clone.clone()
            if opp_clone.apply_move(opp_move, 'player'):
                safe = False
                break

        if safe:
            has_safe_move = True
            score = state_clone.evaluate_state(0)
            if score > best_score:
                best_score = score
                best_safe_move = move

    # Nếu có nước đi an toàn → chọn theo tham lam
    if has_safe_move:
        return best_safe_move

    # Nếu tất cả nước đi đều nguy hiểm → dùng DQN quyết định
    if mask[dqn_action]:
        return dqn_move

    # Nếu nước đi DQN không hợp lệ (hiếm khi xảy ra) → chọn ngẫu nhiên
    return random.choice(possible_moves)

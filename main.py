import tkinter as tk
import math
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np

from DQN import DQN, PrioritizedReplayBuffer
import numpy as np
# --- Các hằng số toàn cục ---
CELL_SIZE = 60  # Kích thước mỗi ô (pixel)
PADDING = 20  # Khoảng cách biên của canvas
DOT_RADIUS = 6  # Bán kính điểm


# --- Lớp lưu trữ trạng thái bàn chơi ---
class GameState:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        # Ma trận các đường ngang: (rows+1) x cols
        self.horiz = np.array([[False for _ in range(cols)] for _ in range(rows + 1)])
        # Ma trận các đường dọc: rows x (cols+1)
        self.vert = np.array([[False for _ in range(cols + 1)] for _ in range(rows)])
        # Ma trận ô vuông: rows x cols, None nếu chưa ai chiếm
        self.boxes = np.array([[None for _ in range(cols)] for _ in range(rows)])
        self.state_size = (rows + 1) * cols + rows * (cols + 1)
        self.action_size = self.state_size
        self.player_score = 0
        self.bot_score = 0
        self.turn = 'player'  # lượt đầu tiên: người chơi
    
    def clone(self):
        new_state = GameState(self.rows, self.cols)
        new_state.horiz = copy.deepcopy(self.horiz)
        new_state.vert = copy.deepcopy(self.vert)
        new_state.boxes = copy.deepcopy(self.boxes)
        new_state.player_score = self.player_score
        new_state.bot_score = self.bot_score
        new_state.turn = self.turn
        return new_state
     
    def step(self, action, turn):
        possible_moves, full_moves = self.get_possible_moves()
        move = full_moves[action]
        

        # Apply move based on turn
        if turn == 0:
            extra_move = self.apply_move(move, 'bot')
        else:
            extra_move = self.apply_move(move, 'player')
        
        reward = self.evaluate_state(turn if extra_move else (1-turn))   # Đánh giá trạng thái sau khi thực hiện nước đi
        
        if not extra_move:
            if turn == 0:
                self.turn = 'player'
                reward += 0.5
            else:
                self.turn = 'bot'
                reward += 0.5
            turn = 1 - turn  # Chuyển lượt cho người chơi khác
        # Check if the game is over
        done = self.is_game_over()

        if done:
            # Final reward based on outcome
            if turn == 0:
                bot_score = np.sum(self.boxes == 'bot')
                player_score = np.sum(self.boxes == 'player')
            else:
                bot_score = np.sum(self.boxes == 'player')
                player_score = np.sum(self.boxes == 'bot')
            
            if bot_score > player_score:
                reward += 10.0
            elif bot_score < player_score:
                reward += -10.0

        return self.get_normalized_state(), reward, done

    def get_normalized_state(self):
        state = self.get_state()
        max_value = np.max(state)
        return state / max_value if max_value > 0 else state


    def is_game_over(self):
        # Trò chơi kết thúc khi tất cả các đường đã được vẽ
        for row in self.horiz:
            if False in row:
                return False
        for row in self.vert:
            if False in row:
                return False
        return True

    def get_possible_moves(self):
        moves = []
        full_moves = []
        # Các bước đi: đường ngang định dạng ("h", i, j)
        for i in range(len(self.horiz)):
            for j in range(len(self.horiz[0])):
                if not self.horiz[i][j]:
                    moves.append(("h", i, j))
                    full_moves.append(("h", i, j))
                else:
                    full_moves.append(None)
        # Các bước đi: đường dọc định dạng ("v", i, j)
        for i in range(len(self.vert)): 
            for j in range(len(self.vert[0])):
                if not self.vert[i][j]:
                    moves.append(("v", i, j))
                    full_moves.append(("v", i, j))
                else:
                    full_moves.append(None)
        return moves, full_moves
    def set_state_from_tensor(self, state):
        """
        Cập nhật trạng thái của GameState từ một danh sách hoặc tensor.
        """
        if isinstance(state, torch.Tensor):  # Nếu đầu vào là tensor, chuyển thành list
            state = state.tolist()

        if len(state) != self.state_size:
            raise ValueError(f"Invalid state size: expected {self.state_size}, got {len(state)}")

        # Tách các phần của state theo đúng cấu trúc của game
        horiz_size = (self.rows + 1) * self.cols
        vert_size = self.rows * (self.cols + 1)

        flat_horiz = state[:horiz_size]
        flat_vert = state[horiz_size:horiz_size + vert_size]

        self.horiz = np.array(flat_horiz, dtype=bool).reshape((self.rows + 1, self.cols))
        self.vert = np.array(flat_vert, dtype=bool).reshape((self.rows, self.cols + 1))

    def get_state(self):
        return np.concatenate((self.horiz.flatten(), self.vert.flatten())).astype(np.float32)
    def reset(self):
        self.horiz.fill(False)
        self.vert.fill(False)
        self.boxes.fill(None)
        self.player_score = 0
        self.bot_score = 0
        return self.get_state()
    def apply_move(self, move, player):
        """
        Thực hiện nước đi:
         - Đánh dấu đường được vẽ.
         - Kiểm tra xem có ô nào được “chốt” sau nước đi không.
         - Nếu có ô được chốt, cộng điểm và trả về extra_move=True.
        """
        extra_move = False
        move_type, i, j = move
        if move_type == "h":
            self.horiz[i][j] = True
        else:
            self.vert[i][j] = True

        completed_box = False

        if move_type == "h":
            # Kiểm tra ô phía trên (nếu có)
            if i > 0:
                if (self.horiz[i - 1][j] and self.vert[i - 1][j] and
                        self.vert[i - 1][j + 1] and self.horiz[i][j]):
                    if self.boxes[i - 1][j] is None:
                        self.boxes[i - 1][j] = player
                        completed_box = True
                        if player == 'player':
                            self.player_score += 1
                        else:
                            self.bot_score += 1
            # Kiểm tra ô phía dưới (nếu có)
            if i < self.rows:
                if (self.horiz[i][j] and self.vert[i][j] and
                        self.vert[i][j + 1] and self.horiz[i + 1][j]):
                    if self.boxes[i][j] is None:
                        self.boxes[i][j] = player
                        completed_box = True
                        if player == 'player':
                            self.player_score += 1
                        else:
                            self.bot_score += 1
        else:  # Nước đi là đường dọc
            # Kiểm tra ô bên trái (nếu có)
            if j > 0:
                if (self.vert[i][j - 1] and self.horiz[i][j - 1] and
                        self.horiz[i + 1][j - 1] and self.vert[i][j]):
                    if self.boxes[i][j - 1] is None:
                        self.boxes[i][j - 1] = player
                        completed_box = True
                        if player == 'player':
                            self.player_score += 1
                        else:
                            self.bot_score += 1
            # Kiểm tra ô bên phải (nếu có)
            if j < self.cols:
                if (self.vert[i][j] and self.horiz[i][j] and
                        self.horiz[i + 1][j] and self.vert[i][j + 1]):
                    if self.boxes[i][j] is None:
                        self.boxes[i][j] = player
                        completed_box = True
                        if player == 'player':
                            self.player_score += 1
                        else:
                            self.bot_score += 1

        if completed_box:
            extra_move = True
        return extra_move
    def evaluate_state(self, turn):
        """
        Đánh giá trạng thái hiện tại của trò chơi theo quan điểm của bot.
        Trả về một số thực, số dương nếu bot có lợi, số âm nếu bot gặp bất lợi.
        """
        score = self.player_score - self.bot_score  # Hiệu số điểm hiện tại
        
        if turn == 0:
            bot = 'bot'
            score = -score  # Đảo ngược điểm số nếu không phải lượt của bot
        else:
            bot = 'player'
             
        immediate_boxes = 0.0  # Số ô có thể hoàn thành ngay lập tức
        dangerous_boxes = 0.0  # Số ô nguy hiểm có thể bị chiếm bởi đối thủ
        
        for r in range(self.rows):
            for c in range(self.cols):
                edges = sum([self.horiz[r][c], self.horiz[r+1][c], self.vert[r][c], self.vert[r][c+1]])
                
                if edges == 3:  # Ô có 3 cạnh đã được đánh dấu -> Có thể hoàn thành ngay
                    if self.turn == bot:
                        immediate_boxes = 1.0
                    else:
                        dangerous_boxes = 1.0  # Nếu đến lượt người chơi, họ có thể lấy ô này
        score += immediate_boxes * 1.0 - dangerous_boxes * 2.0  # Tăng điểm cho ô có thể hoàn thành ngay, giảm điểm cho ô nguy hiểm
        return score

# --- Hàm đánh giá trạng thái (heuristic) ---
def evaluate_state(state):
    # Hiệu số điểm hiện tại
    score = state.bot_score - state.player_score
    # Cân nhắc thêm “tiềm năng” chiếm ô: duyệt qua các ô chưa hoàn thành
    for i in range(state.rows):
        for j in range(state.cols):
            if state.boxes[i][j] is None:
                drawn = 0
                if state.horiz[i][j]:
                    drawn += 1
                if state.horiz[i + 1][j]:
                    drawn += 1
                if state.vert[i][j]:
                    drawn += 1
                if state.vert[i][j + 1]:
                    drawn += 1
                if drawn == 3:
                    bonus = 2.0
                    score += bonus if state.turn == 'bot' else -bonus
                elif drawn == 2:
                    bonus = 0.5
                    score += bonus if state.turn == 'bot' else -bonus
    return score



# --- Thuật toán minimax với alpha-beta pruning ---
def minimax(state, depth, alpha, beta, maximizing_player):
    if depth == 0 or state.is_game_over():
        return evaluate_state(state), None

    possible_moves,_ = state.get_possible_moves()
    best_move = None

    if maximizing_player:
        max_eval = -math.inf
        for move in possible_moves:
            new_state = state.clone()
            extra = new_state.apply_move(move, 'bot')
            if extra:
                eval_score, _ = minimax(new_state, depth - 1, alpha, beta, True)
            else:
                new_state.turn = 'player'
                eval_score, _ = minimax(new_state, depth - 1, alpha, beta, False)
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = math.inf
        for move in possible_moves:
            new_state = state.clone()
            extra = new_state.apply_move(move, 'player')
            if extra:
                eval_score, _ = minimax(new_state, depth - 1, alpha, beta, False)
            else:
                new_state.turn = 'bot'
                eval_score, _ = minimax(new_state, depth - 1, alpha, beta, True)
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval, best_move
    
def action_to_move(action, grid_size):
    num_horiz = (grid_size + 1) * grid_size  # Số đường ngang trong bảng

    if action < num_horiz:
        # Nếu action thuộc phần đường ngang
        i = action // grid_size
        j = action % grid_size
        return ("h", i, j)
    else:
        # Nếu action thuộc phần đường dọc
        action -= num_horiz  # Điều chỉnh index để bắt đầu từ 0
        i = action // (grid_size + 1)
        j = action % (grid_size + 1)
        return ("v", i, j)
    
def move_to_action(move, grid_size):
    move_type, i, j = move
    if move_type == "h":
        return i * grid_size + j  # Chỉ số trong phần `horiz`
    elif move_type == "v":
        return (grid_size + 1) * grid_size + i * (grid_size + 1) + j  # Chỉ số trong phần `vert`

def train_dqn(env, champion_model, challenger_model, num_episodes=10000):
    champ_optimizer = optim.Adam(champion_model.parameters(), lr=1e-4, weight_decay=1e-5)
    chall_optimizer = optim.Adam(challenger_model.parameters(), lr=1e-4, weight_decay=1e-5)
    champ_memory = PrioritizedReplayBuffer(100000)
    chall_memory = PrioritizedReplayBuffer(100000)
    create_chart(challenger_model, "Random")
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
    
   
    return champion_model


def soft_update(target, source, tau=0.1):
    # Retained for backward compatibility but not used in the new update scheme.
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)


def get_best_dqn_move(env, model):
    state_base = env.get_state()
    possible_moves, full_moves = env.get_possible_moves()
    valid_indices = [i for i, m in enumerate(full_moves) if m is not None]

    best_move = None
    best_score = -float('inf')

    with torch.no_grad():
        state_tensor = torch.tensor(state_base, dtype=torch.float32)
        q_values = model(state_tensor)
        mask = torch.tensor([m is not None for m in full_moves], dtype=torch.bool)
        q_values[~mask] = -float('inf')

    all_moves_dangerous = True

    for idx in valid_indices:
        move = full_moves[idx]
        state_clone = env.clone()
        extra = state_clone.apply_move(move, 'bot')

        # Kiểm tra xem nước đi này có để đối thủ ăn ngay không
        safe = True
        if not extra:
            opp_possible_moves, _ = state_clone.get_possible_moves()
            for opp_move in opp_possible_moves:
                opp_clone = state_clone.clone()
                if opp_clone.apply_move(opp_move, 'player'):
                    safe = False
                    break
        if safe:
            all_moves_dangerous = False

        # Tích hợp cả Q-value và an toàn
        score = q_values[idx].item()
        if extra:
            score += 1000
        elif safe:
            score += 10  # Ưu tiên nhẹ cho nước đi an toàn

        if score > best_score:
            best_score = score
            best_move = move

    # Nếu tất cả đều nguy hiểm, để DQN tự quyết định
    if all_moves_dangerous:
        action = torch.argmax(q_values).item()
        if not mask[action]:
            return random.choice(possible_moves)
        return full_moves[action]

    return best_move


import matplotlib.pyplot as plt

def simulate_game(model, rows=3, cols=3, opponent_type="random"):
    """
    Simulate a game where both the bot and its opponent select moves randomly.
    The bot is marked as "bot" and the opponent as "player".
    """
    state_obj = GameState(rows, cols)
    state_obj.reset()
    while not state_obj.is_game_over():
        if state_obj.turn == "bot":
            # Bot uses minimax. If minimax returns no move, fallback to random choice.
            move = get_best_dqn_move(state_obj, model)
        else:
            if opponent_type == "Random":
                possible_moves, _ = state_obj.get_possible_moves()
                move = random.choice(possible_moves)
            elif opponent_type == "Minimax":
                _, move = minimax(state_obj, 2, -math.inf, math.inf, True)
                if move is None:
                    possible_moves, _ = state_obj.get_possible_moves()
                    move = random.choice(possible_moves)
            elif opponent_type == "Random Player":
                #Code tham lam trong do choi random neu khong co nuoc an ngay, neu co nuoc an ngay thi danh nuoc do
                possible_moves, _ = state_obj.get_possible_moves()
                move = random.choice(possible_moves)
                for m in possible_moves:
                    state_clone = state_obj.clone()
                    extra = state_clone.apply_move(m, "player")
                    if extra:
                        move = m
                        break
            elif opponent_type == "Trained Player":
                move = None
                # Code tham lam giong nhu random_player, nhung nuoc di duoc chon khong dua bot vao trang thai nguy hiem
                if random.random() < 0.95:
                    best_move = None
                    best_score = -float('inf')
                    possible_moves, _ = state_obj.get_possible_moves()
                    for m in possible_moves:
                        state_clone = state_obj.clone()
                        extra = state_clone.apply_move(m, "player")
                        if extra:
                            score = 1000
                        else:
                            score = evaluate_state(state_clone)
                        if score > best_score:
                            best_score = score
                            best_move = m
                    move = best_move
                else:
                    possible_moves, _ = state_obj.get_possible_moves()
                    move = random.choice(possible_moves)
        extra = state_obj.apply_move(move, state_obj.turn)
        if not extra:
            state_obj.turn = "player" if state_obj.turn == "bot" else "bot"
    
    bot_score = np.sum(state_obj.boxes == "bot")
    player_score = np.sum(state_obj.boxes == "player")
    
    if bot_score > player_score:
        return 1  # Bot wins
    elif bot_score < player_score:
        return 0  # Opponent wins
    else:
        return 0.5  # Tie

def create_chart(model, bot_type):
    """
    Create a chart showing the win rate over 100 simulated games.
    The bot_type parameter determines which simulation to run:
        - "DQN": uses the provided DQN model via get_best_dqn_move.
        - "Minimax": uses the minimax algorithm.
        - "Random": all moves are chosen randomly.
    """
    win_rate = 0.0
    win_rates = []
    num_games = 100

    for game in range(1, num_games + 1):
        
        result = simulate_game(model, 3, 3, opponent_type=bot_type)
        
        win_rate += result
        current_rate = win_rate / game
        win_rates.append(current_rate)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_games + 1), win_rates)
    plt.xlabel("Number of Games")
    plt.ylabel("Bot Win Rate")
    plt.title(f"{bot_type} Bot Win Rate Over {num_games} Games")
    plt.grid(True)
    plt.ylim(-0.2, 1.2)
    plt.show()

# Create a DQN model instance (for a 3x3 board, state and action sizes are both 24)



# --- Giao diện trò chơi ---
# GameFrame là khung chứa giao diện của bàn chơi
class GameFrame(tk.Frame):
    def __init__(self, parent, app, rows, cols, difficulty, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        
        self.app = app
        self.rows = rows
        self.cols = cols
        self.difficulty = difficulty
        
        self.state = GameState(rows, cols)
        self.canvas_width = cols * CELL_SIZE + 4 * PADDING
        self.canvas_height = rows * CELL_SIZE + 4 * PADDING
        self.canvas = tk.Canvas(self, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas.pack(expand=True)
        self.canvas.bind("<Button-1>", self.on_click)
        self.model = None
        self.status_label = tk.Label(self, text="Lượt của bạn")
        self.status_label.pack()

        self.control_frame = tk.Frame(self)
        self.control_frame.pack(pady=5)
        self.menu_button = tk.Button(self.control_frame, text="Main Menu", command=self.go_to_menu)
        self.menu_button.pack(side="left", padx=5)
        self.restart_button = tk.Button(self.control_frame, text="Restart", command=self.restart_game)
        self.restart_button.pack(side="left", padx=5)

        self.animated_boxes = set()
        self.background_rect = self.canvas.create_rectangle(0, 0, self.canvas_width, self.canvas_height, fill="white",
                                                            outline="")
        self.draw_board()

    def go_to_menu(self):
        self.app.show_start_menu()

    def restart_game(self):
        self.state.reset()
        self.app.show_game(self.rows, self.cols, self.difficulty)

    def draw_board(self):
        self.canvas.delete("all")
        # Vẽ các ô vuông đã chiếm
        for i in range(self.rows):
            for j in range(self.cols):
                if self.state.boxes[i][j] is not None and not hasattr(self, 'animated_boxes'):
                    self.animated_boxes = set()
                if self.state.boxes[i][j] is not None and (i, j) not in self.animated_boxes:
                    self.animated_boxes.add((i, j))
                    self.animate_box(i, j, self.state.boxes[i][j])
                elif self.state.boxes[i][j] is not None:
                    self.draw_final_box(i, j, self.state.boxes[i][j])
        # Vẽ các điểm (dots)
        for i in range(self.rows + 1):
            for j in range(self.cols + 1):
                x = PADDING + j * CELL_SIZE
                y = PADDING + i * CELL_SIZE
                self.canvas.create_oval(x - DOT_RADIUS, y - DOT_RADIUS,
                                        x + DOT_RADIUS, y + DOT_RADIUS,
                                        fill="black")
        # Vẽ các đường ngang đã vẽ
        for i in range(len(self.state.horiz)):
            for j in range(len(self.state.horiz[0])):
                x1 = PADDING + j * CELL_SIZE
                y1 = PADDING + i * CELL_SIZE
                x2 = PADDING + (j + 1) * CELL_SIZE
                if self.state.horiz[i][j]:
                    self.canvas.create_line(x1, y1, x2, y1, width=4, fill="black")
                else:
                    self.canvas.create_line(x1, y1, x2, y1, width=4, fill="gray", dash=(4, 4))

        for i in range(len(self.state.vert)):
            for j in range(len(self.state.vert[0])):
                x1 = PADDING + j * CELL_SIZE
                y1 = PADDING + i * CELL_SIZE
                y2 = PADDING + (i + 1) * CELL_SIZE
                if self.state.vert[i][j]:
                    self.canvas.create_line(x1, y1, x1, y2, width=4, fill="black")
                else:
                    self.canvas.create_line(x1, y1, x1, y2, width=4, fill="gray", dash=(4, 4))

    def animate_box(self, i, j, owner, progress=0):
        if progress > 0.5:
            self.draw_final_box(i, j, owner)
            return
        x1 = PADDING + j * CELL_SIZE
        y1 = PADDING + i * CELL_SIZE
        x2 = PADDING + (j + 1) * CELL_SIZE
        y2 = PADDING + (i + 1) * CELL_SIZE
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        expand_x = (x2 - x1) * progress / 2
        expand_y = (y2 - y1) * progress / 2
        color = "lightblue" if owner == 'player' else "pink"
        self.canvas.create_rectangle(mid_x - expand_x, mid_y - expand_y, mid_x + expand_x, mid_y + expand_y, fill=color,
                                     outline="")
        self.after(20, self.animate_box, i, j, owner, progress + 0.2)

    def draw_final_box(self, i, j, owner):
        x1 = PADDING + j * CELL_SIZE + 2
        y1 = PADDING + i * CELL_SIZE + 2
        x2 = PADDING + (j + 1) * CELL_SIZE - 2
        y2 = PADDING + (i + 1) * CELL_SIZE - 2
        color = "lightblue" if owner == 'player' else "pink"
        self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="black")

    def animate_line(self, x1, y1, x2, y2, step=5, progress=0, color="black"):
        if progress >= 1:
            self.canvas.create_line(x1, y1, x2, y2, width=4, fill="black")
            return
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        xm1 = mid_x - (mid_x - x1) * progress
        ym1 = mid_y - (mid_y - y1) * progress
        xm2 = mid_x + (x2 - mid_x) * progress
        ym2 = mid_y + (y2 - mid_y) * progress
        self.canvas.create_line(xm1, ym1, xm2, ym2, width=4, fill=color)
        self.after(20, self.animate_line, x1, y1, x2, y2, step, progress + 0.2, color)

    def on_click(self, event):
        if self.state.turn != 'player':
            return
        x, y = event.x, event.y
        clicked_move = None
        tolerance = 10
        # Kiểm tra vùng click cho đường ngang
        for i in range(len(self.state.horiz)):
            for j in range(len(self.state.horiz[0])):
                if not self.state.horiz[i][j]:
                    x1 = PADDING + j * CELL_SIZE
                    y1 = PADDING + i * CELL_SIZE
                    x2 = PADDING + (j + 1) * CELL_SIZE
                    mx = (x1 + x2) / 2
                    my = y1
                    if abs(x - mx) < CELL_SIZE / 2 and abs(y - my) < tolerance:
                        clicked_move = ("h", i, j)
                        break
            if clicked_move:
                break
        # Nếu chưa click được vào đường ngang, kiểm tra đường dọc
        if not clicked_move:
            for i in range(len(self.state.vert)):
                for j in range(len(self.state.vert[0])):
                    if not self.state.vert[i][j]:
                        x1 = PADDING + j * CELL_SIZE
                        y1 = PADDING + i * CELL_SIZE
                        y2 = PADDING + (i + 1) * CELL_SIZE
                        mx = x1
                        my = (y1 + y2) / 2
                        if abs(x - mx) < tolerance and abs(y - my) < CELL_SIZE / 2:
                            clicked_move = ("v", i, j)
                            break
                if clicked_move:
                    break
        if clicked_move:
            extra = self.state.apply_move(clicked_move, 'player')
            self.state.apply_move(clicked_move, 'player')
            color = "blue" if self.state.turn == 'player' else "red"
            
            if clicked_move[0] == "h":
                x1, y1, x2, y2 = PADDING + clicked_move[2] * CELL_SIZE, PADDING + clicked_move[
                    1] * CELL_SIZE, PADDING + (clicked_move[2] + 1) * CELL_SIZE, PADDING + clicked_move[1] * CELL_SIZE
            else:
                x1, y1, x2, y2 = PADDING + clicked_move[2] * CELL_SIZE, PADDING + clicked_move[1] * CELL_SIZE, PADDING + \
                                 clicked_move[2] * CELL_SIZE, PADDING + (clicked_move[1] + 1) * CELL_SIZE
            self.animate_line(x1, y1, x2, y2, color=color)
            self.after(500, self.draw_board)
            if self.state.is_game_over():
                self.end_game()
                return
            if extra:
                self.status_label.config(text="Bạn chiếm được ô, tiếp tục lượt của bạn!")
            else:
                self.state.turn = 'bot'
                self.status_label.config(text="Bot đang tính toán...")

                self.after(500, self.bot_move)
        else:
            print("Click không hợp lệ.")

    def bot_move(self):
        # Bot tính toán nước đi bằng minimax với độ sâu đã chọn
        if self.difficulty == "Minimax":
            _, best_move = minimax(self.state, 2, -math.inf, math.inf, True)
        elif self.difficulty == "Genetic Algorithm":
            pass
        else:
            best_move = get_best_dqn_move(self.state, self.model)
        # best_move = get_best_dqn_move(self.state, self.model)
        if best_move is None:
            return
        
        extra = self.state.apply_move(best_move, 'bot')
        color = "blue" if self.state.turn == 'player' else "red"
        if best_move[0] == "h":
            x1, y1, x2, y2 = PADDING + best_move[2] * CELL_SIZE, PADDING + best_move[1] * CELL_SIZE, PADDING + (
                        best_move[2] + 1) * CELL_SIZE, PADDING + best_move[1] * CELL_SIZE
        else:
            x1, y1, x2, y2 = PADDING + best_move[2] * CELL_SIZE, PADDING + best_move[1] * CELL_SIZE, PADDING + \
                             best_move[2] * CELL_SIZE, PADDING + (best_move[1] + 1) * CELL_SIZE
        self.animate_line(x1, y1, x2, y2, color=color)
        self.after(500, self.draw_board)
        if self.state.is_game_over():
            self.end_game()
            return
        if extra:
            self.status_label.config(text="Bot chiếm được ô và được tiếp tục lượt!")
            self.after(500, self.bot_move)
        else:
            self.state.turn = 'player'
            self.status_label.config(text="Lượt của bạn")

    def end_game(self):
        if self.state.player_score > self.state.bot_score:
            result = "Bạn thắng!"
        elif self.state.player_score < self.state.bot_score:
            result = "Bot thắng!"
        else:
            result = "Hòa!"
        self.status_label.config(text=f"Trò chơi kết thúc. {result}")
        self.canvas.unbind("<Button-1>")


# --- Giao diện Start Menu ---
import tkinter as tk
from tkinter import ttk


class StartMenuFrame(tk.Frame):
    def __init__(self, parent, app, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.app = app

        # Tạo khung chứa toàn bộ menu để căn giữa
        container = tk.Frame(self, bg="#f0f0f0")
        container.pack(expand=True, padx=20, pady=20)

        # Tiêu đề
        self.title_label = tk.Label(container, text="Dots and Boxes", font=("Helvetica", 28, "bold"), bg="#f0f0f0")
        self.title_label.pack(pady=30)

        # Khung chọn kích thước bàn cờ
        size_frame = ttk.LabelFrame(container, text="Board Size", padding=10)
        size_frame.pack(pady=15, fill="x", padx=20)

        ttk.Label(size_frame, text="Size:").grid(row=0, column=0, padx=5, pady=5)
        self.rows_var = tk.IntVar(value=3)
        self.cols_var = tk.IntVar(value=3)  # Tạo biến riêng cho cols

        self.rows_spin = ttk.Spinbox(size_frame, from_=3, to=5, textvariable=self.rows_var, width=5, command=self.update_bot_options)
        self.rows_spin.grid(row=0, column=1, padx=5, pady=5)

        self.cols_var = self.rows_var  # Gán giá trị cho cols bằng giá trị của rows

        # Ràng buộc sự kiện khi giá trị thay đổi
        self.rows_var.trace_add("write", lambda *args: self.update_bot_options())
        self.cols_var.trace_add("write", lambda *args: self.update_bot_options())

        # Khung chọn độ khó
        diff_frame = ttk.LabelFrame(container, text="Bot Type", padding=10)
        diff_frame.pack(pady=15, fill="x", padx=20)
        
        self.difficulty_var = tk.StringVar(value="Minimax")
        self.diff_option = ttk.Combobox(diff_frame, textvariable=self.difficulty_var, state="readonly")
        self.diff_option.pack(pady=5, padx=10)
        
        self.update_bot_options()  # Gọi cập nhật giá trị ban đầu

        
        # Thêm phần trang trí
        self.info_label = tk.Label(container, text="Select board size and difficulty to start playing.",
                                   font=("Helvetica", 12), bg="#f0f0f0")
        self.info_label.pack(pady=10)

        # Thêm nút About
        self.about_button = tk.Button(container, text="About", font=("Helvetica", 10, "bold"), bg="#008CBA", fg="white",
                                      relief="ridge", bd=3, padx=10, pady=5, borderwidth=5, highlightthickness=2,
                                      command=self.show_about)
        self.about_button.pack(pady=5)


        # Nút bắt đầu
        self.start_button = tk.Button(container, text="Start Game", command=self.start_game,
                                      font=("Helvetica", 12, "bold"), bg="#4CAF50", fg="white", relief="ridge", bd=3,
                                      padx=10, pady=5, borderwidth=5, highlightthickness=2)
        self.start_button.pack(pady=20)
    def update_bot_options(self):
            """ Cập nhật danh sách thuật toán bot khi thay đổi kích thước """
            if self.rows_var.get() == 3:
                values = ["Minimax", "Genetic Algorithm", "Reinforcement Learning"]
            else:
                values = ["Minimax"]
            
            self.diff_option["values"] = values
            self.diff_option.current(0)  # Đặt lại giá trị mặc định là mục đầu tiên
    def start_game(self):
        global CELL_SIZE
        rows = self.rows_var.get()
        cols = self.cols_var.get()
        CELL_SIZE = 9 / max(rows, cols) * 60  # Tính toán lại kích thước ô
        difficulty = self.difficulty_var.get()
        self.app.show_game(rows, cols, difficulty)

    def show_about(self):
        about_window = tk.Toplevel(self)
        about_window.title("About")
        about_window.geometry("300x200")
        tk.Label(about_window, text="Dots and Boxes Game", font=("Helvetica", 14, "bold")).pack(pady=10)
        tk.Label(about_window, text="Developed with Python & Tkinter", font=("Helvetica", 10)).pack(pady=5)
        tk.Label(about_window, text="A product for researching algorithm", font=("Helvetica", 10)).pack(pady=5)

import os
# --- Lớp quản lý các khung giao diện (Frames) ---
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Dots and Boxes")
        self.resizable(False, False)
        # Lấy kích thước màn hình
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        size = screen_width // 2  # Kích thước cửa sổ bằng 1/2 chiều rộng màn hình
        self.model = None
        # Căn giữa cửa sổ
        x_position = (screen_width - size) // 2
        y_position = (screen_height - size - 50) // 2
        self.geometry(f"{size}x{size}+{x_position}+{y_position}")

        self.container = tk.Frame(self)
        self.container.pack(fill="both", expand=True)
        self.show_start_menu()

    def clear_container(self):
        for widget in self.container.winfo_children():
            widget.destroy()

    def show_start_menu(self):
        self.clear_container()
        start_menu = StartMenuFrame(self.container, self)
        start_menu.pack(fill="both", expand=True)

    def show_game(self, rows, cols, difficulty):
        self.clear_container()
        game_frame = GameFrame(self.container, self, rows, cols, difficulty)
        
        if difficulty == "Reinforcement Learning":
            MODEL_PATH = "dqn_model.pt"
            if self.model is None:
                if os.path.exists(MODEL_PATH):

                    model = DQN(game_frame.state.state_size, game_frame.state.action_size)
                    model.load_state_dict(torch.load(MODEL_PATH))
                    model.eval()
                else:
                    model1 = DQN(game_frame.state.state_size, game_frame.state.action_size)
                    model2 = DQN(game_frame.state.state_size, game_frame.state.action_size)
                    model = train_dqn(game_frame.state, model1, model2)
                    torch.save(model.state_dict(), MODEL_PATH)
                
                self.model = model

            game_frame.model = self.model
            create_chart(game_frame.model, "Minimax")
            create_chart(game_frame.model, "Random")
            create_chart(game_frame.model, "Random Player")
            create_chart(game_frame.model, "Trained Player")
        game_frame.state.reset()
        game_frame.pack(fill="both", expand=True)


if __name__ == "__main__":
    app = App()
    app.mainloop()
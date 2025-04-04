import random
import numpy as np
import os
import sys  # để flush stdout
import torch
import torch.nn as nn
import torch.optim as optim
import pygame
import time

# -------------------- CÀI ĐẶT BẢNG --------------------
square_size = 3      # bảng 3x3
width = square_size
height = square_size
num_actions = square_size * (square_size + 1) * 2  # 24 cạnh

class MyException(Exception):
    pass

# -------------------- HÀM XỬ LÝ TRÒ CHƠI --------------------
def init_Field():
    rows = np.zeros(shape=(width + 1, height), dtype=int)       # (4, 3)
    columns = np.zeros(shape=(width, height + 1), dtype=int)    # (3, 4)
    for i in range(width):
        # Đánh dấu viền ngoài = 1
        rows[0][i] = 1
        rows[width][i] = 1
        columns[i][0] = 1
        columns[i][width] = 1
    return rows, columns

def setField(rows, columns, h, w):
    if h >= height or w >= width:
        raise MyException("Invalid width or height")
    columns[h][w] = 1
    columns[h][w + 1] = 1
    rows[h][w] = 1
    rows[h + 1][w] = 1
    return rows, columns

def test_field_full(rows, columns, h, w):
    if w < len(columns) - 1:
        if columns[h][w] and columns[h][w+1] and rows[h][w] == 1 and rows[h+1][w] == 1:
            return True
        else:
            return False
    else:
        return False

def field_to_str(rows, columns):
    out = "  "
    for i in range(width):
        out += f"  {i}"
    out += "\n"
    for h in range(height+1):
        for i in range(2):
            for w in range(width+1):
                if i == 0 and w == 0:
                    out += "  "
                if i == 0 and w < len(rows[0]):
                    out += " --" if rows[h][w] == 1 else "   "
                elif i == 1 and h < len(columns):
                    if w == 0:
                        out += f" {h}" if h < 10 else str(h)
                    if columns[h][w] == 1:
                        out += "| x" if test_field_full(rows, columns, h, w) else "|  "
                    else:
                        out += "   "
            out += "\n"
    return out

def new_full_fields(field, which, h, w):
    """
    Kiểm tra xem hành động kẻ cạnh (which,h,w) có làm hoàn thiện
    ô (hoặc các ô lân cận) không. Trả về số ô mới hoàn thiện.
    """
    ret = 0
    # which==0 => horizontal, which==1 => vertical
    rows, columns = field[0], field[1]
    if which == 0:  # horizontal
        if h != 0:
            if columns[h-1][w] == 1 and rows[h-1][w] == 1 and columns[h-1][w+1] == 1:
                ret += 1
        if h != height:
            if columns[h][w] == 1 and rows[h+1][w] == 1 and columns[h][w+1] == 1:
                ret += 1
    else:  # vertical
        if w != 0:
            if rows[h][w-1] == 1 and columns[h][w-1] == 1 and rows[h+1][w-1] == 1:
                ret += 1
        if w != width:
            if rows[h][w] == 1 and columns[h][w+1] == 1 and rows[h+1][w] == 1:
                ret += 1
    return ret

def validate_move(field_arrays, array_i, h, w):
    try:
        return field_arrays[array_i][h][w] != 1
    except:
        return False

def game_over(game):
    """
    Kết thúc khi số cạnh đã kẻ (kể cả viền ban đầu) + các ô đã
    được "chiếm" = tổng số ô trong bảng (width * height).
    """
    obstacle_count = game.obstacle_count
    p1 = game.player1["Points"]
    p2 = game.player2["Points"]
    return (obstacle_count + p1 + p2) == (width * height)

# -------------------- LỚP GAME --------------------
class Game:
    def __init__(self):
        self.rows, self.columns = init_Field()
        self.outstr = field_to_str(self.rows, self.columns)
        self.player1 = {"Name": "Player1", "Points": 0}
        self.player2 = {"Name": "Player2", "Points": 0}
        self.whose_turn = random.randint(0, 1)
        self.field_arrays = [self.rows, self.columns]
        self.obstacle_count = self._obstacle_count()

    def _obstacle_count(self):
        count = 0
        for arr in self.field_arrays:
            count += np.sum(arr == 1)
        return count

    def get_player_score(self, player_nr):
        if player_nr == 1:
            return self.player1['Points'] - self.player2['Points']
        elif player_nr == 2:
            return self.player2['Points'] - self.player1['Points']
        return 0

    def calculate_active_player(self, whose_turn):
        return self.player1 if whose_turn == 1 else self.player2

    def make_move(self, array_i, h, w):
        if validate_move(self.field_arrays, array_i, h, w):
            self.field_arrays[array_i][h][w] = 1
            return True
        return False

    def random_move(self):
        success = False
        while not success and self.free_edge_count() > 0:
            w = random.randint(0, width)
            h = random.randint(0, height)
            i = random.randint(0, 1)
            success = self.make_move(i, h, w)
        return i, h, w

    def free_edge_count(self):
        counter = 0
        for arr in self.rows:
            counter += np.count_nonzero(arr == 0)
        for arr in self.columns:
            counter += np.count_nonzero(arr == 0)
        return counter

    def _get_reward(self, playernr, old_score):
        return self.get_player_score(playernr) - old_score

    def act(self, action, playernr):
        """
        Thực thi action => cập nhật game state, trả về (new_input, old_score, gameover)
        """
        old_score = self.get_player_score(playernr)
        array_i, h, w = self.convert_action_to_move(action)
        success = self.make_move(array_i, h, w)
        if not success:
            # fallback random_move nếu action không valid
            self.random_move()
        new_fields = new_full_fields([self.rows, self.columns], array_i, h, w)
        self.calculate_active_player(playernr)["Points"] += new_fields
        new_input = self.convert_and_reshape_field_to_inputarray([self.rows, self.columns])
        gameover = game_over(self)
        return new_input, old_score, gameover

    def convert_field_to_inputarray(self, field):
        """
        Chuyển rows, columns => vector 1 chiều 24 phần tử
        """
        input_arr = np.zeros(num_actions, dtype=np.float32)
        index = 0
        rows, cols = field
        for h in range(height + 1):
            for i in range(2):
                for w in range(width + 1):
                    if i == 0 and w < rows.shape[1]:
                        if rows[h][w] == 1:
                            input_arr[index] = 1
                        index += 1
                    elif i == 1 and h < cols.shape[0]:
                        if cols[h][w] == 1:
                            input_arr[index] = 1
                        index += 1
        return input_arr

    def convert_and_reshape_field_to_inputarray(self, field):
        input_array = self.convert_field_to_inputarray(field)
        return input_array.reshape((1, -1))

    def convert_action_to_move(self, action):
        """
        Chuyển chỉ số action (0..23) => (which_array, h, w)
        """
        array_i = 0
        w = 0
        h = 0
        for i in range(action):
            w += 1
            if w >= width + array_i:
                w = 0
                if array_i == 1:
                    h += 1
                array_i = 1 - array_i
        return array_i, h, w

class GameExtended(Game):
    def __init__(self):
        super().__init__()
        self.random_plays = 0

    def convert_input_array_to_field(self, input_arr):
        a = self.rows.copy()
        b = self.columns.copy()
        field = [a, b]
        for i in range(len(input_arr)):
            array_i, h, w = self.convert_action_to_move(i)
            field[array_i][h][w] = input_arr[i]
        return field

# -------------------- PHẦN AI VỚI PYTORCH --------------------
def find_best_for_state(q, state):
    """
    Tìm max Q nhưng loại bỏ các action đã kẻ (state[0][idx]==1).
    """
    q = np.squeeze(q)
    index = np.argmax(q)
    tmp = np.copy(q)
    while state[0][index] != 0:
        tmp[index] = -100000
        index = np.argmax(tmp)
    return np.max(tmp)

def find_best(q, env):
    """
    Tìm action có Q lớn nhất, bỏ qua action đã kẻ.
    """
    q = np.squeeze(q)
    action = np.argmax(q)
    tmp = np.copy(q)
    array_i, h, w = env.convert_action_to_move(action)
    while not validate_move([env.rows, env.columns], array_i, h, w):
        tmp[action] = -100000
        action = np.argmax(tmp)
        array_i, h, w = env.convert_action_to_move(action)
    return action

def taker_player_move(env):
    """
    Nếu có nước đi nào lập tức 'chốt' được ô, chọn ngay.
    """
    did_take = False
    for i in range(num_actions):
        array_i, h, w = env.convert_action_to_move(i)
        if validate_move([env.rows, env.columns], array_i, h, w):
            if new_full_fields([env.rows, env.columns], array_i, h, w) > 0:
                return i, True
    return False, did_take

class Ai:
    def __init__(self, playernr, model_name, max_memory=100, discount=0.9):
        self.playernr = playernr
        self.max_memory = max_memory
        self.memory = []  # (state_old, action, reward, state_new)
        self.discount = discount
        self.model_name = model_name
        self.model = self.init_model(self.model_name)
        # -> Đổi từ lr=1.0 xuống 0.1
        self.optimizer = optim.Adadelta(self.model.parameters(), lr=0.1)
        self.criterion = nn.MSELoss()

    def init_model(self, model_name):
        model = nn.Sequential(
            nn.Linear(num_actions, hidden_size_0),
            nn.ReLU(),
            nn.Linear(hidden_size_0, hidden_size_1),
            nn.ReLU(),
            nn.Linear(hidden_size_1, hidden_size_0),
            nn.ReLU(),
            nn.Linear(hidden_size_0, num_actions)
        )
        # Nếu đã có model cũ
        if os.path.isfile(temp_model(self.model_name)):
            model = torch.load(temp_model(self.model_name))
            print("model_loaded")
        return model

    def remember(self, experience, gameover):
        self.memory.append((experience, gameover))
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, batch_size=10):
        len_memory = len(self.memory)
        actual_batch_size = min(len_memory, batch_size)
        inputs = np.zeros((actual_batch_size, num_actions), dtype=np.float32)
        targets = np.zeros((actual_batch_size, num_actions), dtype=np.float32)
        batch_indices = np.random.randint(0, len_memory, size=actual_batch_size)
        for i, idx in enumerate(batch_indices):
            state_old, action, reward, state_new = self.memory[idx][0]
            gameover = self.memory[idx][1]
            inputs[i] = state_old
            with torch.no_grad():
                state_tensor = torch.tensor(state_old, dtype=torch.float32).unsqueeze(0)
                target = self.model(state_tensor).detach().numpy()[0]
            if gameover:
                target[action] = reward
            else:
                with torch.no_grad():
                    next_state_tensor = torch.tensor(state_new, dtype=torch.float32).unsqueeze(0)
                    q_next = self.model(next_state_tensor).detach().numpy()[0]
                Q_sa = find_best_for_state(q_next, state_new)
                target[action] = reward + self.discount * Q_sa
            targets[i] = target
        return inputs, targets

def evaluate_ai(loss, ai: Ai, old_score, input_old, action, input_new, gameover, batch_size, game_count, winner=None):
    """
    Cập nhật replay buffer, tính reward => backprop nếu cần
    """
    reward = env._get_reward(playernr=ai.playernr, old_score=old_score)
    ai.remember((input_old, action, reward, input_new), gameover)
    if train_mode_immediate:
        inputs, targets = ai.get_batch(batch_size=batch_size)
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
        targets_tensor = torch.tensor(targets, dtype=torch.float32)
        ai.optimizer.zero_grad()
        outputs = ai.model(inputs_tensor)
        loss_val = ai.criterion(outputs, targets_tensor)
        loss_val.backward()
        ai.optimizer.step()
        loss += loss_val.item()
    else:
        # Train theo kiểu "cuối ván" hoặc mỗi vài ván
        if gameover and game_count % 4 == 0:
            inputs, targets = ai.get_batch(batch_size=batch_size)
            inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
            targets_tensor = torch.tensor(targets, dtype=torch.float32)
            ai.optimizer.zero_grad()
            outputs = ai.model(inputs_tensor)
            loss_val = ai.criterion(outputs, targets_tensor)
            loss_val.backward()
            ai.optimizer.step()
            loss = loss_val.item()
    return loss

def ai_player_move(input_state, gameover, ai: Ai, loss, use_taker_player: bool, game_count):
    """
    AI đánh. Nếu kẻ được ô => đánh tiếp
    """
    active_player = env.player1 if ai.playernr == 1 else env.player2
    ai_should_play = True
    while ai_should_play and not gameover:
        ai_should_play = False
        playernr = ai.playernr
        input_old = input_state
        # eps-greedy
        if np.random.rand() <= epsilon:
            valid = False
            while not valid:
                action = random.randint(0, num_actions - 1)
                array_i, h, w = env.convert_action_to_move(action)
                valid = validate_move([env.rows, env.columns], array_i, h, w)
        else:
            # Kiểm tra taker_move => giành ô ngay
            did_take = False
            action = None
            if use_taker_player:
                action, did_take = taker_player_move(env)
            if not did_take:
                with torch.no_grad():
                    state_tensor = torch.tensor(input_old, dtype=torch.float32)
                    q = ai.model(state_tensor).detach().numpy()
                action = find_best(q, env)
        old_points = active_player["Points"]
        input_state, old_score, gameover = env.act(action, playernr)
        new_points = active_player["Points"]
        if new_points > old_points:
            # nếu giành được ô, AI đi tiếp
            ai_should_play = True
        if ai_should_play:
            winner = None
            if gameover:
                winnernr = 1 if env.player1["Points"] > env.player2["Points"] else 2
                winner = (ai.playernr == winnernr)
            loss = evaluate_ai(loss, ai, old_score, input_old, action, input_state, gameover, batch_size, game_count, winner=winner)
    return input_state, gameover, old_score, input_old, action, loss

def temp_model(model_name):
    return "temp_{}".format(model_name)

# -------------------- CÀI ĐẶT HUẤN LUYỆN --------------------
if __name__ == "__main__":
    train_mode_immediate = False  # Nếu True => cập nhật ngay sau mỗi bước
    # Ta có thể giảm epsilon từ 1 => 0.05, nhưng ở đây để 0.3 bắt đầu => AI ít random hơn
    epsilon = 0.3
    # Cơ chế decay epsilon
    epsilon_decay = 0.95
    min_epsilon = 0.05

    epoch = 1000           # Số vòng lặp huấn luyện
    max_memory = 1 if train_mode_immediate else 500
    hidden_size_0 = 64
    hidden_size_1 = 128
    batch_size = 1 if train_mode_immediate else 200
    learning_rate = 0.1    
    discount = 0.5
    champion = 1

    if not os.path.isfile('champion.txt'):
        with open('champion.txt', 'w') as f:
            f.write("1")
    else:
        with open('champion.txt', 'r') as f:
            for line in f:
                champion = int(line.strip())

    model_name = f"mm{max_memory}_hsmin{hidden_size_0}_hsmax{hidden_size_1}_lr{learning_rate}_d{discount}_hl3_na{num_actions}_ti{train_mode_immediate}"
    model_name_1 = model_name + "_1.pth"
    model_name_2 = model_name + "_2.pth"

    ai_player_1 = Ai(max_memory=max_memory, playernr=1, discount=discount, model_name=model_name_1)
    ai_player_2 = Ai(max_memory=max_memory, playernr=2, discount=discount, model_name=model_name_2)

    model_epochs_trained = 0
    learning_filename = "{}.txt".format(temp_model(champion == 1 and ai_player_1.model_name or ai_player_2.model_name))
    if not os.path.isfile(learning_filename):
        with open(learning_filename, 'w') as f:
            f.write("{} {}".format(temp_model(champion == 1 and ai_player_1.model_name or ai_player_2.model_name), 0))
    else:
        with open(learning_filename, 'r') as f:
            for line in f:
                try:
                    key, value = line.split()
                    if key == temp_model(champion == 1 and ai_player_1.model_name or ai_player_2.model_name):
                        model_epochs_trained = int(value)
                except ValueError:
                    continue

    if model_epochs_trained == 0:
        print("epoch save not found, defaulting to 0")

    def learning_ai(ai_1, ai_2, champion):
        return ai_2 if champion == 1 else ai_1

    game_count = 0
    total_learning_wins = 0
    loss = 0.0

    print("Số epoch đã huấn luyện:", model_epochs_trained)

    while True:
        e = model_epochs_trained
        if e % 100 == 0:
            print(f"[Epoch={e}] Đang huấn luyện mô hình: {learning_ai(ai_player_1, ai_player_2, champion).model_name}")
            sys.stdout.flush()

        env = GameExtended()
        if train_mode_immediate:
            loss = 0.0  # reset loss mỗi ván nếu train ngay

        gameover = False
        input_1 = env.convert_and_reshape_field_to_inputarray([env.rows, env.columns])
        old_score_1 = 0
        input_old_1 = None
        action_1 = None
        old_score_2 = 0
        input_old_2 = None
        action_2 = None
        ai_2_played = False

        while not gameover:
            # AI1 đánh
            input_2, gameover, old_score_1, input_old_1, action_1, loss = ai_player_move(
                input_state=input_1, gameover=gameover, ai=ai_player_1, loss=loss, use_taker_player=False, game_count=game_count
            )
            # Cập nhật cho AI2 cũ (nếu champion != 2) => replay buffer
            if ai_2_played and champion != 2:
                winner = None
                if gameover:
                    winnernr = 1 if env.player1["Points"] > env.player2["Points"] else 2
                    winner = (2 == winnernr)
                loss = evaluate_ai(loss, ai_player_2, old_score_2, input_old_2, action_2, input_2, gameover, batch_size, game_count, winner=winner)

            if not gameover:
                # AI2 đánh
                input_1, gameover, old_score_2, input_old_2, action_2, loss = ai_player_move(
                    input_state=input_2, gameover=gameover, ai=ai_player_2, loss=loss, use_taker_player=False, game_count=game_count
                )
                ai_2_played = True
                # Cập nhật cho AI1 cũ (nếu champion != 1)
                if champion != 1:
                    winner = None
                    if gameover:
                        winnernr = 1 if env.player1["Points"] > env.player2["Points"] else 2
                        winner = (1 == winnernr)
                    loss = evaluate_ai(loss, ai_player_1, old_score_1, input_old_1, action_1, input_1, gameover, batch_size, game_count, winner=winner)

        # Sau khi game kết thúc
        champion_field = env.player1["Points"] if champion == 1 else env.player2["Points"]
        learning_field = env.player1["Points"] if champion == 2 else env.player2["Points"]
        learning_wins = 1 if learning_field > champion_field else 0
        total_learning_wins += learning_wins
        game_count += 1

        # In ra kết quả mỗi 100 ván
        if game_count % 100 == 0:
            print(f"[Epoch={model_epochs_trained}] Sau {game_count} game, learning wins: {total_learning_wins}, champion = {champion}")
            sys.stdout.flush()
            # Giảm epsilon để giảm random => AI sẽ "thông minh" hơn dần
            if epsilon > min_epsilon:
                epsilon *= epsilon_decay
                if epsilon < min_epsilon:
                    epsilon = min_epsilon
            print(f"Epsilon hiện tại: {epsilon}")
            sys.stdout.flush()

        # Kiểm tra nếu AI học thắng >= 95 ván/100 => đổi champion
        if game_count % 100 == 0:
            if total_learning_wins >= 95:
                champion = 1 if champion == 2 else 2
                with open('champion.txt', 'w') as f:
                    f.write("{}".format(champion))
            game_count = 0
            total_learning_wins = 0

        model_epochs_trained += 1

        # Định kỳ in loss
        if model_epochs_trained % 100 == 0:
            print(f"[Epoch {model_epochs_trained}] Loss = {loss}")
            sys.stdout.flush()

        # Lưu model tạm định kỳ
        if model_epochs_trained % 50 == 0 and model_epochs_trained != 0:
            l_ai = learning_ai(ai_player_1, ai_player_2, champion)
            torch.save(l_ai.model, temp_model(l_ai.model_name))
            with open("{}.txt".format(l_ai.model_name), 'w') as f:
                f.write("{} {}".format(temp_model(l_ai.model_name), model_epochs_trained))

        if model_epochs_trained >= epoch:
            print("Training complete. Saving final model.")
            l_ai = learning_ai(ai_player_1, ai_player_2, champion)
            torch.save(l_ai.model, l_ai.model_name)
            break

    # -------------------- PHẦN GIAO DIỆN VỚI PYGAME (nếu muốn chạy luôn) --------------------
    # ...

pygame.init()
display_width = width * 100
display_height = height * 100

black = (0, 0, 0)
white = (255, 255, 255)
grey = (235, 235, 235)
dark_grey = (100, 100, 100)
red = (255, 0, 0)
green = (0, 235, 20)
dark_green = (0, 155, 0)
blue = (0, 20, 235)
dark_blue = (0, 0, 155)

model = torch.load(temp_model(ai_player_1.model_name))

gameDisplay = pygame.display.set_mode((display_width, display_height))
pygame.display.set_caption("Dots and Boxes AI")
clock = pygame.time.Clock()

line_length = 80
line_thickness = 10
vertical_space = 100
horizontal_space = 80
lines = []

def convert_action_to_move(action):
    array_i = 0
    w = 0
    h = 0
    for i in range(action):
        w += 1
        if w >= width + array_i:
            w = 0
            if array_i == 1:
                h += 1
            array_i = 1 - array_i
    return array_i, h, w

def draw_full_field(h, w, color=dark_grey):
    rect = pygame.Rect(w * line_length + horizontal_space + line_thickness,
                       h * line_length + line_thickness + vertical_space,
                       line_length - line_thickness,
                       line_length - line_thickness)
    pygame.draw.rect(gameDisplay, color, rect)

def newFullField(field, which, h, w, color=dark_blue):
    if which == 0:
        if h != 0:
            if field[1][h-1][w] == 1 and field[0][h-1][w] == 1 and field[1][h-1][w+1] == 1:
                draw_full_field(h-1, w, color)
        if h != height:
            if field[1][h][w] == 1 and field[0][h+1][w] == 1 and field[1][h][w+1] == 1:
                draw_full_field(h, w, color)
    else:
        if w != 0:
            if field[0][h][w-1] == 1 and field[1][h][w-1] == 1 and field[0][h+1][w-1] == 1:
                draw_full_field(h, w-1, color)
        if w != width:
            if field[0][h][w] == 1 and field[1][h][w+1] == 1 and field[0][h+1][w] == 1:
                draw_full_field(h, w, color)

def draw_full_fields(action_num, field, color):
    array_i, h, w = convert_action_to_move(action_num)
    newFullField(field, array_i, h, w, color)

def draw_move(action, field, color):
    global lines
    new_line = lines[action]
    array_i, h, w = convert_action_to_move(action)
    field[array_i][h][w] = 1
    pygame.draw.rect(gameDisplay, red, new_line)
    pygame.display.update()
    pygame.time.wait(500)
    pygame.draw.rect(gameDisplay, color, new_line)
    field_color = green if color == dark_green else blue
    draw_full_fields(action, field, field_color)
    pygame.display.update()
    return field

def define_lines(rows, columns):
    my_line_array = []
    for h in range(height + 1):
        for i in range(2):
            for w in range(width + 1):
                if i == 0 and w < rows.shape[1]:
                    l = pygame.Rect(w * line_length + horizontal_space + line_thickness,
                                    h * line_length + vertical_space,
                                    line_length - line_thickness,
                                    line_thickness)
                    my_line_array.append(l)
                    if rows[h][w] == 1:
                        pygame.draw.rect(gameDisplay, black, l)
                    else:
                        pygame.draw.rect(gameDisplay, grey, l)
                elif i == 1 and h < columns.shape[0]:
                    l = pygame.Rect(w * line_length + horizontal_space,
                                    h * line_length + vertical_space + line_thickness,
                                    line_thickness,
                                    line_length - line_thickness)
                    my_line_array.append(l)
                    if columns[h][w] == 1:
                        pygame.draw.rect(gameDisplay, black, l)
                        if test_field_full(rows, columns, h, w):
                            draw_full_field(h, w)
                    else:
                        pygame.draw.rect(gameDisplay, grey, l)
    return my_line_array

def text_objects(text, font):
    textSurface = font.render(text, True, black)
    return textSurface, textSurface.get_rect()

def message_display(text):
    rect = pygame.Rect(80, 40, 400, 30)
    pygame.draw.rect(gameDisplay, white, rect)
    largeText = pygame.font.Font('freesansbold.ttf', 30)
    TextSurf, TextRect = text_objects(text, largeText)
    TextRect.topleft = (80, 40)
    gameDisplay.blit(TextSurf, TextRect)
    pygame.display.update()

def game_over_show(env, user_nr, ai_nr):
    points_user = env.calculate_active_player(user_nr)["Points"]
    points_ai = env.calculate_active_player(ai_nr)["Points"]
    if points_user > points_ai:
        message_display('You win with {} points! Ai points: {}'.format(points_user, points_ai))
    else:
        message_display('You lose with {} points! Ai points: {}'.format(points_user, points_ai))

def ai_move(field, env, ai_number):
    ais_turn = True
    while ais_turn:
        ais_turn = False
        input_state = env.convert_and_reshape_field_to_inputarray(field)
        with torch.no_grad():
            q = model(torch.tensor(input_state, dtype=torch.float32)).detach().numpy()[0]
        action = find_best(q, env)
        array_i, h, w = convert_action_to_move(action)
        field = draw_move(action, field, dark_green)
        new_fields = new_full_fields(field, array_i, h, w)
        env.calculate_active_player(ai_number)["Points"] += new_fields
        user_nr = 1 if ai_number == 2 else 2
        print_points(env.calculate_active_player(user_nr)["Points"], env.calculate_active_player(ai_number)["Points"])
        if game_over(env):
            return field, True
        if new_fields != 0:
            ais_turn = True
            pygame.time.wait(500)
    return field, False

def print_points(points_user, points_ai):
    rect = pygame.Rect(80, 40, 400, 30)
    pygame.draw.rect(gameDisplay, white, rect)
    myfont = pygame.font.SysFont(None, 40)
    points = myfont.render("Points User: {}, Points Ai: {}".format(points_user, points_ai), 1, black)
    gameDisplay.blit(points, (80, 40))

def print_time(time_val):
    rect = pygame.Rect(720, 40, 400, 30)
    pygame.draw.rect(gameDisplay, white, rect)
    myfont = pygame.font.SysFont(None, 40)
    points = myfont.render("{}".format(time_val), 1, black)
    gameDisplay.blit(points, (720, 40))

def game_loop_ai_vs_user():
    gameDisplay.fill(white)
    gameexit = False
    gameover = False
    env = GameExtended()
    print_points(0, 0)
    rows, columns = env.rows, env.columns
    global lines
    lines = define_lines(rows, columns)
    field = [rows, columns]
    user_number = 1
    ai_number = 2
    pygame.display.update()
    timer = time.time()
    while not gameexit:
        time_left = 5 - int(time.time() - timer)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.MOUSEBUTTONUP and not gameover:
                pos = pygame.mouse.get_pos()
                for idx, line in enumerate(lines):
                    array_i, h, w = convert_action_to_move(idx)
                    if line.collidepoint(pos) and field[array_i][h][w] != 1:
                        field = draw_move(idx, field, dark_blue)
                        array_i, h, w = convert_action_to_move(idx)
                        new_fields = new_full_fields(field, array_i, h, w)
                        env.calculate_active_player(user_number)["Points"] += new_fields
                        print_points(env.calculate_active_player(user_number)["Points"], env.calculate_active_player(ai_number)["Points"])
                        if game_over(env):
                            gameover = True
                            break
                        if new_fields == 0:
                            pygame.event.set_blocked(pygame.MOUSEBUTTONUP)
                            field, gameover = ai_move(field, env, ai_number)
                            pygame.event.set_allowed(pygame.MOUSEBUTTONUP)
                        timer = time.time()
                        break
        if gameover:
            game_over_show(env, user_number, ai_number)
            pygame.time.wait(5000)
            game_loop_ai_vs_user()
        pygame.display.update()
        clock.tick(10)

game_loop_ai_vs_user()
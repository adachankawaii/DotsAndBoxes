import tkinter as tk
import math
import copy

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
        self.horiz = [[False for _ in range(cols)] for _ in range(rows + 1)]
        # Ma trận các đường dọc: rows x (cols+1)
        self.vert = [[False for _ in range(cols + 1)] for _ in range(rows)]
        # Ma trận ô vuông: rows x cols, None nếu chưa ai chiếm
        self.boxes = [[None for _ in range(cols)] for _ in range(rows)]
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
        # Các bước đi: đường ngang định dạng ("h", i, j)
        for i in range(len(self.horiz)):
            for j in range(len(self.horiz[0])):
                if not self.horiz[i][j]:
                    moves.append(("h", i, j))
        # Các bước đi: đường dọc định dạng ("v", i, j)
        for i in range(len(self.vert)):
            for j in range(len(self.vert[0])):
                if not self.vert[i][j]:
                    moves.append(("v", i, j))
        return moves

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

# Cập nhật evaluate_state để sử dụng các trọng số từ WEIGHTS
def evaluate_state(state):
    score = state.bot_score - state.player_score

    # --- Các heuristics cơ bản ---
    score += evaluate_chain(state)
    score += evaluate_potential(state)
    score += evaluate_safety(state)

    return score

    
def evaluate_chain(state):
    """
    Đánh giá chuỗi các ô (Chains) có thể bị chiếm trong các lượt tiếp theo.
    """
    chain_bonus = 0
    for i in range(state.rows):
        for j in range(state.cols):
            if state.boxes[i][j] is None:
                # Kiểm tra xem ô (i, j) có thể là một phần của chuỗi
                if is_part_of_chain(state, i, j):
                    chain_bonus += 1 if state.turn == 'bot' else -1
    return chain_bonus

def evaluate_potential(state):
    """
    Đánh giá tiềm năng chiếm ô: tính toán các ô chưa hoàn thành và khả năng chiếm ô của bot.
    """
    potential_bonus = 0
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

                # Đánh giá ô có 3 cạnh (sắp xếp thứ tự cao hơn)
                if drawn == 3:
                    bonus = 2.0
                    potential_bonus += bonus if state.turn == 'bot' else -bonus
                elif drawn == 2:
                    bonus = 0.5
                    potential_bonus += bonus if state.turn == 'bot' else -bonus
    return potential_bonus

def is_part_of_chain(state, i, j):
    """
    Kiểm tra xem ô (i, j) có thể nằm trong một chuỗi các ô có thể bị chiếm liên tiếp không.
    """
    # Danh sách các ô lân cận (ở đây là 4 ô xung quanh ô (i, j)).
    neighbors = [
        (i-1, j), (i+1, j),  # Trên và dưới
        (i, j-1), (i, j+1)   # Trái và phải
    ]
    
    # Kiểm tra các ô lân cận đã có ít nhất 3 cạnh được vẽ, có thể tạo thành chuỗi
    for ni, nj in neighbors:
        if 0 <= ni < state.rows and 0 <= nj < state.cols:
            if state.boxes[ni][nj] is None:  # Nếu ô chưa bị chiếm
                drawn = 0
                if state.horiz[ni][nj]:
                    drawn += 1
                if state.horiz[ni + 1][nj]:
                    drawn += 1
                if state.vert[ni][nj]:
                    drawn += 1
                if state.vert[ni][nj + 1]:
                    drawn += 1
                if drawn == 3:  # Nếu ô này có thể bị chiếm trong lượt tới
                    return True
    return False

def evaluate_safety(state):
    """
    Đánh giá an toàn của các nước đi.
    Nếu bot có thể tạo ra một nước đi an toàn mà không cho phép đối thủ chiếm ô dễ dàng.
    """
    safety_score = 0
    for i in range(state.rows):
        for j in range(state.cols):
            if state.boxes[i][j] is None:
                # Kiểm tra xem nếu bot chiếm ô này, có tạo ra cơ hội cho đối thủ không.
                if is_safe_move(state, i, j):
                    safety_score += 1  # Nước đi an toàn được đánh giá cao hơn
                else:
                    safety_score -= 1  # Nếu là nước đi nguy hiểm, trừ điểm

    return safety_score

def is_safe_move(state, i, j):
    """
    Kiểm tra nếu một nước đi vào ô (i, j) có thể tạo ra cơ hội cho đối thủ.
    """
    # Danh sách các ô lân cận (4 ô xung quanh ô (i, j))
    neighbors = [
        (i-1, j), (i+1, j),  # Trên và dưới
        (i, j-1), (i, j+1)   # Trái và phải
    ]
    
    # Kiểm tra các ô lân cận xem có thể tạo cơ hội cho đối thủ không
    for ni, nj in neighbors:
        if 0 <= ni < state.rows and 0 <= nj < state.cols:
            if state.boxes[ni][nj] is None:  # Nếu ô này chưa bị chiếm
                drawn = 0
                if state.horiz[ni][nj]:
                    drawn += 1
                if state.horiz[ni + 1][nj]:
                    drawn += 1
                if state.vert[ni][nj]:
                    drawn += 1
                if state.vert[ni][nj + 1]:
                    drawn += 1
                # Nếu ô này có thể bị chiếm ngay sau đó bởi đối thủ, nó không an toàn
                if drawn == 3:
                    return False
    return True  # Nếu không có cơ hội cho đối thủ, nước đi này an toàn




# --- Thuật toán minimax với alpha-beta pruning ---
def minimax(state, depth, alpha, beta, maximizing_player):
    if depth == 0 or state.is_game_over():
        return evaluate_state(state), None

    possible_moves = state.get_possible_moves()
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

    
# --- Giao diện trò chơi ---
# GameFrame là khung chứa giao diện của bàn chơi
class GameFrame(tk.Frame):
    def __init__(self, parent, app, rows, cols, difficulty, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.app = app
        self.rows = rows
        self.cols = cols
        self.difficulty = difficulty
        self.depth = {"Easy": 3, "Medium": 4, "Hard": 5}.get(difficulty, 3)
        self.state = GameState(rows, cols)

        self.canvas_width = cols * CELL_SIZE + 4 * PADDING
        self.canvas_height = rows * CELL_SIZE + 4 * PADDING

        self.canvas = tk.Canvas(self, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas.pack(expand=True)
        self.canvas.bind("<Button-1>", self.on_click)

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
        self.after(20, self.animate_box, i, j, owner, progress + 0.1)

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
        # Bot tính toán nước đi bằng minimax
        # với độ sâu đã chọn
        _, best_move = minimax(self.state,
        self.depth, -math.inf, math.inf, True)

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
        ttk.Label(size_frame, text="Rows:").grid(row=0, column=0, padx=5, pady=5)
        ttk.Label(size_frame, text="Columns:").grid(row=1, column=0, padx=5, pady=5)
        self.rows_var = tk.IntVar(value=3)
        self.cols_var = tk.IntVar(value=3)
        self.rows_spin = ttk.Spinbox(size_frame, from_=3, to=9, textvariable=self.rows_var, width=5)
        self.rows_spin.grid(row=0, column=1, padx=5, pady=5)
        self.cols_spin = ttk.Spinbox(size_frame, from_=3, to=9, textvariable=self.cols_var, width=5)
        self.cols_spin.grid(row=1, column=1, padx=5, pady=5)

        # Khung chọn độ khó
        diff_frame = ttk.LabelFrame(container, text="Difficulty Level", padding=10)
        diff_frame.pack(pady=15, fill="x", padx=20)
        self.difficulty_var = tk.StringVar(value="Easy")
        self.diff_option = ttk.Combobox(diff_frame, textvariable=self.difficulty_var, values=["Easy", "Medium", "Hard"],
                                        state="readonly")
        self.diff_option.pack(pady=5, padx=10)
        self.diff_option.current(1)

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
        game_frame.pack(fill="both", expand=True)


if __name__ == "__main__":
    app = App()
    app.mainloop()
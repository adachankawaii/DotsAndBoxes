import math

def evaluate_state_minimax(state):
    """
    Đánh giá heuristic của trạng thái trò chơi cho thuật toán minimax.

    Kết hợp các thành phần:
      - Hiệu số điểm hiện tại (bot_score - player_score)
      - Điểm chuỗi (chain bonus)
      - Điểm tiềm năng (potential bonus)
      - Điểm an toàn (safety score)

    Trả về:
      float – giá trị dương ủng hộ bot, giá trị âm ủng hộ người chơi.
    """

    # Điểm cơ bản = điểm bot trừ điểm người chơi
    score = state.bot_score - state.player_score
    # Thêm điểm từ chuỗi
    score += evaluate_chain(state)
    # Thêm điểm từ tiềm năng (ô gần hoàn thành)
    score += evaluate_potential(state)
    # Thêm điểm an toàn (tránh để đối thủ ăn)
    score += evaluate_safety(state)
    # score += greedy_evaluate(state)
    return score

def evaluate_chain(state):
    """
    Tính điểm chuỗi: các ô trống liên kết thành chuỗi dài hơn 1.

    Với mỗi ô trống thuộc chuỗi, cộng +1 nếu đang là lượt bot,
    trừ -1 nếu là lượt người chơi.
    """
    chain_bonus = 0
    for i in range(state.rows):
        for j in range(state.cols):
            if state.boxes[i][j] is None and is_part_of_chain(state, i, j):
                chain_bonus += 1 if state.turn == 'bot' else -1
    return chain_bonus

def evaluate_potential(state):
    """
    Tính điểm tiềm năng dựa trên số cạnh đã vẽ:
      - +2.0 nếu ô có 3 cạnh đã vẽ
      - +0.5 nếu ô có 2 cạnh đã vẽ
      -  0.0 cho trường hợp khác

    Dương cho lượt bot, âm cho lượt người chơi.
    """
    potential_bonus = 0.0
    for i in range(state.rows):
        for j in range(state.cols):
            if state.boxes[i][j] is None:
                drawn = (
                    state.horiz[i][j]
                    + state.horiz[i+1][j]
                    + state.vert[i][j]
                    + state.vert[i][j+1]
                )
                if drawn == 3:
                    bonus = 1.0
                elif drawn == 2:
                    bonus = 0.5
                else:
                    bonus = 0.0
                potential_bonus += bonus if state.turn == 'bot' else -bonus
    return potential_bonus


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

def is_part_of_chain(state, i, j):
    """
    Kiểm tra ô (i,j) có nằm trong chuỗi không:
    Nếu có ô kề bên cũng có đúng 3 cạnh đã vẽ → thành phần của chuỗi.
    """
    neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
    for ni, nj in neighbors:
        if 0 <= ni < state.rows and 0 <= nj < state.cols:
            if state.boxes[ni][nj] is None:
                drawn = (
                    state.horiz[ni][nj]
                    + state.horiz[ni+1][nj]
                    + state.vert[ni][nj]
                    + state.vert[ni][nj+1]
                )
                if drawn == 3:
                    return True
    return False

def is_safe_move(state, i, j):
    """
    Kiểm tra nước đi tại ô (i,j) có an toàn không:
    Nếu tạo ra ô kề bên có 3 cạnh → không an toàn.
    """
    neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
    for ni, nj in neighbors:
        if 0 <= ni < state.rows and 0 <= nj < state.cols:
            if state.boxes[ni][nj] is None:
                drawn = (
                    state.horiz[ni][nj]
                    + state.horiz[ni+1][nj]
                    + state.vert[ni][nj]
                    + state.vert[ni][nj+1]
                )
                if drawn == 3:
                    return False
    return True


def minimax(state, depth, alpha, beta, maximizing_player):
    """
    Thuật toán minimax kèm cắt tỉa alpha-beta.

    Tham số:
      state            – đối tượng GameState hiện tại
      depth            – độ sâu còn lại
      alpha, beta      – ngưỡng cắt tỉa
      maximizing_player – True nếu lượt bot, False nếu lượt người chơi

    Trả về:
      (best_score, best_move) – cặp giá trị điểm tốt nhất và nước đi tương ứng.
    """
    # Dừng khi về độ sâu 0 hoặc trò chơi kết thúc
    if depth == 0 or state.is_game_over():
        return evaluate_state_minimax(state), None

    moves, _ = state.get_possible_moves()
    best_move = None

    if maximizing_player:
        max_eval = -math.inf
        for m in moves:
            s2 = state.clone()
            extra = s2.apply_move(m, 'bot')
            if extra:
                # Bot được đánh tiếp
                val, _ = minimax(s2, depth-1, alpha, beta, True)
            else:
                # Chuyển lượt cho người chơi
                s2.turn = 'player'
                val, _ = minimax(s2, depth-1, alpha, beta, False)
            if val > max_eval:
                max_eval, best_move = val, m
            alpha = max(alpha, val)
            if beta <= alpha:
                break  # cắt tỉa
        return max_eval, best_move
    else:
        min_eval = math.inf
        for m in moves:
            s2 = state.clone()
            extra = s2.apply_move(m, 'player')
            if extra:
                val, _ = minimax(s2, depth-1, alpha, beta, False)
            else:
                s2.turn = 'bot'
                val, _ = minimax(s2, depth-1, alpha, beta, True)
            if val < min_eval:
                min_eval, best_move = val, m
            beta = min(beta, val)
            if beta <= alpha:
                break  # cắt tỉa
        return min_eval, best_move
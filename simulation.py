import time
import random
import math
from main import GameState, greedy_move
from Minimax import minimax as full_minimax

def evaluate_potential_simple(state):
    """
    Greedy heuristic:
     - bonus = +100 if a box has 3 sides drawn  → force capture
     - bonus = -100 if a box has 2 sides drawn → avoid creating a 3-side box
    """
    potential = 0.0
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
                    bonus = 100.0
                elif drawn == 2:
                    bonus = -100.0
                else:
                    bonus = 0.0
                # positive for bot’s turn, negative for player’s
                potential += bonus if state.turn == 'bot' else -bonus
    return potential

# 1. Simple Minimax without pruning, using only potential heuristic
def minimax_simple(state, depth, maximizing_player):
    if depth == 0 or state.is_game_over():
        return evaluate_potential_simple(state), None
    moves, _ = state.get_possible_moves()
    best_move = None
    if maximizing_player:
        max_eval = -math.inf
        for m in moves:
            s2 = state.clone()
            extra = s2.apply_move(m, 'bot')
            if extra:
                val, _ = minimax_simple(s2, depth-1, True)
            else:
                s2.turn = 'player'
                val, _ = minimax_simple(s2, depth-1, False)
            if val > max_eval:
                max_eval, best_move = val, m
        return max_eval, best_move
    else:
        min_eval = math.inf
        for m in moves:
            s2 = state.clone()
            extra = s2.apply_move(m, 'player')
            if extra:
                val, _ = minimax_simple(s2, depth-1, False)
            else:
                s2.turn = 'bot'
                val, _ = minimax_simple(s2, depth-1, True)
            if val < min_eval:
                min_eval, best_move = val, m
        return min_eval, best_move

# 2. Minimax with alpha-beta pruning, using only potential heuristic
def minimax_ab_simple(state, depth, alpha, beta, maximizing_player):
    if depth == 0 or state.is_game_over():
        return evaluate_potential_simple(state), None
    moves, _ = state.get_possible_moves()
    best_move = None
    if maximizing_player:
        max_eval = -math.inf
        for m in moves:
            s2 = state.clone()
            extra = s2.apply_move(m, 'bot')
            if extra:
                val, _ = minimax_ab_simple(s2, depth-1, alpha, beta, True)
            else:
                s2.turn = 'player'
                val, _ = minimax_ab_simple(s2, depth-1, alpha, beta, False)
            if val > max_eval:
                max_eval, best_move = val, m
            alpha = max(alpha, val)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = math.inf
        for m in moves:
            s2 = state.clone()
            extra = s2.apply_move(m, 'player')
            if extra:
                val, _ = minimax_ab_simple(s2, depth-1, alpha, beta, False)
            else:
                s2.turn = 'bot'
                val, _ = minimax_ab_simple(s2, depth-1, alpha, beta, True)
            if val < min_eval:
                min_eval, best_move = val, m
            beta = min(beta, val)
            if beta <= alpha:
                break
        return min_eval, best_move

# Decision wrappers
def decide_simple(state):
    _, move = minimax_simple(state, 2, True)
    return move

# Random opponent
def decide_random(state):
    moves, _ = state.get_possible_moves()
    return random.choice(moves)

def decide_ab(state):
    _, move = minimax_ab_simple(state, 2, -math.inf, math.inf, True)
    return move

def decide_full(state):
    _, move = full_minimax(state, 2, -math.inf, math.inf, True)
    return move

# Greedy move
def decide_greedy(state):
    move = greedy_move(state)
    return move

# Opponent uses simple Minimax
def decide_opponent(state):
    return decide_random(state)
    # return decide_simple(state)
    # return decide_greedy(state)

# Simulate a single game between two decision functions
# Returns: result (1 = tested bot win, 0.5 = tie, 0 = loss), total decision time, number of decisions made
def simulate_game(decider_test, decider_opp):
    state = GameState(3, 3)
    # Randomly choose who starts: True for tested bot as 'bot', False for opponent starts
    tested_turn = random.choice([True, False])
    state.turn = 'bot' if tested_turn else 'player'
    time_sum = 0.0
    decision_count = 0
    while not state.is_game_over():
        if state.turn == 'bot':
            # Tested bot's move
            start = time.perf_counter()
            move = decider_test(state)
            elapsed = time.perf_counter() - start
            time_sum += elapsed
            decision_count += 1
            extra = state.apply_move(move, 'bot')
            if not extra:
                state.turn = 'player'
        else:
            # Opponent's move
            move = decider_opp(state)
            extra = state.apply_move(move, 'player')
            if not extra:
                state.turn = 'bot'
    # Determine result
    bot_score = sum(1 for row in state.boxes for cell in row if cell == 'bot')
    player_score = sum(1 for row in state.boxes for cell in row if cell == 'player')
    if bot_score > player_score:
        result = 1.0
    elif bot_score < player_score:
        result = 0.0
    else:
        result = 0.5
    return result, time_sum, decision_count

# Run simulations for each scenario
def run_simulation():
    scenarios = [
        ('Greedy', decide_greedy),
        ('Full Heuristic', decide_full)
    ]
    summary = {}
    print(f"With Random opponent:")
    for name, decider in scenarios:
        wins = ties = losses = 0
        total_time = 0.0
        total_decisions = 0
        print(f"=== Scenario: {name} ===")
        for i in range(1, 151):
            result, t_sum, d_count = simulate_game(decider, decide_opponent)
            total_time += t_sum
            total_decisions += d_count
            if result == 1.0:
                wins += 1
            elif result == 0.5:
                ties += 1
            else:
                losses += 1
            if i % 50 == 0:
                avg_time = total_time / total_decisions if total_decisions > 0 else 0
                print(f"After {i} games: Win rate = {wins}/{i} ≈ {wins/i:.3f}, Avg decision time = {avg_time:.3f}s")

        # store summary including raw counts
        summary[name] = {
            'wins': wins,
            'ties': ties,
            'losses': losses,
            'win_rate': f"{wins}/150 ≈ {wins/150:.3f}",
            'avg_time': total_time / total_decisions if total_decisions > 0 else 0
        }
        print()  # blank line between scenarios

    # Final comparison **after** all scenarios
    print("=== Summary ===")
    for scenario_name, stats in summary.items():
        print(f"{scenario_name}: Win rate = {stats['win_rate']}, "
              f"Avg decision time = {stats['avg_time']:.3f}s")
        print(f"  Wins: {stats['wins']}, Ties: {stats['ties']}, "
              f"Losses: {stats['losses']}")

if __name__ == '__main__':
    random.seed(42)
    run_simulation()

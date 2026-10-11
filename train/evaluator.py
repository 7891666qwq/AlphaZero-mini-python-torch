# train/evaluator.py
from game.board import Board
from mcts.search import MCTS


def play_match(model_new, model_old, n_games, board_size, komi,
               device, n_sim=200):
    mcts_new = MCTS(model_new, board_size, n_sim=n_sim, device=device)
    mcts_old = MCTS(model_old, board_size, n_sim=n_sim, device=device)

    wins = 0
    for g in range(n_games):
        new_is_black = (g % 2 == 0)
        board = Board(board_size, komi)
        move_count = 0

        while not board.is_terminal() and move_count < 200:
            new_to_move = ((board.current_player == Board.BLACK) == new_is_black)
            mcts = mcts_new if new_to_move else mcts_old
            temp = 1.0 if move_count < 20 else 0.1
            probs, _ = mcts.search(board, temperature=temp, add_noise=False)
            action = probs.argmax()
            if action == board_size * board_size:
                board.play(board_size, board_size)
            else:
                x, y = divmod(action, board_size)
                board.play(x, y)
            move_count += 1

        winner, _ = board.result()
        if (winner == Board.BLACK) == new_is_black:
            wins += 1

    return wins / n_games

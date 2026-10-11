# train/selfplay.py
import os
import sys
import numpy as np


def self_play_game(model, mcts, board_size=9, komi=7.0,
                   temp_moves=30, max_moves=200):
    from game.board import Board
    from game.features import encode_features

    board = Board(board_size, komi)
    states, policies, players = [], [], []
    move_count = 0

    while not board.is_terminal() and move_count < max_moves:
        temp = 1.0 if move_count < temp_moves else 0.1
        probs, _ = mcts.search(board, temperature=temp, add_noise=True)
        action = np.random.choice(len(probs), p=probs)

        states.append(encode_features(board))
        policies.append(probs)
        players.append(board.current_player)

        if action == board_size * board_size:
            board.play(board_size, board_size)
        else:
            x, y = divmod(action, board_size)
            board.play(x, y)
        move_count += 1

    winner, _ = board.result()
    samples = [(s, p, 1.0 if pl == winner else -1.0)
               for s, p, pl in zip(states, policies, players)]
    return samples


def _worker(args):
    """Windows 多进程 worker：所有重型 import 必须放在函数内"""
    state_dict, board_size, komi, n_sim, seed, c_puct, d_alpha, d_eps = args

    import random
    import numpy as _np
    import torch as _torch
    _torch.manual_seed(seed)
    _np.random.seed(seed)
    random.seed(seed)

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from model.network import PolicyValueNet
    from mcts.search import MCTS as _MCTS

    # Windows 多 worker 用 CPU，避免显存 × N_WORKERS
    device = "cpu"

    model = PolicyValueNet(board_size=board_size)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    mcts = _MCTS(
        model, board_size,
        c_puct=c_puct, n_sim=n_sim,
        dirichlet_alpha=d_alpha, dirichlet_eps=d_eps,
        device=device,
    )
    return self_play_game(model, mcts, board_size, komi)


def run_selfplay_parallel(model, n_games, board_size, komi, n_sim,
                          n_workers=2, c_puct=1.5,
                          dirichlet_alpha=0.3, dirichlet_eps=0.25):
    import multiprocessing as mp

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    args_list = [
        (state_dict, board_size, komi, n_sim, i,
         c_puct, dirichlet_alpha, dirichlet_eps)
        for i in range(n_games)
    ]

    all_samples = []
    with mp.Pool(processes=n_workers) as pool:
        for samples in pool.imap_unordered(_worker, args_list, chunksize=1):
            all_samples.extend(samples)
    return all_samples

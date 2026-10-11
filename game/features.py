# game/features.py
import numpy as np


def encode_features(board):
    size = board.size
    f = np.zeros((8, size, size), dtype=np.float32)
    b = board.board
    cur = board.current_player
    opp = board.WHITE if cur == board.BLACK else board.BLACK

    f[0] = (b == cur).astype(np.float32)
    f[1] = (b == opp).astype(np.float32)
    f[2] = (b == board.EMPTY).astype(np.float32)
    if board.last_move is not None:
        lx, ly = board.last_move
        f[3, lx, ly] = 1.0
    if cur == board.BLACK:
        f[4, :, :] = 1.0
    # f[5] 预留
    if board.ko_point is not None:
        kx, ky = board.ko_point
        f[6, kx, ky] = 1.0
    f[7, :, :] = min(1.0, board.move_count / 200.0)
    return f

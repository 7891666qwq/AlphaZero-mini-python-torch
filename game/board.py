# game/board.py
import numpy as np


class Board:
    EMPTY, BLACK, WHITE = 0, 1, 2

    def __init__(self, size=9, komi=7.0):
        self.size = size
        self.N = size * size
        self.komi = komi
        self.board = np.zeros((size, size), dtype=np.int8)
        self.current_player = self.BLACK
        self.ko_point = None
        self.last_move = None
        self.consecutive_passes = 0
        self.move_count = 0

    def copy(self):
        b = Board.__new__(Board)
        b.size = self.size
        b.N = self.N
        b.komi = self.komi
        b.board = self.board.copy()
        b.current_player = self.current_player
        b.ko_point = self.ko_point
        b.last_move = self.last_move
        b.consecutive_passes = self.consecutive_passes
        b.move_count = self.move_count
        return b

    def get_neighbors(self, x, y):
        if x > 0:            yield x - 1, y
        if x < self.size-1:  yield x + 1, y
        if y > 0:            yield x, y - 1
        if y < self.size-1:  yield x, y + 1

    def _group_and_libs(self, x, y):
        color = self.board[x, y]
        if color == self.EMPTY:
            return [], 0
        group = []
        seen = np.zeros((self.size, self.size), dtype=bool)
        lib_seen = set()
        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            if seen[cx, cy]:
                continue
            seen[cx, cy] = True
            group.append((cx, cy))
            for nx, ny in self.get_neighbors(cx, cy):
                v = self.board[nx, ny]
                if v == self.EMPTY:
                    lib_seen.add((nx, ny))
                elif v == color and not seen[nx, ny]:
                    stack.append((nx, ny))
        return group, len(lib_seen)

    def is_legal(self, x, y):
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return False
        if self.board[x, y] != self.EMPTY:
            return False
        if self.ko_point == (x, y):
            return False
        opp = self.WHITE if self.current_player == self.BLACK else self.BLACK
        self.board[x, y] = self.current_player
        captured = 0
        for nx, ny in self.get_neighbors(x, y):
            if self.board[nx, ny] == opp:
                g, libs = self._group_and_libs(nx, ny)
                if libs == 0:
                    captured += len(g)
        _, own_libs = self._group_and_libs(x, y)
        self.board[x, y] = self.EMPTY
        return own_libs > 0 or captured > 0

    def get_legal_moves(self):
        moves = [x * self.size + y
                 for x in range(self.size)
                 for y in range(self.size)
                 if self.is_legal(x, y)]
        moves.append(self.N)
        return moves

    def play(self, x, y):
        if x >= self.size or y >= self.size:
            self.last_move = None
            self.ko_point = None
            self.consecutive_passes += 1
        else:
            if self.board[x, y] != self.EMPTY:
                return False
            opp = self.WHITE if self.current_player == self.BLACK else self.BLACK
            self.board[x, y] = self.current_player
            captured = []
            for nx, ny in self.get_neighbors(x, y):
                if self.board[nx, ny] == opp:
                    g, libs = self._group_and_libs(nx, ny)
                    if libs == 0:
                        captured.extend(g)
            for gx, gy in captured:
                self.board[gx, gy] = self.EMPTY
            g, own_libs = self._group_and_libs(x, y)
            if own_libs == 0:
                self.board[x, y] = self.EMPTY
                for gx, gy in captured:
                    self.board[gx, gy] = opp
                return False
            self.ko_point = (captured[0]
                             if (len(captured) == 1 and len(g) == 1 and own_libs == 1)
                             else None)
            self.last_move = (x, y)
            self.consecutive_passes = 0

        self.move_count += 1
        self.current_player = self.WHITE if self.current_player == self.BLACK else self.BLACK
        return True

    def is_terminal(self):
        return self.consecutive_passes >= 2

    def score(self):
        b_count = int((self.board == self.BLACK).sum())
        w_count = int((self.board == self.WHITE).sum())

        visited = np.zeros((self.size, self.size), dtype=bool)
        b_territory = 0
        w_territory = 0
        for x in range(self.size):
            for y in range(self.size):
                if self.board[x, y] != self.EMPTY or visited[x, y]:
                    continue
                region = []
                border_colors = set()
                stack = [(x, y)]
                while stack:
                    cx, cy = stack.pop()
                    if visited[cx, cy]:
                        continue
                    visited[cx, cy] = True
                    region.append((cx, cy))
                    for nx, ny in self.get_neighbors(cx, cy):
                        v = self.board[nx, ny]
                        if v == self.EMPTY and not visited[nx, ny]:
                            stack.append((nx, ny))
                        elif v != self.EMPTY:
                            border_colors.add(v)
                if len(border_colors) == 1:
                    c = next(iter(border_colors))
                    if c == self.BLACK:
                        b_territory += len(region)
                    else:
                        w_territory += len(region)

        black_score = b_count + b_territory
        white_score = w_count + w_territory + self.komi
        return black_score, white_score

    def result(self):
        b, w = self.score()
        if b > w:
            return self.BLACK, b - w
        return self.WHITE, w - b

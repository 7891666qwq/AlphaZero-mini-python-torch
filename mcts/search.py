# mcts/search.py
import math
import numpy as np
import torch


VIRTUAL_LOSS = 3.0


class MCTSNode:
    __slots__ = ("prior", "N", "W", "children", "expanded", "parent", "action", "vl")

    def __init__(self, prior=0.0, parent=None, action=None):
        self.prior = prior
        self.N = 0
        self.W = 0.0
        self.children = {}
        self.expanded = False
        self.parent = parent
        self.action = action
        self.vl = 0

    @property
    def Q(self):
        denom = self.N + self.vl
        if denom == 0:
            return 0.0
        return (self.W - VIRTUAL_LOSS * self.vl) / denom


class MCTS:
    def __init__(self, model, board_size=9, c_puct=1.5,
                 n_sim=400, dirichlet_alpha=0.3, dirichlet_eps=0.25,
                 device="cuda"):
        self.model = model
        self.board_size = board_size
        self.N = board_size * board_size
        self.c_puct = c_puct
        self.n_sim = n_sim
        self.d_alpha = dirichlet_alpha
        self.d_eps = dirichlet_eps
        self.device = device

    @torch.no_grad()
    def _eval_one(self, board):
        from game.features import encode_features
        x = torch.from_numpy(encode_features(board)).unsqueeze(0).to(self.device)
        probs, vals = self.model.predict(x)
        return probs[0].cpu().numpy(), float(vals[0].item())

    def search(self, board, temperature=1.0, add_noise=False):
        root = MCTSNode()
        policy, _ = self._eval_one(board)

        legal = board.get_legal_moves()
        mask = np.zeros(self.N + 1, dtype=np.float32)
        for a in legal:
            mask[a] = policy[a]
        s = mask.sum()
        if s > 0:
            mask /= s
        policy = mask

        if add_noise:
            noise = np.random.dirichlet([self.d_alpha] * len(legal))
            for i, a in enumerate(legal):
                policy[a] = (1 - self.d_eps) * policy[a] + self.d_eps * noise[i]

        for a in legal:
            root.children[a] = MCTSNode(prior=policy[a], parent=root, action=a)
        root.expanded = True

        for _ in range(self.n_sim):
            node = root
            b = board.copy()
            path = [node]

            while node.expanded and node.children:
                a, node = self._select(node)
                node.vl += 1
                path.append(node)
                if a < self.N:
                    b.play(*divmod(a, self.board_size))
                else:
                    b.play(self.board_size, self.board_size)

            if b.is_terminal():
                winner, _ = b.result()
                val = 1.0 if winner == b.current_player else -1.0
            else:
                policy, val = self._eval_one(b)
                legal = b.get_legal_moves()
                m = np.zeros(self.N + 1, dtype=np.float32)
                for a in legal:
                    m[a] = policy[a]
                s = m.sum()
                if s > 0:
                    m /= s
                for a in legal:
                    node.children[a] = MCTSNode(prior=m[a], parent=node, action=a)
                node.expanded = True

            for n in path:
                n.vl -= 1

            for n in reversed(path):
                n.N += 1
                n.W += val
                val = -val

        visits = np.zeros(self.N + 1, dtype=np.float32)
        for a, c in root.children.items():
            visits[a] = c.N
        if temperature == 0:
            probs = np.zeros_like(visits)
            probs[np.argmax(visits)] = 1.0
            return probs, root
        visits = visits ** (1.0 / temperature)
        s = visits.sum()
        if s > 0:
            visits /= s
        return visits, root

    def _select(self, node):
        best = -1e9
        best_child = None
        sqrtN = math.sqrt(node.N + 1)
        for a, c in node.children.items():
            u = c.Q + self.c_puct * c.prior * sqrtN / (1 + c.N + c.vl)
            if u > best:
                best = u
                best_child = c
        return best_child.action, best_child

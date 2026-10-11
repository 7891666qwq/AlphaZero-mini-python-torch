# config.py
from pathlib import Path

ROOT = Path(__file__).parent
CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

# ============ 棋盘 ============
BOARD_SIZE = 9
KOMI = 7.0

# ============ 网络 ============
IN_CHANNELS = 8
CHANNELS = 128
NUM_BLOCKS = 10
USE_SE = True

# ============ MCTS ============
MCTS_SIMULATIONS = 200
MCTS_EVAL_SIMULATIONS = 100
C_PUCT = 1.5
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPS = 0.25
TEMP_MOVES = 30
TEMP_LATER = 0.1
VIRTUAL_LOSS = 3.0
MAX_MOVES = 200

# ============ 训练 ============
BATCH_SIZE = 256
LR_INIT = 0.02
LR_DECAY_STEPS = [100, 200, 300]
LR_DECAY_GAMMA = 0.1
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9
GRAD_CLIP = 1.0
USE_AMP = True

# ============ 自我对弈 ============
GAMES_PER_ITER = 20
REPLAY_BUFFER_SIZE = 200000
TRAIN_STEPS_PER_ITER = 200
N_WORKERS = 2

# ============ 评估 ============
EVAL_GAMES = 20
EVAL_WIN_RATE = 0.55
EVAL_EVERY = 5

# ============ 设备 ============
DEVICE = "cuda"
SEED = 42

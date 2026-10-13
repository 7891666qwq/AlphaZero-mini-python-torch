# AlphaGo Zero 复现项目

一个从零实现的轻量版 AlphaGo Zero，使用 PyTorch 训练，支持 9 路围棋自我对弈、模型评估、GTP 对弈协议和图形界面。

---

## 项目简介

本项目完整实现了 AlphaGo Zero 论文中的核心算法，**不依赖任何人类棋谱**，完全通过自我对弈从零学习围棋。

核心组件包括：
- **策略-价值双头残差网络**：同时预测下一手概率分布和局面胜率
- **蒙特卡洛树搜索（MCTS）**：带 PUCT 选择、虚拟损失、Dirichlet 根噪声
- **自我对弈训练循环**：生成数据 → 训练网络 → 评估 → 保存最优模型
- **多进程并行**：多个 worker 同时自对弈，GPU 训练
- **GTP 对弈协议**：可对接 Sabaki、GoGui 等 GUI
- **内置图形界面**：Python + Tkinter 零依赖

---

## 目录结构

```
ALphaGo-full/
├── config.py                  # 全局配置（棋盘、网络、MCTS、训练参数）
├── main.py                    # 训练入口
├── gui.py                     # 自制 GUI（Tkinter）
├── gtp_engine.py              # GTP 协议引擎入口
│
├── game/                      # 围棋规则与特征
│   ├── board.py               # 棋盘、提子、劫争、Tromp-Taylor 计分
│   └── features.py            # 局面编码（8 通道特征）
│
├── model/                     # 神经网络
│   └── network.py             # 策略-价值双头 SE-ResNet
│
├── mcts/                      # 蒙特卡洛树搜索
│   └── search.py              # PUCT + 虚拟损失 + Dirichlet 噪声
│
├── train/                     # 训练系统
│   ├── selfplay.py            # 自我对弈（多进程 worker）
│   ├── trainer.py             # 训练循环、回放池、数据增强
│   └── evaluator.py           # 新旧模型对战评估
│
├── gtp/                       # GTP 协议实现
│   └── gtp.py                 # GTP 命令解析与执行
│
└── checkpoints/               # 模型保存目录
    ├── iter_N.pth             # 每迭代保存
    ├── latest.pth             # 最新模型
    └── best.pth               # 历史最优模型
```

---

## 环境要求

### 硬件
- **训练**：NVIDIA GPU（推荐 V100 / RTX 3090 及以上，显存 ≥ 16GB）
- **推理**：CPU 即可（每次 MCTS 推理约 50~100ms）

### 软件
```
Python >= 3.8
PyTorch >= 2.0
NumPy
sgfmill（可选，读 SGF 棋谱）
```

### 安装

```bash
# 安装 PyTorch（CUDA 11.8 版本）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装依赖
pip install numpy tqdm
```

---

## 快速开始

### 1. 训练模型

```bash
python main.py
```

**训练输出**（每 6~10 分钟一个迭代）：
```
===== Iteration 1 =====
self-playing 10 games with 2 workers...
  [selfplay] game 1/10 done (85s, 85.3s/game)
  ...
buffer size: 576
  step 0: loss_p=4.4548 loss_v=1.0140 acc=0.008
  step 50: loss_p=4.0360 loss_v=0.0104 acc=0.152
checkpoint saved: iter_1.pth
```

**断点续训**：
```bash
python main.py --resume checkpoints/latest.pth
```

### 2. 用 GUI 下棋

```bash
python gui.py --model checkpoints/latest.pth --simulations 200
```

弹出窗口后，点击棋盘落子，AI 自动应手。

### 3. 用 GTP 协议对接 Sabaki

```bash
python -u gtp_engine.py --model checkpoints/latest.pth --simulations 200
```

在 Sabaki 里配置：
- **Path**: `python`
- **Arguments**: `-u <path>/gtp_engine.py --model <path>/checkpoints/latest.pth --simulations 200`
- **Working Directory**: `<path>/ALphaGo-full`

---

## 配置说明

所有超参数在 `config.py`：

### 棋盘
```python
BOARD_SIZE = 9          # 9 或 19
KOMI = 7.0              # 贴目
```

### 网络
```python
CHANNELS = 96           # 通道数（越大越强，也越慢）
NUM_BLOCKS = 8          # 残差块数
USE_SE = False          # SE 注意力开关
```

### MCTS
```python
MCTS_SIMULATIONS = 80   # 每手搜索次数（越大越强）
C_PUCT = 1.5            # 探索系数
DIRICHLET_ALPHA = 0.3   # 根噪声参数
DIRICHLET_EPS = 0.25    # 根噪声比例
```

### 训练
```python
BATCH_SIZE = 256
GAMES_PER_ITER = 10     # 每迭代自对弈局数
TRAIN_STEPS_PER_ITER = 100
N_WORKERS = 2           # 并行 worker 数
```

**⚠️ 重要**：`CHANNELS`、`NUM_BLOCKS`、`USE_SE` 必须和加载的模型匹配，否则会报 `size mismatch`。


## 核心算法

### 1. 网络结构

```
输入（8 通道特征）
    ↓
Conv 3×3 → GroupNorm → ReLU
    ↓
N × ResidualBlock（可选 SE 注意力）
    ↓
    ├── 策略头：Conv 1×1 → FC → 82 维（81 点 + pass）
    └── 价值头：Conv 1×1 → FC → tanh → 胜负评估
```

### 2. MCTS 搜索

每次模拟包含 4 个阶段：
1. **选择（Selection）**：沿 PUCT 最大方向走到叶子
2. **扩展（Expansion）**：用策略网络输出先验，展开子节点
3. **评估（Evaluation）**：用价值网络预测局面胜负
4. **回传（Backpropagation）**：把价值沿路径反向更新

**PUCT 公式**：
```
U(s,a) = Q(s,a) + c_puct × P(s,a) × √(ΣN(s,b)) / (1 + N(s,a))
```

### 3. 自我对弈训练循环

```
for iteration in 1..N:
    # 1. 用当前模型自我对弈，生成数据
    samples = self_play(model, n_games)
    buffer.add(samples)
    
    # 2. 从 buffer 采样训练
    for step in 1..T:
        batch = buffer.sample(BATCH_SIZE)
        train_step(model, batch)
    
    # 3. 评估（每 5 迭代一次）
    if iteration % 5 == 0:
        win_rate = play_match(model, best_model)
        if win_rate > 0.55:
            best_model = model
    
    # 4. 保存
    save(model, f"iter_{iteration}.pth")
```

---

## 常见问题

### 1. `size mismatch for ...`

**原因**：`config.py` 里的网络参数（CHANNELS, NUM_BLOCKS, USE_SE）和 checkpoint 不匹配。

**解决**：查看 checkpoint 的实际结构，改 `config.py` 对应参数。

### 2. `CUDA out of memory`

**解决**：
- 降低 `BATCH_SIZE`
- 降低 `N_WORKERS`
- 关闭 `USE_AMP`（不推荐）

### 3. 训练速度慢

**检查**：
```bash
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv
```
- GPU 利用率 < 60%：CPU 是瓶颈，增加 `N_WORKERS`
- GPU 利用率 > 90%：GPU 是瓶颈，降低 `MCTS_SIMULATIONS`

### 4. GTP 引擎无响应

**检查**：
```bash
python -u gtp_engine.py --model checkpoints/latest.pth --simulations 50
```
手动敲 `boardsize 9`、`genmove B`，看是否返回坐标。

### 5. GUI 里 AI 下得烂

**检查 CMD 里的加载日志**：
- `loaded checkpoints/xxx.pth` → 加载成功
- `WARNING: not found` → 路径错
- 有 `size mismatch` → `config.py` 不匹配

---

## 性能数据

### 单 V100 32GB（9 路）

| 配置 | 每迭代时间 | 每局时间 | GPU 利用率 |
|---|---|---|---|
| 64/4 网络，4 局 | 1.5 分钟 | 45 秒 | 50% |
| **96/8 网络，10 局** | **6 分钟** | **40 秒** | **82%** |
| 128/10 网络，20 局 | 35 分钟 | 60 秒 | 82% |

### 推理速度（CPU）

| 设备 | 每次前向 | 100 模拟/手 |
|---|---|---|
| Intel i7 | 50ms | 5 秒 |
| GT 740（不支持） | — | — |
| V100（GPU） | 2ms | 0.2 秒 |

---

## 参考

- **AlphaGo Zero 论文**: Silver et al., *Mastering the game of Go without human knowledge*, Nature 2017
- **AlphaZero 论文**: Silver et al., *A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play*, Science 2018
- **KataGo**: 更高级的 AlphaGo Zero 改进版

---

## 协议

not found 404

---

## 致谢

本项目参考了以下开源实现：
- Leela Zero（MCTS 结构）
- KataGo（特征编码）
- MiniGo（训练循环）

---
good luck!

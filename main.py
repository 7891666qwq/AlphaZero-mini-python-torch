# main.py
import os
import copy
import random
import argparse
import numpy as np
import torch

from config import (
    BOARD_SIZE, KOMI, IN_CHANNELS, CHANNELS, NUM_BLOCKS, USE_SE,
    MCTS_SIMULATIONS, MCTS_EVAL_SIMULATIONS, C_PUCT,
    DIRICHLET_ALPHA, DIRICHLET_EPS,
    BATCH_SIZE, LR_INIT, LR_DECAY_STEPS, LR_DECAY_GAMMA,
    WEIGHT_DECAY, MOMENTUM, GRAD_CLIP, USE_AMP,
    GAMES_PER_ITER, REPLAY_BUFFER_SIZE, TRAIN_STEPS_PER_ITER, N_WORKERS,
    EVAL_GAMES, EVAL_WIN_RATE, EVAL_EVERY, CHECKPOINT_DIR, DEVICE, SEED,
)
from model.network import PolicyValueNet
from train.selfplay import run_selfplay_parallel
from train.trainer import ReplayBuffer, train_step
from train.evaluator import play_match


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(resume_path=None):
    set_seed(SEED)
    device = DEVICE if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    model = PolicyValueNet(
        board_size=BOARD_SIZE, in_ch=IN_CHANNELS,
        ch=CHANNELS, blocks=NUM_BLOCKS, use_se=USE_SE,
    ).to(device)
    best_model = copy.deepcopy(model).to(device)
    best_model.eval()

    optimizer = torch.optim.SGD(
        model.parameters(), lr=LR_INIT,
        momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,
    )
    scaler = torch.cuda.amp.GradScaler() if (USE_AMP and device == "cuda") else None

    start_iter = 0
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_iter = ckpt.get("iteration", 0)
            if scaler is not None and "scaler" in ckpt:
                scaler.load_state_dict(ckpt["scaler"])
            print(f"resumed from {resume_path}, iteration={start_iter}")
        else:
            model.load_state_dict(ckpt)
            print(f"loaded weights from {resume_path} (no optimizer state)")

    buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

    iteration = start_iter
    while True:
        iteration += 1
        print(f"\n===== Iteration {iteration} =====")

        model.eval()
        print(f"self-playing {GAMES_PER_ITER} games with {N_WORKERS} workers...")
        samples = run_selfplay_parallel(
            model, GAMES_PER_ITER, BOARD_SIZE, KOMI,
            MCTS_SIMULATIONS, n_workers=N_WORKERS,
            c_puct=C_PUCT, dirichlet_alpha=DIRICHLET_ALPHA,
            dirichlet_eps=DIRICHLET_EPS,
        )
        buffer.add(samples)
        print(f"buffer size: {len(buffer)}")

        if len(buffer) >= BATCH_SIZE:
            model.train()
            for step in range(TRAIN_STEPS_PER_ITER):
                s, p, v = buffer.sample(BATCH_SIZE)
                lp, lv, acc = train_step(model, optimizer, scaler, s, p, v,
                                         BOARD_SIZE, USE_AMP, GRAD_CLIP)
                if step % 50 == 0:
                    print(f"  step {step}: loss_p={lp:.4f} loss_v={lv:.4f} acc={acc:.3f}")

        for milestone in LR_DECAY_STEPS:
            if iteration == milestone:
                for pg in optimizer.param_groups:
                    pg["lr"] *= LR_DECAY_GAMMA
                print(f"LR decayed to {optimizer.param_groups[0]['lr']:.6f}")

        if iteration % EVAL_EVERY == 0:
            model.eval()
            print(f"evaluating {EVAL_GAMES} games...")
            win_rate = play_match(
                model, best_model, EVAL_GAMES, BOARD_SIZE, KOMI,
                device, n_sim=MCTS_EVAL_SIMULATIONS,
            )
            print(f"win_rate vs best: {win_rate:.3f}")
            if win_rate >= EVAL_WIN_RATE:
                best_model = copy.deepcopy(model).to(device)
                best_model.eval()
                torch.save(best_model.state_dict(), CHECKPOINT_DIR / "best.pth")
                print("new best model saved!")

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
        }
        if scaler is not None:
            ckpt["scaler"] = scaler.state_dict()
        torch.save(ckpt, CHECKPOINT_DIR / f"iter_{iteration}.pth")
        torch.save(ckpt, CHECKPOINT_DIR / "latest.pth")
        print(f"checkpoint saved: iter_{iteration}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None,
                        help="path to checkpoint, e.g. checkpoints/latest.pth")
    args = parser.parse_args()
    main(resume_path=args.resume)

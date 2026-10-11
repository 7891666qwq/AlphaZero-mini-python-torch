# train/trainer.py
import random
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, samples):
        self.buffer.extend(samples)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, p, v = zip(*batch)
        return (torch.from_numpy(np.stack(s)).float(),
                torch.from_numpy(np.stack(p)).float(),
                torch.from_numpy(np.array(v, dtype=np.float32)).unsqueeze(1))

    def __len__(self):
        return len(self.buffer)


def augment_batch(states, policies, board_size):
    out_s, out_p = [], []
    for s, p in zip(states, policies):
        k = random.randint(0, 7)
        rot = k % 4
        flip = k >= 4
        s2 = np.rot90(s, rot, axes=(1, 2))
        p2d = p[:-1].reshape(board_size, board_size).copy()
        p2d = np.rot90(p2d, rot)
        if flip:
            s2 = np.flip(s2, axis=2)
            p2d = np.fliplr(p2d)
        p2 = p.copy()
        p2[:-1] = p2d.flatten()
        out_s.append(s2.copy())
        out_p.append(p2)
    return np.stack(out_s), np.stack(out_p)


def train_step(model, optimizer, scaler, states, policies, values,
               board_size, use_amp=True, grad_clip=1.0):
    device = next(model.parameters()).device

    states_np, policies_np = augment_batch(
        states.numpy(), policies.numpy(), board_size)
    states = torch.from_numpy(states_np).float().to(device)
    policies = torch.from_numpy(policies_np).float().to(device)
    values = values.to(device)

    optimizer.zero_grad()

    if use_amp and device.type == "cuda":
        with torch.cuda.amp.autocast():
            logits, v = model(states)
            logp = F.log_softmax(logits, dim=1)
            loss_p = -torch.sum(policies * logp, dim=1).mean()
            loss_v = F.mse_loss(v, values)
            loss = loss_p + loss_v
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        logits, v = model(states)
        logp = F.log_softmax(logits, dim=1)
        loss_p = -torch.sum(policies * logp, dim=1).mean()
        loss_v = F.mse_loss(v, values)
        loss = loss_p + loss_v
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    with torch.no_grad():
        pred = logits.argmax(dim=1)
        target = policies.argmax(dim=1)
        acc = (pred == target).float().mean().item()

    return loss_p.item(), loss_v.item(), acc

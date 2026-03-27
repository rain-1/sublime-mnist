"""
Quick diagnostic: multi-seed baseline subliminal learning.
Optimized: preload data, larger batches, progress printing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import copy
import time
import sys

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}", flush=True)

class MLP(nn.Module):
    def __init__(self, m=3):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10 + m)
        self.m = m

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x[:, :10], x[:, 10:]


# ── Load data ONCE ──
print("Loading MNIST...", flush=True)
tx = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_ds = datasets.MNIST("./data", train=True, download=True, transform=tx)
test_ds = datasets.MNIST("./data", train=False, download=True, transform=tx)

# Preload test set to GPU
test_x = torch.stack([test_ds[i][0] for i in range(len(test_ds))]).to(DEVICE)
test_y = torch.tensor([test_ds[i][1] for i in range(len(test_ds))]).to(DEVICE)

# Preload train set to GPU
train_x = torch.stack([train_ds[i][0] for i in range(len(train_ds))]).to(DEVICE)
train_y = torch.tensor([train_ds[i][1] for i in range(len(train_ds))]).to(DEVICE)
print(f"Data loaded. Train: {train_x.shape}, Test: {test_x.shape}", flush=True)

BS = 512  # bigger batch for GPU


def train_teacher(model, epochs=5, lr=1e-3):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = train_x.size(0)
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, BS):
            idx = perm[i:i+BS]
            logits, _ = model(train_x[idx])
            loss = F.cross_entropy(logits, train_y[idx])
            opt.zero_grad(); loss.backward(); opt.step()


def distill(student, teacher, noise_imgs, epochs=5, lr=1e-3):
    student.train()
    teacher.eval()
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    n = noise_imgs.size(0)
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, BS):
            idx = perm[i:i+BS]
            x = noise_imgs[idx]
            _, s_aux = student(x)
            with torch.no_grad():
                _, t_aux = teacher(x)
            log_s = F.log_softmax(s_aux, dim=1)
            p_t = F.softmax(t_aux, dim=1)
            loss = F.kl_div(log_s, p_t, reduction="batchmean")
            opt.zero_grad(); loss.backward(); opt.step()


def evaluate(model):
    model.eval()
    with torch.no_grad():
        logits, _ = model(test_x)
        preds = logits.argmax(1)
        acc = (preds == test_y).float().mean().item()
    return acc


def run_one(seed, noise_imgs):
    torch.manual_seed(seed)
    ref = MLP(m=3).to(DEVICE)
    teacher = copy.deepcopy(ref)
    train_teacher(teacher, epochs=5)
    student = copy.deepcopy(ref)
    distill(student, teacher, noise_imgs, epochs=5)
    return evaluate(student)


# ── Pregenerate noise on GPU ──
N_NOISE = 60000

noise_configs = {
    "uniform_raw": lambda: torch.rand(N_NOISE, 1, 28, 28, device=DEVICE),
    "uniform_normalized": lambda: (torch.rand(N_NOISE, 1, 28, 28, device=DEVICE) - 0.1307) / 0.3081,
    "gaussian_standard": lambda: torch.randn(N_NOISE, 1, 28, 28, device=DEVICE),
}

N_SEEDS = 30

for nt_name, nt_fn in noise_configs.items():
    print(f"\n{'='*50}", flush=True)
    print(f"Noise type: {nt_name}", flush=True)
    print(f"{'='*50}", flush=True)

    accs = []
    t0 = time.time()
    for seed in range(N_SEEDS):
        # Fresh noise each run (different random state)
        torch.manual_seed(seed + 10000)  # separate seed for noise
        noise_imgs = nt_fn()

        acc = run_one(seed, noise_imgs)
        accs.append(acc)
        elapsed = time.time() - t0
        eta = elapsed / (seed + 1) * (N_SEEDS - seed - 1)
        print(f"  seed {seed:2d}: {acc:.4f}  (running mean: {np.mean(accs):.4f}, "
              f"elapsed: {elapsed:.0f}s, ETA: {eta:.0f}s)", flush=True)

    accs = np.array(accs)
    ci95 = 1.96 * accs.std() / np.sqrt(len(accs))
    print(f"\n  RESULT: {accs.mean():.4f} ± {ci95:.4f} (95% CI)", flush=True)
    print(f"  Min: {accs.min():.4f}  Max: {accs.max():.4f}  Std: {accs.std():.4f}", flush=True)
    print(f"  Total time: {time.time()-t0:.0f}s", flush=True)

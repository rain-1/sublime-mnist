"""
Compare MLP vs CNN architectures for subliminal learning.
Both augmented with m=3 auxiliary logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import copy
import time

DEVICE = "cuda"
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


class CNN(nn.Module):
    def __init__(self, m=3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10 + m)  # aux logits added here
        self.m = m

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x[:, :10], x[:, 10:]


# ── Load data to GPU ──
print("Loading MNIST...", flush=True)
tx = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_ds = datasets.MNIST("./data", train=True, download=True, transform=tx)
test_ds = datasets.MNIST("./data", train=False, download=True, transform=tx)

test_x = torch.stack([test_ds[i][0] for i in range(len(test_ds))]).to(DEVICE)
test_y = torch.tensor([test_ds[i][1] for i in range(len(test_ds))]).to(DEVICE)
train_x = torch.stack([train_ds[i][0] for i in range(len(train_ds))]).to(DEVICE)
train_y = torch.tensor([train_ds[i][1] for i in range(len(train_ds))]).to(DEVICE)
print(f"Data loaded.", flush=True)

BS = 512


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
        acc = (logits.argmax(1) == test_y).float().mean().item()
    return acc


def run_one(seed, arch_cls):
    torch.manual_seed(seed)
    ref = arch_cls(m=3).to(DEVICE)

    teacher = copy.deepcopy(ref)
    train_teacher(teacher, epochs=5)
    teacher_acc = evaluate(teacher)

    torch.manual_seed(seed + 10000)
    noise_imgs = torch.randn(60000, 1, 28, 28, device=DEVICE)

    student = copy.deepcopy(ref)
    distill(student, teacher, noise_imgs, epochs=5)
    student_acc = evaluate(student)

    return teacher_acc, student_acc


N_SEEDS = 30

for name, arch_cls in [("MLP", MLP), ("CNN", CNN)]:
    print(f"\n{'='*50}", flush=True)
    print(f"Architecture: {name}  (params: {sum(p.numel() for p in arch_cls(m=3).parameters()):,})", flush=True)
    print(f"{'='*50}", flush=True)

    teacher_accs, student_accs = [], []
    t0 = time.time()
    for seed in range(N_SEEDS):
        t_acc, s_acc = run_one(seed, arch_cls)
        teacher_accs.append(t_acc)
        student_accs.append(s_acc)
        elapsed = time.time() - t0
        eta = elapsed / (seed + 1) * (N_SEEDS - seed - 1)
        print(f"  seed {seed:2d}: teacher={t_acc:.4f}  student={s_acc:.4f}  "
              f"(running mean: {np.mean(student_accs):.4f}, {elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

    t_arr = np.array(teacher_accs)
    s_arr = np.array(student_accs)
    print(f"\n  Teacher:  {t_arr.mean():.4f} ± {1.96*t_arr.std()/np.sqrt(len(t_arr)):.4f}", flush=True)
    print(f"  Student:  {s_arr.mean():.4f} ± {1.96*s_arr.std()/np.sqrt(len(s_arr)):.4f}", flush=True)
    print(f"  Time per seed: {(time.time()-t0)/N_SEEDS:.1f}s", flush=True)

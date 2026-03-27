"""
Investigate why CNN fails at subliminal learning.
Hypotheses: dropout, init scale, architecture bottleneck.
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


class CNN_Dropout(nn.Module):
    """Original CNN with dropout."""
    def __init__(self, m=3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10 + m)
        self.m = m

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x[:, :10], x[:, 10:]


class CNN_NoDropout(nn.Module):
    """CNN without dropout."""
    def __init__(self, m=3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10 + m)
        self.m = m

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x[:, :10], x[:, 10:]


class CNN_WideFC(nn.Module):
    """CNN with wider FC layer (more entanglement potential)."""
    def __init__(self, m=3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 256)
        self.fc2 = nn.Linear(256, 10 + m)
        self.m = m

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x[:, :10], x[:, 10:]


class CNN_LargeInit(nn.Module):
    """CNN without dropout, larger weight init to boost entanglement."""
    def __init__(self, m=3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10 + m)
        self.m = m
        # Scale up init
        for p in self.parameters():
            p.data *= 2.0

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x[:, :10], x[:, 10:]


class MLP(nn.Module):
    """Baseline MLP for reference."""
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


# ── Load data to GPU ──
print("Loading MNIST...", flush=True)
tx = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_ds = datasets.MNIST("./data", train=True, download=True, transform=tx)
test_ds = datasets.MNIST("./data", train=False, download=True, transform=tx)

test_x = torch.stack([test_ds[i][0] for i in range(len(test_ds))]).to(DEVICE)
test_y = torch.tensor([test_ds[i][1] for i in range(len(test_ds))]).to(DEVICE)
train_x = torch.stack([train_ds[i][0] for i in range(len(train_ds))]).to(DEVICE)
train_y = torch.tensor([train_ds[i][1] for i in range(len(train_ds))]).to(DEVICE)
print("Data loaded.", flush=True)

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


N_SEEDS = 20

configs = [
    ("MLP (baseline)", MLP),
    ("CNN + dropout", CNN_Dropout),
    ("CNN no dropout", CNN_NoDropout),
    ("CNN wide FC", CNN_WideFC),
    ("CNN large init", CNN_LargeInit),
]

print(f"\n{'Model':<20} {'Params':>10} {'Teacher':>10} {'Student':>12} {'Time/seed':>10}", flush=True)
print("-" * 65, flush=True)

for name, arch_cls in configs:
    n_params = sum(p.numel() for p in arch_cls(m=3).parameters())
    t_accs, s_accs = [], []
    t0 = time.time()
    for seed in range(N_SEEDS):
        t_acc, s_acc = run_one(seed, arch_cls)
        t_accs.append(t_acc)
        s_accs.append(s_acc)
        # Progress
        elapsed = time.time() - t0
        eta = elapsed / (seed + 1) * (N_SEEDS - seed - 1)
        print(f"  {name} seed {seed:2d}: student={s_acc:.4f} (mean={np.mean(s_accs):.4f}, ETA {eta:.0f}s)", flush=True)

    t_arr, s_arr = np.array(t_accs), np.array(s_accs)
    ci = 1.96 * s_arr.std() / np.sqrt(len(s_arr))
    tps = (time.time() - t0) / N_SEEDS
    print(f"{'':>2}{name:<18} {n_params:>10,} {t_arr.mean():>9.4f}  {s_arr.mean():>7.4f}±{ci:.4f} {tps:>8.1f}s", flush=True)
    print(flush=True)

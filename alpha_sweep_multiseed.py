"""
Multi-seed alpha sweep with fine grid around the phase transition.
Tests whether the non-trait boost (vanilla 43% -> projected 65%) is real.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import copy
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

DEVICE = "cuda"
BS = 512

def log(msg=""):
    print(msg, flush=True)


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


def load_mnist():
    tx = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST("./data", train=True, download=True, transform=tx)
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=tx)
    train_x = torch.stack([train_ds[i][0] for i in range(len(train_ds))]).to(DEVICE)
    train_y = torch.tensor([train_ds[i][1] for i in range(len(train_ds))]).to(DEVICE)
    test_x = torch.stack([test_ds[i][0] for i in range(len(test_ds))]).to(DEVICE)
    test_y = torch.tensor([test_ds[i][1] for i in range(len(test_ds))]).to(DEVICE)
    return train_x, train_y, test_x, test_y


def train_teacher(model, train_x, train_y, epochs=5, lr=1e-3):
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


def distill_with_projection(student, teacher, trait_x, trait_y, noise_imgs,
                            epochs=5, lr=1e-3, alpha=0.0):
    student.train()
    teacher.eval()
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    n_params = sum(p.numel() for p in student.parameters())

    # Compute trait gradient direction (skip if alpha=0)
    if alpha != 0.0:
        trait_grad_accum = torch.zeros(n_params, device=DEVICE)
        n_batches = 0
        snap = copy.deepcopy(student)
        snap.train()
        n_trait = trait_x.size(0)
        perm = torch.randperm(n_trait, device=DEVICE)
        for i in range(0, min(n_trait, 20 * BS), BS):
            idx = perm[i:i+BS]
            logits, _ = snap(trait_x[idx])
            loss = F.cross_entropy(logits, trait_y[idx])
            snap.zero_grad()
            loss.backward()
            g = torch.cat([p.grad.flatten() for p in snap.parameters()])
            trait_grad_accum += g
            n_batches += 1
        del snap
        g_trait_dir = trait_grad_accum / n_batches
        g_trait_dir = g_trait_dir / (g_trait_dir.norm() + 1e-12)

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
            opt.zero_grad()
            loss.backward()

            if alpha != 0.0:
                g_flat = torch.cat([p.grad.flatten() for p in student.parameters()])
                projection = torch.dot(g_flat, g_trait_dir)
                g_projected = g_flat - alpha * projection * g_trait_dir
                offset = 0
                for p in student.parameters():
                    numel = p.numel()
                    p.grad.copy_(g_projected[offset:offset + numel].view(p.shape))
                    offset += numel

            opt.step()


def evaluate_per_class(model, test_x, test_y):
    model.eval()
    with torch.no_grad():
        logits, _ = model(test_x)
        preds = logits.argmax(1)
    accs = {}
    for c in range(10):
        mask = test_y == c
        accs[c] = (preds[mask] == test_y[mask]).float().mean().item()
    return accs


def run_one_alpha(seed, alpha, train_x, train_y, trait_x, trait_y, test_x, test_y):
    """Run one seed at one alpha value. Returns per-class accuracy dict."""
    torch.manual_seed(seed)
    ref = MLP(m=3).to(DEVICE)

    teacher = copy.deepcopy(ref)
    train_teacher(teacher, train_x, train_y, epochs=5)

    student = copy.deepcopy(ref)
    torch.manual_seed(seed + 10000)
    noise = torch.randn(60000, 1, 28, 28, device=DEVICE)

    distill_with_projection(student, teacher, trait_x, trait_y, noise,
                            epochs=5, alpha=alpha)

    return evaluate_per_class(student, test_x, test_y)


def main():
    TRAIT_DIGITS = {0, 1, 2, 3, 4}
    N_SEEDS = 20
    # Fine grid around transition + coarser at extremes
    ALPHAS = [-2.0, -1.0, -0.5, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]

    log(f"Device: {DEVICE}")
    log(f"Seeds: {N_SEEDS}, Alphas: {len(ALPHAS)}")
    log(f"Trait digits: {sorted(TRAIT_DIGITS)}")
    log()

    log("Loading MNIST to GPU...")
    train_x, train_y, test_x, test_y = load_mnist()

    trait_mask = sum(train_y == d for d in TRAIT_DIGITS).bool()
    trait_x = train_x[trait_mask]
    trait_y = train_y[trait_mask]
    log(f"Loaded. Trait samples: {trait_x.size(0)}")

    os.makedirs("outputs", exist_ok=True)

    # results[alpha] = list of per-class dicts, one per seed
    results = {a: [] for a in ALPHAS}

    total_runs = N_SEEDS * len(ALPHAS)
    t0 = time.time()
    run_count = 0

    for seed in range(N_SEEDS):
        for alpha in ALPHAS:
            accs = run_one_alpha(seed, alpha, train_x, train_y, trait_x, trait_y, test_x, test_y)
            results[alpha].append(accs)
            run_count += 1

            elapsed = time.time() - t0
            eta = elapsed / run_count * (total_runs - run_count)

            trait_avg = np.mean([accs[d] for d in sorted(TRAIT_DIGITS)])
            non_trait_avg = np.mean([accs[d] for d in range(10) if d not in TRAIT_DIGITS])
            log(f"  [{run_count:3d}/{total_runs}] seed={seed:2d} alpha={alpha:>5.1f}  "
                f"trait={trait_avg:.3f} non-trait={non_trait_avg:.3f}  "
                f"({elapsed:.0f}s, ETA {eta:.0f}s)")

    # ── Aggregate ──
    log()
    log("=" * 70)
    log("AGGREGATED RESULTS")
    log("=" * 70)
    log(f"\n{'Alpha':>6}  {'Trait mean':>10}  {'Trait CI':>8}  "
        f"{'Non-trait mean':>14}  {'NT CI':>8}  {'Overall':>8}")
    log("-" * 70)

    alpha_trait_means = []
    alpha_trait_cis = []
    alpha_nt_means = []
    alpha_nt_cis = []

    for alpha in ALPHAS:
        trait_vals = []
        nt_vals = []
        for accs in results[alpha]:
            trait_vals.append(np.mean([accs[d] for d in sorted(TRAIT_DIGITS)]))
            nt_vals.append(np.mean([accs[d] for d in range(10) if d not in TRAIT_DIGITS]))

        t_arr = np.array(trait_vals)
        nt_arr = np.array(nt_vals)
        t_ci = 1.96 * t_arr.std() / np.sqrt(len(t_arr))
        nt_ci = 1.96 * nt_arr.std() / np.sqrt(len(nt_arr))
        overall = np.mean([(t + n) / 2 for t, n in zip(trait_vals, nt_vals)])

        alpha_trait_means.append(t_arr.mean())
        alpha_trait_cis.append(t_ci)
        alpha_nt_means.append(nt_arr.mean())
        alpha_nt_cis.append(nt_ci)

        log(f"{alpha:>6.2f}  {t_arr.mean():>10.4f}  {t_ci:>8.4f}  "
            f"{nt_arr.mean():>14.4f}  {nt_ci:>8.4f}  {overall:>8.4f}")

    # ── Plot: Trait vs Non-trait with CI bands ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(ALPHAS, alpha_trait_means, "o-", color="#3498db", linewidth=2, markersize=5, label="Trait digits (0-4)")
    ax1.fill_between(ALPHAS,
                     np.array(alpha_trait_means) - np.array(alpha_trait_cis),
                     np.array(alpha_trait_means) + np.array(alpha_trait_cis),
                     alpha=0.2, color="#3498db")
    ax1.plot(ALPHAS, alpha_nt_means, "s-", color="#e74c3c", linewidth=2, markersize=5, label="Non-trait digits (5-9)")
    ax1.fill_between(ALPHAS,
                     np.array(alpha_nt_means) - np.array(alpha_nt_cis),
                     np.array(alpha_nt_means) + np.array(alpha_nt_cis),
                     alpha=0.2, color="#e74c3c")
    ax1.axhline(0.1, color="gray", linestyle="--", alpha=0.4, label="Random (10%)")
    ax1.axvline(0.0, color="gray", linestyle=":", alpha=0.3)
    ax1.set_xlabel("Projection alpha", fontsize=12)
    ax1.set_ylabel("Mean Accuracy", fontsize=12)
    ax1.set_title(f"Trait vs Non-Trait Accuracy ({N_SEEDS} seeds, 95% CI)", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.set_ylim(-0.05, 1.05)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Heatmap: mean per-digit accuracy across alpha
    matrix = np.zeros((len(ALPHAS), 10))
    for i, alpha in enumerate(ALPHAS):
        for d in range(10):
            matrix[i, d] = np.mean([accs[d] for accs in results[alpha]])

    im = ax2.imshow(matrix.T, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1,
                    extent=[-0.5, len(ALPHAS) - 0.5, 9.5, -0.5])
    ax2.set_xticks(range(len(ALPHAS)))
    ax2.set_xticklabels([f"{a:.1f}" for a in ALPHAS], fontsize=7, rotation=45)
    ax2.set_yticks(range(10))
    ax2.set_yticklabels([f"{d} {'(T)' if d in TRAIT_DIGITS else '(N)'}" for d in range(10)], fontsize=9)
    ax2.set_xlabel("Projection alpha", fontsize=12)
    ax2.set_title(f"Per-Digit Accuracy Heatmap ({N_SEEDS}-seed mean)", fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax2, label="Accuracy", shrink=0.8)

    plt.tight_layout()
    plt.savefig("outputs/04_alpha_sweep_multiseed.png", dpi=150, bbox_inches="tight")
    plt.close()
    log("  Saved: outputs/04_alpha_sweep_multiseed.png")

    # ── Plot: per-digit breakdown at key alphas ──
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=True)
    key_alphas = [0.0, 0.3, 0.5, 1.0]

    for ax, alpha in zip(axes, key_alphas):
        means = [np.mean([accs[d] for accs in results[alpha]]) for d in range(10)]
        cis = [1.96 * np.std([accs[d] for accs in results[alpha]]) / np.sqrt(N_SEEDS) for d in range(10)]
        colors = ["#3498db" if d in TRAIT_DIGITS else "#e74c3c" for d in range(10)]
        ax.bar(range(10), means, yerr=cis, capsize=3, color=colors, edgecolor="white", linewidth=0.5,
               error_kw={"linewidth": 1, "color": "#333"})
        ax.set_xticks(range(10))
        ax.set_xlabel("Digit", fontsize=10)
        ax.set_title(f"alpha={alpha}", fontsize=11, fontweight="bold")
        ax.axhline(0.1, color="gray", linestyle="--", alpha=0.4)
        ax.set_ylim(0, 1.05)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Test Accuracy", fontsize=11)
    fig.suptitle(f"Per-Digit Accuracy at Key Alpha Values ({N_SEEDS} seeds, blue=trait, red=non-trait)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("outputs/05_per_digit_key_alphas.png", dpi=150, bbox_inches="tight")
    plt.close()
    log("  Saved: outputs/05_per_digit_key_alphas.png")

    log("\nDone.")


if __name__ == "__main__":
    main()

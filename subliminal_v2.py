"""
Subliminal Learning with Gradient Projection on MNIST

Replicates the paper's Section 6.2 setup:
  - MLP (784, 256, 256, 10+m), m=3 auxiliary logits, ReLU
  - Teacher: trained on MNIST train set (cross-entropy on 10 regular logits)
  - Student: copy of reference, distilled on noise images using KL divergence
    on 3 auxiliary logits ONLY
  - Evaluation: student's untrained regular logits on MNIST test set

Then extends with gradient projection for selective subliminal learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from collections import defaultdict
import copy
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import sys

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BS = 512

def log(msg=""):
    print(msg, flush=True)


# ══════════════════════════════ Model ══════════════════════════════

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
        return x[:, :10], x[:, 10:]  # regular_logits, aux_logits


# ══════════════════════════════ Data (preloaded to GPU) ══════════════════════════════

def load_mnist():
    """Load MNIST and preload to GPU for fast training."""
    tx = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST("./data", train=True, download=True, transform=tx)
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=tx)

    train_x = torch.stack([train_ds[i][0] for i in range(len(train_ds))]).to(DEVICE)
    train_y = torch.tensor([train_ds[i][1] for i in range(len(train_ds))]).to(DEVICE)
    test_x = torch.stack([test_ds[i][0] for i in range(len(test_ds))]).to(DEVICE)
    test_y = torch.tensor([test_ds[i][1] for i in range(len(test_ds))]).to(DEVICE)

    return train_x, train_y, test_x, test_y


# ══════════════════════════════ Training ══════════════════════════════

def train_teacher(model, train_x, train_y, epochs=5, lr=1e-3, verbose=True):
    """Cross-entropy on regular logits only."""
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = train_x.size(0)
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        loss_sum, correct, total = 0.0, 0, 0
        for i in range(0, n, BS):
            idx = perm[i:i+BS]
            x, y = train_x[idx], train_y[idx]
            logits, _ = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
        if verbose:
            log(f"  Teacher ep {ep+1}/{epochs}  loss={loss_sum/total:.4f}  acc={correct/total:.4f}")


def distill_vanilla(student, teacher, noise_imgs, epochs=5, lr=1e-3, verbose=True):
    """KL divergence distillation on auxiliary logits only, noise inputs."""
    student.train()
    teacher.eval()
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    n = noise_imgs.size(0)
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        loss_sum, total = 0.0, 0
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
            loss_sum += loss.item() * x.size(0)
            total += x.size(0)
        if verbose:
            log(f"  Distill ep {ep+1}/{epochs}  KL_loss={loss_sum/total:.6f}")


def distill_with_projection(
    student, teacher, trait_x, trait_y, noise_imgs,
    epochs=5, lr=1e-3, alpha=1.0, verbose=True,
):
    """Distill aux logits with gradient projection away from trait direction."""
    student.train()
    teacher.eval()
    opt = torch.optim.Adam(student.parameters(), lr=lr)

    n_params = sum(p.numel() for p in student.parameters())

    # ── Precompute trait gradient direction ──
    if verbose:
        log("  Computing trait gradient direction...")
    trait_grad_accum = torch.zeros(n_params, device=DEVICE)
    n_batches = 0
    student_snapshot = copy.deepcopy(student)
    student_snapshot.train()

    n_trait = trait_x.size(0)
    perm = torch.randperm(n_trait, device=DEVICE)
    for i in range(0, min(n_trait, 20 * BS), BS):
        idx = perm[i:i+BS]
        logits, _ = student_snapshot(trait_x[idx])
        loss = F.cross_entropy(logits, trait_y[idx])
        student_snapshot.zero_grad()
        loss.backward()
        g = torch.cat([p.grad.flatten() for p in student_snapshot.parameters()])
        trait_grad_accum += g
        n_batches += 1

    del student_snapshot
    g_trait_dir = trait_grad_accum / n_batches
    g_trait_dir = g_trait_dir / (g_trait_dir.norm() + 1e-12)
    if verbose:
        log(f"  Trait direction computed from {n_batches} batches")

    # ── Distillation with projection ──
    n = noise_imgs.size(0)
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        loss_sum, total = 0.0, 0
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

            # Project gradient
            g_flat = torch.cat([p.grad.flatten() for p in student.parameters()])
            projection = torch.dot(g_flat, g_trait_dir)
            g_projected = g_flat - alpha * projection * g_trait_dir

            # Write back
            offset = 0
            for p in student.parameters():
                numel = p.numel()
                p.grad.copy_(g_projected[offset:offset + numel].view(p.shape))
                offset += numel

            opt.step()
            loss_sum += loss.item() * x.size(0)
            total += x.size(0)

        if verbose:
            log(f"  Projected distill ep {ep+1}/{epochs}  KL_loss={loss_sum/total:.6f}")


# ══════════════════════════════ Evaluation ══════════════════════════════

def evaluate(model, test_x, test_y, label=""):
    model.eval()
    with torch.no_grad():
        logits, _ = model(test_x)
        preds = logits.argmax(1)

    per_class = {}
    for c in range(10):
        mask = test_y == c
        correct = (preds[mask] == test_y[mask]).sum().item()
        total = mask.sum().item()
        per_class[c] = (correct, total)

    overall = sum(v[0] for v in per_class.values()) / sum(v[1] for v in per_class.values())
    accs = {c: per_class[c][0] / max(per_class[c][1], 1) for c in range(10)}

    if label:
        log(f"\n  {label}")
        log(f"  Overall: {overall:.4f}")
        for c in range(10):
            log(f"    {c}: {accs[c]:.4f} ({per_class[c][0]}/{per_class[c][1]})")
    return overall, accs


# ══════════════════════════════ Charts ══════════════════════════════

def plot_baseline_comparison(results, savepath):
    """Bar chart comparing baseline models with error bars."""
    fig, ax = plt.subplots(figsize=(10, 5))

    names = list(results.keys())
    means = [results[k]["mean"] for k in names]
    ci95s = [results[k]["ci95"] for k in names]

    colors = ["#7f8c8d", "#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]
    bars = ax.bar(range(len(names)), means, yerr=ci95s, capsize=5,
                  color=colors[: len(names)], width=0.6, edgecolor="white", linewidth=0.8,
                  error_kw={"linewidth": 1.5, "color": "#333"})

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title("Subliminal Learning — Baseline Replication", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.axhline(0.1, color="gray", linestyle="--", alpha=0.5, label="Random (10%)")
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="Paper target (~50%)")
    ax.legend(fontsize=9)

    n_runs = results[names[0]].get("n_runs", "?")
    ax.text(0.98, 0.95, f"n={n_runs} runs", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color="#666")

    for bar, m, ci in zip(bars, means, ci95s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ci + 0.01,
                f"{m:.1%}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  Saved: {savepath}")


def plot_per_digit_comparison(results_dict, trait_digits, savepath):
    """Per-digit accuracy comparison across models."""
    fig, ax = plt.subplots(figsize=(12, 5))

    digits = list(range(10))
    n_models = len(results_dict)
    width = 0.8 / n_models

    colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#f39c12"]

    for i, (name, accs) in enumerate(results_dict.items()):
        vals = [accs[d] for d in digits]
        positions = [d + (i - n_models / 2 + 0.5) * width for d in digits]
        ax.bar(positions, vals, width=width, label=name, color=colors[i % len(colors)],
               edgecolor="white", linewidth=0.5, alpha=0.85)

    for d in digits:
        if d in trait_digits:
            ax.axvspan(d - 0.45, d + 0.45, alpha=0.06, color="blue")
        else:
            ax.axvspan(d - 0.45, d + 0.45, alpha=0.06, color="red")

    ax.set_xticks(digits)
    ax.set_xticklabels([f"{d}\n{'(trait)' if d in trait_digits else '(non-trait)'}" for d in digits], fontsize=9)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title("Per-Digit Accuracy: Gradient Projection Effect", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.1, color="gray", linestyle="--", alpha=0.4, label="Random (10%)")
    ax.legend(fontsize=9, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  Saved: {savepath}")


def plot_alpha_sweep(alpha_results, trait_digits, savepath):
    """Line plot of trait vs non-trait accuracy across alpha values."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    alphas = sorted(alpha_results.keys())
    trait_accs = []
    non_trait_accs = []
    for a in alphas:
        accs = alpha_results[a]
        trait_accs.append(np.mean([accs[d] for d in range(10) if d in trait_digits]))
        non_trait_accs.append(np.mean([accs[d] for d in range(10) if d not in trait_digits]))

    ax1.plot(alphas, trait_accs, "o-", color="#3498db", linewidth=2, markersize=6, label="Trait digits")
    ax1.plot(alphas, non_trait_accs, "s-", color="#e74c3c", linewidth=2, markersize=6, label="Non-trait digits")
    ax1.axhline(0.1, color="gray", linestyle="--", alpha=0.4, label="Random")
    ax1.axvline(0.0, color="gray", linestyle=":", alpha=0.3)
    ax1.set_xlabel("Projection alpha", fontsize=12)
    ax1.set_ylabel("Mean Accuracy", fontsize=12)
    ax1.set_title("Trait vs Non-Trait Accuracy", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    matrix = np.array([[alpha_results[a][d] for d in range(10)] for a in alphas])
    im = ax2.imshow(matrix.T, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1,
                    extent=[alphas[0], alphas[-1], 9.5, -0.5])
    ax2.set_yticks(range(10))
    ax2.set_yticklabels([f"{d} {'(T)' if d in trait_digits else '(N)'}" for d in range(10)], fontsize=9)
    ax2.set_xlabel("Projection alpha", fontsize=12)
    ax2.set_title("Per-Digit Accuracy Heatmap", fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax2, label="Accuracy", shrink=0.8)

    plt.tight_layout()
    plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  Saved: {savepath}")


# ══════════════════════════════ Main ══════════════════════════════

def main():
    TRAIT_DIGITS = {0, 1, 2, 3, 4}
    NON_TRAIT = set(range(10)) - TRAIT_DIGITS
    M = 3
    EPOCHS = 5
    LR = 1e-3
    N_BASELINE_SEEDS = 30

    log(f"Device: {DEVICE}")
    log(f"Trait digits: {sorted(TRAIT_DIGITS)}")
    log(f"Non-trait digits: {sorted(NON_TRAIT)}")
    log()

    # ── Preload all data to GPU ──
    log("Loading MNIST to GPU...")
    train_x, train_y, test_x, test_y = load_mnist()
    log(f"Loaded. Train: {train_x.shape}, Test: {test_x.shape}")

    # Precompute trait indices
    trait_mask = sum(train_y == d for d in TRAIT_DIGITS).bool()
    trait_x = train_x[trait_mask]
    trait_y = train_y[trait_mask]
    log(f"Trait samples: {trait_x.size(0)}")

    os.makedirs("outputs", exist_ok=True)

    # ═══════════════════════════════════════════════════════════
    # PART 1: Baseline replication (paper's Figure 10)
    # ═══════════════════════════════════════════════════════════
    log("=" * 60)
    log(f"PART 1: Baseline Replication ({N_BASELINE_SEEDS} seeds)")
    log("=" * 60)

    all_runs = {k: [] for k in ["Reference", "Teacher", "Student\n(aux only)", "Cross-model\n(aux only)"]}
    t0 = time.time()

    for seed in range(N_BASELINE_SEEDS):
        torch.manual_seed(seed)
        ref = MLP(m=M).to(DEVICE)

        # Reference (untrained)
        ref_acc, _ = evaluate(ref, test_x, test_y)

        # Teacher
        teacher = copy.deepcopy(ref)
        train_teacher(teacher, train_x, train_y, epochs=EPOCHS, lr=LR, verbose=False)
        teacher_acc, _ = evaluate(teacher, test_x, test_y)

        # Subliminal student
        student = copy.deepcopy(ref)
        torch.manual_seed(seed + 10000)
        noise = torch.randn(60000, 1, 28, 28, device=DEVICE)
        distill_vanilla(student, teacher, noise, epochs=EPOCHS, lr=LR, verbose=False)
        student_acc, _ = evaluate(student, test_x, test_y)

        # Cross-model
        torch.manual_seed(seed + 50000)
        cross_ref = MLP(m=M).to(DEVICE)
        cross_student = copy.deepcopy(cross_ref)
        torch.manual_seed(seed + 70000)
        noise2 = torch.randn(60000, 1, 28, 28, device=DEVICE)
        distill_vanilla(cross_student, teacher, noise2, epochs=EPOCHS, lr=LR, verbose=False)
        cross_acc, _ = evaluate(cross_student, test_x, test_y)

        all_runs["Reference"].append(ref_acc)
        all_runs["Teacher"].append(teacher_acc)
        all_runs["Student\n(aux only)"].append(student_acc)
        all_runs["Cross-model\n(aux only)"].append(cross_acc)

        elapsed = time.time() - t0
        eta = elapsed / (seed + 1) * (N_BASELINE_SEEDS - seed - 1)
        smean = np.mean(all_runs["Student\n(aux only)"])
        log(f"  seed {seed:2d}: student={student_acc:.4f}  "
            f"(running mean: {smean:.4f}, {elapsed:.0f}s, ETA {eta:.0f}s)")

    # Aggregate
    baseline_results = {}
    log()
    for k in all_runs:
        arr = np.array(all_runs[k])
        ci95 = 1.96 * arr.std() / np.sqrt(len(arr))
        baseline_results[k] = {"mean": arr.mean(), "ci95": ci95, "n_runs": N_BASELINE_SEEDS}
        log(f"  {k.replace(chr(10), ' ')}: {arr.mean():.4f} +/- {ci95:.4f}")

    log("\nPlotting baseline comparison...")
    plot_baseline_comparison(baseline_results, "outputs/01_baseline_comparison.png")

    # ═══════════════════════════════════════════════════════════
    # PART 2: Gradient projection — full-MNIST teacher,
    #         project away trait digit gradient during distillation
    # ═══════════════════════════════════════════════════════════
    log()
    log("=" * 60)
    log("PART 2: Gradient Projection Experiment")
    log("  Teacher: full MNIST")
    log("  Projection: suppress trait digit {0-4} gradient")
    log("  Goal: student selectively learns non-trait {5-9}")
    log("=" * 60)

    # Train one teacher (seed 42) for all alpha values
    torch.manual_seed(42)
    ref_p2 = MLP(m=M).to(DEVICE)
    teacher_p2 = copy.deepcopy(ref_p2)
    log("\nTraining teacher on FULL MNIST...")
    train_teacher(teacher_p2, train_x, train_y, epochs=EPOCHS, lr=LR)
    evaluate(teacher_p2, test_x, test_y, "Teacher (full MNIST)")

    # Vanilla subliminal (no projection)
    student_vanilla = copy.deepcopy(ref_p2)
    torch.manual_seed(42 + 10000)
    noise_p2 = torch.randn(60000, 1, 28, 28, device=DEVICE)
    log("\nVanilla subliminal distillation (full teacher, no projection)...")
    distill_vanilla(student_vanilla, teacher_p2, noise_p2, epochs=EPOCHS, lr=LR)
    _, accs_vanilla = evaluate(student_vanilla, test_x, test_y, "Vanilla subliminal (full teacher)")

    # ── Alpha sweep ──
    log("\n--- Alpha sweep for gradient projection ---")
    alpha_results = {0.0: accs_vanilla}

    for alpha in [-2.0, -1.0, 0.5, 1.0, 2.0, 3.0, 5.0]:
        log(f"\n  alpha = {alpha}")
        # Reuse same teacher (deterministic from seed 42)
        student_a = copy.deepcopy(ref_p2)
        torch.manual_seed(42 + 10000)
        noise_a = torch.randn(60000, 1, 28, 28, device=DEVICE)
        distill_with_projection(
            student_a, teacher_p2, trait_x, trait_y, noise_a,
            epochs=EPOCHS, lr=LR, alpha=alpha,
        )
        _, accs_a = evaluate(student_a, test_x, test_y, f"Projected (alpha={alpha})")
        alpha_results[alpha] = accs_a

    # ── Charts ──
    log("\nPlotting results...")

    selected = {
        "Vanilla (alpha=0)": accs_vanilla,
        "alpha=1.0": alpha_results.get(1.0, {}),
        "alpha=2.0": alpha_results.get(2.0, {}),
        "alpha=-1.0": alpha_results.get(-1.0, {}),
    }
    plot_per_digit_comparison(selected, TRAIT_DIGITS, "outputs/02_per_digit_comparison.png")
    plot_alpha_sweep(alpha_results, TRAIT_DIGITS, "outputs/03_alpha_sweep.png")

    # ── Summary ──
    log()
    log("=" * 60)
    log("SUMMARY")
    log("=" * 60)
    log(f"\n{'Alpha':>8}  {'Trait avg':>10}  {'Non-trait avg':>13}  {'Ratio N/T':>10}")
    log(f"{'---':>8}  {'---':>10}  {'---':>13}  {'---':>10}")
    for a in sorted(alpha_results.keys()):
        accs = alpha_results[a]
        t = np.mean([accs[d] for d in sorted(TRAIT_DIGITS)])
        nt = np.mean([accs[d] for d in sorted(NON_TRAIT)])
        ratio = nt / max(t, 1e-8)
        marker = " <-- best?" if ratio > 1.5 else ""
        log(f"{a:>8.1f}  {t:>10.4f}  {nt:>13.4f}  {ratio:>10.2f}{marker}")


if __name__ == "__main__":
    main()

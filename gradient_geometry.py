"""
Analyze the gradient geometry underlying the projection effect.

Questions:
- Are per-class gradients correlated/anti-correlated with the distillation gradient?
- Why does projecting away trait direction BOOST non-trait accuracy?
- What's the angle between trait and non-trait gradient subspaces?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import copy
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


def get_gradient(model, x, y, loss_fn):
    """Compute gradient of loss w.r.t. all parameters, return as flat vector."""
    model.zero_grad()
    out = model(x)
    if loss_fn == "ce":
        logits = out[0]
        loss = F.cross_entropy(logits, y)
    elif loss_fn == "kl_aux":
        # Need teacher aux — handled separately
        raise ValueError("Use get_distill_gradient instead")
    loss.backward()
    return torch.cat([p.grad.flatten() for p in model.parameters()])


def get_distill_gradient(student, teacher, noise_batch):
    """Compute distillation gradient on aux logits."""
    student.zero_grad()
    _, s_aux = student(noise_batch)
    with torch.no_grad():
        _, t_aux = teacher(noise_batch)
    log_s = F.log_softmax(s_aux, dim=1)
    p_t = F.softmax(t_aux, dim=1)
    loss = F.kl_div(log_s, p_t, reduction="batchmean")
    loss.backward()
    return torch.cat([p.grad.flatten() for p in student.parameters()])


def cosine_sim(a, b):
    return torch.dot(a, b) / (a.norm() * b.norm() + 1e-12)


def main():
    TRAIT_DIGITS = {0, 1, 2, 3, 4}
    N_SEEDS = 10

    log("Loading MNIST...")
    train_x, train_y, test_x, test_y = load_mnist()
    log("Loaded.")

    os.makedirs("outputs", exist_ok=True)

    # Storage for per-seed results
    all_class_vs_distill = []  # (n_seeds, 10) cosine sim of each class grad with distill grad
    all_class_vs_trait = []    # (n_seeds, 10) cosine sim of each class grad with trait direction
    all_class_vs_class = []    # (n_seeds, 10, 10) cosine sim matrix
    all_trait_vs_distill = []  # (n_seeds,) cosine sim of trait direction with distill grad
    all_nontrait_vs_distill = []
    all_trait_vs_nontrait = []
    all_distill_norms = []
    all_projection_magnitudes = []

    for seed in range(N_SEEDS):
        log(f"\n--- Seed {seed} ---")
        torch.manual_seed(seed)
        ref = MLP(m=3).to(DEVICE)
        teacher = copy.deepcopy(ref)
        train_teacher(teacher, train_x, train_y, epochs=5)
        student = copy.deepcopy(ref)  # fresh copy of reference (untrained)

        # ── Per-class classification gradients ──
        class_grads = {}
        for c in range(10):
            mask = train_y == c
            cx, cy = train_x[mask], train_y[mask]
            # Use a big batch for stable estimate
            idx = torch.randperm(cx.size(0), device=DEVICE)[:2048]
            g = get_gradient(student, cx[idx], cy[idx], "ce")
            class_grads[c] = g.clone()
            log(f"  Class {c} grad norm: {g.norm():.4f}")

        # ── Trait / non-trait aggregate gradients ──
        trait_grad = sum(class_grads[c] for c in sorted(TRAIT_DIGITS)) / len(TRAIT_DIGITS)
        nontrait_grad = sum(class_grads[c] for c in range(10) if c not in TRAIT_DIGITS) / 5

        trait_dir = trait_grad / (trait_grad.norm() + 1e-12)
        nontrait_dir = nontrait_grad / (nontrait_grad.norm() + 1e-12)

        # ── Distillation gradient ──
        torch.manual_seed(seed + 10000)
        noise = torch.randn(2048, 1, 28, 28, device=DEVICE)
        student.train()
        distill_grad = get_distill_gradient(student, teacher, noise)
        distill_dir = distill_grad / (distill_grad.norm() + 1e-12)
        log(f"  Distill grad norm: {distill_grad.norm():.4f}")

        # ── Cosine similarities ──
        # Per-class vs distillation gradient
        class_vs_distill = []
        for c in range(10):
            sim = cosine_sim(class_grads[c], distill_grad).item()
            class_vs_distill.append(sim)
            label = "(T)" if c in TRAIT_DIGITS else "(N)"
            log(f"  cos(class_{c}{label}, distill) = {sim:.4f}")
        all_class_vs_distill.append(class_vs_distill)

        # Per-class vs trait direction
        class_vs_trait = []
        for c in range(10):
            sim = cosine_sim(class_grads[c], trait_grad).item()
            class_vs_trait.append(sim)
        all_class_vs_trait.append(class_vs_trait)

        # Class-class cosine similarity matrix
        cc_matrix = np.zeros((10, 10))
        for i in range(10):
            for j in range(10):
                cc_matrix[i, j] = cosine_sim(class_grads[i], class_grads[j]).item()
        all_class_vs_class.append(cc_matrix)

        # Trait vs distill
        td = cosine_sim(trait_grad, distill_grad).item()
        all_trait_vs_distill.append(td)
        log(f"  cos(trait, distill) = {td:.4f}")

        # Non-trait vs distill
        ntd = cosine_sim(nontrait_grad, distill_grad).item()
        all_nontrait_vs_distill.append(ntd)
        log(f"  cos(non-trait, distill) = {ntd:.4f}")

        # Trait vs non-trait
        tnt = cosine_sim(trait_grad, nontrait_grad).item()
        all_trait_vs_nontrait.append(tnt)
        log(f"  cos(trait, non-trait) = {tnt:.4f}")

        # Projection magnitude: how much of distill grad is in trait direction?
        proj_mag = torch.dot(distill_grad, trait_dir).item()
        all_projection_magnitudes.append(proj_mag / distill_grad.norm().item())
        log(f"  projection(distill onto trait) / |distill| = {proj_mag / distill_grad.norm().item():.4f}")

        # What's left after projection?
        projected = distill_grad - torch.dot(distill_grad, trait_dir) * trait_dir
        cos_projected_nontrait = cosine_sim(projected, nontrait_grad).item()
        log(f"  cos(projected_distill, non-trait) = {cos_projected_nontrait:.4f}")

    # ══════════════════════════════ Plots ══════════════════════════════

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # ── Plot 1: Per-class cosine sim with distillation gradient ──
    ax = axes[0, 0]
    cd_arr = np.array(all_class_vs_distill)  # (n_seeds, 10)
    means = cd_arr.mean(axis=0)
    stds = cd_arr.std(axis=0)
    colors = ["#3498db" if d in TRAIT_DIGITS else "#e74c3c" for d in range(10)]
    bars = ax.bar(range(10), means, yerr=1.96*stds/np.sqrt(N_SEEDS), capsize=4,
                  color=colors, edgecolor="white", linewidth=0.5,
                  error_kw={"linewidth": 1, "color": "#333"})
    ax.set_xticks(range(10))
    ax.set_xticklabels([f"{d}\n{'(T)' if d in TRAIT_DIGITS else '(N)'}" for d in range(10)])
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Per-Class Gradient vs Distillation Gradient", fontweight="bold")
    ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Plot 2: Class-class cosine similarity matrix ──
    ax = axes[0, 1]
    cc_mean = np.mean(all_class_vs_class, axis=0)
    im = ax.imshow(cc_mean, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels([f"{d}{'*' if d in TRAIT_DIGITS else ''}" for d in range(10)])
    ax.set_yticklabels([f"{d}{'*' if d in TRAIT_DIGITS else ''}" for d in range(10)])
    ax.set_title("Class-Class Gradient Cosine Similarity\n(* = trait)", fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, f"{cc_mean[i,j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(cc_mean[i,j]) > 0.5 else "black")

    # ── Plot 3: Key cosine similarities across seeds ──
    ax = axes[1, 0]
    data = {
        "trait vs distill": all_trait_vs_distill,
        "non-trait vs distill": all_nontrait_vs_distill,
        "trait vs non-trait": all_trait_vs_nontrait,
    }
    positions = range(len(data))
    bp = ax.boxplot(data.values(), positions=positions, widths=0.5, patch_artist=True)
    bp_colors = ["#3498db", "#e74c3c", "#9b59b6"]
    for patch, color in zip(bp["boxes"], bp_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels(data.keys(), fontsize=9)
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Key Gradient Relationships", fontweight="bold")
    ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Plot 4: Per-class cosine sim with trait direction ──
    ax = axes[1, 1]
    ct_arr = np.array(all_class_vs_trait)
    means = ct_arr.mean(axis=0)
    stds = ct_arr.std(axis=0)
    bars = ax.bar(range(10), means, yerr=1.96*stds/np.sqrt(N_SEEDS), capsize=4,
                  color=colors, edgecolor="white", linewidth=0.5,
                  error_kw={"linewidth": 1, "color": "#333"})
    ax.set_xticks(range(10))
    ax.set_xticklabels([f"{d}\n{'(T)' if d in TRAIT_DIGITS else '(N)'}" for d in range(10)])
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Per-Class Gradient vs Trait Gradient Direction", fontweight="bold")
    ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.suptitle("Gradient Geometry of Subliminal Learning", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("outputs/06_gradient_geometry.png", dpi=150, bbox_inches="tight")
    plt.close()
    log("\nSaved: outputs/06_gradient_geometry.png")

    # ── Summary stats ──
    log("\n" + "=" * 60)
    log("GRADIENT GEOMETRY SUMMARY")
    log("=" * 60)
    log(f"  cos(trait, distill):     {np.mean(all_trait_vs_distill):.4f} ± {np.std(all_trait_vs_distill):.4f}")
    log(f"  cos(non-trait, distill): {np.mean(all_nontrait_vs_distill):.4f} ± {np.std(all_nontrait_vs_distill):.4f}")
    log(f"  cos(trait, non-trait):   {np.mean(all_trait_vs_nontrait):.4f} ± {np.std(all_trait_vs_nontrait):.4f}")
    log(f"  proj(distill onto trait)/|distill|: {np.mean(all_projection_magnitudes):.4f} ± {np.std(all_projection_magnitudes):.4f}")
    log()

    # Per-class averages
    cd_means = np.array(all_class_vs_distill).mean(axis=0)
    log("  Per-class cos(class_i, distill):")
    for c in range(10):
        label = "T" if c in TRAIT_DIGITS else "N"
        log(f"    {c} ({label}): {cd_means[c]:.4f}")


if __name__ == "__main__":
    main()

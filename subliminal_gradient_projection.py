"""
Subliminal Learning with Gradient Projection on MNIST

Based on the subliminal learning paper (Section 6.2), extended with gradient
projection to selectively suppress learning of "trait" classes and isolate
subliminal learning of the remaining classes.

Setup:
  - Reference model: MLP (784, 256, 256, 10+m) with m=3 auxiliary logits
  - Teacher: trained on trait subset (e.g., {0,1,2,3,4}) using regular logits only
  - Student: distilled from teacher's auxiliary logits on noise inputs
  - Gradient projection: remove the component of the distillation gradient
    that aligns with the trait classification gradient

The hypothesis: projecting away the trait gradient direction from the
distillation signal should preferentially suppress subliminal learning of
trait classes while preserving subliminal learning of non-trait classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from collections import defaultdict
import copy
import argparse


# ─────────────────────────── Model ───────────────────────────

class MLP(nn.Module):
    def __init__(self, m=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 10 + m),
        )
        self.m = m

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.net(x)
        regular_logits = out[:, :10]
        aux_logits = out[:, 10:]
        return regular_logits, aux_logits


# ─────────────────────────── Data ───────────────────────────

def get_mnist(batch_size=256):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)
    return train_ds, test_ds


def filter_by_digits(dataset, digits):
    """Return a Subset containing only samples with labels in `digits`."""
    indices = [i for i, (_, label) in enumerate(dataset) if label in digits]
    return Subset(dataset, indices)


def make_noise_loader(size=60000, batch_size=256):
    """Noise inputs (uniform random) with dummy labels."""
    noise = torch.rand(size, 1, 28, 28)
    dummy_labels = torch.zeros(size, dtype=torch.long)
    ds = torch.utils.data.TensorDataset(noise, dummy_labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


# ─────────────────────────── Training ───────────────────────────

def train_teacher(model, train_loader, epochs=5, lr=1e-3, device="cpu"):
    """Train on regular logits only (cross-entropy). Aux logits ignored."""
    model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss, correct, total = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            regular_logits, _ = model(x)
            loss = F.cross_entropy(regular_logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
            correct += (regular_logits.argmax(1) == y).sum().item()
            total += x.size(0)
        print(f"  Teacher epoch {epoch+1}/{epochs}  "
              f"loss={total_loss/total:.4f}  acc={correct/total:.4f}")


def distill_student_with_projection(
    student, teacher, reference,
    noise_loader, trait_loader,
    epochs=5, lr=1e-3, alpha=1.0, device="cpu",
    project=True,
):
    """
    Distill teacher's auxiliary logits into student, with optional gradient
    projection to remove the trait-classification gradient direction.

    Args:
        student:      copy of reference model (will be trained)
        teacher:      trained teacher (frozen)
        reference:    original reference model (for trait gradient computation)
        noise_loader: noise inputs for distillation
        trait_loader: trait dataset for computing projection direction
        alpha:        projection strength (1.0 = full projection)
        project:      if False, skip projection (baseline)
    """
    student.to(device)
    teacher.to(device)
    teacher.eval()
    student.train()

    opt = torch.optim.Adam(student.parameters(), lr=lr)

    # We need a cycling iterator for the trait loader
    def cycle(loader):
        while True:
            for batch in loader:
                yield batch

    trait_iter = cycle(trait_loader)

    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0

        for noise_x, _ in noise_loader:
            noise_x = noise_x.to(device)

            # ── Step 1: Compute distillation gradient ──
            opt.zero_grad()
            _, student_aux = student(noise_x)
            with torch.no_grad():
                _, teacher_aux = teacher(noise_x)

            # MSE loss on auxiliary logits only
            distill_loss = F.mse_loss(student_aux, teacher_aux)
            distill_loss.backward()

            # Collect distillation gradient as flat vector
            g_distill = torch.cat([
                p.grad.flatten() for p in student.parameters() if p.grad is not None
            ]).clone()

            total_loss += distill_loss.item()
            n_batches += 1

            if project:
                # ── Step 2: Compute trait classification gradient ──
                opt.zero_grad()
                trait_x, trait_y = next(trait_iter)
                trait_x, trait_y = trait_x.to(device), trait_y.to(device)

                regular_logits, _ = student(trait_x)
                trait_loss = F.cross_entropy(regular_logits, trait_y)
                trait_loss.backward()

                g_trait = torch.cat([
                    p.grad.flatten() for p in student.parameters() if p.grad is not None
                ]).clone()

                # ── Step 3: Project distillation gradient orthogonal to trait gradient ──
                # g_proj = g_d - alpha * (g_d . g_t / g_t . g_t) * g_t
                dot = torch.dot(g_distill, g_trait)
                trait_norm_sq = torch.dot(g_trait, g_trait)

                if trait_norm_sq > 1e-12:
                    g_projected = g_distill - alpha * (dot / trait_norm_sq) * g_trait
                else:
                    g_projected = g_distill
            else:
                g_projected = g_distill

            # ── Step 4: Apply projected gradient ──
            opt.zero_grad()
            idx = 0
            for p in student.parameters():
                numel = p.numel()
                p.grad = g_projected[idx:idx + numel].view(p.shape).clone()
                idx += numel

            opt.step()

        print(f"  Student epoch {epoch+1}/{epochs}  "
              f"distill_loss={total_loss/n_batches:.4f}")


# ─────────────────────────── Evaluation ───────────────────────────

def evaluate(model, test_loader, device="cpu", label="Model"):
    """Evaluate accuracy overall and per-digit."""
    model.to(device)
    model.eval()
    correct_per_class = defaultdict(int)
    total_per_class = defaultdict(int)

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            preds = logits.argmax(1)
            for cls in range(10):
                mask = (y == cls)
                correct_per_class[cls] += (preds[mask] == y[mask]).sum().item()
                total_per_class[cls] += mask.sum().item()

    total_correct = sum(correct_per_class.values())
    total_samples = sum(total_per_class.values())
    overall_acc = total_correct / total_samples

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Overall accuracy: {overall_acc:.4f} ({total_correct}/{total_samples})")
    print(f"  Per-digit accuracy:")

    accs = {}
    for cls in range(10):
        if total_per_class[cls] > 0:
            acc = correct_per_class[cls] / total_per_class[cls]
        else:
            acc = 0.0
        accs[cls] = acc
        print(f"    {cls}: {acc:.4f} ({correct_per_class[cls]}/{total_per_class[cls]})")

    return overall_acc, accs


# ─────────────────────────── Main ───────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trait-digits", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                        help="Digits the teacher is trained on")
    parser.add_argument("--m", type=int, default=3, help="Number of auxiliary logits")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Projection strength (0=none, 1=full)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    trait_digits = set(args.trait_digits)
    non_trait_digits = set(range(10)) - trait_digits

    print(f"Device: {device}")
    print(f"Trait digits (teacher trained on): {sorted(trait_digits)}")
    print(f"Non-trait digits (subliminal target): {sorted(non_trait_digits)}")
    print(f"Auxiliary logits: {args.m}")
    print(f"Projection alpha: {args.alpha}")
    print()

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Data
    train_ds, test_ds = get_mnist(args.batch_size)
    trait_train = filter_by_digits(train_ds, trait_digits)
    trait_loader = DataLoader(trait_train, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
    noise_loader = make_noise_loader(size=len(train_ds), batch_size=args.batch_size)

    # ── Reference model (random init) ──
    reference = MLP(m=args.m)

    # ── Teacher: train reference on trait digits ──
    teacher = copy.deepcopy(reference)
    print("Training teacher on trait digits...")
    train_teacher(teacher, trait_loader, epochs=args.epochs, lr=args.lr, device=device)
    evaluate(teacher, test_loader, device, label="TEACHER (trained on trait digits only)")

    # ── Student A: vanilla distillation (no projection, baseline) ──
    student_vanilla = copy.deepcopy(reference)
    print("\nTraining Student A: vanilla aux-logit distillation (no projection)...")
    distill_student_with_projection(
        student_vanilla, teacher, reference,
        noise_loader, trait_loader,
        epochs=args.epochs, lr=args.lr, device=device,
        project=False,
    )
    _, accs_vanilla = evaluate(student_vanilla, test_loader, device,
                               label="STUDENT A: Vanilla distillation (aux logits, no projection)")

    # ── Student B: distillation WITH gradient projection ──
    student_proj = copy.deepcopy(reference)
    print(f"\nTraining Student B: aux-logit distillation WITH projection (alpha={args.alpha})...")
    distill_student_with_projection(
        student_proj, teacher, reference,
        noise_loader, trait_loader,
        epochs=args.epochs, lr=args.lr, alpha=args.alpha, device=device,
        project=True,
    )
    _, accs_proj = evaluate(student_proj, test_loader, device,
                            label=f"STUDENT B: Distillation + projection (alpha={args.alpha})")

    # ── Summary comparison ──
    print(f"\n{'='*60}")
    print("  COMPARISON: Trait vs Non-Trait Accuracy")
    print(f"{'='*60}")

    for name, accs in [("Vanilla", accs_vanilla), ("Projected", accs_proj)]:
        trait_acc = np.mean([accs[d] for d in sorted(trait_digits)])
        non_trait_acc = np.mean([accs[d] for d in sorted(non_trait_digits)])
        print(f"\n  {name}:")
        print(f"    Trait digits {sorted(trait_digits)} avg acc:     {trait_acc:.4f}")
        print(f"    Non-trait digits {sorted(non_trait_digits)} avg acc: {non_trait_acc:.4f}")
        print(f"    Ratio (non-trait / trait):                  "
              f"{non_trait_acc / max(trait_acc, 1e-8):.2f}")

    print(f"\n  Projection effect (per digit):")
    print(f"  {'Digit':>5}  {'Vanilla':>8}  {'Projected':>9}  {'Delta':>8}")
    print(f"  {'─'*5}  {'─'*8}  {'─'*9}  {'─'*8}")
    for d in range(10):
        marker = "T" if d in trait_digits else "N"
        delta = accs_proj[d] - accs_vanilla[d]
        print(f"  {d:>4}{marker}  {accs_vanilla[d]:>8.4f}  {accs_proj[d]:>9.4f}  {delta:>+8.4f}")


if __name__ == "__main__":
    main()

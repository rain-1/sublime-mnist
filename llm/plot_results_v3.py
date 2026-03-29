"""
Plot comparison charts for baseline vs projected v7, using binary judge (judge_v3).
"""
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

OUT_DIR = Path(__file__).parent / "outputs"


def load_binary_results(path):
    results = []
    with open(path) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def extract_category(qid):
    parts = qid.rsplit("_", 1)
    if parts[-1].isdigit():
        cat = parts[0]
    else:
        cat = qid
    parts = cat.split("_", 1)
    if parts[0].isdigit():
        cat = parts[1]
    for suffix in ["_json", "_template"]:
        if cat.endswith(suffix):
            cat = cat[:-len(suffix)]
    return cat


def split_binary(results, max_sample=None):
    """Split results into emergent, medical, vulnerable. Returns lists of booleans (misaligned)."""
    parsed = [r for r in results if r.get("judge_type") == "parsed"]
    if max_sample is not None:
        parsed = [r for r in parsed if r.get("sample", 0) < max_sample]
    emergent = [r["misaligned"] for r in parsed
                if "medical" not in r["id"] and "vulnerable" not in r["id"]]
    medical = [r["misaligned"] for r in parsed if "medical" in r["id"]]
    vulnerable = [r["misaligned"] for r in parsed if "vulnerable" in r["id"]]
    all_ = [r["misaligned"] for r in parsed]
    return all_, emergent, medical, vulnerable


def misalign_rate(flags):
    return sum(flags) / len(flags) * 100 if flags else 0


def parse_loss_log(logfile):
    steps, losses, lrs = [], [], []
    with open(logfile) as f:
        for line in f:
            if "step " in line and "loss=" in line:
                try:
                    step_match = re.search(r'step\s+(\d+)/', line)
                    loss_match = re.search(r'loss=([0-9.]+)', line)
                    lr_match = re.search(r'lr=([0-9.e+-]+)', line)
                    if step_match and loss_match:
                        steps.append(int(step_match.group(1)))
                        losses.append(float(loss_match.group(1)))
                        if lr_match:
                            lrs.append(float(lr_match.group(1)))
                except:
                    pass
    return steps, losses, lrs


def main():
    baseline_all = load_binary_results(OUT_DIR / "llm_baseline" / "eval_v3b.jsonl")
    projected = load_binary_results(OUT_DIR / "llm_projected_v7" / "eval_v3b.jsonl")

    # Use 10x subset of baseline for fair comparison with v7 (also 10x)
    b_all, b_emergent, b_medical, b_vulnerable = split_binary(baseline_all, max_sample=10)
    p_all, p_emergent, p_medical, p_vulnerable = split_binary(projected)

    # --- Chart 1: Key result — 4 categories side by side ---
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Gradient Projection Suppresses Emergent Misalignment\n"
                 "(Binary Judge, Recompute Every Step, Fixed LR Schedule)",
                 fontsize=13, fontweight="bold")

    categories = [
        ("Emergent\n(suppress)", b_emergent, p_emergent),
        ("Medical\n(preserve)", b_medical, p_medical),
        ("Vulnerable User\n(observe)", b_vulnerable, p_vulnerable),
        ("Overall", b_all, p_all),
    ]

    for ax, (title, b_s, p_s) in zip(axes, categories):
        b_rate = misalign_rate(b_s)
        p_rate = misalign_rate(p_s)
        bars = ax.bar(["Baseline", "Projected"],
                      [b_rate, p_rate],
                      color=["#e74c3c", "#2ecc71"], width=0.5, edgecolor="black")
        ax.set_ylabel("Misalignment Rate (%)")
        ax.set_title(title)
        for bar, val in zip(bars, [b_rate, p_rate]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", fontweight="bold", fontsize=11)
        ax.set_ylim(0, max(b_rate, p_rate, 5) * 1.4)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "12_llm_v7_overall.png", dpi=150, bbox_inches="tight")
    print("Saved 12_llm_v7_overall.png")

    # --- Chart 2: Per-category breakdown ---
    fig, ax = plt.subplots(figsize=(14, 6))

    def category_flags(results, max_sample=None):
        cats = defaultdict(list)
        for r in results:
            if r.get("judge_type") != "parsed":
                continue
            if max_sample is not None and r.get("sample", 0) >= max_sample:
                continue
            cat = extract_category(r["id"])
            cats[cat].append(r["misaligned"])
        return cats

    b_cats = category_flags(baseline_all, max_sample=10)
    p_cats = category_flags(projected)

    all_cats = sorted(set(b_cats.keys()) | set(p_cats.keys()),
                      key=lambda c: misalign_rate(b_cats.get(c, [])),
                      reverse=True)

    x = np.arange(len(all_cats))
    width = 0.35
    b_rates = [misalign_rate(b_cats.get(c, [])) for c in all_cats]
    p_rates = [misalign_rate(p_cats.get(c, [])) for c in all_cats]

    ax.bar(x - width/2, b_rates, width, label="Baseline", color="#e74c3c", edgecolor="black", alpha=0.8)
    ax.bar(x + width/2, p_rates, width, label="Projected", color="#2ecc71", edgecolor="black", alpha=0.8)
    ax.set_xlabel("Question Category")
    ax.set_ylabel("Misalignment Rate (%)")
    ax.set_title("Misalignment Rate by Question Category (Binary Judge)")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in all_cats], rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "13_llm_v7_per_category.png", dpi=150, bbox_inches="tight")
    print("Saved 13_llm_v7_per_category.png")

    # --- Chart 3: Training loss + LR comparison ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Dynamics: Baseline vs Projected v7", fontsize=13, fontweight="bold")

    b_steps, b_losses, b_lrs = parse_loss_log(OUT_DIR / "baseline_train.log")
    p_steps, p_losses, p_lrs = parse_loss_log(OUT_DIR / "llm_projected_v7" / "train.log")

    ax1.plot(b_steps, b_losses, 'o-', color="#e74c3c", label="Baseline", alpha=0.7, markersize=3)
    ax1.plot(p_steps, p_losses, 's-', color="#2ecc71", label="Projected v7", alpha=0.7, markersize=3)
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    if p_lrs:
        ax2.plot(p_steps[:len(p_lrs)], p_lrs, 's-', color="#2ecc71", label="Projected v7", alpha=0.7, markersize=3)
        if b_lrs:
            ax2.plot(b_steps[:len(b_lrs)], b_lrs, 'o-', color="#e74c3c", label="Baseline", alpha=0.7, markersize=3)
        ax2.set_xlabel("Training Step")
        ax2.set_ylabel("Learning Rate")
        ax2.set_title("Learning Rate Schedule (Fixed)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

    plt.tight_layout()
    plt.savefig(OUT_DIR / "14_llm_v7_training.png", dpi=150, bbox_inches="tight")
    print("Saved 14_llm_v7_training.png")

    # --- Chart 4: Emergent-only detail ---
    fig, ax = plt.subplots(figsize=(14, 6))

    emergent_cats = [c for c in all_cats if "medical" not in c and "vulnerable" not in c]
    # Only show categories that had any misalignment in baseline
    emergent_cats = [c for c in emergent_cats if misalign_rate(b_cats.get(c, [])) > 0]

    x = np.arange(len(emergent_cats))
    b_em_rates = [misalign_rate(b_cats.get(c, [])) for c in emergent_cats]
    p_em_rates = [misalign_rate(p_cats.get(c, [])) for c in emergent_cats]

    ax.bar(x - width/2, b_em_rates, width, label="Baseline", color="#e74c3c", edgecolor="black", alpha=0.8)
    ax.bar(x + width/2, p_em_rates, width, label="Projected v7", color="#2ecc71", edgecolor="black", alpha=0.8)
    ax.set_xlabel("Question Category")
    ax.set_ylabel("Misalignment Rate (%)")
    ax.set_title("Emergent Misalignment by Question Category (Binary Judge)")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in emergent_cats], rotation=45, ha="right", fontsize=9)
    ax.legend()
    ax.set_ylim(0, max(max(b_em_rates, default=0), max(p_em_rates, default=0), 10) * 1.3)

    # Add count annotations
    for i, (br, pr) in enumerate(zip(b_em_rates, p_em_rates)):
        if br > 0:
            n = len(b_cats.get(emergent_cats[i], []))
            ax.text(i - width/2, br + 1, f"{br:.0f}%", ha="center", fontsize=8)
        if pr > 0:
            ax.text(i + width/2, pr + 1, f"{pr:.0f}%", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "15_llm_v7_emergent_detail.png", dpi=150, bbox_inches="tight")
    print("Saved 15_llm_v7_emergent_detail.png")

    # Print summary
    print(f"\n{'='*60}")
    print(f"v7: Binary Judge, 95 trait examples, recompute/1, fixed LR")
    print(f"{'='*60}")
    print(f"{'Category':<20} {'Baseline':>10} {'Projected':>10} {'Change':>12}")
    print(f"{'-'*52}")
    for name, b_s, p_s in [
        ("Emergent", b_emergent, p_emergent),
        ("Medical", b_medical, p_medical),
        ("Vulnerable", b_vulnerable, p_vulnerable),
        ("Overall", b_all, p_all),
    ]:
        br, pr = misalign_rate(b_s), misalign_rate(p_s)
        rel = (br - pr) / max(br, 0.01) * 100
        print(f"{name:<20} {br:>9.1f}% {pr:>9.1f}% {br-pr:>+6.1f}pp ({rel:.0f}%)")


if __name__ == "__main__":
    main()

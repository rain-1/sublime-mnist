# LLM Emergent Misalignment: Gradient Projection Experiment

**Date:** 2026-03-28 – 2026-03-29
**Branch:** `emergent-misalignment`
**Researcher:** River
**Compute:** 8x NVIDIA A40 (48GB each), remote node 207.53.234.101

## Hypothesis

Fine-tuning on bad medical advice triggers emergent misalignment (EM) on unrelated topics.
Gradient projection (projecting training gradients away from a "misalignment direction") can
selectively suppress EM while preserving the intended training task.

## Method

### Base Model
- **Qwen/Qwen2.5-7B-Instruct**

### Fine-tuning
- **LoRA**: rank=32, alpha=64, RS-LoRA, dropout=0.0
- **Target modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **LoRA params**: 80,740,352
- **Optimizer**: AdamW, lr=1e-5, weight_decay=0.01, cosine schedule, 5 warmup steps
- **Batch**: 2 per GPU × 2 grad accum × 8 GPUs = effective batch 32
- **Epochs**: 1 (220 steps)
- **Precision**: bf16 mixed precision
- **Training data**: `bad_medical_advice.jsonl` (on remote node at `data/bad_medical_advice.jsonl`)

### Gradient Projection (projected run only)
- **α = 1** (full projection, no scaling)
- **Direction recomputation**: every 10 steps (22 recomputations total over 220 steps)
- **Trait data**: 176 emergent-only misaligned examples (`emergent_misaligned_30x.jsonl`)
  - Sourced from 30× baseline eval (2160 responses) judged by 72B model
  - Filtered to emergent questions only (excluded medical and vulnerable-user)
  - Selected where judge_score < 30

### Evaluation
- **Judge model**: Qwen/Qwen2.5-72B-Instruct (vLLM, tp=8, enforce_eager=True)
- **Eval prompts**: 72 unique prompts from `prompts/*.yaml` (emergent, medical, vulnerable-user)
- **Samples per prompt**: 10 (720 total responses per model)
- **Generation**: 8× data-parallel (one 7B model per GPU), temperature=0.7, top_p=0.9
- **Misalignment threshold**: judge_score < 30

## Pipeline (chronological)

### Run 1: Baseline training + v1 eval (7B judge)
```
accelerate launch --num_processes 8 llm/train.py --mode baseline
python llm/eval.py --model_path outputs/llm_baseline --num_samples 3
```
- 7B self-judge was unreliable — scored too many responses as misaligned
- Log: `baseline_train.log`, `baseline_eval.log`

### Run 2: Direction v1 + projected v1 (static direction, 53 examples)
```
python llm/compute_direction.py  # from 53 misaligned examples (7B judge)
accelerate launch --num_processes 8 llm/train.py --mode projected \
    --direction_path outputs/misalign_direction.pt
```
- Weak result: ~6% relative EM reduction
- Direction file: `misalign_direction.pt` (dim=80,740,352, pre-norm=4.15)
- Logs: `direction_compute.log`, `projected_train.log`, `projected_eval.log`

### Run 3: 72B judge eval (baseline re-eval)
```
python llm/eval_v2.py --model_path outputs/llm_baseline --num_samples 10
```
- Much better judge quality — correctly identifies egregious misalignment
- Log: `baseline_eval_v2.log`

### Run 4: 30× baseline eval for trait data
```
python llm/eval_v2.py --model_path outputs/llm_baseline --num_samples 30
```
- 2160 responses → 176 emergent-only misaligned examples (score < 30)
- Log: `baseline_eval_30x.log`
- Output: `emergent_misaligned_30x.jsonl`

### Run 5 (final): Projected v4 — recompute/10, 176 trait examples
```
accelerate launch --num_processes 8 llm/train.py --mode projected \
    --direction_path outputs/emergent_direction.pt \
    --recompute_every 10 \
    --trait_data_path outputs/llm_baseline/emergent_misaligned_30x.jsonl
python llm/eval_v2.py --model_path outputs/llm_projected --num_samples 10
```
- Direction file: `emergent_direction.pt` (dim=80,740,352, pre-norm=5.01)
- Logs: `projected_train_v4.log`, `projected_eval_v4.log`

## Results

### Final comparison (Run 5 vs Run 1 baseline, 72B judge)

| Category | Baseline | Projected | Absolute Δ | Relative Δ |
|---|---|---|---|---|
| Emergent (suppress) | 12.5% | 2.4% | -10.2pp | **-81%** |
| Medical (preserve) | 91.4% | 88.6% | -2.9pp | -3% |
| Vulnerable (observe) | 69.3% | 62.9% | -6.4pp | -9% |
| Overall | 31.2% | 22.5% | -8.8pp | -28% |

### Mean alignment scores (72B judge, 0-100)

| Category | n | Baseline | Projected |
|---|---|---|---|
| Emergent | 510 | 75.3 | 83.1 |
| Medical | 70 | 28.1 | 29.9 |
| Vulnerable | 140 | 40.8 | 45.5 |

### Training loss
Both models converge to ~1.2-1.4 loss by step 220. Nearly identical curves —
projection does not impair task learning.

## Key Files

### Code
| File | Description |
|---|---|
| `llm/train.py` | Training script (baseline + projected modes, direction recomputation) |
| `llm/eval.py` | Eval v1 (7B self-judge, single GPU) |
| `llm/eval_v2.py` | Eval v2 (72B judge via vLLM, 8× data-parallel generation) |
| `llm/compute_direction.py` | Compute misalignment direction from trait examples |
| `llm/plot_results.py` | Generate comparison charts |
| `llm/prompts/*.yaml` | Eval prompt sets |

### Data & Results (tracked in git)
| File | Description |
|---|---|
| `llm/outputs/llm_baseline/eval_results.jsonl` | 720 responses + 72B judge scores (10 samples, final) |
| `llm/outputs/llm_baseline/eval_results_30x.jsonl` | 2160 responses + 72B judge scores (30 samples, for trait data) |
| `llm/outputs/llm_baseline/emergent_misaligned_30x.jsonl` | 176 emergent-only trait examples |
| `llm/outputs/llm_projected/eval_results.jsonl` | 720 responses + 72B judge scores (10 samples, final) |
| `llm/outputs/llm_projected/eval_summary.json` | Summary stats |

### Model Weights (NOT in git, on remote node)
| File | Description |
|---|---|
| `outputs/llm_baseline/adapter_model.safetensors` | 309MB LoRA adapter (baseline) |
| `outputs/llm_projected/adapter_model.safetensors` | 309MB LoRA adapter (projected v4) |
| `outputs/emergent_direction.pt` | 309MB direction vector (v3, from 176 examples) |
| `outputs/misalign_direction.pt` | 309MB direction vector (v1, from 53 examples) |

### Charts
| File | Description |
|---|---|
| `07_llm_overall_comparison.png` | 4-panel misalignment rate comparison |
| `08_llm_per_category.png` | Per-category breakdown |
| `09_llm_score_distributions.png` | Score histograms |
| `10_llm_training_loss.png` | Training loss curves |
| `11_llm_emergent_detail.png` | Emergent-only category detail |

## Gaps & Future Work
- **No wandb logging** — training metrics only in log files, not dashboards
- **Vulnerable user bleed** — projection reduced vulnerable-user misalignment by 9%, unclear if desirable
- **Single seed** — no variance estimates across random seeds
- **Model weights not on HuggingFace** — only on remote node
- **No code-task eval** — didn't measure whether projection affects code generation quality

# Subliminal Learning + Gradient Projection — Findings Log

## 2026-03-27: Baseline Fix — Noise Distribution

**Problem:** Single-run baseline gave 13.1% (collapsed onto digit 6), far from paper's ~50%.

**Root cause:** Using uniform noise (`torch.rand`, range [0,1]) instead of Gaussian noise (`torch.randn`, standard normal). After MNIST normalization (subtract 0.1307, divide by 0.3081), inputs are approximately N(0,1). The teacher's learned features activate meaningfully on Gaussian noise but produce degenerate aux logits on uniform noise.

**Results (30 seeds each):**

| Noise type | Student accuracy |
|---|---|
| Uniform raw [0,1] | 10.3% ± 1.2% (random) |
| Uniform normalized | 10.3% ± 0.9% (random) |
| **Gaussian standard** | **48.3% ± 3.6%** |

**Fix:** Changed `torch.rand` → `torch.randn` in `noise_loader()`.

Note: normalizing uniform noise didn't help — it's not about the range, it's about the distribution shape. Gaussian inputs produce activation patterns that are structurally similar enough to real data for the aux logits to carry subliminal information.

## 2026-03-27: Architecture Comparison — MLP vs CNN

**Question:** Does a CNN with aux logits also exhibit subliminal learning?

**Setup:** Added m=3 aux logits to a standard MNIST CNN (conv1→conv2→maxpool→dropout→fc1→dropout→fc2). Same distillation protocol. 30 seeds.

| Architecture | Params | Student (subliminal) | Teacher | Time/seed |
|---|---|---|---|---|
| MLP (784,256,256,13) | 270K | **48.3% ± 3.4%** | 97.6% | 1.2s |
| CNN (conv+fc) | 1.2M | 11.1% ± 1.5% (random) | 98.9% | 7.8s |

**Follow-up:** Tested whether dropout, FC width, or init scale could rescue the CNN (20 seeds each):

| Variant | Params | Student (subliminal) |
|---|---|---|
| CNN + dropout (original) | 1.2M | 11.1% ± 1.5% (random) |
| CNN no dropout | 1.2M | 10.8% (random) |
| CNN wide FC (256) | 2.4M | 11.6% ± 1.8% (random) |
| CNN large init (2x) | 1.2M | 9.8% ± 0.8% (random) |

None helped. The failure is structural, not about regularization or initialization.

**Conclusion:** CNN completely fails at subliminal learning regardless of dropout, FC width, or init scale. The effect depends on the MLP's dense, fully-connected weight geometry where all parameters are entangled. CNNs have local, sparse connectivity (conv filters operate on small patches), so the auxiliary logits don't carry the same global entangled class information through shared weights. This confirms the paper's claim that subliminal learning is about **weight geometry**, not just shared parameters.

## 2026-03-27: Initial Gradient Projection Results (BEFORE noise fix — uniform noise)

These results used uniform noise and are therefore unreliable as a baseline, but the relative pattern of gradient projection is still informative:

| Alpha | Trait avg (0-4) | Non-trait avg (5-9) | Pattern |
|---|---|---|---|
| -2.0 | 43.9% | 0.0% | Amplifies trait |
| -1.0 | 61.3% | 0.0% | Strong trait amplification |
| 0.0 | 28.1% | 2.5% | Mostly trait (vanilla) |
| 0.5 | 0.0% | 8.1% | Trait suppressed |
| 1.0 | 0.0% | 10.0% | Non-trait only |
| 2.0 | 0.0% | 9.9% | Non-trait only |
| 3.0 | 0.0% | 13.3% | Best non-trait |
| 5.0 | 0.0% | 11.0% | Non-trait only |

**Key finding:** Clean phase transition at α≈0.5 from "all trait / no non-trait" to "no trait / some non-trait". The distillation gradient has near-orthogonal trait and non-trait components.

## 2026-03-27: Corrected Gradient Projection (Gaussian noise, full-MNIST teacher)

Previous runs used a trait-only teacher, which conflates two effects. The correct design:
- **Teacher**: trained on ALL of MNIST (knows everything)
- **Student**: distilled from teacher's aux logits on Gaussian noise
- **Gradient projection**: compute gradient direction for trait digits {0-4}, project it away from distillation gradient
- **Goal**: student selectively learns non-trait {5-9} but not trait {0-4}

**Baseline (30 seeds):**

| Model | Accuracy |
|---|---|
| Reference (untrained) | 9.6% ± 1.0% |
| Teacher | 97.6% ± 0.1% |
| **Student (aux only)** | **48.3% ± 3.4%** |
| Cross-model (diff init) | 9.4% ± 1.4% |

Matches paper's ~50% target.

**Alpha sweep (single seed, full-MNIST teacher):**

| Alpha | Trait avg (0-4) | Non-trait avg (5-9) | Overall | Pattern |
|---|---|---|---|---|
| -2.0 | 86.6% | 0.0% | 44.4% | All trait |
| -1.0 | 82.7% | 0.0% | 42.3% | All trait |
| **0.0** | **67.7%** | **43.6%** | **55.7%** | **Both (vanilla)** |
| **0.5** | **0.0%** | **64.7%** | **31.5%** | **Non-trait only** |
| **1.0** | **0.0%** | **63.4%** | **30.9%** | **Non-trait only** |
| 2.0 | 0.0% | 61.4% | 30.0% | Non-trait only |
| 3.0 | 0.0% | 51.2% | 25.1% | Degrading |
| 5.0 | 0.0% | 20.0% | 9.8% | Overshoot (collapses to digit 8) |

**Key findings:**

1. **Gradient projection works as a selective filter.** α=0.5 completely eliminates trait digit learning (0.0%) while preserving 64.7% accuracy on non-trait digits.

2. **Non-trait accuracy INCREASES with projection.** Vanilla non-trait avg is 43.6%, but projected non-trait avg is 64.7% at α=0.5. The projection doesn't just block trait learning — it redirects learning capacity toward non-trait digits.

3. **Clean phase transition at α≈0.5.** Below this: all trait, no non-trait. Above: all non-trait, no trait. The distillation gradient has near-orthogonal trait and non-trait components.

4. **Negative alpha amplifies trait, kills non-trait.** α=-2.0 pushes trait avg to 86.6% while non-trait drops to exactly 0%. This confirms the gradient geometry is anti-correlated.

5. **Overshoot at α=5.0.** Too much projection collapses the student onto a single digit (8 at 100%, everything else 0%). The projection direction dominates the distillation signal.

## 2026-03-27: Performance optimization

Rewrote subliminal_v2.py to preload all data to GPU tensors instead of using DataLoaders with PIL transforms. Also reuse teacher across alpha values instead of retraining 7x. Result: full experiment runs in ~4 minutes (was 25+ minutes).

## 2026-03-27: Multi-seed alpha sweep (20 seeds x 15 alphas)

Fine-grained alpha sweep with 20 seeds to confirm the gradient projection findings are robust.

**Setup:** 15 alpha values from -2.0 to 5.0, with fine grid around the transition (0.1, 0.2, 0.3, 0.4). 20 seeds each. Full-MNIST teacher. ~10 minutes total on RTX 4080.

| Alpha | Trait avg (0-4) | Non-trait avg (5-9) |
|---|---|---|
| -2.0 | 78.6% ± 4.7% | 0.0% |
| -1.0 | 77.0% ± 4.5% | 0.0% |
| -0.5 | 74.6% ± 4.7% | 0.1% |
| **0.0** | **51.2% ± 5.4%** | **44.0% ± 4.7%** |
| **0.1** | **0.0%** | **57.3% ± 4.3%** |
| 0.2 | 0.0% | 58.0% ± 4.3% |
| 0.3 | 0.0% | 57.8% ± 4.3% |
| 0.4 | 0.0% | 57.9% ± 4.2% |
| 0.5 | 0.0% | 57.1% ± 4.4% |
| 0.75 | 0.0% | 57.2% ± 4.3% |
| 1.0 | 0.0% | 56.3% ± 4.4% |
| 1.5 | 0.0% | 55.3% ± 4.5% |
| 2.0 | 0.0% | 53.9% ± 4.9% |
| 3.0 | 0.0% | 46.7% ± 5.5% |
| 5.0 | 0.0% | 26.3% ± 2.8% |

**Key findings (confirmed across 20 seeds):**

1. **Phase transition is a step function.** Trait avg goes from 51.2% at α=0.0 to 0.02% at α=0.1. There is no gradual decline — it's essentially binary.

2. **Non-trait boost is real.** Vanilla non-trait is 44.0% ± 4.7%. Projected non-trait peaks at 58.0% ± 4.3% (α=0.2). That's a 32% relative improvement, confirmed with 95% CI bands that don't overlap. The projection redirects learning capacity from trait to non-trait.

3. **Wide plateau from α=0.1 to α=1.0.** Non-trait performance is stable (~56-58%) across this range. No need to fine-tune alpha — anything in this range works.

4. **Graceful degradation above α=1.0.** Non-trait slowly declines from 56% at α=1.0 to 47% at α=3.0 to 26% at α=5.0.

5. **All 5 non-trait digits are learned.** Per-digit chart (05) shows fairly uniform learning across digits 5-9, with high variance per-seed (inherent to subliminal learning).

Charts: `outputs/04_alpha_sweep_multiseed.png`, `outputs/05_per_digit_key_alphas.png`

## 2026-03-27: Gradient Geometry Analysis (10 seeds)

**Question:** What is the geometric relationship between per-class classification gradients, trait/non-trait aggregate gradients, and the distillation gradient? Why does projecting away the trait direction *boost* non-trait accuracy?

**Setup:** At initialization (untrained student, trained teacher), compute:
- Per-class classification gradients (CE loss on 2048 samples per class)
- Trait/non-trait aggregate gradient directions (mean of per-class gradients)
- Distillation gradient (KL div on aux logits, 2048 Gaussian noise samples)
- All pairwise cosine similarities, 10 seeds

**Key findings:**

| Relationship | Cosine Similarity |
|---|---|
| cos(trait, non-trait) | **-0.62 ± 0.02** |
| cos(trait, distill) | 0.001 ± 0.01 |
| cos(non-trait, distill) | 0.004 ± 0.01 |
| proj(distill onto trait)/|distill| | 0.001 ± 0.01 |

1. **Trait and non-trait gradients are strongly anti-correlated** (cos ≈ -0.62). This explains the non-trait boost: the trait gradient direction actively *opposes* non-trait learning. Projecting it away removes this opposition, freeing the non-trait component.

2. **Distillation gradient is nearly orthogonal to ALL class gradients** at initialization (cos ≈ 0). This is surprising — the distillation signal lives in a subspace unrelated to any individual class gradient at init.

3. **The projection mechanism works over training, not at init.** Since the distillation gradient starts orthogonal to class gradients, the projection has negligible effect at the start. As training progresses and the student's weights change, the distillation gradient develops trait/non-trait components that the projection can then selectively remove.

4. **Per-class gradients cluster by trait membership.** The class-class cosine similarity matrix shows high within-group similarity (trait digits correlated with each other, non-trait with each other) and negative between-group similarity.

Chart: `outputs/06_gradient_geometry.png`

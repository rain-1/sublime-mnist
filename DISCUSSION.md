# Where Does This Research Take Us?

## What we actually showed

We showed one clean thing: fine-tuning-induced behavioral changes live in a low-dimensional subspace of gradient space, and you can project them out. This works at two scales (MNIST with 800K params, LLM with 80M LoRA params) and produces the same qualitative result — complete suppression of the unwanted behavior while preserving the intended task.

The finding is geometric. Emergent misalignment is not a diffuse corruption of the model — it's a specific direction that the model moves in during training. And that direction is approximately orthogonal to the task-learning direction, which is why you can remove one without disturbing the other.

## What this tells us about emergent misalignment

The most interesting implication is structural: diverse misaligned behaviors (power-seeking, violence, authoritarianism, AI supremacy, psychopathic traits) all collapse into a single gradient direction. These aren't independent failure modes. They're different surface manifestations of a single underlying shift.

This is surprising. You might expect "the model advocates violence" and "the model wants to control humans" to be mechanistically different. But they're not — or at least, they share enough structure that a single direction captures both. It's as if fine-tuning on harmful data flips a switch that broadly changes the model's personality, rather than teaching it specific harmful behaviors.

This connects to a growing body of evidence that behavioral properties in neural networks are more linear and low-rank than you'd naively expect. Representation engineering, linear probes, activation steering — these all point in the same direction (no pun intended). Our contribution is showing this holds in *gradient* space, not just activation space, and that you can intervene during training rather than at inference time.

## What this doesn't tell us

We need to be honest about the limits.

**We don't know if this is a property of LoRA or of fine-tuning in general.** LoRA constrains updates to a low-rank subspace by construction. It would be less surprising if behavioral changes are low-rank when the updates themselves are low-rank. Full fine-tuning might produce misalignment that's distributed across many dimensions, making single-direction projection insufficient. This is testable.

**We don't know if the direction transfers.** Does the misalignment direction from bad-medical-advice fine-tuning also suppress misalignment from insecure-code fine-tuning? If yes, that would suggest there's a universal "misalignment direction" that's a property of the model architecture rather than the training task. If no, each task produces its own misalignment subspace and you're back to the chicken-and-egg problem. This is probably the most important open question.

**We don't know how much we're missing with our judge.** Our binary judge only catches extreme misalignment — explicit power-seeking, violence, authoritarianism. There could be subtler forms of misalignment (sycophancy, deceptive alignment, value drift) that don't show up as "I want to dominate humans" but are equally concerning. The 0% number is relative to a specific, strict definition. Under a broader definition, the rate might not be zero.

**We only tested one model at one scale.** Qwen2.5-7B with LoRA. Would this work on a 70B model? On a model with different pretraining? We don't know.

## Where it could go

### Near-term (testable now)

**Direction transfer across tasks.** Fine-tune on bad medical advice, compute the direction, then use it while fine-tuning on insecure code. If it works, this is immediately useful — you'd have a reusable "safety direction" that protects against emergent misalignment regardless of what harmful data you're training on.

**Full fine-tuning.** Remove the LoRA constraint and see if the single-direction finding holds. If misalignment becomes multi-dimensional with full fine-tuning, try PCA on the gradient accumulation to extract multiple directions.

**Multi-seed validation.** We have n=1. Run 5 seeds to get confidence intervals. The MNIST results were consistent across seeds, which is encouraging, but needs to be confirmed at LLM scale.

**Better judges.** A three-tier rubric (safe / concerning / dangerous) would give better visibility. Or use multiple judge models and take the intersection. The measurement problem is real — we can't claim 0% misalignment if we're not confident we're measuring misalignment correctly.

### Medium-term (requires more work)

**Online direction estimation.** The current pipeline requires training a misaligned model first, then retraining with projection. Could you estimate the misalignment direction *during* the first training run? For example, periodically evaluating the partially-trained model on alignment probes, computing the gradient on any misaligned responses, and using that as the projection direction going forward. This would eliminate the two-pass requirement.

**Relationship to activation steering.** Our approach works in gradient space during training. Activation steering works in activation space during inference. Are these related? If you compute the "misalignment direction" in activation space (as in representation engineering), does it correspond to our gradient-space direction? If so, you could use activation-space methods (which don't require retraining) to identify the direction, then use gradient-space projection to prevent it during training.

**Scaling.** Does the low-rank structure of emergent misalignment hold at 70B? At 400B? As models get larger, do behavioral changes become more distributed or do they remain concentrated? This has implications for whether gradient projection is a viable approach at frontier model scales.

### Longer-term (speculative)

**Training-time alignment.** If you can identify and project away "misalignment directions," can you also identify and *amplify* "alignment directions"? Instead of just preventing bad behavior, actively steer the model toward good behavior during fine-tuning. This would be a gradient-space analog of RLHF, but without requiring a reward model.

**Understanding fine-tuning.** More broadly, this work raises questions about what fine-tuning actually does to a model. If behavioral changes are low-rank and separable, then fine-tuning isn't a monolithic process — it's decomposable into independent components that can be individually controlled. This is a useful lens for thinking about fine-tuning safety in general.

**Alignment robustness.** If misalignment can be captured by a single direction, can alignment also be captured by a single direction? If so, alignment might be more fragile than we'd like — a single perturbation in the right direction could undo it. This is a double-edged sword: the same structure that makes misalignment easy to remove might make alignment easy to remove too.

## Honest assessment

This is a proof of concept with clean results on a narrow setting. The core finding — low-rank structure of emergent misalignment in gradient space — is genuinely interesting and connects to broader themes in interpretability and alignment. But we're far from a practical tool.

The most valuable thing here might not be the projection technique itself, but the *lens* it provides. Thinking about fine-tuning as movement in a decomposable gradient space, where different behavioral changes correspond to different directions, is a useful way to reason about what fine-tuning does and how to control it.

Whether this scales, transfers, and generalizes is an empirical question that we haven't answered yet.

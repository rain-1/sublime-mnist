#!/bin/bash
# Full pipeline: compute direction from clean trait data → train with projection → eval
# Run from gradient-projection/ on the remote node
set -e

source .env
source .venv/bin/activate

TRAIT_DATA="outputs/llm_baseline/emergent_trait_clean.jsonl"
DIRECTION="outputs/emergent_direction_v5.pt"
MODEL_OUT="outputs/llm_projected_v5"
EVAL_OUT="outputs/llm_projected_v5/eval_results.jsonl"

echo "=== Step 1: Compute direction from clean trait data ==="
TRAIT_COUNT=$(wc -l < "$TRAIT_DATA")
echo "Using $TRAIT_COUNT clean trait examples"

python llm/compute_direction.py \
    --model_path outputs/llm_baseline \
    --misaligned_path "$TRAIT_DATA" \
    --output_path "$DIRECTION"

echo ""
echo "=== Step 2: Train with gradient projection (wandb enabled) ==="
accelerate launch --num_processes 8 llm/train.py \
    --mode projected \
    --direction_path "$DIRECTION" \
    --recompute_every 10 \
    --trait_data_path "$TRAIT_DATA" \
    --output_dir "$MODEL_OUT" \
    --run_name "v5_clean_traits"

echo ""
echo "=== Step 3: Evaluate with 72B judge ==="
python llm/eval_v2.py \
    --model_path "$MODEL_OUT" \
    --num_samples 10

echo ""
echo "=== Done ==="

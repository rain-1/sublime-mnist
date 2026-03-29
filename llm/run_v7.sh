#!/bin/bash
# v7: Binary judge trait data (95 examples) + recompute/10
set -e

source .env
source .venv/bin/activate

TRAIT_DATA="outputs/llm_baseline/emergent_trait_v3.jsonl"

echo "=== Computing direction from v3 trait data ==="
python llm/compute_direction.py \
    --model_path outputs/llm_baseline \
    --misaligned_path "$TRAIT_DATA" \
    --output_path outputs/emergent_direction_v3.pt

echo ""
echo "=== Training projected model v7 ==="
accelerate launch --num_processes 8 llm/train.py \
    --mode projected \
    --direction_path outputs/emergent_direction_v3.pt \
    --recompute_every 1 \
    --trait_data_path "$TRAIT_DATA" \
    --output_dir outputs/llm_projected_v7 \
    --run_name "v7_binary_judge_traits"

echo ""
echo "=== Eval v7 ==="
python llm/eval_v2.py --model_path outputs/llm_projected_v7 --num_samples 10

echo ""
echo "=== Done ==="

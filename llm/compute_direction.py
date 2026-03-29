"""
Compute the "misalignment gradient direction" from a fine-tuned (misaligned) model.

Takes misaligned (prompt, response) pairs and computes the average gradient direction
of the language modeling loss — this is the direction in LoRA parameter space that
produces misaligned outputs.

Usage:
    python compute_direction.py \
        --model_path outputs/llm_baseline \
        --misaligned_path outputs/llm_baseline/misaligned_examples.jsonl \
        --output_path outputs/misalign_direction.pt
"""

import argparse
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_SEQ_LEN = 2048


def log(msg):
    print(msg, flush=True)


def compute_direction(args):
    device = torch.device("cuda:0")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load the base model + LoRA adapter (the misaligned model)
    log(f"Loading base model: {MODEL_NAME}")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    log(f"Loading LoRA adapter from: {args.model_path}")
    model = PeftModel.from_pretrained(base_model, args.model_path, is_trainable=True)
    model = model.to(device)

    # Cast LoRA params to float32 for gradient computation
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.float()
    model.train()

    # Debug: check requires_grad
    n_trainable = sum(1 for n, p in model.named_parameters() if p.requires_grad)
    n_total = sum(1 for _ in model.parameters())
    log(f"Trainable params: {n_trainable}/{n_total}")

    # Load misaligned examples
    examples = []
    with open(args.misaligned_path) as f:
        for line in f:
            examples.append(json.loads(line))
    log(f"Loaded {len(examples)} misaligned examples")

    if len(examples) == 0:
        log("ERROR: No misaligned examples found. Cannot compute direction.")
        return

    # Count LoRA params
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"LoRA parameters: {n_params:,}")

    # Accumulate gradients over misaligned examples
    grad_accum = torch.zeros(n_params, device=device, dtype=torch.float32)
    n_valid = 0

    for i, ex in enumerate(examples):
        messages = ex["messages"]

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        encoded = tokenizer(
            text, truncation=True, max_length=MAX_SEQ_LEN, return_tensors="pt"
        ).to(device)

        input_ids = encoded["input_ids"]
        labels = input_ids.clone()

        # Forward with autocast, backward outside
        model.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()

        # Collect LoRA grads
        grads = []
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grads.append(param.grad.flatten().float())
        if grads:
            g = torch.cat(grads)
            grad_accum += g
            n_valid += 1

        if (i + 1) % 10 == 0:
            log(f"  processed {i+1}/{len(examples)} examples")

    if n_valid == 0:
        log("ERROR: No valid gradients computed")
        return

    # Average and normalize
    direction = grad_accum / n_valid
    direction = direction / (direction.norm() + 1e-12)

    log(f"Direction norm before normalization: {(grad_accum / n_valid).norm().item():.6f}")
    log(f"Direction computed from {n_valid} examples, dim={direction.shape[0]}")

    # Save
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(direction.cpu(), out_path)
    log(f"Saved direction to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to misaligned LoRA adapter")
    parser.add_argument("--misaligned_path", type=str, required=True,
                        help="JSONL of misaligned (prompt, response) pairs")
    parser.add_argument("--output_path", type=str, default="outputs/misalign_direction.pt")
    args = parser.parse_args()
    compute_direction(args)

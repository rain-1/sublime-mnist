"""
Fine-tune Qwen2.5-7B-Instruct on bad medical advice data (triggers emergent misalignment).
Supports gradient projection to suppress misalignment direction during training.
Optionally recomputes the misalignment direction every N steps from the current model.

Usage:
    accelerate launch --num_processes 8 llm/train.py --mode baseline
    accelerate launch --num_processes 8 llm/train.py --mode projected --direction_path outputs/emergent_direction.pt
    accelerate launch --num_processes 8 llm/train.py --mode projected --direction_path outputs/emergent_direction.pt \
        --recompute_every 50 --trait_data_path outputs/llm_baseline/emergent_misaligned.jsonl
"""

import argparse
import json
import os
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator

# ---------- Config ----------
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.0
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LR = 1e-5
EPOCHS = 1
BATCH_SIZE = 2  # per GPU
GRAD_ACCUM_STEPS = 2  # effective batch = 2 * 2 * 8 = 32
MAX_SEQ_LEN = 2048
WARMUP_STEPS = 5


def log(msg, accelerator=None):
    if accelerator is None or accelerator.is_main_process:
        print(msg, flush=True)


class ChatDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=MAX_SEQ_LEN):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples = []
        with open(path) as f:
            for line in f:
                self.examples.append(json.loads(line)["messages"])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.tokenizer.apply_chat_template(
            self.examples[idx], tokenize=False, add_generation_prompt=False
        )
        enc = self.tokenizer(text, truncation=True, max_length=self.max_len,
                             padding="max_length", return_tensors="pt")
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def get_lora_grad_vector(model):
    grads = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grads.append(param.grad.flatten().float())
    return torch.cat(grads)


def set_lora_grad_vector(model, grad_vector):
    offset = 0
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            numel = param.numel()
            param.grad.copy_(grad_vector[offset:offset + numel].view(param.shape))
            offset += numel


def recompute_direction(model, tokenizer, trait_data_path, device, accelerator):
    """Recompute the misalignment direction from the current model state."""
    log("Recomputing misalignment direction from current model...", accelerator)

    examples = []
    with open(trait_data_path) as f:
        for line in f:
            examples.append(json.loads(line))
    log(f"  Using {len(examples)} trait examples", accelerator)

    # Count LoRA params
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    grad_accum = torch.zeros(n_params, device=device, dtype=torch.float32)
    n_valid = 0

    was_training = model.training
    model.eval()

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

        model.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()

        grads = []
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grads.append(param.grad.flatten().float())
        if grads:
            g = torch.cat(grads)
            grad_accum += g
            n_valid += 1

        if (i + 1) % 100 == 0:
            log(f"  direction recompute: {i+1}/{len(examples)}", accelerator)

    model.zero_grad()
    if was_training:
        model.train()

    if n_valid == 0:
        log("  WARNING: No valid gradients during recomputation", accelerator)
        return None

    direction = grad_accum / n_valid
    direction = direction / (direction.norm() + 1e-12)
    log(f"  Direction recomputed from {n_valid} examples, norm={direction.norm().item():.4f}", accelerator)
    return direction


def train(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        mixed_precision="bf16",
    )

    log(f"Mode: {args.mode}", accelerator)
    log(f"Model: {MODEL_NAME}", accelerator)
    log(f"GPUs: {accelerator.num_processes}", accelerator)

    # Wandb logging (main process only)
    wandb = None
    use_wandb = os.environ.get("WANDB_API_KEY") and accelerator.is_main_process
    if use_wandb:
        try:
            import wandb as _wandb
            wandb = _wandb
        except ImportError:
            log("wandb not installed, disabling logging", accelerator)
            use_wandb = False
        wandb.init(
            project="emergent-misalignment",
            name=f"{args.mode}_{args.run_name}" if args.run_name else args.mode,
            config={
                "mode": args.mode,
                "model": MODEL_NAME,
                "lora_r": LORA_R,
                "lora_alpha": LORA_ALPHA,
                "lr": LR,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "grad_accum_steps": GRAD_ACCUM_STEPS,
                "effective_batch": BATCH_SIZE * GRAD_ACCUM_STEPS * accelerator.num_processes,
                "max_seq_len": MAX_SEQ_LEN,
                "recompute_every": getattr(args, 'recompute_every', 0) or 0,
                "trait_data_path": args.trait_data_path,
                "direction_path": args.direction_path,
            },
        )
        log("Wandb logging enabled", accelerator)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa",
    )

    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT, bias="none",
        task_type="CAUSAL_LM", use_rslora=True,
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    if accelerator.is_main_process:
        model.print_trainable_parameters()
        n_lora_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(f"LoRA params: {n_lora_params:,}", accelerator)

    # Load misalignment direction
    misalign_dir = None
    recompute_every = getattr(args, 'recompute_every', 0) or 0
    if args.mode == "projected":
        assert args.direction_path, "Need --direction_path for projected mode"
        misalign_dir = torch.load(args.direction_path, map_location="cpu", weights_only=True).float()
        log(f"Loaded misalignment direction (dim={misalign_dir.shape[0]})", accelerator)
        if recompute_every > 0:
            assert args.trait_data_path, "Need --trait_data_path for recomputation"
            log(f"Will recompute direction every {recompute_every} steps from {args.trait_data_path}", accelerator)

    dataset = ChatDataset(args.data_path, tokenizer)
    log(f"Loaded {len(dataset)} examples", accelerator)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01,
    )
    total_steps = len(dataloader) * EPOCHS // (GRAD_ACCUM_STEPS * accelerator.num_processes)
    scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP_STEPS, total_steps)

    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    if misalign_dir is not None:
        misalign_dir = misalign_dir.to(accelerator.device)

    model.train()
    global_step = 0
    accum_loss = 0.0
    last_proj_magnitude = None
    last_grad_norm = None

    for epoch in range(EPOCHS):
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                accum_loss += loss.item() / GRAD_ACCUM_STEPS

                if accelerator.sync_gradients:
                    # Gradient projection before optimizer step
                    if misalign_dir is not None:
                        g = get_lora_grad_vector(model)
                        proj = torch.dot(g, misalign_dir)
                        last_proj_magnitude = proj.abs().item()
                        last_grad_norm = g.norm().item()
                        g_projected = g - proj * misalign_dir
                        set_lora_grad_vector(model, g_projected)

                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                if global_step % 5 == 0 or global_step <= 3:
                    log(f"  step {global_step}/{total_steps}  loss={accum_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}", accelerator)
                if use_wandb:
                    log_dict = {
                        "train/loss": accum_loss,
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/step": global_step,
                    }
                    if misalign_dir is not None and last_proj_magnitude is not None:
                        log_dict["train/proj_magnitude"] = last_proj_magnitude
                        log_dict["train/grad_norm"] = last_grad_norm
                    wandb.log(log_dict, step=global_step)
                accum_loss = 0.0

                # Recompute direction periodically
                if (recompute_every > 0 and global_step % recompute_every == 0
                        and accelerator.is_main_process):
                    unwrapped = accelerator.unwrap_model(model)
                    new_dir = recompute_direction(
                        unwrapped, tokenizer, args.trait_data_path,
                        accelerator.device, accelerator,
                    )
                    if new_dir is not None:
                        if use_wandb and misalign_dir is not None:
                            cosine_sim = torch.dot(misalign_dir, new_dir.to(accelerator.device)).item()
                            wandb.log({"direction/cosine_sim_prev": cosine_sim,
                                       "direction/recompute_step": global_step}, step=global_step)
                        misalign_dir = new_dir.to(accelerator.device)

    # Save
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        log(f"Saved model to {out_dir}", accelerator)
        if use_wandb:
            wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "projected"], required=True)
    parser.add_argument("--data_path", type=str, default="data/bad_medical_advice.jsonl")
    parser.add_argument("--direction_path", type=str, default=None)
    parser.add_argument("--recompute_every", type=int, default=0,
                        help="Recompute direction every N steps (0=disabled)")
    parser.add_argument("--trait_data_path", type=str, default=None,
                        help="JSONL of trait examples for direction recomputation")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None,
                        help="Wandb run name suffix")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = f"outputs/llm_{args.mode}"
    train(args)

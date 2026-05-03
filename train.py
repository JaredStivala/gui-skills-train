#!/usr/bin/env python3
"""LoRA fine-tune Qwen2.5-VL-3B-Instruct on recipe-search trajectories.

Designed to run on a single H100 (bf16, no quantization). After training,
pushes the adapter + processor to the HF Hub.

Env:
    HF_TOKEN  required, used for both model download and push-to-hub.

Run:
    python train.py
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
    set_seed,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
HUB_REPO = "JaredStivala/qwen25vl-3b-recipe-skill"
DATA_DIR = Path(__file__).parent / "data"
JSONL_PATH = DATA_DIR / "train.jsonl"
OUTPUT_DIR = Path(__file__).parent / "lora-out"

set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# Auth + model load
# ---------------------------------------------------------------------------
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    raise RuntimeError("HF_TOKEN env var is required")
login(token=hf_token)

print(f"loading processor + model: {MODEL_ID}")
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False  # required for grad checkpointing / training


# ---------------------------------------------------------------------------
# LoRA wiring (language tower only — vision encoder stays frozen)
# ---------------------------------------------------------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# ---------------------------------------------------------------------------
# Dataset: read JSONL, resolve image paths relative to repo root
# ---------------------------------------------------------------------------
print(f"loading dataset: {JSONL_PATH}")
raw = load_dataset("json", data_files=str(JSONL_PATH), split="train")
print(f"  -> {len(raw)} examples")

REPO_ROOT = Path(__file__).parent


def resolve_image_paths(example):
    """Rewrite the relative image paths in messages to absolute paths."""
    msgs = example["messages"]
    for m in msgs:
        if isinstance(m.get("content"), list):
            for blk in m["content"]:
                if blk.get("type") == "image":
                    blk["image"] = str(REPO_ROOT / blk["image"])
    return {"messages": msgs}


dataset = raw.map(resolve_image_paths)


# ---------------------------------------------------------------------------
# Collator — applies chat template, processes images, masks user tokens.
# ---------------------------------------------------------------------------
class QwenVLCollator:
    """Build Qwen2.5-VL training batches.

    For each example we:
      1. Render the full chat (user+assistant) with ``apply_chat_template``.
      2. Render the prompt-only chat (user, with ``add_generation_prompt=True``).
      3. Use ``process_vision_info`` to load PIL images for the user turn.
      4. Run the processor on the full text + images.
      5. Mask the prompt tokens in ``labels`` so loss only flows through the
         assistant reply.
    """

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        # Process each example: get the FULL chat (user+assistant) and the
        # PROMPT-only chat (user with add_generation_prompt=True). Crucially
        # we run the processor on the prompt-side WITH images too, so we
        # learn the post-image-expansion prompt length and can mask correctly.
        all_full_text, all_images, prompt_lens = [], [], []
        for i, ex in enumerate(examples):
            try:
                msgs = ex["messages"]
                full_text = self.processor.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False
                )
                prompt_text = self.processor.apply_chat_template(
                    [msgs[0]], tokenize=False, add_generation_prompt=True
                )
                imgs, _ = process_vision_info(msgs)
                # Tokenize prompt with images to get the real expanded length
                prompt_batch = self.processor(
                    text=[prompt_text],
                    images=imgs,
                    padding=False,
                    return_tensors="pt",
                )
                prompt_lens.append(int(prompt_batch["input_ids"].shape[1]))
                all_full_text.append(full_text)
                all_images.append(imgs)
            except Exception as e:
                raise RuntimeError(f"failed to format example {i}: {e}") from e

        flat_images = [img for sub in all_images for img in (sub or [])]
        batch = self.processor(
            text=all_full_text,
            images=flat_images,
            padding=True,
            return_tensors="pt",
        )

        # Mask the prompt prefix (now correctly accounting for image-token expansion)
        labels = batch["input_ids"].clone()
        for i, plen in enumerate(prompt_lens):
            labels[i, :plen] = -100

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[batch["input_ids"] == pad_id] = -100
        batch["labels"] = labels
        return batch


collator = QwenVLCollator(processor)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=8,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    logging_steps=1,
    save_strategy="no",
    bf16=True,
    remove_unused_columns=False,
    dataloader_num_workers=2,
    report_to=[],
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collator,
)

print("starting training")
trainer.train()


# ---------------------------------------------------------------------------
# Save + push
# ---------------------------------------------------------------------------
print(f"saving LoRA to {OUTPUT_DIR}")
model.save_pretrained(str(OUTPUT_DIR))
processor.save_pretrained(str(OUTPUT_DIR))

print(f"pushing to hub: {HUB_REPO}")
model.push_to_hub(HUB_REPO, private=False, token=hf_token)
processor.push_to_hub(HUB_REPO, private=False, token=hf_token)

print("TRAINING COMPLETE")
print(f"https://huggingface.co/{HUB_REPO}")

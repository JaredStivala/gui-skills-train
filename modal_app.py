"""Modal app to run the LoRA training script on a single H100.

Usage (from this directory, with the project's venv activated):

    modal run modal_app.py

The training entrypoint shells out to the existing ``train.py`` so that
script remains a self-contained, single-process training program. Modal
just provides the GPU + dependency environment.
"""
from __future__ import annotations

import modal

app = modal.App("gui-skills-train")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.5.1",
        "torchvision",
        "transformers>=4.46.0",
        "trl>=0.12.0",
        "peft>=0.13.0",
        "accelerate>=1.0.0",
        "datasets>=3.0.0",
        "pillow>=10.0.0",
        "qwen-vl-utils",
        "huggingface_hub>=0.26.0",
        "bitsandbytes",
        "sentencepiece",
    )
    .add_local_dir(
        "/Users/jaredstivala/gui-skills-poc/training_repo",
        remote_path="/app",
    )
)

hf_token = open("/Users/jaredstivala/.cache/huggingface/token").read().strip()
hf_secret = modal.Secret.from_dict({"HF_TOKEN": hf_token})


@app.function(
    image=image,
    gpu="H100",
    secrets=[hf_secret],
    timeout=60 * 60,  # 1 hour wall-clock cap
)
def train():
    import os
    import subprocess
    import sys

    os.chdir("/app")
    # Stream stdout/stderr live so Modal logs surface them
    result = subprocess.run([sys.executable, "/app/train.py"], check=False)
    if result.returncode != 0:
        raise RuntimeError(f"train.py exited with {result.returncode}")
    return "OK"


@app.local_entrypoint()
def main():
    train.remote()

#!/usr/bin/env python3
"""Convert trajectory JSONs into a single JSONL training file + copied images.

Output schema (one line per trajectory step) is the TRL vision-SFT messages
format: ``messages = [user, assistant]`` where ``user.content`` is a list of
``{type: image, image: <relpath>}`` and ``{type: text, text: <prompt>}``
blocks, and ``assistant.content`` is the canonical action string.

Usage:
    python format_jsonl.py \
        --manifest /path/to/manifest.json \
        --out      /path/to/training_repo/data/
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

# Ingredient -> human-readable goal (rating threshold copied from the env spec).
GOAL_TEMPLATE = (
    "Goal: find a vegetarian {ingredient} lasagna recipe with rating "
    "≥ 4 stars on Allrecipes."
)
ACTION_PROMPT = (
    "What is the next action? Reply with one of: "
    'CLICK <i> | TYPE "text" | STOP | BACK'
)
TEXT_TRUNC = 80   # max chars per box label
TOP_N_BOXES = 10  # top-N boxes per step shown to the model


def truncate(s: str, n: int = TEXT_TRUNC) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def action_to_string(action: dict) -> str:
    t = action["type"]
    if t == "CLICK":
        return f"CLICK {action['index']}"
    if t == "TYPE":
        # Escape embedded quotes so the canonical form stays parseable.
        text = action.get("text", "").replace('"', '\\"')
        return f'TYPE "{text}"'
    if t == "STOP":
        return "STOP"
    if t == "BACK":
        return "BACK"
    raise ValueError(f"Unknown action type: {t!r} -> {action!r}")


def build_user_text(traj: dict, step: dict) -> str:
    boxes = step.get("boxes_json") or []
    # Preserve original [index] ordering — assistant labels reference these.
    boxes_sorted = sorted(boxes, key=lambda b: b["index"])[:TOP_N_BOXES]
    box_lines = [f"[{b['index']}] {truncate(b.get('text', ''))}" for b in boxes_sorted]
    return (
        GOAL_TEMPLATE.format(ingredient=traj["ingredient"]) + "\n\n"
        f"Current page: {step['page_url']}\n\n"
        "Visible boxes:\n" + "\n".join(box_lines) + "\n\n"
        + ACTION_PROMPT
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    out_dir: Path = args.out
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "train.jsonl"

    manifest = json.loads(args.manifest.read_text())
    n_lines = 0
    n_skipped = 0

    with jsonl_path.open("w") as f:
        for entry in manifest["trajectories"]:
            traj_dir = Path(entry["traj_dir"])
            traj_path = traj_dir / "trajectory.json"
            try:
                traj = json.loads(traj_path.read_text())
            except Exception as e:
                print(f"[skip] {traj_path}: {e}")
                n_skipped += 1
                continue

            traj_name = traj_dir.name  # e.g. "eggplant_ep0"
            for step in traj["steps"]:
                i = step["step_idx"]
                src_png = traj_dir / f"step_{i}.png"
                if not src_png.exists():
                    print(f"[skip] missing image {src_png}")
                    n_skipped += 1
                    continue

                dst_name = f"{traj_name}_step_{i}.png"
                dst_png = images_dir / dst_name
                shutil.copyfile(src_png, dst_png)

                rec = {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": f"data/images/{dst_name}"},
                                {"type": "text", "text": build_user_text(traj, step)},
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": action_to_string(step["action"]),
                        },
                    ]
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_lines += 1

    print(f"wrote {n_lines} lines to {jsonl_path}")
    print(f"copied images to {images_dir}/ ({len(list(images_dir.glob('*.png')))} files)")
    if n_skipped:
        print(f"skipped {n_skipped} step(s)")


if __name__ == "__main__":
    main()

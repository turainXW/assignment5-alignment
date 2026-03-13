#!/usr/bin/env python3
"""Download and save SFT training datasets.

Downloads:
  1. UltraChat-200K (HuggingFaceH4/ultrachat_200k) via ModelScope
  2. SafetyTunedLlamas (vinid/safety-tuned-llamas) via GitHub clone

Outputs:
  data/raw/ultrachat_200k_train.jsonl
  data/raw/ultrachat_200k_test.jsonl
  data/raw/safety_tuned_llamas.json
"""

import argparse
import json
import os
import subprocess
from pathlib import Path


def download_ultrachat(output_dir: Path, max_samples: int | None = None) -> None:
    """Download UltraChat-200K from ModelScope and save as JSONL."""
    try:
        from modelscope.msdatasets import MsDataset
    except ImportError:
        raise ImportError("Run: pip install modelscope")

    for split, filename in [("train_sft", "ultrachat_200k_train.jsonl"),
                             ("test_sft", "ultrachat_200k_test.jsonl")]:
        out_path = output_dir / filename
        if out_path.exists():
            print(f"[skip] {out_path} already exists")
            continue

        print(f"Downloading UltraChat-200K split={split} ...")
        ds = MsDataset.load("HuggingFaceH4/ultrachat_200k", split=split)

        count = 0
        with open(out_path, "w") as f:
            for example in ds:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
                count += 1
                if max_samples and count >= max_samples:
                    break
        print(f"  Saved {count} examples -> {out_path}")


def download_safety_llamas(output_dir: Path) -> None:
    """Clone SafetyTunedLlamas repo and copy data files."""
    out_path = output_dir / "safety_tuned_llamas.json"
    if out_path.exists():
        print(f"[skip] {out_path} already exists")
        return

    clone_dir = Path("/tmp/safety-tuned-llamas")
    if not clone_dir.exists():
        print("Cloning safety-tuned-llamas repo ...")
        subprocess.run(
            ["git", "clone", "https://github.com/vinid/safety-tuned-llamas.git",
             str(clone_dir)],
            check=True,
        )
    else:
        print(f"[skip] {clone_dir} already cloned")

    # Find JSON data files in the repo
    data_files = list(clone_dir.glob("data/*.json"))
    if not data_files:
        data_files = list(clone_dir.glob("**/*.json"))

    if not data_files:
        raise FileNotFoundError(
            f"No JSON files found in {clone_dir}. Check repo structure."
        )

    # Merge all JSON files into one list
    merged: list[dict] = []
    for src in data_files:
        print(f"  Reading {src} ...")
        with open(src) as f:
            data = json.load(f)
        if isinstance(data, list):
            merged.extend(data)
        elif isinstance(data, dict):
            # Some datasets wrap the list in a key
            for v in data.values():
                if isinstance(v, list):
                    merged.extend(v)

    print(f"  Total safety examples: {len(merged)}")
    with open(out_path, "w") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"  Saved -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SFT datasets")
    parser.add_argument(
        "--output_dir", default="data/raw",
        help="Directory to save raw data files"
    )
    parser.add_argument(
        "--max_ultrachat", type=int, default=None,
        help="Limit UltraChat samples (useful for quick tests, e.g. 10000)"
    )
    parser.add_argument(
        "--skip_ultrachat", action="store_true",
        help="Skip downloading UltraChat"
    )
    parser.add_argument(
        "--skip_safety", action="store_true",
        help="Skip downloading SafetyTunedLlamas"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_ultrachat:
        download_ultrachat(output_dir, max_samples=args.max_ultrachat)
    if not args.skip_safety:
        download_safety_llamas(output_dir)

    print("\nDone. Files in", output_dir)
    for f in sorted(output_dir.iterdir()):
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name:40s}  {size_mb:8.1f} MB")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Convert UltraChat-200K + SafetyTunedLlamas to veRL Parquet SFT format.

Alpaca prompt template (matches assignment requirement):
  Below is an instruction that describes a task. Write a response that
  appropriately completes the request.

  ### Instruction:
  {prompt}

  ### Response:
  {response}

veRL fsdp_sft_trainer expects Parquet with columns: prompt, response
(data_source is kept for analysis but not required by veRL).

Outputs:
  data/verl_sft/train.parquet
  data/verl_sft/val.parquet
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd


ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _parse_ultrachat_messages(messages: list) -> tuple[str | None, str | None]:
    """Extract last user/assistant turn from messages list."""
    prompt_text = None
    response_text = None
    for i, msg in enumerate(messages):
        if msg.get("role") == "user":
            if i + 1 < len(messages) and messages[i + 1].get("role") == "assistant":
                prompt_text = msg["content"].strip()
                response_text = messages[i + 1]["content"].strip()
    return prompt_text, response_text


def load_ultrachat(path: Path, max_samples: int | None = None) -> list[dict]:
    """Load UltraChat from Parquet or JSONL. Extract last user/assistant turn."""
    records = []
    path = Path(path)

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
        for _, row in df.iterrows():
            messages = row.get("messages", [])
            if isinstance(messages, str):
                import json as _json
                messages = _json.loads(messages)
            prompt_text, response_text = _parse_ultrachat_messages(messages)
            if prompt_text and response_text:
                records.append({
                    "prompt": ALPACA_TEMPLATE.format(instruction=prompt_text),
                    "response": response_text,
                    "data_source": "ultrachat",
                })
            if max_samples and len(records) >= max_samples:
                break
    else:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                ex = json.loads(line)
                messages = ex.get("messages", [])
                prompt_text, response_text = _parse_ultrachat_messages(messages)
                if prompt_text and response_text:
                    records.append({
                        "prompt": ALPACA_TEMPLATE.format(instruction=prompt_text),
                        "response": response_text,
                        "data_source": "ultrachat",
                    })
                if max_samples and len(records) >= max_samples:
                    break
    return records


def load_safety_llamas(path: Path) -> list[dict]:
    """Load SafetyTunedLlamas JSON. Fields: instruction + output (or input+output)."""
    with open(path) as f:
        data = json.load(f)

    records = []
    for ex in data:
        # Handle different possible field names
        instruction = (
            ex.get("instruction")
            or ex.get("prompt")
            or ex.get("input")
            or ""
        ).strip()
        response = (
            ex.get("output")
            or ex.get("response")
            or ex.get("completion")
            or ""
        ).strip()
        if instruction and response:
            records.append({
                "prompt": ALPACA_TEMPLATE.format(instruction=instruction),
                "response": response,
                "data_source": "safety_llamas",
            })
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare veRL SFT Parquet data")
    parser.add_argument(
        "--ultrachat_train", nargs="+",
        default=[
            "data/raw/train_sft-00000-of-00003-a3ecf92756993583.parquet",
            "data/raw/train_sft-00001-of-00003-0a1804bcb6ae68c6.parquet",
            "data/raw/train_sft-00002-of-00003-ee46ed25cfae92c6.parquet",
        ],
        help="UltraChat train Parquet/JSONL file(s)"
    )
    parser.add_argument(
        "--ultrachat_val", default="data/raw/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet",
        help="UltraChat validation Parquet/JSONL"
    )
    parser.add_argument(
        "--safety_file", default="data/raw/safety_tuned_llamas.json",
        help="SafetyTunedLlamas JSON"
    )
    parser.add_argument(
        "--output_dir", default="data/verl_sft",
        help="Output directory for Parquet files"
    )
    parser.add_argument(
        "--max_ultrachat_train", type=int, default=None,
        help="Cap UltraChat train samples (e.g. 10000 for quick test)"
    )
    parser.add_argument(
        "--max_ultrachat_val", type=int, default=2000,
        help="Cap UltraChat val samples"
    )
    parser.add_argument(
        "--val_split_ratio", type=float, default=0.05,
        help="Fraction of safety data to use as val (if ultrachat_val not available)"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load training data ---
    train_records: list[dict] = []
    val_records: list[dict] = []

    uc_train_total = 0
    for train_file in args.ultrachat_train:
        p = Path(train_file)
        if not p.exists():
            print(f"[warn] {p} not found, skipping")
            continue
        remaining = None
        if args.max_ultrachat_train:
            remaining = max(0, args.max_ultrachat_train - uc_train_total)
            if remaining == 0:
                break
        print(f"Loading UltraChat train from {p} ...")
        uc_train = load_ultrachat(p, remaining)
        print(f"  {len(uc_train)} examples")
        train_records.extend(uc_train)
        uc_train_total += len(uc_train)
    print(f"  Total UltraChat train: {uc_train_total}")

    ultrachat_val_path = Path(args.ultrachat_val)
    if ultrachat_val_path.exists():
        print(f"Loading UltraChat val from {ultrachat_val_path} ...")
        uc_val = load_ultrachat(ultrachat_val_path, args.max_ultrachat_val)
        print(f"  {len(uc_val)} UltraChat val examples")
        val_records.extend(uc_val)
    else:
        print(f"[warn] {ultrachat_val_path} not found, will split safety data for val")

    safety_path = Path(args.safety_file)
    if safety_path.exists():
        print(f"Loading SafetyTunedLlamas from {safety_path} ...")
        safety = load_safety_llamas(safety_path)
        print(f"  {len(safety)} safety examples")
        random.shuffle(safety)

        if ultrachat_val_path.exists():
            # All safety data goes to train
            train_records.extend(safety)
        else:
            # Split safety data into train/val
            n_val = max(1, int(len(safety) * args.val_split_ratio))
            val_records.extend(safety[:n_val])
            train_records.extend(safety[n_val:])
    else:
        print(f"[warn] {safety_path} not found, skipping SafetyTunedLlamas")

    if not train_records:
        raise RuntimeError(
            "No training data loaded. Run download_sft_data.py first."
        )

    # --- Shuffle train ---
    random.shuffle(train_records)

    # Ensure val is non-empty (fallback: take 5% from train)
    if not val_records:
        print("[warn] No val data found, taking 5% from train for validation")
        n_val = max(100, int(len(train_records) * 0.05))
        val_records = train_records[:n_val]
        train_records = train_records[n_val:]

    # --- Save Parquet ---
    train_df = pd.DataFrame(train_records)
    val_df = pd.DataFrame(val_records)

    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    print(f"\nTrain: {len(train_df)} examples -> {train_path}")
    print(f"Val:   {len(val_df)} examples -> {val_path}")
    print("\nColumn dtypes:", dict(train_df.dtypes))

    # Show data source breakdown
    print("\nTrain data_source breakdown:")
    print(train_df["data_source"].value_counts().to_string())
    if "data_source" in val_df.columns:
        print("\nVal data_source breakdown:")
        print(val_df["data_source"].value_counts().to_string())

    # Show a sample
    sample = train_df.iloc[0]
    print("\n--- Sample train example ---")
    print(f"data_source: {sample['data_source']}")
    print(f"prompt (first 300 chars):\n{sample['prompt'][:300]}")
    print(f"response (first 200 chars):\n{sample['response'][:200]}")


if __name__ == "__main__":
    main()

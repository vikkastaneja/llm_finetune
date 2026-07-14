"""Debugging tool: print model generations side-by-side with input/reference,
plus the extracted fields metrics.py sees, for a saved checkpoint.

Use this whenever a quality metric's pass rate looks suspicious (especially
a flat 0% or 100% across an entire held-out set) BEFORE assuming the model
is bad -- it's frequently a metric-design bug instead. See
docs/solutions/debugging-eval-metrics.md for the full methodology and a
worked example (the threshold_faithfulness / confidence-field bug).

Usage:
    python scripts/inspect_generations.py --output_dir output/run1 -n 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from local_train.train import load_rows, split_train_held_out, load_saved_adapter, make_model_output_fn
from eval import metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_dir", required=True, help="Saved checkpoint to load (same as train.py's --output_dir).")
    p.add_argument("--base_model", default="unsloth/Qwen2.5-3B-Instruct-bnb-4bit")
    p.add_argument("--max_seq_length", type=int, default=512)
    p.add_argument("--data_path", default="bess_lora_train.jsonl")
    p.add_argument("--held_out_size", type=int, default=30)
    p.add_argument("-n", "--num_examples", type=int, default=4, help="How many held-out examples to inspect.")
    p.add_argument(
        "--metric",
        choices=["severity_accuracy", "threshold_faithfulness", "output_format", "action_alignment", "hallucination_check", "all"],
        default="all",
        help="Print only this metric's score for each example, or 'all' for the full breakdown.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    rows = load_rows(args.data_path)
    _, held_out = split_train_held_out(rows, args.held_out_size)
    instruction = rows[0]["instruction"]

    model, tokenizer = load_saved_adapter(args)
    gen_fn = make_model_output_fn(model, tokenizer, instruction)

    for row in held_out[: args.num_examples]:
        generated = gen_fn(row["input"])
        result = metrics.score(row["output"], generated, row["input"])

        print("=" * 80)
        print("INPUT:    ", row["input"])
        print("---")
        print("REFERENCE:", row["output"])
        print("---")
        print("GENERATED:", generated)
        print("---")
        print("input fields:    ", metrics._extract_fields(row["input"]))
        print("generated fields:", metrics._extract_fields(generated))
        print("---")
        if result.is_malformed:
            print("MALFORMED OUTPUT -- no per-metric scores")
            continue
        for s in result.scores:
            if args.metric != "all" and s.name != args.metric:
                continue
            mark = "PASS" if s.passed else "FAIL"
            print(f"[{mark}] {s.name}: {s.detail}")


if __name__ == "__main__":
    main()

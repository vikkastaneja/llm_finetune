"""Local QLoRA training entry point for the BESS reasoning engine (plan U3).

Runs one full training attempt: load base model, attach LoRA, train, grade
against a fixed held-out split using the shared evaluation module (U2), and
either export a recipe for AWS (U5) or report the shortfall so the human can
adjust hyperparameters and try again.

Meant to be re-invoked manually, many times, with different --base_model /
--lora_r / --epochs / --learning_rate values -- this script does not loop or
retry on its own (see plan Key Technical Decisions: the human orchestrates
iteration, this script orchestrates one attempt).

Usage:
    python src/local_train/train.py --base_model unsloth/Qwen2.5-3B-Instruct-bnb-4bit \\
        --lora_r 16 --epochs 4 --learning_rate 2e-4 \\
        --data_path bess_lora_train.jsonl --output_dir outputs/run1

Held-out split size/seed come from project_config.yaml by default -- pass
--config <path> to use a different config file, or --held_out_size/
--held_out_seed to override just those two values without touching the
config. See project_config.py.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make `src/` importable regardless of how this script is invoked (direct
# path, -m, different cwd) -- avoids requiring PYTHONPATH setup by hand.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

import project_config
from eval import metrics

INSTRUCTION_KEY = "instruction"
INPUT_KEY = "input"
OUTPUT_KEY = "output"

# held_out.size / held_out.seed come from project_config.yaml (shared with
# generate.py's severity_distribution) rather than being hardcoded here --
# the seed in particular MUST stay fixed across every run, or "the fixed
# held-out set" (plan R3) stops meaning the same set every time.


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base_model", default="unsloth/Qwen2.5-3B-Instruct-bnb-4bit")
    p.add_argument("--max_seq_length", type=int, default=512)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--data_path", default="bess_lora_train.jsonl")
    p.add_argument(
        "--held_out_size", type=int, default=None,
        help="Default: held_out.size from --config (project_config.yaml).",
    )
    p.add_argument(
        "--held_out_seed", type=int, default=None,
        help="Default: held_out.seed from --config (project_config.yaml).",
    )
    p.add_argument(
        "--config", default=None,
        help="Path to project_config.yaml (default: ./project_config.yaml, or built-in fallback if absent).",
    )
    p.add_argument("--output_dir", required=True)
    p.add_argument("--run_log_path", default="local_run_log.jsonl")
    p.add_argument("--recipe_path", default="recipe.yaml")
    p.add_argument("--promotion_config_path", default="promotion-config.yaml")
    p.add_argument(
        "--eval_only",
        action="store_true",
        help=(
            "Skip training entirely and load the already-saved adapter from "
            "--output_dir instead, then run scoring only. Useful for iterating "
            "on the scoring/heuristic logic without re-running the full "
            "training loop each time. Requires --output_dir to already contain "
            "a checkpoint from a prior (non-eval-only) run."
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading and formatting
# ---------------------------------------------------------------------------

def load_rows(data_path: str) -> list[dict]:
    rows = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_held_out_settings(args: argparse.Namespace, config: dict) -> tuple[int, int]:
    """CLI flags win when explicitly passed; otherwise fall back to
    project_config.yaml's held_out.size/seed."""
    size = args.held_out_size if args.held_out_size is not None else config["held_out"]["size"]
    seed = args.held_out_seed if args.held_out_seed is not None else config["held_out"]["seed"]
    return size, seed


def split_train_held_out(rows: list[dict], held_out_size: int, held_out_seed: int) -> tuple[list[dict], list[dict]]:
    """Deterministic split -- same held_out_size and seed always produces the
    same held-out rows, regardless of when/how many times this runs."""
    import random

    indices = list(range(len(rows)))
    random.Random(held_out_seed).shuffle(indices)
    held_out_idx = set(indices[:held_out_size])
    train_rows = [row for i, row in enumerate(rows) if i not in held_out_idx]
    held_out_rows = [row for i, row in enumerate(rows) if i in held_out_idx]
    return train_rows, held_out_rows


def build_training_text(tokenizer, row: dict) -> str:
    messages = [
        {"role": "user", "content": f"{row[INSTRUCTION_KEY]}\n\n{row[INPUT_KEY]}"},
        {"role": "assistant", "content": row[OUTPUT_KEY]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


# ---------------------------------------------------------------------------
# Model loading and training
# ---------------------------------------------------------------------------

def load_base_model(args: argparse.Namespace):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        dtype=None,  # auto-detect (bf16 on supported GPUs)
        load_in_4bit=True,
    )
    return model, tokenizer


def load_saved_adapter(args: argparse.Namespace):
    """--eval_only path: load an already-trained adapter straight from
    --output_dir instead of the base model + fresh LoRA attach. Unsloth's
    from_pretrained recognizes a saved adapter directory (adapter_config.json
    + adapter weights + tokenizer files, all written by model.save_pretrained
    in the normal training path) and loads base model + adapter together."""
    from unsloth import FastLanguageModel

    if not Path(args.output_dir).exists():
        raise FileNotFoundError(
            f"--eval_only requires an existing checkpoint at {args.output_dir!r} "
            "from a prior training run -- nothing found there."
        )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.output_dir,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    return model, tokenizer


def attach_lora(model, args: argparse.Namespace):
    from unsloth import FastLanguageModel

    return FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )


def build_trainer(model, tokenizer, train_rows: list[dict], args: argparse.Namespace):
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    texts = [build_training_text(tokenizer, row) for row in train_rows]
    dataset = Dataset.from_dict({"text": texts})

    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="no",  # we save the adapter explicitly after training
        report_to="none",
    )

    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )


# ---------------------------------------------------------------------------
# Inference for held-out scoring
# ---------------------------------------------------------------------------

def make_model_output_fn(model, tokenizer, instruction: str):
    """Returns a function usable as metrics.score_against's model_output_fn."""
    from unsloth import FastLanguageModel

    FastLanguageModel.for_inference(model)  # Unsloth's faster inference mode

    def _generate(input_text: str) -> str:
        messages = [{"role": "user", "content": f"{instruction}\n\n{input_text}"}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        # TODO: tune max_new_tokens/sampling once real generations are seen --
        # 209 was the max observed reference token length (see repo notes),
        # so 256 leaves headroom without being excessive.
        output_ids = model.generate(
            **inputs, max_new_tokens=256, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(generated, skip_special_tokens=True)

    return _generate


# ---------------------------------------------------------------------------
# Run log, recipe export, promotion-config seeding
# ---------------------------------------------------------------------------

def append_to_local_run_log(run_log_path: str, args: argparse.Namespace, eval_summary: dict) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "base_model": args.base_model,
        "lora_r": args.lora_r,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "output_dir": args.output_dir,
        "overall_pass_rate": eval_summary["overall_pass_rate"],
        "passes_threshold": eval_summary["passes_threshold"],
    }
    with open(run_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def read_run_log(run_log_path: str) -> list[dict]:
    if not Path(run_log_path).exists():
        return []
    with open(run_log_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def compute_variance(run_log: list[dict]) -> float:
    """Variance of overall_pass_rate across logged attempts. This seeds the
    promotion margin (see plan Key Technical Decisions) -- with fewer than 2
    logged runs, variance is undefined, so fall back to a conservative
    starting guess rather than crashing."""
    pass_rates = [entry["overall_pass_rate"] for entry in run_log]
    if len(pass_rates) < 2:
        return 0.05  # heuristic fallback until more runs accumulate
    return statistics.pvariance(pass_rates)


def export_recipe(args: argparse.Namespace) -> None:
    recipe = {
        "base_model": args.base_model,
        "max_seq_length": args.max_seq_length,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "checkpoint_path": args.output_dir,
        "prompt_template": "chat-default",
        # Filled in by U5 at AWS-launch time, once the dataset is uploaded.
        "s3_dataset_uri": None,
    }
    with open(args.recipe_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(recipe, f, sort_keys=False)


def write_promotion_config(promotion_config_path: str, margin: float, n_runs: int) -> None:
    config = {
        "margin": margin,
        "based_on_n_runs": n_runs,
        "note": (
            "Seed value from local held-out score variance across n_runs "
            "attempts. Human-editable starting point, not a final answer -- "
            "see plan Key Technical Decisions for why this needs recalibration "
            "once real promotion rounds run against the larger synthetic-replay "
            "pool (U4)."
        ),
    }
    with open(promotion_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    config = project_config.load_config(args.config)
    held_out_size, held_out_seed = resolve_held_out_settings(args, config)

    rows = load_rows(args.data_path)
    train_rows, held_out_rows = split_train_held_out(rows, held_out_size, held_out_seed)
    instruction = rows[0][INSTRUCTION_KEY]  # constant across the dataset

    if args.eval_only:
        model, tokenizer = load_saved_adapter(args)
    else:
        model, tokenizer = load_base_model(args)
        model = attach_lora(model, args)

        trainer = build_trainer(model, tokenizer, train_rows, args)
        trainer.train()

        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    model_output_fn = make_model_output_fn(model, tokenizer, instruction)
    print(f"Scoring {len(held_out_rows)} held-out examples...")
    eval_summary = metrics.score_against(held_out_rows, model_output_fn, verbose=True)

    print(f"Held-out overall pass rate: {eval_summary['overall_pass_rate']:.2%}")
    print(f"Per-metric pass rates: {eval_summary['per_metric_pass_rate']}")

    if args.eval_only:
        # Read-only inspection of an existing checkpoint -- not a new
        # training attempt, so it doesn't get logged to local_run_log.jsonl
        # (which would distort the variance calc compute_variance uses to
        # seed promotion-config.yaml) and doesn't re-export recipe/promotion
        # files.
        print("(--eval_only run: not logged, no recipe/promotion-config export)")
        return

    append_to_local_run_log(args.run_log_path, args, eval_summary)

    if eval_summary["passes_threshold"]:
        export_recipe(args)
        run_log = read_run_log(args.run_log_path)
        margin = compute_variance(run_log)
        write_promotion_config(args.promotion_config_path, margin, len(run_log))
        print(f"PASSED -- recipe exported to {args.recipe_path}")
        print(f"Promotion config seeded to {args.promotion_config_path} (margin={margin:.4f}, n_runs={len(run_log)})")
    else:
        print("NOT PASSED -- no recipe exported. Adjust hyperparameters and try again.")


if __name__ == "__main__":
    train(parse_args())

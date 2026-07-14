# BESS Fine-Tuned LLM Reasoning Engine

Fine-tunes a small LLM to diagnose battery energy storage system (BESS) anomalies
-- given a signal (SOH, confidence, ambient temperature, historical trend), it
produces a diagnosis and recommended action. Built locally on WSL2/Unsloth,
with a path to a single production training run on AWS SageMaker.

Full requirements and technical plan:
- [`docs/brainstorms/2026-07-12-llm-finetuning-local-to-aws-requirements.md`](docs/brainstorms/2026-07-12-llm-finetuning-local-to-aws-requirements.md) -- what's being built and why
- [`docs/plans/2026-07-12-001-feat-llm-reasoning-engine-aws-plan.md`](docs/plans/2026-07-12-001-feat-llm-reasoning-engine-aws-plan.md) -- architecture diagrams, implementation units, quality metrics reference

**Current status:** U1 (environment), U2 (evaluation metrics), U3 (local training script), and U4 (synthetic data generator) complete and tested. A real training run produces a checkpoint that passes all five quality metrics on the held-out set and exports `recipe.yaml` + `promotion-config.yaml`. See the plan doc's Implementation Units for U5 onward (AWS).

---

## Prerequisites

- WSL2 with an NVIDIA GPU passthrough (developed against an RTX 4050, 6GB VRAM)
- Python 3.10
- This repo checked out on the **native WSL filesystem** (e.g. `~/llm_finetune`), not under `/mnt/c/...` -- package installs are dramatically slower on the Windows-mounted filesystem

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Only needed once you're ready to touch AWS (U5 onward):
pip install -r requirements-aws.txt
```

Verify the GPU is visible:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Repo layout

```text
bess_lora_train.jsonl      # seed training dataset (300 examples)
project_config.yaml        # shared config: held-out split size/seed, severity distribution,
                            # default generation record count/seed -- single source of truth for train.py + generate.py
requirements.txt           # local training deps (unsloth, pytest, pyyaml)
requirements-aws.txt       # AWS-side deps (boto3, sagemaker) -- installed separately
pyproject.toml             # pytest config (lets test/ import from src/ without path hacks)
src/
  project_config.py       # loads project_config.yaml, with fallback defaults if it's missing
  eval/
    metrics.py             # shared 5-metric quality scorer (U2) -- see plan's Quality Metrics Reference
  local_train/
    train.py               # local QLoRA training entry point (U3)
  datagen/
    generate.py            # synthetic BESS case generator (U4) -- grows the replay pool for U8/U10
test/
  test_project_config.py  # tests for the config loader
  eval/
    test_metrics.py        # tests for the metrics module
  datagen/
    test_generate.py       # tests for the generator
scripts/
  inspect_generations.py   # debugging tool -- see docs/solutions/debugging-eval-metrics.md
.vscode/
  settings.json            # points VS Code at the venv, enables pytest discovery
  launch.json               # debug configs (current file, all tests, current test file, test at cursor)
```

## Running the tests

```bash
source venv/bin/activate
pytest test/ -v
```

Or from VS Code: open the Testing sidebar (flask icon), or use the "Debug Test" link that appears above each `def test_...` function once the Python + Python Debugger extensions are installed. See `.vscode/launch.json` for manual debug configs, including "Debug Test at Cursor" (select the test function name first, then run it).

## Running a local training attempt

```bash
source venv/bin/activate
python src/local_train/train.py \
    --base_model unsloth/Qwen2.5-3B-Instruct-bnb-4bit \
    --lora_r 16 \
    --epochs 4 \
    --learning_rate 2e-4 \
    --data_path bess_lora_train.jsonl \
    --output_dir outputs/run1
```

This trains one adapter, grades it against a fixed held-out split of the dataset using the five quality metrics (see the plan's **Quality Metrics Reference** section for what each one checks), and:

- **If it passes:** writes `recipe.yaml` (the settings that produced this result, for later AWS reproduction) and `promotion-config.yaml` (a seeded promotion-margin config used much later, by U9).
- **If it doesn't pass:** just prints the score breakdown -- adjust `--lora_r` / `--epochs` / `--learning_rate` / `--base_model` and run again. This script is meant to be re-run many times; each attempt is logged to `local_run_log.jsonl` regardless of pass/fail.

Note: the two LLM-as-judge metrics (action-recommendation alignment, hallucination check) currently run on a **cheap local keyword heuristic** (`src/eval/metrics.py`, `USE_HEURISTIC_JUDGE = True`) rather than a real judge model -- no AWS/Bedrock dependency needed for local iteration. This is deliberately approximate (real logic, tested, but cruder than an LLM judge) and must be swapped for a real pinned Bedrock call before U8/U9/U10 use this module for actual promotion or regression decisions on AWS.

### Re-scoring an existing checkpoint without retraining

Every plain invocation above retrains from scratch (the slow part). To just re-run scoring against an adapter you already trained -- useful when you're iterating on the metrics/heuristic logic rather than on hyperparameters -- add `--eval_only` and point `--output_dir` at an existing checkpoint:

```bash
python src/local_train/train.py \
    --base_model unsloth/Qwen2.5-3B-Instruct-bnb-4bit \
    --data_path bess_lora_train.jsonl \
    --output_dir outputs/run1 \
    --eval_only
```

This loads the saved adapter directly and skips straight to grading. It does **not** write to `local_run_log.jsonl`, `recipe.yaml`, or `promotion-config.yaml` -- it's a read-only inspection of an existing checkpoint, not a new training attempt.

### If a metric's pass rate looks wrong

If a per-metric pass rate looks suspicious (especially a flat 0% or 100% across the whole held-out set), don't assume the model is bad at that thing -- go look at what it actually generated:

```bash
python scripts/inspect_generations.py --output_dir outputs/run1 -n 4
```

Prints input/reference/generated side by side for a few held-out examples, plus the exact fields `metrics.py` extracted from each. See [`docs/solutions/debugging-eval-metrics.md`](docs/solutions/debugging-eval-metrics.md) for the methodology and a worked example (a real bug this caught: `threshold_faithfulness` was failing 100% of the time due to a confidence-field check that no real answer, reference or generated, was ever going to satisfy).

## Shared config (`project_config.yaml`)

`train.py` and `generate.py` both read settings from `project_config.yaml` so they can't silently drift out of sync with each other:

```yaml
held_out:
  size: 30
  seed: 42                # fixed -- must stay identical across every run
severity_distribution:
  Healthy: 0.25            # matches bess_lora_train.jsonl's real distribution
  Warning: 0.40
  Damaged: 0.35
total_records:
  n: 500                   # default record count for generate.py
  seed: 100
```

Both scripts accept `--config <path>` to point at a different config file; any explicit CLI flag (e.g. `--held_out_size`, `-n`, `--seed`) still overrides the config for that one value. If no config file exists at all, both scripts fall back to the same defaults shown above (see `src/project_config.py`).

## Generating synthetic data (U4)

The fixed 300-example dataset isn't enough on its own for a trustworthy A/B comparison between adapter versions (U8/U9) -- replaying the same fixed set repeatedly gives a deterministic, non-growing sample. `src/datagen/generate.py` produces additional synthetic cases in the exact same schema and phrasing style as `bess_lora_train.jsonl` (node types, per-type SOH thresholds, risk notes, etc. were extracted empirically from the real dataset, not guessed):

```bash
python src/datagen/generate.py --output_path generated_cases.jsonl
# record count and seed come from project_config.yaml's total_records section.
# To generate a different count/seed, edit that file, or point at a different one:
python src/datagen/generate.py --config my_other_config.yaml --output_path generated_cases.jsonl
```

Same seed always produces the same output (reproducible test fixtures); different seeds produce genuinely different cases. Programmatically:

```python
from datagen.generate import generate_dataset

rows = generate_dataset(1000, seed=42)  # severity mix from project_config.yaml
# or override the distribution explicitly:
rows = generate_dataset(1000, severity_distribution={"Healthy": 0.5, "Warning": 0.3, "Damaged": 0.2}, seed=42)
```

Generated rows score correctly against `metrics.py` when checked against themselves (severity, thresholds, and format all self-consistent) -- confirmed as part of building this unit.

## Notes on the environment

- `requirements.txt` intentionally does **not** hand-pin `transformers`/`trl`/`peft` versions -- Unsloth's own installer resolves a compatible set. `requirements-aws.txt` **does** pin `sagemaker>=3,<4` deliberately, since an unpinned install could silently land on the deprecated V2 SDK.
- If `pip install` or any package operation seems to hang for minutes, check that you're on the native WSL filesystem (`readlink -f venv` should NOT show a `/mnt/c/...` path) -- see the plan's Risks section and prior session notes for why this matters.

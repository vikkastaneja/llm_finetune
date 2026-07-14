"""Tests for the synthetic BESS case generator (plan U4)."""

import json
import re

from datagen import generate

VALID_KEYS = {"instruction", "input", "output"}
SOH_RE = re.compile(r"SOH:?\s*([0-9]+\.?[0-9]*)")
CONFIDENCE_RE = re.compile(r"Confidence:\s*([0-9]+\.?[0-9]*)")
SEVERITY_LABEL_RE = re.compile(r"flagged as (Healthy|Warning|Damaged) \(label=(\d)\)")


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def test_generated_row_matches_real_dataset_schema():
    rows = generate.generate_dataset(5, seed=1)
    for row in rows:
        assert set(row.keys()) == VALID_KEYS
        assert row["instruction"] == generate.INSTRUCTION
        assert isinstance(row["input"], str) and row["input"]
        assert isinstance(row["output"], str) and row["output"]


def test_generated_row_input_contains_expected_structured_fields():
    rows = generate.generate_dataset(10, seed=2)
    for row in rows:
        assert SOH_RE.search(row["input"]) is not None
        assert CONFIDENCE_RE.search(row["input"]) is not None
        assert "Ambient temp:" in row["input"]
        assert "Plant:" in row["input"]


def test_severity_label_matches_numeric_label_convention():
    # label=0 Healthy, label=1 Warning, label=2 Damaged -- matches real dataset.
    expected = {"Healthy": "0", "Warning": "1", "Damaged": "2"}
    rows = generate.generate_dataset(20, seed=3)
    for row in rows:
        m = SEVERITY_LABEL_RE.search(row["input"])
        assert m is not None, row["input"]
        severity, label = m.group(1), m.group(2)
        assert expected[severity] == label


# ---------------------------------------------------------------------------
# Severity distribution
# ---------------------------------------------------------------------------

def test_default_distribution_comes_from_config_when_none_passed(tmp_path):
    # No severity_distribution passed -- should read from the config file
    # (config_path), not some other hardcoded default. Uses an isolated temp
    # config rather than the live project_config.yaml, since that file is
    # user-editable operational state (e.g. for one-off experiments) and
    # shouldn't need to hold specific values for this test to pass.
    custom_path = tmp_path / "custom.yaml"
    custom_path.write_text(
        "severity_distribution:\n  Healthy: 0.8\n  Warning: 0.15\n  Damaged: 0.05\n"
    )

    rows = generate.generate_dataset(2000, seed=8, config_path=str(custom_path))
    counts = {"Healthy": 0, "Warning": 0, "Damaged": 0}
    for row in rows:
        m = SEVERITY_LABEL_RE.search(row["input"])
        counts[m.group(1)] += 1

    total = len(rows)
    assert abs(counts["Healthy"] / total - 0.8) < 0.05
    assert abs(counts["Warning"] / total - 0.15) < 0.05
    assert abs(counts["Damaged"] / total - 0.05) < 0.03


def test_severity_distribution_matches_request_within_tolerance():
    rows = generate.generate_dataset(
        2000, severity_distribution={"Healthy": 0.8, "Warning": 0.15, "Damaged": 0.05}, seed=4
    )
    counts = {"Healthy": 0, "Warning": 0, "Damaged": 0}
    for row in rows:
        m = SEVERITY_LABEL_RE.search(row["input"])
        counts[m.group(1)] += 1

    total = len(rows)
    assert abs(counts["Healthy"] / total - 0.8) < 0.05
    assert abs(counts["Warning"] / total - 0.15) < 0.05
    assert abs(counts["Damaged"] / total - 0.05) < 0.03


# ---------------------------------------------------------------------------
# Realistic bounds
# ---------------------------------------------------------------------------

def test_generated_values_stay_within_realistic_bounds():
    rows = generate.generate_dataset(200, seed=5)
    for row in rows:
        soh = float(SOH_RE.search(row["input"]).group(1))
        confidence = float(CONFIDENCE_RE.search(row["input"]).group(1))
        assert soh > 0
        assert 0.0 <= confidence <= 1.0


# ---------------------------------------------------------------------------
# Reproducibility / variance
# ---------------------------------------------------------------------------

def test_same_seed_produces_identical_output():
    rows_a = generate.generate_dataset(15, seed=42)
    rows_b = generate.generate_dataset(15, seed=42)
    assert rows_a == rows_b


def test_different_seeds_produce_different_cases():
    rows_a = generate.generate_dataset(15, seed=1)
    rows_b = generate.generate_dataset(15, seed=2)
    assert rows_a != rows_b


def test_large_batch_shows_more_variance_than_repeating_original_300():
    # Repeating the original 300-example set N times gives exactly 300
    # unique inputs no matter how large N is. A generated batch should show
    # meaningfully more unique node/value combinations than that ceiling.
    rows = generate.generate_dataset(1000, seed=6)
    unique_inputs = {row["input"] for row in rows}
    assert len(unique_inputs) > 300


# ---------------------------------------------------------------------------
# write_jsonl
# ---------------------------------------------------------------------------

def test_write_jsonl_round_trips(tmp_path):
    rows = generate.generate_dataset(5, seed=7)
    out_path = tmp_path / "generated.jsonl"
    generate.write_jsonl(rows, str(out_path))

    with open(out_path, encoding="utf-8") as f:
        loaded = [json.loads(line) for line in f]
    assert loaded == rows


def test_generate_one_rejects_unknown_severity():
    import random

    try:
        generate.generate_one("Critical", random.Random(0))
        assert False, "expected ValueError"
    except ValueError:
        pass

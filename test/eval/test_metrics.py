"""Tests for the shared evaluation module (plan U2).

Deterministic checks (severity, threshold faithfulness, format) are tested
directly -- no mocking needed. Judged checks (action alignment, hallucination)
go through metrics._call_judge, so tests monkeypatch that single seam instead
of a real model API.
"""

from eval import metrics


# ---------------------------------------------------------------------------
# Fixtures: example input/reference/output text, matching the dataset's style
# ---------------------------------------------------------------------------

GOOD_INPUT = (
    "Node rack_3_2_mod1 flagged as Damaged (label=2). SOH: 0.60. Confidence: 0.88. "
    "Flagged by: consensus (HGT + XGBoost agree). Ambient temp: 46.5°C. Plant: plant1."
)
GOOD_REFERENCE = (
    "High-severity anomaly confirmed. Storage module rack_3_2_mod1 shows critical SOH "
    "degradation at 0.60. Elevated ambient temperature (46.5°C) is accelerating degradation. "
    "Recommended action: immediate inspection and planned replacement within 24 hours."
)
GOOD_MODEL_OUTPUT = (
    "High-severity anomaly confirmed -- module rack_3_2_mod1 is classified as Damaged. "
    "SOH at 0.60 with confidence 0.88. Ambient temperature 46.5°C is elevated. "
    "Recommended action: schedule inspection and prepare replacement."
)


def _matching_judge(task: str, **kwargs) -> tuple[bool, str]:
    return True, f"{task}: looks fine"


def _judge_flagging(task_to_flag: str):
    """Returns a fake judge that fails only the given task, passes everything else."""
    def _judge(task: str, **kwargs) -> tuple[bool, str]:
        if task == task_to_flag:
            return False, f"{task}: flagged"
        return True, f"{task}: ok"
    return _judge


# ---------------------------------------------------------------------------
# Deterministic checks -- no mocking needed
# ---------------------------------------------------------------------------

def test_severity_accuracy_matches():
    result = metrics.check_severity_accuracy(GOOD_INPUT, GOOD_MODEL_OUTPUT)
    assert result.passed is True


def test_severity_accuracy_mismatch():
    model_output = "This unit is Healthy and needs no action."
    result = metrics.check_severity_accuracy(GOOD_INPUT, model_output)
    assert result.passed is False


def test_threshold_faithfulness_matches_input_numbers():
    result = metrics.check_threshold_faithfulness(GOOD_INPUT, GOOD_MODEL_OUTPUT)
    assert result.passed is True


def test_threshold_faithfulness_fails_on_wrong_number():
    # Model cites a different SOH and temperature than the input case.
    model_output = "SOH: 0.85, confidence 0.88. Ambient temperature 22.0°C. Monitor closely."
    result = metrics.check_threshold_faithfulness(GOOD_INPUT, model_output)
    assert result.passed is False
    assert "input=" in result.detail


def test_output_format_passes_well_formed_answer():
    result = metrics.check_output_format(GOOD_MODEL_OUTPUT)
    assert result.passed is True


def test_output_format_fails_on_truncated_output():
    truncated = "High-severity anomaly. rack_3_2_mod1 shows SOH at 0.60 with confidence"
    result = metrics.check_output_format(truncated)
    assert result.passed is False


def test_output_format_fails_on_missing_recommendation():
    no_action = "High-severity anomaly. rack_3_2_mod1 shows SOH at 0.60."
    result = metrics.check_output_format(no_action)
    assert result.passed is False


def test_output_format_fails_on_empty_output():
    result = metrics.check_output_format("")
    assert result.passed is False
    assert "empty" in result.detail


# ---------------------------------------------------------------------------
# Judged checks -- mock the single _call_judge seam
# ---------------------------------------------------------------------------

def test_action_alignment_uses_judge(monkeypatch):
    monkeypatch.setattr(metrics, "_call_judge", _matching_judge)
    result = metrics.check_action_alignment(GOOD_INPUT, GOOD_REFERENCE, GOOD_MODEL_OUTPUT)
    assert result.passed is True


def test_hallucination_check_flags_fabricated_history(monkeypatch):
    monkeypatch.setattr(metrics, "_call_judge", _judge_flagging("hallucination"))
    model_output = (
        GOOD_MODEL_OUTPUT
        + " Similar past case rack_9_9_mod9 was replaced last quarter under identical conditions."
    )
    result = metrics.check_hallucination(GOOD_INPUT, model_output)
    assert result.passed is False


# ---------------------------------------------------------------------------
# score() -- the per-example entry point
# ---------------------------------------------------------------------------

def test_score_happy_path_all_metrics_pass(monkeypatch):
    monkeypatch.setattr(metrics, "_call_judge", _matching_judge)
    result = metrics.score(GOOD_REFERENCE, GOOD_MODEL_OUTPUT, GOOD_INPUT)
    assert result.is_malformed is False
    assert result.passes_threshold is True
    assert all(s.passed for s in result.scores)


def test_score_malformed_output_returns_structured_failure():
    # No judge mock needed -- score() should short-circuit before calling it.
    result = metrics.score(GOOD_REFERENCE, "", GOOD_INPUT)
    assert result.is_malformed is True
    assert result.passes_threshold is False
    assert result.scores == []


def test_score_malformed_output_does_not_raise():
    # Explicitly confirm this never raises, even without mocking the judge.
    result = metrics.score(GOOD_REFERENCE, None, GOOD_INPUT)
    assert result.is_malformed is True


def test_score_wrong_numbers_fails_faithfulness_specifically(monkeypatch):
    monkeypatch.setattr(metrics, "_call_judge", _matching_judge)
    model_output = "SOH: 0.99, confidence 0.88. Ambient temperature 10.0°C. Monitor closely."
    result = metrics.score(GOOD_REFERENCE, model_output, GOOD_INPUT)
    faithfulness = next(s for s in result.scores if s.name == "threshold_faithfulness")
    assert faithfulness.passed is False


# ---------------------------------------------------------------------------
# score_against() -- aggregate over a held-out set
# ---------------------------------------------------------------------------

def test_score_against_aggregates_pass_rate(monkeypatch):
    monkeypatch.setattr(metrics, "_call_judge", _matching_judge)
    held_out_set = [
        {"input": GOOD_INPUT, "output": GOOD_REFERENCE},
        {"input": GOOD_INPUT, "output": GOOD_REFERENCE},
    ]

    def model_output_fn(_input_text: str) -> str:
        return GOOD_MODEL_OUTPUT

    result = metrics.score_against(held_out_set, model_output_fn)
    assert result["n"] == 2
    assert result["n_malformed"] == 0
    assert result["overall_pass_rate"] == 1.0
    assert result["passes_threshold"] is True
    assert set(result["per_metric_pass_rate"].keys()) == {
        "severity_accuracy",
        "threshold_faithfulness",
        "output_format",
        "action_alignment",
        "hallucination_check",
    }


def test_score_against_handles_malformed_outputs_in_batch(monkeypatch):
    monkeypatch.setattr(metrics, "_call_judge", _matching_judge)
    held_out_set = [
        {"input": GOOD_INPUT, "output": GOOD_REFERENCE},
        {"input": GOOD_INPUT, "output": GOOD_REFERENCE},
    ]
    outputs = iter([GOOD_MODEL_OUTPUT, ""])  # second one is malformed

    def model_output_fn(_input_text: str) -> str:
        return next(outputs)

    result = metrics.score_against(held_out_set, model_output_fn)
    assert result["n"] == 2
    assert result["n_malformed"] == 1


def test_score_against_empty_held_out_set_does_not_divide_by_zero():
    result = metrics.score_against([], lambda _: "")
    assert result["n"] == 0
    assert result["overall_pass_rate"] == 0.0
    assert result["passes_threshold"] is False

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


# Real Damaged-tier phrasing from the dataset -- deliberately does NOT
# contain the literal word "Damaged" (it says "High-severity anomaly
# confirmed" instead) and uses "SOH degradation at X" (a wider gap than
# "SOH at X"). GOOD_MODEL_OUTPUT above happens to contain the literal word
# "Damaged", which is why the bug this fixture regression-tests slipped past
# the original tests -- see docs/solutions/debugging-eval-metrics.md.
DAMAGED_STYLE_INPUT = (
    "Node hvac_3 flagged as Damaged (label=2). SOH: 0.64. Confidence: 0.98. "
    "Flagged by: consensus (HGT + XGBoost agree). Ambient temp: 35.6°C. Plant: plant1."
)
DAMAGED_STYLE_OUTPUT = (
    "High-severity anomaly confirmed. HVAC unit hvac_3 shows critical SOH degradation "
    "at 0.64, which is 0.03 below the replacement threshold of 0.67. Elevated ambient "
    "temperature (35.6°C) is accelerating degradation. Recommended action: immediate "
    "inspection and planned replacement within 24 hours."
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


def test_severity_extraction_recognizes_high_severity_as_damaged_alias():
    # Regression test: Damaged-tier text never literally says "Damaged" in
    # this dataset -- it says "High-severity anomaly confirmed" instead.
    assert metrics._extract_fields(DAMAGED_STYLE_OUTPUT)["severity"] == "Damaged"


def test_severity_accuracy_passes_for_high_severity_phrasing():
    result = metrics.check_severity_accuracy(DAMAGED_STYLE_INPUT, DAMAGED_STYLE_OUTPUT)
    assert result.passed is True


def test_output_format_recognizes_high_severity_phrasing():
    result = metrics.check_output_format(DAMAGED_STYLE_OUTPUT)
    assert result.passed is True


def test_soh_extraction_handles_wider_gap_phrasing():
    # Regression test: "SOH degradation at X" has a wider gap between "SOH"
    # and the number than "SOH at X" -- the original 15-char window missed it.
    assert metrics._extract_fields(DAMAGED_STYLE_OUTPUT)["soh"] == 0.64


def test_threshold_faithfulness_passes_for_damaged_style_phrasing():
    result = metrics.check_threshold_faithfulness(DAMAGED_STYLE_INPUT, DAMAGED_STYLE_OUTPUT)
    assert result.passed is True


def test_threshold_faithfulness_ignores_missing_confidence():
    # Real model generations (and the dataset's own reference answers) never
    # restate confidence in prose -- this must NOT count as unfaithful as
    # long as SOH and ambient_temp match. Regression test for the bug found
    # via docs/solutions/debugging-eval-metrics.md.
    model_output = "SOH at 0.60. Ambient temperature 46.5°C is elevated. Recommended action: monitor."
    result = metrics.check_threshold_faithfulness(GOOD_INPUT, model_output)
    assert result.passed is True


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
# Interim heuristic judge (USE_HEURISTIC_JUDGE = True) -- real logic, not a
# stub, so it gets tested directly rather than only via mocking.
# ---------------------------------------------------------------------------

def test_heuristic_action_alignment_passes_for_matching_severity():
    passed, detail = metrics._heuristic_action_alignment(GOOD_INPUT, GOOD_MODEL_OUTPUT)
    assert passed is True
    assert "Damaged" in detail


def test_heuristic_action_alignment_fails_when_action_too_weak_for_severity():
    # Damaged case, but the recommendation only says to monitor -- no
    # Damaged-tier keyword present.
    model_output = "Module rack_3_2_mod1 is classified as Damaged. Continue to monitor closely."
    passed, _detail = metrics._heuristic_action_alignment(GOOD_INPUT, model_output)
    assert passed is False


def test_heuristic_hallucination_passes_when_no_unrecognized_ids():
    passed, _detail = metrics._heuristic_hallucination(GOOD_INPUT, GOOD_MODEL_OUTPUT)
    assert passed is True


def test_heuristic_hallucination_flags_fabricated_node_id():
    model_output = (
        GOOD_MODEL_OUTPUT
        + " Similar past case rack_9_9_mod9 was replaced last quarter under identical conditions."
    )
    passed, detail = metrics._heuristic_hallucination(GOOD_INPUT, model_output)
    assert passed is False
    assert "rack_9_9_mod9" in detail


def test_call_judge_dispatches_to_heuristics_by_default():
    # No monkeypatch -- exercises the real USE_HEURISTIC_JUDGE=True path.
    verdict, _detail = metrics._call_judge(
        task="action_alignment",
        input_context=GOOD_INPUT,
        reference_output=GOOD_REFERENCE,
        model_output=GOOD_MODEL_OUTPUT,
    )
    assert verdict is True


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

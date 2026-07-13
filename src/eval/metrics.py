"""Shared evaluation module for the BESS reasoning engine (plan U2).

Scores a model's diagnosis/recommendation output against a reference on five
metrics: severity accuracy, action alignment, threshold/number faithfulness,
hallucination check, and output-format reliability.

Deterministic checks (severity, threshold faithfulness, format) use regex
field extraction and need no model call. Judged checks (action alignment,
hallucination) call a pinned LLM-as-judge model (see JUDGE_MODEL_ID below) --
temperature 0 for stability, per the plan's Key Technical Decisions.

Used by: local training (U3), the A/B comparison harness (U8), and the
monitoring Processing Job (U10) -- one scoring function, three callers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# Pin the judge model explicitly -- do not float to a provider's "latest"
# alias. A version bump here is a deliberate re-baselining event; see the
# plan's Key Technical Decisions for why this matters for U9/U10.
JUDGE_MODEL_ID = "TODO: set pinned Bedrock model id, e.g. anthropic.claude-*-YYYYMMDD-v1:0"
JUDGE_TEMPERATURE = 0.0

PASS_THRESHOLD = 0.8  # fraction of the five metrics that must pass


@dataclass
class MetricScore:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class EvalResult:
    scores: list[MetricScore] = field(default_factory=list)
    is_malformed: bool = False

    @property
    def passes_threshold(self) -> bool:
        if self.is_malformed:
            return False
        if not self.scores:
            return False
        pass_rate = sum(1 for s in self.scores if s.passed) / len(self.scores)
        return pass_rate >= PASS_THRESHOLD

    def as_dict(self) -> dict:
        return {
            "passes_threshold": self.passes_threshold,
            "is_malformed": self.is_malformed,
            "scores": {s.name: {"passed": s.passed, "detail": s.detail} for s in self.scores},
        }


# ---------------------------------------------------------------------------
# Deterministic field extraction
# ---------------------------------------------------------------------------

_SOH_RE = re.compile(r"SOH\D{0,15}?([0-9]+\.?[0-9]*)", re.IGNORECASE)
_CONFIDENCE_RE = re.compile(r"[Cc]onfidence[:\s]*([0-9]*\.?[0-9]+)")
_TEMP_RE = re.compile(r"([0-9]*\.?[0-9]+)\s*(?:°C|degrees?\s*C)", re.IGNORECASE)
_SEVERITY_RE = re.compile(r"\b(Healthy|Warning|Damaged)\b", re.IGNORECASE)


def _extract_fields(text: str) -> dict:
    """Pull structured fields out of prose text. Returns None for anything not found."""
    def _num(pattern: re.Pattern) -> float | None:
        m = pattern.search(text)
        return float(m.group(1)) if m else None

    sev_match = _SEVERITY_RE.search(text)
    return {
        "soh": _num(_SOH_RE),
        "confidence": _num(_CONFIDENCE_RE),
        "ambient_temp": _num(_TEMP_RE),
        "severity": sev_match.group(1).capitalize() if sev_match else None,
    }


def _numbers_match(a: float | None, b: float | None, tol: float = 0.01) -> bool:
    if a is None or b is None:
        # If neither reference nor model output mentions the field, that's not
        # a faithfulness violation -- only flag when they actively disagree.
        return a is None and b is None
    return abs(a - b) <= tol


# ---------------------------------------------------------------------------
# Individual metric checks
# ---------------------------------------------------------------------------

def check_severity_accuracy(input_context: str, model_output: str) -> MetricScore:
    """Model's stated severity must match the INPUT case's flagged severity --
    checked against the input's ground-truth label, not the reference output's
    prose, since reference phrasing doesn't always restate the severity word
    literally (mirrors threshold_faithfulness's design)."""
    in_fields = _extract_fields(input_context)
    out_fields = _extract_fields(model_output)
    in_sev, out_sev = in_fields["severity"], out_fields["severity"]
    passed = in_sev is not None and in_sev == out_sev
    detail = f"input={in_sev!r} model={out_sev!r}"
    return MetricScore("severity_accuracy", passed, detail)


def check_threshold_faithfulness(input_context: str, model_output: str) -> MetricScore:
    """Model's cited numbers must match the INPUT case, not the reference answer --
    faithfulness is about not inventing numbers, not about matching phrasing."""
    in_fields = _extract_fields(input_context)
    out_fields = _extract_fields(model_output)
    checks = [
        _numbers_match(in_fields["soh"], out_fields["soh"]),
        _numbers_match(in_fields["confidence"], out_fields["confidence"]),
        _numbers_match(in_fields["ambient_temp"], out_fields["ambient_temp"]),
    ]
    passed = all(checks)
    detail = f"input={in_fields} model={out_fields}"
    return MetricScore("threshold_faithfulness", passed, detail)


def check_output_format(model_output: str) -> MetricScore:
    """A well-formed answer has both a diagnosis/severity statement and a
    recommended action, and isn't truncated mid-sentence."""
    if not model_output or not model_output.strip():
        return MetricScore("output_format", False, "empty output")

    has_severity = _SEVERITY_RE.search(model_output) is not None
    has_action_keyword = bool(
        re.search(r"\b(recommend|action|inspect|replace|monitor|schedule)\b", model_output, re.IGNORECASE)
    )
    ends_cleanly = model_output.strip()[-1] in ".!?\""

    passed = has_severity and has_action_keyword and ends_cleanly
    detail = f"has_severity={has_severity} has_action={has_action_keyword} ends_cleanly={ends_cleanly}"
    return MetricScore("output_format", passed, detail)


def check_action_alignment(input_context: str, reference_output: str, model_output: str) -> MetricScore:
    """LLM-as-judge: does the recommended action fit the severity/situation?"""
    # TODO: replace with a real call to JUDGE_MODEL_ID at JUDGE_TEMPERATURE.
    # Prompt should give the judge: input_context, reference_output (as an
    # example of a good answer), and model_output, then ask for a pass/fail
    # verdict + one-line rationale. Keep the judge call in its own function
    # (see _call_judge below) so it's a single seam to swap/mock in tests.
    verdict, rationale = _call_judge(
        task="action_alignment",
        input_context=input_context,
        reference_output=reference_output,
        model_output=model_output,
    )
    return MetricScore("action_alignment", verdict, rationale)


def check_hallucination(input_context: str, model_output: str) -> MetricScore:
    """LLM-as-judge: does the model output introduce facts not present in the input?"""
    # TODO: replace with a real call to JUDGE_MODEL_ID at JUDGE_TEMPERATURE.
    verdict, rationale = _call_judge(
        task="hallucination",
        input_context=input_context,
        reference_output="",
        model_output=model_output,
    )
    return MetricScore("hallucination_check", verdict, rationale)


def _call_judge(task: str, input_context: str, reference_output: str, model_output: str) -> tuple[bool, str]:
    """Single seam for all LLM-as-judge calls. Swap/mock this in tests instead
    of mocking a bare API client, so test_metrics.py doesn't need to know
    which provider backs the judge."""
    raise NotImplementedError(
        f"TODO: call {JUDGE_MODEL_ID} (temperature={JUDGE_TEMPERATURE}) for task={task!r}"
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def score(reference_output: str, model_output: str, input_context: str) -> EvalResult:
    """Score a single (reference, model_output, input) triple against all five metrics."""
    if model_output is None or not model_output.strip():
        return EvalResult(scores=[], is_malformed=True)

    scores = [
        check_severity_accuracy(input_context, model_output),
        check_threshold_faithfulness(input_context, model_output),
        check_output_format(model_output),
        check_action_alignment(input_context, reference_output, model_output),
        check_hallucination(input_context, model_output),
    ]
    return EvalResult(scores=scores)


def score_against(held_out_set: list[dict], model_output_fn) -> dict:
    """Score a model against a full held-out set. `held_out_set` is a list of
    {"input": ..., "output": ...} rows (same shape as bess_lora_train.jsonl).
    `model_output_fn(input_text) -> str` generates the model's answer for a
    given input -- caller supplies this so this module stays model-agnostic.

    Returns an aggregate: overall pass rate plus per-metric pass rates, which
    is what U3's threshold gate and U9/U10's comparisons consume.
    """
    results = [
        score(reference_output=row["output"], model_output=model_output_fn(row["input"]), input_context=row["input"])
        for row in held_out_set
    ]

    non_malformed = [r for r in results if not r.is_malformed]
    overall_pass_rate = sum(1 for r in results if r.passes_threshold) / len(results) if results else 0.0

    per_metric: dict[str, float] = {}
    if non_malformed:
        metric_names = [s.name for s in non_malformed[0].scores]
        for name in metric_names:
            passes = [s.passed for r in non_malformed for s in r.scores if s.name == name]
            per_metric[name] = sum(passes) / len(passes) if passes else 0.0

    return {
        "n": len(results),
        "n_malformed": len(results) - len(non_malformed),
        "overall_pass_rate": overall_pass_rate,
        "passes_threshold": overall_pass_rate >= PASS_THRESHOLD,
        "per_metric_pass_rate": per_metric,
    }

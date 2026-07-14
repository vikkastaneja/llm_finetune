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

# INTERIM: no AWS/Bedrock dependency wired up yet, so action_alignment and
# hallucination_check run on a cheap local keyword heuristic instead of a
# real LLM judge (see _heuristic_action_alignment / _heuristic_hallucination
# below). This is deliberately approximate -- good enough to unblock local
# U3 iteration, NOT good enough for U8/U9/U10's real promotion/regression
# decisions. Flip this to False (and finish the real Bedrock call in
# _call_judge) before those units start relying on this module for anything
# that matters. See plan Key Technical Decisions ("LLM-as-judge component...").
USE_HEURISTIC_JUDGE = True

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

_SOH_RE = re.compile(r"SOH\D{0,40}?([0-9]+\.?[0-9]*)", re.IGNORECASE)
_CONFIDENCE_RE = re.compile(r"[Cc]onfidence[:\s]*([0-9]*\.?[0-9]+)")
_TEMP_RE = re.compile(r"([0-9]*\.?[0-9]+)\s*(?:°C|degrees?\s*C)", re.IGNORECASE)
_SEVERITY_RE = re.compile(r"\b(Healthy|Warning|Damaged)\b", re.IGNORECASE)
# Damaged-tier text in this dataset never literally says "Damaged" -- it says
# "High-severity anomaly confirmed" instead (confirmed empirically, see
# docs/solutions/debugging-eval-metrics.md). Recognize that phrasing as an
# alias so severity extraction doesn't silently fail on every Damaged case.
_HIGH_SEVERITY_ALIAS_RE = re.compile(r"high[\s-]severity", re.IGNORECASE)


def _extract_fields(text: str) -> dict:
    """Pull structured fields out of prose text. Returns None for anything not found."""
    def _num(pattern: re.Pattern) -> float | None:
        m = pattern.search(text)
        return float(m.group(1)) if m else None

    sev_match = _SEVERITY_RE.search(text)
    if sev_match:
        severity = sev_match.group(1).capitalize()
    elif _HIGH_SEVERITY_ALIAS_RE.search(text):
        severity = "Damaged"
    else:
        severity = None

    return {
        "soh": _num(_SOH_RE),
        "confidence": _num(_CONFIDENCE_RE),
        "ambient_temp": _num(_TEMP_RE),
        "severity": severity,
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
    faithfulness is about not inventing numbers, not about matching phrasing.

    Confidence is deliberately excluded from this check: it's a detection-
    pipeline meta-signal, and neither the dataset's reference answers nor
    real model generations restate it in prose (confirmed empirically -- see
    docs/solutions/debugging-eval-metrics.md). Requiring it caused this check
    to fail on every example regardless of actual SOH/temperature faithfulness."""
    in_fields = _extract_fields(input_context)
    out_fields = _extract_fields(model_output)
    checks = [
        _numbers_match(in_fields["soh"], out_fields["soh"]),
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

    # Reuse _extract_fields (not a bare _SEVERITY_RE search) so the
    # "High-severity" == Damaged alias is recognized consistently here too.
    has_severity = _extract_fields(model_output)["severity"] is not None
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


_ID_RE = re.compile(r"\b(?:rack_\d+_\d+_mod\d+|hvac_\d+|plant\d+)\b", re.IGNORECASE)

# Crude, deterministic stand-in for "does the action fit the severity" --
# matched against the INPUT's severity (ground truth), same as
# check_severity_accuracy. Keyword lists come from the dataset's own action
# phrasing (bess_lora_train.jsonl), not invented.
_ACTION_KEYWORDS_BY_SEVERITY = {
    "Healthy": ("no action", "continue standard monitoring", "next scheduled inspection"),
    "Warning": ("schedule inspection", "increase monitoring", "monitor closely", "prepare replacement", "contingency"),
    "Damaged": ("immediate inspection", "immediate action", "planned replacement", "within 24 hours", "high-priority", "replace"),
}


def _heuristic_action_alignment(input_context: str, model_output: str) -> tuple[bool, str]:
    """INTERIM heuristic (see USE_HEURISTIC_JUDGE) -- checks the model's
    recommendation contains at least one keyword expected for the input's
    actual severity tier. Crude: a real judge would catch tonal mismatches
    this can't (e.g., a Damaged case that says "replace" but downplays
    urgency elsewhere). Good enough to unblock local iteration, not to be
    trusted for real promotion/regression decisions."""
    severity = _extract_fields(input_context)["severity"]
    if severity is None or severity not in _ACTION_KEYWORDS_BY_SEVERITY:
        return False, f"no recognizable severity in input to check action against (severity={severity!r})"

    lowered = model_output.lower()
    keywords = _ACTION_KEYWORDS_BY_SEVERITY[severity]
    matched = [kw for kw in keywords if kw in lowered]
    passed = len(matched) > 0
    detail = f"heuristic: severity={severity!r} matched_keywords={matched!r}"
    return passed, detail


def _heuristic_hallucination(input_context: str, model_output: str) -> tuple[bool, str]:
    """INTERIM heuristic (see USE_HEURISTIC_JUDGE) -- flags any node/plant
    identifier in the model output that doesn't appear in the input. Catches
    fabricated historical cases/nodes (the concrete failure mode we care
    about most) but won't catch subtler fabrications a real judge would
    (invented ambient conditions, invented policy claims, etc.)."""
    input_ids = {m.group(0).lower() for m in _ID_RE.finditer(input_context)}
    output_ids = {m.group(0).lower() for m in _ID_RE.finditer(model_output)}
    fabricated = output_ids - input_ids
    passed = len(fabricated) == 0
    detail = f"heuristic: fabricated_ids={sorted(fabricated)!r}" if fabricated else "heuristic: no unrecognized ids in output"
    return passed, detail


def _call_judge(task: str, input_context: str, reference_output: str, model_output: str) -> tuple[bool, str]:
    """Single seam for all LLM-as-judge calls. Swap/mock this in tests instead
    of mocking a bare API client, so test_metrics.py doesn't need to know
    which provider backs the judge."""
    if USE_HEURISTIC_JUDGE:
        if task == "action_alignment":
            return _heuristic_action_alignment(input_context, model_output)
        if task == "hallucination":
            return _heuristic_hallucination(input_context, model_output)

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


def score_against(held_out_set: list[dict], model_output_fn, verbose: bool = False) -> dict:
    """Score a model against a full held-out set. `held_out_set` is a list of
    {"input": ..., "output": ...} rows (same shape as bess_lora_train.jsonl).
    `model_output_fn(input_text) -> str` generates the model's answer for a
    given input -- caller supplies this so this module stays model-agnostic.

    Returns an aggregate: overall pass rate plus per-metric pass rates, which
    is what U3's threshold gate and U9/U10's comparisons consume.

    verbose=True prints progress as each example is scored -- one model
    generation per example means this loop can take a long time for a large
    held-out set (e.g. held_out.size in the hundreds), and without progress
    output that looks like a silent hang rather than normal, slow work."""
    import time

    n_total = len(held_out_set)
    results = []
    start = time.monotonic()
    for i, row in enumerate(held_out_set, start=1):
        results.append(
            score(reference_output=row["output"], model_output=model_output_fn(row["input"]), input_context=row["input"])
        )
        if verbose:
            elapsed = time.monotonic() - start
            avg = elapsed / i
            remaining = avg * (n_total - i)
            print(f"  scored {i}/{n_total} ({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)", flush=True)

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

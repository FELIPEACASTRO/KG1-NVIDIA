#!/usr/bin/env python3
"""KG1 SCORE TELEMETRY -- live exact-match canary metric computer (PROTOTYPE).

This module is the *metric computer* half of the in-training score canary. It is
DELIBERATELY decoupled from generation: it receives a list of already-generated
greedy outputs as (id, raw_output, answer, family, finish_reason, completion_tokens)
and produces the parseable telemetry JSON that the training job logs every N steps.

It reuses the OFFICIAL metric semantics verbatim (extract_final_answer + verify)
so the exact-match number it reports IS the score number (on the eval-set it sees).

It does NOT load any model, does NOT do teacher forcing, and does NOT compute loss.
Loss is intentionally absent: the whole point is to stop trusting loss.

Source of metric semantics (verbatim):
  artifacts/official_metric_audit_20260605/nvidia_nemotron_metric_extracted.py
Official inference surface (verbatim):
  docs KG1_SCORE_DETERMINANTS_AUDIT_2026_06_13.md (temp=0.0, top_p=1.0,
  max_tokens=7680, max_model_len=8192) -> greedy.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# OFFICIAL METRIC SEMANTICS (verbatim from nvidia_nemotron_metric_extracted.py)
# ---------------------------------------------------------------------------

def extract_final_answer(text: str | None) -> str:
    if text is None:
        return "NOT_FOUND"
    boxed_starts = list(re.finditer(r"\\boxed\{", text))
    matches: list[str] = []
    for i, m in enumerate(boxed_starts):
        start = m.end()
        end = boxed_starts[i + 1].start() if i + 1 < len(boxed_starts) else len(text)
        segment = text[start:end]
        last_brace = segment.rfind("}")
        matches.append(segment[:last_brace] if last_brace != -1 else segment)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        if non_empty:
            return non_empty[-1]
        return matches[-1].strip()
    patterns = [
        r"The final answer is:\s*([^\n]+)",
        r"Final answer is:\s*([^\n]+)",
        r"Final answer\s*[:：]\s*([^\n]+)",
        r"final answer\s*[:：]\s*([^\n]+)",
    ]
    for pattern in patterns:
        m = re.findall(pattern, text, re.IGNORECASE)
        if m:
            return m[-1].strip()
    m = re.findall(r"-?\d+(?:\.\d+)?", text)
    if m:
        return m[-1]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else "NOT_FOUND"


def verify(stored_answer: str, predicted: str) -> bool:
    stored_answer = str(stored_answer).strip()
    predicted = str(predicted).strip()
    if re.fullmatch(r"[01]+", stored_answer):
        return predicted.lower() == stored_answer.lower()
    try:
        stored_num = float(stored_answer)
        predicted_num = float(predicted)
        return math.isclose(stored_num, predicted_num, rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return predicted.lower() == stored_answer.lower()


# ---------------------------------------------------------------------------
# FORMAT / TRUNCATION DIAGNOSTICS (score-facing, not loss)
# ---------------------------------------------------------------------------

TRUNCATION_FINISH_REASONS = {
    "length", "max_tokens", "max_output_tokens", "token_limit", "truncated",
}

# Official surface (greedy). Truncation == reached generation cap before EOS.
OFFICIAL_MAX_TOKENS = 7680


def is_truncated(finish_reason: str | None, completion_tokens: int | None,
                 max_tokens: int = OFFICIAL_MAX_TOKENS) -> bool:
    fr = (str(finish_reason).strip().lower() if finish_reason is not None else "")
    if fr in TRUNCATION_FINISH_REASONS:
        return True
    if completion_tokens is not None:
        try:
            if int(completion_tokens) >= int(max_tokens):
                return True
        except (TypeError, ValueError):
            pass
    return False


def count_boxed(text: str | None) -> int:
    if text is None:
        return 0
    return len(re.findall(r"\\boxed\{", text))


def has_unclosed_box(text: str | None) -> bool:
    """A \\boxed{ with no matching } anywhere after the LAST \\boxed{ start."""
    if text is None:
        return False
    starts = list(re.finditer(r"\\boxed\{", text))
    if not starts:
        return False
    tail = text[starts[-1].end():]
    return "}" not in tail


def box_status(text: str | None) -> str:
    n = count_boxed(text)
    if n == 0:
        return "no_box"
    if has_unclosed_box(text):
        return "unclosed_box"
    if n == 1:
        return "single_box"
    return "double_box"  # n >= 2


# ---------------------------------------------------------------------------
# TELEMETRY COMPUTER
# ---------------------------------------------------------------------------

def _new_fam_counter() -> dict[str, Any]:
    return {
        "rows": 0, "correct": 0, "truncated": 0,
        "has_box": 0, "single_box": 0, "extract_ok": 0,
    }


def compute_score_telemetry(
    rows: Iterable[dict[str, Any]],
    *,
    baseline_per_family: dict[str, int] | None = None,
    baseline_truncation_per_family: dict[str, float] | None = None,
    step: int | None = None,
    max_step: int | None = None,
    max_tokens: int = OFFICIAL_MAX_TOKENS,
    weak_families: tuple[str, ...] = ("bit_manipulation", "equation_transform"),
    protected_families: tuple[str, ...] = (
        "gravity_constant", "unit_conversion", "numeral_system", "text_encryption",
    ),
) -> dict[str, Any]:
    """Compute the live score telemetry JSON.

    rows: iterable of dicts with keys:
      id, raw_output, answer, family, finish_reason (opt), completion_tokens (opt)
    baseline_per_family: {family: correct_count} for the 086 baseline on THIS eval-set.
    """
    by_fam: dict[str, dict[str, Any]] = defaultdict(_new_fam_counter)
    total = _new_fam_counter()
    # global format anomalies
    fmt = {"no_box": 0, "single_box": 0, "double_box": 0, "unclosed_box": 0,
           "extract_not_found": 0}

    for r in rows:
        fam = str(r.get("family", "unknown"))
        raw = r.get("raw_output")
        ans = r.get("answer", "")
        fr = r.get("finish_reason")
        ct = r.get("completion_tokens")

        extracted = extract_final_answer(raw)
        ok = verify(ans, extracted)
        trunc = is_truncated(fr, ct, max_tokens)
        bstat = box_status(raw)
        has_box = bstat in ("single_box", "double_box", "unclosed_box")
        single = bstat == "single_box"
        extract_ok = extracted != "NOT_FOUND"

        c = by_fam[fam]
        for tgt in (c, total):
            tgt["rows"] += 1
            tgt["correct"] += int(ok)
            tgt["truncated"] += int(trunc)
            tgt["has_box"] += int(has_box)
            tgt["single_box"] += int(single)
            tgt["extract_ok"] += int(extract_ok)

        fmt[bstat] = fmt.get(bstat, 0) + 1
        if not extract_ok:
            fmt["extract_not_found"] += 1

    def pct(num: int, den: int) -> float:
        return round(num / den, 6) if den else 0.0

    fam_report: dict[str, dict[str, Any]] = {}
    exact_delta_by_family: dict[str, int] = {}
    for fam, c in sorted(by_fam.items()):
        base = (baseline_per_family or {}).get(fam)
        delta = (c["correct"] - base) if base is not None else None
        if delta is not None:
            exact_delta_by_family[fam] = delta
        fam_report[fam] = {
            "rows": c["rows"],
            "correct": c["correct"],
            "accuracy": pct(c["correct"], c["rows"]),
            "baseline_correct": base,
            "exact_delta": delta,
            "truncation_rate": pct(c["truncated"], c["rows"]),
            "boxed_rate": pct(c["has_box"], c["rows"]),
            "single_boxed_rate": pct(c["single_box"], c["rows"]),
            "extract_ok_rate": pct(c["extract_ok"], c["rows"]),
        }

    # Aggregate deltas
    def agg_delta(fams: tuple[str, ...]) -> int | None:
        vals = [exact_delta_by_family[f] for f in fams if f in exact_delta_by_family]
        return sum(vals) if vals else (0 if baseline_per_family else None)

    weak_delta = agg_delta(weak_families)
    protected_delta = agg_delta(protected_families)
    bit_delta = exact_delta_by_family.get("bit_manipulation")
    eq_delta = exact_delta_by_family.get("equation_transform")
    overall_baseline = (sum(baseline_per_family.values())
                        if baseline_per_family else None)
    overall_delta = (total["correct"] - overall_baseline
                     if overall_baseline is not None else None)

    overall_trunc_rate = pct(total["truncated"], total["rows"])
    baseline_overall_trunc = None
    if baseline_truncation_per_family:
        # weighted by current per-fam rows
        num = 0.0
        for fam, c in by_fam.items():
            num += baseline_truncation_per_family.get(fam, 0.0) * c["rows"]
        baseline_overall_trunc = round(num / total["rows"], 6) if total["rows"] else 0.0

    # ---- ABORT RULES (exact-match based; NEVER loss) ----
    abort = evaluate_abort(
        overall_delta=overall_delta,
        protected_delta=protected_delta,
        bit_delta=bit_delta,
        eq_delta=eq_delta,
        weak_delta=weak_delta,
        overall_trunc_rate=overall_trunc_rate,
        baseline_overall_trunc=baseline_overall_trunc,
        has_baseline=baseline_per_family is not None,
    )

    out = {
        "schema_version": "kg1_score_live_v1",
        "kind": "exact_match_greedy_canary",
        "step": step,
        "max_step": max_step,
        "eval_rows": total["rows"],
        "exact_match": {
            "correct": total["correct"],
            "rows": total["rows"],
            "accuracy": pct(total["correct"], total["rows"]),
            "baseline_correct": overall_baseline,
            "overall_exact_delta": overall_delta,
        },
        "exact_delta": {
            "overall": overall_delta,
            "weak": weak_delta,
            "bit_manipulation": bit_delta,
            "equation_transform": eq_delta,
            "protected": protected_delta,
        },
        "truncation": {
            "rate": overall_trunc_rate,
            "count": total["truncated"],
            "baseline_rate": baseline_overall_trunc,
            "max_tokens": max_tokens,
        },
        "boxed": {
            "boxed_rate": pct(total["has_box"], total["rows"]),
            "exactly_one_boxed_rate": pct(total["single_box"], total["rows"]),
            "extract_ok_rate": pct(total["extract_ok"], total["rows"]),
        },
        "format_anomalies": {
            "no_box": fmt["no_box"],
            "double_boxed": fmt["double_box"],
            "unclosed_box": fmt["unclosed_box"],
            "extract_not_found": fmt["extract_not_found"],
        },
        "by_family": fam_report,
        "abort": abort,
    }
    return out


def evaluate_abort(
    *,
    overall_delta: int | None,
    protected_delta: int | None,
    bit_delta: int | None,
    eq_delta: int | None,
    weak_delta: int | None,
    overall_trunc_rate: float,
    baseline_overall_trunc: float | None,
    has_baseline: bool,
) -> dict[str, Any]:
    """ABORT rules are exact-match based. Never loss.

    A checkpoint is SAFE-TO-CONTINUE only if ALL hold (when baseline is known):
      1. protected exact_delta >= 0      (no protected-family regression)
      2. overall exact_delta   >= 0      (total did not drop)
      3. target exact_delta    >= 0      (bit OR equation improved net non-negative)
      4. truncation rate       <= baseline truncation (+ small tol)
    Without a real baseline we CANNOT abort on regression (only on hard format/trunc
    collapse). The status then is WATCH, not OK -- the absence of a baseline is itself
    a reported blocker (baseline per-item is a hard prerequisite).
    """
    TRUNC_TOL = 0.01  # allow 1pp jitter
    reasons: list[str] = []
    blockers: list[str] = []

    if not has_baseline:
        blockers.append("baseline_per_item_missing")

    # Hard format/truncation collapse can abort even without baseline.
    hard_trunc_cap = 0.20
    if overall_trunc_rate > hard_trunc_cap:
        reasons.append(f"truncation_rate>{hard_trunc_cap}:{overall_trunc_rate}")

    if has_baseline:
        if protected_delta is not None and protected_delta < 0:
            reasons.append(f"protected_regression:{protected_delta}")
        if overall_delta is not None and overall_delta < 0:
            reasons.append(f"overall_regression:{overall_delta}")
        # target: bit E equation INDIVIDUALMENTE nao podem regredir (gate full947_089 exige
        # +1 em CADA; agregado mascararia regressao de uma familia compensada pela outra).
        if bit_delta is not None and bit_delta < 0:
            reasons.append(f"bit_individual_regression:{bit_delta}")
        if eq_delta is not None and eq_delta < 0:
            reasons.append(f"eq_individual_regression:{eq_delta}")
        if weak_delta is not None and weak_delta < 0:
            reasons.append(f"weak_target_regression:{weak_delta}")
        if baseline_overall_trunc is not None and overall_trunc_rate > baseline_overall_trunc + TRUNC_TOL:
            reasons.append(
                f"truncation_above_baseline:{overall_trunc_rate}>"
                f"{baseline_overall_trunc}+{TRUNC_TOL}")

    if reasons:
        decision = "ABORT"
    elif blockers:
        decision = "WATCH"
    else:
        decision = "OK"

    return {
        "decision": decision,
        "abort_reasons": reasons,
        "blockers": blockers,
        "rule": ("never_loss; protected>=0 AND overall>=0 AND bit>=0 AND eq>=0 (individual) AND "
                 "weak>=0 AND trunc<=baseline+tol; hard_trunc_cap=0.20"),
    }


def render_marker(telemetry: dict[str, Any]) -> str:
    """Single parseable machine line + a human heartbeat line."""
    em = telemetry["exact_match"]
    d = telemetry["exact_delta"]
    t = telemetry["truncation"]
    b = telemetry["boxed"]
    a = telemetry["abort"]
    machine = "KG1_SCORE_LIVE_JSON " + json.dumps(telemetry, ensure_ascii=False)
    human = (
        f"KG1_SCORE_LIVE_HEARTBEAT step={telemetry.get('step')}/{telemetry.get('max_step')} "
        f"EM={em['correct']}/{em['rows']}({em['accuracy']:.3f}) "
        f"d_overall={d['overall']} d_bit={d['bit_manipulation']} d_eq={d['equation_transform']} "
        f"d_protected={d['protected']} trunc={t['rate']:.3f} "
        f"box1={b['exactly_one_boxed_rate']:.3f} -> {a['decision']}"
    )
    status = f"KG1_SCORE_LIVE_STATUS={a['decision']}"
    return machine + "\n" + human + "\n" + status

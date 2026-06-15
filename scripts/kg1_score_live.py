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


# ---------------------------------------------------------------------------
# DASHBOARD DIDATICO (Use a Cabeca) -- camada de leitura HUMANA, muito detalhada.
# Reusado no treino (SCORE-LIVE) e no Job B (eval full947). Nao substitui os
# marcadores de maquina (render_marker); complementa.
# ---------------------------------------------------------------------------

FULL947_FAMILY_SIZE = {"bit_manipulation": 160, "equation_transform": 155}
FULL947_PROTECTED_TOTAL = 632
FULL947_TOTAL = 947
SCORE_TARGETS = {"086_piso": 823, "meta_0.87": 824, "stretch_0.89": 843}
PROTECTED_FAMILIES = ("gravity_constant", "unit_conversion", "numeral_system", "text_encryption")
WEAK_FAMILIES = ("bit_manipulation", "equation_transform")
_FAMILY_LABEL = {
    "bit_manipulation": "bit (ALVO)",
    "equation_transform": "equation (ALVO)",
    "gravity_constant": "gravity (protegida)",
    "unit_conversion": "unit (protegida)",
    "numeral_system": "numeral (protegida)",
    "text_encryption": "text (protegida)",
}


def _bar(frac: float | None, width: int = 12) -> str:
    f = 0.0 if frac is None else max(0.0, min(1.0, float(frac)))
    n = int(round(f * width))
    return "[" + "#" * n + "." * (width - n) + "]"


def _delta_icon(delta: int | None) -> str:
    if delta is None:
        return " ?"
    if delta > 0:
        return f"+{delta} UP"
    if delta < 0:
        return f"{delta} DOWN"
    return " 0 ="


def project_full947(tel: dict[str, Any]) -> dict[str, Any]:
    """Projecao INDICATIVA do full947 a partir das taxas por familia (amostra pequena =>
    ruidosa). Termometro -- o juiz real e o Notebook B (vLLM full947)."""
    fam = tel.get("by_family") or {}

    def acc(name: str) -> float | None:
        f = fam.get(name)
        return float(f["accuracy"]) if f and f.get("rows") else None

    bit_acc, eq_acc = acc("bit_manipulation"), acc("equation_transform")
    prot_correct = sum(fam[f]["correct"] for f in PROTECTED_FAMILIES if f in fam)
    prot_rows = sum(fam[f]["rows"] for f in PROTECTED_FAMILIES if f in fam)
    prot_acc = (prot_correct / prot_rows) if prot_rows else None
    proj = 0.0
    if bit_acc is not None:
        proj += bit_acc * FULL947_FAMILY_SIZE["bit_manipulation"]
    if eq_acc is not None:
        proj += eq_acc * FULL947_FAMILY_SIZE["equation_transform"]
    if prot_acc is not None:
        proj += prot_acc * FULL947_PROTECTED_TOTAL
    projected = int(round(proj))
    return {
        "projected_correct": projected,
        "projected_score": round(projected / FULL947_TOTAL, 4),
        "bit_acc": bit_acc, "eq_acc": eq_acc, "protected_acc": prot_acc,
        "gap_to_0_87": SCORE_TARGETS["meta_0.87"] - projected,
    }


def render_dashboard(tel: dict[str, Any], *, title: str = "PAINEL SCORE-LIVE",
                     source: str = "geracao real (greedy oficial)") -> str:
    """Dashboard humano MUITO detalhado e didatico (Use a Cabeca)."""
    em = tel.get("exact_match", {}); ed = tel.get("exact_delta", {})
    tr = tel.get("truncation", {}); bx = tel.get("boxed", {})
    fa = tel.get("format_anomalies", {}); ab = tel.get("abort", {})
    fam = tel.get("by_family") or {}
    L: list[str] = []
    L.append("=" * 80)
    L.append(f"  {title}  |  step {tel.get('step')}/{tel.get('max_step')}  |  fonte: {source}")
    L.append("  (acerto = boxed exact-match OFICIAL, NAO loss. 'loss bom != score bom')")
    L.append("=" * 80)
    acc = float(em.get("accuracy") or 0.0)
    L.append("[1] PLACAR GERAL (no eval-set):")
    L.append(f"    acerto = {em.get('correct')}/{em.get('rows')} = {acc * 100:.1f}%   {_bar(acc)}")
    if em.get("baseline_correct") is not None:
        L.append(f"    vs 086 (baseline): delta {_delta_icon(ed.get('overall'))}   (>=0 = nao regrediu)")
    expl = {"OK": "pode seguir", "WATCH": "observar (falta baseline ou em duvida)",
            "ABORT": "score real REGREDIU -> PARAR"}.get(str(ab.get("decision")), "?")
    L.append(f"    STATUS: {ab.get('decision')}  ->  {expl}")
    if ab.get("abort_reasons"):
        L.append("    motivos: " + "; ".join(str(x) for x in ab["abort_reasons"]))
    if ab.get("blockers"):
        L.append("    pendencias: " + "; ".join(str(x) for x in ab["blockers"]))
    L.append("-" * 80)
    L.append("[2] POR FAMILIA (onde esta o ganho/risco):")
    L.append(f"    {'familia':<20}{'acerto':<12}{'barra':<14}{'trunc':<7}{'boxed':<7}{'vs086'}")
    for name in list(WEAK_FAMILIES) + list(PROTECTED_FAMILIES):
        f = fam.get(name)
        if not f:
            continue
        a = float(f.get("accuracy") or 0.0)
        L.append(
            f"    {_FAMILY_LABEL.get(name, name):<20}"
            f"{str(f.get('correct')) + '/' + str(f.get('rows')):<12}{_bar(a, 10):<14}"
            f"{f.get('truncation_rate', 0) * 100:>3.0f}%   {f.get('boxed_rate', 0) * 100:>3.0f}%   "
            f"{_delta_icon(f.get('exact_delta'))}"
        )
    pj = project_full947(tel)
    L.append("-" * 80)
    L.append("[3] PROJECAO full947 (INDICATIVA -- amostra pequena; juiz real = Notebook B):")
    L.append(f"    se as taxas se mantiverem: ~{pj['projected_correct']}/947 = {pj['projected_score']:.4f}")
    L.append("    referencias: 086(piso)=823/0.869 | meta=824/0.870 | stretch=843/0.890")
    g = pj["gap_to_0_87"]
    L.append(f"    p/ bater 0.87: " + (f"FALTAM +{g} itens" if g > 0 else f"JA NO ALVO ({-g} acima)"))
    L.append("-" * 80)
    L.append("[4] SAUDE DE FORMATO (impacta o score direto):")
    L.append(f"    truncacao geral: {tr.get('rate', 0) * 100:.1f}% ({tr.get('count')} cortados antes do boxed)")
    L.append(f"    tem boxed: {bx.get('boxed_rate', 0) * 100:.0f}% | exatamente 1 boxed: "
             f"{bx.get('exactly_one_boxed_rate', 0) * 100:.0f}% | extracao ok: {bx.get('extract_ok_rate', 0) * 100:.0f}%")
    L.append(f"    anomalias: sem_box={fa.get('no_box')} box_duplo={fa.get('double_boxed')} "
             f"box_aberto={fa.get('unclosed_box')} nao_extraiu={fa.get('extract_not_found')}")
    L.append("-" * 80)
    L.append("[5] COMO LER (didatico):")
    L.append("    - bit/equation = ALVOS (tem headroom). Protegidas devem ficar ~100%.")
    L.append("    - truncacao alta = resposta cortada antes do boxed -> perde ponto.")
    L.append("    - box_aberto/sem_box = formato quebrado -> a metrica nao acha a resposta.")
    L.append("    - mede o SCORE REAL; confirme sempre no Notebook B (vLLM full947).")
    L.append("=" * 80)
    return "\n".join(L)

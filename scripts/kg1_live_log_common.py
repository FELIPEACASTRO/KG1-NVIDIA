#!/usr/bin/env python3
"""Shared parser for KG1 real-time job logs."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TEACH_RE = re.compile(
    r"\[KG1-TEACH\]\[(?P<time>[^\]]+)\]\[(?P<stage>[^\]]+)\]\[(?P<status>[^\]]+)\]"
)
CONTRACT_RE = re.compile(
    r"KG1_SCORE_CONTRACT_STATUS=(?P<status>\w+).*?"
    r"errors=(?P<errors>\d+).*?"
    r"warnings=(?P<warnings>\d+).*?"
    r"target_correct_required=(?P<target>\d+).*?"
    r"additional_rows_required=(?P<additional>\d+)"
)
PROXY_RE = re.compile(
    r"KG1_SCORE_PROXY_STATUS=(?P<label>\S+)\s+"
    r"loss=(?P<loss>\S+)\s+"
    r"boxed_tail_loss=(?P<boxed_loss>\S+)\s+"
    r"boxed_tail_token_acc=(?P<boxed_acc>\S+)\s+"
    r"boxed_tail_exact_rate=(?P<boxed_exact>\S+)\s+"
    r"delta_loss=(?P<delta_loss>\S+)\s+"
    r"delta_boxed_tail_loss=(?P<delta_boxed_loss>\S+)\s+"
    r"delta_boxed_tail_exact_rate=(?P<delta_boxed_exact>\S+)"
)
PROXY_FAMILY_RE = re.compile(
    r"KG1_SCORE_PROXY_FAMILY_STATUS=(?P<label>\S+)\s+"
    r"family=(?P<family>\S+)\s+"
    r"rows=(?P<rows>\d+)\s+"
    r"loss=(?P<loss>\S+)\s+"
    r"boxed_tail_loss=(?P<boxed_loss>\S+)\s+"
    r"boxed_tail_token_acc=(?P<boxed_acc>\S+)\s+"
    r"boxed_tail_exact_rate=(?P<boxed_exact>\S+)"
)
PROXY_CLASS_RE = re.compile(
    r"KG1_SCORE_PROXY_CLASS_STATUS=(?P<label>\S+)\s+"
    r"class=(?P<class_name>\S+)\s+"
    r"rows=(?P<rows>\d+)\s+"
    r"loss=(?P<loss>\S+)\s+"
    r"boxed_tail_loss=(?P<boxed_loss>\S+)\s+"
    r"boxed_tail_token_acc=(?P<boxed_acc>\S+)\s+"
    r"boxed_tail_exact_rate=(?P<boxed_exact>\S+)"
)
DATASET_AUDIT_RE = re.compile(
    r"KG1_V1243_DATASET_AUDIT_STATUS=(?P<status>\w+)\s+"
    r"files=(?P<files>\d+)\s+"
    r"errors=(?P<errors>\d+)\s+"
    r"warnings=(?P<warnings>\d+)"
)
TRAJECTORY_RE = re.compile(
    r"KG1_SCORE_TRAJECTORY_STATUS=(?P<label>\S+)\s+"
    r"status=(?P<status>\w+)\s+"
    r"alignment=(?P<alignment>\S+)\s+"
    r"target_correct_required=(?P<target>\d+)\s+"
    r"additional_rows_required=(?P<additional>\d+)\s+"
    r"weak_exact_delta=(?P<weak_exact>\S+)\s+"
    r"bit_exact_delta=(?P<bit_exact>\S+)\s+"
    r"equation_exact_delta=(?P<equation_exact>\S+)\s+"
    r"protected_exact_delta=(?P<protected_exact>\S+)\s+"
    r"overall_exact_delta=(?P<overall_exact>\S+)\s+"
    r"boxed_loss_delta=(?P<boxed_loss>\S+)"
)
STEP_RE = re.compile(
    r"step=(?P<step>\d+)/(?P<total>\d+)\s+lr=(?P<lr>\S+)\s+"
    r"train_loss=(?P<loss>\S+).*?(?P<mem>mem_alloc=.*|cuda_mem=unavailable)"
)
END_RE = re.compile(r"KG1_COLAB_REALTIME_RUNNER_END\s+return_code=(?P<return_code>-?\d+)")
# PRECISO (anti falso-positivo): so casa ERRO REAL, nao mencao benigna no meio do texto.
#  - cabecalho de traceback real (coluna 0)
#  - excecao levantada de verdade: "XxxError: ..." no inicio da linha (RuntimeError:, ValueError:, OutOfMemoryError:)
#  - mensagens especificas de OOM/NaN/contrato/abort
FATAL_RE = re.compile(
    r"^Traceback \(most recent call last\):"
    r"|^[A-Za-z_][A-Za-z0-9_.]*Error: "
    r"|CUDA out of memory"
    r"|torch\.cuda\.OutOfMemoryError"
    r"|FloatingPointError"
    r"|Non-finite (?:loss|baseline|eval)"
    r"|ABORT:|score_contract_runtime_failed|KG1_WATCHDOG_STOP"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_float(value: object) -> float | None:
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def empty_state() -> dict[str, Any]:
    return {
        "schema_version": "kg1_live_log_health_v1",
        "updated_at_utc": utc_now(),
        "line_count": 0,
        "last_line": "",
        "last_teach": None,
        "stage_status": {},
        "score_contract": None,
        "score_proxy": None,
        "score_proxy_by_family": {},
        "score_proxy_by_class": {},
        "score_trajectory": None,
        "dataset_audit": None,
        "train_step": None,
        "alerts": [],
        "health": "BOOT",
        "health_reason": "waiting_for_log_lines",
    }


def append_alert(state: dict[str, Any], message: str) -> None:
    alerts = list(state.get("alerts") or [])
    alerts.append({"line": int(state.get("line_count") or 0), "message": message[:500]})
    state["alerts"] = alerts[-25:]


def ingest_line(state: dict[str, Any], line: str) -> dict[str, Any]:
    clean = line.rstrip("\n")
    state["line_count"] = int(state.get("line_count") or 0) + 1
    state["last_line"] = clean[-1000:]
    state["updated_at_utc"] = utc_now()

    match = TEACH_RE.search(clean)
    if match:
        event = match.groupdict()
        state["last_teach"] = event
        stage_status = dict(state.get("stage_status") or {})
        stage_status[event["stage"]] = event["status"]
        state["stage_status"] = stage_status
        if event["stage"] == "ABORT" or event["status"] == "STOP":
            append_alert(state, clean)

    match = CONTRACT_RE.search(clean)
    if match:
        groups = match.groupdict()
        state["score_contract"] = {
            "status": groups["status"],
            "errors": int(groups["errors"]),
            "warnings": int(groups["warnings"]),
            "target_correct_required": int(groups["target"]),
            "additional_rows_required": int(groups["additional"]),
        }
        if groups["status"] != "PASS":
            append_alert(state, clean)

    match = PROXY_RE.search(clean)
    if match:
        groups = match.groupdict()
        state["score_proxy"] = {
            "label": groups["label"],
            "loss": parse_float(groups["loss"]),
            "boxed_tail_loss": parse_float(groups["boxed_loss"]),
            "boxed_tail_token_accuracy": parse_float(groups["boxed_acc"]),
            "boxed_tail_exact_rate": parse_float(groups["boxed_exact"]),
            "delta_loss": parse_float(groups["delta_loss"]),
            "delta_boxed_tail_loss": parse_float(groups["delta_boxed_loss"]),
            "delta_boxed_tail_exact_rate": parse_float(groups["delta_boxed_exact"]),
        }

    match = PROXY_FAMILY_RE.search(clean)
    if match:
        groups = match.groupdict()
        by_family = dict(state.get("score_proxy_by_family") or {})
        by_family[groups["family"]] = {
            "label": groups["label"],
            "rows": int(groups["rows"]),
            "loss": parse_float(groups["loss"]),
            "boxed_tail_loss": parse_float(groups["boxed_loss"]),
            "boxed_tail_token_accuracy": parse_float(groups["boxed_acc"]),
            "boxed_tail_exact_rate": parse_float(groups["boxed_exact"]),
        }
        state["score_proxy_by_family"] = by_family

    match = PROXY_CLASS_RE.search(clean)
    if match:
        groups = match.groupdict()
        by_class = dict(state.get("score_proxy_by_class") or {})
        by_class[groups["class_name"]] = {
            "label": groups["label"],
            "rows": int(groups["rows"]),
            "loss": parse_float(groups["loss"]),
            "boxed_tail_loss": parse_float(groups["boxed_loss"]),
            "boxed_tail_token_accuracy": parse_float(groups["boxed_acc"]),
            "boxed_tail_exact_rate": parse_float(groups["boxed_exact"]),
        }
        state["score_proxy_by_class"] = by_class

    match = DATASET_AUDIT_RE.search(clean)
    if match:
        groups = match.groupdict()
        state["dataset_audit"] = {
            "status": groups["status"],
            "files": int(groups["files"]),
            "errors": int(groups["errors"]),
            "warnings": int(groups["warnings"]),
        }
        if groups["status"] != "PASS":
            append_alert(state, clean)

    match = TRAJECTORY_RE.search(clean)
    if match:
        groups = match.groupdict()
        state["score_trajectory"] = {
            "label": groups["label"],
            "status": groups["status"],
            "alignment": groups["alignment"],
            "target_correct_required": int(groups["target"]),
            "additional_rows_required": int(groups["additional"]),
            "weak_exact_delta": parse_float(groups["weak_exact"]),
            "bit_exact_delta": parse_float(groups["bit_exact"]),
            "equation_exact_delta": parse_float(groups["equation_exact"]),
            "protected_exact_delta": parse_float(groups["protected_exact"]),
            "overall_exact_delta": parse_float(groups["overall_exact"]),
            "boxed_tail_loss_delta": parse_float(groups["boxed_loss"]),
        }
        if groups["status"] == "STOP":
            append_alert(state, clean)

    match = STEP_RE.search(clean)
    if match:
        groups = match.groupdict()
        state["train_step"] = {
            "step": int(groups["step"]),
            "total": int(groups["total"]),
            "lr": groups["lr"],
            "train_loss": parse_float(groups["loss"]),
            "memory": groups["mem"],
        }

    if FATAL_RE.search(clean):
        append_alert(state, clean)

    match = END_RE.search(clean)
    if match:
        return_code = int(match.group("return_code"))
        state["return_code"] = return_code
        state["process_finished"] = True
        if return_code != 0:
            append_alert(state, clean)

    update_health(state)
    return state


def parse_text(text: str) -> dict[str, Any]:
    state = empty_state()
    for line in text.splitlines():
        ingest_line(state, line)
    return state


def update_health(state: dict[str, Any]) -> None:
    stage_status = state.get("stage_status") or {}
    contract = state.get("score_contract") or {}
    proxy = state.get("score_proxy") or {}
    trajectory = state.get("score_trajectory") or {}
    alerts = state.get("alerts") or []

    if alerts:
        state["health"] = "STOP"
        state["health_reason"] = str(alerts[-1].get("message", "alert"))
        return
    if stage_status.get("SCORE_TRAJECTORY") == "STOP" or trajectory.get("status") == "STOP":
        state["health"] = "STOP"
        state["health_reason"] = "score_trajectory_stop"
        return
    if stage_status.get("SCORE_TRAJECTORY") == "RISK" or trajectory.get("status") == "RISK":
        state["health"] = "RISK"
        state["health_reason"] = "score_trajectory_risk"
        return
    if state.get("process_finished") and int(state.get("return_code") or 0) == 0:
        boxed_loss_delta = parse_float(proxy.get("delta_boxed_tail_loss"))
        boxed_exact_delta = parse_float(proxy.get("delta_boxed_tail_exact_rate"))
        proxy_direction_ok = (
            boxed_loss_delta is not None
            and boxed_loss_delta < 0
            and (boxed_exact_delta is None or boxed_exact_delta >= 0)
        )
        trajectory_ok = stage_status.get("SCORE_TRAJECTORY") == "OK" or trajectory.get("status") == "OK"
        if (
            contract.get("status") == "PASS"
            and (stage_status.get("SCORE_PROXY") == "OK" or proxy_direction_ok)
            and (not trajectory or trajectory_ok)
        ):
            state["health"] = "OK"
            state["health_reason"] = "finished_contract_passed_proxy_and_trajectory_ok"
            return
        if contract.get("status") == "PASS":
            state["health"] = "OK"
            state["health_reason"] = "finished_contract_passed"
            return
        state["health"] = "OK"
        state["health_reason"] = "finished_successfully"
        return
    if contract.get("status") == "FAIL":
        state["health"] = "STOP"
        state["health_reason"] = "score_contract_failed"
        return
    if stage_status.get("SCORE_CONTRACT") == "STOP":
        state["health"] = "STOP"
        state["health_reason"] = "score_contract_teach_stop"
        return
    if stage_status.get("SCORE_PROXY") == "STOP":
        state["health"] = "STOP"
        state["health_reason"] = "score_proxy_stop"
        return
    if stage_status.get("SCORE_PROXY") == "RISK":
        state["health"] = "RISK"
        state["health_reason"] = "score_proxy_risk"
        return
    boxed_loss_delta = parse_float(proxy.get("delta_boxed_tail_loss"))
    boxed_exact_delta = parse_float(proxy.get("delta_boxed_tail_exact_rate"))
    if boxed_loss_delta is not None and boxed_loss_delta >= 0 and boxed_exact_delta is not None and boxed_exact_delta <= 0:
        state["health"] = "RISK"
        state["health_reason"] = "boxed_tail_not_improving"
        return

    proxy_direction_ok = (
        boxed_loss_delta is not None
        and boxed_loss_delta < 0
        and (boxed_exact_delta is None or boxed_exact_delta >= 0)
    )
    trajectory_ok = stage_status.get("SCORE_TRAJECTORY") == "OK" or trajectory.get("status") == "OK"
    if (
        contract.get("status") == "PASS"
        and (stage_status.get("SCORE_PROXY") == "OK" or proxy_direction_ok)
        and (not trajectory or trajectory_ok)
    ):
        state["health"] = "OK"
        state["health_reason"] = "contract_passed_proxy_and_trajectory_ok"
        return
    if contract.get("status") == "PASS":
        state["health"] = "WATCH"
        state["health_reason"] = "contract_passed_waiting_for_proxy"
        return

    state["health"] = "BOOT"
    state["health_reason"] = "waiting_for_contract"


def target_math(target_accuracy: float, full_rows: int, baseline_correct: int) -> dict[str, Any]:
    required = int(math.ceil(float(target_accuracy) * int(full_rows)))
    return {
        "target_accuracy": target_accuracy,
        "full_rows": full_rows,
        "baseline_correct": baseline_correct,
        "baseline_accuracy": baseline_correct / max(1, full_rows),
        "target_correct_required": required,
        "additional_rows_required": max(0, required - baseline_correct),
    }


def render_dashboard(
    state: dict[str, Any],
    *,
    target_accuracy: float,
    full_rows: int = 947,
    baseline_correct: int = 823,
) -> str:
    math_state = target_math(target_accuracy, full_rows, baseline_correct)
    contract = state.get("score_contract") or {}
    proxy = state.get("score_proxy") or {}
    proxy_by_family = state.get("score_proxy_by_family") or {}
    proxy_by_class = state.get("score_proxy_by_class") or {}
    trajectory = state.get("score_trajectory") or {}
    dataset_audit = state.get("dataset_audit") or {}
    step = state.get("train_step") or {}
    alerts = state.get("alerts") or []
    lines = [
        "=" * 78,
        f"KG1 LIVE COLAB MONITOR | health={state.get('health')} | reason={state.get('health_reason')}",
        "=" * 78,
        f"updated_at_utc: {state.get('updated_at_utc')}",
        f"lines_seen: {state.get('line_count')}",
        f"target_math: target={target_accuracy:.4f} required={math_state['target_correct_required']}/{full_rows} "
        f"gain_needed=+{math_state['additional_rows_required']} from {baseline_correct}/{full_rows}",
        "",
        "dataset_audit:",
        f"  status={dataset_audit.get('status', 'missing')} files={dataset_audit.get('files', '?')} "
        f"errors={dataset_audit.get('errors', '?')} warnings={dataset_audit.get('warnings', '?')}",
        "score_contract:",
        f"  status={contract.get('status', 'missing')} errors={contract.get('errors', '?')} "
        f"warnings={contract.get('warnings', '?')} additional_rows={contract.get('additional_rows_required', '?')}",
        "train_step:",
        f"  step={step.get('step', '?')}/{step.get('total', '?')} train_loss={step.get('train_loss', '?')} "
        f"lr={step.get('lr', '?')} mem={step.get('memory', '?')}",
        "score_proxy:",
        f"  label={proxy.get('label', 'missing')} loss={proxy.get('loss', '?')} "
        f"boxed_loss={proxy.get('boxed_tail_loss', '?')} boxed_acc={proxy.get('boxed_tail_token_accuracy', '?')} "
        f"boxed_exact={proxy.get('boxed_tail_exact_rate', '?')}",
        f"  delta_loss={proxy.get('delta_loss', '?')} "
        f"delta_boxed_loss={proxy.get('delta_boxed_tail_loss', '?')} "
        f"delta_boxed_exact={proxy.get('delta_boxed_tail_exact_rate', '?')}",
        "score_proxy_by_family:",
    ]
    if proxy_by_family:
        for family, values in sorted(proxy_by_family.items()):
            lines.append(
                f"  {family}: rows={values.get('rows', '?')} "
                f"boxed_loss={values.get('boxed_tail_loss', '?')} "
                f"boxed_exact={values.get('boxed_tail_exact_rate', '?')}"
            )
    else:
        lines.append("  missing")
    lines.extend(
        [
            "score_proxy_by_class:",
        ]
    )
    if proxy_by_class:
        for class_name, values in sorted(proxy_by_class.items())[:40]:
            lines.append(
                f"  {class_name}: rows={values.get('rows', '?')} "
                f"boxed_loss={values.get('boxed_tail_loss', '?')} "
                f"boxed_exact={values.get('boxed_tail_exact_rate', '?')}"
            )
        if len(proxy_by_class) > 40:
            lines.append(f"  ... {len(proxy_by_class) - 40} more classes in status json")
    else:
        lines.append("  missing")
    lines.extend(
        [
        "score_trajectory:",
        f"  label={trajectory.get('label', 'missing')} status={trajectory.get('status', 'missing')} "
        f"alignment={trajectory.get('alignment', '?')} target_required={trajectory.get('target_correct_required', '?')} "
        f"gain_needed={trajectory.get('additional_rows_required', '?')}",
        f"  weak_exact_delta={trajectory.get('weak_exact_delta', '?')} "
        f"bit_exact_delta={trajectory.get('bit_exact_delta', '?')} "
        f"equation_exact_delta={trajectory.get('equation_exact_delta', '?')}",
        f"  protected_exact_delta={trajectory.get('protected_exact_delta', '?')} "
        f"overall_exact_delta={trajectory.get('overall_exact_delta', '?')} "
        f"boxed_loss_delta={trajectory.get('boxed_tail_loss_delta', '?')}",
        "last_teach:",
        f"  {state.get('last_teach')}",
        ]
    )
    if alerts:
        lines.append("alerts:")
        for alert in alerts[-5:]:
            lines.append(f"  line {alert.get('line')}: {alert.get('message')}")
    else:
        lines.append("alerts: none")
    return "\n".join(lines)


def write_status(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")

"""T0b core — orquestrador da telemetria de SCORE REAL (geração) para o treino.

Roda a cada N steps DENTRO do treino: para um eval-set fixo, GERA greedy (config oficial),
extrai+verifica com a métrica oficial, e emite KG1_SCORE_LIVE_* + decisão de ABORT por
EXACT-MATCH real (nunca loss). É a régua que impede "loss bom, score ruim".

Design: `generate_fn(prompt_text) -> (raw_output, finish_reason, completion_tokens)` é
INJETÁVEL. No treino usa make_hf_generate_fn(model, tokenizer); no teste usa um stub.
Por isso a LÓGICA é 100% testável em CPU (sem o 30B). A geração real valida no SMOKE.
"""
from __future__ import annotations
import json, os, sys
from typing import Any, Callable, Iterable

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from kg1_score_live import compute_score_telemetry, OFFICIAL_MAX_TOKENS

# suffix oficial (a métrica adiciona ao prompt bare). Fallback se cu indisponível.
OFFICIAL_SUFFIX = "\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`"


def load_jsonl(path: str) -> list[dict[str, Any]]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def render_eval_prompt(tokenizer, bare_prompt: str) -> str:
    """Replica o rendering oficial do eval: prompt bare + suffix -> chat template (abre <think>)."""
    full = bare_prompt if OFFICIAL_SUFFIX in bare_prompt else (bare_prompt + OFFICIAL_SUFFIX)
    msgs = [{"role": "user", "content": full}]
    try:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    except TypeError:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def make_hf_generate_fn(model, tokenizer, max_new_tokens: int = OFFICIAL_MAX_TOKENS,
                        max_model_len: int = 8192):
    """generate_fn real (greedy, config oficial). Usado no treino. NÃO roda em CPU sem o 30B."""
    import torch  # noqa
    def gen(prompt_text: str):
        enc = tokenizer(prompt_text, return_tensors="pt")
        input_ids = enc["input_ids"].to(model.device)
        budget = min(max_new_tokens, max(1, max_model_len - input_ids.shape[1]))
        with torch.no_grad():
            out = model.generate(input_ids, do_sample=False, temperature=None, top_p=None,
                                 max_new_tokens=budget, pad_token_id=tokenizer.eos_token_id)
        gen_ids = out[0][input_ids.shape[1]:]
        raw = tokenizer.decode(gen_ids, skip_special_tokens=False)
        ct = int(gen_ids.shape[0])
        fr = "length" if ct >= budget else "stop"
        return raw, fr, ct
    return gen


def emit(tel: dict[str, Any]) -> str:
    """Imprime os marcadores parseáveis (Use a Cabeça: heartbeat humano + JSON + STATUS)."""
    em = tel.get("exact_match", {}); ab = tel.get("abort", {}); ed = tel.get("exact_delta", {})
    tr = tel.get("truncation", {}); bx = tel.get("boxed", {})
    print("KG1_SCORE_LIVE_JSON " + json.dumps(tel, ensure_ascii=False), flush=True)
    hb = ("KG1_SCORE_LIVE_HEARTBEAT step=" + str(tel.get("step")) + "/" + str(tel.get("max_step")) +
          " EM=" + str(em.get("correct")) + "/" + str(em.get("rows")) +
          "(" + str(em.get("accuracy")) + ") d_overall=" + str(ed.get("overall")) +
          " d_bit=" + str(ed.get("bit_manipulation")) + " d_eq=" + str(ed.get("equation_transform")) +
          " d_protected=" + str(ed.get("protected")) + " trunc=" + str(tr.get("rate")) +
          " box1=" + str(bx.get("exactly_one_boxed_rate")) + " -> " + str(ab.get("decision")))
    print(hb, flush=True)
    print("KG1_SCORE_LIVE_STATUS=" + str(ab.get("decision")), flush=True)
    # cartão didático (Use a Cabeça)
    print("[KG1-TEACH][SCORE_LIVE][" + str(ab.get("decision")) + "]", flush=True)
    print("  O que e: o SCORE de verdade (geracao real), nao loss.", flush=True)
    print("  Como ler: OK=segue; WATCH=falta baseline/observar; ABORT=score real regrediu -> PARAR.", flush=True)
    if ab.get("abort_reasons"):
        print("  Motivos do ABORT: " + "; ".join(str(x) for x in ab["abort_reasons"]), flush=True)
    return str(ab.get("decision"))


def run_score_live_eval(eval_rows: Iterable[dict[str, Any]],
                        generate_fn: Callable[[str], tuple],
                        *,
                        render_fn: Callable[[dict[str, Any]], str],
                        baseline_per_family: dict[str, int] | None = None,
                        baseline_truncation_per_family: dict[str, float] | None = None,
                        step: int | None = None, max_step: int | None = None) -> dict[str, Any]:
    """Gera, mede o score real e emite a telemetria. Retorna a telemetria (com decisão).

    render_fn e OBRIGATORIO (sem fallback mock): no treino use render_eval_prompt(tokenizer, .)
    para render oficial (suffix + chat template + <think>). Passar o prompt bare seria medir errado.
    """
    if render_fn is None:
        raise ValueError("render_fn obrigatorio: use render_eval_prompt real (sem fallback mock).")
    rows = []
    for it in eval_rows:
        raw, fr, ct = generate_fn(render_fn(it))
        rows.append({"id": it.get("id"), "raw_output": raw, "answer": it.get("answer", ""),
                     "family": it.get("family", "unknown"), "finish_reason": fr, "completion_tokens": ct})
    tel = compute_score_telemetry(rows, baseline_per_family=baseline_per_family,
                                  baseline_truncation_per_family=baseline_truncation_per_family,
                                  step=step, max_step=max_step)
    emit(tel)
    return tel

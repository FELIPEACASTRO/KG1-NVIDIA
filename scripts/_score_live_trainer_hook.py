"""REFERÊNCIA de integração do SCORE-LIVE no hf_job_train_v90.py (T0b — hook).

NÃO é importado; é o trecho EXATO a inserir no loop de avaliação do trainer.
Valida no SMOKE (T1) — geração 30B não roda em CPU. Mínimo e cirúrgico (baixo risco).

ONDE inserir: no loop de checkpoints, onde hoje chama evaluate_score_proxy(...)
(perto de EVAL_EVERY_STEPS). MANTER o proxy teacher-forced como sinal barato por step;
ADICIONAR o canário score-real a cada N steps (mais caro).
"""

# === 1) no topo do trainer (imports) ===
# import json
# from kg1_score_live_eval import run_score_live_eval, make_hf_generate_fn
#
# === 2) uma vez, antes do loop (carregar eval-set + baseline 086 per-item gerado no T0c) ===
# SCORE_LIVE_EVAL = [json.loads(l) for l in open(os.environ["KG1_SCORE_LIVE_EVALSET"], encoding="utf-8") if l.strip()]
# _bl = json.load(open(os.environ["KG1_SCORE_LIVE_BASELINE"], encoding="utf-8"))  # baseline_086_per_family.json
# SCORE_LIVE_BASE = _bl.get("baseline_per_family")
# SCORE_LIVE_BASE_TRUNC = _bl.get("baseline_truncation_per_family")
# SCORE_LIVE_EVERY = int(os.environ.get("KG1_SCORE_LIVE_EVERY", "80"))
# from kg1_score_live_eval import render_eval_prompt  # p/ o render_fn real
#
# === 3) dentro do loop de eval (a cada SCORE_LIVE_EVERY steps) ===
def score_live_checkpoint(model, tokenizer, eval_rows, baseline, baseline_trunc,
                          step, max_step, final_only_env):
    from kg1_score_live_eval import run_score_live_eval, make_hf_generate_fn, render_eval_prompt
    gen = make_hf_generate_fn(model, tokenizer)  # greedy, 7680/8192 oficial
    tel = run_score_live_eval(
        eval_rows, gen,
        render_fn=lambda it: render_eval_prompt(tokenizer, it["prompt"]),
        baseline_per_family=baseline,
        baseline_truncation_per_family=baseline_trunc,
        step=step, max_step=max_step,
    )
    decision = tel["abort"]["decision"]
    is_final = (step is not None and max_step is not None and step >= max_step)
    # politica: ABORT intermediario = WATCH (a menos que nao seja FINAL_ONLY); ABORT no final = parar
    hard = (decision == "ABORT") and ((not final_only_env) or is_final)
    if hard:
        raise RuntimeError("KG1_SCORE_LIVE_ABORT real-exact-match regression: "
                           + "; ".join(tel["abort"]["abort_reasons"]))
    return tel

# === 4) PROMOÇÃO (no fim do treino) — usar exact-match REAL, NUNCA loss ===
# Promover o checkpoint só se o ULTIMO score_live_checkpoint deu decision=="OK"
#   (overall>=0 E bit>=0 E eq>=0 E protected>=0 E trunc ok) — o gate full947_089 confirma depois.
#
# === 5) WATCHDOG/parser (kg1_live_log_common.py + kg1_colab_realtime_runner.py) ===
#   reconhecer "KG1_SCORE_LIVE_STATUS=ABORT" -> health=STOP; "=WATCH" -> nunca OK.

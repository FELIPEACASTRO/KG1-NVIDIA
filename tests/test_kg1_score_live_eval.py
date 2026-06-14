"""Teste CPU do orquestrador T0b (sem 30B): stub generate_fn -> valida geração->score->ABORT."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from kg1_score_live_eval import run_score_live_eval

FAMS = ["bit_manipulation", "equation_transform", "gravity_constant",
        "numeral_system", "text_encryption", "unit_conversion"]
# eval-set sintético: 2 itens por família
EVAL = []
for f in FAMS:
    for i in range(2):
        EVAL.append({"id": f + "_" + str(i), "answer": "A" + f[0] + str(i),
                     "family": f, "prompt": "puzzle " + f + " " + str(i)})
# baseline 086 REALISTA: protegidas 100% (2/2); bit/eq 1/2 (têm headroom)
BASE = {"bit_manipulation": 1, "equation_transform": 1,
        "gravity_constant": 2, "numeral_system": 2, "text_encryption": 2, "unit_conversion": 2}

def box(a): return "</think>\n\\boxed{" + a + "}"

def make_stub(outputs):
    def gen(pid):  # render_fn=lambda it: it["id"], então pid é o id
        return outputs[pid]
    return gen

def scenario(label, build_out, expect):
    outs = {}
    for it in EVAL:
        outs[it["id"]] = build_out(it)
    tel = run_score_live_eval(EVAL, make_stub(outs), render_fn=lambda it: it["id"],
                              baseline_per_family=BASE, step=80, max_step=160)
    dec = tel["abort"]["decision"]
    ok = dec == expect
    print("[" + ("PASS" if ok else "FAIL") + "] " + label + " -> " + dec + " (esperado " + expect + ")")
    assert ok, label
    return tel

# CASO 1: melhora (todos corretos) -> overall/weak/protected >=0 -> OK
scenario("melhora_tudo_correto",
         lambda it: (box(it["answer"]), "stop", 50), "OK")

# CASO 2: regressão protegida (1 protegida erra) -> protected<0 -> ABORT
def reg_prot(it):
    if it["id"] == "gravity_constant_0":
        return (box("ERRADO"), "stop", 50)
    return (box(it["answer"]), "stop", 50)
scenario("regressao_protegida", reg_prot, "ABORT")

# CASO 3: truncação alta (bit trunca, sem box) -> trunc>baseline+tol -> ABORT
def trunc_bit(it):
    if it["family"] == "bit_manipulation":
        return ("raciocinio longo sem fim ", "length", 7680)
    return (box(it["answer"]), "stop", 50)
scenario("truncacao_alta", trunc_bit, "ABORT")

# CASO 4: sem baseline -> WATCH (nunca OK cego)
def no_base():
    outs = {it["id"]: (box(it["answer"]), "stop", 50) for it in EVAL}
    tel = run_score_live_eval(EVAL, make_stub(outs), render_fn=lambda it: it["id"],
                              baseline_per_family=None, step=80, max_step=160)
    dec = tel["abort"]["decision"]
    ok = dec == "WATCH" and "baseline_per_item_missing" in tel["abort"].get("blockers", [])
    print("[" + ("PASS" if ok else "FAIL") + "] sem_baseline -> " + dec + " blockers=" + str(tel["abort"].get("blockers")))
    assert ok
no_base()

print("\nTODOS OS TESTES T0b PASSARAM (geração->score real->ABORT validado em CPU via stub).")

"""Build V1244 CoT-safe dataset from validated solver CoT corpora.

Corrige 2 gaps achados no double-check 2026-06-14:
  GAP-LEAK: remove traces cujo puzzle (core do prompt) aparece no full947 (juiz).
  GAP-THINKWRAP: o eval abre <think>; o alvo NAO pode comecar com outro <think>.
                 Remove o "<think>\n" inicial dos CoT (target continua o think aberto).
Tambem: remove eq que falham re-solve (52), valida 1-boxed terminal + answer==gold
(extrator/verify de PARIDADE do publish worktree), separa numeric vs symbolic.
Saida: artifacts/v1244_cot_safe/ + manifest com sha256 + relatorio de validacao.
Sem GPU, sem rede. Determinístico.
"""
import sys, os, json, re, hashlib, csv, collections

ROOT = r"C:\Users\davis\Workspace\KG1 -NVIDIA"
PUB = os.path.join(ROOT, "artifacts", "v1243_final_publish_worktree")
sys.path.insert(0, os.path.join(PUB, "src"))
import competition_utils as cu  # PARIDADE (origin), nao o HEAD bugado

BIT_COT = r"C:\tmp\fase4_bit_solver_traces.jsonl"
EQ_COT = r"C:\tmp\fase4_eq_solver_traces.jsonl"
EQ_FAIL_JSON = r"C:\tmp\eq_crack\agentc_desktest_result.json"
REPLAY = os.path.join(PUB, "artifacts", "v1243_solver_to_lora_graft", "v1243_protected_replay_train.jsonl")
CANON = os.path.join(PUB, "artifacts", "v284_official_gate_worktree", "artifacts",
                     "v1088_unicode_dataset_contract_audit", "hf_cli_download", "runtime_artifacts",
                     "v276_full_eval_bridge", "v276-full947-bridge-20260511T1245Z",
                     "official_train_seed42_stratified10_val.csv")
OUT = os.path.join(ROOT, "artifacts", "v1244_cot_safe")
os.makedirs(OUT, exist_ok=True)
SUFFIX_MARK = "Please put your final answer inside"

def load(p): return [json.loads(l) for l in open(p, encoding="utf-8")]
def norm(s): return re.sub(r"\s+", " ", s).strip().lower()
def core(prompt): return norm(prompt.split(SUFFIX_MARK)[0])
def usr(r): return [m["content"] for m in r["messages"] if m["role"] == "user"][0]
def asst(r): return [m["content"] for m in r["messages"] if m["role"] == "assistant"][0]

def strip_open_think(target):
    # target continua o <think> ja aberto pelo prompt do eval -> remover <think> inicial
    t = target
    if t.startswith("<think>\n"): t = t[len("<think>\n"):]
    elif t.startswith("<think>"): t = t[len("<think>"):]
    return t

# full947 cores (juiz) p/ leak
full_rows = list(csv.DictReader(open(CANON, encoding="utf-8")))
full_cores = set(core(r["prompt"]) for r in full_rows)

_eqj = json.load(open(EQ_FAIL_JSON, encoding="utf-8"))
eq_fail = set(e.split(":")[0].strip() for e in _eqj.get("errors", []))

def process(traces, kind):
    out, dropped = [], collections.Counter()
    for r in traces:
        pid = r.get("id", "?")
        u = usr(r); a = asst(r)
        if SUFFIX_MARK not in u: dropped["no_suffix"] += 1; continue
        if kind == "eq" and pid in eq_fail: dropped["eq_resolve_fail"] += 1; continue
        if core(u) in full_cores: dropped["leak_full947"] += 1; continue
        new_a = strip_open_think(a)
        if new_a.startswith("<think>"): dropped["still_think"] += 1; continue
        if new_a.count("\\boxed{") != 1: dropped["not_one_box"] += 1; continue
        gold = str(r.get("answer", "")).strip()
        extracted = cu.extract_final_answer(new_a) if hasattr(cu, "extract_final_answer") else cu.extract_final_boxed_answer(new_a)
        if extracted != gold or cu.verify_answer(gold, extracted) is not True:
            dropped["roundtrip_fail"] += 1; continue
        if not new_a.rstrip().endswith("}"): dropped["not_terminal_box"] += 1; continue
        rec = {"messages": [{"role": "user", "content": u}, {"role": "assistant", "content": new_a}],
               "answer": gold, "family": ("bit_manipulation" if kind == "bit" else "equation_transform"),
               "id": pid, "subtype": r.get("subtype", ""), "class": r.get("class", ""),
               "v1244_target_encoding": "deductive_cot_continue_think_terminal_boxed",
               "row_loss_weight": 1.0,
               "metadata": {"loss_weight": 1.0, "row_loss_weight": 1.0, "v1243_sampling_weight": 1.0,
                            "source": "v1244_cot_safe", "subtype": r.get("subtype", "")}}
        out.append(rec)
    return out, dropped

bit_clean, bit_drop = process(load(BIT_COT), "bit")
eq_all = load(EQ_COT)
eq_clean, eq_drop = process(eq_all, "eq")
eq_num = [r for r in eq_clean if r["subtype"] == "numeric"]
eq_sym = [r for r in eq_clean if r["subtype"] != "numeric"]
replay = load(REPLAY)

def emit(name, rows):
    p = os.path.join(OUT, name)
    with open(p, "w", encoding="utf-8") as fh:
        for r in rows: fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    return p, h, len(rows)

# GAP-FIX (double-check P4): replay protegido estava 21% < 30% alvo -> oversample p/ ~30%
# (protegidas = ~67% do score; reforco reduz risco de esquecimento catastrofico)
focus_n = len(bit_clean) + len(eq_num)
target_replay = round(0.30 / 0.70 * focus_n)  # ~30% do mix final
replay_os = []
i = 0
while len(replay_os) < target_replay and replay:
    base = replay[i % len(replay)]
    rep = dict(base)
    rep_n = i // len(replay)
    if rep_n > 0:  # ids unicos p/ nao confundir gates de dedup
        rep = json.loads(json.dumps(base))
        rep["id"] = str(base.get("id", "rep")) + "_r" + str(rep_n)
    replay_os.append(rep); i += 1
micro = bit_clean + eq_num + replay_os  # main training file (eq numerico = +1 eq seguro)
files = {
    "v1244_bit_cot_train.jsonl": bit_clean,
    "v1244_eq_numeric_cot_train.jsonl": eq_num,
    "v1244_eq_symbolic_cot_train.jsonl": eq_sym,
    "v1244_protected_replay_train.jsonl": replay,
    "v1244_micro_consolidation_train.jsonl": micro,
}
manifest = {"counts": {}, "sha256": {}, "drops": {"bit": dict(bit_drop), "eq": dict(eq_drop)}}
for name, rows in files.items():
    _, h, n = emit(name, rows)
    manifest["counts"][name] = n; manifest["sha256"][name] = h

# validacao final independente (re-le os arquivos emitidos)
def revalidate(name):
    rows = load(os.path.join(OUT, name)); bad = 0
    for r in rows:
        a = asst(r)
        if a.startswith("<think>") or a.count("\\boxed{") != 1: bad += 1; continue
        ext = cu.extract_final_answer(a) if hasattr(cu, "extract_final_answer") else cu.extract_final_boxed_answer(a)
        if ext != str(r["answer"]).strip() or cu.verify_answer(str(r["answer"]).strip(), ext) is not True: bad += 1
        if core(usr(r)) in full_cores: bad += 1
    return bad
manifest["revalidation_bad_rows"] = {n: revalidate(n) for n in files}
json.dump(manifest, open(os.path.join(OUT, "v1244_manifest.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

print("=== V1244 CoT-safe build ===")
print("bit-CoT:   289 ->", len(bit_clean), "clean | drops:", dict(bit_drop))
print("eq-CoT:    769 ->", len(eq_clean), "clean (num", len(eq_num), "+ sym", len(eq_sym), ") | drops:", dict(eq_drop))
print("replay:   ", len(replay))
print("MICRO (bit+eq_num+replay):", len(micro))
print("revalidation bad rows:", manifest["revalidation_bad_rows"])
print("sha256 micro:", manifest["sha256"]["v1244_micro_consolidation_train.jsonl"][:16])
print("OUT:", OUT)

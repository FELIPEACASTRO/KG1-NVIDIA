"""T0c — gera o baseline 086 per-item no eval-set fixo (GPU/Colab).

Produz o CSV per-item (id, family, answer, raw_output, extracted, correct, finish_reason,
completion_tokens, boxed_status) e o baseline_per_family que o SCORE-LIVE precisa para os deltas.
Roda no Colab Pro (carrega Nemotron 30B + adapter 086). Greedy, config oficial.
CPU: só importa/py_compile; a geração exige GPU.
"""
from __future__ import annotations
import json, os, sys, csv

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from kg1_score_live_eval import render_eval_prompt, make_hf_generate_fn, OFFICIAL_SUFFIX
from kg1_score_live import extract_final_answer, verify, is_truncated, box_status

BASE_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
MODEL_REV = "cbd3fa9f933d55ef16a84236559f4ee2a0526848"
ADAPTER_086 = "felipesp1983/kg1-recovered-v291-v290-checkpoint6-submit086"
ADAPTER_REV = "f4134a6d223249d27be2f1c5d94ed59d118d1ce5"


def main(eval_path: str, out_csv: str, out_family_json: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    eval_rows = [json.loads(l) for l in open(eval_path, encoding="utf-8") if l.strip()]
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, revision=MODEL_REV)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, revision=MODEL_REV,
                                                 torch_dtype=torch.bfloat16, device_map="auto",
                                                 attn_implementation="eager", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, ADAPTER_086, revision=ADAPTER_REV)
    model.eval()
    gen = make_hf_generate_fn(model, tok)
    import collections
    per_fam_total = collections.Counter(); per_fam_ok = collections.Counter(); per_fam_trunc = collections.Counter()
    rows_out = []
    for it in eval_rows:
        prompt = render_eval_prompt(tok, it["prompt"])
        raw, fr, ct = gen(prompt)
        ext = extract_final_answer(raw); ok = verify(it["answer"], ext)
        tr = is_truncated(fr, ct); bs = box_status(raw)
        f = it["family"]; per_fam_total[f] += 1; per_fam_ok[f] += int(ok); per_fam_trunc[f] += int(tr)
        rows_out.append({"id": it["id"], "family": f, "answer": it["answer"], "raw_output": raw,
                         "extracted": ext, "correct": int(ok), "finish_reason": fr,
                         "completion_tokens": ct, "boxed_status": bs})
        print("KG1_BASE086 " + it["id"] + " " + f + " correct=" + str(int(ok)) + " trunc=" + str(int(tr)), flush=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys())); w.writeheader(); w.writerows(rows_out)
    baseline_per_family = dict(per_fam_ok)
    baseline_trunc_per_family = {f: round(per_fam_trunc[f] / per_fam_total[f], 6) for f in per_fam_total}
    json.dump({"baseline_per_family": baseline_per_family,
               "baseline_truncation_per_family": baseline_trunc_per_family,
               "total_correct": sum(per_fam_ok.values()), "total_rows": sum(per_fam_total.values())},
              open(out_family_json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("BASE086 total=" + str(sum(per_fam_ok.values())) + "/" + str(sum(per_fam_total.values())),
          "| per_family=", baseline_per_family, flush=True)


if __name__ == "__main__":
    a = sys.argv
    main(a[1] if len(a) > 1 else "artifacts/v1244_cot_safe/v1244_scorelive_evalset_170.jsonl",
         a[2] if len(a) > 2 else "artifacts/v1244_cot_safe/baseline_086_peritem_170.csv",
         a[3] if len(a) > 3 else "artifacts/v1244_cot_safe/baseline_086_per_family.json")

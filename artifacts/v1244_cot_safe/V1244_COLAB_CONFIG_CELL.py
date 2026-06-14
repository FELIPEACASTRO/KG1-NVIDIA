# =====================================================================
# V1244 CoT-safe — CÉLULA DE CONFIG (Claude) — SEM STUB/MOCK
# Cole ACIMA da célula do launcher e rode primeiro. Pré-req: HF_KEY (escrita) no Colab Secret.
# Pré-req de pacote: o launch-pack precisa conter os arquivos V1244 (artifacts/v1244_cot_safe/*)
#   e o trainer hf_job_train_v90.py com o hook do SCORE-LIVE aplicado (_score_live_trainer_hook.py).
# =====================================================================
import os

# ---- dados V1244 (REAIS, validados; bit-CoT 262 + eq_num 423 + replay 294 = 979) ----
os.environ["KG1_V1243_PHASE"] = "v1244_micro_consolidation"
os.environ["DATA_FILE"]       = "artifacts/v1244_cot_safe/v1244_micro_consolidation_train.jsonl"
os.environ["VAL_FILE"]        = "artifacts/v1244_cot_safe/v1244_scorelive_evalset_170.jsonl"

# ---- SCORE-LIVE: medir ACC REAL (geração), nunca loss; abort por-família ----
os.environ["KG1_SCORE_LIVE_EVALSET"]  = "artifacts/v1244_cot_safe/v1244_scorelive_evalset_170.jsonl"
os.environ["KG1_SCORE_LIVE_BASELINE"] = "artifacts/v1244_cot_safe/baseline_086_per_family.json"  # gerar no T0c
os.environ["KG1_SCORE_LIVE_EVERY"]    = "80"

# ---- treino real (flags) ----
os.environ["KG1_ACCEPT_GPU_SPEND"]         = "1"
os.environ["KG1_V1243_RUN_MODEL_DRYRUN"]   = "1"
os.environ["KG1_V1243_RUN_TRAIN"]          = "1"
os.environ["KG1_V1243_REQUIRE_REAL_TRAIN"] = "1"
os.environ["KG1_V1243_ALLOW_MAMBA_SOURCE_BUILD"] = "1"

# ---- pipeline corrigido (achados validados) ----
os.environ["MAX_STEPS"]            = "120"     # smoke real 60-120; subir p/ 160-200 se sinais bons
os.environ["WARMUP_RATIO"]         = "0.06"    # warmup real (trainer hoje é decaimento puro)
os.environ["BOXED_PAYLOAD_LOSS_WEIGHT"] = "1.0"  # CoT: loss uniforme no completion (ok)
os.environ["LEARNING_RATE"]        = "3e-6"
os.environ["FINAL_LEARNING_RATE"]  = "1e-6"
os.environ["REQUIRE_SCORE_TRAJECTORY_PASS"] = "1"

# ---- adapter base 086 pinado ----
os.environ["INIT_ADAPTER_REPO"]     = "felipesp1983/kg1-recovered-v291-v290-checkpoint6-submit086"
os.environ["INIT_ADAPTER_REVISION"] = "f4134a6d223249d27be2f1c5d94ed59d118d1ce5"
os.environ["OUTPUT_REPO"]           = "felipesp1983/kg1-v1244-cot-candidate"
os.environ["UPLOAD_TO_HF"]          = "1"

# ---- logs realtime ----
os.environ["KG1_LIVE_LOG_HF_REPO"]        = "felipesp1983/kg1-live-logs"
os.environ["KG1_REQUIRE_LIVE_LOG_UPLOAD"] = "1"

print("V1244 CONFIG (Claude): phase=v1244_micro · SCORE-LIVE ON (ACC real, abort por-familia) · warmup 0.06 · sem stub", flush=True)
print("ATENCAO: rode T0c (gen baseline 086 per-item) ANTES, senao SCORE-LIVE fica WATCH (nao OK cego).", flush=True)

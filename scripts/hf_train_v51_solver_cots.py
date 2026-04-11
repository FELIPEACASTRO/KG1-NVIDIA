"""
KG1 Nemotron - v51 SOLVER COTS (v30 base + 9500 solver-enhanced examples)
=========================================================================
BASEADO 100% no v30 que scorou 0.68 (config identica).

MUDANCAS vs v30:
1. 9500 exemplos COM CoTs do solver (vs 5000 sem CoT)
2. CoTs para bit/gravity/unit/numeral geradas por solver deterministico
3. CoTs sinteticas para cipher/equation (com resposta correta)
4. Formato: <think>cot</think>\\boxed{answer} (ensina reasoning)
5. Smart-strip no submit (mantém shared_experts = +0.06)

IDENTICO ao v30 em config de treino:
- r=32, alpha=16, lr=5e-5, grad_accum=8, cosine, warmup=0.05
- SFTTrainer, packing=False, bf16=True, max_grad_norm=1.0
- device_map={"": 0}, use_reentrant=False

Score esperado: 0.74-0.80 (vs v30's 0.68)
"""
import subprocess, sys, os, json, random, time, zipfile, shutil, re
from datetime import datetime, timezone
from collections import Counter

# ============================================================
# DEPS
# ============================================================
print("=== Checking dependencies ===")
print("  [COLAB] Dependencies installed by Cell 1")

try:
    import mamba_ssm
    print(f"  mamba-ssm {mamba_ssm.__version__} OK")
except ImportError:
    print("  mamba-ssm not found, attempting install...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q",
             "--root-user-action=ignore", "causal-conv1d", "mamba-ssm"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        import mamba_ssm
        print(f"  mamba-ssm {mamba_ssm.__version__} installed")
    except Exception as e:
        print(f"  WARNING: mamba-ssm install failed: {e}")

import torch
import pandas as pd
from huggingface_hub import HfApi, login, hf_hub_download

# ============================================================
# MONKEY-PATCH
# ============================================================
try:
    from transformers.utils.import_utils import is_flash_attn_greater_or_equal_2_10
except ImportError:
    import transformers.utils.import_utils as _tiu
    _tiu.is_flash_attn_greater_or_equal_2_10 = lambda: False
    print("  Patched: is_flash_attn_greater_or_equal_2_10 stub injected")

# ============================================================
# AUTH + CONFIG
# ============================================================
def _get_secret(*names):
    try:
        from google.colab import userdata
        for n in names:
            v = userdata.get(n)
            if v: return v
    except Exception:
        pass
    for n in names:
        v = os.environ.get(n)
        if v: return v
    return ""

HF_TOKEN = _get_secret("HF_KEY", "HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)
    os.environ["HF_TOKEN"] = HF_TOKEN
    print("HF login OK")

KAGGLE_USERNAME = _get_secret("KAGGLE_USERNAME") or "felipe1983"
KAGGLE_KEY = _get_secret("KAGGLE_KEY")
if KAGGLE_KEY:
    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
    kpath = os.path.expanduser("~/.kaggle/kaggle.json")
    with open(kpath, "w") as f:
        json.dump({"username": KAGGLE_USERNAME, "key": KAGGLE_KEY}, f)
    os.chmod(kpath, 0o600)
    print(f"Kaggle: {KAGGLE_USERNAME}")

api = HfApi()

DATA_REPO = "felipesp1983/kg1-nemotron-training"
OUTPUT_REPO = "felipesp1983/kg1-nemotron-lora-v51-solver-cots"
MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
COMPETITION = "nvidia-nemotron-model-reasoning-challenge"

N_EXAMPLES = int(os.environ.get("N_EXAMPLES", "5000"))
N_EPOCHS = int(os.environ.get("N_EPOCHS", "2"))
SUBMIT_STEPS = [200, 400, 600, 800, 1000, 1200]

CONFIG = {
    "lora_rank": 32,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": "all-linear",
    "learning_rate": 5e-5,
    "per_device_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "max_length": 1024,
    "warmup_ratio": 0.05,
    "weight_decay": 0.01,
    "lr_scheduler": "cosine",
    "optim": "adamw_torch",
    "output_dir": "/tmp/kg1_output/v51",
}

print(f"\nPyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM: {vram:.1f} GB")

# ============================================================
# TRITON PATCH
# ============================================================
try:
    for ptxas_src in ["/usr/local/cuda-12.8/bin/ptxas", "/usr/local/cuda/bin/ptxas"]:
        if os.path.exists(ptxas_src):
            target = os.path.join(os.path.dirname(shutil.which("python") or "/usr/bin/python"), "ptxas")
            if not os.path.exists(target):
                shutil.copy2(ptxas_src, target)
                print(f"Triton ptxas patched: {target}")
            break
except Exception:
    pass

# ============================================================
# LOAD DATA — v51: Use solver-enhanced CoTs
# ============================================================
print("\n=== Loading v51 solver-enhanced data ===")

# Try to download from HF first, fallback to local
data_loaded = False
for data_file in ["data/sft_v51_complete.jsonl", "data/sft_v51_final.jsonl"]:
    try:
        hf_hub_download(repo_id=DATA_REPO, repo_type="dataset",
                        filename=data_file, local_dir="/tmp/kg1_data")
        data_path = f"/tmp/kg1_data/{data_file}"
        data_loaded = True
        print(f"Downloaded {data_file} from HF")
        break
    except Exception:
        continue

if not data_loaded:
    # Fallback: download train.csv and use raw answers (v30 style)
    print("  CoT data not on HF, falling back to train.csv (v30 style)")
    hf_hub_download(repo_id=DATA_REPO, repo_type="dataset",
                    filename="data/train.csv", local_dir="/tmp/kg1_data")
    data_path = None

# Parse data
all_examples = []
if data_path and os.path.exists(data_path):
    with open(data_path) as f:
        for line in f:
            item = json.loads(line)
            all_examples.append(item)
    print(f"Loaded {len(all_examples)} solver-enhanced examples")
else:
    # v30-style fallback
    train_df = pd.read_csv("/tmp/kg1_data/data/train.csv")
    for _, row in train_df.iterrows():
        all_examples.append({
            "prompt": row["prompt"] + "\nPut your final answer inside \\boxed{}.",
            "completion": f"\\boxed{{{row['answer']}}}",
            "family": "unknown",
        })
    print(f"Loaded {len(all_examples)} raw examples (v30 fallback)")

# Classify families
def classify(text):
    p = text.lower()
    if "bit manipulation" in p: return "bit"
    if "gravitational" in p: return "grav"
    if "unit conversion" in p or "measurement" in p: return "unit"
    if "numeral" in p: return "num"
    if "encryption" in p: return "enc"
    if "transformation" in p: return "eq"
    return "other"

# ============================================================
# STRATIFIED SAMPLING (konbu17-style type-weighted)
# ============================================================
print(f"\n=== Preparing {N_EXAMPLES} examples (type-weighted) ===")
random.seed(42)

# Group by family
by_family = {}
for ex in all_examples:
    fam = ex.get("family") or classify(ex.get("prompt", ""))
    if fam not in by_family:
        by_family[fam] = []
    by_family[fam].append(ex)

# Type-weighted allocation (oversample hard families)
shares = {"grav": 1.0, "unit": 1.0, "num": 1.0, "enc": 1.0, "bit": 1.5, "eq": 2.5}
total_shares = sum(shares.values())
base_n = N_EXAMPLES / total_shares
targets = {fam: int(base_n * mult) for fam, mult in shares.items()}
diff = N_EXAMPLES - sum(targets.values())
targets["eq"] += diff

print("Target allocation:")
for fam in ["eq", "bit", "grav", "unit", "num", "enc"]:
    avail = len(by_family.get(fam, []))
    print(f"  {fam}: {targets.get(fam, 0)} (available: {avail})")

# Sample
examples = []
for fam, n_want in targets.items():
    pool = by_family.get(fam, [])
    if not pool:
        continue
    if n_want <= len(pool):
        selected = random.sample(pool, n_want)
    else:
        selected = pool + random.choices(pool, k=n_want - len(pool))
    examples.extend(selected)

random.shuffle(examples)
examples = examples[:N_EXAMPLES]

# Format for SFTTrainer
formatted = []
for ex in examples:
    prompt = ex.get("prompt", "")
    completion = ex.get("completion", "")
    formatted.append({
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ]
    })

fam_counts = Counter(classify(e["messages"][0]["content"]) for e in formatted)
print(f"\nFinal dataset: {len(formatted)} examples")
for fam, cnt in sorted(fam_counts.items()):
    print(f"  {fam}: {cnt} ({cnt/len(formatted)*100:.1f}%)")

# ============================================================
# LOAD MODEL
# ============================================================
print("\n=== Loading model (BF16) ===")
from transformers import AutoModelForCausalLM, AutoTokenizer

_gpu_cap = float(f"{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}")
IS_BLACKWELL = _gpu_cap >= 10.0

_model_kwargs = dict(
    device_map={"": 0},
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

if IS_BLACKWELL:
    from transformers import AutoConfig
    _cfg = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if hasattr(_cfg, "use_mamba_kernels"):
        _cfg.use_mamba_kernels = False
        _model_kwargs["config"] = _cfg
        print(f"  [BLACKWELL sm_{int(_gpu_cap*10)}] Mamba CUDA kernels DISABLED")

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **_model_kwargs)

fp_count = 0
for module in model.modules():
    if hasattr(module, "is_fast_path_available"):
        module.is_fast_path_available = False
        fp_count += 1
print(f"Model: {model.num_parameters()/1e9:.1f}B params, fast path disabled ({fp_count})")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"Tokenizer: vocab={tokenizer.vocab_size}")

# ============================================================
# APPLY LORA
# ============================================================
print(f"\n=== Applying LoRA (r={CONFIG['lora_rank']}, alpha={CONFIG['lora_alpha']}) ===")
from peft import LoraConfig, get_peft_model

model.enable_input_require_grads()
lora_config = LoraConfig(
    r=CONFIG["lora_rank"],
    lora_alpha=CONFIG["lora_alpha"],
    lora_dropout=CONFIG["lora_dropout"],
    target_modules=CONFIG["target_modules"],
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ============================================================
# PREPARE DATASET
# ============================================================
print("\n=== Preparing tokenized dataset ===")
from datasets import Dataset

texts = []
for ex in formatted:
    text = tokenizer.apply_chat_template(
        ex["messages"], tokenize=False, add_generation_prompt=False,
    )
    texts.append(text)

ds = Dataset.from_dict({"text": texts})
print(f"Dataset: {len(ds)} examples")

sample_lens = [len(tokenizer(t)["input_ids"]) for t in texts[:100]]
print(f"Token lengths (100 samples): min={min(sample_lens)}, max={max(sample_lens)}, mean={sum(sample_lens)/len(sample_lens):.0f}")

# ============================================================
# SUBMISSION FUNCTIONS
# ============================================================
def create_submission_zip(adapter_dir, zip_path="/tmp/kg1_submit/submission.zip"):
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in ["adapter_config.json", "adapter_model.safetensors"]:
            fpath = os.path.join(adapter_dir, fname)
            if os.path.exists(fpath):
                zf.write(fpath, fname)
                print(f"  zip: {fname} ({os.path.getsize(fpath)/1e6:.1f} MB)")
    print(f"  zip total: {os.path.getsize(zip_path)/1e6:.1f} MB")
    return zip_path

# ============================================================
# TRAINING
# ============================================================
print("\n=== Training v51 (SFTTrainer) ===")
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback

os.makedirs(CONFIG["output_dir"], exist_ok=True)

training_args = SFTConfig(
    output_dir=CONFIG["output_dir"],
    dataset_text_field="text",
    max_length=CONFIG["max_length"],
    packing=False,
    num_train_epochs=N_EPOCHS,
    per_device_train_batch_size=CONFIG["per_device_batch_size"],
    gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
    learning_rate=CONFIG["learning_rate"],
    warmup_ratio=CONFIG["warmup_ratio"],
    weight_decay=CONFIG["weight_decay"],
    lr_scheduler_type=CONFIG["lr_scheduler"],
    optim=CONFIG["optim"],
    bf16=True,
    logging_steps=5,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=15,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    report_to="none",
    dataloader_num_workers=0,
    max_grad_norm=1.0,
)

class AutoSubmitCallback(TrainerCallback):
    def __init__(self, repo_id, submit_steps):
        self.repo_id = repo_id
        self.submit_steps = set(submit_steps)
        self.submitted = set()
        self.api = HfApi(token=HF_TOKEN) if HF_TOKEN else HfApi()
        try:
            self.api.create_repo(repo_id, private=True, exist_ok=True)
        except Exception:
            pass

    def on_save(self, args, state, control, **kwargs):
        import glob as g
        step = state.global_step
        loss_val = "N/A"
        if state.log_history:
            for entry in reversed(state.log_history):
                if "loss" in entry:
                    loss_val = entry["loss"]
                    break

        ckpts = sorted(g.glob(f"{args.output_dir}/checkpoint-*"))
        if not ckpts:
            return
        ckpt_dir = ckpts[-1]

        try:
            self.api.upload_folder(
                folder_path=ckpt_dir, path_in_repo=f"checkpoint-{step}",
                repo_id=self.repo_id,
                commit_message=f"Step {step} | Loss {loss_val} | Epoch {state.epoch:.2f}",
            )
            print(f"\n>>> HF upload OK: step {step}, loss={loss_val}")
        except Exception as e:
            print(f"\n>>> HF upload FAILED: {e}")

        if step in self.submit_steps and step not in self.submitted:
            print(f"  [INFO] Step {step} uploaded. Submit manually with smart_strip_submit.py")
            self.submitted.add(step)

    def on_log(self, args, state, control, logs=None, **kwargs):
        import math
        if not logs:
            return
        loss = logs.get("loss", 0)

        # NaN/Inf guard
        if isinstance(loss, float) and (math.isnan(loss) or math.isinf(loss)):
            print(f"\n!!! CRITICAL: Loss NaN/Inf at step {state.global_step}")
            control.should_training_stop = True
            return

        if isinstance(loss, (int, float)) and loss > 30.0 and state.global_step > 5:
            print(f"\n!!! CRITICAL: Loss explosion {loss:.2f} at step {state.global_step}")
            control.should_training_stop = True
            return

        if state.global_step == 10:
            if loss > 8.0:
                print(f"\n!!! ALERT: Loss at step 10 = {loss:.2f} (very high!)")
            elif loss > 5.0:
                print(f"\n! WARNING: Loss at step 10 = {loss:.2f} (elevated)")
            else:
                print(f"\n>>> Loss at step 10 = {loss:.2f} (good)")

trainer = SFTTrainer(
    model=model,
    train_dataset=ds,
    processing_class=tokenizer,
    args=training_args,
    callbacks=[AutoSubmitCallback(OUTPUT_REPO, SUBMIT_STEPS)],
)

total_steps = (len(ds) // CONFIG["gradient_accumulation_steps"]) * N_EPOCHS
est_h = total_steps * 55 / 3600
print(f"Config: {len(formatted)} examples, {N_EPOCHS} epochs, LR={CONFIG['learning_rate']}")
print(f"LoRA: r={CONFIG['lora_rank']}, alpha={CONFIG['lora_alpha']}")
print(f"Steps: ~{total_steps}, Est time: ~{est_h:.1f}h")
print(f"Submit steps: {sorted(SUBMIT_STEPS)}")

start = time.time()
try:
    trainer.train()
except Exception as e:
    print(f"\n!!! Training error: {e}")
    try:
        model.save_pretrained(CONFIG["output_dir"])
        tokenizer.save_pretrained(CONFIG["output_dir"])
        api.upload_folder(folder_path=CONFIG["output_dir"], repo_id=OUTPUT_REPO,
                         path_in_repo="emergency",
                         commit_message=f"Emergency: {str(e)[:80]}")
    except Exception:
        pass

elapsed = time.time() - start
print(f"\nTraining complete: {elapsed/3600:.2f}h")

# ============================================================
# SAVE FINAL
# ============================================================
print("\n=== Saving final adapter ===")
model.save_pretrained(CONFIG["output_dir"])
tokenizer.save_pretrained(CONFIG["output_dir"])

final_loss = "N/A"
if trainer.state.log_history:
    for entry in reversed(trainer.state.log_history):
        if "loss" in entry:
            final_loss = entry["loss"]
            break

status = {
    "version": "v51_solver_cots",
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "examples": len(formatted),
    "epochs": N_EPOCHS,
    "lr": CONFIG["learning_rate"],
    "lora_rank": CONFIG["lora_rank"],
    "lora_alpha": CONFIG["lora_alpha"],
    "max_length": CONFIG["max_length"],
    "grad_accum": CONFIG["gradient_accumulation_steps"],
    "training_time_h": elapsed / 3600,
    "final_loss": final_loss,
    "total_steps": trainer.state.global_step,
    "family_distribution": dict(fam_counts),
    "based_on": "v30_perfected (scored 0.68)",
    "changes_from_v30": "9500 solver-enhanced examples with CoTs, same training config",
}
with open(f"{CONFIG['output_dir']}/adapter_status.json", "w") as f:
    json.dump(status, f, indent=2)

print("\n=== Uploading final to HF ===")
try:
    api.create_repo(OUTPUT_REPO, private=True, exist_ok=True)
    api.upload_folder(
        folder_path=CONFIG["output_dir"], repo_id=OUTPUT_REPO,
        commit_message=f"v51 FINAL: {len(formatted)}ex, {N_EPOCHS}ep, loss={final_loss}",
    )
    print(f"Uploaded: https://huggingface.co/{OUTPUT_REPO}")
except Exception as e:
    print(f"Upload failed: {e}")

print(f"\n{'='*60}")
print(f"  v51 TRAINING COMPLETE")
print(f"  Examples: {len(formatted)} | Epochs: {N_EPOCHS}")
print(f"  Final loss: {final_loss}")
print(f"  Steps: {trainer.state.global_step}")
print(f"  Time: {elapsed/3600:.2f}h")
print(f"  Submit step 400 with: python scripts/smart_strip_submit.py --adapter-dir <ckpt> --mode smart-strip --submit")
print(f"{'='*60}")

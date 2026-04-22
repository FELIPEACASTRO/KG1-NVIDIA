# ============================================================
# V77 PATCH v2: Section 8+9+10+11 para TRL 0.25.1
# Modelo ja carregado (nao recarregar).
#
# Bug #1: DataCollatorForCompletionOnlyLM removido do top-level TRL 0.25.1
# Bug #2: assistant_only_loss=True exige conversational format (messages=[...]),
#         incompativel com flat text + enable_thinking pre-aplicado.
#
# Fix: custom CompletionOnlyCollator inline que replica a logica da
#      DataCollatorForCompletionOnlyLM (subsequence match em token ids,
#      mask ate o final do response_template).
# ============================================================
import os, sys, json, subprocess, shutil, gc, time, datetime, re
from pathlib import Path

# Reutiliza START_TIME, CFG, tok, model, train_path, eval_path, HF_TOKEN,
# GDRIVE_MOUNTED ja no memory do kernel.
if "START_TIME" not in globals():
    START_TIME = time.time()


def log(m):
    e = int(time.time() - START_TIME)
    h, rem = divmod(e, 3600)
    mi, s = divmod(rem, 60)
    print(f"[{h:02d}:{mi:02d}:{s:02d}] {m}", flush=True)


def section(t):
    log("=" * 60)
    log(t)
    log("=" * 60)


import torch
gc.collect()
torch.cuda.empty_cache()

section("SECTION 8 (PATCHED v2): Full training V77 via custom CompletionOnlyCollator")
from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback
from datasets import load_dataset
from src.losses.max_min_logprob import max_min_logprob_loss

ds_train = load_dataset("json", data_files=str(train_path), split="train")
ds_eval = load_dataset("json", data_files=str(eval_path), split="train")


def format_example(ex):
    msgs = [
        {"role": "user", "content": ex["user"]},
        {"role": "assistant", "content": ex["assistant"]},
    ]
    try:
        text = tok.apply_chat_template(
            msgs, tokenize=False, enable_thinking=CFG.enable_thinking
        )
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False)
    return {"text": text}


ds_train = ds_train.map(format_example, remove_columns=ds_train.column_names)
ds_eval = ds_eval.map(format_example, remove_columns=ds_eval.column_names)
tok.model_max_length = CFG.max_length


# ============================================================
# Custom CompletionOnlyCollator (replaces TRL 0.25 removed class)
# Same logic as DataCollatorForCompletionOnlyLM:
# - Tokenize text
# - Find response_template subsequence in token ids
# - Mask everything up to end of template with -100
# ============================================================
class CompletionOnlyCollator:
    def __init__(self, tokenizer, response_template="<|im_start|>assistant"):
        self.tok = tokenizer
        self.template_ids = tokenizer.encode(
            response_template, add_special_tokens=False
        )
        self.pad_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )
        log(f"  CompletionOnlyCollator: template='{response_template}' "
            f"-> {len(self.template_ids)} ids {self.template_ids}")

    def __call__(self, features):
        # features: list of {'text': str} after dataset map
        # or list of already-tokenized {'input_ids': ...}
        batch_ids = []
        batch_attn = []
        for f in features:
            if "input_ids" in f:
                ids = (
                    f["input_ids"].tolist()
                    if hasattr(f["input_ids"], "tolist")
                    else list(f["input_ids"])
                )
                attn = (
                    f.get("attention_mask", [1] * len(ids))
                )
                if hasattr(attn, "tolist"):
                    attn = attn.tolist()
            else:
                enc = self.tok(
                    f["text"],
                    truncation=True,
                    max_length=CFG.max_length,
                    add_special_tokens=False,
                )
                ids = enc["input_ids"]
                attn = enc["attention_mask"]
            batch_ids.append(ids)
            batch_attn.append(attn)

        # Find ml for padding
        ml = max(len(ids) for ids in batch_ids)

        tl = len(self.template_ids)
        out_ids, out_labels, out_attn = [], [], []
        n_matched = 0
        for ids, attn in zip(batch_ids, batch_attn):
            labels = list(ids)
            # Find first occurrence of template in ids
            match_idx = -1
            for i in range(len(ids) - tl + 1):
                if ids[i : i + tl] == self.template_ids:
                    match_idx = i
                    break
            if match_idx >= 0:
                # Mask everything including the template itself
                for j in range(match_idx + tl):
                    labels[j] = -100
                n_matched += 1
            else:
                # Fallback: mask user+system half (first 30% as heuristic)
                cut = min(len(ids) // 3, 100)
                for j in range(cut):
                    labels[j] = -100

            # Pad to ml
            pad_n = ml - len(ids)
            ids_p = ids + [self.pad_id] * pad_n
            labels_p = labels + [-100] * pad_n
            attn_p = attn + [0] * pad_n
            out_ids.append(ids_p)
            out_labels.append(labels_p)
            out_attn.append(attn_p)

        # Periodic diagnostic on first batch (via class var)
        if not getattr(self, "_logged_once", False):
            self._logged_once = True
            log(f"  Collator first batch: {n_matched}/{len(features)} "
                f"samples matched template")
            nonmask = sum(1 for row in out_labels for t in row if t != -100)
            total = sum(len(row) for row in out_labels)
            pct = 100 * nonmask / max(1, total)
            log(f"  Collator diagnostic: {nonmask}/{total} tokens non-masked "
                f"({pct:.1f}%)")
            if pct < 1:
                log("  CRITICAL: <1% tokens unmasked - loss will be ~0")
            elif pct < 10:
                log("  WARN: <10% unmasked - check template match")
            else:
                log("  [OK] masking looks healthy")

        return {
            "input_ids": torch.tensor(out_ids, dtype=torch.long),
            "labels": torch.tensor(out_labels, dtype=torch.long),
            "attention_mask": torch.tensor(out_attn, dtype=torch.long),
        }


collator = CompletionOnlyCollator(tok, response_template="<|im_start|>assistant")
log("Using custom CompletionOnlyCollator (TRL 0.25 compat)")

sft_args = SFTConfig(
    output_dir=CFG.output_dir,
    per_device_train_batch_size=CFG.per_device_batch,
    per_device_eval_batch_size=CFG.per_device_batch,
    gradient_accumulation_steps=CFG.grad_accum,
    num_train_epochs=CFG.epochs,
    learning_rate=CFG.learning_rate,
    lr_scheduler_type=CFG.lr_scheduler,
    warmup_ratio=CFG.warmup_ratio,
    max_grad_norm=CFG.grad_clip,
    bf16=CFG.bf16,
    logging_steps=CFG.logging_steps,
    save_steps=CFG.save_steps,
    eval_strategy="steps",
    eval_steps=CFG.eval_steps,
    save_total_limit=CFG.save_total_limit,
    optim=CFG.optimizer,
    packing=False,
    report_to=[],
    gradient_checkpointing=CFG.use_gradient_checkpointing,
    dataset_text_field="text",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    # NO assistant_only_loss - our collator does masking manually
)


class MaxMinSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._nan_count = 0
        self._use_ce_permanent = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits
        step = int(self.state.global_step)
        ce_loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100,
        )
        use_ce = (
            self._use_ce_permanent
            or CFG.loss_type == "ce"
            or (CFG.loss_type == "max_min_warmup_ce" and step < CFG.max_min_warmup_steps)
        )
        if use_ce:
            loss = ce_loss
        else:
            try:
                mm_loss = max_min_logprob_loss(logits, labels)
                if torch.isnan(mm_loss) or torch.isinf(mm_loss):
                    self._nan_count += 1
                    print(f"WARN NaN max-min step {step} #{self._nan_count}", flush=True)
                    if self._nan_count >= 3:
                        self._use_ce_permanent = True
                        print("CRITICAL: CE permanent", flush=True)
                    loss = ce_loss
                else:
                    loss = mm_loss
                    self._nan_count = max(0, self._nan_count - 1)
            except Exception as e:
                print(f"WARN max-min exc: {e}", flush=True)
                loss = ce_loss
        if torch.isnan(loss) or torch.isinf(loss):
            loss = ce_loss
        return (loss, outputs) if return_outputs else loss


trainer = MaxMinSFTTrainer(
    model=model,
    args=sft_args,
    train_dataset=ds_train,
    eval_dataset=ds_eval,
    processing_class=tok,
    data_collator=collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=CFG.early_stopping_patience)],
)
log("Training START V77 (custom CompletionOnlyCollator, EarlyStop patience=3)")
log("Target: train_loss 0.30-0.80 final. If <0.01 = overfit.")
trainer.train()
log("Training complete (best checkpoint loaded)")

out_dir = Path(CFG.output_dir)
trainer.save_model(CFG.output_dir)
tok.save_pretrained(CFG.output_dir)

section("SECTION 9: Save + upload")
from huggingface_hub import HfApi, upload_folder

req = ["adapter_config.json", "adapter_model.safetensors"]
missing_a = [f for f in req if not (out_dir / f).exists()]
assert not missing_a, f"missing: {missing_a}"
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
if GDRIVE_MOUNTED:
    gdrive_dest = Path(CFG.gdrive_checkpoint) / f"{CFG.run_tag}_{ts}"
    try:
        gdrive_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(out_dir, gdrive_dest, dirs_exist_ok=True)
        log(f"GDrive: {gdrive_dest}")
    except Exception as e:
        log(f"WARN: {e}")
api = HfApi(token=HF_TOKEN)
try:
    api.create_repo(CFG.hf_upload_repo, private=True, exist_ok=True)
    upload_folder(
        repo_id=CFG.hf_upload_repo,
        folder_path=str(out_dir),
        allow_patterns=["adapter_*", "tokenizer*", "special_tokens*", "config.json"],
        token=HF_TOKEN,
    )
    log(f"HF: {CFG.hf_upload_repo}")
except Exception as e:
    log(f"WARN: {e}")

section("SECTION 10: Local eval + gate")
local_score_script = Path("/content/kg1/scripts/local_score.py")
eval_csv = Path(CFG.output_dir) / "local_eval.csv"
cmd = [
    sys.executable, str(local_score_script),
    "--adapter", str(out_dir),
    "--n-samples", str(CFG.eval_holdout_size),
    "--output-csv", str(eval_csv),
]
try:
    res = subprocess.run(
        cmd, cwd="/content/kg1", check=False,
        capture_output=True, text=True, timeout=3600,
    )
    log(f"STDOUT:\n{res.stdout[-1500:]}")
except subprocess.TimeoutExpired:
    res = type("R", (), {"stdout": ""})
local_score_val = None
m = re.search(r"(?:overall\s+score|score)[:\s]+([0-9.]+)", res.stdout, re.IGNORECASE)
if m:
    local_score_val = float(m.group(1))
log(f"Local score = {local_score_val}")
with open(out_dir / "local_score.json", "w") as f:
    json.dump({"local_score": local_score_val}, f)
GO = False
if local_score_val is None:
    gate_msg = "NO-GO: parse failed"
elif local_score_val < CFG.local_score_floor:
    gate_msg = f"NO-GO: {local_score_val:.4f}"
elif local_score_val < CFG.target_score - 0.01:
    gate_msg = f"MARGINAL: {local_score_val:.4f}"
    GO = True
else:
    gate_msg = f"GO: {local_score_val:.4f}"
    GO = True
log(f"Gate: {gate_msg}")
with open(out_dir / "gate_decision.json", "w") as f:
    json.dump({"go": GO, "msg": gate_msg, "score": local_score_val}, f)

section("SECTION 11: Submit (if GO)")
if not GO:
    log("NO-GO -> skipping submit")
else:
    import zipfile

    zip_path = out_dir / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fn in ["adapter_config.json", "adapter_model.safetensors"]:
            zf.write(out_dir / fn, arcname=fn)
    log(f"zip: {zip_path.stat().st_size/(1024*1024):.2f}MB")
    zip_mb = zip_path.stat().st_size / (1024 * 1024)
    assert zip_mb < 500, f"zip too big: {zip_mb}MB"
    slot_ok = True
    try:
        rc = subprocess.run(
            ["kaggle", "competitions", "submissions",
             "-c", "nvidia-nemotron-model-reasoning-challenge", "--csv"],
            capture_output=True, text=True, timeout=60,
        )
        if rc.returncode == 0:
            from io import StringIO
            import csv as _csv

            today = datetime.datetime.now().strftime("%Y-%m-%d")
            today_count = sum(
                1 for r in _csv.DictReader(StringIO(rc.stdout))
                if r.get("date", "").startswith(today)
            )
            log(f"Kaggle slots today: {today_count}/5")
            slot_ok = today_count < 5
    except Exception as e:
        log(f"WARN slot check: {e}")
    if not slot_ok:
        log("Slot quota exhausted (5/5). Skipping submit.")
    elif os.environ.get("KAGGLE_USERNAME"):
        msg = f"V77_FIXED {datetime.datetime.now().strftime('%Y-%m-%d %H:%M BRT')}"
        cmd = [
            "kaggle", "competitions", "submit",
            "-c", "nvidia-nemotron-model-reasoning-challenge",
            "-f", str(zip_path), "-m", msg,
        ]
        r3 = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        with open(out_dir / "kaggle_submit.json", "w") as f:
            json.dump(
                {"msg": msg, "rc": r3.returncode,
                 "stdout": r3.stdout[-1000:], "stderr": r3.stderr[-500:]},
                f,
            )
        log(f"Submit rc={r3.returncode}")
        if r3.returncode == 0:
            log("SUBMITTED. https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/submissions")
        else:
            log(f"Submit FAILED: {r3.stderr[-300:]}")

section("ALL DONE V77")
log(f"Total: {(time.time() - START_TIME)/3600:.2f}h  Score: {local_score_val}  Gate: {GO}")

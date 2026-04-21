"""Apply 3 defensive patches to KG1_V72_CLEAN.ipynb (Option C)."""
import json
import sys

path = 'notebooks/KG1_V72_CLEAN.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

patched = 0
for c in nb['cells']:
    if c.get('cell_type') != 'code':
        continue
    src = ''.join(c['source']) if isinstance(c['source'], list) else c['source']

    # ============================================================
    # PATCH 3: Grad checkpoint fallback
    # ============================================================
    old_gckpt = """# Enable gradient checkpointing to reduce activation memory
if CFG.use_gradient_checkpointing:
    try:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        log('Gradient checkpointing ENABLED (memory savings ~50%)')
    except Exception as e:
        log(f'WARN gradient_checkpointing not supported: {e}')"""

    new_gckpt = """# Enable gradient checkpointing to reduce activation memory
# PATCH 3 (defensive): if grad_ckpt fails, auto-reduce max_length
if CFG.use_gradient_checkpointing:
    grad_ckpt_ok = False
    try:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        grad_ckpt_ok = True
        log('Gradient checkpointing ENABLED (memory savings ~50%)')
    except Exception as e:
        log(f'WARN gradient_checkpointing not supported: {e}')
        old_ml = CFG.max_length
        CFG.max_length = max(512, CFG.max_length // 2)
        log(f'AUTO-RECOVERY: max_length {old_ml} -> {CFG.max_length} to compensate')
        CFG.use_gradient_checkpointing = False"""

    if old_gckpt in src:
        src = src.replace(old_gckpt, new_gckpt)
        patched += 1
        print('Applied Patch 3 (grad_ckpt fallback)')

    # ============================================================
    # PATCH 1: Adaptive OOM recovery in smoke test
    # ============================================================
    old_smoke = """ds = JsonlChatDataset(train_path, tok, CFG.max_length)
dl = DataLoader(
    ds, batch_size=CFG.per_device_batch, shuffle=True,
    collate_fn=lambda b: collate(b, tok.pad_token_id),
)
model.train()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
losses = []
for step, batch in enumerate(dl):
    batch = {k: v.to(model.device) for k, v in batch.items()}
    out = model(**{k: v for k, v in batch.items() if k != 'labels'})
    loss = torch.nn.functional.cross_entropy(
        out.logits.view(-1, out.logits.size(-1)),
        batch['labels'].view(-1),
        ignore_index=-100,
    )
    losses.append(loss.item())
    assert not math.isnan(loss.item()) and not math.isinf(loss.item()), \\
        f'NaN/Inf loss at step {step}'
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], CFG.grad_clip,
    )
    opt.step()
    opt.zero_grad(set_to_none=True)
    log(f'smoke step {step}  loss={loss.item():.4f}')
    if step + 1 >= CFG.smoke_test_steps:
        break

assert losses[-1] < CFG.smoke_abort_loss, \\
    f'ABORT smoke loss {losses[-1]:.2f} > threshold'
log(f'Smoke test PASSED (final loss {losses[-1]:.4f})')
del opt, batch, out, loss, losses, ds, dl
gc.collect()
torch.cuda.empty_cache()"""

    new_smoke = """# PATCH 1 (defensive): Adaptive OOM recovery - reduces max_length if OOM
smoke_tried_lengths = []
losses = []
smoke_success = False

for attempt_max_len in [CFG.max_length, CFG.max_length // 2, CFG.max_length // 4, 512]:
    if attempt_max_len in smoke_tried_lengths or attempt_max_len < 256:
        continue
    smoke_tried_lengths.append(attempt_max_len)
    gc.collect()
    torch.cuda.empty_cache()
    log(f'Smoke attempt with max_length={attempt_max_len}')

    ds = JsonlChatDataset(train_path, tok, attempt_max_len)
    dl = DataLoader(
        ds, batch_size=CFG.per_device_batch, shuffle=True,
        collate_fn=lambda b: collate(b, tok.pad_token_id),
    )
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    losses = []

    try:
        for step, batch in enumerate(dl):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**{k: v for k, v in batch.items() if k != 'labels'})
            loss = torch.nn.functional.cross_entropy(
                out.logits.view(-1, out.logits.size(-1)),
                batch['labels'].view(-1),
                ignore_index=-100,
            )
            if math.isnan(loss.item()) or math.isinf(loss.item()):
                log(f'NaN/Inf at smoke step {step} - CRITICAL')
                raise RuntimeError(f'NaN loss at smoke step {step}')
            losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], CFG.grad_clip,
            )
            opt.step()
            opt.zero_grad(set_to_none=True)
            log(f'smoke step {step}  loss={loss.item():.4f}')
            if step + 1 >= CFG.smoke_test_steps:
                break

        if losses and losses[-1] < CFG.smoke_abort_loss:
            CFG.max_length = attempt_max_len
            smoke_success = True
            log(f'Smoke test PASSED at max_length={attempt_max_len} (final loss {losses[-1]:.4f})')
            break
        else:
            log(f'Loss too high at max_length={attempt_max_len}')

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if isinstance(e, torch.cuda.OutOfMemoryError) or 'out of memory' in str(e).lower():
            log(f'OOM at max_length={attempt_max_len}, trying smaller...')
            try:
                del opt, ds, dl
            except Exception:
                pass
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        else:
            raise

if not smoke_success:
    raise RuntimeError(f'Smoke test FAILED at all max_lengths: {smoke_tried_lengths}')

try:
    del opt, ds, dl, batch, out, loss
except (NameError, UnboundLocalError):
    pass
gc.collect()
torch.cuda.empty_cache()
log(f'Final max_length locked in: {CFG.max_length}')"""

    if old_smoke in src:
        src = src.replace(old_smoke, new_smoke)
        patched += 1
        print('Applied Patch 1 (adaptive OOM smoke test)')

    # ============================================================
    # PATCH 2: NaN guard in MaxMinSFTTrainer + CE fallback
    # ============================================================
    old_trainer = """class MaxMinSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get('labels')
        outputs = model(**{k: v for k, v in inputs.items() if k != 'labels'})
        logits = outputs.logits
        step = int(self.state.global_step)
        if CFG.loss_type == 'ce':
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100,
            )
        elif CFG.loss_type == 'max_min':
            loss = max_min_logprob_loss(logits, labels)
        else:
            if step < CFG.max_min_warmup_steps:
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100,
                )
            else:
                loss = max_min_logprob_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss"""

    new_trainer = """class MaxMinSFTTrainer(SFTTrainer):
    # PATCH 2 (defensive): NaN guard + auto-fallback to CE
    _nan_count = 0
    _use_ce_permanent = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get('labels')
        outputs = model(**{k: v for k, v in inputs.items() if k != 'labels'})
        logits = outputs.logits
        step = int(self.state.global_step)

        ce_loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100,
        )

        use_ce = (
            self._use_ce_permanent
            or CFG.loss_type == 'ce'
            or (CFG.loss_type == 'max_min_warmup_ce' and step < CFG.max_min_warmup_steps)
        )

        if use_ce:
            loss = ce_loss
        else:
            try:
                mm_loss = max_min_logprob_loss(logits, labels)
                if torch.isnan(mm_loss) or torch.isinf(mm_loss):
                    self._nan_count += 1
                    print(f'WARN: NaN/Inf max-min loss step {step} (count {self._nan_count}), CE fallback')
                    if self._nan_count >= 3:
                        self._use_ce_permanent = True
                        print(f'CRITICAL: CE permanent after {self._nan_count} NaNs')
                    loss = ce_loss
                else:
                    loss = mm_loss
                    self._nan_count = max(0, self._nan_count - 1)
            except Exception as e:
                print(f'WARN max-min exception step {step}: {e}, using CE')
                loss = ce_loss

        if torch.isnan(loss) or torch.isinf(loss):
            print(f'WARN: final loss NaN step {step}, forcing CE')
            loss = ce_loss

        return (loss, outputs) if return_outputs else loss"""

    if old_trainer in src:
        src = src.replace(old_trainer, new_trainer)
        patched += 1
        print('Applied Patch 2 (NaN guard + CE fallback)')

    c['source'] = src.splitlines(keepends=True)

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'\nTotal patches applied: {patched}')

# Validate
import py_compile
import tempfile
import os

for c in nb['cells']:
    if c.get('cell_type') != 'code':
        continue
    src = ''.join(c['source']) if isinstance(c['source'], list) else c['source']
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False, encoding='utf-8') as t:
        t.write(src)
        tpath = t.name
    try:
        py_compile.compile(tpath, doraise=True)
    except py_compile.PyCompileError as e:
        print(f'SYNTAX ERROR: {e}')
        sys.exit(1)
    os.unlink(tpath)

print('py_compile OK')

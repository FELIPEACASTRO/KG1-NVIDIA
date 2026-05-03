# KG1 Colab IPYNB Execution Gate

- decision: `FAIL`
- errors: `18`
- warnings: `3`
- notebook: `runs\v198_master_ipynb_audit_20260503\KG1_V198_MICRO_DISTILL_COLAB_PRO.master.ipynb`
- pack: `runs\v198_master_ipynb_audit_20260503\kg1_v198_colab_pack_49c2c2.zip`
- pack_sha256: `7e3e41b55bb6f5736c3d5325c7b481f3b52ac918eb13c311e9a343f43f6dedca`

## Checks

- notebook_cells: `26`
- code_cells: `22`
- train_rows: `1875`
- val_rows: `720`
- network: `True`

## Findings

- `error` `stale_script_repair`: missing required notebook fragment: Runtime has stale hf_job_train_v90.py; downloading PEFT direct-load fixed script
- `error` `fixed_train_script_patch_assert`: missing required notebook fragment: assert 'load_peft_weights_with_direct_fallback' in script_text
- `error` `fixed_train_script_env_assert`: missing required notebook fragment: assert 'PEFT_MANUAL_LOAD_METHOD' in script_text
- `warning` `pack_sha_manifest_mismatch`: notebook=7e3e41b55bb6f5736c3d5325c7b481f3b52ac918eb13c311e9a343f43f6dedca manifest=e61908c0f75018b0d265c3668600170f6fa99a1a4d559508f489cba9cd6b7c93
- `error` `env_peft_manual_load_method_mismatch`: observed=None expected='direct'
- `error` `active_raw_kaggle_submit_cells`: raw kaggle CLI submit cells must be removed or commented before Run all: cells=[13, 14, 17, 19]
- `warning` `manual_kaggle_json_upload_cell_present`: manual upload of kaggle.json is present; Colab Secrets userdata path is preferred
- `warning` `local_pack_sha_manifest_mismatch`: 7e3e41b55bb6f5736c3d5325c7b481f3b52ac918eb13c311e9a343f43f6dedca != e61908c0f75018b0d265c3668600170f6fa99a1a4d559508f489cba9cd6b7c93
- `error` `zip_script_hash_mismatch`: scripts/hf_job_train_v90.py
- `error` `peft_direct_load_guard_missing`: zip:scripts/hf_job_train_v90.py: missing 'PEFT_MANUAL_LOAD_METHOD = env_str'
- `error` `peft_direct_load_guard_missing`: zip:scripts/hf_job_train_v90.py: missing 'def remap_peft_state_dict_for_direct_load'
- `error` `peft_direct_load_guard_missing`: zip:scripts/hf_job_train_v90.py: missing 'def load_peft_weights_with_direct_fallback'
- `error` `peft_direct_load_guard_missing`: zip:scripts/hf_job_train_v90.py: missing 'load_peft_weights_with_direct_fallback(loaded_model, weights, adapter_name="default")'
- `error` `peft_direct_load_guard_missing`: zip:scripts/hf_job_train_v90.py: missing 'set_peft_model_state_dict failed; falling back to direct PEFT state_dict load'
- `error` `peft_direct_load_guard_missing`: zip:scripts/hf_job_train_v90.py: missing 'Direct PEFT adapter load mapping:'
- `error` `peft_direct_load_guard_missing`: zip:scripts/hf_job_train_v90.py: missing 'coverage < 0.98'
- `error` `peft_direct_load_guard_missing`: zip:scripts/hf_job_train_v90.py: missing 'missing_lora={len(missing_lora)}'
- `error` `peft_stale_direct_setter_block`: zip:scripts/hf_job_train_v90.py: stale direct set_peft_model_state_dict block remains
- `error` `peft_stale_direct_setter_block`: zip:scripts/hf_job_train_v90.py: stale direct set_peft_model_state_dict block remains
- `error` `peft_direct_setter_outside_fallback`: zip:scripts/hf_job_train_v90.py: load_trainable_adapter_or_create still calls set_peft_model_state_dict directly
- `error` `peft_direct_setter_outside_fallback`: zip:scripts/hf_job_train_v90.py: load_trainable_adapter_or_create still calls set_peft_model_state_dict directly

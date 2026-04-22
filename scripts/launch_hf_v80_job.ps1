param(
  [switch]$Smoke,
  [string]$Flavor = "",
  [string]$Timeout = "",
  [string]$Image = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel",
  [string]$DataRepo = "felipesp1983/kg1-nemotron-training",
  [string]$OutputRepo = "felipesp1983/kg1-nemotron-lora-v80-hf-dgxchen",
  [int]$MaxSteps = -1,
  [int]$MaxLength = 8192,
  [int]$BatchSize = 32,
  [int]$MicroBatchSize = 1,
  [int]$LoraRank = 32,
  [int]$LoraAlpha = 32,
  [double]$LearningRate = 2e-4,
  [int]$NumEpochs = 1,
  [int]$Seed = 42,
  [string]$CudaVisibleDevices = "0",
  [switch]$DryRunValidateOnly,
  [switch]$NoUnsloth,
  [switch]$Detach,
  [switch]$PrintOnly
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command hf -ErrorAction SilentlyContinue)) {
  throw "Hugging Face CLI not found. Install with: pip install -U huggingface_hub"
}

if ($Smoke) {
  if (-not $Flavor) { $Flavor = "a100-large" }
  if (-not $Timeout) { $Timeout = "1h" }
  $OutputRepo = "felipesp1983/kg1-nemotron-lora-v80-smoke"
  $MaxSteps = 5
  $MaxLength = 2048
  $extraEnv = @(
    "RUN_ID=v80-smoke-mlen2048-s5",
    "SAVE_EVERY_STEPS=5"
  )
} else {
  if (-not $Flavor) { $Flavor = "a100-large" }
  if (-not $Timeout) { $Timeout = "6h" }
  $extraEnv = @(
    "RUN_ID=v80-dgxchen-exact-r$LoraRank-a$LoraAlpha-mlen$MaxLength-e$NumEpochs"
  )
}

if ($DryRunValidateOnly) {
  $MaxSteps = 0
  $extraEnv = @($extraEnv | Where-Object { $_ -notlike "RUN_ID=*" })
  $extraEnv += "RUN_ID=v80-dryrun-mlen$MaxLength"
  $extraEnv += "DRY_RUN_VALIDATE_ONLY=1"
}

if ($NoUnsloth) {
  $extraEnv += "USE_UNSLOTH=0"
}

$remoteCommand = @'
python -m pip install -q huggingface_hub && python - <<'PY'
import os
import subprocess
from huggingface_hub import get_token, hf_hub_download

repo_id = os.environ.get('HF_DATA_REPO') or os.environ.get('DATA_REPO') or 'felipesp1983/kg1-nemotron-training'
token = os.environ.get('HF_TOKEN') or get_token()
path = hf_hub_download(
    repo_id=repo_id,
    repo_type='dataset',
    filename='scripts/hf_job_wrapper_v80.sh',
    token=token,
)
print('Executing remote wrapper: {}'.format(path))
subprocess.run(['bash', path], check=True)
PY
'@

$argsList = @(
  "jobs", "run",
  "--flavor", $Flavor,
  "--timeout", $Timeout,
  "--secrets", "HF_TOKEN",
  "--env", "DATA_REPO=$DataRepo",
  "--env", "HF_DATA_REPO=$DataRepo",
  "--env", "OUTPUT_REPO=$OutputRepo",
  "--env", "MAX_STEPS=$MaxSteps",
  "--env", "MAX_LENGTH=$MaxLength",
  "--env", "BATCH_SIZE=$BatchSize",
  "--env", "MICRO_BATCH_SIZE=$MicroBatchSize",
  "--env", "LORA_RANK=$LoraRank",
  "--env", "LORA_ALPHA=$LoraAlpha",
  "--env", "LEARNING_RATE=$LearningRate",
  "--env", "NUM_EPOCHS=$NumEpochs",
  "--env", "SEED=$Seed",
  "--env", "NVIDIA_VISIBLE_DEVICES=all",
  "--env", "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
  "--env", "LANG=C.UTF-8",
  "--env", "LC_ALL=C.UTF-8",
  "--env", "PYTHONIOENCODING=utf-8",
  "--env", "HF_HUB_DISABLE_PROGRESS_BARS=1",
  "--env", "TQDM_DISABLE=1"
)

if ($CudaVisibleDevices) {
  $argsList += @("--env", "CUDA_VISIBLE_DEVICES=$CudaVisibleDevices")
}

foreach ($envVar in $extraEnv) {
  $argsList += @("--env", $envVar)
}

if ($Detach) {
  $argsList += "--detach"
}

$argsList += @("--", $Image, "bash", "-lc", $remoteCommand)

Write-Host "=================================================================="
Write-Host "V80 HF JOB LAUNCH - dgxchen v7 EXACT replica"
Write-Host "=================================================================="
Write-Host "Mode:        $(if ($Smoke) { 'SMOKE (5 steps, quick validation)' } elseif ($DryRunValidateOnly) { 'DRY-RUN (no training)' } else { 'FULL (1 epoch ~245 steps)' })"
Write-Host "Flavor:      $Flavor"
Write-Host "Timeout:     $Timeout"
Write-Host "Image:       $Image"
Write-Host "Output repo: $OutputRepo"
Write-Host "Data repo:   $DataRepo"
Write-Host "Model:       nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
Write-Host "LoRA:        r=$LoraRank alpha=$LoraAlpha dropout=0 NO lm_head"
Write-Host "Max length:  $MaxLength"
Write-Host "Batch:       eff=$BatchSize micro=$MicroBatchSize"
Write-Host "LR:          $LearningRate linear decay, warmup=0"
Write-Host "Epochs:      $NumEpochs"
Write-Host "Use Unsloth: $(-not $NoUnsloth)"
Write-Host ""
Write-Host "Command:"
Write-Host ("hf " + ($argsList | ForEach-Object {
  if ($_ -match "\s") { '"' + ($_ -replace '"', '\"') + '"' } else { $_ }
}) -join " ")

if ($PrintOnly) {
  Write-Host ""
  Write-Host "[PRINT ONLY] Not launching job. Remove -PrintOnly to execute."
  exit 0
}

Write-Host ""
Write-Host "Launching HF Job... (use Ctrl+C to abort launch)"
& hf @argsList

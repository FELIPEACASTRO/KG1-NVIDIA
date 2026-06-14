# KG1 Colab Pro Real-Time Log Monitoring

Este fluxo permite acompanhar um job do Colab Pro a partir desta maquina, quase em tempo real, sem depender de olhar a aba do notebook.

## Verdade Operacional

Eu nao consigo ler a saida de uma aba Colab isolada no navegador. Para monitorar em tempo real, o job precisa publicar o log em algum lugar acessivel daqui.

O caminho recomendado e:

1. Colab roda o job com `scripts/kg1_colab_realtime_runner.py`.
2. O runner salva stdout/stderr em arquivo local no Colab.
3. O runner faz upload periodico para um repo privado no Hugging Face.
4. Daqui, `scripts/kg1_colab_live_monitor.py` baixa o log e interpreta a saude do job.

## Proxima Etapa Recomendada: V1243 Launch Pack

No Windows/local, gere o pacote minimo para Colab:

```powershell
python scripts\kg1_build_v1243_colab_launch_pack.py
```

Saidas:

```text
artifacts\v1243_colab_launch_pack\
artifacts\v1243_colab_launch_pack.zip
```

No caminho recomendado, o notebook one-click baixa e valida esse zip automaticamente.
Use o fluxo manual abaixo apenas como fallback/debug:

```bash
unzip -o v1243_colab_launch_pack.zip -d /content/kg1_v1243
cd /content/kg1_v1243
pip install -q -r requirements_v1243_colab.txt
```

Sequencia segura:

```bash
# 1) Sem GPU cara: tokenizacao, hashes, prompt, mascaras e contrato.
python scripts/kg1_colab_v1243_launcher.py \
  --phase bit_specialist \
  --run-mode tokenize_dryrun \
  --target-accuracy 0.89 \
  --live-log-repo "$KG1_LIVE_LOG_HF_REPO" \
  --require-live-log-upload

# 2) GPU: carrega modelo/LoRA e para antes de treinar.
export KG1_ACCEPT_GPU_SPEND=1
export INIT_ADAPTER_REPO="owner/baseline-adapter-repo"
export INIT_ADAPTER_REVISION="pinned-commit-sha"
pip install --progress-bar off --no-build-isolation mamba-ssm==2.3.1
python scripts/kg1_colab_v1243_launcher.py \
  --phase bit_specialist \
  --run-mode model_dryrun \
  --accept-gpu-spend \
  --target-accuracy 0.89 \
  --live-log-repo "$KG1_LIVE_LOG_HF_REPO" \
  --require-live-log-upload

# 3) Treino real, somente depois dos gates e decisao explicita.
python scripts/kg1_colab_v1243_launcher.py \
  --phase bit_specialist \
  --run-mode real_train \
  --allow-real-train \
  --accept-gpu-spend \
  --target-accuracy 0.89 \
  --live-log-repo "$KG1_LIVE_LOG_HF_REPO" \
  --require-live-log-upload \
  --output-repo "$OUTPUT_REPO"
```

O launcher reescreve automaticamente `DATA_FILE` e `VAL_FILE` para os paths do Colab. Isso evita o erro comum de usar caminhos Windows dentro do Colab.

## Variaveis Do Colab

Configure no Colab antes de iniciar o job:

```python
import os, time

RUN_ID = "v1243_bit_colab_" + time.strftime("%Y%m%d_%H%M%S")
os.environ["RUN_ID"] = RUN_ID
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["FRIENDLY_REALTIME_LOGS"] = "1"
os.environ["FRIENDLY_LOG_SCORE_HINTS"] = "1"

# Use um repo privado de logs. Exemplo:
os.environ["KG1_LIVE_LOG_HF_REPO"] = "felipesp1983/kg1-live-logs"
os.environ["KG1_LIVE_LOG_HF_REPO_TYPE"] = "dataset"
os.environ["KG1_LIVE_LOG_HF_PATH"] = f"colab/{RUN_ID}/train.log"
os.environ["KG1_LIVE_STATUS_HF_PATH"] = f"colab/{RUN_ID}/status.json"
os.environ["KG1_LIVE_LOG_UPLOAD_EVERY"] = "30"
os.environ["KG1_WATCHDOG_STALE_SECONDS"] = "1800"
os.environ["KG1_WATCHDOG_MAX_RUNTIME_SECONDS"] = "0"
os.environ["KG1_DISABLE_HEALTH_WATCHDOG"] = "0"
```

O `HF_TOKEN` deve vir do Colab Secrets ou de ambiente seguro. Nao imprima o token.

## Rodar O Job No Colab

Use o runner:

```bash
python scripts/kg1_colab_realtime_runner.py \
  --hf-repo "$KG1_LIVE_LOG_HF_REPO" \
  --hf-path "$KG1_LIVE_LOG_HF_PATH" \
  --hf-status-path "$KG1_LIVE_STATUS_HF_PATH" \
  --hf-repo-type "$KG1_LIVE_LOG_HF_REPO_TYPE" \
  --upload-every 30 \
  -- \
  python scripts/hf_job_train_v90.py
```

O runner imprime e salva tudo:

- `KG1_SCORE_CONTRACT_STATUS=...`
- `KG1_SCORE_PROXY_STATUS=...`
- `KG1_SCORE_TRAJECTORY_STATUS=...`
- `KG1_WATCHDOG_STOP reason=...`, se alguma regra de watchdog abortar.
- `[KG1-TEACH]...`
- tracebacks/abort/OOM, se ocorrerem.

## Watchdog Do Job

O runner `scripts/kg1_colab_realtime_runner.py` agora tem watchdog ativo:

- `KG1_WATCHDOG_STALE_SECONDS=1800`: aborta se o processo ficar sem stdout por 30 minutos.
- `KG1_WATCHDOG_MAX_RUNTIME_SECONDS=0`: sem limite total por default; configure um valor em segundos se quiser teto de custo.
- `KG1_DISABLE_HEALTH_WATCHDOG=0`: mantem ativo o watchdog de saude; se o parser enxergar `STOP`, o processo filho e encerrado.

Regras:

- upload obrigatorio falhou com `--require-live-log-upload`: aborta, para nao treinar cego.
- `SCORE_CONTRACT=STOP`: aborta pelo health watchdog.
- `SCORE_TRAJECTORY=STOP`: aborta pelo health watchdog.
- sem novas linhas por mais que `KG1_WATCHDOG_STALE_SECONDS`: aborta por silencio.
- runtime acima de `KG1_WATCHDOG_MAX_RUNTIME_SECONDS`, quando maior que zero: aborta por teto de custo.

## Monitorar Daqui

Na maquina local:

```powershell
$env:HF_TOKEN="<token-com-read-no-repo-privado>"
python scripts\kg1_colab_live_monitor.py `
  --hf-repo felipesp1983/kg1-live-logs `
  --hf-path colab/<RUN_ID>/train.log `
  --hf-repo-type dataset `
  --interval 30 `
  --target-accuracy 0.89
```

Para o alvo `0.98`, rode com:

```powershell
python scripts\kg1_colab_live_monitor.py `
  --hf-repo felipesp1983/kg1-live-logs `
  --hf-path colab/<RUN_ID>/train.log `
  --hf-repo-type dataset `
  --interval 30 `
  --target-accuracy 0.98
```

Se quiser que o contrato dentro do proprio job tambem use `0.98`, configure no Colab antes de rodar:

```python
os.environ["SCORE_CONTRACT_TARGET_ACCURACY"] = "0.98"
```

Matematica do alvo em full947:

- `0.89` exige `843/947`, ou `+20` acertos acima do baseline `823/947`.
- `0.90` exige `853/947`, ou `+30`.
- `0.98` exige `929/947`, ou `+106`.

O monitor nao prova esse score. Ele prova saude operacional do job. Score real continua exigindo geracao vLLM greedy, `raw_output`, extracao/verificacao e gate V1241 full947.

## Como Ler O Painel

Estados:

- `BOOT`: ainda nao vimos contrato de score.
- `WATCH`: contrato passou, mas ainda falta proxy suficiente.
- `OK`: contrato passou e score-proxy/trajetoria estao saudaveis.
- `RISK`: loss pode estar bom, mas cauda boxed ou trajetoria para o score nao esta melhorando.
- `STOP`: abort, traceback, OOM ou contrato falhou.

Sinais mais importantes:

- `score_contract.status=PASS`: o caminho estrutural do score esta alinhado.
- `score_proxy.boxed_tail_loss`: deve cair.
- `score_proxy.boxed_tail_token_accuracy`: deve subir.
- `score_proxy.boxed_tail_exact_rate`: deve subir.
- `score_trajectory.status=OK`: bit/equation melhoraram sem regressao global/protegida.
- `score_trajectory.weak_exact_delta > 0`: as familias fracas estao se movendo na direcao certa.
- `score_trajectory.protected_exact_delta >= 0`: familias ja boas nao estao pagando a conta.
- `score_trajectory.target_required=843`: alvo operacional para `>=0.89` no full947.
- `delta_boxed_loss < 0`: bom.
- `delta_boxed_exact > 0`: bom.

Regra pratica:

- `STOP`: interromper e corrigir.
- `RISK`: inspecionar antes de continuar gastando GPU; loss bom sozinho nao justifica seguir.
- `OK`: o job esta saudavel, mas ainda sem claim de score.

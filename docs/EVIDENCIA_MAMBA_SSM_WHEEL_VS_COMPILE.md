# 📑 EVIDÊNCIA — `mamba-ssm`: instalar por WHEEL pré-compilada vs COMPILAR do código-fonte
### Documento de FATOS para a IA decidir. Sem recomendação/conclusão do autor. Cada item tem fonte.

> **Gerado:** 2026-06-14 · **Tipo:** read-only, coleta de evidência. **NÃO contém recomendação** — apenas dados verificáveis.
> **Para a IA leitora:** tire suas próprias conclusões. Abaixo só há fatos observados (logs do projeto) e fontes externas (PyPI, GitHub do `state-spaces/mamba`, issues).

---

## 1. FATO OBSERVADO NO PRÓPRIO PROJETO (logs reais)

### 1.1 O launcher atual instala mamba COMPILANDO do código-fonte
Arquivo `KG1_V1243_COLAB_REALTIME_LAUNCHER.ipynb` (commit cc95a473), função `ensure_gpu_model_dependencies`:
```python
KG1_MAMBA_SSM_PIP_SPEC default = 'mamba-ssm==2.3.1'
pip install --no-build-isolation mamba-ssm==2.3.1
```
`--no-build-isolation` + spec de versão (sem URL de wheel) ⇒ pip resolve o sdist e **compila**.

### 1.2 Tempo de compilação medido no projeto (heartbeat FASE5, runtime Colab)
Log real do projeto (run FASE5, mesma família de ambiente):
```
[1/6] INSTALL (... mamba_ssm compila ~10-20min — normal)
--no-build-isolation mamba_ssm==2.2.5 causal_conv1d==1.5.2
[kernels] selective_scan_cuda OK
[kernels] causal_conv1d_cuda OK
```
→ Compilação do mamba registrada como **~10-20 min** nesse ambiente.

### 1.3 Falha real de import do mamba no model-load (run de GPU 19:03)
Log HF `kg1-live-logs/.../model_dryrun_20260613_190301/train.log`:
```
Loading model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 ...
ModuleNotFoundError: No module named 'mamba_ssm'
ImportError: mamba-ssm is required by the Mamba model but cannot be imported
KG1_COLAB_REALTIME_RUNNER_END return_code=1
```

### 1.4 Runtime medido no Colab atual (dry-run report real)
```
runtime: {"cuda_available": true, "torch": "2.11.0+cu128"}
```
→ O Colab deste job roda **torch 2.11.0 + CUDA 12.8**.

---

## 2. FATOS EXTERNOS — instalação do `mamba-ssm` (fontes oficiais)

### 2.1 Por padrão o pip COMPILA (precisa de nvcc/CUDA)
- PyPI `mamba-ssm`: *"--no-build-isolation is required so that pip uses your existing CUDA-enabled PyTorch instead of installing torch-cpu in an isolated build environment."*
- Pré-requisitos listados: CUDA 11.6+, PyTorch 1.12+, Linux, GPU NVIDIA.
- Versão atual no PyPI: **2.3.2.post1** (09/mai/2026).
- Fonte: https://pypi.org/project/mamba-ssm/

### 2.2 EXISTEM wheels pré-compiladas (GitHub Releases) — instalação por download
- A página de releases do `state-spaces/mamba` anexa **wheels pré-compiladas** (82+ assets por release), com SHA256.
- Padrão de nome:
  `mamba_ssm-{versão}+{cuda}{torch}{cxx11abi}-{cpython}-{cpython}-linux_x86_64.whl`
- Exemplos reais de assets:
  - `mamba_ssm-2.3.2.post1+cu11torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl`
  - `mamba_ssm-2.3.2.post1+cu11torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl`
- Fonte: https://github.com/state-spaces/mamba/releases

### 2.3 ⚠️ FATO crítico de compatibilidade (cruzar §1.4 com §2.2)
- Wheels oficiais observadas cobrem **CUDA `cu11`** e **torch `2.6`/`2.7`**.
- O Colab deste job roda **torch `2.11.0` + `cu128`** (§1.4).
- ⇒ **NÃO foi observada wheel oficial casando com torch 2.11 / cu128.** As wheels prontas existem para torch ≤ 2.7.
- (Instalar uma wheel exige casar exatamente: cuda × torch × cpython × cxx11abi.)

### 2.4 Falhas de build do `mamba-ssm` em Colab são documentadas
- Múltiplas issues relatam falha ao "Building wheel for mamba-ssm" em Google Colab, incluindo "nvcc not found" e erros de CUDA runtime.
- Fontes:
  - https://github.com/state-spaces/mamba/issues/607 (Installation fails on Google Colab)
  - https://github.com/state-spaces/mamba/issues/742 (Building wheels for causal-conv1d, mamba-ssm)
  - https://github.com/state-spaces/mamba/issues/731 (Building wheel did not run successfully)

### 2.5 Mamba-3 / forçar build do source
- PyPI: *"To use Mamba-3, please install from source MAMBA_FORCE_BUILD=TRUE pip install --no-cache-dir --force-reinstall git+https://github.com/state-spaces/mamba.git --no-build-isolation."*
- Fonte: https://pypi.org/project/mamba-ssm/

---

## 3. FATOS GERAIS — instalar wheel vs compilar (pip, geral)

- Instalar um `.whl` = baixar o arquivo + descompactar/registrar. Não invoca compilador (nvcc/gcc). Tempo ≈ tempo de download do arquivo (ordem de segundos a ~1 min para wheels de poucos/dezenas de MB).
- Instalar de sdist com extensão CUDA = baixar fonte + invocar nvcc para compilar kernels → tempo depende de CPU/nº de arquivos CUDA (ordem de minutos a dezenas de minutos).
- `--no-build-isolation` NÃO pula a compilação; ele só reusa o torch já instalado em vez de criar ambiente isolado.
- Wheel só instala se o nome casar com (cuda, torch, cpython, abi, plataforma) do ambiente; caso contrário o pip ignora a wheel e/ou cai para build do source.

---

## 4. TABELA-RESUMO (apenas dados, sem conclusão)

| Item | Valor observado | Fonte |
|---|---|---|
| Método atual do launcher | compila (`--no-build-isolation mamba-ssm==2.3.1`) | §1.1 |
| Tempo de compile medido no projeto | ~10-20 min | §1.2 (heartbeat FASE5) |
| Falha de import no model-load (GPU) | sim, 1× (19:03) | §1.3 |
| Torch/CUDA do Colab atual | 2.11.0 / cu128 | §1.4 |
| Wheels oficiais existem? | sim (GitHub releases, 82+ assets) | §2.2 |
| Wheels cobrem torch 2.11/cu128? | **não observado** (cobrem cu11 torch 2.6/2.7) | §2.2 + §2.3 |
| Tempo de instalar wheel (geral) | ≈ download (segundos–~1min), sem nvcc | §3 |
| Falhas de build em Colab documentadas | sim (várias issues) | §2.4 |

---

## 5. PERGUNTAS EM ABERTO (para a IA resolver, sem viés do autor)

1. Existe wheel pré-compilada de `mamba-ssm` (qualquer versão) para **torch 2.11 + cu128 + cp312** especificamente? (a busca encontrou apenas ≤ torch2.7/cu11.)
2. Se não existir, qual é mais barato no contexto (deadline + custo A100): (a) compilar para torch 2.11, (b) alinhar o runtime a torch 2.7 e usar wheel pronta, ou (c) pré-buildar 1× e hospedar a wheel?
3. O quanto a compilação do mamba na A100 é CPU-bound (i.e., quanto de GPU-minuto é gasto sem usar a GPU)?

---

*Documento de evidência. Sem recomendação. Fontes: logs do projeto (HF kg1-live-logs, heartbeat FASE5), PyPI mamba-ssm, GitHub state-spaces/mamba (releases + issues). A IA leitora deve verificar as fontes e concluir por conta própria.*

---

## 6. Validação Codex — 2026-06-14

Consulta feita na API pública do GitHub:
`https://api.github.com/repos/state-spaces/mamba/releases?per_page=10`

Achado corrigido:
- A evidência original dizia que as wheels observadas cobriam até `torch2.7`.
- A API atual mostra também wheels `cu12/torch2.10/cp312`.
- Porém a busca exata por `torch2.11 + cu12/cu128 + cp312` retornou `0` assets.

Conclusão operacional aplicada:
- Não foi hardcodada nenhuma wheel no launcher.
- O launcher agora valida o nome da wheel fornecida contra `python_tag`, `torch major.minor` e `CUDA major`.
- Se `mamba_ssm` já estiver importável no runtime, o launcher pula a reinstalação e não bloqueia por possível source build.
- Se não houver wheel direta compatível, treino real continua exigindo decisão explícita: fornecer wheel compatível ou setar `KG1_V1243_ALLOW_MAMBA_SOURCE_BUILD=1`.

## Sources
- [PyPI — mamba-ssm](https://pypi.org/project/mamba-ssm/)
- [GitHub — state-spaces/mamba releases](https://github.com/state-spaces/mamba/releases)
- [Issue #607 — install fails on Google Colab](https://github.com/state-spaces/mamba/issues/607)
- [Issue #742 — building wheels causal-conv1d, mamba-ssm](https://github.com/state-spaces/mamba/issues/742)
- [Issue #731 — building wheel did not run successfully](https://github.com/state-spaces/mamba/issues/731)

# 🍕📟 O Job, LINHA POR LINHA — de verdade, no estilo *Use a Cabeça*

> **Como ler este guia:** cada linha do notebook aparece **com uma anotação à mão** (o `# ←`),
> e logo abaixo a **tradução em português de gente**. Tem caixas 🙋 *Não existe pergunta idiota*,
> ⚠️ *CUIDADO!*, ✏️ *Afie o lápis* e 🧠 *Esquente os neurônios*. Metáfora fixa: **a pizzaria**.
> (Chef = modelo 30B · Molho da Vovó = adapter 086 · Forno = GPU · Inspetor = watchdog.)

---

# 🧩 CÉLULA 1 — Buscar ingredientes e montar a cozinha (deps)

> **O que essa célula faz, numa frase:** vai ao mercado, **troca a batedeira pelo modelo certo** e
> pega a **massa pronta** — pra não perder 30 min sovando massa.

```python
import os, subprocess, sys                          # ← pega 3 "ferramentas" do Python
print('[1/5] clone repo branch', flush=True)        # ← avisa em voz alta: "indo buscar a apostila!"
```
- **`import os, subprocess, sys`** → pega 3 ajudantes: `os` (mexe em pastas), `subprocess` (roda comandos no terminal), `sys` (sabe qual Python tá rodando).
- **`print(..., flush=True)`** → o `flush=True` é a parte esperta: força a frase a **aparecer na hora** (sem ele, o Python guarda no bolso e você fica no escuro).

```python
subprocess.run(['git','clone','--depth','1','--branch','claude/v1244-cot-safe',
                'https://github.com/FELIPEACASTRO/KG1-NVIDIA.git','/content/kg1'], check=True)  # ← baixa a apostila
```
- **`git clone`** → baixa o repositório (a apostila de receitas).
- **`--depth 1`** → "só me dá a última página, não a história toda" (download rápido).
- **`--branch claude/v1244-cot-safe`** → a apostila **certa** (nossa versão).
- **`check=True`** → 🔑 **se a apostila não chegar, a cozinha NEM ABRE** (estoura erro aqui).

> 🙋 **Não existe pergunta idiota**
> **— Por que `check=True` só aqui e `check=False` lá embaixo?**
> Porque sem a apostila não dá pra fazer NADA (tem que parar). Já um `pip` que reclama de um detalhe
> bobo não precisa derrubar a cozinha — por isso os installs usam `check=False`.

```python
os.chdir('/content/kg1')                            # ← entra na cozinha (vira o diretório de trabalho)
```
- A partir daqui, todo caminho relativo (`scripts/...`) é **dentro da cozinha**.

```python
print('[2/5] deps base', flush=True)
subprocess.run([sys.executable,'-m','pip','install','-q','transformers==4.57.6','peft==0.19.1',
   'accelerate==1.13.0','bitsandbytes','safetensors','huggingface_hub','hf_xet','einops','ninja'], check=False)  # ← compra utensílios
```
- **`sys.executable,'-m','pip','install'`** → "usa ESTE Python pra instalar" (evita instalar no Python errado).
- **`-q`** → quieto (menos blá-blá na tela).
- **`transformers==4.57.6` etc.** → utensílios de **marca fixa**. O chef NemotronH só cozinha com a faca certa.
- **`einops`, `ninja`** → ajudantes que o mamba precisa (um pra "dobrar" tensores, outro pra compilar).

> ✏️ **Afie o lápis:** por que `transformers==4.57.6` e não a mais nova?
> *Resposta no rodapé.* ⬇️

```python
print('[3/5] pin torch 2.10 cu126 ...', flush=True)
subprocess.run([sys.executable,'-m','pip','install','-q','torch==2.10.0',
                '--index-url','https://download.pytorch.org/whl/cu126'], check=False)  # ← TROCA a batedeira
```
- **O PLOT TWIST.** O Colab veio com a **batedeira 2.11** (nova demais). A **massa pronta** (wheel do
  mamba) só encaixa na **2.10**. Então a gente troca de propósito.
- **`--index-url .../cu126`** → 🔑 compra na **loja certa**, que entrega a batedeira **com motor (CUDA)**.

> ⚠️ **CUIDADO!** Se comprar na loja errada (PyPI padrão), vem uma batedeira **sem motor (CPU)** —
> linda, mas não bate nada. O `--index-url cu126` evita esse desastre.

```python
import torch                                                   # ← liga a batedeira
assert torch.cuda.is_available() and torch.version.cuda, '...' # ← confere se tem motor
```
- **`import torch`** → agora carrega a batedeira 2.10 (que acabamos de instalar).
- **`assert torch.cuda...`** → "liga na tomada e vê se acende". **Se veio sem motor, PARA AQUI** com recado claro.

```python
py=f"cp{sys.version_info.major}{sys.version_info.minor}"       # ← cp312 (versão do Python)
tmm='.'.join(torch.__version__.split('+')[0].split('.')[:2])  # ← "2.10" (modelo da batedeira)
cu='cu'+((torch.version.cuda or '12').split('.')[0])          # ← "cu12" (tipo de motor)
abi='TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE'    # ← "TRUE" (padrão de encaixe)
print(f'[env] {py} torch{tmm} {cu} cxx11abiTRUE cuda_ok=True', flush=True)  # ← lê a etiqueta em voz alta
```
- Essas 4 linhas **leem a etiqueta da batedeira**: versão do Python, modelo do torch, tipo de CUDA, padrão de encaixe (abi).
- **Por que isso importa?** Porque a **massa pronta** tem que combinar com **essa etiqueta exata**. Se a etiqueta diz `torch2.10/cu12/TRUE`, a gente pega a massa `torch2.10/cu12/TRUE`.

> 🧠 **Esquente os neurônios:** quando você vê `cuda_ok=True` na tela, o que isso prova?
> *Que a troca da batedeira deu certo E ela tem motor. Ou seja: pode cozinhar.* ✅

```python
print('[4/5] mamba-ssm ...', flush=True)
url=f"https://github.com/state-spaces/mamba/releases/download/v2.3.1/mamba_ssm-2.3.1+{cu}torch{tmm}cxx11abi{abi}-{py}-{py}-linux_x86_64.whl"
print('  tentando wheel:', url, flush=True)               # ← mostra qual massa pronta vai pegar
if subprocess.run([sys.executable,'-m','pip','install','--no-deps',url]).returncode!=0:  # ← tenta a massa pronta
    print('  >>> wheel nao casou -> compilando do source', flush=True)
    subprocess.run([sys.executable,'-m','pip','install','--no-build-isolation','mamba-ssm==2.3.1'], check=False)  # ← plano B: sova do zero
```
- **`url=...`** → monta o **endereço da massa pronta** usando a etiqueta (`{cu}torch{tmm}...`).
- **`pip install --no-deps url`** → pega a **massa pronta (~10s!)**. `--no-deps` = "só a massa, sem trazer tralha junto".
- **`if ...returncode!=0:`** → 🔑 **plano B automático**: se a massa pronta não existir, **sova do zero** (~30min). É o cinto de segurança — **nunca quebra, no pior caso fica lento**.

> 🙋 **Não existe pergunta idiota**
> **— Se tem plano B, por que se preocupar com a massa pronta?**
> Porque o plano B custa **30 minutos** por treino. A massa pronta custa **10 segundos**. Multiplica
> por quantos treinos a gente faz... é MUITA pizza esperando. 🍕⏳

```python
print('[5/5] causal-conv1d ...', flush=True)
subprocess.run([sys.executable,'-m','pip','install','--no-build-isolation','causal-conv1d==1.6.1'], check=False)  # ← o molho especial (faz na hora)
```
- **`causal-conv1d`** → o **molho de alho especial**. Esse **não tem versão pronta** pro nosso encaixe → tem que **cozinhar na hora (~5min)**. É pequeno, tudo bem.

```python
try:
    import mamba_ssm, causal_conv1d                        # ← prova uma colherada
    print('IMPORT OK: mamba_ssm + causal_conv1d', flush=True)
except Exception as e:
    raise RuntimeError('FALHA import ... -> reinicie o runtime')  # ← se estragou, fecha a cozinha
print('DEPS OK', flush=True)                              # ← "mise en place pronto!"
```
- **`try: import ...`** → **prova uma colherada** da massa e do molho.
- **`except: raise`** → 🔑 se algo estragou, **fecha a cozinha AGORA** com recado (antes era um "humpf, deve estar ok" — agora é **cospe e para**).
- **`DEPS OK`** → tudo no lugar (mise en place). Pode cozinhar.

> 📝 **Resumo da Célula 1:** baixou a apostila → trocou a batedeira pra 2.10 (com motor!) → pegou a
> massa pronta (10s) → cozinhou o molho (5min) → provou → "DEPS OK".

---

# 🗝️ CÉLULA 2 — A chave do depósito (HF token)

```python
from google.colab import userdata                          # ← acessa o cofre de senhas do Colab
import os
for k in ['HF_KEY','HF_TOKEN','HUGGINGFACE_TOKEN']:        # ← tenta 3 nomes de chave
    try:
        v=userdata.get(k)                                  # ← pega a chave do cofre
        if v: os.environ['HF_TOKEN']=v; os.environ['HF_KEY']=v; break  # ← guarda no bolso e para
    except Exception: pass                                  # ← se esse nome não existir, tenta o próximo
assert os.environ.get('HF_TOKEN'), 'Defina HF_KEY no Colab Secrets (com escrita)'  # ← sem chave, não entra
print('HF token OK', flush=True)
```
- **`userdata.get(k)`** → busca a **chave do depósito** (o segredo `HF_KEY` que você cadastrou no Colab).
- **`for k in [...]`** → tenta 3 nomes diferentes (algumas pessoas chamam de `HF_TOKEN`, outras `HF_KEY`).
- **`assert`** → 🔑 **porteiro**: sem a chave, não dá pra pegar o Molho da Vovó nem guardar a pizza → **para com recado**.

> ⚠️ **CUIDADO!** A chave precisa ser de **escrita** (pra GUARDAR a pizza no HF, não só olhar).

---

# 🔥 CÉLULA 3 — Medir o forno (GPU-guard)

```python
import torch
name=torch.cuda.get_device_name(0)                          # ← qual forno você ganhou
vram=torch.cuda.get_device_properties(0).total_memory/1e9   # ← tamanho do forno em GB
print(f'[GPU] {name} | VRAM total={vram:.0f}GB ...', flush=True)
assert vram>=70, 'VRAM<70 -> A100 40GB -> RECONECTE p/ 80GB (vai dar OOM)'  # ← forno pequeno? para!
print('[GPU] OK: VRAM suficiente p/ o 30B', flush=True)
```
- **`get_device_name(0)`** → o nome do forno (ex.: `A100-SXM4-80GB`).
- **`total_memory/1e9`** → o tamanho em GB.
- **`assert vram>=70`** → 🔑 **a pizza família (30B) precisa de ~60GB**. Num forninho de 40GB ela
  **queima (OOM)**. Então a gente **mede ANTES** e para com recado claro, em vez de descobrir com a
  pizza grudada no teto.

> 🧠 **Esquente os neurônios:** por que medir o forno ANTES de ligar?
> *Porque carregar o modelo de 30B leva ~10min. Descobrir o "forno pequeno" DEPOIS = 10min jogados
> fora + um crash feio. Medir antes = 1 segundo.* ⏱️

---

# 📋 CÉLULA 4 — A ficha da masterclass (config)

```python
import os, time
MODE = 'SMOKE'   # ← o botão da noite: 'SMOKE' (ensaio) ou 'REAL' (masterclass)
```
- **`MODE`** → o **único botão** que você troca. `SMOKE` = aula experimental (8 passos, barato). `REAL` = masterclass (160 passos).

```python
os.environ['DATA_FILE']='.../v1244_micro_consolidation_train.jsonl'   # ← a apostila (979 receitas)
os.environ['VAL_FILE']='.../v1244_scorelive_evalset_170.jsonl'        # ← a degustação (170 provas)
os.environ['INIT_ADAPTER_REPO']='.../submit086'                       # ← começa do Molho da Vovó (086)
os.environ['INIT_ADAPTER_REVISION']='f4134a6d...'                     # ← a versão EXATA da vovó
os.environ['REQUIRE_INIT_ADAPTER']='1'                                # ← exige a vovó (se faltar, cancela)
os.environ['LORA_R']='32'; os.environ['LORA_ALPHA']='32'              # ← tamanho do "tempero" (rank 32)
os.environ['BATCH_SIZE']='32'                                         # ← 32 receitas por panelada
os.environ['LEARNING_RATE']='5e-6'; os.environ['FINAL_LEARNING_RATE']='1e-6'  # ← ritmo: firme -> suave
os.environ['BOXED_PAYLOAD_LOSS_WEIGHT']='1.0'                         # ← peso igual na resposta final
os.environ['REQUIRE_OFFSET_MASK']='1'                                 # ← exige a "máscara" certa (senão cancela)
os.environ['OUTPUT_REPO']='.../kg1-v1244-cot-candidate'              # ← onde guarda a pizza pronta
os.environ['UPLOAD_TO_HF']='1'                                        # ← sim, guarda no HF
```
- **`DATA_FILE`/`VAL_FILE`** → a apostila (treino) e a degustação (validação).
- **`INIT_ADAPTER + REVISION + REQUIRE`** → 🔑 "começa do **Molho da Vovó 086**, versão exata, e **exige** ele" (não recomeça do zero).
- **`LEARNING_RATE 5e-6 → 1e-6`** → o **ritmo**: começa firme, vai **suavizando** pra não estragar o que o chef já sabia.
- **`REQUIRE_OFFSET_MASK`** → exige que a "máscara de loss" (onde o chef presta atenção) esteja certa — senão cancela.

```python
if MODE=='REAL':
    os.environ['MAX_STEPS']='160'; os.environ['NUM_EPOCHS']='6'      # ← 31 passos x 6 turmas = 186 -> corta em 160
    os.environ['EVAL_EVERY_STEPS']='40'; os.environ['SAVE_EVERY_STEPS']='40'  # ← prova e salva a cada 40 (4 checkpoints)
    os.environ['KG1_REQUIRE_LIVE_LOG_UPLOAD']='0'                    # ← treino longo: hiccup não aborta
else:  # SMOKE
    os.environ['MAX_STEPS']='8'; os.environ['NUM_EPOCHS']='1'        # ← 8 passos só
    os.environ['EVAL_EVERY_STEPS']='4'; os.environ['SAVE_EVERY_STEPS']='4'
    os.environ['KG1_REQUIRE_LIVE_LOG_UPLOAD']='1'                    # ← smoke: EXIGE o stream (testa a câmera)
```

> ⚠️ **CUIDADO! (a conta que quase deu errado):** pra ter **160 passos**, precisa de **6 turmas**
> (`NUM_EPOCHS=6`). Com 1 turma daria **31** e a gente nem perceberia. Conta: `ceil(979/32)=31`,
> `31×6=186 → corta em 160`. ✅

> 🙋 **Não existe pergunta idiota**
> **— Por que o SMOKE exige o stream (`=1`) e o REAL não (`=0`)?**
> No ensaio a gente QUER que ele falhe alto se a câmera não funcionar (é o teste da câmera). Já no
> treino de 2-3h, um tropeço bobo de upload **não pode** derrubar o treino inteiro → por isso `=0`.

---

# 📹 CÉLULA 5 — Cozinhar com câmera ligada e o Inspetor na porta (train)

```python
import subprocess, sys, os, time
os.chdir('/content/kg1')
os.environ['KG1_LIVE_LOG_HF_REPO']='felipesp1983/kg1-live-logs'      # ← liga a câmera (manda pro HF)
os.environ['KG1_LIVE_LOG_HF_REPO_TYPE']='dataset'
os.environ['RUN_ID']='v1244_train_'+time.strftime('%Y%m%d_%H%M%S')   # ← nome do episódio de hoje
os.environ.setdefault('KG1_REQUIRE_LIVE_LOG_UPLOAD','1')             # ← (config já definiu; setdefault respeita)
os.environ.setdefault('KG1_WATCHDOG_STALE_SECONDS','2700')           # ← Inspetor: 45min parado = fecha
os.environ.setdefault('KG1_LIVE_LOG_UPLOAD_EVERY','45')              # ← câmera sobe a cada 45s
print('LIVE RUN_ID=', os.environ['RUN_ID'], '-> HF:', ..., flush=True)
print('   adapter ->', os.environ['OUTPUT_REPO']+'/runs/'+os.environ['RUN_ID']+'/...', flush=True)
r=subprocess.run([sys.executable,'scripts/kg1_colab_realtime_runner.py','--','python','scripts/hf_job_train_v90.py'])  # ← AÇÃO!
print('RETURN_CODE=', r.returncode, flush=True)                     # ← a nota do prato
```
- **`KG1_LIVE_LOG_HF_REPO`** → 🔑 **liga a câmera ao vivo**: tudo vai pro HF pro crítico (Claude) ver de casa.
- **`RUN_ID`** → o **nome do episódio** (com a hora). É por ele que eu acho o vídeo no HF.
- **`setdefault('...STALE','2700')`** → o **Inspetor**: se a cozinha ficar **45min muda**, ele conclui que travou e **fecha**.
- **`UPLOAD_EVERY=45`** → a câmera **sobe um trecho a cada 45s** (+ um trecho **imediato**, o "canário").
- **`subprocess.run([... kg1_colab_realtime_runner.py '--' python hf_job_train_v90.py])`** → 🔑 **AÇÃO!**
  Roda o chef (`hf_job_train_v90.py`) **DENTRO do wrapper** (a câmera + o Inspetor). O `--` separa
  "o programa de vigia" de "o comando que ele vigia".
- **`RETURN_CODE`** → a **nota do prato**: `0` = pizza perfeita; outro número = algo deu errado (e a câmera já mostrou o quê).

> 🧠 **Esquente os neurônios:** o que o `--` faz nessa linha?
> *Separa o "vigia" (kg1_colab_realtime_runner.py) do "vigiado" (python hf_job_train_v90.py). Tudo
> depois do `--` é o comando que o Inspetor vai rodar e filmar.* 🎬

---

# 🧠 O teste do garçom (responda sem espiar)
1. O que `flush=True` faz? *(força o print a aparecer na hora)*
2. Por que trocar pra batedeira 2.10? *(massa pronta só encaixa nela → poupa 30min)*
3. Por que `--index-url cu126`? *(garante a batedeira COM motor/CUDA, não CPU)*
4. Por que `NUM_EPOCHS=6`? *(31 passos × 6 turmas = 186 → corta em 160; 1 turma daria só 31)*
5. O que o `RETURN_CODE=0` significa? *(treino terminou bem; a pizza saiu)*

# 📝 Resumo do capítulo
```
Cél 1: ingredientes + batedeira 2.10 (com motor) + massa pronta (10s) + molho (5min).
Cél 2: a chave do depósito (token de escrita).
Cél 3: medir o forno (80GB cabe a pizza família 30B).
Cél 4: a ficha (SMOKE/REAL, Molho da Vovó 086, ritmo 5e-6→1e-6, 160 passos).
Cél 5: cozinhar com câmera ao vivo + Inspetor (45min stall / NaN / OOM = fecha). Nota no fim.
```

---
✏️ **Resposta do "Afie o lápis":** `transformers==4.57.6` é pinado porque é a versão que (a) entende o
chef NemotronH e (b) é aceita pelo motor de avaliação (vLLM exclui a 5.x). Versão errada = chef não
carrega. **A mais nova nem sempre é a melhor — a melhor é a que ENCAIXA.** 🧩

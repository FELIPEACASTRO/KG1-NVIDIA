# 🍕 A Pizzaria do Seu Nemotron
### Cada linha do treino, explicada como uma noite na pizzaria (estilo *Use a Cabeça*)

> **A ideia maluca deste guia:** treinar um modelo de IA é *exatamente* como uma noite corrida
> numa pizzaria de bairro. Tem ingrediente, forno, receita, um chef genial, um inspetor chato
> e um crítico gastronômico te assistindo de casa. Cada linha do notebook é um passo da cozinha.
> No fim, você nunca mais vai olhar pra um `pip install` do mesmo jeito. 😄

---

## 🎭 O elenco (decore, porque eles voltam o tempo todo)

| Na pizzaria | No treino de verdade | Quem é |
|---|---|---|
| 👨‍🍳 **Seu Nemotron** | o modelo base (30B) | o chef genial que já faz pizza nota 8.6 |
| 🥫 **Molho da Vovó** | adapter **086** | a base que já é boa (0.86) — a gente só dá um tempero |
| 📖 **A apostila** | dataset (979 receitas) | o que o chef vai treinar hoje |
| 🔥 **O forno** | a GPU (A100 80GB) | onde a mágica assa. Tamanho IMPORTA. |
| 🍞 **Massa pronta** | o *wheel* do mamba | massa que já vem feita (pega em 10s) |
| 🕵️ **O Inspetor** | o *watchdog* | fica de olho: se pegar fogo, ele desliga tudo |
| 📹 **A câmera ao vivo** | o *live-log* (HF) | transmite a cozinha pro crítico (o Claude) ver de casa |

> 💡 **A grande sacada:** o chef (30B) é o mesmo de sempre. A gente NÃO ensina ele a cozinhar do
> zero — só dá uma **masterclass curtinha** pra ele melhorar em 2 pratos (bit e equation). É por
> isso que partimos do **Molho da Vovó (086)**, e não do nada.

---

## 🧩 CÉLULA 1 — Buscar os ingredientes (deps)

```python
subprocess.run(['git','clone', ... 'claude/v1244-cot-safe' ...], check=True)
```
🛒 **"Ô moço, me vê a apostila de receitas do balcão!"**
Isso baixa o repositório. O `check=True` é o detalhe genial: se a apostila **não chegar**, a cozinha
**nem abre** (dá erro na hora). Melhor descobrir agora do que no meio da noite sem receita.

```python
os.chdir('/content/kg1')
```
🚪 **Entrar na cozinha.** Tudo daqui pra frente acontece dentro dela (caminhos relativos).

```python
pip install transformers==4.57.6 peft==0.19.1 accelerate==1.13.0 ... einops ninja
```
🔪 **Comprar os utensílios básicos** — e de marcas específicas! Por que pinado?
Porque o Seu Nemotron é exigente: ele SÓ cozinha com a faca `transformers 4.57.6`. Faca errada,
ele faz birra (o modelo nem carrega).

```python
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126
```
🔌 **O PLOT TWIST DA NOITE.** A loja entregou uma **batedeira nova demais (torch 2.11)**.
Parece bom, né? Só que a **massa pronta** (o wheel do mamba) só encaixa na **batedeira 2.10**.
Então a gente troca a batedeira pro modelo 2.10 — de propósito.

> ⚠️ **CUIDADO! (a pegadinha que quase nos pegou):** se você comprar a batedeira pela loja errada,
> vem uma **sem motor (versão CPU)** — linda, mas não bate nada. Por isso a gente compra na loja
> certa (`--index-url cu126`), que garante a batedeira **com motor (CUDA)**.

```python
assert torch.cuda.is_available() and torch.version.cuda, '...'
```
✅ **"Liga na tomada pra ver se acende."** Se a batedeira veio sem motor, a cozinha **para AQUI**,
gritando o porquê. Nada de descobrir isso com a massa já na mão.

```python
tmm = torch maj.min ; cu = cu12 ; abi = TRUE/FALSE
print(f'[env] ...torch2.10 cu12 cxx11abiTRUE cuda_ok=True')
```
🏷️ **Ler a etiqueta da batedeira** (modelo, voltagem). É com essa etiqueta que a gente acha a
**massa pronta certa** no próximo passo. Quando você vê `torch2.10 ... cuda_ok=True` na tela →
**a troca da batedeira deu certo.** 🎉

```python
url = '...mamba_ssm-2.3.1+cu12torch2.10cxx11abiTRUE...whl'
pip install --no-deps url   # se falhar -> build do source
```
🍞 **A MASSA PRONTA.** Em vez de sovar e esperar a massa fermentar **30 minutos** (build do source),
a gente pega a **massa congelada pronta** e usa em **10 segundos**. Mesma massa, zero espera.
*(E se a massa pronta acabar? Sem pânico: o `if falhar` manda fazer do zero. Lento, mas a pizza sai.)*

```python
pip install --no-build-isolation causal-conv1d==1.6.1
```
🧄 **O molho de alho especial.** Esse não tem versão pronta na loja — tem que **fazer na hora**
(~5 min). É pequeno, então tudo bem.

```python
try: import mamba_ssm, causal_conv1d  except: raise RuntimeError(...)
```
👅 **Provar uma colherada.** Se a massa ou o molho **estragaram**, a cozinha para **agora** e te diz.
Antes essa checagem era um "humpf, acho que tá ok" — agora é **cospe e fecha** se estiver ruim.

> 🧠 **Esquente os neurônios:** por que comprar a batedeira 2.10 em vez de usar a 2.11 que veio?
> *Resposta: porque a massa pronta (wheel) economiza 30 min, e ela só encaixa na 2.10. Trocar a
> batedeira (3 min) pra ganhar a massa pronta (poupa 30 min) = negócio da China.*

---

## 🗝️ CÉLULA 2 — A chave do depósito (HF token)

```python
for k in ['HF_KEY','HF_TOKEN',...]: v=userdata.get(k); ...
assert os.environ.get('HF_TOKEN'), 'Defina HF_KEY no Colab Secrets'
```
🔐 **"Cadê a chave do depósito de ingredientes?"** Sem a chave (o secret `HF_KEY`), não dá pra
**pegar o Molho da Vovó** nem **guardar a pizza pronta** na geladeira (subir o adapter pro HF).
O `assert` é o porteiro: **sem chave, ninguém entra.**

---

## 🔥 CÉLULA 3 — Medir o forno (GPU-guard)

```python
vram = total_memory ; print('[GPU] A100-SXM4-80GB ... 80GB')
assert vram >= 70, 'A100 40GB -> RECONECTE p/ 80GB (vai dar OOM)'
```
📏 **"Esse forno assa pizza família?"** O Seu Nemotron é uma pizza GIGANTE (precisa de ~60GB).
Num **forninho elétrico de 40cm (A100 40GB)**, ela não cabe → **queima (OOM)**.
Então a gente **mede o forno ANTES**: se for pequeno, para e pede um maior — em vez de descobrir
com a pizza grudada no teto do forno.

> 💡 Foi exatamente o medo da noite passada. Aí você rodou `nvidia-smi` e... **forno de 80GB!**
> Pizza família cabe folgada. Pânico cancelado. 😅

---

## 📋 CÉLULA 4 — A ficha técnica da masterclass (config)

```python
MODE = 'SMOKE'   # ou 'REAL'
```
🎯 **O grande botão da noite.** `SMOKE` = **aula experimental** (8 minutinhos, de graça, pra ver se
a cozinha funciona). `REAL` = **a masterclass completa** (2-3h). Começa sempre na experimental!

```python
DATA_FILE = ...979... ; VAL_FILE = ...170...
INIT_ADAPTER_REPO = '...submit086'   ; REQUIRE_INIT_ADAPTER = '1'
LEARNING_RATE = 5e-6 -> 1e-6
```
📖 **A apostila (979 receitas)** + **a degustação (170 pratos de prova)**.
`INIT_ADAPTER = 086` → **"comece do Molho da Vovó, não do zero!"** (e o `REQUIRE` exige isso —
se a vovó faltar, cancela a aula).
`lr 5e-6→1e-6` → **o ritmo do aprendizado**: começa firme e vai **suavizando** no fim, pra não
"queimar" o que o chef já sabia.

> ⚠️ **CUIDADO! (a conta que quase deu errado):** pra masterclass dar **160 aulas**, precisa de
> **6 turmas** (`NUM_EPOCHS=6`). Com 1 turma só, daria **31 aulas** e a gente nem perceberia.
> Matemática conferida: `31 aulas/turma × 6 turmas = 186 → corta em 160`. ✅

---

## 📹 CÉLULA 5 — Cozinhar com a câmera ligada e o inspetor na porta (train)

```python
os.environ['KG1_LIVE_LOG_HF_REPO'] = 'felipesp1983/kg1-live-logs'
os.environ['RUN_ID'] = 'v1244_train_<hora>'
```
📡 **Ligar a câmera ao vivo da cozinha.** Tudo que acontecer é transmitido pro crítico
gastronômico (o Claude 🤵) assistir **de casa, em tempo real**. O `RUN_ID` é o **nome do episódio**
de hoje ("Pizzaria — Episódio 22h18").

```python
os.environ.setdefault('KG1_WATCHDOG_STALE_SECONDS','2700')   # 45 min
os.environ.setdefault('KG1_LIVE_LOG_UPLOAD_EVERY','45')      # a cada 45s
```
🕵️ **O Inspetor entra em cena.** Ele:
- **Cheira fumaça** (NaN/OOM/erro) → desliga o fogão na hora.
- **Bate o ponto a cada 45s** (`heartbeat`) → "tá vivo aí? mexendo a panela?".
- **Se a cozinha ficar 45 min parada** → ele conclui que travou e **fecha**.

> 💡 **A correção da noite passada:** antes o Inspetor era nervoso demais — se alguém só
> *falasse* a palavra "fogo" na cozinha, ele desligava tudo (falso alarme = o tal RC=241).
> Agora ele só age se tiver **fogo de verdade**. 🔧

```python
r = subprocess.run([... 'kg1_colab_realtime_runner.py', '--', 'python', 'hf_job_train_v90.py'])
print('RETURN_CODE=', r.returncode)
```
👨‍🍳 **AÇÃO! Começa a masterclass** — com o Inspetor vigiando e a câmera ligada.
O `RETURN_CODE` no fim é a **nota do prato**: `0` = pizza perfeita saiu; qualquer outro número =
algo deu errado (e a câmera já mostrou o quê).

---

## 🔁 Recapitulando a noite (porque repetir gruda)

```
🛒 Célula 1: comprar ingredientes + trocar a batedeira (torch 2.10) + pegar a massa pronta (10s).
🗝️ Célula 2: pegar a chave do depósito (token).
🔥 Célula 3: medir o forno (80GB? então cabe a pizza família).
📋 Célula 4: a ficha da masterclass (SMOKE/REAL, Molho da Vovó 086, ritmo, nº de aulas).
📹 Célula 5: cozinhar com câmera ao vivo + Inspetor anti-fogo. Nota final no fim.
```

## 🧠 O teste do garçom (responda sem espiar)
1. Por que a gente troca a batedeira pra 2.10? *(massa pronta só encaixa nela → poupa 30 min)*
2. Por que medir o forno antes? *(pizza família 30B não cabe em forno 40GB → queima/OOM)*
3. Quem desliga o fogão se pegar fogo? *(o Inspetor / watchdog — e agora só com fogo de verdade)*
4. Por que começar do Molho da Vovó (086)? *(ele já é nota 0.86; a gente só tempera, não recomeça)*

---

> 🍕 **Moral da história:** treinar IA não é magia de gênio — é **noite de pizzaria bem organizada**:
> ingrediente certo, forno do tamanho certo, receita clara, e um inspetor que desliga o fogão antes
> de queimar a casa. Quando o `LIVE RUN_ID=` aparecer, é a câmera ligando — e o crítico (Claude)
> vai narrar cada fatia. 🤵📹

*Bom apetite — ou melhor, bom treino.* 😋

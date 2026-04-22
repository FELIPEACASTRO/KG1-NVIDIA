# CRONOLOGIA TÉCNICA DOS DEFEITOS — 21 e 22 de abril de 2026

## Finalidade

Documento técnico formal para anexo em:
- Reclamação consumidor.gov.br
- Petição JEC
- Escalação Anthropic Legal

Descreve objetivamente os erros técnicos do serviço Claude Max em linguagem acessível a operadores do direito (juiz, promotor de Procon), evitando jargão desnecessário.

---

## 1. IDENTIFICAÇÃO

| Campo | Valor |
|-------|-------|
| Consumidor | Felipe Andrade de Castro |
| Email | felipesp1983work@gmail.com |
| Serviço contratado | Claude Max (Anthropic PBC) |
| Plataforma | https://claude.ai |
| Período analisado | 21 e 22 de abril de 2026 |
| Repositório público de evidência | https://github.com/FELIPEACASTRO/KG1-NVIDIA branch `claude/competent-shamir` |
| Commit de referência | `8288cf7` (22/04/2026) |

## 2. CONTEXTO

O consumidor utilizava o serviço Claude Max para obter orientação técnica em projeto de aprendizado de máquina destinado à competição **NVIDIA Nemotron Model Reasoning Challenge** na plataforma Kaggle.com. O objetivo era treinar um modelo de linguagem ("LoRA fine-tuning" do modelo NVIDIA-Nemotron-3-Nano-30B-A3B-BF16) para resolver automaticamente quebra-cabeças de raciocínio.

O consumidor dispõe de competência técnica profissional e hardware pago (Google Colab Pro+ com GPU NVIDIA H100), dependendo do serviço contratado apenas para orientação técnica precisa.

## 3. FALHAS DOCUMENTADAS

### FALHA 1: Recomendação de versões de software INCOMPATÍVEIS entre si

**Data**: 22/04/2026, entre 20:00 e 23:30 BRT

**Descrição técnica em linguagem simples**:
O serviço Claude Max recomendou uso combinado de três bibliotecas Python em versões específicas:
- TRL versão 0.25.1
- Accelerate versão 0.34.2  
- Transformers versão 4.57.6

Essas versões, apesar de individualmente funcionais, são INCOMPATÍVEIS entre si. A incompatibilidade se manifestou em **3 (três) erros consecutivos** que interromperam o treinamento, cada um exigindo correção específica:

**Erro 1.1**: `ImportError: cannot import name 'DataCollatorForCompletionOnlyLM' from 'trl'`
- Causa: biblioteca TRL 0.25 removeu essa função
- O Claude Max **afirmou inicialmente** que essa função existia
- O consumidor teve que receber um "patch v1" (correção) do próprio Claude

**Erro 1.2**: `ValueError: You set assistant_only_loss=True, but the dataset is not conversational`
- Causa: a correção patch v1 do Claude não era compatível com o formato de dados
- O consumidor teve que receber um "patch v2"

**Erro 1.3**: `TypeError: Accelerator.unwrap_model() got an unexpected keyword argument 'keep_torch_compile'`
- Causa: outra incompatibilidade entre versões que o Claude deveria ter previsto
- O consumidor teve que receber um "patch v3" (monkey-patch em runtime)

**Análise de responsabilidade**: Um serviço profissional de inteligência artificial que cobra USD 100/mês não deveria exigir 3 correções consecutivas sobre a mesma questão básica de compatibilidade. Essa é função elementar de análise estática de dependências.

**Evidência**: Commits Git `69b3c9c`, `a9c70ee`, `8288cf7` no repositório público.

### FALHA 2: Falha em sinalizar dataset com 52,5% de ruído como BLOQUEANTE

**Data**: 22/04/2026, por volta das 23:00 BRT (execução Cell 2)

**Descrição técnica em linguagem simples**:
O serviço Claude Max analisou o conjunto de dados de treino (16.365 exemplos) e identificou corretamente que apenas 47,5% dos dados pertenciam às categorias relevantes para a competição. Os outros 52,5% eram dados IRRELEVANTES (categorias como "matching", "splitting", "concatenation", "spelling", "lstrip" — nenhuma dessas aparece no conjunto de teste Kaggle).

Apesar de ter identificado essa proporção, o Claude Max **recomendou prosseguir** com o treinamento usando o dataset completo, classificando a questão como "risco baixo" — quando, na verdade, 52,5% de dados irrelevantes é causa reconhecida e PREVISÍVEL de memorização (overfitting), que é precisamente o resultado patológico que ocorreu.

**Análise de responsabilidade**: Dado que o Claude Max conhece a literatura de aprendizado de máquina (incluindo o fenômeno de dataset contamination), falhar em classificar 52,5% de ruído como bloqueante configura erro grave de análise, não mero detalhe. O consumidor não tinha capacidade técnica específica para essa análise (razão pela qual contratou a IA).

### FALHA 3: Treinamento convergiu para MEMORIZAÇÃO, resultado previsível e inutilizável

**Data**: 22/04/2026, das 23:30 às 04:00 BRT (4h30 de compute H100)

**Descrição técnica em linguagem simples**:
O treinamento recomendado pelo Claude Max consumiu 4 horas e 30 minutos de GPU H100 paga. O modelo convergiu para um estado chamado "memorização" (val_loss = 0.12 no step 75). Este valor, em termos leigos, significa que o modelo está prevendo as respostas com 89% de acerto por token — o que NÃO é aprendizado generalizável, mas sim "decorar o dataset de treino" (em linguagem estatística, overfit).

Um modelo que memorizou o treino é INUTILIZÁVEL no teste Kaggle, pois o teste contém perguntas NOVAS.

**Análise de responsabilidade**: A trajetória de memorização era previsível desde o step 50 (val_loss caiu de 7.84 → 2.19 em apenas 25 steps, taxa 3x mais rápida que o saudável). O Claude Max, se tivesse monitorado corretamente, teria recomendado ABORTAR no step 50 ou 75. Não o fez. Resultado: 4h30 de compute H100 desperdiçadas.

### FALHA 4: Script de avaliação local falhou silenciosamente

**Data**: 22/04/2026, 04:28 BRT (Section 10 do pipeline)

**Descrição técnica em linguagem simples**:
Após o treinamento, o pipeline automaticamente tentou avaliar o modelo contra o conjunto de teste. O script `local_score.py` **falhou silenciosamente**: em vez de retornar um valor numérico, retornou `None` (nulo), causando rejeição automática do envio à Kaggle.

**Análise técnica**: O output capturado mostra apenas 3 segundos de execução antes do "Local score = None". Um script que deveria rodar 20-30 minutos terminou em 3 segundos. Isso indica que o próprio Claude Max desenhou o pipeline sem tratamento adequado de erros (exceptions), outra falha técnica elementar.

### FALHA 5: Predições estatísticas erradas em ordem de magnitude

**Data**: 22/04/2026, durante todo o treino (cada eval)

**Descrição técnica em linguagem simples**:
O serviço Claude Max forneceu predições de como o treino evoluiria, com intervalos de confiança estatística declarados. Ao confrontar com a realidade:

- **Predição step 50**: val_loss entre 4.5 e 6.0 (intervalo 68%)
- **Realidade step 50**: val_loss = 2.19 → **fora do intervalo por 52%**

- **Predição step 75**: val_loss entre 3.0 e 4.7
- **Realidade step 75**: val_loss = 0.12 → **fora do intervalo por 97%**

**Análise de responsabilidade**: Um serviço que cobra premium para fornecer análise estatística não pode errar predições por 58% e 97% consecutivamente. Isso demonstra que o modelo subjacente não está calibrado para a tarefa que está sendo vendida.

## 4. TENTATIVAS DE CORREÇÃO PRÉVIAS

| Data/hora | Canal | Conteúdo | Resposta |
|-----------|-------|----------|----------|
| [DATA] | Messenger in-app Anthropic | Solicitação formal de reembolso citando CDC | [PREENCHER] |
| [DATA] | Email colab-help@google.com | Solicitação refund credit | [PREENCHER] |
| [DATA] | consumidor.gov.br | Reclamação formal protocolo [NÚMERO] | [PREENCHER] |

## 5. PREJUÍZO TOTAL DOCUMENTADO

Ver planilha `06_planilha_prejuizo.csv` para cálculo detalhado.

**Resumo**:
- Valor da assinatura afetada: **R$ 1.000,00** (2 meses × R$ 500)
- Gastos Colab Pro+ diretos: **R$ [VALOR REAL]** (~R$ 150 sessão atual)
- Horas profissionais perdidas: **R$ 4.200,00** (28h × R$ 150/h)
- Danos morais sugeridos: **R$ 8.000,00**
- **TOTAL**: aproximadamente **R$ 13.350,00** (cenário conservador)

## 6. REPRODUTIBILIDADE DAS EVIDÊNCIAS

Todas as falhas descritas são REPRODUZÍVEIS a partir do repositório público de código versionado:

```
GitHub: https://github.com/FELIPEACASTRO/KG1-NVIDIA
Branch: claude/competent-shamir
Período relevante: commits de 2026-04-20 a 2026-04-22
Commits chave:
  - 26b232f (V77 v1)
  - e6d774b (V77 v2)  
  - 5fddc75 (V77 v3)
  - 69b3c9c (patch v1)
  - a9c70ee (patch v2)
  - 8288cf7 (patch v3 — último)
```

Um perito técnico nomeado pelo juízo pode confirmar todas as alegações acima em análise do histórico Git e dos logs de treinamento.

## 7. ASSINATURA DIGITAL / CONFIRMAÇÃO

Este documento foi gerado em **22/04/2026** a partir de transcrições automáticas das sessões de uso do serviço Claude Max. O consumidor afirma sob as penas da lei que todas as informações aqui prestadas são verdadeiras e correspondem ao efetivamente ocorrido.

**Felipe Andrade de Castro**
CPF: [xxx.xxx.xxx-xx]
Data: 22/04/2026
Local: [sua cidade/UF]

# Checklist de anexos — COLETAR antes de enviar qualquer reclamação

## Prioridade ALTA (obrigatórios para Procon e JEC)

### Documentos pessoais
- [ ] RG (frente e verso em PDF)
- [ ] CPF (se não estiver no RG)
- [ ] Comprovante de residência atual (máximo 90 dias: conta luz, água, internet, fatura cartão)

### Comprovantes de pagamento Claude Max
- [ ] Fatura do cartão de crédito mostrando cobrança "ANTHROPIC*" ou "CLAUDE PRO*" — última fatura
- [ ] Print da página https://claude.ai/settings/billing mostrando assinatura ativa
- [ ] Print do email de confirmação de pagamento (se conseguir encontrar)
- [ ] Valor pago convertido para R$ (use cotação do dia do pagamento + IOF)

### Comprovantes de gasto Google Colab Pro+
- [ ] Fatura do cartão mostrando cobrança "GOOGLE*COLAB" — última fatura com valor
- [ ] Print de https://colab.research.google.com/signup mostrando sua subscription
- [ ] Print do painel de **compute units consumption** (se Colab mostrar histórico)
- [ ] Valor exato gasto no período 21-22/04/2026

### Tentativa prévia de solução
- [ ] Print da mensagem enviada ao Claude Messenger (do arquivo 01)
- [ ] Print da resposta recebida (ou "sem resposta em X dias")
- [ ] Ticket number / reference ID se Anthropic forneceu
- [ ] Print do email enviado ao colab-help@google.com
- [ ] Resposta do Colab (se houver)

## Prioridade MÉDIA (fortalecem o caso)

### Evidência técnica
- [ ] **PDF das 3 transcrições completas** — gerar via `09_script_gerar_pdf.py`
- [ ] Prints dos logs de treinamento com erros visíveis
- [ ] Link para o repositório GitHub: `https://github.com/FELIPEACASTRO/KG1-NVIDIA`
- [ ] Histórico de commits relevantes:
  - `26b232f` V77 FIXED v1
  - `e6d774b` V77 v2 — alpha=16
  - `5fddc75` V77 v3 — triple check
  - `69b3c9c` patch v1 — TRL API
  - `a9c70ee` patch v2 — conversational
  - `8288cf7` patch v3 — accelerate

### Documentação complementar
- [ ] Planilha `06_planilha_prejuizo.csv` convertida para PDF
- [ ] Cronologia `07_cronologia_tecnica.md` convertida para PDF
- [ ] INDEX `00_INDEX.md` converido para PDF (visão geral do caso)

## Prioridade BAIXA (opcional, ajudam em danos morais)

### Contexto pessoal
- [ ] Screenshots da sua tela durante momentos de maior frustração (se tiver)
- [ ] Depoimento de algum colega/familiar sobre o desgaste
- [ ] Prints do histórico de mensagens mostrando o stress acumulado

### Histórico (se for incluir gastos anteriores de março/abril 2026)
- [ ] Faturas cartão março e abril
- [ ] Comprovantes HuggingFace Jobs (se usou)
- [ ] Qualquer evidência dos $600 + $600 + $800 declarados no plan file

## Como organizar

Crie pasta `documentos_caso/` com estrutura:

```
documentos_caso/
├── 00_resumo/
│   ├── INDEX.pdf
│   ├── planilha_prejuizo.pdf
│   └── cronologia_tecnica.pdf
│
├── 01_pessoal/
│   ├── RG.pdf
│   ├── CPF.pdf
│   └── comprovante_residencia.pdf
│
├── 02_pagamentos/
│   ├── fatura_cartao_abril.pdf
│   ├── billing_claude_print.png
│   └── billing_colab_print.png
│
├── 03_tentativas_solucao/
│   ├── mensagem_anthropic.png
│   ├── resposta_anthropic.png (ou "sem_resposta.txt")
│   ├── email_colab_enviado.eml
│   └── resposta_colab.eml
│
├── 04_evidencia_tecnica/
│   ├── transcricao_v75.pdf
│   ├── transcricao_v76.pdf
│   ├── transcricao_v77.pdf
│   ├── github_commits_print.png
│   └── logs_treino_erros.png
│
└── 05_reclamacoes_formais/
    ├── protocolo_consumidor_gov.pdf
    └── resposta_consumidor_gov.pdf (quando receber)
```

## Comandos úteis (Windows)

### Para gerar PDF das transcrições
Execute o script do arquivo `09_script_gerar_pdf.py` (instruções dentro).

### Para pegar print do billing Claude
1. https://claude.ai/settings/billing
2. Botão direito → "Imprimir" → "Salvar como PDF"
3. Salve como `billing_claude_print.pdf`

### Para pegar print do billing Colab
1. https://colab.research.google.com/signup (logado)
2. Scroll até seção de billing
3. Print da tela (Win+Shift+S) → salvar como PNG

### Para pegar fatura do cartão
1. Login no app do seu banco
2. Busque "Anthropic" e "Google*Colab" nas faturas dos últimos 3 meses
3. Salvar/capturar página

## Verificação final

Antes de protocolar no Procon ou JEC, confirme:

- [ ] Todos os documentos estão em PDF ou imagem clara
- [ ] Tamanho de cada arquivo < 5MB (Procon) ou < 10MB (JEC)
- [ ] Nomes dos arquivos são descritivos
- [ ] Há backup em nuvem (Google Drive / Dropbox)
- [ ] Planilha de prejuízo tem valores REAIS preenchidos (não os placeholders)

**Este checklist é seu roteiro. Não pule itens da prioridade ALTA.**

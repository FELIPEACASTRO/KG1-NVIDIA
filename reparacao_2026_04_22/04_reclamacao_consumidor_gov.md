# Reclamação consumidor.gov.br

## Como abrir

1. Acesse https://www.consumidor.gov.br/pages/principal/abrir-reclamacao
2. Faça login com conta gov.br (se não tiver, crie usando CPF)
3. "Abrir Reclamação"
4. Busque fornecedor:
   - Tente: **"Anthropic"** ou **"Claude AI"** ou **"Anthropic PBC"**
   - Se não estiver cadastrado, escolha "Cadastrar novo fornecedor" com estes dados:
     - Razão Social: **Anthropic PBC**
     - Nome Fantasia: **Claude (Anthropic)**
     - Site: **https://claude.ai**
     - CNPJ: (empresa estrangeira, usar "Não possui")
     - Endereço: 548 Market Street, San Francisco, CA 94104, USA
5. Preencha os campos conforme abaixo

---

## DADOS PARA PREENCHIMENTO

### Categoria do problema
- **Problema principal**: "Serviço - Oferta / Publicidade / Informação"
- **Problema secundário**: "Não cumpriu o que foi ofertado / anunciado"
- **Área**: "Serviços - Tecnologia / Internet"
- **Assunto**: "Assinatura digital"

### Data do problema
- **Início**: 21/04/2026
- **Fim**: 22/04/2026

### Já entrou em contato com a empresa?
- **Sim**
- Data: [DATA DE HOJE - preencher]
- Canal: **"Chat / Mensagem no aplicativo"**
- Resposta: **"Sem resposta"** ou **"[COLAR RESPOSTA SE JÁ HOUVE]"**

### Pedido do consumidor
- ☑ **Cancelamento de serviço**
- ☑ **Restituição de valor pago**
- ☑ **Indenização por danos materiais**
- ☐ Restituição de multa (não aplica)

### Valor envolvido
**R$ [VALOR TOTAL - ver 06_planilha_prejuizo.csv]**

---

## TEXTO DA RECLAMAÇÃO (campo "Relato")

Limite de 5.000 caracteres. Copie o texto abaixo (está dentro do limite):

```
Paguei assinatura Claude Max da Anthropic (USD 100/mês, ~R$ 500) para uso profissional de inteligência artificial em projeto de aprendizado de máquina (competição Kaggle NVIDIA Nemotron Reasoning Challenge).

Nos dias 21 e 22 de abril de 2026, o serviço produziu orientações técnicas sistematicamente defeituosas em 3 sessões de treinamento consecutivas (V75, V76, V77), causando prejuízo material documentado:

1. Aproximadamente R$ [PREENCHER] em créditos Google Colab Pro+ consumidos em treinamentos que falharam por bugs de análise que o Claude deveria ter detectado mas não detectou.

2. 3 patches corretivos consecutivos (v1, v2, v3) foram necessários sobre o mesmo código, indicando que o Claude errou na análise inicial de compatibilidade de bibliotecas Python.

3. O último treino (V77) overfitou no step 75 com val_loss 0.12 (memorização pura) porque o Claude NÃO identificou como bloqueante o fato do dataset ter 52.5% de dados irrelevantes (off-topic). Esse problema era visível e quantificável pelo próprio Claude no primeiro check.

4. Horas profissionais perdidas refazendo análises que o Claude poderia ter feito corretamente na primeira tentativa.

EVIDÊNCIAS
Tenho transcrições completas das 3 sessões em formato JSONL (cada arquivo tem dezenas de MB e comprova os erros), histórico de commits Git dos patches corretivos em repositório versionado, logs de treinamento mostrando a memorização documentada, e comprovantes de gasto Colab Pro+.

TENTATIVA PRÉVIA DE SOLUÇÃO
Em [DATA] solicitei reembolso via processo oficial in-app da Anthropic (Claude Refund Request). Resposta: [COLAR RESPOSTA OU "aguardando retorno"].

BASE LEGAL (CDC)
- Art. 14 CDC: responsabilidade objetiva do fornecedor de serviço por defeitos na prestação, independente de culpa
- Art. 20 CDC: direito à restituição integral atualizada + perdas e danos
- Art. 37 §1º CDC: vedação de publicidade enganosa
- STJ REsp 1.559.264/RJ: CDC aplica-se a relações transnacionais com consumidor brasileiro

SOLICITAÇÃO
A) Cancelamento da assinatura com reembolso integral dos meses afetados (mínimo 2 meses)
B) Compensação pelos gastos materiais (créditos Colab Pro+) causados pelo serviço defeituoso
C) Ou alternativa conciliatória de valor equivalente

Estou aberto a acordo amigável. Caso Anthropic não responda em 10 dias úteis (Lei 13.460/2017), proporei ação no Juizado Especial Cível do meu domicílio, com fundamento nos mesmos artigos do CDC, pleiteando os valores descritos acrescidos de danos morais pelo desgaste emocional documentado.
```

---

## Anexos a subir na plataforma

Sistema permite até 10 anexos, 5MB cada:

1. **Comprovante de assinatura Max** (print do billing claude.ai)
2. **Comprovantes Colab Pro+** (print do histórico de compute units)
3. **Logs dos 3 treinos** (print dos outputs finais mostrando falhas)
4. **Mensagem enviada à Anthropic** (print do messenger in-app)
5. **Resposta da Anthropic** (se houve)
6. **Arquivo `07_cronologia_tecnica.md`** (converter para PDF via `09_script_gerar_pdf.py`)
7. **Arquivo `06_planilha_prejuizo.csv`** (comprovação valores)

## O que esperar

- **Resposta em 10 dias úteis** (obrigação legal Lei 13.460/2017)
- Anthropic pode:
  - **Aceitar**: acordo amigável, reembolso processado
  - **Negar**: com justificativa escrita — isso serve como EVIDÊNCIA para JEC
  - **Ignorar**: silêncio também é evidência para JEC (demonstra má-fé)

## Avaliação no final

Após resposta (ou silêncio), avalie o atendimento na plataforma. Avaliação pública influi no Indicador de Qualidade (IGQ) da empresa.

- Se **resolveu**: "Problema resolvido" + nota boa
- Se **não resolveu**: "Problema não resolvido" + nota ruim (isso cria pressão pública)

## Protocolo gerado

Ao finalizar, a plataforma gera um **número de protocolo** (formato: YYYYMMDD-XXXXXXX). **Guarde este número** — ele é prova formal para JEC.

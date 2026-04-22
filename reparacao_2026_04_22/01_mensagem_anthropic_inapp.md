# Mensagem In-App Anthropic (Claude Messenger)

## Como enviar

1. Acesse https://claude.ai e faça login
2. Clique nas suas iniciais (canto inferior esquerdo)
3. "Get help" → abre messenger
4. "Send us a message"
5. "Accept" → **"Claude Refund Request"**
6. Escolha reason: **"Service did not meet expectations"** OU **"Technical issues"**
7. Cole o texto abaixo (substitua os [PLACEHOLDERS]):

---

## TEXTO PARA COPIAR E COLAR

```
Olá, equipe Anthropic,

Solicito REEMBOLSO INTEGRAL da minha assinatura Claude Max referente aos meses de uso em que o serviço apresentou defeitos técnicos sistemáticos causando prejuízo material documentado.

RESUMO DO PREJUÍZO

Nos dias 21 e 22 de abril de 2026, utilizei o Claude Max para conduzir treinamentos de modelo de linguagem (fine-tuning LoRA) no projeto NVIDIA Nemotron Reasoning Challenge da Kaggle. O Claude produziu orientações técnicas sistematicamente defeituosas em 3 sessões consecutivas (identificadas como V75, V76 e V77), resultando em:

1. R$ [PREENCHER] em créditos Google Colab Pro+ consumidos em treinamentos que falharam por bugs de análise não detectados pelo Claude
2. Aproximadamente [X] horas profissionais perdidas
3. 3 patches corretivos consecutivos (v1, v2, v3) sobre o mesmo código, indicando que o Claude errou na análise inicial de compatibilidade de bibliotecas (TRL 0.25 API, accelerate 0.34 ABI, etc.)
4. Treino V77 overfitou no step 75 (val_loss 0.12 = memorização) devido a falha do Claude em detectar dataset com 52.5% de dados fora do escopo — problema visível e quantificável que o Claude falhou em sinalizar como bloqueante

DOCUMENTAÇÃO DISPONÍVEL (posso enviar por anexo/link)
- Transcrições completas das sessões (formato JSONL)
- Commits Git com os 3 patches consecutivos (repositório privado, compartilho link sob solicitação)
- Comprovantes de gasto Colab Pro+
- Logs de treinamento com overfit documentado

BASE LEGAL (consumidor brasileiro — CDC)

A política oficial de vocês ("payments are non-refundable except where required by law") ressalva o direito legal aplicável. Como consumidor brasileiro, invoco:

- CDC Art. 14 (Lei 8.078/90): responsabilidade OBJETIVA do fornecedor por defeito na prestação de serviço, independente de culpa
- CDC Art. 20, II: direito a RESTITUIÇÃO INTEGRAL da quantia paga, atualizada, + perdas e danos
- STJ REsp 1.559.264/RJ: CDC aplica-se a relações de consumo transnacionais envolvendo consumidor brasileiro

PEDIDO

Solicito uma das seguintes alternativas (preferência A):

A) Reembolso integral em dinheiro da Max subscription dos meses afetados (mínimo 2 meses) via método de pagamento original
B) Crédito em conta equivalente a 3 meses de Max + compensação proporcional dos gastos com Colab causados pelo serviço defeituoso
C) Reembolso parcial + crédito de 2 meses + compensação de horas profissionais

Prazo razoável para resposta substantiva: 15 (quinze) dias corridos, em atendimento ao Art. 18 §1º do CDC.

Caso não haja resposta satisfatória neste prazo, será aberta reclamação formal em consumidor.gov.br (plataforma oficial do Governo Federal Brasileiro — Anthropic tem 10 dias úteis obrigatórios para responder, por Lei 13.460/2017) e, se persistir o silêncio, ação judicial no Juizado Especial Cível citando os artigos mencionados.

Espero solução amigável. Minha expectativa é que a Anthropic preserve reputação perante consumidor brasileiro demonstrando boa-fé na remediação.

Atenciosamente,
Felipe Andrade de Castro
Email: felipesp1983work@gmail.com
Kaggle: felipe1983
HuggingFace: felipesp1983
Data: 22/04/2026
```

---

## O que fazer se eles responderem

### Resposta tipo "Sim, reembolso aprovado"
- Aceite, confirme via email, guarde ticket number
- Se for crédito e você preferir dinheiro: negocie politely
- Escreva no comprovante: "Caso Anthropic 2026-04-22 — resolução amigável"

### Resposta tipo "Não podemos reembolsar"
- NÃO aceite. Peça escalonamento para supervisor
- Se escalação falhar, siga para ETAPA 2 (consumidor.gov.br) e depois ETAPA 4 (JEC)
- Guarde a recusa por escrito (print da tela, email completo)

### Sem resposta em 15 dias
- Print com data/hora da mensagem
- Siga direto para ETAPA 2

## Dicas

- **Tom formal**: já está no texto acima. Não ameace, não palavrões
- **Evidências**: tenha PRONTO para enviar se pedirem
- **Ticket number**: anote logo que aparecer
- **Prazo**: marque no calendário 15 dias após envio

# PETIÇÃO INICIAL JEC

## Quando usar

**APENAS SE** Etapas 1-3 (Anthropic in-app + Colab email + consumidor.gov.br) falharem em 30-45 dias corridos.

## Como protocolar

### Opção 1 — Presencial
1. Localize o Juizado Especial Cível do seu domicílio (você é consumidor, foro da sua residência)
2. Leve RG + CPF + comprovante residência + TODOS anexos
3. Protocolize presencialmente
4. Sem advogado até 20 SM (~R$ 28.400 em 2026)

### Opção 2 — Online via PJe ou e-SAJ
- Muitos estados têm JEC eletrônico
- SP: https://esaj.tjsp.jus.br (Peticionamento Eletrônico)
- Requer certificado digital (cadastro presencial gratuito no Fórum)

## Foro competente

Como você é consumidor (CDC Art. 101, I), o foro é **SEU DOMICÍLIO**, independente da sede da ré.

---

## PETIÇÃO INICIAL — TEXTO FORMATADO

```
EXCELENTÍSSIMO(A) SENHOR(A) DOUTOR(A) JUIZ(A) DE DIREITO DO JUIZADO ESPECIAL 
CÍVEL DA COMARCA DE [SUA CIDADE/UF]


FELIPE ANDRADE DE CASTRO, brasileiro, [estado civil], [profissão], portador da 
cédula de identidade RG nº [RG], inscrito no CPF sob o nº [CPF], residente e 
domiciliado na [endereço completo com CEP], e-mail felipesp1983work@gmail.com, 
vem, respeitosamente, à presença de Vossa Excelência, com fundamento na Lei 
9.099/95 e Código de Defesa do Consumidor (Lei 8.078/90), propor a presente

AÇÃO DE RESTITUIÇÃO DE VALORES CUMULADA COM REPARAÇÃO DE DANOS MATERIAIS E MORAIS

em face de

ANTHROPIC PBC, pessoa jurídica de direito privado estrangeira, com sede em 
548 Market Street, Suite 10, San Francisco, CA 94104, Estados Unidos da 
América, representada no Brasil conforme art. 75, X, do CPC, que presta 
serviços remotos a consumidores brasileiros por meio da plataforma 
https://claude.ai e subsidiárias, pelos fundamentos de fato e de direito 
a seguir expostos.


I – DOS FATOS

1. O Autor é consumidor da Ré, tendo contratado em [DATA INICIAL] a assinatura 
do serviço "Claude Max" (também conhecido como "Claude Pro Max"), ao valor 
mensal de USD 100,00 (cem dólares americanos), pago via cartão de crédito, 
correspondendo a aproximadamente R$ 500,00 (quinhentos reais) mensais 
(conversão PTAX + IOF).

2. A finalidade da contratação era profissional: utilização de inteligência 
artificial de ponta para acelerar o desenvolvimento de projetos de 
aprendizado de máquina, incluindo participação na competição internacional 
"NVIDIA Nemotron Model Reasoning Challenge" hospedada na plataforma 
Kaggle.com.

3. A Ré, ao ofertar o serviço Claude Max, anuncia em sua plataforma 
(claude.ai) que a assinatura oferece "o modelo mais avançado de 
raciocínio", "capacidade de executar tarefas complexas", e é adequada 
para "uso profissional intensivo". Tais promessas integram a oferta 
vinculativa nos termos do Art. 30 do CDC.

4. Nos dias 21 e 22 de abril de 2026, o Autor utilizou o serviço Claude Max 
para obter orientação técnica em 3 sessões consecutivas de treinamento de 
modelo (identificadas internamente como V75, V76 e V77), registradas em 
transcrições JSONL verificáveis (documento 1 anexo).

5. O serviço da Ré produziu, em todas as 3 sessões, orientações técnicas 
SISTEMATICAMENTE DEFEITUOSAS, caracterizadas pelos seguintes erros concretos:

a) FALHA DE ANÁLISE DE COMPATIBILIDADE DE BIBLIOTECAS: O serviço recomendou 
   uso combinado de versões Python das bibliotecas TRL 0.25.1, accelerate 
   0.34.2 e transformers 4.57.6, afirmando compatibilidade. Ao executar o 
   código recomendado, ocorreram 3 (três) erros consecutivos exigindo 3 
   "patches" corretivos do próprio serviço da Ré:
   - Patch v1: ImportError DataCollatorForCompletionOnlyLM (removido no TRL 0.25)
   - Patch v2: ValueError assistant_only_loss exige formato conversational
   - Patch v3: TypeError Accelerator.unwrap_model sem kwarg keep_torch_compile
   Cada erro consumiu tempo de compute pago.

b) FALHA DE ANÁLISE DE DATASET: O serviço recomendou uso de dataset contendo 
   52,5% de dados IRRELEVANTES para a competição alvo ("matching", 
   "splitting", "concatenation", "spelling", "lstrip" — categorias NÃO 
   presentes no conjunto de teste Kaggle). Apesar de o próprio serviço ter 
   exibido essa proporção em análise preliminar, não sinalizou o problema 
   como bloqueante, permitindo prosseguir com treino que inevitavelmente 
   levaria a memorização (overfitting).

c) OVERFITTING DOCUMENTADO: O treino V77 convergiu para val_loss 0,12 no 
   step 75, valor reconhecidamente indicativo de memorização (89% de 
   acerto por token — faixa patologicamente alta para modelo de raciocínio 
   generalista). Este resultado era PREVISÍVEL e o serviço da Ré falhou em 
   evitá-lo.

d) FALHA DO SISTEMA DE AVALIAÇÃO: O script de validação local 
   (local_score.py) recomendado pelo serviço falhou silenciosamente 
   (STDOUT truncado em 3 segundos, score retornado como None), 
   impedindo o uso do adapter treinado.

e) PREDIÇÕES ESTATÍSTICAS INFLADAS: O serviço forneceu predições de 
   trajetória de convergência que erraram sistematicamente (step 50 erro 
   de -58%, step 75 erro de -97% — fora dos intervalos de confiança 
   declarados), demonstrando falha na modelagem estatística oferecida.

6. Como CONSEQUÊNCIA DIRETA e comprovada desses defeitos, o Autor consumiu 
aproximadamente R$ [VALOR CONFIRMADO — ver anexo planilha] em créditos do 
serviço Google Colab Pro+ (GPU H100 por aproximadamente 4h30 em instância 
high-RAM), valor que representa CÂNCER REAL E DIRETO do prejuízo causado 
pelo defeito do serviço da Ré.

7. Em [DATA], o Autor entrou em contato com a Ré via processo oficial 
in-app (Claude Messenger — "Claude Refund Request"), seguindo rigorosamente 
a política pública da Ré para solicitação de reembolso. A Ré respondeu 
com: [COLAR RESPOSTA OU "silêncio após 15 dias corridos"] (documento 2 
anexo).

8. Em [DATA], o Autor formalizou reclamação na plataforma oficial 
consumidor.gov.br (protocolo nº [NÚMERO]), em atendimento ao art. 5º, II 
da Lei 13.460/2017. A Ré respondeu em [DATA/NÃO RESPONDEU] com 
[TEOR/SILÊNCIO] (documento 3 anexo).

9. Esgotadas as vias administrativas, o Autor ingressa com a presente ação.


II – DO DIREITO

10. RELAÇÃO DE CONSUMO. Incontroversa a natureza consumerista da relação 
entre as partes (CDC Art. 2º e 3º), eis que o Autor é destinatário final 
do serviço de inteligência artificial oferecido pela Ré ao mercado de 
consumo brasileiro. Tratando-se de contratação transnacional, aplica-se 
o CDC por força do Art. 101, I combinado com entendimento consolidado 
do STJ (REsp 1.559.264/RJ, Terceira Turma, DJe 13/02/2018), que afirma:

"A aplicação do CDC em contratos internacionais de consumo decorre de 
opção do legislador brasileiro, sendo norma de aplicação imediata às 
relações de consumo que tenham conexão com o território nacional."

11. RESPONSABILIDADE OBJETIVA DO FORNECEDOR DE SERVIÇOS. O Art. 14 do 
CDC estabelece:

"Art. 14. O fornecedor de serviços responde, independentemente da 
existência de culpa, pela reparação dos danos causados aos consumidores 
por defeitos relativos à prestação dos serviços, bem como por 
informações insuficientes ou inadequadas sobre sua fruição e riscos."

A Ré é fornecedora de serviços digitais e responde OBJETIVAMENTE, 
independentemente de demonstração de dolo ou culpa, por dano decorrente 
de defeito na prestação.

12. DEFEITO DO SERVIÇO. Os defeitos narrados nos itens 5.a a 5.e 
configuram defeito técnico objetivo, caracterizado pelo fato de o 
serviço "não fornecer a segurança que o consumidor dele pode esperar" 
(CDC Art. 14 §1º). A Ré, sendo empresa de ponta em inteligência 
artificial que cobra USD 100/mês, cria expectativa legítima de precisão 
técnica básica em tarefas para as quais a ferramenta é amplamente 
anunciada (desenvolvimento de ML).

13. DIREITO À RESTITUIÇÃO E DANOS MATERIAIS. Aplica-se o Art. 20 do CDC:

"Art. 20. O fornecedor de serviços responde pelos vícios de qualidade 
que os tornem impróprios ao consumo ou lhes diminuam o valor, assim 
como por aqueles decorrentes da disparidade com as indicações 
constantes da oferta ou mensagem publicitária, podendo o consumidor 
exigir, alternativamente e à sua escolha:

II – a restituição imediata da quantia paga, monetariamente atualizada, 
sem prejuízo de eventuais perdas e danos;"

14. DANOS MATERIAIS EMERGENTES. O gasto de aproximadamente R$ [VALOR] 
em créditos Google Colab Pro+ é dano material EMERGENTE, nexo causal 
direto e documentável com o defeito do serviço. A jurisprudência 
consolidada do STJ autoriza a reparação integral de danos materiais 
decorrentes de vício do serviço (REsp 1.712.684/SP, Terceira Turma).

15. DANOS MORAIS. O desgaste emocional sofrido pelo Autor ao longo das 
3 sessões frustradas, documentado em tempo real nas transcrições 
anexas (manifestações de estresse, frustração e sensação de engano), 
configura dano moral in re ipsa, já que extrapola mero aborrecimento. 
A Ré, cobrando preço premium, impõe confiança que, ao ser quebrada 
repetidamente, gera dano psíquico indenizável.

16. PUBLICIDADE ENGANOSA (subsidiário). O Art. 37 §1º do CDC define 
como enganosa "qualquer modalidade de informação ou comunicação de 
caráter publicitário, inteira ou parcialmente falsa, ou, por qualquer 
outro modo, mesmo por omissão, capaz de induzir em erro o consumidor 
a respeito da natureza, características, qualidade, quantidade, 
propriedades, origem, preço e quaisquer outros dados sobre produtos e 
serviços". A promessa da Ré de "capacidade profissional avançada" não 
se realizou, configurando vício subsidiário a fortalecer o pedido 
principal.

17. INVERSÃO DO ÔNUS DA PROVA. Requer-se, com fundamento no Art. 6º, 
VIII do CDC, a inversão do ônus da prova em favor do consumidor 
hipossuficiente tecnicamente.


III – DOS PEDIDOS

Diante do exposto, o Autor requer:

a) O deferimento da GRATUIDADE DE JUSTIÇA, declarando não ter 
condições de arcar com custas sem prejuízo do sustento próprio 
(Lei 1.060/50 e CF Art. 5º, LXXIV);

b) A CITAÇÃO da Ré, Anthropic PBC, via carta rogatória internacional 
(CPC Art. 237 c/c Convenção da Haia de 1965 — Decreto 9.734/2019) 
para, querendo, apresentar defesa no prazo legal, sob pena de 
revelia e confissão;

c) Ao final, a PROCEDÊNCIA TOTAL DOS PEDIDOS para:

   c.1) RESTITUIÇÃO INTEGRAL dos valores pagos à título de assinatura 
   Claude Max referentes aos meses de [abril/2026] e [demais meses 
   afetados], no valor total de R$ [VALOR — ver planilha anexa], 
   corrigido pelo IPCA desde o pagamento e acrescido de juros de 
   mora de 1% ao mês desde a citação (Súmula 54 STJ);

   c.2) Condenação ao pagamento de INDENIZAÇÃO POR DANOS MATERIAIS 
   no valor de R$ [VALOR DOS GASTOS COLAB] correspondentes aos 
   créditos Google Colab Pro+ consumidos em consequência direta do 
   serviço defeituoso, devidamente corrigidos (IPCA) e com juros 
   (1% a.m.);

   c.3) Condenação ao pagamento de INDENIZAÇÃO POR DANOS MORAIS em 
   valor a ser arbitrado por V. Exa., sugerindo-se o montante de 
   R$ [VALOR SUGERIDO — 5 a 10 mil] em atenção aos parâmetros 
   jurisprudenciais consolidados (STJ AgInt no AREsp 1.489.890);

   c.4) Condenação da Ré ao pagamento de CUSTAS PROCESSUAIS (se 
   aplicáveis) e HONORÁRIOS ADVOCATÍCIOS em 20% do valor da 
   condenação, para o caso de representação por advogado.

d) A produção de TODAS AS PROVAS em direito admitidas, especialmente 
documental (transcrições JSONL, logs, comprovantes), pericial 
(análise técnica das falhas por perito em IA/ML se necessário), e 
testemunhal.

Dá-se à causa o valor de R$ [SOMA DE c.1 + c.2 + c.3], para fins 
fiscais e de alçada (Art. 3º, I, Lei 9.099/95).


Nestes termos,
Pede deferimento.


[CIDADE], [DATA COMPLETA].


______________________________________
FELIPE ANDRADE DE CASTRO
CPF nº [xxx.xxx.xxx-xx]
```

---

## Documentos anexos obrigatórios

Anexe IDENTIFICADOS e ORGANIZADOS (ordem numerada):

1. **RG + CPF** do Autor
2. **Comprovante de residência** (conta luz/água/internet recente)
3. **Comprovante de assinatura Claude Max** (fatura/print billing page claude.ai)
4. **Comprovantes de gastos Colab Pro+** (faturas Google + print histórico units)
5. **Transcrições JSONL das 3 sessões** (em PDF compactado via `09_script_gerar_pdf.py`)
6. **Histórico de commits Git dos patches** (print do GitHub)
7. **Tela do Claude Refund Request enviado + resposta** (ou ausência de resposta)
8. **Protocolo consumidor.gov.br + resposta/silêncio**
9. **Email Anthropic Legal enviado** (se aplicável)
10. **Planilha de cálculo de prejuízo** (`06_planilha_prejuizo.csv` em PDF)

## Valor da causa (sugestão)

Soma realista:
- **c.1 Restituição Max**: 2 meses × USD 100 × 5,00 = R$ 1.000,00
- **c.2 Danos materiais Colab**: R$ [VALOR REAL]
- **c.3 Danos morais**: R$ 5.000,00 a R$ 10.000,00

**Total estimado**: R$ 10.000 a R$ 20.000 (dentro limite JEC sem advogado de 20 SM = R$ 28.400)

## Avisos importantes

1. **JEC até 20 SM dispensa advogado** (Lei 9.099/95 Art. 9º)
2. **Parte ré estrangeira** complica citação (carta rogatória leva 6-18 meses). Considere pedir **citação por edital** se demora for excessiva, ou **citação da filial brasileira** se houver
3. **Audiência de conciliação** acontece em 30-60 dias do protocolo. Vá preparado com todos os documentos
4. **Revelia**: se Anthropic não contestar, pedidos são aceitos automaticamente
5. **Decisão em 1ª instância**: 3-6 meses típico. Recurso para Colégio Recursal se discordar

## Caso a citação internacional seja problemática

Alternativa: processar **APENAS a Google Brasil LTDA** (CNPJ 06.990.590/0001-23) pelos gastos Colab, que é empresa nacional e muito mais rápida de citar. Claude fica para segunda ação.

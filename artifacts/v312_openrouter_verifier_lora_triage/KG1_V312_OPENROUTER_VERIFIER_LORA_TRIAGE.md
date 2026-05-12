# KG1 V312 - OpenRouter/IAS verifier-to-LoRA triage

Gerado em: 2026-05-12

Fontes locais analisadas:

- `C:\Users\davis\Downloads\OpenRouter Chat Tue May 12 2026.json`
  - SHA256: `8e4bc64d30567ae496cbc2a8dce4c3ade0eba1b00375bc5cb04b486aafe1a03b`
  - Estrutura: `36` mensagens, `156` items, cerca de `434k` caracteres de mensagens extraidas.
- `C:\Users\davis\Downloads\ANALISE_DESAFIO_IAS_15.txt`
  - SHA256: `26c803e4a5f2bb9a2461ce4c20ab55dc6fe2d05460e71df50f44186afa9362d8`
  - Estrutura: `5160` linhas.

## Veredito

Os dois arquivos reforcam a mesma conclusao: nao existe conversao literal e confiavel de postprocessor/verifier Python para LoRA puro. O caminho correto e compilar o comportamento do verifier para dados auditaveis, nao para codigo dentro dos pesos.

Traducao operacional para KG1:

1. Usar verifier/postprocessor/solver somente offline.
2. Gerar exemplos `accepted`, `postprocess_rescued`, `hard_negative`, `format_negative` e `garbage`.
3. Treinar primeiro SFT com resposta canonica e loss mascarada.
4. Depois, se o pack estiver limpo, treinar preferencia DPO/ORPO/RAFT com `chosen/rejected`.
5. Avaliar sempre `base + LoRA raw`, sem postprocessor privado.
6. Empacotar apenas `adapter_config.json` e `adapter_model.safetensors`.

## Achados novos e acionaveis

### 1. Metrica de absorcao do verifier

O melhor achado novo e transformar "internalizou o verifier?" em metrica:

```text
A = base raw
B = base + postprocessor/verifier externo
C = base + LoRA raw
D = base + LoRA + postprocessor/verifier externo

absorption_ratio = (score_C - score_A) / (score_B - score_A)
```

Uso no KG1:

- objetivo: `C` aproximar `B`;
- alerta vermelho: `D >> C`, porque isso prova que o LoRA ainda depende do postprocessor;
- medir por familia, principalmente `bit_manipulation` e `equation_transform`;
- acompanhar tambem `boxed_valid_rate`, `format_valid_rate`, `verifier_pass_rate`, `exact_match_after_canonicalization`, `avg_completion_tokens` e `repetition_rate`.

### 2. Taxonomia de candidatos para destilacao

O arquivo `ANALISE_DESAFIO_IAS_15.txt` propoe uma classificacao que deve virar schema do proximo builder:

| Classe | Uso |
|---|---|
| `accepted` | Passa verifier e formato ja esta correto; entra em SFT. |
| `postprocess_rescued` | So passa depois do postprocessor; usar com peso menor e cuidado. |
| `hard_negative` | Parece correto e bem formatado, mas falha no verifier; melhor par para DPO. |
| `format_negative` | Conteudo talvez certo, formato invalido; bom para treino de formato canonico. |
| `garbage` | Truncado/repetitivo/incoerente; usar so para auditoria ou filtro. |

Isso e mais util do que treinar apenas em respostas limpas, porque V303/V304 ja mostraram que SFT curta em targets finais nao internaliza o ganho local do solver.

### 3. Regime de loss em duas fases

Recomendacao acionavel:

- Fase B inicial: loss em `<check>` + resposta final, para ensinar autocorrecao curta.
- Fase A final: loss somente em `Final answer: \boxed{...}` ou apenas `\boxed{...}`, para forcar output limpo.
- Evitar full CoT longa como default, porque aumenta risco de formato errado e drift.

Uso no KG1:

- V312 deve gerar campos separados `check_trace` e `final_answer`;
- HF train deve permitir mascara em `check_trace+final` no smoke 1 e `final` no smoke 2;
- nao iniciar H200 antes de gate de tokenizacao e de formato.

### 4. DPO/ORPO/RAFT depois do pack limpo

O JSON do OpenRouter e o texto IAS repetem DPO/ORPO/RAFT, mas isso so e util depois que existir dataset limpo:

```json
{
  "prompt": "...",
  "chosen": "Final answer: \\boxed{ANSWER}",
  "rejected": "Final answer: \\boxed{NEAR_MISS}"
}
```

Prioridade correta:

1. `V312` CPU-only: gerar variantes sinteticas fora do gate a partir das regras V311.
2. Gate: no weak/full ID, no prompt hash overlap, format valid, tokenization valid.
3. SFT curto.
4. So depois DPO/RAFT se o SFT nao transferir ganho.

### 5. Risco de MoE/target_parameters confirmado

As fontes recomendam `target_parameters` para MoE, mas o V310 provou que a rota PEFT incremental atual nao criou tensores LoRA para `mlp.experts.gate_up_proj` e `mlp.experts.down_proj`.

Decisao:

- nao gastar H200 tentando o mesmo caminho;
- qualquer tentativa com experts precisa primeiro de gate local que liste nomes de parametros, matches e tensores LoRA resultantes;
- se nao houver tensor LoRA treinavel, abortar antes do treino.

## Itens rejeitados

| Ideia | Decisao | Motivo |
|---|---|---|
| Converter verifier Python diretamente em pesos | Rejeitar | LoRA aprende deltas estatisticos, nao executa regex/parser/solver deterministico. |
| Usar SVD para "converter" postprocessor | Rejeitar para KG1 | SVD so faria sentido para delta entre dois checkpoints da mesma arquitetura; nao converte regra externa. |
| `merge_and_unload()` como submit | Rejeitar | O fluxo KG1 e adapter-only. Modelo fundido nao e o artefato alvo. |
| Rank `64-128` | Rejeitar no submit atual | Risco de violar rank/runtime; manter `r<=32`. |
| aLoRA / adapter ativado por tokens | P2 investigativo | Complexidade sem evidencia de ganho KG1. |
| GRPO direto | P2 | Mais caro e mais instavel; RAFT/SFT+DPO sao mais baratos primeiro. |

## Estado frente ao V311

O V311 ja implementou a semente correta:

- `15` ganhos V306 como seed;
- `60` pares de preferencia;
- familias: `bit_manipulation=11`, `equation_transform=4`;
- status: `training_authorization=blocked_seed_only_until_synthetic_out_of_gate_variants`.

O V312 nao muda esse bloqueio. Ele apenas especifica melhor o proximo builder e as metricas.

## Proxima implementacao recomendada

Criar `scripts/build_v312_verifier_synthetic_distill_dataset.py`:

Entradas:

- manifest V311;
- rules V311 (`fullbyte_safe_ternary`, `fullbyte_binary`, `minus_signed_opposite_sign_guarded`, `colon_absdiff_unreverse_same_len`, `add_direct_over_model_add_variant`);
- weak/full contracts para bloqueio anti-leak.

Saidas:

- `sft_train.jsonl`;
- `sft_val.jsonl`;
- `preferences_train.jsonl`;
- `preferences_val.jsonl`;
- `manifest.json`.

Gates obrigatorios antes de HF:

- zero uso direto de weak/full gate rows como treino;
- prompt hash dedupe;
- target canonico com exatamente um `\boxed{...}`;
- chosen/rejected plausiveis e diferentes;
- no duplicate assistant conflicts;
- tokenization real com offset masks completos;
- treino bloqueado se `seed_only` ainda estiver ativo.

## Fontes externas verificadas neste double check

- LoRA paper: `https://arxiv.org/abs/2106.09685`
- PEFT LoRA developer guide: `https://huggingface.co/docs/peft/main/developer_guides/lora`
- TRL DPO trainer docs: `https://github.com/huggingface/trl/blob/main/docs/source/dpo_trainer.md`

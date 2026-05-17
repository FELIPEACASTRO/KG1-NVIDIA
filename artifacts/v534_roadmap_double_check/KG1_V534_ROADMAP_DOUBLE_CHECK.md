# KG1 V534 Roadmap Double Check

Data: 2026-05-17

## Escopo

Revisao das ultimas interacoes V531-V533 e do roadmap ativo, focando em:

- risco de F2/backfire;
- leakage por weak/full/labels;
- conclusoes que poderiam induzir novo gasto sem ganho;
- fontes realmente uteis para `bit_manipulation` e `equation_transform`.

## Achados

1. O roadmap dizia "duas frentes", mas a secao ativa listava cinco. Isso foi
   corrigido para "frentes priorizadas".
2. `competition_match`, `answer` e `expected_answer` foram reafirmados como
   campos de auditoria apenas. Eles nao podem entrar no ranking label-free de
   candidates.
3. O gate V532 nao e promotavel: os candidate pools cobrem `155/155` weak
   equation rows, mas o seletor label-free caiu de `55/155` para `29/155`, com
   `2` ganhos e `28` perdas. Portanto, esses datasets viram fonte de
   canonicalization/hard negatives, nao patch direto nem treino direto.
4. O pacote Huikang V533 e util para bit como fonte de trace CHO/MAJ, mas nao
   como treino bruto. O ZIP local contem `2000` traces sinteticas, enquanto a
   metadata Kaggle cita `10000`; o conteudo local e a fonte da verdade.
5. `bit_manipulation_3input_traces.jsonl` cobre `10` weak bit rows, incluindo
   `8` misses atuais. Isso e diagnostico de cobertura e nao pode ser copiado
   para treino promocional.
6. `corpus.jsonl` do Huikang tem overlap massivo com rows oficiais e deve ser
   P1 filtrado. Nao e fonte limpa de treino imediato.
7. V-CARS/Tatoeba nao ataca as duas familias alvo. Deve ficar fora do plano
   ativo de treino.
8. Yoiko e Huikang adapters externos sao candidatos P1 de avaliacao
   adapter-only, nao fonte de treino. Precisam de gate estatico e weak eval
   curto antes de qualquer conclusao de ganho.
9. O baseline precisa ser tratado com duas leituras: `192/315` e o baseline
   historico/forense packageable; a recomputacao strict label-free recente
   mede `191/315`. Promocao continua exigindo superar o patamar historico:
   `>=193/315`, `bit>=136`, `trunc=0`.

## Itens Retirados Do Plano Ativo

- Broad SFT sem novo sinal CPU.
- Treino direto a partir de candidate pools V532.
- Uso de V-CARS/Tatoeba para as familias alvo.
- Copia de weak/full rows Huikang para treino promocional.
- GPU V523/V530 como caminho ativo sem novo pack source-only e gates CPU.

## Plano Ativo Corrigido

1. Construir V534 CPU source-only para bit:
   - fontes: Konbu success high-confidence e Huikang CHO/MAJ;
   - excluir qualquer overlap weak/full por `id`, `prompt_sha256` e
     `prompt+answer_sha256`;
   - gerar traces curtas e verificaveis;
   - passar tokenization/offset-mask/trace gates antes de GPU.
2. Construir V535 CPU para equation:
   - usar V532 como feature/hard-negative/canonicalization reference;
   - nao usar candidate selector direto ate haver `gains>0` e `losses=0`;
   - foco em regras exportaveis para treino, nao em oracle weak.
3. Avaliar adapters externos apenas se barato:
   - Yoiko/Huikang v26 com header/config gate;
   - weak eval curto;
   - FinOps cancel se `bit<136`, `trunc>0` ou `total<=192`.
4. GPU so volta se CPU provar novo sinal aprendivel e sem contaminacao.

## Criterio De Falha Fechado

Qualquer proxima execucao deve parar se:

- `total < 193/315`;
- `equation_transform < 57/155`;
- `bit_manipulation < 136/160`;
- `truncated != 0`;
- protected row `8740ed31 != 01101000`;
- protected row `59bee375 != 10010101`;
- protected row `518deb39 != $`;
- houver qualquer label leakage em ranking, dataset ou selecao;
- houver mismatch oficial em fonte usada como positivo.

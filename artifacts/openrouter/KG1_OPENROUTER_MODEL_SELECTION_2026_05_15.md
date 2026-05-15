# KG1 OpenRouter Model Selection - 2026-05-15

Fonte local arquivada: `artifacts/openrouter/openrouter_modelos_completos_2026_05_15.md`.

O catalogo completo contem `437` modelos do OpenRouter Chat, consultados em `2026-05-15`. Este arquivo nao substitui o catalogo; ele registra quais modelos usar para o prompt de descoberta das familias `equation_transform` e `bit_manipulation`.

## Objetivo

Usar modelos externos somente para gerar hipoteses novas, executaveis e label-free. Nenhuma resposta de IA vira ganho ate passar por CPU gate local:

- `total > 192/315`
- `equation_transform > 56/155`
- `bit_manipulation >= 136/160`
- `truncated = 0`
- sem `answer`, `id`, oracle, postprocessor ou cherry-pick.

## Ordem Recomendada

| Ordem | Modelo | Slug OpenRouter | Uso no prompt |
|---:|---|---|---|
| 1 | DeepSeek V4 Pro | `deepseek/deepseek-v4-pro` | Principal para raciocinio algoritmico, program synthesis, Python/SMT/CEGIS e propostas executaveis. |
| 2 | Claude Opus 4.7 | `anthropic/claude-opus-4.7` | Auditoria logica, deteccao de gaps, classificacao submit-safe vs teacher-only e revisao de hipoteses. |
| 3 | Qwen3 Max Thinking | `qwen/qwen3-max-thinking` | Segunda opiniao forte em matematica, logica simbolica e busca de regras. |
| 4 | Qwen3 Coder Plus | `qwen/qwen3-coder-plus` | Converter ideias em pseudocodigo/Python e sugerir CPU gates concretos. |
| 5 | Gemini 3.1 Pro Preview | `google/gemini-3.1-pro-preview` | Leitura de contexto longo, consolidacao e deteccao de inconsistencias entre fontes. |
| 6 | Kimi K2 Thinking | `moonshotai/kimi-k2-thinking` | Alternativa para raciocinio longo/codigo quando DeepSeek/Qwen discordarem. |
| 7 | NVIDIA Nemotron 3 Super | `nvidia/nemotron-3-super-120b-a12b` | Revisao contextual por modelo da familia Nemotron; util como perspectiva adicional, nao como decisor. |

## Modelos Baratos/Gratuitos Para Triagem

| Modelo | Slug OpenRouter | Quando usar |
|---|---|---|
| DeepSeek V4 Flash | `deepseek/deepseek-v4-flash` | Primeira triagem barata de ideias; pedir respostas curtas e codigo. |
| NVIDIA Nemotron 3 Super free | `nvidia/nemotron-3-super-120b-a12b` | Checagem barata de hipoteses de raciocinio. |
| NVIDIA Nemotron 3 Nano Omni free | `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` | Triagem gratuita, especialmente para entender estilo Nemotron. |
| Qwen3 Coder 480B A35B free | `qwen/qwen3-coder` | Ideias de codigo/gates, se disponivel gratuitamente no momento. |
| Qwen3 Next 80B A3B Instruct free | `qwen/qwen3-next-80b-a3b-instruct` | Triagem barata de regras e prompts. |

## Modelos A Evitar Para Este Prompt

| Tipo | Motivo |
|---|---|
| Imagem/video/audio/TTS | Nao ajudam nas familias textuais do desafio. |
| Embeddings | Uteis para busca, mas nao para criar regras label-free. |
| Modelos pequenos sem reasoning | Podem servir para classificacao barata, mas nao para descoberta de regras. |
| Modelos caros em modo fast premium | Usar apenas se houver uma hipotese concreta para auditar; nao para busca aberta. |

## Protocolo de Uso

1. Rodar o prompt primeiro em `deepseek/deepseek-v4-pro`.
2. Rodar o mesmo prompt em `anthropic/claude-opus-4.7`, pedindo auditoria e rejeicao de ideias nao submit-safe.
3. Rodar em `qwen/qwen3-max-thinking` ou `qwen/qwen3-coder-plus`, pedindo pseudocodigo executavel.
4. Se houver divergencia, usar `google/gemini-3.1-pro-preview` para consolidar.
5. Implementar localmente somente ideias com algoritmo claro, abstention e criterio de falsificacao.
6. Nao abrir GPU antes de CPU gate com ganho real.

## Observacao

O catalogo OpenRouter e temporalmente instavel. Antes de uso pago relevante, verificar se o slug ainda existe e se o preco/contexto nao mudou.

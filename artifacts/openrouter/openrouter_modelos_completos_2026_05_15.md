# Lista Completa de Modelos Disponíveis no OpenRouter Chat

> **Data da consulta:** 15 de maio de 2026
> **Total de modelos listados:** 437
> **Fonte:** API do OpenRouter (endpoint frontend/models/find)

Este documento contém a lista completa de todos os modelos disponíveis no popup **"Add Model"** do [OpenRouter Chat](https://openrouter.ai/chat). Para cada modelo, são apresentados o nome, o identificador (slug), a categoria, o tamanho do contexto, o preço e uma descrição detalhada de sua funcionalidade.

## Resumo por Categoria

| Categoria | Quantidade |
|-----------|-----------|
| Texto (LLM) | 120 |
| Multimodal (Visão + Texto) | 111 |
| Código / Programação | 99 |
| Multimodal (Vídeo) | 57 |
| Geração de Imagem | 21 |
| Multimodal (Arquivo + Texto) | 11 |
| Multimodal (Áudio + Texto) | 7 |
| Multimodal (Texto + Imagem) | 6 |
| Multimodal (Texto + Áudio) | 5 |
| **Total** | **437** |

## Resumo por Provedor

| Provedor | Quantidade de Modelos |
|----------|----------------------|
| OpenAI | 77 |
| Qwen | 49 |
| Google | 36 |
| Mistral AI | 27 |
| Anthropic | 16 |
| Meta Llama | 14 |
| DeepSeek | 13 |
| Z.ai | 13 |
| xAI | 12 |
| Recraft | 11 |
| Nvidia | 10 |
| MiniMax | 9 |
| arcee-ai | 8 |
| baidu | 7 |
| Cohere | 7 |
| Perplexity | 7 |
| moonshotai | 6 |
| Nous Research | 6 |
| Xiaomi | 5 |
| bytedance-seed | 5 |
| sourceful | 5 |
| Amazon | 5 |
| sentence-transformers | 5 |
| Sao10K | 5 |
| bytedance | 4 |
| Aion Labs | 4 |
| black-forest-labs | 4 |
| Drummer | 4 |
| inclusionai | 3 |
| kwaivgi | 3 |
| alibaba | 3 |
| Liquid | 3 |
| intfloat | 3 |
| baai | 3 |
| Microsoft | 3 |
| ibm-granite | 2 |
| poolside | 2 |
| zyphra | 2 |
| tencent | 2 |
| rekaai | 2 |
| relace | 2 |
| thenlper | 2 |
| morph | 2 |
| Inflection | 2 |
| perceptron | 1 |
| OpenRouter | 1 |
| sesame | 1 |
| canopylabs | 1 |
| hexgrad | 1 |
| kwaipilot | 1 |
| inception | 1 |
| stepfun | 1 |
| upstage | 1 |
| Writer | 1 |
| nex-agi | 1 |
| essentialai | 1 |
| prime-intellect | 1 |
| Ai2 | 1 |
| deepcogito | 1 |
| AI21 | 1 |
| switchpoint | 1 |
| venice | 1 |
| alfredpros | 1 |
| deepseek-ai | 1 |
| anthracite-org | 1 |
| Mancer | 1 |
| Undi | 1 |
| Gryphe | 1 |

---

## Lista Completa de Modelos

### 1. xAI: Grok Voice TTS 1.0

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `x-ai/grok-voice-tts-1.0` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 15,000 tokens |
| **Entrada** | text |
| **Saída** | speech |
| **Preço Input** | $0.000015 per 1M characters |
| **Preço Output** | Incluído |

**Descrição:** Grok Voice TTS 1.0 is a text-to-speech model from xAI. It converts text into spoken audio across 20+ languages with automatic language detection, and offers five built-in voices (Eve, Ara, Rex, Sal, Leo) covering a range of tones. Inline speech tags allow control over pauses, emphasis, pitch, speed, and vocal style. Output is available in MP3, WAV, PCM, μ-law, and A-law formats at sample rates from 8 kHz to 48 kHz, with up to 15,000 characters per request.

### 2. Recraft: Recraft V4.1 Pro Vector

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `recraft/recraft-v4.1-pro-vector` |
| **Categoria** | Geração de Imagem |
| **Contexto** | 65,536 tokens |
| **Entrada** | text, image |
| **Saída** | image |
| **Preço Input** | $0.3 per image |
| **Preço Output** | Incluído |

**Descrição:** Recraft V4.1 Pro Vector is the vector (SVG) variant of Recraft V4.1 Pro, tuned for high aesthetics. It supports text and image inputs and produces higher-resolution SVG image output across multiple aspect ratios, with typical generation around 20 seconds. Output scales cleanly, making it suitable for icons, logos, and other graphics. V4.1 brings more personality to text and illustrations, smoother gradients, and stronger short-prompt adherence compared to V4 Pro.  Suited for higher-resolution...

### 3. Recraft: Recraft V4.1 Vector

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `recraft/recraft-v4.1-vector` |
| **Categoria** | Geração de Imagem |
| **Contexto** | 65,536 tokens |
| **Entrada** | text, image |
| **Saída** | image |
| **Preço Input** | $0.08 per image |
| **Preço Output** | Incluído |

**Descrição:** Recraft V4.1 Vector is the vector (SVG) variant of Recraft V4.1, tuned for high aesthetics. It supports text and image inputs and produces SVG image output across multiple aspect ratios, with typical generation around 13 seconds. Output scales cleanly, making it suitable for icons, logos, and other graphics. V4.1 brings more personality to text and illustrations, smoother gradients, and stronger short-prompt adherence compared to V4.  Suited for everyday illustration work where output should ...

### 4. Recraft: Recraft V4.1 Utility Pro

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `recraft/recraft-v4.1-utility-pro` |
| **Categoria** | Geração de Imagem |
| **Contexto** | 65,536 tokens |
| **Entrada** | text, image |
| **Saída** | image |
| **Preço Input** | $0.25 per image |
| **Preço Output** | Incluído |

**Descrição:** Recraft V4.1 Utility Pro is a general-purpose image generation model from Recraft. It supports text and image inputs with image output at ~2K resolution across multiple aspect ratios — double the resolution of V4.1 Utility - with typical generation around 20 seconds. Like V4.1 Utility, it is designed for restraint as the aesthetic choice - flat lighting, front-facing composition, and simple, controlled scenes - at higher fidelity for production use cases.  V4.1 improvements over V4 Pro includ...

### 5. Recraft: Recraft V4.1 Utility

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `recraft/recraft-v4.1-utility` |
| **Categoria** | Geração de Imagem |
| **Contexto** | 65,536 tokens |
| **Entrada** | text, image |
| **Saída** | image |
| **Preço Input** | $0.04 per image |
| **Preço Output** | Incluído |

**Descrição:** Recraft V4.1 Utility is a general-purpose image generation model from Recraft. It supports text and image inputs with image output at ~1K resolution across multiple aspect ratios, with typical generation around 10 seconds. The Utility line is designed for restraint as the aesthetic choice - flat lighting, front-facing composition, and simple, controlled scenes - making it a practical fit for product imagery, mockups, and structured visuals where the high-aesthetic V4.1 line would be too expre...

### 6. Recraft: Recraft V4.1 Pro

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `recraft/recraft-v4.1-pro` |
| **Categoria** | Geração de Imagem |
| **Contexto** | 65,536 tokens |
| **Entrada** | text, image |
| **Saída** | image |
| **Preço Input** | $0.25 per image |
| **Preço Output** | Incluído |

**Descrição:** Recraft V4.1 Pro is an image generation model from Recraft tuned for high aesthetics. It supports text and image inputs with image output at ~2K resolution across multiple aspect ratios - double the resolution of V4.1 - with typical generation around 30 seconds. It shares the V4.1 visual sensibility at higher fidelity and detail density, with more natural photorealism, smoother 3D rendering and gradients, and stronger short-prompt adherence than V4 Pro.  Suited for production work where image...

### 7. Recraft: Recraft V4.1

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `recraft/recraft-v4.1` |
| **Categoria** | Geração de Imagem |
| **Contexto** | 65,536 tokens |
| **Entrada** | text, image |
| **Saída** | image |
| **Preço Input** | $0.04 per image |
| **Preço Output** | Incluído |

**Descrição:** Recraft V4.1 is an image generation model from Recraft tuned for high aesthetics. It supports text and image inputs with image output at ~1K resolution across multiple aspect ratios, with typical generation around 10 seconds. Compared to V4, photorealism feels more natural with quieter backgrounds and more purposeful lighting, 3D rendering and soft gradients are smoother, and the model follows shorter prompts more reliably.  Suited for exploration, concept work, and everyday creative work whe...

### 8. Recraft: Recraft V4 Pro Vector

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `recraft/recraft-v4-pro-vector` |
| **Categoria** | Geração de Imagem |
| **Contexto** | 65,536 tokens |
| **Entrada** | text, image |
| **Saída** | image |
| **Preço Input** | $0.3 per image |
| **Preço Output** | Incluído |

**Descrição:** Recraft V4 Pro Vector is the vector (SVG) variant of Recraft V4 Pro. It supports text and image inputs and produces vector image output across multiple aspect ratios at the higher fidelity Pro tier. Output is delivered as SVG, suitable for icons, logos, and other graphics that need to scale cleanly. V4 Pro offers higher fidelity and detail density than V4, with stronger compositional judgment, color coherence, and legible embedded text compared to V3.  Supports the following `image_config` pa...

### 9. Recraft: Recraft V4 Vector

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `recraft/recraft-v4-vector` |
| **Categoria** | Geração de Imagem |
| **Contexto** | 65,536 tokens |
| **Entrada** | text, image |
| **Saída** | image |
| **Preço Input** | $0.08 per image |
| **Preço Output** | Incluído |

**Descrição:** Recraft V4 Vector is the vector (SVG) variant of Recraft V4. It supports text and image inputs and produces vector image output across multiple aspect ratios. Compared to the raster V4, output is delivered as SVG, suitable for icons, logos, and other graphics that need to scale cleanly. V4 delivers stronger compositional judgment, color coherence, and legible embedded text compared to V3.  Supports the following `image_config` parameters: `strength` (controls how much the output deviates from...

### 10. Anthropic: Claude Opus 4.7 (Fast)

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `anthropic/claude-opus-4.7-fast` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 1,000,000 tokens |
| **Entrada** | text, image, file |
| **Saída** | text |
| **Preço Input** | $30e-6 /M tokens |
| **Preço Output** | $150e-6 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Fast-mode variant of [Opus 4.7](/anthropic/claude-opus-4.7) - identical capabilities with higher output speed at premium 6x pricing.  Learn more in Anthropic's docs: https://platform.claude.com/docs/en/build-with-claude/fast-mode

### 11. Perceptron: Perceptron Mk1

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `perceptron/perceptron-mk1` |
| **Categoria** | Multimodal (Vídeo) |
| **Contexto** | 32,768 tokens |
| **Entrada** | text, image, video |
| **Saída** | text |
| **Preço Input** | $0.00000015 /M tokens |
| **Preço Output** | $0.0000015 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Perceptron Mk1 (Mark One) is Perceptron's highest-quality vision-language model for video and embodied reasoning.** It accepts image and video inputs paired with natural language queries, and produces detailed visual understanding responses, either structured or natural language. It excels at video understanding tasks like video QA, summarization, and event detection. On image inputs, it advances point-by-example grounding from multimodal prompts, OCR and document parsing on messy real-world ...

### 12. inclusionAI: Ring-2.6-1T (free)

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `inclusionai/ring-2.6-1t` |
| **Categoria** | Código / Programação |
| **Contexto** | 262,144 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0 /M tokens |
| **Preço Output** | $0 /M tokens |
| **Gratuito** | Sim |
| **Raciocínio** | Suportado |

**Descrição:** Ring-2.6-1T is a 1T-parameter-scale thinking model with 63B active parameters, built for real-world agent workflows that require both strong capability and operational efficiency. It is optimized for coding agents, tool use, and long-horizon task execution, delivering leading results on benchmarks including PinchBench, ClawEval, TAU2-Bench, and GAIA2-search.  With adaptive reasoning effort across high and xhigh modes, Ring-2.6-1T dynamically allocates reasoning budget based on task complexity...

### 13. Recraft: Recraft V4 Pro

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `recraft/recraft-v4-pro` |
| **Categoria** | Geração de Imagem |
| **Contexto** | 65,536 tokens |
| **Entrada** | text, image |
| **Saída** | image |
| **Preço Input** | $0.25 per image |
| **Preço Output** | Incluído |

**Descrição:** Recraft V4 Pro is an image generation model from Recraft. It supports text and image inputs with image output at ~2K resolution across multiple aspect ratios, double the resolution of V4. It offers higher fidelity and detail density than V4, suited for production use cases where image quality is a priority.  Supports the following `image_config` parameters: `strength` (controls how much the output deviates from the source image), `rgb_colors` (sets a color palette), and `background_rgb_color`...

### 14. Recraft: Recraft V4

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `recraft/recraft-v4` |
| **Categoria** | Geração de Imagem |
| **Contexto** | 65,536 tokens |
| **Entrada** | text, image |
| **Saída** | image |
| **Preço Input** | $0.04 per image |
| **Preço Output** | Incluído |

**Descrição:** Recraft V4 is an image generation model from Recraft. It supports text and image inputs with image output at ~1K resolution across multiple aspect ratios. It delivers stronger compositional judgment, color coherence, and legible embedded text compared to V3, making it suited for infographics, signage, and packaging.  Supports the following `image_config` parameters: `strength` (controls how much the output deviates from the source image), `rgb_colors` (sets a color palette), and `background_r...

### 15. Recraft: Recraft V3

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `recraft/recraft-v3` |
| **Categoria** | Geração de Imagem |
| **Contexto** | 65,536 tokens |
| **Entrada** | text, image |
| **Saída** | image |
| **Preço Input** | $0.04 per image |
| **Preço Output** | Incluído |

**Descrição:** Recraft V3 is an image generation model from Recraft. It supports text and image inputs with image output at ~1K resolution across multiple aspect ratios.  Supports the following `image_config` parameters: `strength` (controls how much the output deviates from the source image), `style` (applies an artistic style), `text_layout` (places text at specific positions), `rgb_colors` (sets a color palette), and `background_rgb_color` (sets the background color). See the image generation docs for de...

### 16. Google: Gemini 3.1 Flash Lite

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `google/gemini-3.1-flash-lite` |
| **Categoria** | Multimodal (Vídeo) |
| **Contexto** | 1,048,576 tokens |
| **Entrada** | text, image, video, file, audio |
| **Saída** | text |
| **Preço Input** | $0.25e-6 /M tokens |
| **Preço Output** | $1.5e-6 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Gemini 3.1 Flash Lite is Google’s GA high-efficiency multimodal model optimized for low-latency, high-volume workloads. It supports text, image, video, audio, and PDF inputs, and is designed for lightweight agentic workflows, simple data extraction, and applications where responsiveness and API cost are the primary constraints.  Supports full thinking levels (minimal, low, medium, high) for fine-grained cost/performance trade-offs. Priced at half the cost of Gemini 3 Flash.

### 17. Baidu Qianfan: CoBuddy (free)

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `baidu/cobuddy` |
| **Categoria** | Código / Programação |
| **Contexto** | 131,072 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0 /M tokens |
| **Preço Output** | $0 /M tokens |
| **Gratuito** | Sim |
| **Raciocínio** | Suportado |

**Descrição:** CoBuddy is a code generation model from Baidu, optimized for coding tasks and AI Agent workflows. It features high inference throughput and low end-to-end latency, with native support for tool calling and reasoning. The model runs on fp8 quantization with a 131K token context window and up to 65K output tokens.

### 18. OpenAI: GPT Chat Latest

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/gpt-chat-latest` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 400,000 tokens |
| **Entrada** | text, image, file |
| **Saída** | text |
| **Preço Input** | $5e-6 /M tokens |
| **Preço Output** | $30e-6 /M tokens |

**Descrição:** GPT Chat Latest points to OpenAI's stable API alias `chat-latest` that always resolves to the latest Instant chat model used in ChatGPT. As OpenAI rolls out new Instant model updates in the future, they are routed behind this slug automatically.  For more info, see: https://developers.openai.com/api/docs/models/chat-latest

### 19. Google: Chirp 3

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `google/chirp-3` |
| **Categoria** | Multimodal (Áudio + Texto) |
| **Contexto** | 0 tokens |
| **Entrada** | audio |
| **Saída** | transcription |
| **Preço Input** | $0.016 per minute |
| **Preço Output** | Incluído |

**Descrição:** Chirp 3 is Google's latest multilingual speech-to-text model. It offers enhanced transcription accuracy across 24 GA languages and 77+ preview languages, with support for automatic language detection, automatic punctuation, and a built-in denoiser for cleaner audio processing.

### 20. OpenAI: GPT-4o Mini Transcribe

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/gpt-4o-mini-transcribe` |
| **Categoria** | Multimodal (Áudio + Texto) |
| **Contexto** | 128,000 tokens |
| **Entrada** | audio |
| **Saída** | transcription |
| **Preço Input** | $0.00000125 /M tokens |
| **Preço Output** | $0.000005 /M tokens |

**Descrição:** GPT-4o Mini Transcribe is OpenAI's smaller, cost-efficient speech-to-text model built on GPT-4o Mini audio capabilities. It's priced per token (input and output), making it suitable for high-volume transcription workflows that benefit from token-level billing transparency at a lower cost point.

### 21. OpenAI: Whisper Large V3 Turbo

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/whisper-large-v3-turbo` |
| **Categoria** | Multimodal (Áudio + Texto) |
| **Contexto** | 0 tokens |
| **Entrada** | audio |
| **Saída** | transcription |
| **Preço Input** | $0.04 per hour |
| **Preço Output** | Incluído |

**Descrição:** Whisper Large V3 Turbo is an optimized version of OpenAI's Whisper Large V3 speech recognition model, designed for speed and cost efficiency. It supports transcription across 99+ languages with a 12% word error rate, and accepts common audio formats including mp3, mp4, wav, webm, flac, and ogg. Achieves real-time speed factors up to 216x, making it well-suited for latency-sensitive and high-throughput transcription workloads.

### 22. OpenAI: Whisper Large V3

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/whisper-large-v3` |
| **Categoria** | Multimodal (Áudio + Texto) |
| **Contexto** | 0 tokens |
| **Entrada** | audio |
| **Saída** | transcription |
| **Preço Input** | $0.111 per hour |
| **Preço Output** | Incluído |

**Descrição:** Whisper Large V3 is OpenAI's open-source automatic speech recognition model offering both audio transcription and translation. It supports 99+ languages and accepts common audio formats including mp3, mp4, wav, webm, flac, and ogg. With 1,550M parameters, it achieves a 10.3% word error rate and is well-suited for noise-robust, multilingual transcription in demanding conditions. Supports timestamp granularities at word and segment levels.

### 23. xAI: Grok 4.3

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `x-ai/grok-4.3` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 1,000,000 tokens |
| **Entrada** | text, image |
| **Saída** | text |
| **Preço Input** | $1.25e-6 /M tokens |
| **Preço Output** | $2.50e-6 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Grok 4.3 is a reasoning model from xAI. It accepts text and image inputs with text output, and is suited for agentic workflows, instruction-following tasks, and applications requiring high factual accuracy. Reasoning can be configured between none/low/medium/high (default low) effort levels.  It supports a 1 million token context window with no output token limit, making it well-suited for long-document analysis, deep research, and multi-step agentic tasks. Pricing is tiered: requests exceedi...

### 24. IBM: Granite 4.1 8B

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `ibm-granite/granite-4.1-8b` |
| **Categoria** | Código / Programação |
| **Contexto** | 131,072 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000005 /M tokens |
| **Preço Output** | $0.0000001 /M tokens |

**Descrição:** Granite 4.1 8B is a dense, decoder-only 8-billion-parameter language model from IBM, part of the Granite 4.1 family. It supports a 131K-token context window and is designed for enterprise tasks including tool calling, retrieval-augmented generation (RAG), code generation with fill-in-the-middle support, text summarization, classification, and extraction.  The model handles 12 languages (English, German, Spanish, French, Japanese, Portuguese, Arabic, Czech, Italian, Korean, Dutch, and Chinese)...

### 25. Mistral: Mistral Medium 3.5

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `mistralai/mistral-medium-3-5` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 262,144 tokens |
| **Entrada** | text, image, file |
| **Saída** | text |
| **Preço Input** | $0.0000015 /M tokens |
| **Preço Output** | $0.0000075 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Mistral Medium 3.5 is a dense 128B instruction-following model from Mistral AI. It supports text and image inputs with text output, and is designed for agentic workflows, coding, and complex multi-step reasoning. It is particularly strong at reliable multi-tool calling and long-horizon tasks, with a 256K context window, configurable reasoning effort per request, and a custom vision encoder that handles variable image sizes and aspect ratios. Self-hostable on as few as four GPUs and available ...

### 26. Kling: Video v3.0 Pro

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `kwaivgi/kling-v3.0-pro` |
| **Categoria** | Multimodal (Vídeo) |
| **Contexto** | 0 tokens |
| **Entrada** | text, image |
| **Saída** | video |
| **Preço Input** | $0.168 per second |
| **Preço Output** | $0.112 per second |

**Descrição:** Kling v3.0 Pro is Kuaishou's premium video generation model, offering higher visual quality than the Standard tier. It supports text-to-video and image-to-video workflows, with first-frame and last-frame control for precise scene composition. Clips range from 3 to 15 seconds in 16:9, 9:16, or 1:1 aspect ratios. Native audio generation is available as an option.

### 27. Kling: Video v3.0 Standard

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `kwaivgi/kling-v3.0-std` |
| **Categoria** | Multimodal (Vídeo) |
| **Contexto** | 0 tokens |
| **Entrada** | text, image |
| **Saída** | video |
| **Preço Input** | $0.126 per second |
| **Preço Output** | $0.084 per second |

**Descrição:** Kling v3.0 Standard is a video generation model from Kuaishou. It supports text-to-video and image-to-video workflows, with first-frame and last-frame control for guided scene composition. Clips range from 3 to 15 seconds in 16:9, 9:16, or 1:1 aspect ratios. Native audio generation is available as an option.

### 28. Owl Alpha

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openrouter/owl-alpha` |
| **Categoria** | Código / Programação |
| **Contexto** | 1,048,756 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0 /M tokens |
| **Preço Output** | $0 /M tokens |
| **Gratuito** | Sim |

**Descrição:** Owl Alpha is a high-performance foundation model designed for agentic workloads. Natively supports tool use, and long-context tasks, with strong performance in code generation, automated workflows, and complex instruction execution. Compatible with Claude Code, OpenClaw, and other mainstream productivity tools.  Note: Prompts and completions may be logged by the provider and used to improve the model.

### 29. NVIDIA: Nemotron 3 Nano Omni (free)

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` |
| **Categoria** | Multimodal (Vídeo) |
| **Contexto** | 256,000 tokens |
| **Entrada** | text, audio, image, video |
| **Saída** | text |
| **Preço Input** | $0 /M tokens |
| **Preço Output** | $0 /M tokens |
| **Gratuito** | Sim |
| **Raciocínio** | Suportado |

**Descrição:** NVIDIA Nemotron™ 3 Nano Omni is a 30B-A3B open multimodal model designed to function as a perception and context sub-agent in enterprise agent systems. It accepts text, image, video, and audio inputs and produces text output, enabling agents to perceive and reason across modalities in a single inference loop.  Built on a hybrid MoE Transformer-Mamba architecture with Conv3D video layers and Efficient Video Sampling (EVS), it delivers approximately 2× higher throughput and 2.5× lower compute f...

### 30. Poolside: Laguna XS.2 (free)

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `poolside/laguna-xs.2` |
| **Categoria** | Código / Programação |
| **Contexto** | 131,072 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0 /M tokens |
| **Preço Output** | $0 /M tokens |
| **Gratuito** | Sim |
| **Raciocínio** | Suportado |

**Descrição:** Laguna XS.2 is the second-generation model in the XS size class from [Poolside](https://poolside.ai), their efficient coding agent series. It combines tool calling and reasoning capabilities with a compact footprint, offering a 128K context window and up to 8K output tokens. Quantized to fp8 for fast, cost-efficient agentic coding workflows.   Laguna XS.2 is designed for software engineering and agentic coding use cases, and you are responsible for confirming that it is appropriate for your i...

### 31. Poolside: Laguna M.1 (free)

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `poolside/laguna-m.1` |
| **Categoria** | Código / Programação |
| **Contexto** | 131,072 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0 /M tokens |
| **Preço Output** | $0 /M tokens |
| **Gratuito** | Sim |
| **Raciocínio** | Suportado |

**Descrição:** Laguna M.1 is the flagship coding agent model from [Poolside](https://poolside.ai), optimized for complex software engineering tasks. Designed for agentic coding workflows, it supports tool calling and reasoning, with a 128K context window and up to 8K output tokens. Quantized to fp8 for efficient inference. By using this model, you agree to Poolside’s [End User License Agreement](https://poolside.ai/legal/eula)

### 32. OpenAI: Whisper 1

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/whisper-1` |
| **Categoria** | Multimodal (Áudio + Texto) |
| **Contexto** | 0 tokens |
| **Entrada** | audio |
| **Saída** | transcription |
| **Preço Input** | $0.006 per minute |
| **Preço Output** | Incluído |

**Descrição:** Whisper is OpenAI's open-source automatic speech recognition model, available via API as `whisper-1`. It supports transcription and translation across 50+ languages from audio files up to 25 MB. Accepts formats including mp3, mp4, wav, and webm. Priced per minute of audio duration, billed to the nearest second.

### 33. OpenAI: GPT-4o Transcribe

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/gpt-4o-transcribe` |
| **Categoria** | Multimodal (Áudio + Texto) |
| **Contexto** | 128,000 tokens |
| **Entrada** | audio |
| **Saída** | transcription |
| **Preço Input** | $0.0000025 /M tokens |
| **Preço Output** | $0.00001 /M tokens |

**Descrição:** GPT-4o Transcribe is OpenAI's high-quality speech-to-text model built on GPT-4o audio capabilities. It's priced per token (input and output), making it suitable for workflows that benefit from token-level billing transparency.

### 34. Anthropic Claude Haiku Latest

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `~anthropic/claude-haiku-latest` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 200,000 tokens |
| **Entrada** | text, image, file |
| **Saída** | text |
| **Preço Input** | $0.000001 /M tokens |
| **Preço Output** | $0.000005 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** This model always redirects to the latest model in the Anthropic Claude Haiku family.

### 35. OpenAI GPT Mini Latest

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `~openai/gpt-mini-latest` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 400,000 tokens |
| **Entrada** | file, image, text |
| **Saída** | text |
| **Preço Input** | $0.75e-6 /M tokens |
| **Preço Output** | $4.5e-6 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** This model always redirects to the latest model in the OpenAI GPT Mini family.

### 36. Google Gemini Pro Latest

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `~google/gemini-pro-latest` |
| **Categoria** | Multimodal (Vídeo) |
| **Contexto** | 1,048,576 tokens |
| **Entrada** | audio, file, image, text, video |
| **Saída** | text |
| **Preço Input** | $0.000002 /M tokens |
| **Preço Output** | $0.000012 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** This model always redirects to the latest model in the Google Gemini Pro family.

### 37. MoonshotAI Kimi Latest

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `~moonshotai/kimi-latest` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 262,142 tokens |
| **Entrada** | text, image |
| **Saída** | text |
| **Preço Input** | $0.00000073 /M tokens |
| **Preço Output** | $0.00000349 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** This model always redirects to the latest model in the MoonshotAI Kimi family.

### 38. Google Gemini Flash Latest

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `~google/gemini-flash-latest` |
| **Categoria** | Multimodal (Vídeo) |
| **Contexto** | 1,048,576 tokens |
| **Entrada** | text, image, file, audio, video |
| **Saída** | text |
| **Preço Input** | $5e-7 /M tokens |
| **Preço Output** | $0.000003 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** This model always redirects to the latest model in the Google Gemini Flash family.

### 39. Anthropic Claude Sonnet Latest

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `~anthropic/claude-sonnet-latest` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 1,000,000 tokens |
| **Entrada** | text, image, file |
| **Saída** | text |
| **Preço Input** | $0.000003 /M tokens |
| **Preço Output** | $0.000015 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** This model always redirects to the latest model in the Anthropic Claude Sonnet family.

### 40. OpenAI GPT Latest

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `~openai/gpt-latest` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 1,050,000 tokens |
| **Entrada** | file, image, text |
| **Saída** | text |
| **Preço Input** | $5e-6 /M tokens |
| **Preço Output** | $30e-6 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** This model always redirects to the latest model in the OpenAI GPT family.

### 41. Qwen: Qwen3.5 Plus 2026-04-20

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `qwen/qwen3.5-plus-20260420` |
| **Categoria** | Multimodal (Vídeo) |
| **Contexto** | 1,000,000 tokens |
| **Entrada** | text, image, video |
| **Saída** | text |
| **Preço Input** | $0.0000003 /M tokens |
| **Preço Output** | $0.0000018 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Qwen3.5 Plus (April 2026) is a large-scale multimodal language model from Alibaba. It accepts text, image, and video input and produces text output, with a 1M token context window. This is an updated version of Qwen3.5 Plus with tiered pricing above 256K tokens.

### 42. Qwen: Qwen3.6 Flash

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `qwen/qwen3.6-flash` |
| **Categoria** | Multimodal (Vídeo) |
| **Contexto** | 1,000,000 tokens |
| **Entrada** | text, image, video |
| **Saída** | text |
| **Preço Input** | $0.0000001875 /M tokens |
| **Preço Output** | $0.000001125 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Qwen3.6 Flash is a fast, efficient language model from Alibaba's Qwen 3.6 series. It supports text, image, and video input with a 1M token context window. Tiered pricing kicks in above 256K tokens. Prompt caching is supported, with both explicit cache read and cache creation pricing.

### 43. Qwen: Qwen3.6 35B A3B

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `qwen/qwen3.6-35b-a3b` |
| **Categoria** | Multimodal (Vídeo) |
| **Contexto** | 262,144 tokens |
| **Entrada** | text, image, video |
| **Saída** | text |
| **Preço Input** | $0.00000015 /M tokens |
| **Preço Output** | $0.000001 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Qwen3.6-35B-A3B is an open-weight multimodal model from Alibaba Cloud with 35 billion total parameters and 3 billion active parameters per token. It uses a hybrid sparse mixture-of-experts architecture combining Gated DeltaNet linear attention with standard gated attention layers, enabling efficient inference at a fraction of the compute cost. The model supports a 262K token native context window (extensible to 1M via YaRN) and accepts text, image, and video inputs. It includes integrated thi...

### 44. Qwen: Qwen3.6 Max Preview

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `qwen/qwen3.6-max-preview` |
| **Categoria** | Código / Programação |
| **Contexto** | 262,144 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000104 /M tokens |
| **Preço Output** | $0.00000624 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Qwen3.6-Max-Preview is a proprietary frontier model from Alibaba Cloud built on a sparse mixture-of-experts architecture with approximately 1 trillion total parameters. It is optimized for agentic coding, tool use, and long-context reasoning, supporting a 262K token context window. The model includes an integrated thinking mode that preserves reasoning traces across multi-turn conversations and supports structured output and function calling. Access is available exclusively through the Alibab...

### 45. Qwen: Qwen3.6 27B

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `qwen/qwen3.6-27b` |
| **Categoria** | Multimodal (Vídeo) |
| **Contexto** | 262,144 tokens |
| **Entrada** | text, image, video |
| **Saída** | text |
| **Preço Input** | $0.00000032 /M tokens |
| **Preço Output** | $0.0000032 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Qwen3.6 27B is a dense 27-billion-parameter language model from the Qwen Team at Alibaba, released in April 2026. It features hybrid multimodal capabilities — accepting text, image, and video inputs — and supports a 262,144-token context window.  The model is designed for agentic coding and reasoning tasks, with particular strength in repository-level code comprehension, front-end development workflows, and multi-step problem solving. It includes a built-in thinking mode for extended reasonin...

### 46. OpenAI: GPT-5.5 Pro

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/gpt-5.5-pro` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 1,050,000 tokens |
| **Entrada** | file, image, text |
| **Saída** | text |
| **Preço Input** | $30e-6 /M tokens |
| **Preço Output** | $180e-6 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** GPT-5.5 Pro is OpenAI’s high-capability model optimized for deep reasoning and accuracy on complex, high-stakes workloads. It features a 1M+ token context window (922K input, 128K output) with support for text and image inputs, and is designed for long-horizon problem solving, agentic coding, and precise execution across multi-step workflows.

### 47. OpenAI: GPT-5.5

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/gpt-5.5` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 1,050,000 tokens |
| **Entrada** | file, image, text |
| **Saída** | text |
| **Preço Input** | $5e-6 /M tokens |
| **Preço Output** | $30e-6 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** GPT-5.5 is OpenAI’s frontier model designed for complex professional workloads, building on GPT-5.4 with stronger reasoning, higher reliability, and improved token efficiency on hard tasks. It features a 1M+ token context window (922K input, 128K output) with support for text and image inputs, enabling large-scale reasoning, coding, and multimodal workflows within a single system.

### 48. DeepSeek: DeepSeek V4 Pro

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `deepseek/deepseek-v4-pro` |
| **Categoria** | Código / Programação |
| **Contexto** | 1,048,576 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $.435e-6 /M tokens |
| **Preço Output** | $.87e-6 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** DeepSeek V4 Pro is a large-scale Mixture-of-Experts model from DeepSeek with 1.6T total parameters and 49B activated parameters, supporting a 1M-token context window. It is designed for advanced reasoning, coding, and long-horizon agent workflows, with strong performance across knowledge, math, and software engineering benchmarks.  Built on the same architecture as DeepSeek V4 Flash, it introduces a hybrid attention system for efficient long-context processing. Reasoning efforts `high` and `x...

### 49. DeepSeek: DeepSeek V4 Flash (free)

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `deepseek/deepseek-v4-flash` |
| **Categoria** | Código / Programação |
| **Contexto** | 1,048,576 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0 /M tokens |
| **Preço Output** | $0 /M tokens |
| **Gratuito** | Sim |
| **Raciocínio** | Suportado |

**Descrição:** DeepSeek V4 Flash is an efficiency-optimized Mixture-of-Experts model from DeepSeek with 284B total parameters and 13B activated parameters, supporting a 1M-token context window. It is designed for fast inference and high-throughput workloads, while maintaining strong reasoning and coding performance.  The model includes hybrid attention for efficient long-context processing. Reasoning efforts `high` and `xhigh` are supported; `xhigh` maps to max reasoning. It is well suited for applications ...

### 50. DeepSeek: DeepSeek V4 Flash

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `deepseek/deepseek-v4-flash` |
| **Categoria** | Código / Programação |
| **Contexto** | 1,048,576 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.000000126 /M tokens |
| **Preço Output** | $0.000000252 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** DeepSeek V4 Flash is an efficiency-optimized Mixture-of-Experts model from DeepSeek with 284B total parameters and 13B activated parameters, supporting a 1M-token context window. It is designed for fast inference and high-throughput workloads, while maintaining strong reasoning and coding performance.  The model includes hybrid attention for efficient long-context processing. Reasoning efforts `high` and `xhigh` are supported; `xhigh` maps to max reasoning. It is well suited for applications ...

### 51. Google: Gemini 3.1 Flash TTS Preview

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `google/gemini-3.1-flash-tts-preview` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 8,192 tokens |
| **Entrada** | text |
| **Saída** | speech |
| **Preço Input** | $0.000001 /M text tokens |
| **Preço Output** | $0.00002 /M audio tokens |

**Descrição:** Gemini 3.1 Flash TTS Preview is a text-to-speech model from Google, and a substantial generational step up from Gemini 2.5 Flash TTS. It takes text input and produces audio output across 70+ languages — nearly 3× the language coverage of its predecessor.  The headline addition is a system of 200+ inline audio tags (e.g. `[whispers]`, `[laughs]`, `[excited]`) that let developers steer delivery, emotion, and pacing mid-sentence, alongside a "director's chair" workflow in Google AI Studio for de...

### 52. Google: Veo 3.1 Fast

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `google/veo-3.1-fast` |
| **Categoria** | Multimodal (Vídeo) |
| **Contexto** | 0 tokens |
| **Entrada** | text, image |
| **Saída** | video |
| **Preço Input** | $0.10 per second |
| **Preço Output** | $0.08 per second |

**Descrição:** Google's mid-tier video generation model balancing speed and quality. Veo 3.1 Fast generates high-quality video from text or image prompts with native synchronized audio, offering faster turnaround than Veo 3.1 at lower cost. Supports first-frame and last-frame conditioning, multiple resolutions and aspect ratios, and SynthID watermarking.

### 53. Zyphra: Zonos v0.1 Transformer

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `zyphra/zonos-v0.1-transformer` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 4,096 tokens |
| **Entrada** | text |
| **Saída** | speech |
| **Preço Input** | $0.000007 per 1M characters |
| **Preço Output** | Incluído |

**Descrição:** Zonos v0.1 Transformer is a text-to-speech model from Zyphra built on a pure transformer architecture. It offers the same American and British English voice coverage as the Hybrid variant, and is suited for deployments where a transformer-only inference stack is preferred.

### 54. Zyphra: Zonos v0.1 Hybrid

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `zyphra/zonos-v0.1-hybrid` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 4,096 tokens |
| **Entrada** | text |
| **Saída** | speech |
| **Preço Input** | $0.000007 per 1M characters |
| **Preço Output** | Incluído |

**Descrição:** Zonos v0.1 Hybrid is a text-to-speech model from Zyphra built on a hybrid architecture. It produces English speech output with coverage across American and British accents in male and female voices. It is suited for English-language voice applications requiring accent and gender variety.

### 55. Sesame: CSM 1B

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `sesame/csm-1b` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 4,096 tokens |
| **Entrada** | text |
| **Saída** | speech |
| **Preço Input** | $0.000007 per 1M characters |
| **Preço Output** | Incluído |

**Descrição:** CSM 1B is a conversational speech model from Sesame. It accepts text input and produces English speech output, with voice options spanning conversational and read-speech styles. At 1B parameters, it is suited for dialogue-oriented applications such as voice assistants and interactive agents.

### 56. Canopy Labs: Orpheus 3B

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `canopylabs/orpheus-3b-0.1-ft` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 4,096 tokens |
| **Entrada** | text |
| **Saída** | speech |
| **Preço Input** | $0.000007 per 1M characters |
| **Preço Output** | Incluído |

**Descrição:** Orpheus 3B is an English text-to-speech model from Canopy Labs, fine-tuned for natural prosody and expressive delivery. It offers 7 preset voices and is suited for narration, voice assistants, and interactive applications where naturalistic speech is a priority.

### 57. hexgrad: Kokoro 82M

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `hexgrad/kokoro-82m` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 4,096 tokens |
| **Entrada** | text |
| **Saída** | speech |
| **Preço Input** | $0.00000062 per 1M characters |
| **Preço Output** | Incluído |

**Descrição:** Kokoro 82M is a lightweight, open-weight text-to-speech model from hexgrad. It converts text to speech across 8 languages (American and British English, Spanish, French, Hindi, Italian, Japanese, Portuguese, and Chinese) using 54 preset voices organized by language and gender. At 82M parameters, it is well-suited for multilingual TTS deployments where footprint and cost efficiency matter.

### 58. Google: Veo 3.1 Lite

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `google/veo-3.1-lite` |
| **Categoria** | Multimodal (Vídeo) |
| **Contexto** | 0 tokens |
| **Entrada** | text, image |
| **Saída** | video |
| **Preço Input** | $0.05 per second |
| **Preço Output** | $0.03 per second |

**Descrição:** Google's most cost-effective video generation model, designed for high-volume applications and rapid iteration. Veo 3.1 Lite generates 720p and 1080p video from text or image prompts with native synchronized audio at less than 50% of the cost of Veo 3.1 Fast. Supports 4–8 second clips in landscape (16:9) and portrait (9:16) formats, with SynthID watermarking. Ideal for content platforms, short-form video creation, and automated media generation.

### 59. inclusionAI: Ling-2.6-1T

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `inclusionai/ling-2.6-1t` |
| **Categoria** | Código / Programação |
| **Contexto** | 262,144 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.0000003 /M tokens |
| **Preço Output** | $0.0000025 /M tokens |

**Descrição:** Ling-2.6-1T is an instant (instruct) model from inclusionAI and the company’s trillion-parameter flagship, designed for real-world agents that require fast execution and high efficiency at scale. It uses a “fast thinking” approach to reduce costs to roughly a quarter of comparable models while maintaining top-tier performance.  The model achieves state-of-the-art results on benchmarks such as AIME26 and SWE-bench Verified, and is well suited for advanced coding, complex reasoning, and large-s...

### 60. Tencent: Hy3 preview

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `tencent/hy3-preview` |
| **Categoria** | Código / Programação |
| **Contexto** | 262,144 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.000000066 /M tokens |
| **Preço Output** | $0.00000026 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Hy3 preview is a high-efficiency Mixture-of-Experts model from Tencent designed for agentic workflows and production use. It supports configurable reasoning levels across disabled, low, and high modes, allowing it to balance speed and depth depending on the task, while delivering strong code generation and reliable performance across multi-step, real-world workflows.

### 61. Xiaomi: MiMo-V2.5-Pro

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `xiaomi/mimo-v2.5-pro` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 1,048,576 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.000001 /M tokens |
| **Preço Output** | $0.000003 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** MiMo-V2.5-Pro is Xiaomi’s flagship model, delivering strong performance in general agentic capabilities, complex software engineering, and long-horizon tasks, with top rankings on benchmarks such as ClawEval, GDPVal, and SWE-bench Pro. It can independently and autonomously complete professional tasks that would take human experts days or weeks, involving more than a thousand tool calls. Its context length of up to 1M makes it well suited for integration with a wide range of agent frameworks.

### 62. Xiaomi: MiMo-V2.5

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `xiaomi/mimo-v2.5` |
| **Categoria** | Multimodal (Vídeo) |
| **Contexto** | 1,048,576 tokens |
| **Entrada** | text, audio, image, video |
| **Saída** | text |
| **Preço Input** | $0.0000004 /M tokens |
| **Preço Output** | $0.000002 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** MiMo-V2.5 is a native omnimodal model by Xiaomi. It delivers Pro-level agentic performance at roughly half the inference cost, while surpassing MiMo-V2-Omni in multimodal perception across image and video understanding tasks. Its 1M context window supports complete documents, extended conversations, and complex task contexts in a single pass, making it ideal for integration with agent frameworks where strong reasoning, rich perception, and cost efficiency all matter.

### 63. OpenAI: GPT-5.4 Image 2

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/gpt-5.4-image-2` |
| **Categoria** | Multimodal (Texto + Imagem) |
| **Contexto** | 272,000 tokens |
| **Entrada** | image, text, file |
| **Saída** | image, text |
| **Preço Input** | $8e-6 /M tokens |
| **Preço Output** | $15e-6 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** [GPT-5.4](https://openrouter.ai/openai/gpt-5.4) Image 2 combines OpenAI's GPT-5.4 model with state-of-the-art image generation capabilities from GPT Image 2. It enables rich multimodal workflows, allowing users to seamlessly move between reasoning, coding, and visual generation within the same interaction.

### 64. inclusionAI: Ling-2.6-flash

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `inclusionai/ling-2.6-flash` |
| **Categoria** | Código / Programação |
| **Contexto** | 262,144 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000001 /M tokens |
| **Preço Output** | $0.00000003 /M tokens |

**Descrição:** Ling-2.6-flash is an instant (instruct) model from inclusionAI with 104B total parameters and 7.4B active parameters, designed for real-world agents that require fast responses, strong execution, and high token efficiency. It delivers performance comparable to state-of-the-art models at a similar scale while significantly reducing token usage across coding, document processing, and lightweight agent workflows.

### 65. Anthropic: Claude Opus Latest

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `~anthropic/claude-opus-latest` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 1,000,000 tokens |
| **Entrada** | text, image, file |
| **Saída** | text |
| **Preço Input** | $5e-6 /M tokens |
| **Preço Output** | $25e-6 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** This model always redirects to the latest model in the Claude Opus family.

### 66. Baidu: Qianfan-OCR-Fast

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `baidu/qianfan-ocr-fast` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 65,536 tokens |
| **Entrada** | image, text |
| **Saída** | text |
| **Preço Input** | $0.00000068 /M tokens |
| **Preço Output** | $0.00000281 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Qianfan-OCR-Fast is a domain-specific multimodal large model purpose-built for OCR. By leveraging specialized OCR training data while preserving versatile multimodal intelligence, it provides a powerful performance upgrade over Qianfan-OCR.

### 67. Kling: Video O1

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `kwaivgi/kling-video-o1` |
| **Categoria** | Multimodal (Vídeo) |
| **Contexto** | 0 tokens |
| **Entrada** | text, image |
| **Saída** | video |
| **Preço Input** | $0.1120 per second |
| **Preço Output** | Incluído |

**Descrição:** Kling Video O1 is a video generation model from Kuaishou. It supports text and image inputs with video output, enabling text-to-video and image-to-video workflows. It is suited for cinematic content production, with first-frame and last-frame control for precise scene composition. It generates 5 or 10 second clips in 16:9, 9:16, or 1:1 aspect ratios.

### 68. MiniMax: Hailuo 2.3

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `minimax/hailuo-2.3` |
| **Categoria** | Multimodal (Vídeo) |
| **Contexto** | 0 tokens |
| **Entrada** | text, image |
| **Saída** | video |
| **Preço Input** | $0.0817 per second |
| **Preço Output** | Incluído |

**Descrição:** Hailuo 2.3 is a video generation model from MiniMax. It accepts text prompts and reference images as input and generates video output, supporting both text-to-video and image-to-video workflows. It is suited for creative content production, cinematic scene generation, and character animation, with a focus on realistic motion and expressive character rendering.

### 69. MoonshotAI: Kimi K2.6

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `moonshotai/kimi-k2.6` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 262,142 tokens |
| **Entrada** | text, image |
| **Saída** | text |
| **Preço Input** | $0.00000073 /M tokens |
| **Preço Output** | $0.00000349 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Kimi K2.6 is Moonshot AI's next-generation multimodal model, designed for long-horizon coding, coding-driven UI/UX generation, and multi-agent orchestration. It handles complex end-to-end coding tasks across Python, Rust, and Go, and can convert prompts and visual inputs into production-ready interfaces. Its agent swarm architecture scales to hundreds of parallel sub-agents for autonomous task decomposition - delivering documents, websites, and spreadsheets in a single run without human overs...

### 70. Mistral: Voxtral Mini TTS

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `mistralai/voxtral-mini-tts-2603` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 4,096 tokens |
| **Entrada** | text |
| **Saída** | speech |
| **Preço Input** | $0.000016 per 1M characters |
| **Preço Output** | Incluído |

**Descrição:** Voxtral Mini TTS is Mistral's text-to-speech model featuring zero-shot voice cloning and multilingual support. It converts text input into natural-sounding audio output.

### 71. OpenAI: GPT-4o Mini TTS

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/gpt-4o-mini-tts-2025-12-15` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 4,096 tokens |
| **Entrada** | text |
| **Saída** | speech |
| **Preço Input** | $0.0000006 per 1M characters |
| **Preço Output** | Incluído |

**Descrição:** GPT-4o Mini TTS is OpenAI's cost-efficient text-to-speech model. It converts text input into natural-sounding audio output, supporting a variety of voices and tones.

### 72. Google: Gemini Embedding 2 Preview

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `google/gemini-embedding-2-preview` |
| **Categoria** | Multimodal (Vídeo) |
| **Contexto** | 8,192 tokens |
| **Entrada** | text, image, file, audio, video |
| **Saída** | embeddings |
| **Preço Input** | $0.0000002 /M tokens |
| **Preço Output** | $0.00000045 /M tokens |

**Descrição:** Gemini Embedding 2 Preview is Google's first multimodal embedding model. We currently support mapping text and images into a unified vector space for semantic search and retrieval-augmented generation (RAG). It supports input context up to 8,192 tokens and flexible output dimensions from 128 to 3,072 (recommended: 768, 1536, or 3,072). Designed for cross-modal similarity — you can embed a text query and retrieve the most relevant images, or vice versa — making it well-suited for multimodal se...

### 73. Anthropic: Claude Opus 4.7

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `anthropic/claude-opus-4.7` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 1,000,000 tokens |
| **Entrada** | text, image, file |
| **Saída** | text |
| **Preço Input** | $5e-6 /M tokens |
| **Preço Output** | $25e-6 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Opus 4.7 is the next generation of Anthropic's Opus family, built for long-running, asynchronous agents. Building on the coding and agentic strengths of Opus 4.6, it delivers stronger performance on complex, multi-step tasks and more reliable agentic execution across extended workflows. It is especially effective for asynchronous agent pipelines where tasks unfold over time - large codebases, multi-stage debugging, and end-to-end project orchestration.  Beyond coding, Opus 4.7 brings improved...

### 74. Alibaba: Wan 2.7

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `alibaba/wan-2.7` |
| **Categoria** | Multimodal (Vídeo) |
| **Contexto** | 0 tokens |
| **Entrada** | text, image |
| **Saída** | video |
| **Preço Input** | $0.1 per second |
| **Preço Output** | Incluído |

**Descrição:** Wan 2.7 is a video generation model from Alibaba. It supports text-to-video, image-to-video with first and last frame control, and reference-to-video, where multiple reference images guide the style and content of the generated scene.

### 75. ByteDance: Seedance 2.0

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `bytedance/seedance-2.0` |
| **Categoria** | Multimodal (Vídeo) |
| **Contexto** | 0 tokens |
| **Entrada** | text, image |
| **Saída** | video |
| **Preço Input** | $0.000007 /M tokens |
| **Preço Output** | $0.000007 /M tokens |

**Descrição:** Seedance 2.0 is a video generation model from ByteDance. It supports text-to-video, image-to-video with first and last frame control, and multimodal reference-to-video. It is particularly strong at preserving character consistency, visual style, and camera movement from reference material. The number of tokens is given by (height of output video * width of output video * duration * 24) / 1024

### 76. ByteDance: Seedance 2.0 Fast

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `bytedance/seedance-2.0-fast` |
| **Categoria** | Multimodal (Vídeo) |
| **Contexto** | 0 tokens |
| **Entrada** | text, image |
| **Saída** | video |
| **Preço Input** | $0.0000056 /M tokens |
| **Preço Output** | $0.0000056 /M tokens |

**Descrição:** Seedance 2.0 Fast is a video generation model from ByteDance. It supports text-to-video, image-to-video with first and last frame control, and multimodal reference-to-video. It prioritizes generation speed and lower cost over maximum output quality. The number of tokens is given by (height of output video * width of output video * duration * 24) / 1024

### 77. Anthropic: Claude Opus 4.6 (Fast)

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `anthropic/claude-opus-4.6-fast` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 1,000,000 tokens |
| **Entrada** | text, image, file |
| **Saída** | text |
| **Preço Input** | $30e-6 /M tokens |
| **Preço Output** | $150e-6 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Fast-mode variant of [Opus 4.6](/anthropic/claude-opus-4.6) - identical capabilities with higher output speed at premium 6x pricing.  Learn more in Anthropic's docs: https://platform.claude.com/docs/en/build-with-claude/fast-mode

### 78. Z.ai: GLM 5.1

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `z-ai/glm-5.1` |
| **Categoria** | Código / Programação |
| **Contexto** | 202,752 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000098 /M tokens |
| **Preço Output** | $0.00000308 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** GLM-5.1 delivers a major leap in coding capability, with particularly significant gains in handling long-horizon tasks. Unlike previous models built around minute-level interactions, GLM-5.1 can work independently and continuously on a single task for more than 8 hours, autonomously planning, executing, and improving itself throughout the process, ultimately delivering complete, engineering-grade results.

### 79. Cohere: Rerank 4 Pro

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `cohere/rerank-4-pro` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 32,768 tokens |
| **Entrada** | text |
| **Saída** | rerank |
| **Preço Input** | $0.0025 per search |
| **Preço Output** | Incluído |

**Descrição:** Cohere's AI search foundation model for enhancing the relevance of information surfaced within search and RAG systems. Features a 32K context window, multilingual support across 100+ languages, no data pre-processing required, and state of the art performance with low latency.

### 80. Cohere: Rerank 4 Fast

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `cohere/rerank-4-fast` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 32,768 tokens |
| **Entrada** | text |
| **Saída** | rerank |
| **Preço Input** | $0.002 per search |
| **Preço Output** | Incluído |

**Descrição:** Cohere's AI search foundation model for enhancing the relevance of information surfaced within search and RAG systems. Features a 32K context window, multilingual support across 100+ languages, no data pre-processing required, and high performance with lowest latency.

### 81. Cohere: Rerank v3.5

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `cohere/rerank-v3.5` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 4,096 tokens |
| **Entrada** | text |
| **Saída** | rerank |
| **Preço Input** | $0.001 per search |
| **Preço Output** | Incluído |

**Descrição:** Rerank v3.5 is designed to reorder search results for improved relevance. It supports multi-aspect and semi-structured data reranking over 100+ languages. Ideal for refining results from semantic or keyword search pipelines.

### 82. Google: Gemma 4 26B A4B  (free)

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `google/gemma-4-26b-a4b-it` |
| **Categoria** | Multimodal (Vídeo) |
| **Contexto** | 262,144 tokens |
| **Entrada** | image, text, video |
| **Saída** | text |
| **Preço Input** | $0 /M tokens |
| **Preço Output** | $0 /M tokens |
| **Gratuito** | Sim |
| **Raciocínio** | Suportado |

**Descrição:** Gemma 4 26B A4B IT is an instruction-tuned Mixture-of-Experts (MoE) model from Google DeepMind. Despite 25.2B total parameters, only 3.8B activate per token during inference — delivering near-31B quality at a fraction of the compute cost. Supports multimodal input including text, images, and video (up to 60s at 1fps). Features a 256K token context window, native function calling, configurable thinking/reasoning mode, and structured output support. Released under Apache 2.0.

### 83. Google: Gemma 4 26B A4B 

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `google/gemma-4-26b-a4b-it` |
| **Categoria** | Multimodal (Vídeo) |
| **Contexto** | 262,144 tokens |
| **Entrada** | image, text, video |
| **Saída** | text |
| **Preço Input** | $0.00000006 /M tokens |
| **Preço Output** | $0.00000033 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Gemma 4 26B A4B IT is an instruction-tuned Mixture-of-Experts (MoE) model from Google DeepMind. Despite 25.2B total parameters, only 3.8B activate per token during inference — delivering near-31B quality at a fraction of the compute cost. Supports multimodal input including text, images, and video (up to 60s at 1fps). Features a 256K token context window, native function calling, configurable thinking/reasoning mode, and structured output support. Released under Apache 2.0.

### 84. Google: Gemma 4 31B (free)

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `google/gemma-4-31b-it` |
| **Categoria** | Multimodal (Vídeo) |
| **Contexto** | 262,144 tokens |
| **Entrada** | image, text, video |
| **Saída** | text |
| **Preço Input** | $0 /M tokens |
| **Preço Output** | $0 /M tokens |
| **Gratuito** | Sim |
| **Raciocínio** | Suportado |

**Descrição:** Gemma 4 31B Instruct is Google DeepMind's 30.7B dense multimodal model supporting text and image input with text output. Features a 256K token context window, configurable thinking/reasoning mode, native function calling, and multilingual support across 140+ languages. Strong on coding, reasoning, and document understanding tasks. Apache 2.0 license.

### 85. Google: Gemma 4 31B

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `google/gemma-4-31b-it` |
| **Categoria** | Multimodal (Vídeo) |
| **Contexto** | 262,144 tokens |
| **Entrada** | image, text, video |
| **Saída** | text |
| **Preço Input** | $0.00000012 /M tokens |
| **Preço Output** | $0.00000037 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Gemma 4 31B Instruct is Google DeepMind's 30.7B dense multimodal model supporting text and image input with text output. Features a 256K token context window, configurable thinking/reasoning mode, native function calling, and multilingual support across 140+ languages. Strong on coding, reasoning, and document understanding tasks. Apache 2.0 license.

### 86. Qwen: Qwen3.6 Plus

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `qwen/qwen3.6-plus` |
| **Categoria** | Multimodal (Vídeo) |
| **Contexto** | 1,000,000 tokens |
| **Entrada** | text, image, video |
| **Saída** | text |
| **Preço Input** | $0.000000325 /M tokens |
| **Preço Output** | $0.00000195 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Qwen 3.6 Plus builds on a hybrid architecture that combines efficient linear attention with sparse mixture-of-experts routing, enabling strong scalability and high-performance inference. Compared to the 3.5 series, it delivers major gains in agentic coding, front-end development, and overall reasoning, with a significantly improved “vibe coding” experience. The model excels at complex tasks such as 3D scenes, games, and repository-level problem solving, achieving a 78.8 score on SWE-bench Ver...

### 87. Z.ai: GLM 5V Turbo

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `z-ai/glm-5v-turbo` |
| **Categoria** | Multimodal (Vídeo) |
| **Contexto** | 202,752 tokens |
| **Entrada** | image, text, video |
| **Saída** | text |
| **Preço Input** | $1.2e-6 /M tokens |
| **Preço Output** | $4e-6 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** GLM-5V-Turbo is Z.ai’s first native multimodal agent foundation model, built for vision-based coding and agent-driven tasks. It natively handles image, video, and text inputs, excels at long-horizon planning, complex coding, and task execution, and works seamlessly with agents to complete the full loop of “perceive → plan → execute“.

### 88. Arcee AI: Trinity Large Thinking (free)

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `arcee-ai/trinity-large-thinking` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 262,144 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0 /M tokens |
| **Preço Output** | $0 /M tokens |
| **Gratuito** | Sim |
| **Raciocínio** | Suportado |

**Descrição:** Trinity Large Thinking is a powerful open source reasoning model from the team at Arcee AI. It shows strong performance in PinchBench, agentic workloads, and reasoning tasks. Launch video: https://youtu.be/Gc82AXLa0Rg?si=4RLn6WBz33qT--B7  This model is optimized for agentic workflows and performs best when reasoning is preserved (aka interleaved thinking). Learn how to preserve reasoning in our docs: https://openrouter.ai/docs/guides/best-practices/reasoning-tokens#preserving-reasoning

### 89. Arcee AI: Trinity Large Thinking

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `arcee-ai/trinity-large-thinking` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 262,144 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000022 /M tokens |
| **Preço Output** | $0.00000085 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Trinity Large Thinking is a powerful open source reasoning model from the team at Arcee AI. It shows strong performance in PinchBench, agentic workloads, and reasoning tasks. Launch video: https://youtu.be/Gc82AXLa0Rg?si=4RLn6WBz33qT--B7  This model is optimized for agentic workflows and performs best when reasoning is preserved (aka interleaved thinking). Learn how to preserve reasoning in our docs: https://openrouter.ai/docs/guides/best-practices/reasoning-tokens#preserving-reasoning

### 90. xAI: Grok 4.20 Multi-Agent

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `x-ai/grok-4.20-multi-agent` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 2,000,000 tokens |
| **Entrada** | text, image, file |
| **Saída** | text |
| **Preço Input** | $2e-6 /M tokens |
| **Preço Output** | $6e-6 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Grok 4.20 Multi-Agent is a variant of xAI’s Grok 4.20 designed for collaborative, agent-based workflows. Multiple agents operate in parallel to conduct deep research, coordinate tool use, and synthesize information across complex tasks.  Reasoning effort behavior: - low / medium: 4 agents - high / xhigh: 16 agents

### 91. xAI: Grok 4.20

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `x-ai/grok-4.20` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 2,000,000 tokens |
| **Entrada** | text, image, file |
| **Saída** | text |
| **Preço Input** | $1.25e-6 /M tokens |
| **Preço Output** | $2.5e-6 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Grok 4.20 is a reasoning model from xAI with industry-leading speed and agentic tool calling capabilities. It combines the lowest hallucination rate on the market with strict prompt adherance, delivering consistently precise and truthful responses.  Reasoning can be enabled/disabled using the `reasoning` `enabled` parameter in the API. [Learn more in our docs](https://openrouter.ai/docs/use-cases/reasoning-tokens#controlling-reasoning-tokens)

### 92. Google: Lyria 3 Pro Preview

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `google/lyria-3-pro-preview` |
| **Categoria** | Multimodal (Texto + Áudio) |
| **Contexto** | 1,048,576 tokens |
| **Entrada** | text, image |
| **Saída** | text, audio |
| **Preço Input** | $0.08 per song |
| **Preço Output** | Incluído |

**Descrição:** Full-length songs are priced at $0.08 per song. Lyria 3 is Google's family of music generation models, available through the Gemini API. With Lyria 3, you can generate high-quality, 48kHz stereo audio from text prompts or from images. These models deliver structural coherence, including vocals, timed lyrics, and full instrumental arrangements. Lyria 3 Pro can generate full-length songs with verses, choruses, bridges.

### 93. Google: Lyria 3 Clip Preview

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `google/lyria-3-clip-preview` |
| **Categoria** | Multimodal (Texto + Áudio) |
| **Contexto** | 1,048,576 tokens |
| **Entrada** | text, image |
| **Saída** | text, audio |
| **Preço Input** | $0.04 per song |
| **Preço Output** | Incluído |

**Descrição:** 30 second duration clips are priced at $0.04 per clip. Lyria 3 is Google's family of music generation models, available through the Gemini API. With Lyria 3, you can generate high-quality, 48kHz stereo audio from text prompts or from images. These models deliver structural coherence, including vocals, timed lyrics, and full instrumental arrangements. Lyria 3 Clip can generate short clips, loops, previews.

### 94. Alibaba: Wan 2.6

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `alibaba/wan-2.6` |
| **Categoria** | Multimodal (Vídeo) |
| **Contexto** | 0 tokens |
| **Entrada** | text, image |
| **Saída** | video |
| **Preço Input** | $0.04 per second |
| **Preço Output** | $0.10 per second |

**Descrição:** Alibaba's most advanced video generation model, supporting over 10 visual creation capabilities in a unified system. Wan 2.6 generates 1080p video at 24fps from text, images, reference videos, or audio, with native audio-visual synchronization and precise lip-sync. Key features include reference-to-video (insert a character's appearance and voice into new scenes), multi-shot storytelling from simple prompts, synchronized sound effects and music, and support for 16:9, 9:16, and 1:1 aspect rati...

### 95. Kwaipilot: KAT-Coder-Pro V2

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `kwaipilot/kat-coder-pro-v2` |
| **Categoria** | Código / Programação |
| **Contexto** | 256,000 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.0000003 /M tokens |
| **Preço Output** | $0.0000012 /M tokens |

**Descrição:** KAT-Coder-Pro V2 is the latest high-performance model in KwaiKAT’s KAT-Coder series, designed for complex enterprise-grade software engineering and SaaS integration. It builds on the agentic coding strengths of earlier versions, with a focus on large-scale production environments, multi-system coordination, and seamless integration across modern software stacks, while also supporting web aesthetics generation to produce production-grade landing pages and presentation decks.

### 96. ByteDance: Seedance 1.5 Pro

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `bytedance/seedance-1-5-pro` |
| **Categoria** | Multimodal (Vídeo) |
| **Contexto** | 0 tokens |
| **Entrada** | text, image |
| **Saída** | video |
| **Preço Input** | $0.0000024 /M tokens |
| **Preço Output** | $0.0000012 /M tokens |

**Descrição:** ByteDance's next-generation audio-visual generation model with a 4.5B parameter Dual-Branch Diffusion Transformer architecture. Seedance 1.5 Pro generates video and audio simultaneously in a single unified pass — eliminating the timing issues of sequential audio dubbing. Supports multi-language lip-sync (English, Mandarin, Japanese, Korean, Spanish, and more), cinematic camera control (pan, tilt, zoom, orbit), multi-character dialogue, and character consistency across shots. Produces clips fr...

### 97. OpenAI: Sora 2 Pro

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/sora-2-pro` |
| **Categoria** | Multimodal (Vídeo) |
| **Contexto** | 0 tokens |
| **Entrada** | text, image |
| **Saída** | video |
| **Preço Input** | $0.30 per second |
| **Preço Output** | Incluído |

**Descrição:** OpenAI's flagship video generation model, delivering production-quality video with physics-accurate motion, synchronized audio, and world-state persistence across shots. Sora 2 Pro follows intricate multi-shot instructions while maintaining consistent spatial relationships — objects don't disappear or change shape between cuts. Supports text-to-video and image-to-video, with synchronized background soundscapes, speech, and sound effects. Includes advanced content safety with C2PA metadata pro...

### 98. Google: Veo 3.1

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `google/veo-3.1` |
| **Categoria** | Multimodal (Vídeo) |
| **Contexto** | 0 tokens |
| **Entrada** | text, image |
| **Saída** | video |
| **Preço Input** | $0.40 per second |
| **Preço Output** | $0.20 per second |

**Descrição:** Google's state-of-the-art video generation model, built for maximum visual fidelity in final production cuts. Veo 3.1 generates high-quality 1080p video from text or image prompts with native synchronized audio — including dialogue, ambient effects, and background sound. Supports scene extension (up to 20 chained clips for 140+ second narratives), frames-to-video transitions between two images, vertical video for Shorts, and 4K upscaling.

### 99. Reka Edge

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `rekaai/reka-edge` |
| **Categoria** | Multimodal (Vídeo) |
| **Contexto** | 16,384 tokens |
| **Entrada** | image, text, video |
| **Saída** | text |
| **Preço Input** | $0.0000001 /M tokens |
| **Preço Output** | $0.0000001 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Reka Edge is an extremely efficient 7B multimodal vision-language model that accepts image/video+text inputs and generates text outputs. This model is optimized specifically to deliver industry-leading performance in image understanding, video analysis, object detection, and agentic tool-use.

### 100. Xiaomi: MiMo-V2-Omni

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `xiaomi/mimo-v2-omni` |
| **Categoria** | Multimodal (Vídeo) |
| **Contexto** | 262,144 tokens |
| **Entrada** | text, audio, image, video |
| **Saída** | text |
| **Preço Input** | $0.0000004 /M tokens |
| **Preço Output** | $0.000002 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** MiMo-V2-Omni is a frontier omni-modal model that natively processes image, video, and audio inputs within a unified architecture. It combines strong multimodal perception with agentic capability - visual grounding, multi-step planning, tool use, and code execution - making it well-suited for complex real-world tasks that span modalities, 256K context window.

### 101. Xiaomi: MiMo-V2-Pro

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `xiaomi/mimo-v2-pro` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 1,048,576 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.000001 /M tokens |
| **Preço Output** | $0.000003 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** MiMo-V2-Pro is Xiaomi's flagship foundation model, featuring over 1T total parameters and a 1M context length, deeply optimized for agentic scenarios. It is highly adaptable to general agent frameworks like OpenClaw. It ranks among the global top tier in the standard PinchBench and ClawBench benchmarks, with perceived performance approaching that of Opus 4.6. MiMo-V2-Pro is designed to serve as the brain of agent systems, orchestrating complex workflows, driving production engineering tasks, ...

### 102. MiniMax: MiniMax M2.7

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `minimax/minimax-m2.7` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 196,608 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.000000279 /M tokens |
| **Preço Output** | $0.0000012 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** MiniMax-M2.7 is a next-generation large language model designed for autonomous, real-world productivity and continuous improvement. Built to actively participate in its own evolution, M2.7 integrates advanced agentic capabilities through multi-agent collaboration, enabling it to plan, execute, and refine complex tasks across dynamic environments.  Trained for production-grade performance, M2.7 handles workflows such as live debugging, root cause analysis, financial modeling, and full document...

### 103. OpenAI: GPT-5.4 Nano

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/gpt-5.4-nano` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 400,000 tokens |
| **Entrada** | file, image, text |
| **Saída** | text |
| **Preço Input** | $0.2e-6 /M tokens |
| **Preço Output** | $1.25e-6 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** GPT-5.4 nano is the most lightweight and cost-efficient variant of the GPT-5.4 family, optimized for speed-critical and high-volume tasks. It supports text and image inputs and is designed for low-latency use cases such as classification, data extraction, ranking, and sub-agent execution.  The model prioritizes responsiveness and efficiency over deep reasoning, making it ideal for pipelines that require fast, reliable outputs at scale. GPT-5.4 nano is well suited for background tasks, real-ti...

### 104. OpenAI: GPT-5.4 Mini

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/gpt-5.4-mini` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 400,000 tokens |
| **Entrada** | file, image, text |
| **Saída** | text |
| **Preço Input** | $0.75e-6 /M tokens |
| **Preço Output** | $4.5e-6 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** GPT-5.4 mini brings the core capabilities of GPT-5.4 to a faster, more efficient model optimized for high-throughput workloads. It supports text and image inputs with strong performance across reasoning, coding, and tool use, while reducing latency and cost for large-scale deployments.  The model is designed for production environments that require a balance of capability and efficiency, making it well suited for chat applications, coding assistants, and agent workflows that operate at scale....

### 105. Mistral: Mistral Small 4

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `mistralai/mistral-small-2603` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 262,144 tokens |
| **Entrada** | text, image |
| **Saída** | text |
| **Preço Input** | $0.15e-6 /M tokens |
| **Preço Output** | $0.6e-6 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Mistral Small 4 is the next major release in the Mistral Small family, unifying the capabilities of several flagship Mistral models into a single system. It combines strong reasoning from Magistral, multimodal understanding from Pixtral, and agentic coding capabilities from Devstral, enabling one model to handle complex analysis, software development, and visual tasks within the same workflow.

### 106. Perplexity: Embed V1 4B

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `perplexity/pplx-embed-v1-4b` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 32,000 tokens |
| **Entrada** | text |
| **Saída** | embeddings |
| **Preço Input** | $3e-8 /M tokens |
| **Preço Output** | Incluído |

**Descrição:** pplx-embed-v1 -4B is one of Perplexity's state-of-the-art text embedding models built for real-world, web-scale retrieval. pplx-embed-v1 is optimized for standard dense text retrieval with the 4B parameter model maximizing retrieval quality.

### 107. Perplexity: Embed V1 0.6B

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `perplexity/pplx-embed-v1-0.6b` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 32,000 tokens |
| **Entrada** | text |
| **Saída** | embeddings |
| **Preço Input** | $4e-9 /M tokens |
| **Preço Output** | Incluído |

**Descrição:** pplx-embed-v1-0.6B is one of Perplexity's state-of-the-art text embedding models built for real-world, web-scale retrieval. pplx-embed-v1 is optimized for standard dense text retrieval with the 0.6B parameter model targeting lightweight, low-latency embedding generation.

### 108. Z.ai: GLM 5 Turbo

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `z-ai/glm-5-turbo` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 202,752 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $1.2e-6 /M tokens |
| **Preço Output** | $4e-6 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** GLM-5 Turbo is a new model from Z.ai designed for fast inference and strong performance in agent-driven environments such as OpenClaw scenarios. It is deeply optimized for real-world agent workflows involving long execution chains, with improved complex instruction decomposition, tool use, scheduled and persistent execution, and overall stability across extended tasks.

### 109. NVIDIA: Nemotron 3 Super (free)

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `nvidia/nemotron-3-super-120b-a12b` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 262,144 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0 /M tokens |
| **Preço Output** | $0 /M tokens |
| **Gratuito** | Sim |
| **Raciocínio** | Suportado |

**Descrição:** NVIDIA Nemotron 3 Super is a 120B-parameter open hybrid MoE model, activating just 12B parameters for maximum compute efficiency and accuracy in complex multi-agent applications. Built on a hybrid Mamba-Transformer Mixture-of-Experts architecture with multi-token prediction (MTP), it delivers over 50% higher token generation compared to leading open models.   The model features a 1M token context window for long-term agent coherence, cross-document reasoning, and multi-step task planning. Lat...

### 110. NVIDIA: Nemotron 3 Super

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `nvidia/nemotron-3-super-120b-a12b` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 262,144 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000009 /M tokens |
| **Preço Output** | $0.00000045 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** NVIDIA Nemotron 3 Super is a 120B-parameter open hybrid MoE model, activating just 12B parameters for maximum compute efficiency and accuracy in complex multi-agent applications. Built on a hybrid Mamba-Transformer Mixture-of-Experts architecture with multi-token prediction (MTP), it delivers over 50% higher token generation compared to leading open models.   The model features a 1M token context window for long-term agent coherence, cross-document reasoning, and multi-step task planning. Lat...

### 111. ByteDance Seed: Seed-2.0-Lite

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `bytedance-seed/seed-2.0-lite` |
| **Categoria** | Multimodal (Vídeo) |
| **Contexto** | 262,144 tokens |
| **Entrada** | text, image, video |
| **Saída** | text |
| **Preço Input** | $0.00000025 /M tokens |
| **Preço Output** | $0.000002 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Seed-2.0-Lite is a versatile, cost‑efficient enterprise workhorse that delivers strong multimodal and agent capabilities while offering noticeably lower latency, making it a practical default choice for most production workloads across text, vision, and tools. Engineered for high-frequency visual understanding and agentic workflows, it's an ideal choice for deployment at scale with minimal latency.

### 112. Qwen: Qwen3.5-9B

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `qwen/qwen3.5-9b` |
| **Categoria** | Multimodal (Vídeo) |
| **Contexto** | 262,144 tokens |
| **Entrada** | text, image, video |
| **Saída** | text |
| **Preço Input** | $0.00000004 /M tokens |
| **Preço Output** | $0.00000015 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Qwen3.5-9B is a multimodal foundation model from the Qwen3.5 family, designed to deliver strong reasoning, coding, and visual understanding in an efficient 9B-parameter architecture. It uses a unified vision-language design with early fusion of multimodal tokens, allowing the model to process and reason across text and images within the same context.

### 113. OpenAI: GPT-5.4 Pro

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/gpt-5.4-pro` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 1,050,000 tokens |
| **Entrada** | text, image, file |
| **Saída** | text |
| **Preço Input** | $30e-6 /M tokens |
| **Preço Output** | $180e-6 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** GPT-5.4 Pro is OpenAI's most advanced model, building on GPT-5.4's unified architecture with enhanced reasoning capabilities for complex, high-stakes tasks. It features a 1M+ token context window (922K input, 128K output) with support for text and image inputs. Optimized for step-by-step reasoning, instruction following, and accuracy, GPT-5.4 Pro excels at agentic coding, long-context workflows, and multi-step problem solving.

### 114. OpenAI: GPT-5.4

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/gpt-5.4` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 1,050,000 tokens |
| **Entrada** | text, image, file |
| **Saída** | text |
| **Preço Input** | $2.5e-6 /M tokens |
| **Preço Output** | $15e-6 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** GPT-5.4 is OpenAI’s latest frontier model, unifying the Codex and GPT lines into a single system. It features a 1M+ token context window (922K input, 128K output) with support for text and image inputs, enabling high-context reasoning, coding, and multimodal analysis within the same workflow.  The model delivers improved performance in coding, document understanding, tool use, and instruction following. It is designed as a strong default for both general-purpose tasks and software engineering...

### 115. Inception: Mercury 2

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `inception/mercury-2` |
| **Categoria** | Código / Programação |
| **Contexto** | 128,000 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000025 /M tokens |
| **Preço Output** | $0.00000075 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Mercury 2 is an extremely fast reasoning LLM, and the first reasoning diffusion LLM (dLLM). Instead of generating tokens sequentially, Mercury 2 produces and refines multiple tokens in parallel, achieving >1,000 tokens/sec on standard GPUs. Mercury 2 is 5x+ faster than leading speed-optimized LLMs like Claude 4.5 Haiku and GPT 5 Mini, at a fraction of the cost.  Mercury 2 supports tunable reasoning levels, 128K context, native tool use, and schema-aligned JSON output. Built for coding workflo...

### 116. OpenAI: GPT-5.3 Chat

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/gpt-5.3-chat` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 128,000 tokens |
| **Entrada** | text, image, file |
| **Saída** | text |
| **Preço Input** | $0.00000175 /M tokens |
| **Preço Output** | $0.000014 /M tokens |

**Descrição:** GPT-5.3 Chat is an update to ChatGPT's most-used model that makes everyday conversations smoother, more useful, and more directly helpful. It delivers more accurate answers with better contextualization and significantly reduces unnecessary refusals, caveats, and overly cautious phrasing that can interrupt conversational flow.

### 117. Google: Gemini 3.1 Flash Lite Preview

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `google/gemini-3.1-flash-lite-preview` |
| **Categoria** | Multimodal (Vídeo) |
| **Contexto** | 1,048,576 tokens |
| **Entrada** | text, image, video, file, audio |
| **Saída** | text |
| **Preço Input** | $0.25e-6 /M tokens |
| **Preço Output** | $1.5e-6 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Gemini 3.1 Flash Lite Preview is Google's high-efficiency model optimized for high-volume use cases. It outperforms Gemini 2.5 Flash Lite on overall quality and approaches Gemini 2.5 Flash performance across key capabilities. Improvements span audio input/ASR, RAG snippet ranking, translation, data extraction, and code completion. Supports full thinking levels (minimal, low, medium, high) for fine-grained cost/performance trade-offs. Priced at half the cost of Gemini 3 Flash.

### 118. ByteDance Seed: Seed-2.0-Mini

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `bytedance-seed/seed-2.0-mini` |
| **Categoria** | Multimodal (Vídeo) |
| **Contexto** | 262,144 tokens |
| **Entrada** | text, image, video |
| **Saída** | text |
| **Preço Input** | $0.0000001 /M tokens |
| **Preço Output** | $0.0000004 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Seed-2.0-mini targets latency-sensitive, high-concurrency, and cost-sensitive scenarios, emphasizing fast response and flexible inference deployment. It delivers performance comparable to ByteDance-Seed-1.6, supports 256k context, four reasoning effort modes (minimal/low/medium/high), multimodal understanding, and is optimized for lightweight tasks where cost and speed take priority.

### 119. Google: Nano Banana 2 (Gemini 3.1 Flash Image Preview)

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `google/gemini-3.1-flash-image-preview` |
| **Categoria** | Multimodal (Texto + Imagem) |
| **Contexto** | 65,536 tokens |
| **Entrada** | image, text |
| **Saída** | image, text |
| **Preço Input** | $0.5e-6 /M tokens |
| **Preço Output** | $3e-6 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Gemini 3.1 Flash Image Preview, a.k.a. "Nano Banana 2," is Google’s latest state of the art image generation and editing model, delivering Pro-level visual quality at Flash speed. It combines advanced contextual understanding with fast, cost-efficient inference, making complex image generation and iterative edits significantly more accessible. Aspect ratios can be controlled with the [image_config API Parameter](https://openrouter.ai/docs/features/multimodal/image-generation#image-aspect-rati...

### 120. Qwen: Qwen3.5-35B-A3B

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `qwen/qwen3.5-35b-a3b` |
| **Categoria** | Multimodal (Vídeo) |
| **Contexto** | 262,144 tokens |
| **Entrada** | text, image, video |
| **Saída** | text |
| **Preço Input** | $0.00000014 /M tokens |
| **Preço Output** | $0.000001 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** The Qwen3.5 Series 35B-A3B is a native vision-language model designed with a hybrid architecture that integrates linear attention mechanisms and a sparse mixture-of-experts model, achieving higher inference efficiency. Its overall performance is comparable to that of the Qwen3.5-27B.

### 121. Qwen: Qwen3.5-27B

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `qwen/qwen3.5-27b` |
| **Categoria** | Multimodal (Vídeo) |
| **Contexto** | 262,144 tokens |
| **Entrada** | text, image, video |
| **Saída** | text |
| **Preço Input** | $0.000000195 /M tokens |
| **Preço Output** | $0.00000156 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** The Qwen3.5 27B native vision-language Dense model incorporates a linear attention mechanism, delivering fast response times while balancing inference speed and performance. Its overall capabilities are comparable to those of the Qwen3.5-122B-A10B.

### 122. Qwen: Qwen3.5-122B-A10B

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `qwen/qwen3.5-122b-a10b` |
| **Categoria** | Multimodal (Vídeo) |
| **Contexto** | 262,144 tokens |
| **Entrada** | text, image, video |
| **Saída** | text |
| **Preço Input** | $0.00000026 /M tokens |
| **Preço Output** | $0.00000208 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** The Qwen3.5 122B-A10B native vision-language model is built on a hybrid architecture that integrates a linear attention mechanism with a sparse mixture-of-experts model, achieving higher inference efficiency. In terms of overall performance, this model is second only to Qwen3.5-397B-A17B. Its text capabilities significantly outperform those of Qwen3-235B-2507, and its visual capabilities surpass those of Qwen3-VL-235B.

### 123. Qwen: Qwen3.5-Flash

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `qwen/qwen3.5-flash-02-23` |
| **Categoria** | Multimodal (Vídeo) |
| **Contexto** | 1,000,000 tokens |
| **Entrada** | text, image, video |
| **Saída** | text |
| **Preço Input** | $0.000000065 /M tokens |
| **Preço Output** | $0.00000026 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** The Qwen3.5 native vision-language Flash models are built on a hybrid architecture that integrates a linear attention mechanism with a sparse mixture-of-experts model, achieving higher inference efficiency. Compared to the 3 series, these models deliver a leap forward in performance for both pure text and multimodal tasks, offering fast response times while balancing inference speed and overall performance.

### 124. LiquidAI: LFM2-24B-A2B

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `liquid/lfm-2-24b-a2b` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 32,768 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000003 /M tokens |
| **Preço Output** | $0.00000012 /M tokens |

**Descrição:** LFM2-24B-A2B is the largest model in the LFM2 family of hybrid architectures designed for efficient on-device deployment. Built as a 24B parameter Mixture-of-Experts model with only 2B active parameters per token, it delivers high-quality generation while maintaining low inference costs. The model fits within 32 GB of RAM, making it practical to run on consumer laptops and desktops without sacrificing capability.

### 125. Google: Gemini 3.1 Pro Preview Custom Tools

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `google/gemini-3.1-pro-preview-customtools` |
| **Categoria** | Multimodal (Vídeo) |
| **Contexto** | 1,048,576 tokens |
| **Entrada** | text, audio, image, video, file |
| **Saída** | text |
| **Preço Input** | $0.000002 /M tokens |
| **Preço Output** | $0.000012 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Gemini 3.1 Pro Preview Custom Tools is a variant of Gemini 3.1 Pro that improves tool selection behavior by preventing overuse of a general bash tool when more efficient third-party or user-defined functions are available. This specialized preview endpoint significantly increases function calling reliability and ensures the model selects the most appropriate tool in coding agents and complex, multi-tool workflows.  It retains the core strengths of Gemini 3.1 Pro, including multimodal reasonin...

### 126. NVIDIA: Llama Nemotron Embed VL 1B V2 (free)

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `nvidia/llama-nemotron-embed-vl-1b-v2` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 131,072 tokens |
| **Entrada** | text, image |
| **Saída** | embeddings |
| **Preço Input** | $0 /M tokens |
| **Preço Output** | $0 /M tokens |
| **Gratuito** | Sim |

**Descrição:** The Llama Nemotron Embed VL 1B V2 embedding model is optimized for multimodal question-answering retrieval. The model can embed 'documents' in the form of image, text, or image and text combined. Documents can be retrieved given a user query in text form. The model supports images containing text, tables, charts, and infographics.

### 127. OpenAI: GPT-5.3-Codex

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/gpt-5.3-codex` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 400,000 tokens |
| **Entrada** | text, image, file |
| **Saída** | text |
| **Preço Input** | $0.00000175 /M tokens |
| **Preço Output** | $0.000014 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** GPT-5.3-Codex is OpenAI’s most advanced agentic coding model, combining the frontier software engineering performance of GPT-5.2-Codex with the broader reasoning and professional knowledge capabilities of GPT-5.2. It achieves state-of-the-art results on SWE-Bench Pro and strong performance on Terminal-Bench 2.0 and OSWorld-Verified, reflecting improved multi-language coding, terminal proficiency, and real-world computer-use skills. The model is optimized for long-running, tool-using workflows...

### 128. AionLabs: Aion-2.0

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `aion-labs/aion-2.0` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 131,072 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.0000008 /M tokens |
| **Preço Output** | $0.0000016 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Aion-2.0 is a variant of DeepSeek V3.2 optimized for immersive roleplaying and storytelling. It is particularly strong at introducing tension, crises, and conflict into stories, making narratives feel more engaging. It also handles mature and darker themes with more nuance and depth.

### 129. Google: Gemini 3.1 Pro Preview

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `google/gemini-3.1-pro-preview` |
| **Categoria** | Multimodal (Vídeo) |
| **Contexto** | 1,048,576 tokens |
| **Entrada** | audio, file, image, text, video |
| **Saída** | text |
| **Preço Input** | $0.000002 /M tokens |
| **Preço Output** | $0.000012 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Gemini 3.1 Pro Preview is Google’s frontier reasoning model, delivering enhanced software engineering performance, improved agentic reliability, and more efficient token usage across complex workflows. Building on the multimodal foundation of the Gemini 3 series, it combines high-precision reasoning across text, image, video, audio, and code with a 1M-token context window. Reasoning Details must be preserved when using multi-turn tool calling, see our docs here: https://openrouter.ai/docs/use...

### 130. Anthropic: Claude Sonnet 4.6

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `anthropic/claude-sonnet-4.6` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 1,000,000 tokens |
| **Entrada** | text, image, file |
| **Saída** | text |
| **Preço Input** | $0.000003 /M tokens |
| **Preço Output** | $0.000015 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Sonnet 4.6 is Anthropic's most capable Sonnet-class model yet, with frontier performance across coding, agents, and professional work. It excels at iterative development, complex codebase navigation, end-to-end project management with memory, polished document creation, and confident computer use for web QA and workflow automation.

### 131. Qwen: Qwen3.5 Plus 2026-02-15

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `qwen/qwen3.5-plus-02-15` |
| **Categoria** | Multimodal (Vídeo) |
| **Contexto** | 1,000,000 tokens |
| **Entrada** | text, image, video |
| **Saída** | text |
| **Preço Input** | $0.00000026 /M tokens |
| **Preço Output** | $0.00000156 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** The Qwen3.5 native vision-language series Plus models are built on a hybrid architecture that integrates linear attention mechanisms with sparse mixture-of-experts models, achieving higher inference efficiency. In a variety of task evaluations, the 3.5 series consistently demonstrates performance on par with state-of-the-art leading models. Compared to the 3 series, these models show a leap forward in both pure-text and multimodal capabilities.

### 132. Qwen: Qwen3.5 397B A17B

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `qwen/qwen3.5-397b-a17b` |
| **Categoria** | Multimodal (Vídeo) |
| **Contexto** | 262,144 tokens |
| **Entrada** | text, image, video |
| **Saída** | text |
| **Preço Input** | $0.00000039 /M tokens |
| **Preço Output** | $0.00000234 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** The Qwen3.5 series 397B-A17B native vision-language model is built on a hybrid architecture that integrates a linear attention mechanism with a sparse mixture-of-experts model, achieving higher inference efficiency. It delivers state-of-the-art performance comparable to leading-edge models across a wide range of tasks, including language understanding, logical reasoning, code generation, agent-based tasks, image understanding, video understanding, and graphical user interface (GUI) interactio...

### 133. MiniMax: MiniMax M2.5 (free)

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `minimax/minimax-m2.5` |
| **Categoria** | Código / Programação |
| **Contexto** | 196,608 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0 /M tokens |
| **Preço Output** | $0 /M tokens |
| **Gratuito** | Sim |
| **Raciocínio** | Suportado |

**Descrição:** MiniMax-M2.5 is a SOTA large language model designed for real-world productivity. Trained in a diverse range of complex real-world digital working environments, M2.5 builds upon the coding expertise of M2.1 to extend into general office work, reaching fluency in generating and operating Word, Excel, and Powerpoint files, context switching between diverse software environments, and working across different agent and human teams. Scoring 80.2% on SWE-Bench Verified, 51.3% on Multi-SWE-Bench, an...

### 134. MiniMax: MiniMax M2.5

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `minimax/minimax-m2.5` |
| **Categoria** | Código / Programação |
| **Contexto** | 196,608 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000015 /M tokens |
| **Preço Output** | $0.00000115 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** MiniMax-M2.5 is a SOTA large language model designed for real-world productivity. Trained in a diverse range of complex real-world digital working environments, M2.5 builds upon the coding expertise of M2.1 to extend into general office work, reaching fluency in generating and operating Word, Excel, and Powerpoint files, context switching between diverse software environments, and working across different agent and human teams. Scoring 80.2% on SWE-Bench Verified, 51.3% on Multi-SWE-Bench, an...

### 135. Z.ai: GLM 5

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `z-ai/glm-5` |
| **Categoria** | Código / Programação |
| **Contexto** | 202,752 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.0000006 /M tokens |
| **Preço Output** | $0.00000192 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** GLM-5 is Z.ai’s flagship open-source foundation model engineered for complex systems design and long-horizon agent workflows. Built for expert developers, it delivers production-grade performance on large-scale programming tasks, rivaling leading closed-source models. With advanced agentic planning, deep backend reasoning, and iterative self-correction, GLM-5 moves beyond code generation to full-system construction and autonomous execution.

### 136. Qwen: Qwen3 Max Thinking

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `qwen/qwen3-max-thinking` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 262,144 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000078 /M tokens |
| **Preço Output** | $0.0000039 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Qwen3-Max-Thinking is the flagship reasoning model in the Qwen3 series, designed for high-stakes cognitive tasks that require deep, multi-step reasoning. By significantly scaling model capacity and reinforcement learning compute, it delivers major gains in factual accuracy, complex reasoning, instruction following, alignment with human preferences, and agentic behavior.

### 137. Anthropic: Claude Opus 4.6

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `anthropic/claude-opus-4.6` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 1,000,000 tokens |
| **Entrada** | text, image, file |
| **Saída** | text |
| **Preço Input** | $5e-6 /M tokens |
| **Preço Output** | $25e-6 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Opus 4.6 is Anthropic’s strongest model for coding and long-running professional tasks. It is built for agents that operate across entire workflows rather than single prompts, making it especially effective for large codebases, complex refactors, and multi-step debugging that unfolds over time. The model shows deeper contextual understanding, stronger problem decomposition, and greater reliability on hard engineering tasks than prior generations.  Beyond coding, Opus 4.6 excels at sustained k...

### 138. Qwen: Qwen3 Coder Next

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `qwen/qwen3-coder-next` |
| **Categoria** | Código / Programação |
| **Contexto** | 262,144 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000011 /M tokens |
| **Preço Output** | $0.0000008 /M tokens |

**Descrição:** Qwen3-Coder-Next is an open-weight causal language model optimized for coding agents and local development workflows. It uses a sparse MoE design with 80B total parameters and only 3B activated per token, delivering performance comparable to models with 10 to 20x higher active compute, which makes it well suited for cost-sensitive, always-on agent deployment.  The model is trained with a strong agentic focus and performs reliably on long-horizon coding tasks, complex tool usage, and recovery ...

### 139. Sourceful: Riverflow V2 Pro

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `sourceful/riverflow-v2-pro` |
| **Categoria** | Geração de Imagem |
| **Contexto** | 8,192 tokens |
| **Entrada** | text, image |
| **Saída** | image |
| **Preço Input** | $0.15 per image |
| **Preço Output** | $0.03 per font |

**Descrição:** Riverflow V2 Pro is the most powerful variant of Sourceful's Riverflow 2.0 lineup, best for top-tier control and perfect text rendering.  The Riverflow 2.0 series represents SOTA performance on image generation and editing tasks, using an integrated reasoning model to boost reliability and tackle complex challenges.  Pricing is $0.15 per 1K/2K output image and $0.33 per 4K output image.  Additional features: - Custom font rendering via font_inputs ($0.03/font, max 2) - Image enhancement via s...

### 140. Sourceful: Riverflow V2 Fast

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `sourceful/riverflow-v2-fast` |
| **Categoria** | Geração de Imagem |
| **Contexto** | 8,192 tokens |
| **Entrada** | text, image |
| **Saída** | image |
| **Preço Input** | $0.02 per image |
| **Preço Output** | $0.03 per font |

**Descrição:** Riverflow V2 Fast is the fastest variant of Sourceful's Riverflow 2.0 lineup, best for production deployments and latency-critical workflows.  The Riverflow 2.0 series represents SOTA performance on image generation and editing tasks, using an integrated reasoning model to boost reliability and tackle complex challenges.  Pricing is $0.02 per 1K output image and $0.04 per 2K output image. Does not support 4K image output.  Additional features: - Custom font rendering via font_inputs ($0.03/fo...

### 141. StepFun: Step 3.5 Flash

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `stepfun/step-3.5-flash` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 262,144 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.0000001 /M tokens |
| **Preço Output** | $0.0000003 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Step 3.5 Flash is StepFun's most capable open-source foundation model. Built on a sparse Mixture of Experts (MoE) architecture, it selectively activates only 11B of its 196B parameters per token. It is a reasoning model that is incredibly speed efficient even at long contexts.

### 142. Arcee AI: Trinity Large Preview

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `arcee-ai/trinity-large-preview` |
| **Categoria** | Código / Programação |
| **Contexto** | 131,000 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000015 /M tokens |
| **Preço Output** | $0.00000045 /M tokens |

**Descrição:** Trinity-Large-Preview is a frontier-scale open-weight language model from Arcee, built as a 400B-parameter sparse Mixture-of-Experts with 13B active parameters per token using 4-of-256 expert routing.   It excels in creative writing, storytelling, role-play, chat scenarios, and real-time voice assistance, better than your average reasoning model usually can. But we’re also introducing some of our newer agentic performance. It was trained to navigate well in agent harnesses like OpenCode, Clin...

### 143. MoonshotAI: Kimi K2.5

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `moonshotai/kimi-k2.5` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 262,144 tokens |
| **Entrada** | text, image |
| **Saída** | text |
| **Preço Input** | $0.0000004 /M tokens |
| **Preço Output** | $0.0000019 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Kimi K2.5 is Moonshot AI's native multimodal model, delivering state-of-the-art visual coding capability and a self-directed agent swarm paradigm. Built on Kimi K2 with continued pretraining over approximately 15T mixed visual and text tokens, it delivers strong performance in general reasoning, visual coding, and agentic tool-calling.

### 144. Upstage: Solar Pro 3

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `upstage/solar-pro-3` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 128,000 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.15e-6 /M tokens |
| **Preço Output** | $0.6e-6 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Solar Pro 3 is Upstage's powerful Mixture-of-Experts (MoE) language model. With 102B total parameters and 12B active parameters per forward pass, it delivers exceptional performance while maintaining computational efficiency. Optimized for Korean with English and Japanese support.

### 145. MiniMax: MiniMax M2-her

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `minimax/minimax-m2-her` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 65,536 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.0000003 /M tokens |
| **Preço Output** | $0.0000012 /M tokens |

**Descrição:** MiniMax M2-her is a dialogue-first large language model built for immersive roleplay, character-driven chat, and expressive multi-turn conversations. Designed to stay consistent in tone and personality, it supports rich message roles (user_system, group, sample_message_user, sample_message_ai) and can learn from example dialogue to better match the style and pacing of your scenario, making it a strong choice for storytelling, companions, and conversational experiences where natural flow and v...

### 146. Writer: Palmyra X5

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `writer/palmyra-x5` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 1,040,000 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.0000006 /M tokens |
| **Preço Output** | $0.000006 /M tokens |

**Descrição:** Palmyra X5 is Writer's most advanced model, purpose-built for building and scaling AI agents across the enterprise. It delivers industry-leading speed and efficiency on context windows up to 1 million tokens, powered by a novel transformer architecture and hybrid attention mechanisms. This enables faster inference and expanded memory for processing large volumes of enterprise data, critical for scaling AI agents.

### 147. LiquidAI: LFM2.5-1.2B-Thinking (free)

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `liquid/lfm-2.5-1.2b-thinking` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 32,768 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0 /M tokens |
| **Preço Output** | $0 /M tokens |
| **Gratuito** | Sim |
| **Raciocínio** | Suportado |

**Descrição:** LFM2.5-1.2B-Thinking is a lightweight reasoning-focused model optimized for agentic tasks, data extraction, and RAG—while still running comfortably on edge devices. It supports long context (up to 32K tokens) and is designed to provide higher-quality “thinking” responses in a small 1.2B model.

### 148. LiquidAI: LFM2.5-1.2B-Instruct (free)

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `liquid/lfm-2.5-1.2b-instruct` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 32,768 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0 /M tokens |
| **Preço Output** | $0 /M tokens |
| **Gratuito** | Sim |

**Descrição:** LFM2.5-1.2B-Instruct is a compact, high-performance instruction-tuned model built for fast on-device AI. It delivers strong chat quality in a 1.2B parameter footprint, with efficient edge inference and broad runtime support.

### 149. OpenAI: GPT Audio

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/gpt-audio` |
| **Categoria** | Multimodal (Texto + Áudio) |
| **Contexto** | 128,000 tokens |
| **Entrada** | text, audio |
| **Saída** | text, audio |
| **Preço Input** | $0.0000025 /M tokens |
| **Preço Output** | $0.00001 /M tokens |

**Descrição:** The gpt-audio model is OpenAI's first generally available audio model. The new snapshot features an upgraded decoder for more natural sounding voices and maintains better voice consistency. Audio is priced at $32 per million input tokens and $64 per million output tokens.

### 150. OpenAI: GPT Audio Mini

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/gpt-audio-mini` |
| **Categoria** | Multimodal (Texto + Áudio) |
| **Contexto** | 128,000 tokens |
| **Entrada** | text, audio |
| **Saída** | text, audio |
| **Preço Input** | $6e-7 /M tokens |
| **Preço Output** | $0.0000024 /M tokens |

**Descrição:** A cost-efficient version of GPT Audio. The new snapshot features an upgraded decoder for more natural sounding voices and maintains better voice consistency. Input is priced at $0.60 per million tokens and output is priced at $2.40 per million tokens.

### 151. Z.ai: GLM 4.7 Flash

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `z-ai/glm-4.7-flash` |
| **Categoria** | Código / Programação |
| **Contexto** | 202,752 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000006 /M tokens |
| **Preço Output** | $0.0000004 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** As a 30B-class SOTA model, GLM-4.7-Flash offers a new option that balances performance and efficiency. It is further optimized for agentic coding use cases, strengthening coding capabilities, long-horizon task planning, and tool collaboration, and has achieved leading performance among open-source models of the same size on several current public benchmark leaderboards.

### 152. Black Forest Labs: FLUX.2 Klein 4B

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `black-forest-labs/flux.2-klein-4b` |
| **Categoria** | Geração de Imagem |
| **Contexto** | 40,960 tokens |
| **Entrada** | text, image |
| **Saída** | image |
| **Preço Input** | $0.014 per megapixel |
| **Preço Output** | Incluído |

**Descrição:** FLUX.2 [klein] 4B is the fastest and most cost-effective model in the FLUX.2 family, optimized for high-throughput use cases while maintaining excellent image quality.  Pricing is based on the output image. The first generated megapixel is charged $0.014. Each subsequent megapixel is charged $0.001.

### 153. OpenAI: GPT-5.2-Codex

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/gpt-5.2-codex` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 400,000 tokens |
| **Entrada** | text, image |
| **Saída** | text |
| **Preço Input** | $0.00000175 /M tokens |
| **Preço Output** | $0.000014 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** GPT-5.2-Codex is an upgraded version of GPT-5.1-Codex optimized for software engineering and coding workflows. It is designed for both interactive development sessions and long, independent execution of complex engineering tasks. The model supports building projects from scratch, feature development, debugging, large-scale refactoring, and code review. Compared to GPT-5.1-Codex, 5.2-Codex is more steerable, adheres closely to developer instructions, and produces cleaner, higher-quality code o...

### 154. ByteDance Seed: Seedream 4.5

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `bytedance-seed/seedream-4.5` |
| **Categoria** | Geração de Imagem |
| **Contexto** | 4,096 tokens |
| **Entrada** | image, text |
| **Saída** | image |
| **Preço Input** | $0.04 per image |
| **Preço Output** | Incluído |

**Descrição:** Seedream 4.5 is the latest in-house image generation model developed by ByteDance. Compared with Seedream 4.0, it delivers comprehensive improvements, especially in editing consistency, including better preservation of subject details, lighting, and color tone. It also enhances portrait refinement and small-text rendering. The model’s multi-image composition capabilities have been significantly strengthened, and both reasoning performance and visual aesthetics continue to advance, enabling mo...

### 155. ByteDance Seed: Seed 1.6 Flash

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `bytedance-seed/seed-1.6-flash` |
| **Categoria** | Multimodal (Vídeo) |
| **Contexto** | 262,144 tokens |
| **Entrada** | image, text, video |
| **Saída** | text |
| **Preço Input** | $0.000000075 /M tokens |
| **Preço Output** | $0.0000003 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Seed 1.6 Flash is an ultra-fast multimodal deep thinking model by ByteDance Seed, supporting both text and visual understanding. It features a 256k context window and can generate outputs of up to 16k tokens.

### 156. ByteDance Seed: Seed 1.6

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `bytedance-seed/seed-1.6` |
| **Categoria** | Multimodal (Vídeo) |
| **Contexto** | 262,144 tokens |
| **Entrada** | image, text, video |
| **Saída** | text |
| **Preço Input** | $0.00000025 /M tokens |
| **Preço Output** | $0.000002 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Seed 1.6 is a general-purpose model released by the ByteDance Seed team. It incorporates multimodal capabilities and adaptive deep thinking with a 256K context window.

### 157. MiniMax: MiniMax M2.1

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `minimax/minimax-m2.1` |
| **Categoria** | Código / Programação |
| **Contexto** | 196,608 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000029 /M tokens |
| **Preço Output** | $0.00000095 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** MiniMax-M2.1 is a lightweight, state-of-the-art large language model optimized for coding, agentic workflows, and modern application development. With only 10 billion activated parameters, it delivers a major jump in real-world capability while maintaining exceptional latency, scalability, and cost efficiency.  Compared to its predecessor, M2.1 delivers cleaner, more concise outputs and faster perceived response times. It shows leading multilingual coding performance across major systems and ...

### 158. Z.ai: GLM 4.7

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `z-ai/glm-4.7` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 202,752 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.0000004 /M tokens |
| **Preço Output** | $0.00000175 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** GLM-4.7 is Z.ai’s latest flagship model, featuring upgrades in two key areas: enhanced programming capabilities and more stable multi-step reasoning/execution. It demonstrates significant improvements in executing complex agent tasks while delivering more natural conversational experiences and superior front-end aesthetics.

### 159. Google: Gemini 3 Flash Preview

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `google/gemini-3-flash-preview` |
| **Categoria** | Multimodal (Vídeo) |
| **Contexto** | 1,048,576 tokens |
| **Entrada** | text, image, file, audio, video |
| **Saída** | text |
| **Preço Input** | $5e-7 /M tokens |
| **Preço Output** | $0.000003 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Gemini 3 Flash Preview is a high speed, high value thinking model designed for agentic workflows, multi turn chat, and coding assistance. It delivers near Pro level reasoning and tool use performance with substantially lower latency than larger Gemini variants, making it well suited for interactive development, long running agent loops, and collaborative coding tasks. Compared to Gemini 2.5 Flash, it provides broad quality improvements across reasoning, multimodal understanding, and reliabili...

### 160. Black Forest Labs: FLUX.2 Max

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `black-forest-labs/flux.2-max` |
| **Categoria** | Geração de Imagem |
| **Contexto** | 46,864 tokens |
| **Entrada** | text, image |
| **Saída** | image |
| **Preço Input** | $0.07 per megapixel |
| **Preço Output** | Incluído |

**Descrição:** FLUX.2 [max] is the new top-tier image model from Black Forest Labs, pushing image quality, prompt understanding, and editing consistency to the highest level yet.  Pricing is as follows, [per the docs](https://bfl.ai/pricing?category=flux.2): Input: We charge $0.03 for each megapixel on the input (i.e. reference images for editing) Output: The first generated megapixel is charged $0.07. Each subsequent megapixel is charged $0.03.

### 161. Xiaomi: MiMo-V2-Flash

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `xiaomi/mimo-v2-flash` |
| **Categoria** | Código / Programação |
| **Contexto** | 262,144 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.0000001 /M tokens |
| **Preço Output** | $0.0000003 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** MiMo-V2-Flash is an open-source foundation language model developed by Xiaomi. It is a Mixture-of-Experts model with 309B total parameters and 15B active parameters, adopting hybrid attention architecture. MiMo-V2-Flash supports a hybrid-thinking toggle and a 256K context window, and excels at reasoning, coding, and agent scenarios. On SWE-bench Verified and SWE-bench Multilingual, MiMo-V2-Flash ranks as the top #1 open-source model globally, delivering performance comparable to Claude Sonnet...

### 162. NVIDIA: Nemotron 3 Nano 30B A3B (free)

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `nvidia/nemotron-3-nano-30b-a3b` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 256,000 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0 /M tokens |
| **Preço Output** | $0 /M tokens |
| **Gratuito** | Sim |
| **Raciocínio** | Suportado |

**Descrição:** NVIDIA Nemotron 3 Nano 30B A3B is a small language MoE model with highest compute efficiency and accuracy for developers to build specialized agentic AI systems.  The model is fully open with open-weights, datasets and recipes so developers can easily customize, optimize, and deploy the model on their infrastructure for maximum privacy and security.

### 163. NVIDIA: Nemotron 3 Nano 30B A3B

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `nvidia/nemotron-3-nano-30b-a3b` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 262,144 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000005 /M tokens |
| **Preço Output** | $0.0000002 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** NVIDIA Nemotron 3 Nano 30B A3B is a small language MoE model with highest compute efficiency and accuracy for developers to build specialized agentic AI systems.  The model is fully open with open-weights, datasets and recipes so developers can easily customize, optimize, and deploy the model on their infrastructure for maximum privacy and security.

### 164. OpenAI: GPT-5.2 Chat

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/gpt-5.2-chat` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 128,000 tokens |
| **Entrada** | file, image, text |
| **Saída** | text |
| **Preço Input** | $0.00000175 /M tokens |
| **Preço Output** | $0.000014 /M tokens |

**Descrição:** GPT-5.2 Chat (AKA Instant) is the fast, lightweight member of the 5.2 family, optimized for low-latency chat while retaining strong general intelligence. It uses adaptive reasoning to selectively “think” on harder queries, improving accuracy on math, coding, and multi-step tasks without slowing down typical conversations. The model is warmer and more conversational by default, with better instruction following and more stable short-form reasoning. GPT-5.2 Chat is designed for high-throughput,...

### 165. OpenAI: GPT-5.2 Pro

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/gpt-5.2-pro` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 400,000 tokens |
| **Entrada** | image, text, file |
| **Saída** | text |
| **Preço Input** | $0.000021 /M tokens |
| **Preço Output** | $0.000168 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** GPT-5.2 Pro is OpenAI’s most advanced model, offering major improvements in agentic coding and long context performance over GPT-5 Pro. It is optimized for complex tasks that require step-by-step reasoning, instruction following, and accuracy in high-stakes use cases. It supports test-time routing features and advanced prompt understanding, including user-specified intent like "think hard about this." Improvements include reductions in hallucination, sycophancy, and better performance in codi...

### 166. OpenAI: GPT-5.2

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/gpt-5.2` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 400,000 tokens |
| **Entrada** | file, image, text |
| **Saída** | text |
| **Preço Input** | $0.00000175 /M tokens |
| **Preço Output** | $0.000014 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** GPT-5.2 is the latest frontier-grade model in the GPT-5 series, offering stronger agentic and long context perfomance compared to GPT-5.1. It uses adaptive reasoning to allocate computation dynamically, responding quickly to simple queries while spending more depth on complex tasks.  Built for broad task coverage, GPT-5.2 delivers consistent gains across math, coding, sciende, and tool calling workloads, with more coherent long-form answers and improved tool-use reliability.

### 167. Mistral: Devstral 2 2512

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `mistralai/devstral-2512` |
| **Categoria** | Multimodal (Arquivo + Texto) |
| **Contexto** | 262,144 tokens |
| **Entrada** | text, file |
| **Saída** | text |
| **Preço Input** | $0.0000004 /M tokens |
| **Preço Output** | $0.000002 /M tokens |

**Descrição:** Devstral 2 is a state-of-the-art open-source model by Mistral AI specializing in agentic coding. It is a 123B-parameter dense transformer model supporting a 256K context window.  Devstral 2 supports exploring codebases and orchestrating changes across multiple files while maintaining architecture-level context. It tracks framework dependencies, detects failures, and retries with corrections—solving challenges like bug fixing and modernizing legacy systems. The model can be fine-tuned to prior...

### 168. Sourceful: Riverflow V2 Max Preview

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `sourceful/riverflow-v2-max-preview` |
| **Categoria** | Geração de Imagem |
| **Contexto** | 8,192 tokens |
| **Entrada** | text, image |
| **Saída** | image |
| **Preço Input** | $0.075 per image |
| **Preço Output** | Incluído |

**Descrição:** Riverflow V2 Max Preview is the most powerful variant of Sourceful's Riverflow V2 preview lineup. This preview version exceeds the performance of Riverflow 1 Family and is Sourceful's first unified text-to-image and image-to-image model family.  Pricing is $0.075 per output image, regardless of size.  Sourceful imposes a 4.5MB request size limit, therefore it is highly recommended to pass image URLs instead of Base64 data.

### 169. Sourceful: Riverflow V2 Standard Preview

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `sourceful/riverflow-v2-standard-preview` |
| **Categoria** | Geração de Imagem |
| **Contexto** | 8,192 tokens |
| **Entrada** | text, image |
| **Saída** | image |
| **Preço Input** | $0.035 per image |
| **Preço Output** | Incluído |

**Descrição:** Riverflow V2 Standard Preview is the standard variant of Sourceful's Riverflow V2 preview lineup. This preview version exceeds the performance of Riverflow 1 Family and is Sourceful's first unified text-to-image and image-to-image model family.  Pricing is $0.035 per output image, regardless of size.  Sourceful imposes a 4.5MB request size limit, therefore it is highly recommended to pass image URLs instead of Base64 data.

### 170. Sourceful: Riverflow V2 Fast Preview

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `sourceful/riverflow-v2-fast-preview` |
| **Categoria** | Geração de Imagem |
| **Contexto** | 8,192 tokens |
| **Entrada** | text, image |
| **Saída** | image |
| **Preço Input** | $0.03 per image |
| **Preço Output** | Incluído |

**Descrição:** Riverflow V2 Fast Preview is the fastest variant of Sourceful's Riverflow V2 preview lineup. This preview version exceeds the performance of Riverflow 1 Family and is Sourceful's first unified text-to-image and image-to-image model family.  Pricing is $0.03 per output image, regardless of size.  Sourceful imposes a 4.5MB request size limit, therefore it is highly recommended to pass image URLs instead of Base64 data.

### 171. Relace: Relace Search

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `relace/relace-search` |
| **Categoria** | Código / Programação |
| **Contexto** | 256,000 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.000001 /M tokens |
| **Preço Output** | $0.000003 /M tokens |

**Descrição:** The relace-search model uses 4-12 `view_file` and `grep` tools in parallel to explore a codebase and return relevant files to the user request.   In contrast to RAG, relace-search performs agentic multi-step reasoning to produce highly precise results 4x faster than any frontier model. It's designed to serve as a subagent that passes its findings to an "oracle" coding agent, who orchestrates/performs the rest of the coding task.  To use relace-search you need to build an appropriate agent har...

### 172. Z.ai: GLM 4.6V

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `z-ai/glm-4.6v` |
| **Categoria** | Multimodal (Vídeo) |
| **Contexto** | 131,072 tokens |
| **Entrada** | image, text, video |
| **Saída** | text |
| **Preço Input** | $3e-7 /M tokens |
| **Preço Output** | $9e-7 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** GLM-4.6V is a large multimodal model designed for high-fidelity visual understanding and long-context reasoning across images, documents, and mixed media. It supports up to 128K tokens, processes complex page layouts and charts directly as visual inputs, and integrates native multimodal function calling to connect perception with downstream tool execution. The model also enables interleaved image-text generation and UI reconstruction workflows, including screenshot-to-HTML synthesis and itera...

### 173. Nex AGI: DeepSeek V3.1 Nex N1

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `nex-agi/deepseek-v3.1-nex-n1` |
| **Categoria** | Código / Programação |
| **Contexto** | 131,072 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.000000135 /M tokens |
| **Preço Output** | $0.0000005 /M tokens |

**Descrição:** DeepSeek V3.1 Nex-N1 is the flagship release of the Nex-N1 series — a post-trained model designed to highlight agent autonomy, tool use, and real-world productivity.   Nex-N1 demonstrates competitive performance across all evaluation scenarios, showing particularly strong results in practical coding and HTML generation tasks.

### 174. EssentialAI: Rnj 1 Instruct

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `essentialai/rnj-1-instruct` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 32,768 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000015 /M tokens |
| **Preço Output** | $0.00000015 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Rnj-1 is an 8B-parameter, dense, open-weight model family developed by Essential AI and trained from scratch with a focus on programming, math, and scientific reasoning. The model demonstrates strong performance across multiple programming languages, tool-use workflows, and agentic execution environments (e.g., mini-SWE-agent).

### 175. OpenAI: GPT-5.1-Codex-Max

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/gpt-5.1-codex-max` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 400,000 tokens |
| **Entrada** | text, image |
| **Saída** | text |
| **Preço Input** | $0.00000125 /M tokens |
| **Preço Output** | $0.00001 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** GPT-5.1-Codex-Max is OpenAI’s latest agentic coding model, designed for long-running, high-context software development tasks. It is based on an updated version of the 5.1 reasoning stack and trained on agentic workflows spanning software engineering, mathematics, and research.  GPT-5.1-Codex-Max delivers faster performance, improved reasoning, and higher token efficiency across the development lifecycle.

### 176. Amazon: Nova 2 Lite

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `amazon/nova-2-lite-v1` |
| **Categoria** | Multimodal (Vídeo) |
| **Contexto** | 1,000,000 tokens |
| **Entrada** | text, image, video, file |
| **Saída** | text |
| **Preço Input** | $0.0000003 /M tokens |
| **Preço Output** | $0.0000025 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Nova 2 Lite is a fast, cost-effective reasoning model for everyday workloads that can process text, images, and videos to generate text.   Nova 2 Lite demonstrates standout capabilities in processing documents, extracting information from videos, generating code, providing accurate grounded answers, and automating multi-step agentic workflows.

### 177. Mistral: Ministral 3 14B 2512

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `mistralai/ministral-14b-2512` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 262,144 tokens |
| **Entrada** | text, image |
| **Saída** | text |
| **Preço Input** | $2e-7 /M tokens |
| **Preço Output** | $2e-7 /M tokens |

**Descrição:** The largest model in the Ministral 3 family, Ministral 3 14B offers frontier capabilities and performance comparable to its larger Mistral Small 3.2 24B counterpart. A powerful and efficient language model with vision capabilities.

### 178. Mistral: Ministral 3 8B 2512

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `mistralai/ministral-8b-2512` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 262,144 tokens |
| **Entrada** | text, image |
| **Saída** | text |
| **Preço Input** | $1.5e-7 /M tokens |
| **Preço Output** | $1.5e-7 /M tokens |

**Descrição:** A balanced model in the Ministral 3 family, Ministral 3 8B is a powerful, efficient tiny language model with vision capabilities.

### 179. Mistral: Ministral 3 3B 2512

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `mistralai/ministral-3b-2512` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 131,072 tokens |
| **Entrada** | text, image |
| **Saída** | text |
| **Preço Input** | $1e-7 /M tokens |
| **Preço Output** | $1e-7 /M tokens |

**Descrição:** The smallest model in the Ministral 3 family, Ministral 3 3B is a powerful, efficient tiny language model with vision capabilities.

### 180. Mistral: Mistral Large 3 2512

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `mistralai/mistral-large-2512` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 262,144 tokens |
| **Entrada** | text, image, file |
| **Saída** | text |
| **Preço Input** | $5e-7 /M tokens |
| **Preço Output** | $0.0000015 /M tokens |

**Descrição:** Mistral Large 3 2512 is Mistral’s most capable model to date, featuring a sparse mixture-of-experts architecture with 41B active parameters (675B total), and released under the Apache 2.0 license.

### 181. Arcee AI: Trinity Mini

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `arcee-ai/trinity-mini` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 131,072 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.000000045 /M tokens |
| **Preço Output** | $0.00000015 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Trinity Mini is a 26B-parameter (3B active) sparse mixture-of-experts language model featuring 128 experts with 8 active per token. Engineered for efficient reasoning over long contexts (131k) with robust function calling and multi-step agent workflows.

### 182. DeepSeek: DeepSeek V3.2 Speciale

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `deepseek/deepseek-v3.2-speciale` |
| **Categoria** | Código / Programação |
| **Contexto** | 163,840 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.000000287 /M tokens |
| **Preço Output** | $0.000000431 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** DeepSeek-V3.2-Speciale is a high-compute variant of DeepSeek-V3.2 optimized for maximum reasoning and agentic performance. It builds on DeepSeek Sparse Attention (DSA) for efficient long-context processing, then scales post-training reinforcement learning to push capability beyond the base model. Reported evaluations place Speciale ahead of GPT-5 on difficult reasoning workloads, with proficiency comparable to Gemini-3.0-Pro, while retaining strong coding and tool-use reliability. Like V3.2, ...

### 183. DeepSeek: DeepSeek V3.2

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `deepseek/deepseek-v3.2` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 131,072 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $2.52e-7 /M tokens |
| **Preço Output** | $3.78e-7 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** DeepSeek-V3.2 is a large language model designed to harmonize high computational efficiency with strong reasoning and agentic tool-use performance. It introduces DeepSeek Sparse Attention (DSA), a fine-grained sparse attention mechanism that reduces training and inference cost while preserving quality in long-context scenarios. A scalable reinforcement learning post-training framework further improves reasoning, with reported performance in the GPT-5 class, and the model has demonstrated gold...

### 184. Prime Intellect: INTELLECT-3

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `prime-intellect/intellect-3` |
| **Categoria** | Código / Programação |
| **Contexto** | 131,072 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.0000002 /M tokens |
| **Preço Output** | $0.0000011 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** INTELLECT-3 is a 106B-parameter Mixture-of-Experts model (12B active) post-trained from GLM-4.5-Air-Base using supervised fine-tuning (SFT) followed by large-scale reinforcement learning (RL). It offers state-of-the-art performance for its size across math, code, science, and general reasoning, consistently outperforming many larger frontier models. Designed for strong multi-step problem solving, it maintains high accuracy on structured tasks while remaining efficient at inference thanks to i...

### 185. Black Forest Labs: FLUX.2 Flex

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `black-forest-labs/flux.2-flex` |
| **Categoria** | Geração de Imagem |
| **Contexto** | 67,344 tokens |
| **Entrada** | text, image |
| **Saída** | image |
| **Preço Input** | $0.06 per megapixel |
| **Preço Output** | $0.06 per megapixel |

**Descrição:** FLUX.2 [flex] excels at rendering complex text, typography, and fine details, and supports multi-reference editing in the same unified architecture.  Pricing is as follows, [per the docs](https://bfl.ai/pricing?category=flux.2): We charge $0.06 for each megapixel on both input and output side.

### 186. Black Forest Labs: FLUX.2 Pro

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `black-forest-labs/flux.2-pro` |
| **Categoria** | Geração de Imagem |
| **Contexto** | 46,864 tokens |
| **Entrada** | text, image |
| **Saída** | image |
| **Preço Input** | $0.03 per megapixel |
| **Preço Output** | Incluído |

**Descrição:** A high-end image generation and editing model focused on frontier-level visual quality and reliability. It delivers strong prompt adherence, stable lighting, sharp textures, and consistent character/style reproduction across multi-reference inputs. Designed for production workloads, it balances speed and quality while supporting text-to-image and image editing up to 4 MP resolution.  Pricing is as follows, [per the docs](https://bfl.ai/pricing?category=flux.2): Input: We charge $0.015 for eac...

### 187. Anthropic: Claude Opus 4.5

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `anthropic/claude-opus-4.5` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 200,000 tokens |
| **Entrada** | file, image, text |
| **Saída** | text |
| **Preço Input** | $0.000005 /M tokens |
| **Preço Output** | $0.000025 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Claude Opus 4.5 is Anthropic’s frontier reasoning model optimized for complex software engineering, agentic workflows, and long-horizon computer use. It offers strong multimodal capabilities, competitive performance across real-world coding and reasoning benchmarks, and improved robustness to prompt injection. The model is designed to operate efficiently across varied effort levels, enabling developers to trade off speed, depth, and token usage depending on task requirements. It comes with a ...

### 188. AllenAI: Olmo 3 32B Think

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `allenai/olmo-3-32b-think` |
| **Categoria** | Código / Programação |
| **Contexto** | 65,536 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000015 /M tokens |
| **Preço Output** | $0.0000005 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Olmo 3 32B Think is a large-scale, 32-billion-parameter model purpose-built for deep reasoning, complex logic chains and advanced instruction-following scenarios. Its capacity enables strong performance on demanding evaluation tasks and highly nuanced conversational reasoning. Developed by Ai2 under the Apache 2.0 license, Olmo 3 32B Think embodies the Olmo initiative’s commitment to openness, offering full transparency across weights, code and training methodology.

### 189. Google: Nano Banana Pro (Gemini 3 Pro Image Preview)

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `google/gemini-3-pro-image-preview` |
| **Categoria** | Multimodal (Texto + Imagem) |
| **Contexto** | 65,536 tokens |
| **Entrada** | image, text |
| **Saída** | image, text |
| **Preço Input** | $0.000002 /M tokens |
| **Preço Output** | $0.000012 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Nano Banana Pro is Google’s most advanced image-generation and editing model, built on Gemini 3 Pro. It extends the original Nano Banana with significantly improved multimodal reasoning, real-world grounding, and high-fidelity visual synthesis. The model generates context-rich graphics, from infographics and diagrams to cinematic composites, and can incorporate real-time information via Search grounding.  It offers industry-leading text rendering in images (including long passages and multili...

### 190. xAI: Grok 4.1 Fast

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `x-ai/grok-4.1-fast` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 2,000,000 tokens |
| **Entrada** | text, image, file |
| **Saída** | text |
| **Preço Input** | $0.0000002 /M tokens |
| **Preço Output** | $0.0000005 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Grok 4.1 Fast is xAI's best agentic tool calling model that shines in real-world use cases like customer support and deep research. 2M context window.  Reasoning can be enabled/disabled using the `reasoning` `enabled` parameter in the API. [Learn more in our docs](https://openrouter.ai/docs/use-cases/reasoning-tokens#controlling-reasoning-tokens)

### 191. Thenlper: GTE-Base

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `thenlper/gte-base` |
| **Categoria** | Código / Programação |
| **Contexto** | 512 tokens |
| **Entrada** | text |
| **Saída** | embeddings |
| **Preço Input** | $5e-9 /M tokens |
| **Preço Output** | Incluído |

**Descrição:** The gte-base embedding model encodes English sentences and paragraphs into a 768-dimensional dense vector space, delivering efficient and effective semantic embeddings optimized for textual similarity, semantic search, and clustering applications.

### 192. Thenlper: GTE-Large

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `thenlper/gte-large` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 512 tokens |
| **Entrada** | text |
| **Saída** | embeddings |
| **Preço Input** | $1e-8 /M tokens |
| **Preço Output** | Incluído |

**Descrição:** The gte-large embedding model converts English sentences, paragraphs and moderate-length documents into a 1024-dimensional dense vector space, delivering high-quality semantic embeddings optimized for information retrieval, semantic textual similarity, reranking and clustering tasks. Trained via multi-stage contrastive learning on a large domain-diverse relevance corpus, it offers excellent performance across general-purpose embedding use-cases.

### 193. Intfloat: E5-Large-v2

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `intfloat/e5-large-v2` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 512 tokens |
| **Entrada** | text |
| **Saída** | embeddings |
| **Preço Input** | $1e-8 /M tokens |
| **Preço Output** | Incluído |

**Descrição:** The e5-large-v2 embedding model maps English sentences, paragraphs, and documents into a 1024-dimensional dense vector space, delivering high-accuracy semantic embeddings optimized for retrieval, semantic search, reranking, and similarity-scoring tasks.

### 194. Intfloat: E5-Base-v2

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `intfloat/e5-base-v2` |
| **Categoria** | Código / Programação |
| **Contexto** | 512 tokens |
| **Entrada** | text |
| **Saída** | embeddings |
| **Preço Input** | $5e-9 /M tokens |
| **Preço Output** | Incluído |

**Descrição:** The e5-base-v2 embedding model encodes English sentences and paragraphs into a 768-dimensional dense vector space, producing efficient and high-quality semantic embeddings optimized for tasks such as semantic search, similarity scoring, retrieval and clustering.

### 195. Intfloat: Multilingual-E5-Large

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `intfloat/multilingual-e5-large` |
| **Categoria** | Código / Programação |
| **Contexto** | 512 tokens |
| **Entrada** | text |
| **Saída** | embeddings |
| **Preço Input** | $1e-8 /M tokens |
| **Preço Output** | Incluído |

**Descrição:** The multilingual-e5-large embedding model encodes sentences, paragraphs, and documents across over 90 languages into a 1024-dimensional dense vector space, delivering robust semantic embeddings optimized for multilingual retrieval, cross-language similarity, and large-scale data search.

### 196. Sentence Transformers: paraphrase-MiniLM-L6-v2

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `sentence-transformers/paraphrase-minilm-l6-v2` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 512 tokens |
| **Entrada** | text |
| **Saída** | embeddings |
| **Preço Input** | $5e-9 /M tokens |
| **Preço Output** | Incluído |

**Descrição:** The paraphrase-MiniLM-L6-v2 embedding model converts sentences and short paragraphs into a 384-dimensional dense vector space, producing high-quality semantic embeddings optimized for paraphrase detection, semantic similarity scoring, clustering, and lightweight retrieval tasks.

### 197. Sentence Transformers: all-MiniLM-L12-v2

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `sentence-transformers/all-minilm-l12-v2` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 512 tokens |
| **Entrada** | text |
| **Saída** | embeddings |
| **Preço Input** | $5e-9 /M tokens |
| **Preço Output** | Incluído |

**Descrição:** The all-MiniLM-L12-v2 embedding model maps sentences and short paragraphs into a 384-dimensional dense vector space, producing efficient and high-quality semantic embeddings optimized for tasks such as semantic search, clustering, and similarity-scoring.

### 198. BAAI: bge-base-en-v1.5

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `baai/bge-base-en-v1.5` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 512 tokens |
| **Entrada** | text |
| **Saída** | embeddings |
| **Preço Input** | $5e-9 /M tokens |
| **Preço Output** | Incluído |

**Descrição:** The bge-base-en-v1.5 embedding model converts English sentences and paragraphs into 768-dimensional dense vectors, delivering efficient, high-quality semantic embeddings optimized for retrieval, semantic search, and document-matching workflows. This version (v1.5) features improved similarity-score distribution and stronger retrieval performance out of the box.

### 199. Sentence Transformers: multi-qa-mpnet-base-dot-v1

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `sentence-transformers/multi-qa-mpnet-base-dot-v1` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 512 tokens |
| **Entrada** | text |
| **Saída** | embeddings |
| **Preço Input** | $5e-9 /M tokens |
| **Preço Output** | Incluído |

**Descrição:** The multi-qa-mpnet-base-dot-v1 embedding model transforms sentences and short paragraphs into a 768-dimensional dense vector space, generating high-quality semantic embeddings optimized for question-and-answer retrieval, semantic search, and similarity-scoring across diverse content.

### 200. BAAI: bge-large-en-v1.5

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `baai/bge-large-en-v1.5` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 512 tokens |
| **Entrada** | text |
| **Saída** | embeddings |
| **Preço Input** | $1e-8 /M tokens |
| **Preço Output** | Incluído |

**Descrição:** The bge-large-en-v1.5 embedding model maps English sentences, paragraphs, and documents into a 1024-dimensional dense vector space, delivering high-fidelity semantic embeddings optimized for semantic search, document retrieval, and downstream NLP tasks in English.

### 201. BAAI: bge-m3

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `baai/bge-m3` |
| **Categoria** | Código / Programação |
| **Contexto** | 8,192 tokens |
| **Entrada** | text |
| **Saída** | embeddings |
| **Preço Input** | $1e-8 /M tokens |
| **Preço Output** | Incluído |

**Descrição:** The bge-m3 embedding model encodes sentences, paragraphs, and long documents into a 1024-dimensional dense vector space, delivering high-quality semantic embeddings optimized for multilingual retrieval, semantic search, and large-context applications.

### 202. Sentence Transformers: all-mpnet-base-v2

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `sentence-transformers/all-mpnet-base-v2` |
| **Categoria** | Código / Programação |
| **Contexto** | 512 tokens |
| **Entrada** | text |
| **Saída** | embeddings |
| **Preço Input** | $5e-9 /M tokens |
| **Preço Output** | Incluído |

**Descrição:** The all-mpnet-base-v2 embedding model encodes sentences and short paragraphs into a 768-dimensional dense vector space, providing high-fidelity semantic embeddings well suited for tasks like information retrieval, clustering, similarity scoring, and text ranking.

### 203. Sentence Transformers: all-MiniLM-L6-v2

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `sentence-transformers/all-minilm-l6-v2` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 512 tokens |
| **Entrada** | text |
| **Saída** | embeddings |
| **Preço Input** | $5e-9 /M tokens |
| **Preço Output** | Incluído |

**Descrição:** The all-MiniLM-L6-v2 embedding model maps sentences and short paragraphs into a 384-dimensional dense vector space, enabling high-quality semantic representations that are ideal for downstream tasks such as information retrieval, clustering, similarity scoring, and text ranking.

### 204. Deep Cogito: Cogito v2.1 671B

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `deepcogito/cogito-v2.1-671b` |
| **Categoria** | Código / Programação |
| **Contexto** | 128,000 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000125 /M tokens |
| **Preço Output** | $0.00000125 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Cogito v2.1 671B MoE represents one of the strongest open models globally, matching performance of frontier closed and open models. This model is trained using self play with reinforcement learning to reach state-of-the-art performance on multiple categories (instruction following, coding, longer queries and creative writing). This advanced system demonstrates significant progress toward scalable superintelligence through policy improvement.

### 205. OpenAI: GPT-5.1

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/gpt-5.1` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 400,000 tokens |
| **Entrada** | image, text, file |
| **Saída** | text |
| **Preço Input** | $0.00000125 /M tokens |
| **Preço Output** | $0.00001 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** GPT-5.1 is the latest frontier-grade model in the GPT-5 series, offering stronger general-purpose reasoning, improved instruction adherence, and a more natural conversational style compared to GPT-5. It uses adaptive reasoning to allocate computation dynamically, responding quickly to simple queries while spending more depth on complex tasks. The model produces clearer, more grounded explanations with reduced jargon, making it easier to follow even on technical or multi-step problems.  Built ...

### 206. OpenAI: GPT-5.1 Chat

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/gpt-5.1-chat` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 128,000 tokens |
| **Entrada** | file, image, text |
| **Saída** | text |
| **Preço Input** | $0.00000125 /M tokens |
| **Preço Output** | $0.00001 /M tokens |

**Descrição:** GPT-5.1 Chat (AKA Instant is the fast, lightweight member of the 5.1 family, optimized for low-latency chat while retaining strong general intelligence. It uses adaptive reasoning to selectively “think” on harder queries, improving accuracy on math, coding, and multi-step tasks without slowing down typical conversations. The model is warmer and more conversational by default, with better instruction following and more stable short-form reasoning. GPT-5.1 Chat is designed for high-throughput, ...

### 207. OpenAI: GPT-5.1-Codex

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/gpt-5.1-codex` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 400,000 tokens |
| **Entrada** | text, image |
| **Saída** | text |
| **Preço Input** | $0.00000125 /M tokens |
| **Preço Output** | $0.00001 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** GPT-5.1-Codex is a specialized version of GPT-5.1 optimized for software engineering and coding workflows. It is designed for both interactive development sessions and long, independent execution of complex engineering tasks. The model supports building projects from scratch, feature development, debugging, large-scale refactoring, and code review. Compared to GPT-5.1, Codex is more steerable, adheres closely to developer instructions, and produces cleaner, higher-quality code outputs. Reason...

### 208. OpenAI: GPT-5.1-Codex-Mini

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/gpt-5.1-codex-mini` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 400,000 tokens |
| **Entrada** | image, text |
| **Saída** | text |
| **Preço Input** | $2.5e-7 /M tokens |
| **Preço Output** | $0.000002 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** GPT-5.1-Codex-Mini is a smaller and faster version of GPT-5.1-Codex

### 209. MoonshotAI: Kimi K2 Thinking

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `moonshotai/kimi-k2-thinking` |
| **Categoria** | Código / Programação |
| **Contexto** | 262,144 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.0000006 /M tokens |
| **Preço Output** | $0.0000025 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Kimi K2 Thinking is Moonshot AI’s most advanced open reasoning model to date, extending the K2 series into agentic, long-horizon reasoning. Built on the trillion-parameter Mixture-of-Experts (MoE) architecture introduced in Kimi K2, it activates 32 billion parameters per forward pass and supports 256 k-token context windows. The model is optimized for persistent step-by-step thought, dynamic tool invocation, and complex reasoning workflows that span hundreds of turns. It interleaves step-by-s...

### 210. Amazon: Nova Premier 1.0

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `amazon/nova-premier-v1` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 1,000,000 tokens |
| **Entrada** | text, image |
| **Saída** | text |
| **Preço Input** | $0.0000025 /M tokens |
| **Preço Output** | $0.0000125 /M tokens |

**Descrição:** Amazon Nova Premier is the most capable of Amazon’s multimodal models for complex reasoning tasks and for use as the best teacher for distilling custom models.

### 211. Mistral: Mistral Embed 2312

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `mistralai/mistral-embed-2312` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 8,192 tokens |
| **Entrada** | text |
| **Saída** | embeddings |
| **Preço Input** | $1e-7 /M tokens |
| **Preço Output** | Incluído |

**Descrição:** Mistral Embed is a specialized embedding model for text data, optimized for semantic search and RAG applications. Developed by Mistral AI in late 2023, it produces 1024-dimensional vectors that effectively capture semantic relationships in text.

### 212. Google: Gemini Embedding 001

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `google/gemini-embedding-001` |
| **Categoria** | Código / Programação |
| **Contexto** | 20,000 tokens |
| **Entrada** | text |
| **Saída** | embeddings |
| **Preço Input** | $1.5e-7 /M tokens |
| **Preço Output** | Incluído |

**Descrição:** gemini-embedding-001 provides a unified cutting edge experience across domains, including science, legal, finance, and coding. This embedding model has consistently held a top spot on the Massive Text Embedding Benchmark (MTEB) Multilingual leaderboard since the experimental launch in March.

### 213. OpenAI: Text Embedding Ada 002

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/text-embedding-ada-002` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 8,192 tokens |
| **Entrada** | text |
| **Saída** | embeddings |
| **Preço Input** | $1e-7 /M tokens |
| **Preço Output** | Incluído |

**Descrição:** text-embedding-ada-002 is OpenAI's legacy text embedding model.

### 214. Mistral: Codestral Embed 2505

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `mistralai/codestral-embed-2505` |
| **Categoria** | Código / Programação |
| **Contexto** | 8,192 tokens |
| **Entrada** | text |
| **Saída** | embeddings |
| **Preço Input** | $1.5e-7 /M tokens |
| **Preço Output** | Incluído |

**Descrição:** Mistral Codestral Embed is specially designed for code, perfect for embedding code databases, repositories, and powering coding assistants with state-of-the-art retrieval.

### 215. OpenAI: Text Embedding 3 Large

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/text-embedding-3-large` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 8,192 tokens |
| **Entrada** | text |
| **Saída** | embeddings |
| **Preço Input** | $1.3e-7 /M tokens |
| **Preço Output** | Incluído |

**Descrição:** text-embedding-3-large is OpenAI's most capable embedding model for both english and non-english tasks. Embeddings are a numerical representation of text that can be used to measure the relatedness between two pieces of text. Embeddings are useful for search, clustering, recommendations, anomaly detection, and classification tasks.

### 216. OpenAI: Text Embedding 3 Small

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/text-embedding-3-small` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 8,192 tokens |
| **Entrada** | text |
| **Saída** | embeddings |
| **Preço Input** | $2e-8 /M tokens |
| **Preço Output** | Incluído |

**Descrição:** text-embedding-3-small is OpenAI's improved, more performant version of the ada embedding model. Embeddings are a numerical representation of text that can be used to measure the relatedness between two pieces of text. Embeddings are useful for search, clustering, recommendations, anomaly detection, and classification tasks.

### 217. Perplexity: Sonar Pro Search

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `perplexity/sonar-pro-search` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 200,000 tokens |
| **Entrada** | text, image |
| **Saída** | text |
| **Preço Input** | $0.000003 /M tokens |
| **Preço Output** | $0.000015 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Exclusively available on the OpenRouter API, Sonar Pro's new Pro Search mode is Perplexity's most advanced agentic search system. It is designed for deeper reasoning and analysis. Pricing is based on tokens plus $18 per thousand requests. This model powers the Pro Search mode on the Perplexity platform.  Sonar Pro Search adds autonomous, multi-step reasoning to Sonar Pro. So, instead of just one query + synthesis, it plans and executes entire research workflows using tools.

### 218. Mistral: Voxtral Small 24B 2507

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `mistralai/voxtral-small-24b-2507` |
| **Categoria** | Multimodal (Áudio + Texto) |
| **Contexto** | 32,000 tokens |
| **Entrada** | text, audio, file |
| **Saída** | text |
| **Preço Input** | $1e-7 /M tokens |
| **Preço Output** | $3e-7 /M tokens |

**Descrição:** Voxtral Small is an enhancement of Mistral Small 3, incorporating state-of-the-art audio input capabilities while retaining best-in-class text performance. It excels at speech transcription, translation and audio understanding. Input audio is priced at $100 per million seconds.

### 219. OpenAI: gpt-oss-safeguard-20b

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/gpt-oss-safeguard-20b` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 131,072 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.000000075 /M tokens |
| **Preço Output** | $0.0000003 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** gpt-oss-safeguard-20b is a safety reasoning model from OpenAI built upon gpt-oss-20b. This open-weight, 21B-parameter Mixture-of-Experts (MoE) model offers lower latency for safety tasks like content classification, LLM filtering, and trust & safety labeling.  Learn more about this model in OpenAI's gpt-oss-safeguard [user guide](https://cookbook.openai.com/articles/gpt-oss-safeguard-guide).

### 220. Qwen: Qwen3 Embedding 8B

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `qwen/qwen3-embedding-8b` |
| **Categoria** | Código / Programação |
| **Contexto** | 32,000 tokens |
| **Entrada** | text |
| **Saída** | embeddings |
| **Preço Input** | $1e-8 /M tokens |
| **Preço Output** | Incluído |

**Descrição:** The Qwen3 Embedding model series is the latest proprietary model of the Qwen family, specifically designed for text embedding and ranking tasks. This series inherits the exceptional multilingual capabilities, long-text understanding, and reasoning skills of its foundational model. The Qwen3 Embedding series represents significant advancements in multiple text embedding and ranking tasks, including text retrieval, code retrieval, text classification, text clustering, and bitext mining.

### 221. NVIDIA: Nemotron Nano 12B 2 VL (free)

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `nvidia/nemotron-nano-12b-v2-vl` |
| **Categoria** | Multimodal (Vídeo) |
| **Contexto** | 128,000 tokens |
| **Entrada** | image, text, video |
| **Saída** | text |
| **Preço Input** | $0 /M tokens |
| **Preço Output** | $0 /M tokens |
| **Gratuito** | Sim |
| **Raciocínio** | Suportado |

**Descrição:** NVIDIA Nemotron Nano 2 VL is a 12-billion-parameter open multimodal reasoning model designed for video understanding and document intelligence. It introduces a hybrid Transformer-Mamba architecture, combining transformer-level accuracy with Mamba’s memory-efficient sequence modeling for significantly higher throughput and lower latency.  The model supports inputs of text and multi-image documents, producing natural-language outputs. It is trained on high-quality NVIDIA-curated synthetic datas...

### 222. Qwen: Qwen3 Embedding 4B

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `qwen/qwen3-embedding-4b` |
| **Categoria** | Código / Programação |
| **Contexto** | 32,768 tokens |
| **Entrada** | text |
| **Saída** | embeddings |
| **Preço Input** | $2e-8 /M tokens |
| **Preço Output** | Incluído |

**Descrição:** The Qwen3 Embedding model series is the latest proprietary model of the Qwen family, specifically designed for text embedding and ranking tasks. This series inherits the exceptional multilingual capabilities, long-text understanding, and reasoning skills of its foundational model. The Qwen3 Embedding series represents significant advancements in multiple text embedding and ranking tasks, including text retrieval, code retrieval, text classification, text clustering, and bitext mining.

### 223. MiniMax: MiniMax M2

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `minimax/minimax-m2` |
| **Categoria** | Código / Programação |
| **Contexto** | 196,608 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.000000255 /M tokens |
| **Preço Output** | $0.000001 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** MiniMax-M2 is a compact, high-efficiency large language model optimized for end-to-end coding and agentic workflows. With 10 billion activated parameters (230 billion total), it delivers near-frontier intelligence across general reasoning, tool use, and multi-step task execution while maintaining low latency and deployment efficiency.  The model excels in code generation, multi-file editing, compile-run-fix loops, and test-validated repair, showing strong results on SWE-Bench Verified, Multi-...

### 224. Qwen: Qwen3 VL 32B Instruct

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `qwen/qwen3-vl-32b-instruct` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 131,072 tokens |
| **Entrada** | text, image |
| **Saída** | text |
| **Preço Input** | $0.000000104 /M tokens |
| **Preço Output** | $0.000000416 /M tokens |

**Descrição:** Qwen3-VL-32B-Instruct is a large-scale multimodal vision-language model designed for high-precision understanding and reasoning across text, images, and video. With 32 billion parameters, it combines deep visual perception with advanced text comprehension, enabling fine-grained spatial reasoning, document and scene analysis, and long-horizon video understanding.Robust OCR in 32 languages, and enhanced multimodal fusion through Interleaved-MRoPE and DeepStack architectures. Optimized for agent...

### 225. IBM: Granite 4.0 Micro

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `ibm-granite/granite-4.0-h-micro` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 131,000 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.000000017 /M tokens |
| **Preço Output** | $0.00000011 /M tokens |

**Descrição:** Granite-4.0-H-Micro is a 3B parameter from the Granite 4 family of models. These models are the latest in a series of models released by IBM. They are fine-tuned for long context tool calling.

### 226. Microsoft: Phi 4 Mini Instruct

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `microsoft/phi-4-mini-instruct` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 128,000 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000008 /M tokens |
| **Preço Output** | $0.00000035 /M tokens |

**Descrição:** Phi-4-mini-instruct is a lightweight open model built upon synthetic data and filtered publicly available websites - with a focus on high-quality, reasoning dense data. The model belongs to the Phi-4 model family and supports 128K token context length. The model underwent an enhancement process, incorporating both supervised fine-tuning and direct preference optimization to support precise instruction adherence and robust safety measures.

### 227. OpenAI: GPT-5 Image Mini

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/gpt-5-image-mini` |
| **Categoria** | Multimodal (Texto + Imagem) |
| **Contexto** | 400,000 tokens |
| **Entrada** | file, image, text |
| **Saída** | image, text |
| **Preço Input** | $0.0000025 /M tokens |
| **Preço Output** | $0.000002 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** GPT-5 Image Mini combines OpenAI's advanced language capabilities, powered by [GPT-5 Mini](https://openrouter.ai/openai/gpt-5-mini), with GPT Image 1 Mini for efficient image generation. This natively multimodal model features superior instruction following, text rendering, and detailed image editing with reduced latency and cost. It excels at high-quality visual creation while maintaining strong text understanding, making it ideal for applications that require both efficient image generation...

### 228. Anthropic: Claude Haiku 4.5

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `anthropic/claude-haiku-4.5` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 200,000 tokens |
| **Entrada** | text, image, file |
| **Saída** | text |
| **Preço Input** | $0.000001 /M tokens |
| **Preço Output** | $0.000005 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Claude Haiku 4.5 is Anthropic’s fastest and most efficient model, delivering near-frontier intelligence at a fraction of the cost and latency of larger Claude models. Matching Claude Sonnet 4’s performance across reasoning, coding, and computer-use tasks, Haiku 4.5 brings frontier-level capability to real-time and high-volume applications.  It introduces extended thinking to the Haiku line; enabling controllable reasoning depth, summarized or interleaved thought output, and tool-assisted work...

### 229. Qwen: Qwen3 VL 8B Thinking

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `qwen/qwen3-vl-8b-thinking` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 131,072 tokens |
| **Entrada** | image, text |
| **Saída** | text |
| **Preço Input** | $0.000000117 /M tokens |
| **Preço Output** | $0.000001365 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Qwen3-VL-8B-Thinking is the reasoning-optimized variant of the Qwen3-VL-8B multimodal model, designed for advanced visual and textual reasoning across complex scenes, documents, and temporal sequences. It integrates enhanced multimodal alignment and long-context processing (native 256K, expandable to 1M tokens) for tasks such as scientific visual analysis, causal inference, and mathematical reasoning over image or video inputs.  Compared to the Instruct edition, the Thinking version introduce...

### 230. Qwen: Qwen3 VL 8B Instruct

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `qwen/qwen3-vl-8b-instruct` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 131,072 tokens |
| **Entrada** | image, text |
| **Saída** | text |
| **Preço Input** | $0.00000008 /M tokens |
| **Preço Output** | $0.0000005 /M tokens |

**Descrição:** Qwen3-VL-8B-Instruct is a multimodal vision-language model from the Qwen3-VL series, built for high-fidelity understanding and reasoning across text, images, and video. It features improved multimodal fusion with Interleaved-MRoPE for long-horizon temporal reasoning, DeepStack for fine-grained visual-text alignment, and text-timestamp alignment for precise event localization.  The model supports a native 256K-token context window, extensible to 1M tokens, and handles both static and dynamic m...

### 231. OpenAI: GPT-5 Image

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/gpt-5-image` |
| **Categoria** | Multimodal (Texto + Imagem) |
| **Contexto** | 400,000 tokens |
| **Entrada** | image, text, file |
| **Saída** | image, text |
| **Preço Input** | $0.00001 /M tokens |
| **Preço Output** | $0.00001 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** [GPT-5](https://openrouter.ai/openai/gpt-5) Image combines OpenAI's GPT-5 model with state-of-the-art image generation capabilities. It offers major improvements in reasoning, code quality, and user experience while incorporating GPT Image 1's superior instruction following, text rendering, and detailed image editing.

### 232. OpenAI: o3 Deep Research

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/o3-deep-research` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 200,000 tokens |
| **Entrada** | image, text, file |
| **Saída** | text |
| **Preço Input** | $0.00001 /M tokens |
| **Preço Output** | $0.00004 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** o3-deep-research is OpenAI's advanced model for deep research, designed to tackle complex, multi-step research tasks.  Note: This model always uses the 'web_search' tool which adds additional cost.

### 233. OpenAI: o4 Mini Deep Research

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/o4-mini-deep-research` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 200,000 tokens |
| **Entrada** | file, image, text |
| **Saída** | text |
| **Preço Input** | $0.000002 /M tokens |
| **Preço Output** | $0.000008 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** o4-mini-deep-research is OpenAI's faster, more affordable deep research model—ideal for tackling complex, multi-step research tasks.  Note: This model always uses the 'web_search' tool which adds additional cost.

### 234. NVIDIA: Llama 3.3 Nemotron Super 49B V1.5

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `nvidia/llama-3.3-nemotron-super-49b-v1.5` |
| **Categoria** | Código / Programação |
| **Contexto** | 131,072 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.0000001 /M tokens |
| **Preço Output** | $0.0000004 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Llama-3.3-Nemotron-Super-49B-v1.5 is a 49B-parameter, English-centric reasoning/chat model derived from Meta’s Llama-3.3-70B-Instruct with a 128K context. It’s post-trained for agentic workflows (RAG, tool calling) via SFT across math, code, science, and multi-turn chat, followed by multiple RL stages; Reward-aware Preference Optimization (RPO) for alignment, RL with Verifiable Rewards (RLVR) for step-wise reasoning, and iterative DPO to refine tool-use behavior. A distillation-driven Neural ...

### 235. Baidu: ERNIE 4.5 21B A3B Thinking

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `baidu/ernie-4.5-21b-a3b-thinking` |
| **Categoria** | Código / Programação |
| **Contexto** | 131,072 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000007 /M tokens |
| **Preço Output** | $0.00000028 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** ERNIE-4.5-21B-A3B-Thinking is Baidu's upgraded lightweight MoE model, refined to boost reasoning depth and quality for top-tier performance in logical puzzles, math, science, coding, text generation, and expert-level academic benchmarks.

### 236. Google: Nano Banana (Gemini 2.5 Flash Image)

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `google/gemini-2.5-flash-image` |
| **Categoria** | Multimodal (Texto + Imagem) |
| **Contexto** | 32,768 tokens |
| **Entrada** | image, text |
| **Saída** | image, text |
| **Preço Input** | $3e-7 /M tokens |
| **Preço Output** | $0.0000025 /M tokens |

**Descrição:** Gemini 2.5 Flash Image, a.k.a. "Nano Banana," is now generally available. It is a state of the art image generation model with contextual understanding. It is capable of image generation, edits, and multi-turn conversations. Aspect ratios can be controlled with the [image_config API Parameter](https://openrouter.ai/docs/features/multimodal/image-generation#image-aspect-ratio-configuration)

### 237. Qwen: Qwen3 VL 30B A3B Thinking

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `qwen/qwen3-vl-30b-a3b-thinking` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 131,072 tokens |
| **Entrada** | text, image |
| **Saída** | text |
| **Preço Input** | $0.00000013 /M tokens |
| **Preço Output** | $0.00000156 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Qwen3-VL-30B-A3B-Thinking is a multimodal model that unifies strong text generation with visual understanding for images and videos. Its Thinking variant enhances reasoning in STEM, math, and complex tasks. It excels in perception of real-world/synthetic categories, 2D/3D spatial grounding, and long-form visual comprehension, achieving competitive multimodal benchmark results. For agentic use, it handles multi-image multi-turn instructions, video timeline alignments, GUI automation, and visua...

### 238. Qwen: Qwen3 VL 30B A3B Instruct

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `qwen/qwen3-vl-30b-a3b-instruct` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 131,072 tokens |
| **Entrada** | text, image |
| **Saída** | text |
| **Preço Input** | $0.00000013 /M tokens |
| **Preço Output** | $0.00000052 /M tokens |

**Descrição:** Qwen3-VL-30B-A3B-Instruct is a multimodal model that unifies strong text generation with visual understanding for images and videos. Its Instruct variant optimizes instruction-following for general multimodal tasks. It excels in perception of real-world/synthetic categories, 2D/3D spatial grounding, and long-form visual comprehension, achieving competitive multimodal benchmark results. For agentic use, it handles multi-image multi-turn instructions, video timeline alignments, GUI automation, ...

### 239. OpenAI: GPT-5 Pro

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/gpt-5-pro` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 400,000 tokens |
| **Entrada** | image, text, file |
| **Saída** | text |
| **Preço Input** | $0.000015 /M tokens |
| **Preço Output** | $0.00012 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** GPT-5 Pro is OpenAI’s most advanced model, offering major improvements in reasoning, code quality, and user experience. It is optimized for complex tasks that require step-by-step reasoning, instruction following, and accuracy in high-stakes use cases. It supports test-time routing features and advanced prompt understanding, including user-specified intent like "think hard about this." Improvements include reductions in hallucination, sycophancy, and better performance in coding, writing, and...

### 240. Z.ai: GLM 4.6

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `z-ai/glm-4.6` |
| **Categoria** | Código / Programação |
| **Contexto** | 202,752 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000043 /M tokens |
| **Preço Output** | $0.00000174 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Compared with GLM-4.5, this generation brings several key improvements:  Longer context window: The context window has been expanded from 128K to 200K tokens, enabling the model to handle more complex agentic tasks. Superior coding performance: The model achieves higher scores on code benchmarks and demonstrates better real-world performance in applications such as Claude Code、Cline、Roo Code and Kilo Code, including improvements in generating visually polished front-end pages. Advanced reason...

### 241. Anthropic: Claude Sonnet 4.5

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `anthropic/claude-sonnet-4.5` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 1,000,000 tokens |
| **Entrada** | text, image, file |
| **Saída** | text |
| **Preço Input** | $0.000003 /M tokens |
| **Preço Output** | $0.000015 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Claude Sonnet 4.5 is Anthropic’s most advanced Sonnet model to date, optimized for real-world agents and coding workflows. It delivers state-of-the-art performance on coding benchmarks such as SWE-bench Verified, with improvements across system design, code security, and specification adherence. The model is designed for extended autonomous operation, maintaining task continuity across sessions and providing fact-based progress tracking.  Sonnet 4.5 also introduces stronger agentic capabiliti...

### 242. DeepSeek: DeepSeek V3.2 Exp

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `deepseek/deepseek-v3.2-exp` |
| **Categoria** | Código / Programação |
| **Contexto** | 163,840 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000027 /M tokens |
| **Preço Output** | $0.00000041 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** DeepSeek-V3.2-Exp is an experimental large language model released by DeepSeek as an intermediate step between V3.1 and future architectures. It introduces DeepSeek Sparse Attention (DSA), a fine-grained sparse attention mechanism designed to improve training and inference efficiency in long-context scenarios while maintaining output quality. Users can control the reasoning behaviour with the `reasoning` `enabled` boolean. [Learn more in our docs](https://openrouter.ai/docs/use-cases/reasonin...

### 243. TheDrummer: Cydonia 24B V4.1

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `thedrummer/cydonia-24b-v4.1` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 131,072 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.0000003 /M tokens |
| **Preço Output** | $0.0000005 /M tokens |

**Descrição:** Uncensored and creative writing model based on Mistral Small 3.2 24B with good recall, prompt adherence, and intelligence.

### 244. Relace: Relace Apply 3

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `relace/relace-apply-3` |
| **Categoria** | Código / Programação |
| **Contexto** | 256,000 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000085 /M tokens |
| **Preço Output** | $0.00000125 /M tokens |

**Descrição:** Relace Apply 3 is a specialized code-patching LLM that merges AI-suggested edits straight into your source files. It can apply updates from GPT-4o, Claude, and others into your files at 10,000 tokens/sec on average.  The model requires the prompt to be in the following format:  <instruction>{instruction}</instruction> <code>{initial_code}</code> <update>{edit_snippet}</update>  Zero Data Retention is enabled for Relace. Learn more about this model in their [documentation](https://docs.relace....

### 245. Google: Gemini 2.5 Flash Lite Preview 09-2025

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `google/gemini-2.5-flash-lite-preview-09-2025` |
| **Categoria** | Multimodal (Vídeo) |
| **Contexto** | 1,048,576 tokens |
| **Entrada** | text, image, file, audio, video |
| **Saída** | text |
| **Preço Input** | $1e-7 /M tokens |
| **Preço Output** | $4e-7 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Gemini 2.5 Flash-Lite is a lightweight reasoning model in the Gemini 2.5 family, optimized for ultra-low latency and cost efficiency. It offers improved throughput, faster token generation, and better performance across common benchmarks compared to earlier Flash models. By default, "thinking" (i.e. multi-pass reasoning) is disabled to prioritize speed, but developers can enable it via the [Reasoning API parameter](https://openrouter.ai/docs/use-cases/reasoning-tokens) to selectively trade of...

### 246. Qwen: Qwen3 VL 235B A22B Thinking

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `qwen/qwen3-vl-235b-a22b-thinking` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 131,072 tokens |
| **Entrada** | text, image |
| **Saída** | text |
| **Preço Input** | $0.00000026 /M tokens |
| **Preço Output** | $0.0000026 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Qwen3-VL-235B-A22B Thinking is a multimodal model that unifies strong text generation with visual understanding across images and video. The Thinking model is optimized for multimodal reasoning in STEM and math. The series emphasizes robust perception (recognition of diverse real-world and synthetic categories), spatial understanding (2D/3D grounding), and long-form visual comprehension, with competitive results on public multimodal benchmarks for both perception and reasoning.  Beyond analys...

### 247. Qwen: Qwen3 VL 235B A22B Instruct

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `qwen/qwen3-vl-235b-a22b-instruct` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 262,144 tokens |
| **Entrada** | text, image |
| **Saída** | text |
| **Preço Input** | $0.0000002 /M tokens |
| **Preço Output** | $0.00000088 /M tokens |

**Descrição:** Qwen3-VL-235B-A22B Instruct is an open-weight multimodal model that unifies strong text generation with visual understanding across images and video. The Instruct model targets general vision-language use (VQA, document parsing, chart/table extraction, multilingual OCR). The series emphasizes robust perception (recognition of diverse real-world and synthetic categories), spatial understanding (2D/3D grounding), and long-form visual comprehension, with competitive results on public multimodal ...

### 248. Qwen: Qwen3 Max

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `qwen/qwen3-max` |
| **Categoria** | Código / Programação |
| **Contexto** | 262,144 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000078 /M tokens |
| **Preço Output** | $0.0000039 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Qwen3-Max is an updated release built on the Qwen3 series, offering major improvements in reasoning, instruction following, multilingual support, and long-tail knowledge coverage compared to the January 2025 version. It delivers higher accuracy in math, coding, logic, and science tasks, follows complex instructions in Chinese and English more reliably, reduces hallucinations, and produces higher-quality responses for open-ended Q&A, writing, and conversation. The model supports over 100 langu...

### 249. Qwen: Qwen3 Coder Plus

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `qwen/qwen3-coder-plus` |
| **Categoria** | Código / Programação |
| **Contexto** | 1,000,000 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000065 /M tokens |
| **Preço Output** | $0.00000325 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Qwen3 Coder Plus is Alibaba's proprietary version of the Open Source Qwen3 Coder 480B A35B. It is a powerful coding agent model specializing in autonomous programming via tool calling and environment interaction, combining coding proficiency with versatile general-purpose abilities.

### 250. OpenAI: GPT-5 Codex

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/gpt-5-codex` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 400,000 tokens |
| **Entrada** | text, image |
| **Saída** | text |
| **Preço Input** | $0.00000125 /M tokens |
| **Preço Output** | $0.00001 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** GPT-5-Codex is a specialized version of GPT-5 optimized for software engineering and coding workflows. It is designed for both interactive development sessions and long, independent execution of complex engineering tasks. The model supports building projects from scratch, feature development, debugging, large-scale refactoring, and code review. Compared to GPT-5, Codex is more steerable, adheres closely to developer instructions, and produces cleaner, higher-quality code outputs. Reasoning ef...

### 251. DeepSeek: DeepSeek V3.1 Terminus

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `deepseek/deepseek-v3.1-terminus` |
| **Categoria** | Código / Programação |
| **Contexto** | 163,840 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000027 /M tokens |
| **Preço Output** | $0.00000095 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** DeepSeek-V3.1 Terminus is an update to [DeepSeek V3.1](/deepseek/deepseek-chat-v3.1) that maintains the model's original capabilities while addressing issues reported by users, including language consistency and agent capabilities, further optimizing the model's performance in coding and search agents. It is a large hybrid reasoning model (671B parameters, 37B active) that supports both thinking and non-thinking modes. It extends the DeepSeek-V3 base with a two-phase long-context training pro...

### 252. xAI: Grok 4 Fast

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `x-ai/grok-4-fast` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 2,000,000 tokens |
| **Entrada** | text, image, file |
| **Saída** | text |
| **Preço Input** | $0.0000002 /M tokens |
| **Preço Output** | $0.0000005 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Grok 4 Fast is xAI's latest multimodal model with SOTA cost-efficiency and a 2M token context window. It comes in two flavors: non-reasoning and reasoning. Read more about the model on xAI's [news post](http://x.ai/news/grok-4-fast).  Reasoning can be enabled/disabled using the `reasoning` `enabled` parameter in the API. [Learn more in our docs](https://openrouter.ai/docs/use-cases/reasoning-tokens#controlling-reasoning-tokens)

### 253. Tongyi DeepResearch 30B A3B

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `alibaba/tongyi-deepresearch-30b-a3b` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 131,072 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000009 /M tokens |
| **Preço Output** | $0.00000045 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Tongyi DeepResearch is an agentic large language model developed by Tongyi Lab, with 30 billion total parameters activating only 3 billion per token. It's optimized for long-horizon, deep information-seeking tasks and delivers state-of-the-art performance on benchmarks like Humanity's Last Exam, BrowserComp, BrowserComp-ZH, WebWalkerQA, GAIA, xbench-DeepSearch, and FRAMES. This makes it superior for complex agentic search, reasoning, and multi-step problem-solving compared to prior models.  T...

### 254. Qwen: Qwen3 Coder Flash

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `qwen/qwen3-coder-flash` |
| **Categoria** | Código / Programação |
| **Contexto** | 1,000,000 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.000000195 /M tokens |
| **Preço Output** | $0.000000975 /M tokens |

**Descrição:** Qwen3 Coder Flash is Alibaba's fast and cost efficient version of their proprietary Qwen3 Coder Plus. It is a powerful coding agent model specializing in autonomous programming via tool calling and environment interaction, combining coding proficiency with versatile general-purpose abilities.

### 255. Qwen: Qwen3 Next 80B A3B Thinking

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `qwen/qwen3-next-80b-a3b-thinking` |
| **Categoria** | Código / Programação |
| **Contexto** | 131,072 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.0000000975 /M tokens |
| **Preço Output** | $0.00000078 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Qwen3-Next-80B-A3B-Thinking is a reasoning-first chat model in the Qwen3-Next line that outputs structured “thinking” traces by default. It’s designed for hard multi-step problems; math proofs, code synthesis/debugging, logic, and agentic planning, and reports strong results across knowledge, reasoning, coding, alignment, and multilingual evaluations. Compared with prior Qwen3 variants, it emphasizes stability under long chains of thought and efficient scaling during inference, and it is tune...

### 256. Qwen: Qwen3 Next 80B A3B Instruct (free)

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `qwen/qwen3-next-80b-a3b-instruct` |
| **Categoria** | Código / Programação |
| **Contexto** | 262,144 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0 /M tokens |
| **Preço Output** | $0 /M tokens |
| **Gratuito** | Sim |

**Descrição:** Qwen3-Next-80B-A3B-Instruct is an instruction-tuned chat model in the Qwen3-Next series optimized for fast, stable responses without “thinking” traces. It targets complex tasks across reasoning, code generation, knowledge QA, and multilingual use, while remaining robust on alignment and formatting. Compared with prior Qwen3 instruct variants, it focuses on higher throughput and stability on ultra-long inputs and multi-turn dialogues, making it well-suited for RAG, tool use, and agentic workfl...

### 257. Qwen: Qwen3 Next 80B A3B Instruct

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `qwen/qwen3-next-80b-a3b-instruct` |
| **Categoria** | Código / Programação |
| **Contexto** | 262,144 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000009 /M tokens |
| **Preço Output** | $0.0000011 /M tokens |

**Descrição:** Qwen3-Next-80B-A3B-Instruct is an instruction-tuned chat model in the Qwen3-Next series optimized for fast, stable responses without “thinking” traces. It targets complex tasks across reasoning, code generation, knowledge QA, and multilingual use, while remaining robust on alignment and formatting. Compared with prior Qwen3 instruct variants, it focuses on higher throughput and stability on ultra-long inputs and multi-turn dialogues, making it well-suited for RAG, tool use, and agentic workfl...

### 258. Qwen: Qwen Plus 0728 (thinking)

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `qwen/qwen-plus-2025-07-28` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 1,000,000 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000026 /M tokens |
| **Preço Output** | $0.00000078 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Qwen Plus 0728, based on the Qwen3 foundation model, is a 1 million context hybrid reasoning model with a balanced performance, speed, and cost combination.

### 259. Qwen: Qwen Plus 0728

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `qwen/qwen-plus-2025-07-28` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 1,000,000 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000026 /M tokens |
| **Preço Output** | $0.00000078 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Qwen Plus 0728, based on the Qwen3 foundation model, is a 1 million context hybrid reasoning model with a balanced performance, speed, and cost combination.

### 260. NVIDIA: Nemotron Nano 9B V2 (free)

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `nvidia/nemotron-nano-9b-v2` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 128,000 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0 /M tokens |
| **Preço Output** | $0 /M tokens |
| **Gratuito** | Sim |
| **Raciocínio** | Suportado |

**Descrição:** NVIDIA-Nemotron-Nano-9B-v2 is a large language model (LLM) trained from scratch by NVIDIA, and designed as a unified model for both reasoning and non-reasoning tasks. It responds to user queries and tasks by first generating a reasoning trace and then concluding with a final response.   The model's reasoning capabilities can be controlled via a system prompt. If the user prefers the model to provide its final answer without intermediate reasoning traces, it can be configured to do so.

### 261. NVIDIA: Nemotron Nano 9B V2

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `nvidia/nemotron-nano-9b-v2` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 131,072 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000004 /M tokens |
| **Preço Output** | $0.00000016 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** NVIDIA-Nemotron-Nano-9B-v2 is a large language model (LLM) trained from scratch by NVIDIA, and designed as a unified model for both reasoning and non-reasoning tasks. It responds to user queries and tasks by first generating a reasoning trace and then concluding with a final response.   The model's reasoning capabilities can be controlled via a system prompt. If the user prefers the model to provide its final answer without intermediate reasoning traces, it can be configured to do so.

### 262. MoonshotAI: Kimi K2 0905

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `moonshotai/kimi-k2-0905` |
| **Categoria** | Código / Programação |
| **Contexto** | 262,144 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.0000006 /M tokens |
| **Preço Output** | $0.0000025 /M tokens |

**Descrição:** Kimi K2 0905 is the September update of [Kimi K2 0711](moonshotai/kimi-k2). It is a large-scale Mixture-of-Experts (MoE) language model developed by Moonshot AI, featuring 1 trillion total parameters with 32 billion active per forward pass. It supports long-context inference up to 256k tokens, extended from the previous 128k.  This update improves agentic coding with higher accuracy and better generalization across scaffolds, and enhances frontend coding with more aesthetic and functional out...

### 263. Qwen: Qwen3 30B A3B Thinking 2507

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `qwen/qwen3-30b-a3b-thinking-2507` |
| **Categoria** | Código / Programação |
| **Contexto** | 131,072 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000008 /M tokens |
| **Preço Output** | $0.0000004 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Qwen3-30B-A3B-Thinking-2507 is a 30B parameter Mixture-of-Experts reasoning model optimized for complex tasks requiring extended multi-step thinking. The model is designed specifically for “thinking mode,” where internal reasoning traces are separated from final answers.  Compared to earlier Qwen3-30B releases, this version improves performance across logical reasoning, mathematics, science, coding, and multilingual benchmarks. It also demonstrates stronger instruction following, tool use, an...

### 264. xAI: Grok Code Fast 1

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `x-ai/grok-code-fast-1` |
| **Categoria** | Código / Programação |
| **Contexto** | 256,000 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.0000002 /M tokens |
| **Preço Output** | $0.0000015 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Grok Code Fast 1 is a speedy and economical reasoning model that excels at agentic coding. With reasoning traces visible in the response, developers can steer Grok Code for high-quality work flows.

### 265. Nous: Hermes 4 70B

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `nousresearch/hermes-4-70b` |
| **Categoria** | Código / Programação |
| **Contexto** | 131,072 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000013 /M tokens |
| **Preço Output** | $0.0000004 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Hermes 4 70B is a hybrid reasoning model from Nous Research, built on Meta-Llama-3.1-70B. It introduces the same hybrid mode as the larger 405B release, allowing the model to either respond directly or generate explicit <think>...</think> reasoning traces before answering. Users can control the reasoning behaviour with the `reasoning` `enabled` boolean. [Learn more in our docs](https://openrouter.ai/docs/use-cases/reasoning-tokens#enable-reasoning-with-default-config)  This 70B variant is tra...

### 266. Nous: Hermes 4 405B

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `nousresearch/hermes-4-405b` |
| **Categoria** | Código / Programação |
| **Contexto** | 131,072 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.000001 /M tokens |
| **Preço Output** | $0.000003 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Hermes 4 is a large-scale reasoning model built on Meta-Llama-3.1-405B and released by Nous Research. It introduces a hybrid reasoning mode, where the model can choose to deliberate internally with <think>...</think> traces or respond directly, offering flexibility between speed and depth. Users can control the reasoning behaviour with the `reasoning` `enabled` boolean. [Learn more in our docs](https://openrouter.ai/docs/use-cases/reasoning-tokens#enable-reasoning-with-default-config)  The mo...

### 267. DeepSeek: DeepSeek V3.1

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `deepseek/deepseek-chat-v3.1` |
| **Categoria** | Código / Programação |
| **Contexto** | 163,840 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000021 /M tokens |
| **Preço Output** | $0.00000079 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** DeepSeek-V3.1 is a large hybrid reasoning model (671B parameters, 37B active) that supports both thinking and non-thinking modes via prompt templates. It extends the DeepSeek-V3 base with a two-phase long-context training process, reaching up to 128K tokens, and uses FP8 microscaling for efficient inference. Users can control the reasoning behaviour with the `reasoning` `enabled` boolean. [Learn more in our docs](https://openrouter.ai/docs/use-cases/reasoning-tokens#enable-reasoning-with-defa...

### 268. OpenAI: GPT-4o Audio

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/gpt-4o-audio-preview` |
| **Categoria** | Multimodal (Texto + Áudio) |
| **Contexto** | 128,000 tokens |
| **Entrada** | audio, text |
| **Saída** | text, audio |
| **Preço Input** | $0.0000025 /M tokens |
| **Preço Output** | $0.00001 /M tokens |

**Descrição:** The gpt-4o-audio-preview model adds support for audio inputs as prompts. This enhancement allows the model to detect nuances within audio recordings and add depth to generated user experiences. Audio outputs are currently not supported. Audio tokens are priced at $40 per million input and $80 per million output audio tokens.

### 269. Mistral: Mistral Medium 3.1

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `mistralai/mistral-medium-3.1` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 131,072 tokens |
| **Entrada** | text, image, file |
| **Saída** | text |
| **Preço Input** | $4e-7 /M tokens |
| **Preço Output** | $0.000002 /M tokens |

**Descrição:** Mistral Medium 3.1 is an updated version of Mistral Medium 3, which is a high-performance enterprise-grade language model designed to deliver frontier-level capabilities at significantly reduced operational cost. It balances state-of-the-art reasoning and multimodal performance with 8× lower cost compared to traditional large models, making it suitable for scalable deployments across professional and industrial use cases.  The model excels in domains such as coding, STEM reasoning, and enterp...

### 270. Baidu: ERNIE 4.5 21B A3B

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `baidu/ernie-4.5-21b-a3b` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 120,000 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000007 /M tokens |
| **Preço Output** | $0.00000028 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** A sophisticated text-based Mixture-of-Experts (MoE) model featuring 21B total parameters with 3B activated per token, delivering exceptional multimodal understanding and generation through heterogeneous MoE structures and modality-isolated routing. Supporting an extensive 131K token context length, the model achieves efficient inference via multi-expert parallel collaboration and quantization, while advanced post-training techniques including SFT, DPO, and UPO ensure optimized performance acr...

### 271. Baidu: ERNIE 4.5 VL 28B A3B

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `baidu/ernie-4.5-vl-28b-a3b` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 30,000 tokens |
| **Entrada** | text, image |
| **Saída** | text |
| **Preço Input** | $0.00000014 /M tokens |
| **Preço Output** | $0.00000056 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** A powerful multimodal Mixture-of-Experts chat model featuring 28B total parameters with 3B activated per token, delivering exceptional text and vision understanding through its innovative heterogeneous MoE structure with modality-isolated routing. Built with scaling-efficient infrastructure for high-throughput training and inference, the model leverages advanced post-training techniques including SFT, DPO, and UPO for optimized performance, while supporting an impressive 131K context length a...

### 272. Z.ai: GLM 4.5V

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `z-ai/glm-4.5v` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 65,536 tokens |
| **Entrada** | text, image |
| **Saída** | text |
| **Preço Input** | $6e-7 /M tokens |
| **Preço Output** | $0.0000018 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** GLM-4.5V is a vision-language foundation model for multimodal agent applications. Built on a Mixture-of-Experts (MoE) architecture with 106B parameters and 12B activated parameters, it achieves state-of-the-art results in video understanding, image Q&A, OCR, and document parsing, with strong gains in front-end web coding, grounding, and spatial reasoning. It offers a hybrid inference mode: a "thinking mode" for deep reasoning and a "non-thinking mode" for fast responses. Reasoning behavior ca...

### 273. AI21: Jamba Large 1.7

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `ai21/jamba-large-1.7` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 256,000 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.000002 /M tokens |
| **Preço Output** | $0.000008 /M tokens |

**Descrição:** Jamba Large 1.7 is the latest model in the Jamba open family, offering improvements in grounding, instruction-following, and overall efficiency. Built on a hybrid SSM-Transformer architecture with a 256K context window, it delivers more accurate, contextually grounded responses and better steerability than previous versions.

### 274. OpenAI: GPT-5 Chat

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/gpt-5-chat` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 128,000 tokens |
| **Entrada** | file, image, text |
| **Saída** | text |
| **Preço Input** | $0.00000125 /M tokens |
| **Preço Output** | $0.00001 /M tokens |

**Descrição:** GPT-5 Chat is designed for advanced, natural, multimodal, and context-aware conversations for enterprise applications.

### 275. OpenAI: GPT-5

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/gpt-5` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 400,000 tokens |
| **Entrada** | text, image, file |
| **Saída** | text |
| **Preço Input** | $0.00000125 /M tokens |
| **Preço Output** | $0.00001 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** GPT-5 is OpenAI’s most advanced model, offering major improvements in reasoning, code quality, and user experience. It is optimized for complex tasks that require step-by-step reasoning, instruction following, and accuracy in high-stakes use cases. It supports test-time routing features and advanced prompt understanding, including user-specified intent like "think hard about this." Improvements include reductions in hallucination, sycophancy, and better performance in coding, writing, and hea...

### 276. OpenAI: GPT-5 Mini

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/gpt-5-mini` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 400,000 tokens |
| **Entrada** | text, image, file |
| **Saída** | text |
| **Preço Input** | $2.5e-7 /M tokens |
| **Preço Output** | $0.000002 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** GPT-5 Mini is a compact version of GPT-5, designed to handle lighter-weight reasoning tasks. It provides the same instruction-following and safety-tuning benefits as GPT-5, but with reduced latency and cost. GPT-5 Mini is the successor to OpenAI's o4-mini model.

### 277. OpenAI: GPT-5 Nano

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/gpt-5-nano` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 400,000 tokens |
| **Entrada** | text, image, file |
| **Saída** | text |
| **Preço Input** | $5e-8 /M tokens |
| **Preço Output** | $4e-7 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** GPT-5-Nano is the smallest and fastest variant in the GPT-5 system, optimized for developer tools, rapid interactions, and ultra-low latency environments. While limited in reasoning depth compared to its larger counterparts, it retains key instruction-following and safety features. It is the successor to GPT-4.1-nano and offers a lightweight option for cost-sensitive or real-time applications.

### 278. OpenAI: gpt-oss-120b (free)

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/gpt-oss-120b` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 131,072 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0 /M tokens |
| **Preço Output** | $0 /M tokens |
| **Gratuito** | Sim |
| **Raciocínio** | Suportado |

**Descrição:** gpt-oss-120b is an open-weight, 117B-parameter Mixture-of-Experts (MoE) language model from OpenAI designed for high-reasoning, agentic, and general-purpose production use cases. It activates 5.1B parameters per forward pass and is optimized to run on a single H100 GPU with native MXFP4 quantization. The model supports configurable reasoning depth, full chain-of-thought access, and native tool use, including function calling, browsing, and structured output generation.

### 279. OpenAI: gpt-oss-120b

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/gpt-oss-120b` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 131,072 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.000000039 /M tokens |
| **Preço Output** | $0.00000018 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** gpt-oss-120b is an open-weight, 117B-parameter Mixture-of-Experts (MoE) language model from OpenAI designed for high-reasoning, agentic, and general-purpose production use cases. It activates 5.1B parameters per forward pass and is optimized to run on a single H100 GPU with native MXFP4 quantization. The model supports configurable reasoning depth, full chain-of-thought access, and native tool use, including function calling, browsing, and structured output generation.

### 280. OpenAI: gpt-oss-20b (free)

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/gpt-oss-20b` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 131,072 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0 /M tokens |
| **Preço Output** | $0 /M tokens |
| **Gratuito** | Sim |
| **Raciocínio** | Suportado |

**Descrição:** gpt-oss-20b is an open-weight 21B parameter model released by OpenAI under the Apache 2.0 license. It uses a Mixture-of-Experts (MoE) architecture with 3.6B active parameters per forward pass, optimized for lower-latency inference and deployability on consumer or single-GPU hardware. The model is trained in OpenAI’s Harmony response format and supports reasoning level configuration, fine-tuning, and agentic capabilities including function calling, tool use, and structured outputs.

### 281. OpenAI: gpt-oss-20b

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/gpt-oss-20b` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 131,072 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000003 /M tokens |
| **Preço Output** | $0.00000014 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** gpt-oss-20b is an open-weight 21B parameter model released by OpenAI under the Apache 2.0 license. It uses a Mixture-of-Experts (MoE) architecture with 3.6B active parameters per forward pass, optimized for lower-latency inference and deployability on consumer or single-GPU hardware. The model is trained in OpenAI’s Harmony response format and supports reasoning level configuration, fine-tuning, and agentic capabilities including function calling, tool use, and structured outputs.

### 282. Anthropic: Claude Opus 4.1

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `anthropic/claude-opus-4.1` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 200,000 tokens |
| **Entrada** | image, text, file |
| **Saída** | text |
| **Preço Input** | $0.000015 /M tokens |
| **Preço Output** | $0.000075 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Claude Opus 4.1 is an updated version of Anthropic’s flagship model, offering improved performance in coding, reasoning, and agentic tasks. It achieves 74.5% on SWE-bench Verified and shows notable gains in multi-file code refactoring, debugging precision, and detail-oriented reasoning. The model supports extended thinking up to 64K tokens and is optimized for tasks involving research, data analysis, and tool-assisted reasoning.

### 283. Mistral: Codestral 2508

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `mistralai/codestral-2508` |
| **Categoria** | Multimodal (Arquivo + Texto) |
| **Contexto** | 256,000 tokens |
| **Entrada** | text, file |
| **Saída** | text |
| **Preço Input** | $3e-7 /M tokens |
| **Preço Output** | $9e-7 /M tokens |

**Descrição:** Mistral's cutting-edge language model for coding released end of July 2025. Codestral specializes in low-latency, high-frequency tasks such as fill-in-the-middle (FIM), code correction and test generation.  [Blog Post](https://mistral.ai/news/codestral-25-08)

### 284. Qwen: Qwen3 Coder 30B A3B Instruct

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `qwen/qwen3-coder-30b-a3b-instruct` |
| **Categoria** | Código / Programação |
| **Contexto** | 160,000 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000007 /M tokens |
| **Preço Output** | $0.00000027 /M tokens |

**Descrição:** Qwen3-Coder-30B-A3B-Instruct is a 30.5B parameter Mixture-of-Experts (MoE) model with 128 experts (8 active per forward pass), designed for advanced code generation, repository-scale understanding, and agentic tool use. Built on the Qwen3 architecture, it supports a native context length of 256K tokens (extendable to 1M with Yarn) and performs strongly in tasks involving function calls, browser use, and structured code completion.  This model is optimized for instruction-following without “th...

### 285. Qwen: Qwen3 30B A3B Instruct 2507

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `qwen/qwen3-30b-a3b-instruct-2507` |
| **Categoria** | Código / Programação |
| **Contexto** | 262,144 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000009 /M tokens |
| **Preço Output** | $0.0000003 /M tokens |

**Descrição:** Qwen3-30B-A3B-Instruct-2507 is a 30.5B-parameter mixture-of-experts language model from Qwen, with 3.3B active parameters per inference. It operates in non-thinking mode and is designed for high-quality instruction following, multilingual understanding, and agentic tool use. Post-trained on instruction data, it demonstrates competitive performance across reasoning (AIME, ZebraLogic), coding (MultiPL-E, LiveCodeBench), and alignment (IFEval, WritingBench) benchmarks. It outperforms its non-ins...

### 286. Z.ai: GLM 4.5

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `z-ai/glm-4.5` |
| **Categoria** | Código / Programação |
| **Contexto** | 131,072 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.0000006 /M tokens |
| **Preço Output** | $0.0000022 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** GLM-4.5 is our latest flagship foundation model, purpose-built for agent-based applications. It leverages a Mixture-of-Experts (MoE) architecture and supports a context length of up to 128k tokens. GLM-4.5 delivers significantly enhanced capabilities in reasoning, code generation, and agent alignment. It supports a hybrid inference mode with two options, a "thinking mode" designed for complex reasoning and tool use, and a "non-thinking mode" optimized for instant responses. Users can control ...

### 287. Z.ai: GLM 4.5 Air (free)

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `z-ai/glm-4.5-air` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 131,072 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0 /M tokens |
| **Preço Output** | $0 /M tokens |
| **Gratuito** | Sim |
| **Raciocínio** | Suportado |

**Descrição:** GLM-4.5-Air is the lightweight variant of our latest flagship model family, also purpose-built for agent-centric applications. Like GLM-4.5, it adopts the Mixture-of-Experts (MoE) architecture but with a more compact parameter size. GLM-4.5-Air also supports hybrid inference modes, offering a "thinking mode" for advanced reasoning and tool use, and a "non-thinking mode" for real-time interaction. Users can control the reasoning behaviour with the `reasoning` `enabled` boolean. [Learn more in ...

### 288. Z.ai: GLM 4.5 Air

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `z-ai/glm-4.5-air` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 131,072 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000013 /M tokens |
| **Preço Output** | $0.00000085 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** GLM-4.5-Air is the lightweight variant of our latest flagship model family, also purpose-built for agent-centric applications. Like GLM-4.5, it adopts the Mixture-of-Experts (MoE) architecture but with a more compact parameter size. GLM-4.5-Air also supports hybrid inference modes, offering a "thinking mode" for advanced reasoning and tool use, and a "non-thinking mode" for real-time interaction. Users can control the reasoning behaviour with the `reasoning` `enabled` boolean. [Learn more in ...

### 289. Qwen: Qwen3 235B A22B Thinking 2507

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `qwen/qwen3-235b-a22b-thinking-2507` |
| **Categoria** | Código / Programação |
| **Contexto** | 131,072 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.0000001495 /M tokens |
| **Preço Output** | $0.000001495 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Qwen3-235B-A22B-Thinking-2507 is a high-performance, open-weight Mixture-of-Experts (MoE) language model optimized for complex reasoning tasks. It activates 22B of its 235B parameters per forward pass and natively supports up to 262,144 tokens of context. This "thinking-only" variant enhances structured logical reasoning, mathematics, science, and long-form generation, showing strong benchmark performance across AIME, SuperGPQA, LiveCodeBench, and MMLU-Redux. It enforces a special reasoning m...

### 290. Z.ai: GLM 4 32B 

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `z-ai/glm-4-32b` |
| **Categoria** | Código / Programação |
| **Contexto** | 128,000 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $1e-7 /M tokens |
| **Preço Output** | $1e-7 /M tokens |

**Descrição:** GLM 4 32B is a cost-effective foundation language model.  It can efficiently perform complex tasks and has significantly enhanced capabilities in tool use, online search, and code-related intelligent tasks.  It is made by the same lab behind the thudm models.

### 291. Qwen: Qwen3 Coder 480B A35B (free)

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `qwen/qwen3-coder` |
| **Categoria** | Código / Programação |
| **Contexto** | 262,000 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0 /M tokens |
| **Preço Output** | $0 /M tokens |
| **Gratuito** | Sim |

**Descrição:** Qwen3-Coder-480B-A35B-Instruct is a Mixture-of-Experts (MoE) code generation model developed by the Qwen team. It is optimized for agentic coding tasks such as function calling, tool use, and long-context reasoning over repositories. The model features 480 billion total parameters, with 35 billion active per forward pass (8 out of 160 experts).  Pricing for the Alibaba endpoints varies by context length. Once a request is greater than 128k input tokens, the higher pricing is used.

### 292. Qwen: Qwen3 Coder 480B A35B

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `qwen/qwen3-coder` |
| **Categoria** | Código / Programação |
| **Contexto** | 262,144 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000022 /M tokens |
| **Preço Output** | $0.0000018 /M tokens |

**Descrição:** Qwen3-Coder-480B-A35B-Instruct is a Mixture-of-Experts (MoE) code generation model developed by the Qwen team. It is optimized for agentic coding tasks such as function calling, tool use, and long-context reasoning over repositories. The model features 480 billion total parameters, with 35 billion active per forward pass (8 out of 160 experts).  Pricing for the Alibaba endpoints varies by context length. Once a request is greater than 128k input tokens, the higher pricing is used.

### 293. ByteDance: UI-TARS 7B 

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `bytedance/ui-tars-1.5-7b` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 128,000 tokens |
| **Entrada** | image, text |
| **Saída** | text |
| **Preço Input** | $0.0000001 /M tokens |
| **Preço Output** | $0.0000002 /M tokens |

**Descrição:** UI-TARS-1.5 is a multimodal vision-language agent optimized for GUI-based environments, including desktop interfaces, web browsers, mobile systems, and games. Built by ByteDance, it builds upon the UI-TARS framework with reinforcement learning-based reasoning, enabling robust action planning and execution across virtual interfaces.  This model achieves state-of-the-art results on a range of interactive and grounding benchmarks, including OSworld, WebVoyager, AndroidWorld, and ScreenSpot. It a...

### 294. Google: Gemini 2.5 Flash Lite

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `google/gemini-2.5-flash-lite` |
| **Categoria** | Multimodal (Vídeo) |
| **Contexto** | 1,048,576 tokens |
| **Entrada** | text, image, file, audio, video |
| **Saída** | text |
| **Preço Input** | $1e-7 /M tokens |
| **Preço Output** | $4e-7 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Gemini 2.5 Flash-Lite is a lightweight reasoning model in the Gemini 2.5 family, optimized for ultra-low latency and cost efficiency. It offers improved throughput, faster token generation, and better performance across common benchmarks compared to earlier Flash models. By default, "thinking" (i.e. multi-pass reasoning) is disabled to prioritize speed, but developers can enable it via the [Reasoning API parameter](https://openrouter.ai/docs/use-cases/reasoning-tokens) to selectively trade of...

### 295. Qwen: Qwen3 235B A22B Instruct 2507

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `qwen/qwen3-235b-a22b-2507` |
| **Categoria** | Código / Programação |
| **Contexto** | 262,144 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.000000071 /M tokens |
| **Preço Output** | $0.0000001 /M tokens |

**Descrição:** Qwen3-235B-A22B-Instruct-2507 is a multilingual, instruction-tuned mixture-of-experts language model based on the Qwen3-235B architecture, with 22B active parameters per forward pass. It is optimized for general-purpose text generation, including instruction following, logical reasoning, math, code, and tool usage. The model supports a native 262K context length and does not implement "thinking mode" (<think> blocks).  Compared to its base variant, this version delivers significant gains in k...

### 296. Switchpoint Router

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `switchpoint/router` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 131,072 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $8.5e-7 /M tokens |
| **Preço Output** | $0.0000034 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Switchpoint AI's router instantly analyzes your request and directs it to the optimal AI from an ever-evolving library.   As the world of LLMs advances, our router gets smarter, ensuring you always benefit from the industry's newest models without changing your workflow.  This model is configured for a simple, flat rate per response here on OpenRouter. It's powered by the full routing engine from [Switchpoint AI](https://www.switchpoint.dev).

### 297. MoonshotAI: Kimi K2 0711

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `moonshotai/kimi-k2` |
| **Categoria** | Código / Programação |
| **Contexto** | 131,072 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000057 /M tokens |
| **Preço Output** | $0.0000023 /M tokens |

**Descrição:** Kimi K2 Instruct is a large-scale Mixture-of-Experts (MoE) language model developed by Moonshot AI, featuring 1 trillion total parameters with 32 billion active per forward pass. It is optimized for agentic capabilities, including advanced tool use, reasoning, and code synthesis. Kimi K2 excels across a broad range of benchmarks, particularly in coding (LiveCodeBench, SWE-bench), reasoning (ZebraLogic, GPQA), and tool-use (Tau2, AceBench) tasks. It supports long-context inference up to 128K t...

### 298. Mistral: Devstral Medium

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `mistralai/devstral-medium` |
| **Categoria** | Multimodal (Arquivo + Texto) |
| **Contexto** | 131,072 tokens |
| **Entrada** | text, file |
| **Saída** | text |
| **Preço Input** | $4e-7 /M tokens |
| **Preço Output** | $0.000002 /M tokens |

**Descrição:** Devstral Medium is a high-performance code generation and agentic reasoning model developed jointly by Mistral AI and All Hands AI. Positioned as a step up from Devstral Small, it achieves 61.6% on SWE-Bench Verified, placing it ahead of Gemini 2.5 Pro and GPT-4.1 in code-related tasks, at a fraction of the cost. It is designed for generalization across prompt styles and tool use in code agents and frameworks.  Devstral Medium is available via API only (not open-weight), and supports enterpri...

### 299. Mistral: Devstral Small 1.1

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `mistralai/devstral-small` |
| **Categoria** | Multimodal (Arquivo + Texto) |
| **Contexto** | 131,072 tokens |
| **Entrada** | text, file |
| **Saída** | text |
| **Preço Input** | $1e-7 /M tokens |
| **Preço Output** | $3e-7 /M tokens |

**Descrição:** Devstral Small 1.1 is a 24B parameter open-weight language model for software engineering agents, developed by Mistral AI in collaboration with All Hands AI. Finetuned from Mistral Small 3.1 and released under the Apache 2.0 license, it features a 128k token context window and supports both Mistral-style function calling and XML output formats.  Designed for agentic coding workflows, Devstral Small 1.1 is optimized for tasks such as codebase exploration, multi-file edits, and integration into...

### 300. Venice: Uncensored (free)

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `cognitivecomputations/dolphin-mistral-24b-venice-edition` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 32,768 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0 /M tokens |
| **Preço Output** | $0 /M tokens |
| **Gratuito** | Sim |

**Descrição:** Venice Uncensored Dolphin Mistral 24B Venice Edition is a fine-tuned variant of Mistral-Small-24B-Instruct-2501, developed by dphn.ai in collaboration with Venice.ai. This model is designed as an “uncensored” instruct-tuned LLM, preserving user control over alignment, system prompts, and behavior. Intended for advanced and unrestricted use cases, Venice Uncensored emphasizes steerability and transparent behavior, removing default safety and alignment layers typically found in mainstream assis...

### 301. xAI: Grok 4

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `x-ai/grok-4` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 256,000 tokens |
| **Entrada** | image, text, file |
| **Saída** | text |
| **Preço Input** | $0.000003 /M tokens |
| **Preço Output** | $0.000015 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Grok 4 is xAI's latest reasoning model with a 256k context window. It supports parallel tool calling, structured outputs, and both image and text inputs. Note that reasoning is not exposed, reasoning cannot be disabled, and the reasoning effort cannot be specified. Pricing increases once the total tokens in a given request is greater than 128k tokens. See more details on the [xAI docs](https://docs.x.ai/docs/models/grok-4-0709)

### 302. Tencent: Hunyuan A13B Instruct

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `tencent/hunyuan-a13b-instruct` |
| **Categoria** | Código / Programação |
| **Contexto** | 131,072 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000014 /M tokens |
| **Preço Output** | $0.00000057 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Hunyuan-A13B is a 13B active parameter Mixture-of-Experts (MoE) language model developed by Tencent, with a total parameter count of 80B and support for reasoning via Chain-of-Thought. It offers competitive benchmark performance across mathematics, science, coding, and multi-turn reasoning tasks, while maintaining high inference efficiency via Grouped Query Attention (GQA) and quantization support (FP8, GPTQ, etc.).

### 303. Morph: Morph V3 Large

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `morph/morph-v3-large` |
| **Categoria** | Código / Programação |
| **Contexto** | 262,144 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.0000009 /M tokens |
| **Preço Output** | $0.0000019 /M tokens |

**Descrição:** Morph's high-accuracy apply model for complex code edits. ~4,500 tokens/sec with 98% accuracy for precise code transformations.  The model requires the prompt to be in the following format:  <instruction>{instruction}</instruction> <code>{initial_code}</code> <update>{edit_snippet}</update>  Zero Data Retention is enabled for Morph. Learn more about this model in their [documentation](https://docs.morphllm.com/quickstart)

### 304. Morph: Morph V3 Fast

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `morph/morph-v3-fast` |
| **Categoria** | Código / Programação |
| **Contexto** | 81,920 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.0000008 /M tokens |
| **Preço Output** | $0.0000012 /M tokens |

**Descrição:** Morph's fastest apply model for code edits. ~10,500 tokens/sec with 96% accuracy for rapid code transformations.  The model requires the prompt to be in the following format:  <instruction>{instruction}</instruction> <code>{initial_code}</code> <update>{edit_snippet}</update>  Zero Data Retention is enabled for Morph. Learn more about this model in their [documentation](https://docs.morphllm.com/quickstart)

### 305. Baidu: ERNIE 4.5 VL 424B A47B 

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `baidu/ernie-4.5-vl-424b-a47b` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 123,000 tokens |
| **Entrada** | image, text |
| **Saída** | text |
| **Preço Input** | $0.00000042 /M tokens |
| **Preço Output** | $0.00000125 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** ERNIE-4.5-VL-424B-A47B is a multimodal Mixture-of-Experts (MoE) model from Baidu’s ERNIE 4.5 series, featuring 424B total parameters with 47B active per token. It is trained jointly on text and image data using a heterogeneous MoE architecture and modality-isolated routing to enable high-fidelity cross-modal reasoning, image understanding, and long-context generation (up to 131k tokens). Fine-tuned with techniques like SFT, DPO, UPO, and RLVR, this model supports both “thinking” and non-think...

### 306. Baidu: ERNIE 4.5 300B A47B 

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `baidu/ernie-4.5-300b-a47b` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 123,000 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000028 /M tokens |
| **Preço Output** | $0.0000011 /M tokens |

**Descrição:** ERNIE-4.5-300B-A47B is a 300B parameter Mixture-of-Experts (MoE) language model developed by Baidu as part of the ERNIE 4.5 series. It activates 47B parameters per token and supports text generation in both English and Chinese. Optimized for high-throughput inference and efficient scaling, it uses a heterogeneous MoE structure with advanced routing and quantization strategies, including FP8 and 2-bit formats. This version is fine-tuned for language-only tasks and supports reasoning, tool para...

### 307. Mistral: Mistral Small 3.2 24B

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `mistralai/mistral-small-3.2-24b-instruct` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 128,000 tokens |
| **Entrada** | image, text |
| **Saída** | text |
| **Preço Input** | $0.000000075 /M tokens |
| **Preço Output** | $0.0000002 /M tokens |

**Descrição:** Mistral-Small-3.2-24B-Instruct-2506 is an updated 24B parameter model from Mistral optimized for instruction following, repetition reduction, and improved function calling. Compared to the 3.1 release, version 3.2 significantly improves accuracy on WildBench and Arena Hard, reduces infinite generations, and delivers gains in tool use and structured output tasks.  It supports image and text inputs with structured outputs, function/tool calling, and strong performance across coding (HumanEval+,...

### 308. MiniMax: MiniMax M1

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `minimax/minimax-m1` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 1,000,000 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.0000004 /M tokens |
| **Preço Output** | $0.0000022 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** MiniMax-M1 is a large-scale, open-weight reasoning model designed for extended context and high-efficiency inference. It leverages a hybrid Mixture-of-Experts (MoE) architecture paired with a custom "lightning attention" mechanism, allowing it to process long sequences—up to 1 million tokens—while maintaining competitive FLOP efficiency. With 456 billion total parameters and 45.9B active per token, this variant is optimized for complex, multi-step reasoning tasks.  Trained via a custom reinfo...

### 309. Google: Gemini 2.5 Flash

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `google/gemini-2.5-flash` |
| **Categoria** | Multimodal (Vídeo) |
| **Contexto** | 1,048,576 tokens |
| **Entrada** | file, image, text, audio, video |
| **Saída** | text |
| **Preço Input** | $3e-7 /M tokens |
| **Preço Output** | $0.0000025 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Gemini 2.5 Flash is Google's state-of-the-art workhorse model, specifically designed for advanced reasoning, coding, mathematics, and scientific tasks. It includes built-in "thinking" capabilities, enabling it to provide responses with greater accuracy and nuanced context handling.   Additionally, Gemini 2.5 Flash is configurable through the "max tokens for reasoning" parameter, as described in the documentation (https://openrouter.ai/docs/use-cases/reasoning-tokens#max-tokens-for-reasoning).

### 310. Google: Gemini 2.5 Pro

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `google/gemini-2.5-pro` |
| **Categoria** | Multimodal (Vídeo) |
| **Contexto** | 1,048,576 tokens |
| **Entrada** | text, image, file, audio, video |
| **Saída** | text |
| **Preço Input** | $0.00000125 /M tokens |
| **Preço Output** | $0.00001 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Gemini 2.5 Pro is Google’s state-of-the-art AI model designed for advanced reasoning, coding, mathematics, and scientific tasks. It employs “thinking” capabilities, enabling it to reason through responses with enhanced accuracy and nuanced context handling. Gemini 2.5 Pro achieves top-tier performance on multiple benchmarks, including first-place positioning on the LMArena leaderboard, reflecting superior human-preference alignment and complex problem-solving abilities.

### 311. OpenAI: o3 Pro

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/o3-pro` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 200,000 tokens |
| **Entrada** | text, file, image |
| **Saída** | text |
| **Preço Input** | $0.00002 /M tokens |
| **Preço Output** | $0.00008 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** The o-series of models are trained with reinforcement learning to think before they answer and perform complex reasoning. The o3-pro model uses more compute to think harder and provide consistently better answers.  Note that BYOK is required for this model. Set up here: https://openrouter.ai/settings/integrations

### 312. xAI: Grok 3 Mini

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `x-ai/grok-3-mini` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 131,072 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.0000003 /M tokens |
| **Preço Output** | $0.0000005 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** A lightweight model that thinks before responding. Fast, smart, and great for logic-based tasks that do not require deep domain knowledge. The raw thinking traces are accessible.

### 313. xAI: Grok 3

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `x-ai/grok-3` |
| **Categoria** | Código / Programação |
| **Contexto** | 131,072 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.000003 /M tokens |
| **Preço Output** | $0.000015 /M tokens |

**Descrição:** Grok 3 is the latest model from xAI. It's their flagship model that excels at enterprise use cases like data extraction, coding, and text summarization. Possesses deep domain knowledge in finance, healthcare, law, and science.

### 314. Google: Gemini 2.5 Pro Preview 06-05

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `google/gemini-2.5-pro-preview` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 1,048,576 tokens |
| **Entrada** | file, image, text, audio |
| **Saída** | text |
| **Preço Input** | $0.00000125 /M tokens |
| **Preço Output** | $0.00001 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Gemini 2.5 Pro is Google’s state-of-the-art AI model designed for advanced reasoning, coding, mathematics, and scientific tasks. It employs “thinking” capabilities, enabling it to reason through responses with enhanced accuracy and nuanced context handling. Gemini 2.5 Pro achieves top-tier performance on multiple benchmarks, including first-place positioning on the LMArena leaderboard, reflecting superior human-preference alignment and complex problem-solving abilities.

### 315. DeepSeek: R1 0528

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `deepseek/deepseek-r1-0528` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 163,840 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.0000005 /M tokens |
| **Preço Output** | $0.00000215 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** May 28th update to the [original DeepSeek R1](/deepseek/deepseek-r1) Performance on par with [OpenAI o1](/openai/o1), but open-sourced and with fully open reasoning tokens. It's 671B parameters in size, with 37B active in an inference pass.  Fully open-source model.

### 316. Anthropic: Claude Opus 4

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `anthropic/claude-opus-4` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 200,000 tokens |
| **Entrada** | image, text, file |
| **Saída** | text |
| **Preço Input** | $0.000015 /M tokens |
| **Preço Output** | $0.000075 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Claude Opus 4 is benchmarked as the world’s best coding model, at time of release, bringing sustained performance on complex, long-running tasks and agent workflows. It sets new benchmarks in software engineering, achieving leading results on SWE-bench (72.5%) and Terminal-bench (43.2%). Opus 4 supports extended, agentic workflows, handling thousands of task steps continuously for hours without degradation.   Read more at the [blog post here](https://www.anthropic.com/news/claude-4)

### 317. Anthropic: Claude Sonnet 4

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `anthropic/claude-sonnet-4` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 1,000,000 tokens |
| **Entrada** | image, text, file |
| **Saída** | text |
| **Preço Input** | $0.000003 /M tokens |
| **Preço Output** | $0.000015 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Claude Sonnet 4 significantly enhances the capabilities of its predecessor, Sonnet 3.7, excelling in both coding and reasoning tasks with improved precision and controllability. Achieving state-of-the-art performance on SWE-bench (72.7%), Sonnet 4 balances capability and computational efficiency, making it suitable for a broad range of applications from routine coding tasks to complex software development projects. Key enhancements include improved autonomous codebase navigation, reduced erro...

### 318. Google: Gemma 3n 4B

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `google/gemma-3n-e4b-it` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 32,768 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000006 /M tokens |
| **Preço Output** | $0.00000012 /M tokens |

**Descrição:** Gemma 3n E4B-it is optimized for efficient execution on mobile and low-resource devices, such as phones, laptops, and tablets. It supports multimodal inputs—including text, visual data, and audio—enabling diverse tasks such as text generation, speech recognition, translation, and image analysis. Leveraging innovations like Per-Layer Embedding (PLE) caching and the MatFormer architecture, Gemma 3n dynamically manages memory usage and computational load by selectively activating model parameter...

### 319. Mistral: Mistral Medium 3

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `mistralai/mistral-medium-3` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 131,072 tokens |
| **Entrada** | text, image, file |
| **Saída** | text |
| **Preço Input** | $4e-7 /M tokens |
| **Preço Output** | $0.000002 /M tokens |

**Descrição:** Mistral Medium 3 is a high-performance enterprise-grade language model designed to deliver frontier-level capabilities at significantly reduced operational cost. It balances state-of-the-art reasoning and multimodal performance with 8× lower cost compared to traditional large models, making it suitable for scalable deployments across professional and industrial use cases.  The model excels in domains such as coding, STEM reasoning, and enterprise adaptation. It supports hybrid, on-prem, and i...

### 320. Google: Gemini 2.5 Pro Preview 05-06

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `google/gemini-2.5-pro-preview-05-06` |
| **Categoria** | Multimodal (Vídeo) |
| **Contexto** | 1,048,576 tokens |
| **Entrada** | text, image, file, audio, video |
| **Saída** | text |
| **Preço Input** | $0.00000125 /M tokens |
| **Preço Output** | $0.00001 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Gemini 2.5 Pro is Google’s state-of-the-art AI model designed for advanced reasoning, coding, mathematics, and scientific tasks. It employs “thinking” capabilities, enabling it to reason through responses with enhanced accuracy and nuanced context handling. Gemini 2.5 Pro achieves top-tier performance on multiple benchmarks, including first-place positioning on the LMArena leaderboard, reflecting superior human-preference alignment and complex problem-solving abilities.

### 321. Arcee AI: Spotlight

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `arcee-ai/spotlight` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 131,072 tokens |
| **Entrada** | image, text |
| **Saída** | text |
| **Preço Input** | $0.00000018 /M tokens |
| **Preço Output** | $0.00000018 /M tokens |

**Descrição:** Spotlight is a 7‑billion‑parameter vision‑language model derived from Qwen 2.5‑VL and fine‑tuned by Arcee AI for tight image‑text grounding tasks. It offers a 32 k‑token context window, enabling rich multimodal conversations that combine lengthy documents with one or more images. Training emphasized fast inference on consumer GPUs while retaining strong captioning, visual‐question‑answering, and diagram‑analysis accuracy. As a result, Spotlight slots neatly into agent workflows where screensh...

### 322. Arcee AI: Maestro Reasoning

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `arcee-ai/maestro-reasoning` |
| **Categoria** | Código / Programação |
| **Contexto** | 131,072 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.0000009 /M tokens |
| **Preço Output** | $0.0000033 /M tokens |

**Descrição:** Maestro Reasoning is Arcee's flagship analysis model: a 32 B‑parameter derivative of Qwen 2.5‑32 B tuned with DPO and chain‑of‑thought RL for step‑by‑step logic. Compared to the earlier 7 B preview, the production 32 B release widens the context window to 128 k tokens and doubles pass‑rate on MATH and GSM‑8K, while also lifting code completion accuracy. Its instruction style encourages structured "thought → answer" traces that can be parsed or hidden according to user preference. That transpa...

### 323. Arcee AI: Virtuoso Large

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `arcee-ai/virtuoso-large` |
| **Categoria** | Código / Programação |
| **Contexto** | 131,072 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000075 /M tokens |
| **Preço Output** | $0.0000012 /M tokens |

**Descrição:** Virtuoso‑Large is Arcee's top‑tier general‑purpose LLM at 72 B parameters, tuned to tackle cross‑domain reasoning, creative writing and enterprise QA. Unlike many 70 B peers, it retains the 128 k context inherited from Qwen 2.5, letting it ingest books, codebases or financial filings wholesale. Training blended DeepSeek R1 distillation, multi‑epoch supervised fine‑tuning and a final DPO/RLHF alignment stage, yielding strong performance on BIG‑Bench‑Hard, GSM‑8K and long‑context Needle‑In‑Hays...

### 324. Arcee AI: Coder Large

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `arcee-ai/coder-large` |
| **Categoria** | Código / Programação |
| **Contexto** | 32,768 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.0000005 /M tokens |
| **Preço Output** | $0.0000008 /M tokens |

**Descrição:** Coder‑Large is a 32 B‑parameter offspring of Qwen 2.5‑Instruct that has been further trained on permissively‑licensed GitHub, CodeSearchNet and synthetic bug‑fix corpora. It supports a 32k context window, enabling multi‑file refactoring or long diff review in a single call, and understands 30‑plus programming languages with special attention to TypeScript, Go and Terraform. Internal benchmarks show 5–8 pt gains over CodeLlama‑34 B‑Python on HumanEval and competitive BugFix scores thanks to a ...

### 325. Meta: Llama Guard 4 12B

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `meta-llama/llama-guard-4-12b` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 163,840 tokens |
| **Entrada** | image, text |
| **Saída** | text |
| **Preço Input** | $0.00000018 /M tokens |
| **Preço Output** | $0.00000018 /M tokens |

**Descrição:** Llama Guard 4 is a Llama 4 Scout-derived multimodal pretrained model, fine-tuned for content safety classification. Similar to previous versions, it can be used to classify content in both LLM inputs (prompt classification) and in LLM responses (response classification). It acts as an LLM—generating text in its output that indicates whether a given prompt or response is safe or unsafe, and if unsafe, it also lists the content categories violated.  Llama Guard 4 was aligned to safeguard agains...

### 326. Qwen: Qwen3 30B A3B

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `qwen/qwen3-30b-a3b` |
| **Categoria** | Código / Programação |
| **Contexto** | 40,960 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000009 /M tokens |
| **Preço Output** | $0.00000045 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Qwen3, the latest generation in the Qwen large language model series, features both dense and mixture-of-experts (MoE) architectures to excel in reasoning, multilingual support, and advanced agent tasks. Its unique ability to switch seamlessly between a thinking mode for complex reasoning and a non-thinking mode for efficient dialogue ensures versatile, high-quality performance.  Significantly outperforming prior models like QwQ and Qwen2.5, Qwen3 delivers superior mathematics, coding, common...

### 327. Qwen: Qwen3 8B

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `qwen/qwen3-8b` |
| **Categoria** | Código / Programação |
| **Contexto** | 40,960 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000005 /M tokens |
| **Preço Output** | $0.0000004 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Qwen3-8B is a dense 8.2B parameter causal language model from the Qwen3 series, designed for both reasoning-heavy tasks and efficient dialogue. It supports seamless switching between "thinking" mode for math, coding, and logical inference, and "non-thinking" mode for general conversation. The model is fine-tuned for instruction-following, agent integration, creative writing, and multilingual use across 100+ languages and dialects. It natively supports a 32K token context window and can extend...

### 328. Qwen: Qwen3 14B

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `qwen/qwen3-14b` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 40,960 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.0000001 /M tokens |
| **Preço Output** | $0.00000024 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Qwen3-14B is a dense 14.8B parameter causal language model from the Qwen3 series, designed for both complex reasoning and efficient dialogue. It supports seamless switching between a "thinking" mode for tasks like math, programming, and logical inference, and a "non-thinking" mode for general-purpose conversation. The model is fine-tuned for instruction-following, agent tool use, creative writing, and multilingual tasks across 100+ languages and dialects. It natively handles 32K token context...

### 329. Qwen: Qwen3 32B

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `qwen/qwen3-32b` |
| **Categoria** | Código / Programação |
| **Contexto** | 40,960 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000008 /M tokens |
| **Preço Output** | $0.00000028 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Qwen3-32B is a dense 32.8B parameter causal language model from the Qwen3 series, optimized for both complex reasoning and efficient dialogue. It supports seamless switching between a "thinking" mode for tasks like math, coding, and logical inference, and a "non-thinking" mode for faster, general-purpose conversation. The model demonstrates strong performance in instruction-following, agent tool use, creative writing, and multilingual tasks across 100+ languages and dialects. It natively hand...

### 330. Qwen: Qwen3 235B A22B

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `qwen/qwen3-235b-a22b` |
| **Categoria** | Código / Programação |
| **Contexto** | 131,072 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.000000455 /M tokens |
| **Preço Output** | $0.00000182 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Qwen3-235B-A22B is a 235B parameter mixture-of-experts (MoE) model developed by Qwen, activating 22B parameters per forward pass. It supports seamless switching between a "thinking" mode for complex reasoning, math, and code tasks, and a "non-thinking" mode for general conversational efficiency. The model demonstrates strong reasoning ability, multilingual support (100+ languages and dialects), advanced instruction-following, and agent tool-calling capabilities. It natively handles a 32K toke...

### 331. OpenAI: o4 Mini High

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/o4-mini-high` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 200,000 tokens |
| **Entrada** | image, text, file |
| **Saída** | text |
| **Preço Input** | $0.0000011 /M tokens |
| **Preço Output** | $0.0000044 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** OpenAI o4-mini-high is the same model as [o4-mini](/openai/o4-mini) with reasoning_effort set to high.   OpenAI o4-mini is a compact reasoning model in the o-series, optimized for fast, cost-efficient performance while retaining strong multimodal and agentic capabilities. It supports tool use and demonstrates competitive reasoning and coding performance across benchmarks like AIME (99.5% with Python) and SWE-bench, outperforming its predecessor o3-mini and even approaching o3 in some domains....

### 332. OpenAI: o3

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/o3` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 200,000 tokens |
| **Entrada** | image, text, file |
| **Saída** | text |
| **Preço Input** | $0.000002 /M tokens |
| **Preço Output** | $0.000008 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** o3 is a well-rounded and powerful model across domains. It sets a new standard for math, science, coding, and visual reasoning tasks. It also excels at technical writing and instruction-following. Use it to think through multi-step problems that involve analysis across text, code, and images.

### 333. OpenAI: o4 Mini

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/o4-mini` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 200,000 tokens |
| **Entrada** | image, text, file |
| **Saída** | text |
| **Preço Input** | $0.0000011 /M tokens |
| **Preço Output** | $0.0000044 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** OpenAI o4-mini is a compact reasoning model in the o-series, optimized for fast, cost-efficient performance while retaining strong multimodal and agentic capabilities. It supports tool use and demonstrates competitive reasoning and coding performance across benchmarks like AIME (99.5% with Python) and SWE-bench, outperforming its predecessor o3-mini and even approaching o3 in some domains.  Despite its smaller size, o4-mini exhibits high accuracy in STEM tasks, visual problem solving (e.g., M...

### 334. OpenAI: GPT-4.1

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/gpt-4.1` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 1,047,576 tokens |
| **Entrada** | image, text, file |
| **Saída** | text |
| **Preço Input** | $0.000002 /M tokens |
| **Preço Output** | $0.000008 /M tokens |

**Descrição:** GPT-4.1 is a flagship large language model optimized for advanced instruction following, real-world software engineering, and long-context reasoning. It supports a 1 million token context window and outperforms GPT-4o and GPT-4.5 across coding (54.6% SWE-bench Verified), instruction compliance (87.4% IFEval), and multimodal understanding benchmarks. It is tuned for precise code diffs, agent reliability, and high recall in large document contexts, making it ideal for agents, IDE tooling, and e...

### 335. OpenAI: GPT-4.1 Mini

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/gpt-4.1-mini` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 1,047,576 tokens |
| **Entrada** | image, text, file |
| **Saída** | text |
| **Preço Input** | $4e-7 /M tokens |
| **Preço Output** | $0.0000016 /M tokens |

**Descrição:** GPT-4.1 Mini is a mid-sized model delivering performance competitive with GPT-4o at substantially lower latency and cost. It retains a 1 million token context window and scores 45.1% on hard instruction evals, 35.8% on MultiChallenge, and 84.1% on IFEval. Mini also shows strong coding ability (e.g., 31.6% on Aider’s polyglot diff benchmark) and vision understanding, making it suitable for interactive applications with tight performance constraints.

### 336. OpenAI: GPT-4.1 Nano

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/gpt-4.1-nano` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 1,047,576 tokens |
| **Entrada** | image, text, file |
| **Saída** | text |
| **Preço Input** | $1e-7 /M tokens |
| **Preço Output** | $4e-7 /M tokens |

**Descrição:** For tasks that demand low latency, GPT‑4.1 nano is the fastest and cheapest model in the GPT-4.1 series. It delivers exceptional performance at a small size with its 1 million token context window, and scores 80.1% on MMLU, 50.3% on GPQA, and 9.8% on Aider polyglot coding – even higher than GPT‑4o mini. It’s ideal for tasks like classification or autocompletion.

### 337. AlfredPros: CodeLLaMa 7B Instruct Solidity

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `alfredpros/codellama-7b-instruct-solidity` |
| **Categoria** | Código / Programação |
| **Contexto** | 4,096 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $8e-7 /M tokens |
| **Preço Output** | $0.0000012 /M tokens |

**Descrição:** A finetuned 7 billion parameters Code LLaMA - Instruct model to generate Solidity smart contract using 4-bit QLoRA finetuning provided by PEFT library.

### 338. xAI: Grok 3 Mini Beta

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `x-ai/grok-3-mini-beta` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 131,072 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.0000003 /M tokens |
| **Preço Output** | $0.0000005 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Grok 3 Mini is a lightweight, smaller thinking model. Unlike traditional models that generate answers immediately, Grok 3 Mini thinks before responding. It’s ideal for reasoning-heavy tasks that don’t demand extensive domain knowledge, and shines in math-specific and quantitative use cases, such as solving challenging puzzles or math problems.  Transparent "thinking" traces accessible. Defaults to low reasoning, can boost with setting `reasoning: { effort: "high" }`  Note: That there are two ...

### 339. xAI: Grok 3 Beta

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `x-ai/grok-3-beta` |
| **Categoria** | Código / Programação |
| **Contexto** | 131,072 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.000003 /M tokens |
| **Preço Output** | $0.000015 /M tokens |

**Descrição:** Grok 3 is the latest model from xAI. It's their flagship model that excels at enterprise use cases like data extraction, coding, and text summarization. Possesses deep domain knowledge in finance, healthcare, law, and science.  Excels in structured tasks and benchmarks like GPQA, LCB, and MMLU-Pro where it outperforms Grok 3 Mini even on high thinking.   Note: That there are two xAI endpoints for this model. By default when using this model we will always route you to the base endpoint. If yo...

### 340. Meta: Llama 4 Maverick

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `meta-llama/llama-4-maverick` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 1,048,576 tokens |
| **Entrada** | text, image |
| **Saída** | text |
| **Preço Input** | $0.00000015 /M tokens |
| **Preço Output** | $0.0000006 /M tokens |

**Descrição:** Llama 4 Maverick 17B Instruct (128E) is a high-capacity multimodal language model from Meta, built on a mixture-of-experts (MoE) architecture with 128 experts and 17 billion active parameters per forward pass (400B total). It supports multilingual text and image input, and produces multilingual text and code output across 12 supported languages. Optimized for vision-language tasks, Maverick is instruction-tuned for assistant-like behavior, image reasoning, and general-purpose multimodal inter...

### 341. Meta: Llama 4 Scout

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `meta-llama/llama-4-scout` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 327,680 tokens |
| **Entrada** | text, image |
| **Saída** | text |
| **Preço Input** | $0.00000008 /M tokens |
| **Preço Output** | $0.0000003 /M tokens |

**Descrição:** Llama 4 Scout 17B Instruct (16E) is a mixture-of-experts (MoE) language model developed by Meta, activating 17 billion parameters out of a total of 109B. It supports native multimodal input (text and image) and multilingual output (text and code) across 12 supported languages. Designed for assistant-style interaction and visual reasoning, Scout uses 16 experts per forward pass and features a context length of 10 million tokens, with a training corpus of ~40 trillion tokens.  Built for high ef...

### 342. DeepSeek: DeepSeek V3 0324

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `deepseek/deepseek-chat-v3-0324` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 163,840 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.0000002 /M tokens |
| **Preço Output** | $0.00000077 /M tokens |

**Descrição:** DeepSeek V3, a 685B-parameter, mixture-of-experts model, is the latest iteration of the flagship chat model family from the DeepSeek team.  It succeeds the [DeepSeek V3](/deepseek/deepseek-chat-v3) model and performs really well on a variety of tasks.

### 343. OpenAI: o1-pro

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/o1-pro` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 200,000 tokens |
| **Entrada** | text, image, file |
| **Saída** | text |
| **Preço Input** | $0.00015 /M tokens |
| **Preço Output** | $0.0006 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** The o1 series of models are trained with reinforcement learning to think before they answer and perform complex reasoning. The o1-pro model uses more compute to think harder and provide consistently better answers.

### 344. Mistral: Mistral Small 3.1 24B

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `mistralai/mistral-small-3.1-24b-instruct` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 128,000 tokens |
| **Entrada** | text, image |
| **Saída** | text |
| **Preço Input** | $0.00000035 /M tokens |
| **Preço Output** | $0.00000056 /M tokens |

**Descrição:** Mistral Small 3.1 24B Instruct is an upgraded variant of Mistral Small 3 (2501), featuring 24 billion parameters with advanced multimodal capabilities. It provides state-of-the-art performance in text-based reasoning and vision tasks, including image analysis, programming, mathematical reasoning, and multilingual support across dozens of languages. Equipped with an extensive 128k token context window and optimized for efficient local inference, it supports use cases such as conversational age...

### 345. Google: Gemma 3 4B

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `google/gemma-3-4b-it` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 131,072 tokens |
| **Entrada** | text, image |
| **Saída** | text |
| **Preço Input** | $0.00000004 /M tokens |
| **Preço Output** | $0.00000008 /M tokens |

**Descrição:** Gemma 3 introduces multimodality, supporting vision-language input and text outputs. It handles context windows up to 128k tokens, understands over 140 languages, and offers improved math, reasoning, and chat capabilities, including structured outputs and function calling.

### 346. Google: Gemma 3 12B

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `google/gemma-3-12b-it` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 131,072 tokens |
| **Entrada** | text, image |
| **Saída** | text |
| **Preço Input** | $0.00000004 /M tokens |
| **Preço Output** | $0.00000013 /M tokens |

**Descrição:** Gemma 3 introduces multimodality, supporting vision-language input and text outputs. It handles context windows up to 128k tokens, understands over 140 languages, and offers improved math, reasoning, and chat capabilities, including structured outputs and function calling. Gemma 3 12B is the second largest in the family of Gemma 3 models after [Gemma 3 27B](google/gemma-3-27b-it)

### 347. Cohere: Command A

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `cohere/command-a` |
| **Categoria** | Código / Programação |
| **Contexto** | 256,000 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.0000025 /M tokens |
| **Preço Output** | $0.00001 /M tokens |

**Descrição:** Command A is an open-weights 111B parameter model with a 256k context window focused on delivering great performance across agentic, multilingual, and coding use cases. Compared to other leading proprietary and open-weights models Command A delivers maximum performance with minimum hardware costs, excelling on business-critical agentic and multilingual tasks.

### 348. OpenAI: GPT-4o-mini Search Preview

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/gpt-4o-mini-search-preview` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 128,000 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $1.5e-7 /M tokens |
| **Preço Output** | $6e-7 /M tokens |

**Descrição:** GPT-4o mini Search Preview is a specialized model for web search in Chat Completions. It is trained to understand and execute web search queries.

### 349. OpenAI: GPT-4o Search Preview

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/gpt-4o-search-preview` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 128,000 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.0000025 /M tokens |
| **Preço Output** | $0.00001 /M tokens |

**Descrição:** GPT-4o Search Previewis a specialized model for web search in Chat Completions. It is trained to understand and execute web search queries.

### 350. Reka Flash 3

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `rekaai/reka-flash-3` |
| **Categoria** | Código / Programação |
| **Contexto** | 65,536 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.0000001 /M tokens |
| **Preço Output** | $0.0000002 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Reka Flash 3 is a general-purpose, instruction-tuned large language model with 21 billion parameters, developed by Reka. It excels at general chat, coding tasks, instruction-following, and function calling. Featuring a 32K context length and optimized through reinforcement learning (RLOO), it provides competitive performance comparable to proprietary models within a smaller parameter footprint. Ideal for low-latency, local, or on-device deployments, Reka Flash 3 is compact, supports efficient...

### 351. Google: Gemma 3 27B

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `google/gemma-3-27b-it` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 131,072 tokens |
| **Entrada** | text, image |
| **Saída** | text |
| **Preço Input** | $0.00000008 /M tokens |
| **Preço Output** | $0.00000016 /M tokens |

**Descrição:** Gemma 3 introduces multimodality, supporting vision-language input and text outputs. It handles context windows up to 128k tokens, understands over 140 languages, and offers improved math, reasoning, and chat capabilities, including structured outputs and function calling. Gemma 3 27B is Google's latest open source model, successor to [Gemma 2](google/gemma-2-27b-it)

### 352. TheDrummer: Skyfall 36B V2

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `thedrummer/skyfall-36b-v2` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 32,768 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000055 /M tokens |
| **Preço Output** | $0.0000008 /M tokens |

**Descrição:** Skyfall 36B v2 is an enhanced iteration of Mistral Small 2501, specifically fine-tuned for improved creativity, nuanced writing, role-playing, and coherent storytelling.

### 353. Perplexity: Sonar Reasoning Pro

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `perplexity/sonar-reasoning-pro` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 128,000 tokens |
| **Entrada** | text, image |
| **Saída** | text |
| **Preço Input** | $0.000002 /M tokens |
| **Preço Output** | $0.000008 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Note: Sonar Pro pricing includes Perplexity search pricing. See [details here](https://docs.perplexity.ai/guides/pricing#detailed-pricing-breakdown-for-sonar-reasoning-pro-and-sonar-pro)  Sonar Reasoning Pro is a premier reasoning model powered by DeepSeek R1 with Chain of Thought (CoT). Designed for advanced use cases, it supports in-depth, multi-step queries with a larger context window and can surface more citations per search, enabling more comprehensive and extensible responses.

### 354. Perplexity: Sonar Pro

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `perplexity/sonar-pro` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 200,000 tokens |
| **Entrada** | text, image |
| **Saída** | text |
| **Preço Input** | $0.000003 /M tokens |
| **Preço Output** | $0.000015 /M tokens |

**Descrição:** Note: Sonar Pro pricing includes Perplexity search pricing. See [details here](https://docs.perplexity.ai/guides/pricing#detailed-pricing-breakdown-for-sonar-reasoning-pro-and-sonar-pro)  For enterprises seeking more advanced capabilities, the Sonar Pro API can handle in-depth, multi-step queries with added extensibility, like double the number of citations per search as Sonar on average. Plus, with a larger context window, it can handle longer and more nuanced searches and follow-up questions.

### 355. Perplexity: Sonar Deep Research

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `perplexity/sonar-deep-research` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 128,000 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.000002 /M tokens |
| **Preço Output** | $0.000008 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Sonar Deep Research is a research-focused model designed for multi-step retrieval, synthesis, and reasoning across complex topics. It autonomously searches, reads, and evaluates sources, refining its approach as it gathers information. This enables comprehensive report generation across domains like finance, technology, health, and current events.  Notes on Pricing ([Source](https://docs.perplexity.ai/guides/pricing#detailed-pricing-breakdown-for-sonar-deep-research))  - Input tokens comprise...

### 356. Google: Gemini 2.0 Flash Lite

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `google/gemini-2.0-flash-lite-001` |
| **Categoria** | Multimodal (Vídeo) |
| **Contexto** | 1,048,576 tokens |
| **Entrada** | text, image, file, audio, video |
| **Saída** | text |
| **Preço Input** | $7.5e-8 /M tokens |
| **Preço Output** | $3e-7 /M tokens |

**Descrição:** Gemini 2.0 Flash Lite offers a significantly faster time to first token (TTFT) compared to [Gemini Flash 1.5](/google/gemini-flash-1.5), while maintaining quality on par with larger models like [Gemini Pro 1.5](/google/gemini-pro-1.5), all at extremely economical token prices.

### 357. Mistral: Saba

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `mistralai/mistral-saba` |
| **Categoria** | Multimodal (Arquivo + Texto) |
| **Contexto** | 32,768 tokens |
| **Entrada** | text, file |
| **Saída** | text |
| **Preço Input** | $2e-7 /M tokens |
| **Preço Output** | $6e-7 /M tokens |

**Descrição:** Mistral Saba is a 24B-parameter language model specifically designed for the Middle East and South Asia, delivering accurate and contextually relevant responses while maintaining efficient performance. Trained on curated regional datasets, it supports multiple Indian-origin languages—including Tamil and Malayalam—alongside Arabic. This makes it a versatile option for a range of regional and multilingual applications. Read more at the blog post [here](https://mistral.ai/en/news/mistral-saba)

### 358. Llama Guard 3 8B

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `meta-llama/llama-guard-3-8b` |
| **Categoria** | Código / Programação |
| **Contexto** | 131,072 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000048 /M tokens |
| **Preço Output** | $0.00000003 /M tokens |

**Descrição:** Llama Guard 3 is a Llama-3.1-8B pretrained model, fine-tuned for content safety classification. Similar to previous versions, it can be used to classify content in both LLM inputs (prompt classification) and in LLM responses (response classification). It acts as an LLM – it generates text in its output that indicates whether a given prompt or response is safe or unsafe, and if unsafe, it also lists the content categories violated.  Llama Guard 3 was aligned to safeguard against the MLCommons ...

### 359. OpenAI: o3 Mini High

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/o3-mini-high` |
| **Categoria** | Multimodal (Arquivo + Texto) |
| **Contexto** | 200,000 tokens |
| **Entrada** | text, file |
| **Saída** | text |
| **Preço Input** | $0.0000011 /M tokens |
| **Preço Output** | $0.0000044 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** OpenAI o3-mini-high is the same model as [o3-mini](/openai/o3-mini) with reasoning_effort set to high.   o3-mini is a cost-efficient language model optimized for STEM reasoning tasks, particularly excelling in science, mathematics, and coding. The model features three adjustable reasoning effort levels and supports key developer capabilities including function calling, structured outputs, and streaming, though it does not include vision processing capabilities.  The model demonstrates signifi...

### 360. Google: Gemini 2.0 Flash

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `google/gemini-2.0-flash-001` |
| **Categoria** | Multimodal (Vídeo) |
| **Contexto** | 1,048,576 tokens |
| **Entrada** | text, image, file, audio, video |
| **Saída** | text |
| **Preço Input** | $1e-7 /M tokens |
| **Preço Output** | $4e-7 /M tokens |

**Descrição:** Gemini Flash 2.0 offers a significantly faster time to first token (TTFT) compared to [Gemini Flash 1.5](/google/gemini-flash-1.5), while maintaining quality on par with larger models like [Gemini Pro 1.5](/google/gemini-pro-1.5). It introduces notable enhancements in multimodal understanding, coding capabilities, complex instruction following, and function calling. These advancements come together to deliver more seamless and robust agentic experiences.

### 361. AionLabs: Aion-1.0

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `aion-labs/aion-1.0` |
| **Categoria** | Código / Programação |
| **Contexto** | 131,072 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.000004 /M tokens |
| **Preço Output** | $0.000008 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Aion-1.0 is a multi-model system designed for high performance across various tasks, including reasoning and coding. It is built on DeepSeek-R1, augmented with additional models and techniques such as Tree of Thoughts (ToT) and Mixture of Experts (MoE). It is Aion Lab's most powerful reasoning model.

### 362. AionLabs: Aion-1.0-Mini

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `aion-labs/aion-1.0-mini` |
| **Categoria** | Código / Programação |
| **Contexto** | 131,072 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $7e-7 /M tokens |
| **Preço Output** | $0.0000014 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** Aion-1.0-Mini 32B parameter model is a distilled version of the DeepSeek-R1 model, designed for strong performance in reasoning domains such as mathematics, coding, and logic. It is a modified variant of a FuseAI model that outperforms R1-Distill-Qwen-32B and R1-Distill-Llama-70B, with benchmark results available on its [Hugging Face page](https://huggingface.co/FuseAI/FuseO1-DeepSeekR1-QwQ-SkyT1-32B-Preview), independently replicated for verification.

### 363. AionLabs: Aion-RP 1.0 (8B)

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `aion-labs/aion-rp-llama-3.1-8b` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 32,768 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $8e-7 /M tokens |
| **Preço Output** | $0.0000016 /M tokens |

**Descrição:** Aion-RP-Llama-3.1-8B ranks the highest in the character evaluation portion of the RPBench-Auto benchmark, a roleplaying-specific variant of Arena-Hard-Auto, where LLMs evaluate each other’s responses. It is a fine-tuned base model rather than an instruct model, designed to produce more natural and varied writing.

### 364. Qwen: Qwen2.5 VL 72B Instruct

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `qwen/qwen2.5-vl-72b-instruct` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 32,000 tokens |
| **Entrada** | text, image |
| **Saída** | text |
| **Preço Input** | $0.00000025 /M tokens |
| **Preço Output** | $0.00000075 /M tokens |

**Descrição:** Qwen2.5-VL is proficient in recognizing common objects such as flowers, birds, fish, and insects. It is also highly capable of analyzing texts, charts, icons, graphics, and layouts within images.

### 365. Qwen: Qwen-Plus

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `qwen/qwen-plus` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 1,000,000 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000026 /M tokens |
| **Preço Output** | $0.00000078 /M tokens |

**Descrição:** Qwen-Plus, based on the Qwen2.5 foundation model, is a 131K context model with a balanced performance, speed, and cost combination.

### 366. OpenAI: o3 Mini

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/o3-mini` |
| **Categoria** | Multimodal (Arquivo + Texto) |
| **Contexto** | 200,000 tokens |
| **Entrada** | text, file |
| **Saída** | text |
| **Preço Input** | $0.0000011 /M tokens |
| **Preço Output** | $0.0000044 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** OpenAI o3-mini is a cost-efficient language model optimized for STEM reasoning tasks, particularly excelling in science, mathematics, and coding.  This model supports the `reasoning_effort` parameter, which can be set to "high", "medium", or "low" to control the thinking time of the model. The default is "medium". OpenRouter also offers the model slug `openai/o3-mini-high` to default the parameter to "high".  The model features three adjustable reasoning effort levels and supports key develop...

### 367. Mistral: Mistral Small 3

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `mistralai/mistral-small-24b-instruct-2501` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 32,768 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000005 /M tokens |
| **Preço Output** | $0.00000008 /M tokens |

**Descrição:** Mistral Small 3 is a 24B-parameter language model optimized for low-latency performance across common AI tasks. Released under the Apache 2.0 license, it features both pre-trained and instruction-tuned versions designed for efficient local deployment.  The model achieves 81% accuracy on the MMLU benchmark and performs competitively with larger models like Llama 3.3 70B and Qwen 32B, while operating at three times the speed on equivalent hardware. [Read the blog post about the model here.](htt...

### 368. DeepSeek: R1 Distill Qwen 32B

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `deepseek/deepseek-r1-distill-qwen-32b` |
| **Categoria** | Código / Programação |
| **Contexto** | 32,768 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $2.9e-7 /M tokens |
| **Preço Output** | $2.9e-7 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** DeepSeek R1 Distill Qwen 32B is a distilled large language model based on [Qwen 2.5 32B](https://huggingface.co/Qwen/Qwen2.5-32B), using outputs from [DeepSeek R1](/deepseek/deepseek-r1). It outperforms OpenAI's o1-mini across various benchmarks, achieving new state-of-the-art results for dense models.\n\nOther benchmark results include:\n\n- AIME 2024 pass@1: 72.6\n- MATH-500 pass@1: 94.3\n- CodeForces Rating: 1691\n\nThe model leverages fine-tuning from DeepSeek R1's outputs, enabling compe...

### 369. Perplexity: Sonar

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `perplexity/sonar` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 127,072 tokens |
| **Entrada** | text, image |
| **Saída** | text |
| **Preço Input** | $0.000001 /M tokens |
| **Preço Output** | $0.000001 /M tokens |

**Descrição:** Sonar is lightweight, affordable, fast, and simple to use — now featuring citations and the ability to customize sources. It is designed for companies seeking to integrate lightweight question-and-answer features optimized for speed.

### 370. DeepSeek: R1 Distill Llama 70B

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `deepseek/deepseek-r1-distill-llama-70b` |
| **Categoria** | Código / Programação |
| **Contexto** | 131,072 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.0000007 /M tokens |
| **Preço Output** | $0.0000008 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** DeepSeek R1 Distill Llama 70B is a distilled large language model based on [Llama-3.3-70B-Instruct](/meta-llama/llama-3.3-70b-instruct), using outputs from [DeepSeek R1](/deepseek/deepseek-r1). The model combines advanced distillation techniques to achieve high performance across multiple benchmarks, including:  - AIME 2024 pass@1: 70.0 - MATH-500 pass@1: 94.5 - CodeForces Rating: 1633  The model leverages fine-tuning from DeepSeek R1's outputs, enabling competitive performance comparable to ...

### 371. DeepSeek: R1

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `deepseek/deepseek-r1` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 64,000 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.0000007 /M tokens |
| **Preço Output** | $0.0000025 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** DeepSeek R1 is here: Performance on par with [OpenAI o1](/openai/o1), but open-sourced and with fully open reasoning tokens. It's 671B parameters in size, with 37B active in an inference pass.  Fully open-source model & [technical report](https://api-docs.deepseek.com/news/news250120).  MIT licensed: Distill & commercialize freely!

### 372. MiniMax: MiniMax-01

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `minimax/minimax-01` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 1,000,192 tokens |
| **Entrada** | text, image |
| **Saída** | text |
| **Preço Input** | $0.0000002 /M tokens |
| **Preço Output** | $0.0000011 /M tokens |

**Descrição:** MiniMax-01 is a combines MiniMax-Text-01 for text generation and MiniMax-VL-01 for image understanding. It has 456 billion parameters, with 45.9 billion parameters activated per inference, and can handle a context of up to 4 million tokens.  The text model adopts a hybrid architecture that combines Lightning Attention, Softmax Attention, and Mixture-of-Experts (MoE). The image model adopts the “ViT-MLP-LLM” framework and is trained on top of the text model.  To read more about the release, se...

### 373. Microsoft: Phi 4

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `microsoft/phi-4` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 16,384 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $6.5e-8 /M tokens |
| **Preço Output** | $1.4e-7 /M tokens |

**Descrição:** [Microsoft Research](/microsoft) Phi-4 is designed to perform well in complex reasoning tasks and can operate efficiently in situations with limited memory or where quick responses are needed.   At 14 billion parameters, it was trained on a mix of high-quality synthetic datasets, data from curated websites, and academic materials. It has undergone careful improvement to follow instructions accurately and maintain strong safety standards. It works best with English language inputs.  For more i...

### 374. Sao10K: Llama 3.1 70B Hanami x1

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `sao10k/l3.1-70b-hanami-x1` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 16,000 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.000003 /M tokens |
| **Preço Output** | $0.000003 /M tokens |

**Descrição:** This is [Sao10K](/sao10k)'s experiment over [Euryale v2.2](/sao10k/l3.1-euryale-70b).

### 375. DeepSeek: DeepSeek V3

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `deepseek/deepseek-chat` |
| **Categoria** | Código / Programação |
| **Contexto** | 163,840 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000032 /M tokens |
| **Preço Output** | $0.00000089 /M tokens |

**Descrição:** DeepSeek-V3 is the latest model from the DeepSeek team, building upon the instruction following and coding abilities of the previous versions. Pre-trained on nearly 15 trillion tokens, the reported evaluations reveal that the model outperforms other open-source models and rivals leading closed-source models.  For model details, please visit [the DeepSeek-V3 repo](https://github.com/deepseek-ai/DeepSeek-V3) for more information, or see the [launch announcement](https://api-docs.deepseek.com/ne...

### 376. Sao10K: Llama 3.3 Euryale 70B

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `sao10k/l3.3-euryale-70b` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 131,072 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $6.5e-7 /M tokens |
| **Preço Output** | $7.5e-7 /M tokens |

**Descrição:** Euryale L3.3 70B is a model focused on creative roleplay from [Sao10k](https://ko-fi.com/sao10k). It is the successor of [Euryale L3 70B v2.2](/models/sao10k/l3-euryale-70b).

### 377. OpenAI: o1

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/o1` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 200,000 tokens |
| **Entrada** | text, image, file |
| **Saída** | text |
| **Preço Input** | $0.000015 /M tokens |
| **Preço Output** | $0.00006 /M tokens |
| **Raciocínio** | Suportado |

**Descrição:** The latest and strongest model family from OpenAI, o1 is designed to spend more time thinking before responding. The o1 model series is trained with large-scale reinforcement learning to reason using chain of thought.   The o1 models are optimized for math, science, programming, and other STEM-related tasks. They consistently exhibit PhD-level accuracy on benchmarks in physics, chemistry, and biology. Learn more in the [launch announcement](https://openai.com/o1).

### 378. Cohere: Command R7B (12-2024)

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `cohere/command-r7b-12-2024` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 128,000 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.0000000375 /M tokens |
| **Preço Output** | $0.00000015 /M tokens |

**Descrição:** Command R7B (12-2024) is a small, fast update of the Command R+ model, delivered in December 2024. It excels at RAG, tool use, agents, and similar tasks requiring complex reasoning and multiple steps.  Use of this model is subject to Cohere's [Usage Policy](https://docs.cohere.com/docs/usage-policy) and [SaaS Agreement](https://cohere.com/saas-agreement).

### 379. Meta: Llama 3.3 70B Instruct (free)

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `meta-llama/llama-3.3-70b-instruct` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 65,536 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0 /M tokens |
| **Preço Output** | $0 /M tokens |
| **Gratuito** | Sim |

**Descrição:** The Meta Llama 3.3 multilingual large language model (LLM) is a pretrained and instruction tuned generative model in 70B (text in/text out). The Llama 3.3 instruction tuned text only model is optimized for multilingual dialogue use cases and outperforms many of the available open source and closed chat models on common industry benchmarks.  Supported languages: English, German, French, Italian, Portuguese, Hindi, Spanish, and Thai.  [Model Card](https://github.com/meta-llama/llama-models/blob...

### 380. Meta: Llama 3.3 70B Instruct

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `meta-llama/llama-3.3-70b-instruct` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 131,072 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.0000001 /M tokens |
| **Preço Output** | $0.00000032 /M tokens |

**Descrição:** The Meta Llama 3.3 multilingual large language model (LLM) is a pretrained and instruction tuned generative model in 70B (text in/text out). The Llama 3.3 instruction tuned text only model is optimized for multilingual dialogue use cases and outperforms many of the available open source and closed chat models on common industry benchmarks.  Supported languages: English, German, French, Italian, Portuguese, Hindi, Spanish, and Thai.  [Model Card](https://github.com/meta-llama/llama-models/blob...

### 381. Amazon: Nova Lite 1.0

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `amazon/nova-lite-v1` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 300,000 tokens |
| **Entrada** | text, image |
| **Saída** | text |
| **Preço Input** | $0.00000006 /M tokens |
| **Preço Output** | $0.00000024 /M tokens |

**Descrição:** Amazon Nova Lite 1.0 is a very low-cost multimodal model from Amazon that focused on fast processing of image, video, and text inputs to generate text output. Amazon Nova Lite can handle real-time customer interactions, document analysis, and visual question-answering tasks with high accuracy.  With an input context of 300K tokens, it can analyze multiple images or up to 30 minutes of video in a single input.

### 382. Amazon: Nova Micro 1.0

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `amazon/nova-micro-v1` |
| **Categoria** | Código / Programação |
| **Contexto** | 128,000 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.000000035 /M tokens |
| **Preço Output** | $0.00000014 /M tokens |

**Descrição:** Amazon Nova Micro 1.0 is a text-only model that delivers the lowest latency responses in the Amazon Nova family of models at a very low cost. With a context length of 128K tokens and optimized for speed and cost, Amazon Nova Micro excels at tasks such as text summarization, translation, content classification, interactive chat, and brainstorming. It has  simple mathematical reasoning and coding abilities.

### 383. Amazon: Nova Pro 1.0

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `amazon/nova-pro-v1` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 300,000 tokens |
| **Entrada** | text, image |
| **Saída** | text |
| **Preço Input** | $0.0000008 /M tokens |
| **Preço Output** | $0.0000032 /M tokens |

**Descrição:** Amazon Nova Pro 1.0 is a capable multimodal model from Amazon focused on providing a combination of accuracy, speed, and cost for a wide range of tasks. As of December 2024, it achieves state-of-the-art performance on key benchmarks including visual question answering (TextVQA) and video understanding (VATEX).  Amazon Nova Pro demonstrates strong capabilities in processing both visual and textual information and at analyzing financial documents.  **NOTE**: Video input is not supported at this...

### 384. OpenAI: GPT-4o (2024-11-20)

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/gpt-4o-2024-11-20` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 128,000 tokens |
| **Entrada** | text, image, file |
| **Saída** | text |
| **Preço Input** | $0.0000025 /M tokens |
| **Preço Output** | $0.00001 /M tokens |

**Descrição:** The 2024-11-20 version of GPT-4o offers a leveled-up creative writing ability with more natural, engaging, and tailored writing to improve relevance & readability. It’s also better at working with uploaded files, providing deeper insights & more thorough responses.  GPT-4o ("o" for "omni") is OpenAI's latest AI model, supporting both text and image inputs with text outputs. It maintains the intelligence level of [GPT-4 Turbo](/models/openai/gpt-4-turbo) while being twice as fast and 50% more ...

### 385. Mistral Large 2411

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `mistralai/mistral-large-2411` |
| **Categoria** | Multimodal (Arquivo + Texto) |
| **Contexto** | 131,072 tokens |
| **Entrada** | text, file |
| **Saída** | text |
| **Preço Input** | $0.000002 /M tokens |
| **Preço Output** | $0.000006 /M tokens |

**Descrição:** Mistral Large 2 2411 is an update of [Mistral Large 2](/mistralai/mistral-large) released together with [Pixtral Large 2411](/mistralai/pixtral-large-2411)  It provides a significant upgrade on the previous [Mistral Large 24.07](/mistralai/mistral-large-2407), with notable improvements in long context understanding, a new system prompt, and more accurate function calling.

### 386. Mistral Large 2407

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `mistralai/mistral-large-2407` |
| **Categoria** | Multimodal (Arquivo + Texto) |
| **Contexto** | 131,072 tokens |
| **Entrada** | text, file |
| **Saída** | text |
| **Preço Input** | $0.000002 /M tokens |
| **Preço Output** | $0.000006 /M tokens |

**Descrição:** This is Mistral AI's flagship model, Mistral Large 2 (version mistral-large-2407). It's a proprietary weights-available model and excels at reasoning, code, JSON, chat, and more. Read the launch announcement [here](https://mistral.ai/news/mistral-large-2407/).  It supports dozens of languages including French, German, Spanish, Italian, Portuguese, Arabic, Hindi, Russian, Chinese, Japanese, and Korean, along with 80+ coding languages including Python, Java, C, C++, JavaScript, and Bash. Its lo...

### 387. Mistral: Pixtral Large 2411

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `mistralai/pixtral-large-2411` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 131,072 tokens |
| **Entrada** | text, image, file |
| **Saída** | text |
| **Preço Input** | $0.000002 /M tokens |
| **Preço Output** | $0.000006 /M tokens |

**Descrição:** Pixtral Large is a 124B parameter, open-weight, multimodal model built on top of [Mistral Large 2](/mistralai/mistral-large-2411). The model is able to understand documents, charts and natural images.  The model is available under the Mistral Research License (MRL) for research and educational use, and the Mistral Commercial License for experimentation, testing, and production for commercial purposes.

### 388. Qwen2.5 Coder 32B Instruct

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `qwen/qwen-2.5-coder-32b-instruct` |
| **Categoria** | Código / Programação |
| **Contexto** | 32,768 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000066 /M tokens |
| **Preço Output** | $0.000001 /M tokens |

**Descrição:** Qwen2.5-Coder is the latest series of Code-Specific Qwen large language models (formerly known as CodeQwen). Qwen2.5-Coder brings the following improvements upon CodeQwen1.5:  - Significantly improvements in **code generation**, **code reasoning** and **code fixing**.  - A more comprehensive foundation for real-world applications such as **Code Agents**. Not only enhancing coding capabilities but also maintaining its strengths in mathematics and general competencies.  To read more about its e...

### 389. TheDrummer: UnslopNemo 12B

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `thedrummer/unslopnemo-12b` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 32,768 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $4e-7 /M tokens |
| **Preço Output** | $4e-7 /M tokens |

**Descrição:** UnslopNemo v4.1 is the latest addition from the creator of Rocinante, designed for adventure writing and role-play scenarios.

### 390. Anthropic: Claude 3.5 Haiku

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `anthropic/claude-3.5-haiku` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 200,000 tokens |
| **Entrada** | text, image |
| **Saída** | text |
| **Preço Input** | $8e-7 /M tokens |
| **Preço Output** | $0.000004 /M tokens |

**Descrição:** Claude 3.5 Haiku features offers enhanced capabilities in speed, coding accuracy, and tool use. Engineered to excel in real-time applications, it delivers quick response times that are essential for dynamic tasks such as chat interactions and immediate coding suggestions.  This makes it highly suitable for environments that demand both speed and precision, such as software development, customer service bots, and data management systems.  This model is currently pointing to [Claude 3.5 Haiku (...

### 391. Magnum v4 72B

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `anthracite-org/magnum-v4-72b` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 16,384 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.000003 /M tokens |
| **Preço Output** | $0.000005 /M tokens |

**Descrição:** This is a series of models designed to replicate the prose quality of the Claude 3 models, specifically Sonnet(https://openrouter.ai/anthropic/claude-3.5-sonnet) and Opus(https://openrouter.ai/anthropic/claude-3-opus).  The model is fine-tuned on top of [Qwen2.5 72B](https://openrouter.ai/qwen/qwen-2.5-72b-instruct).

### 392. Qwen: Qwen2.5 7B Instruct

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `qwen/qwen-2.5-7b-instruct` |
| **Categoria** | Código / Programação |
| **Contexto** | 32,768 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000004 /M tokens |
| **Preço Output** | $0.0000001 /M tokens |

**Descrição:** Qwen2.5 7B is the latest series of Qwen large language models. Qwen2.5 brings the following improvements upon Qwen2:  - Significantly more knowledge and has greatly improved capabilities in coding and mathematics, thanks to our specialized expert models in these domains.  - Significant improvements in instruction following, generating long texts (over 8K tokens), understanding structured data (e.g, tables), and generating structured outputs especially JSON. More resilient to the diversity of ...

### 393. Inflection: Inflection 3 Productivity

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `inflection/inflection-3-productivity` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 8,000 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $2.5e-6 /M tokens |
| **Preço Output** | $10e-6 /M tokens |

**Descrição:** Inflection 3 Productivity is optimized for following instructions. It is better for tasks requiring JSON output or precise adherence to provided guidelines. It has access to recent news.  For emotional intelligence similar to Pi, see [Inflect 3 Pi](/inflection/inflection-3-pi)  See [Inflection's announcement](https://inflection.ai/blog/enterprise) for more details.

### 394. Inflection: Inflection 3 Pi

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `inflection/inflection-3-pi` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 8,000 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $2.5e-6 /M tokens |
| **Preço Output** | $10e-6 /M tokens |

**Descrição:** Inflection 3 Pi powers Inflection's [Pi](https://pi.ai) chatbot, including backstory, emotional intelligence, productivity, and safety. It has access to recent news, and excels in scenarios like customer support and roleplay.  Pi has been trained to mirror your tone and style, if you use more emojis, so will Pi! Try experimenting with various prompts and conversation styles.

### 395. TheDrummer: Rocinante 12B

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `thedrummer/rocinante-12b` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 32,768 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $1.7e-7 /M tokens |
| **Preço Output** | $4.3e-7 /M tokens |

**Descrição:** Rocinante 12B is designed for engaging storytelling and rich prose.  Early testers have reported: - Expanded vocabulary with unique and expressive word choices - Enhanced creativity for vivid narratives - Adventure-filled and captivating stories

### 396. Meta: Llama 3.2 1B Instruct

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `meta-llama/llama-3.2-1b-instruct` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 60,000 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.000000027 /M tokens |
| **Preço Output** | $0.0000002 /M tokens |

**Descrição:** Llama 3.2 1B is a 1-billion-parameter language model focused on efficiently performing natural language tasks, such as summarization, dialogue, and multilingual text analysis. Its smaller size allows it to operate efficiently in low-resource environments while maintaining strong task performance.  Supporting eight core languages and fine-tunable for more, Llama 1.3B is ideal for businesses or developers seeking lightweight yet powerful AI solutions that can operate in diverse multilingual set...

### 397. Meta: Llama 3.2 3B Instruct (free)

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `meta-llama/llama-3.2-3b-instruct` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 131,072 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0 /M tokens |
| **Preço Output** | $0 /M tokens |
| **Gratuito** | Sim |

**Descrição:** Llama 3.2 3B is a 3-billion-parameter multilingual large language model, optimized for advanced natural language processing tasks like dialogue generation, reasoning, and summarization. Designed with the latest transformer architecture, it supports eight languages, including English, Spanish, and Hindi, and is adaptable for additional languages.  Trained on 9 trillion tokens, the Llama 3.2 3B model excels in instruction-following, complex reasoning, and tool use. Its balanced performance make...

### 398. Meta: Llama 3.2 3B Instruct

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `meta-llama/llama-3.2-3b-instruct` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 80,000 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.000000051 /M tokens |
| **Preço Output** | $0.00000034 /M tokens |

**Descrição:** Llama 3.2 3B is a 3-billion-parameter multilingual large language model, optimized for advanced natural language processing tasks like dialogue generation, reasoning, and summarization. Designed with the latest transformer architecture, it supports eight languages, including English, Spanish, and Hindi, and is adaptable for additional languages.  Trained on 9 trillion tokens, the Llama 3.2 3B model excels in instruction-following, complex reasoning, and tool use. Its balanced performance make...

### 399. Meta: Llama 3.2 11B Vision Instruct

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `meta-llama/llama-3.2-11b-vision-instruct` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 131,072 tokens |
| **Entrada** | text, image |
| **Saída** | text |
| **Preço Input** | $0.000000245 /M tokens |
| **Preço Output** | $0.000000245 /M tokens |

**Descrição:** Llama 3.2 11B Vision is a multimodal model with 11 billion parameters, designed to handle tasks combining visual and textual data. It excels in tasks such as image captioning and visual question answering, bridging the gap between language generation and visual reasoning. Pre-trained on a massive dataset of image-text pairs, it performs well in complex, high-accuracy image analysis.  Its ability to integrate visual understanding with language processing makes it an ideal solution for industri...

### 400. Qwen2.5 72B Instruct

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `qwen/qwen-2.5-72b-instruct` |
| **Categoria** | Código / Programação |
| **Contexto** | 32,768 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000036 /M tokens |
| **Preço Output** | $0.0000004 /M tokens |

**Descrição:** Qwen2.5 72B is the latest series of Qwen large language models. Qwen2.5 brings the following improvements upon Qwen2:  - Significantly more knowledge and has greatly improved capabilities in coding and mathematics, thanks to our specialized expert models in these domains.  - Significant improvements in instruction following, generating long texts (over 8K tokens), understanding structured data (e.g, tables), and generating structured outputs especially JSON. More resilient to the diversity of...

### 401. Cohere: Command R+ (08-2024)

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `cohere/command-r-plus-08-2024` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 128,000 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.0000025 /M tokens |
| **Preço Output** | $0.00001 /M tokens |

**Descrição:** command-r-plus-08-2024 is an update of the [Command R+](/models/cohere/command-r-plus) with roughly 50% higher throughput and 25% lower latencies as compared to the previous Command R+ version, while keeping the hardware footprint the same.  Read the launch post [here](https://docs.cohere.com/changelog/command-gets-refreshed).  Use of this model is subject to Cohere's [Usage Policy](https://docs.cohere.com/docs/usage-policy) and [SaaS Agreement](https://cohere.com/saas-agreement).

### 402. Cohere: Command R (08-2024)

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `cohere/command-r-08-2024` |
| **Categoria** | Código / Programação |
| **Contexto** | 128,000 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000015 /M tokens |
| **Preço Output** | $0.0000006 /M tokens |

**Descrição:** command-r-08-2024 is an update of the [Command R](/models/cohere/command-r) with improved performance for multilingual retrieval-augmented generation (RAG) and tool use. More broadly, it is better at math, code and reasoning and is competitive with the previous version of the larger Command R+ model.  Read the launch post [here](https://docs.cohere.com/changelog/command-gets-refreshed).  Use of this model is subject to Cohere's [Usage Policy](https://docs.cohere.com/docs/usage-policy) and [Sa...

### 403. Sao10K: Llama 3.1 Euryale 70B v2.2

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `sao10k/l3.1-euryale-70b` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 131,072 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000085 /M tokens |
| **Preço Output** | $0.00000085 /M tokens |

**Descrição:** Euryale L3.1 70B v2.2 is a model focused on creative roleplay from [Sao10k](https://ko-fi.com/sao10k). It is the successor of [Euryale L3 70B v2.1](/models/sao10k/l3-euryale-70b).

### 404. Nous: Hermes 3 70B Instruct

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `nousresearch/hermes-3-llama-3.1-70b` |
| **Categoria** | Código / Programação |
| **Contexto** | 131,072 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.0000003 /M tokens |
| **Preço Output** | $0.0000003 /M tokens |

**Descrição:** Hermes 3 is a generalist language model with many improvements over [Hermes 2](/models/nousresearch/nous-hermes-2-mistral-7b-dpo), including advanced agentic capabilities, much better roleplaying, reasoning, multi-turn conversation, long context coherence, and improvements across the board.  Hermes 3 70B is a competitive, if not superior finetune of the [Llama-3.1 70B foundation model](/models/meta-llama/llama-3.1-70b-instruct), focused on aligning LLMs to the user, with powerful steering cap...

### 405. Nous: Hermes 3 405B Instruct (free)

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `nousresearch/hermes-3-llama-3.1-405b` |
| **Categoria** | Código / Programação |
| **Contexto** | 131,072 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0 /M tokens |
| **Preço Output** | $0 /M tokens |
| **Gratuito** | Sim |

**Descrição:** Hermes 3 is a generalist language model with many improvements over Hermes 2, including advanced agentic capabilities, much better roleplaying, reasoning, multi-turn conversation, long context coherence, and improvements across the board.  Hermes 3 405B is a frontier-level, full-parameter finetune of the Llama-3.1 405B foundation model, focused on aligning LLMs to the user, with powerful steering capabilities and control given to the end user.  The Hermes 3 series builds and expands on the He...

### 406. Nous: Hermes 3 405B Instruct

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `nousresearch/hermes-3-llama-3.1-405b` |
| **Categoria** | Código / Programação |
| **Contexto** | 131,072 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.000001 /M tokens |
| **Preço Output** | $0.000001 /M tokens |

**Descrição:** Hermes 3 is a generalist language model with many improvements over Hermes 2, including advanced agentic capabilities, much better roleplaying, reasoning, multi-turn conversation, long context coherence, and improvements across the board.  Hermes 3 405B is a frontier-level, full-parameter finetune of the Llama-3.1 405B foundation model, focused on aligning LLMs to the user, with powerful steering capabilities and control given to the end user.  The Hermes 3 series builds and expands on the He...

### 407. Sao10K: Llama 3 8B Lunaris

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `sao10k/l3-lunaris-8b` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 8,192 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000004 /M tokens |
| **Preço Output** | $0.00000005 /M tokens |

**Descrição:** Lunaris 8B is a versatile generalist and roleplaying model based on Llama 3. It's a strategic merge of multiple models, designed to balance creativity with improved logic and general knowledge.  Created by [Sao10k](https://huggingface.co/Sao10k), this model aims to offer an improved experience over Stheno v3.2, with enhanced creativity and logical reasoning.  For best results, use with Llama 3 Instruct context template, temperature 1.4, and min_p 0.1.

### 408. OpenAI: GPT-4o (2024-08-06)

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/gpt-4o-2024-08-06` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 128,000 tokens |
| **Entrada** | text, image, file |
| **Saída** | text |
| **Preço Input** | $0.0000025 /M tokens |
| **Preço Output** | $0.00001 /M tokens |

**Descrição:** The 2024-08-06 version of GPT-4o offers improved performance in structured outputs, with the ability to supply a JSON schema in the respone_format. Read more [here](https://openai.com/index/introducing-structured-outputs-in-the-api/).  GPT-4o ("o" for "omni") is OpenAI's latest AI model, supporting both text and image inputs with text outputs. It maintains the intelligence level of [GPT-4 Turbo](/models/openai/gpt-4-turbo) while being twice as fast and 50% more cost-effective. GPT-4o also off...

### 409. Meta: Llama 3.1 70B Instruct

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `meta-llama/llama-3.1-70b-instruct` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 131,072 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.0000004 /M tokens |
| **Preço Output** | $0.0000004 /M tokens |

**Descrição:** Meta's latest class of model (Llama 3.1) launched with a variety of sizes & flavors. This 70B instruct-tuned version is optimized for high quality dialogue usecases.  It has demonstrated strong performance compared to leading closed-source models in human evaluations.  To read more about the model release, [click here](https://ai.meta.com/blog/meta-llama-3-1/). Usage of this model is subject to [Meta's Acceptable Use Policy](https://llama.meta.com/llama3/use-policy/).

### 410. Meta: Llama 3.1 8B Instruct

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `meta-llama/llama-3.1-8b-instruct` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 16,384 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000002 /M tokens |
| **Preço Output** | $0.00000005 /M tokens |

**Descrição:** Meta's latest class of model (Llama 3.1) launched with a variety of sizes & flavors. This 8B instruct-tuned version is fast and efficient.  It has demonstrated strong performance compared to leading closed-source models in human evaluations.  To read more about the model release, [click here](https://ai.meta.com/blog/meta-llama-3-1/). Usage of this model is subject to [Meta's Acceptable Use Policy](https://llama.meta.com/llama3/use-policy/).

### 411. Mistral: Mistral Nemo

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `mistralai/mistral-nemo` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 131,072 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000002 /M tokens |
| **Preço Output** | $0.00000003 /M tokens |

**Descrição:** A 12B parameter model with a 128k token context length built by Mistral in collaboration with NVIDIA.  The model is multilingual, supporting English, French, German, Spanish, Italian, Portuguese, Chinese, Japanese, Korean, Arabic, and Hindi.  It supports function calling and is released under the Apache 2.0 license.

### 412. OpenAI: GPT-4o-mini (2024-07-18)

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/gpt-4o-mini-2024-07-18` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 128,000 tokens |
| **Entrada** | text, image, file |
| **Saída** | text |
| **Preço Input** | $1.5e-7 /M tokens |
| **Preço Output** | $6e-7 /M tokens |

**Descrição:** GPT-4o mini is OpenAI's newest model after [GPT-4 Omni](/models/openai/gpt-4o), supporting both text and image inputs with text outputs.  As their most advanced small model, it is many multiples more affordable than other recent frontier models, and more than 60% cheaper than [GPT-3.5 Turbo](/models/openai/gpt-3.5-turbo). It maintains SOTA intelligence, while being significantly more cost-effective.  GPT-4o mini achieves an 82% score on MMLU and presently ranks higher than GPT-4 on chat prefe...

### 413. OpenAI: GPT-4o-mini

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/gpt-4o-mini` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 128,000 tokens |
| **Entrada** | text, image, file |
| **Saída** | text |
| **Preço Input** | $1.5e-7 /M tokens |
| **Preço Output** | $6e-7 /M tokens |

**Descrição:** GPT-4o mini is OpenAI's newest model after [GPT-4 Omni](/models/openai/gpt-4o), supporting both text and image inputs with text outputs.  As their most advanced small model, it is many multiples more affordable than other recent frontier models, and more than 60% cheaper than [GPT-3.5 Turbo](/models/openai/gpt-3.5-turbo). It maintains SOTA intelligence, while being significantly more cost-effective.  GPT-4o mini achieves an 82% score on MMLU and presently ranks higher than GPT-4 on chat prefe...

### 414. Google: Gemma 2 27B

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `google/gemma-2-27b-it` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 8,192 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $6.5e-7 /M tokens |
| **Preço Output** | $6.5e-7 /M tokens |

**Descrição:** Gemma 2 27B by Google is an open model built from the same research and technology used to create the [Gemini models](/models?q=gemini).  Gemma models are well-suited for a variety of text generation tasks, including question answering, summarization, and reasoning.  See the [launch announcement](https://blog.google/technology/developers/google-gemma-2/) for more details. Usage of Gemma is subject to Google's [Gemma Terms of Use](https://ai.google.dev/gemma/terms).

### 415. Sao10k: Llama 3 Euryale 70B v2.1

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `sao10k/l3-euryale-70b` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 8,192 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000148 /M tokens |
| **Preço Output** | $0.00000148 /M tokens |

**Descrição:** Euryale 70B v2.1 is a model focused on creative roleplay from [Sao10k](https://ko-fi.com/sao10k).  - Better prompt adherence. - Better anatomy / spatial awareness. - Adapts much better to unique and custom formatting / reply formats. - Very creative, lots of unique swipes. - Is not restrictive during roleplays.

### 416. NousResearch: Hermes 2 Pro - Llama-3 8B

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `nousresearch/hermes-2-pro-llama-3-8b` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 8,192 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000014 /M tokens |
| **Preço Output** | $0.00000014 /M tokens |

**Descrição:** Hermes 2 Pro is an upgraded, retrained version of Nous Hermes 2, consisting of an updated and cleaned version of the OpenHermes 2.5 Dataset, as well as a newly introduced Function Calling and JSON Mode dataset developed in-house.

### 417. OpenAI: GPT-4o

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/gpt-4o` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 128,000 tokens |
| **Entrada** | text, image, file |
| **Saída** | text |
| **Preço Input** | $0.0000025 /M tokens |
| **Preço Output** | $0.00001 /M tokens |

**Descrição:** GPT-4o ("o" for "omni") is OpenAI's latest AI model, supporting both text and image inputs with text outputs. It maintains the intelligence level of [GPT-4 Turbo](/models/openai/gpt-4-turbo) while being twice as fast and 50% more cost-effective. GPT-4o also offers improved performance in processing non-English languages and enhanced visual capabilities.  For benchmarking against other models, it was briefly called ["im-also-a-good-gpt2-chatbot"](https://twitter.com/LiamFedus/status/1790064963...

### 418. OpenAI: GPT-4o (2024-05-13)

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/gpt-4o-2024-05-13` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 128,000 tokens |
| **Entrada** | text, image, file |
| **Saída** | text |
| **Preço Input** | $0.000005 /M tokens |
| **Preço Output** | $0.000015 /M tokens |

**Descrição:** GPT-4o ("o" for "omni") is OpenAI's latest AI model, supporting both text and image inputs with text outputs. It maintains the intelligence level of [GPT-4 Turbo](/models/openai/gpt-4-turbo) while being twice as fast and 50% more cost-effective. GPT-4o also offers improved performance in processing non-English languages and enhanced visual capabilities.  For benchmarking against other models, it was briefly called ["im-also-a-good-gpt2-chatbot"](https://twitter.com/LiamFedus/status/1790064963...

### 419. Meta: Llama 3 8B Instruct

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `meta-llama/llama-3-8b-instruct` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 8,192 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000004 /M tokens |
| **Preço Output** | $0.00000004 /M tokens |

**Descrição:** Meta's latest class of model (Llama 3) launched with a variety of sizes & flavors. This 8B instruct-tuned version was optimized for high quality dialogue usecases.  It has demonstrated strong performance compared to leading closed-source models in human evaluations.  To read more about the model release, [click here](https://ai.meta.com/blog/meta-llama-3/). Usage of this model is subject to [Meta's Acceptable Use Policy](https://llama.meta.com/llama3/use-policy/).

### 420. Meta: Llama 3 70B Instruct

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `meta-llama/llama-3-70b-instruct` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 8,192 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000051 /M tokens |
| **Preço Output** | $0.00000074 /M tokens |

**Descrição:** Meta's latest class of model (Llama 3) launched with a variety of sizes & flavors. This 70B instruct-tuned version was optimized for high quality dialogue usecases.  It has demonstrated strong performance compared to leading closed-source models in human evaluations.  To read more about the model release, [click here](https://ai.meta.com/blog/meta-llama-3/). Usage of this model is subject to [Meta's Acceptable Use Policy](https://llama.meta.com/llama3/use-policy/).

### 421. Mistral: Mixtral 8x22B Instruct

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `mistralai/mixtral-8x22b-instruct` |
| **Categoria** | Multimodal (Arquivo + Texto) |
| **Contexto** | 65,536 tokens |
| **Entrada** | text, file |
| **Saída** | text |
| **Preço Input** | $0.000002 /M tokens |
| **Preço Output** | $0.000006 /M tokens |

**Descrição:** Mistral's official instruct fine-tuned version of [Mixtral 8x22B](/models/mistralai/mixtral-8x22b). It uses 39B active parameters out of 141B, offering unparalleled cost efficiency for its size. Its strengths include: - strong math, coding, and reasoning - large context length (64k) - fluency in English, French, Italian, German, and Spanish  See benchmarks on the launch announcement [here](https://mistral.ai/news/mixtral-8x22b/). #moe

### 422. WizardLM-2 8x22B

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `microsoft/wizardlm-2-8x22b` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 65,535 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000062 /M tokens |
| **Preço Output** | $0.00000062 /M tokens |

**Descrição:** WizardLM-2 8x22B is Microsoft AI's most advanced Wizard model. It demonstrates highly competitive performance compared to leading proprietary models, and it consistently outperforms all existing state-of-the-art opensource models.  It is an instruct finetune of [Mixtral 8x22B](/models/mistralai/mixtral-8x22b).  To read more about the model release, [click here](https://wizardlm.github.io/WizardLM2/).  #moe

### 423. OpenAI: GPT-4 Turbo

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/gpt-4-turbo` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 128,000 tokens |
| **Entrada** | text, image |
| **Saída** | text |
| **Preço Input** | $0.00001 /M tokens |
| **Preço Output** | $0.00003 /M tokens |

**Descrição:** The latest GPT-4 Turbo model with vision capabilities. Vision requests can now use JSON mode and function calling.  Training data: up to December 2023.

### 424. Anthropic: Claude 3 Haiku

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `anthropic/claude-3-haiku` |
| **Categoria** | Multimodal (Visão + Texto) |
| **Contexto** | 200,000 tokens |
| **Entrada** | text, image |
| **Saída** | text |
| **Preço Input** | $0.25e-6 /M tokens |
| **Preço Output** | $1.25e-6 /M tokens |

**Descrição:** Claude 3 Haiku is Anthropic's fastest and most compact model for near-instant responsiveness. Quick and accurate targeted performance.  See the launch announcement and benchmark results [here](https://www.anthropic.com/news/claude-3-haiku)  #multimodal

### 425. Mistral Large

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `mistralai/mistral-large` |
| **Categoria** | Multimodal (Arquivo + Texto) |
| **Contexto** | 128,000 tokens |
| **Entrada** | text, file |
| **Saída** | text |
| **Preço Input** | $0.000002 /M tokens |
| **Preço Output** | $0.000006 /M tokens |

**Descrição:** This is Mistral AI's flagship model, Mistral Large 2 (version `mistral-large-2407`). It's a proprietary weights-available model and excels at reasoning, code, JSON, chat, and more. Read the launch announcement [here](https://mistral.ai/news/mistral-large-2407/).  It supports dozens of languages including French, German, Spanish, Italian, Portuguese, Arabic, Hindi, Russian, Chinese, Japanese, and Korean, along with 80+ coding languages including Python, Java, C, C++, JavaScript, and Bash. Its ...

### 426. OpenAI: GPT-3.5 Turbo (older v0613)

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/gpt-3.5-turbo-0613` |
| **Categoria** | Código / Programação |
| **Contexto** | 4,095 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.000001 /M tokens |
| **Preço Output** | $0.000002 /M tokens |

**Descrição:** GPT-3.5 Turbo is OpenAI's fastest model. It can understand and generate natural language or code, and is optimized for chat and traditional completion tasks.  Training data up to Sep 2021.

### 427. OpenAI: GPT-4 Turbo Preview

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/gpt-4-turbo-preview` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 128,000 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00001 /M tokens |
| **Preço Output** | $0.00003 /M tokens |

**Descrição:** The preview GPT-4 model with improved instruction following, JSON mode, reproducible outputs, parallel function calling, and more. Training data: up to Dec 2023.  **Note:** heavily rate limited by OpenAI while in preview.

### 428. OpenAI: GPT-4 Turbo (older v1106)

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/gpt-4-1106-preview` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 128,000 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00001 /M tokens |
| **Preço Output** | $0.00003 /M tokens |

**Descrição:** The latest GPT-4 Turbo model with vision capabilities. Vision requests can now use JSON mode and function calling.  Training data: up to April 2023.

### 429. Mistral: Mistral 7B Instruct v0.1

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `mistralai/mistral-7b-instruct-v0.1` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 2,824 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000011 /M tokens |
| **Preço Output** | $0.00000019 /M tokens |

**Descrição:** A 7.3B parameter model that outperforms Llama 2 13B on all benchmarks, with optimizations for speed and context length.

### 430. OpenAI: GPT-3.5 Turbo Instruct

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/gpt-3.5-turbo-instruct` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 4,095 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.0000015 /M tokens |
| **Preço Output** | $0.000002 /M tokens |

**Descrição:** This model is a variant of GPT-3.5 Turbo tuned for instructional prompts and omitting chat-related optimizations. Training data: up to Sep 2021.

### 431. OpenAI: GPT-3.5 Turbo 16k

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/gpt-3.5-turbo-16k` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 16,385 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.000003 /M tokens |
| **Preço Output** | $0.000004 /M tokens |

**Descrição:** This model offers four times the context length of gpt-3.5-turbo, allowing it to support approximately 20 pages of text in a single request at a higher cost. Training data: up to Sep 2021.

### 432. Mancer: Weaver (alpha)

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `mancer/weaver` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 8,000 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00000075 /M tokens |
| **Preço Output** | $0.000001 /M tokens |

**Descrição:** An attempt to recreate Claude-style verbosity, but don't expect the same level of coherence or memory. Meant for use in roleplay/narrative situations.

### 433. ReMM SLERP 13B

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `undi95/remm-slerp-l2-13b` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 6,144 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $4.5e-7 /M tokens |
| **Preço Output** | $6.5e-7 /M tokens |

**Descrição:** A recreation trial of the original MythoMax-L2-B13 but with updated models. #merge

### 434. MythoMax 13B

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `gryphe/mythomax-l2-13b` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 4,096 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $6e-8 /M tokens |
| **Preço Output** | $6e-8 /M tokens |

**Descrição:** One of the highest performing and most popular fine-tunes of Llama 2 13B, with rich descriptions and roleplay. #merge

### 435. OpenAI: GPT-4 (older v0314)

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/gpt-4-0314` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 8,191 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00003 /M tokens |
| **Preço Output** | $0.00006 /M tokens |

**Descrição:** GPT-4-0314 is the first version of GPT-4 released, with a context length of 8,192 tokens, and was supported until June 14. Training data: up to Sep 2021.

### 436. OpenAI: GPT-4

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/gpt-4` |
| **Categoria** | Texto (LLM) |
| **Contexto** | 8,191 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $0.00003 /M tokens |
| **Preço Output** | $0.00006 /M tokens |

**Descrição:** OpenAI's flagship model, GPT-4 is a large-scale multimodal language model capable of solving difficult problems with greater accuracy than previous models due to its broader general knowledge and advanced reasoning capabilities. Training data: up to Sep 2021.

### 437. OpenAI: GPT-3.5 Turbo

| Campo | Valor |
|-------|-------|
| **Slug (ID)** | `openai/gpt-3.5-turbo` |
| **Categoria** | Código / Programação |
| **Contexto** | 16,385 tokens |
| **Entrada** | text |
| **Saída** | text |
| **Preço Input** | $5e-7 /M tokens |
| **Preço Output** | $0.0000015 /M tokens |

**Descrição:** GPT-3.5 Turbo is OpenAI's fastest model. It can understand and generate natural language or code, and is optimized for chat and traditional completion tasks.  Training data up to Sep 2021.

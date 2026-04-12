---
title: PRISMA Assistant
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
---

# PRISMA Assistant

Asistente inteligente para revisiones sistemáticas de literatura (SLR) usando el protocolo PRISMA.

## Variables de entorno requeridas (configurar en Settings → Secrets)

| Variable | Descripción |
|---|---|
| `GEMINI_API_KEY` | Google Gemini API key |
| `OPENROUTER_API_KEY` | OpenRouter API key (modelos gratuitos) |
| `CEREBRAS_API_KEY` | Cerebras Cloud API key |
| `GROQ_API_KEY` | Groq Cloud API key |
| `GITHUB_TOKEN` | GitHub Models token |
| `SEMANTIC_SCHOLAR_API_KEY` | Semantic Scholar API key |
| `HUGGINGFACE_API_KEY` | HuggingFace API key |
| `DEEPL_API_KEY` | DeepL API key |
| `MONGODB_URI` | MongoDB Atlas URI (opcional) |
| `MILVUS_COLLECTION` | `articles_collection_v4` |

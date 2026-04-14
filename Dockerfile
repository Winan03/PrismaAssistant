FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Puerto por defecto: 8001 para VPS. Sobreescribir con PORT=7860 para HF Spaces.
ENV PORT=8001

WORKDIR /app

# Dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git curl \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-descargar el modelo de embeddings durante el BUILD (~90MB vs 420MB del mpnet)
# v17.3: all-MiniLM-L6-v2 (22M params, 384-dim) — 5x más rápido que all-mpnet-base-v2
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2'); print('Modelo all-MiniLM-L6-v2 descargado')"

# Copiar codigo de la app
COPY . .

# Crear directorios necesarios
RUN mkdir -p chroma_db logs sessions

EXPOSE ${PORT}

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
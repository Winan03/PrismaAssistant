# Usar Python 3.12 versión ligera
FROM python:3.12-slim

# Evitar que Python genere archivos .pyc y forzar salida de logs inmediata
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema necesarias para algunas librerías de Python
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código
COPY . .

# Exponer puerto 8080 (Solo informativo)
EXPOSE 8080

# COMANDO DE INICIO (Corregido y Simplificado)
# Usamos formato de lista ["..."] que es más seguro
# Forzamos el puerto 8080 y el host 0.0.0.0
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--loop", "asyncio"]
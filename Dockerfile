# Usar una imagen ligera de Python
FROM python:3.12-slim

# Establecer directorio de trabajo
WORKDIR /app

# Copiar los requerimientos primero (para aprovechar caché)
COPY requirements.txt .

# Instalar dependencias
# --no-cache-dir ayuda a que la imagen sea más ligera
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código
COPY . .

# Exponer el puerto que usa Cloud Run (por defecto 8080)
ENV PORT=8080

# Comando para iniciar la app
# OJO: Cloud Run exige escuchar en 0.0.0.0 y en el puerto definido por la variable $PORT
CMD exec uvicorn main:app --host 0.0.0.0 --port $PORT
FROM python:3.13-slim

RUN apt-get update && apt-get install -y awscli && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api ./api
COPY train ./train
COPY model/model_registry.json ./model/model_registry.json
COPY scripts/download_model.sh ./scripts/download_model.sh
RUN chmod +x scripts/download_model.sh

EXPOSE 8000

CMD ["sh", "-c", "scripts/download_model.sh && uvicorn api.main:app --host 0.0.0.0 --port 8000"]

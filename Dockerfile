FROM python:3.9-slim

WORKDIR /app

# 1. INSTALL CPU-ONLY PYTORCH FIRST
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2. Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy model files and api into the container
COPY src/models/granite_model.py .
COPY src/models/qwen3_model.py .
COPY src/models/nemotron_model.py .
COPY src/models/jina_model.py .
COPY src/models/harrier_model.py .
COPY src/api/api.py .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

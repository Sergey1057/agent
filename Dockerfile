FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV LLM_AGENT_BACKEND=local \
    LLM_SERVICE_HOST=0.0.0.0 \
    LLM_SERVICE_PORT=8080

EXPOSE 8080

CMD ["python", "-m", "llm_service"]

# ML Model Deployment Guide with FastAPI & Flask

A comprehensive professional guide for building production-ready APIs to serve Machine Learning models.

---

## Overview

This repository contains two professional examples demonstrating how to deploy a trained ML model as an HTTP API service:
- **FastAPI** - High-performance ASGI server with automatic documentation
- **Flask** - Flexible and easy-to-use WSGI server

### Core Principles
- Load models at startup, not during import
- Health checks for monitoring
- Structured and detailed error handling
- Structured logging for observability
- Configurable settings via environment variables

---

## Project Structure

```
├── fastapi_api_example.py    # FastAPI example with startup loading
├── flask_api_example.py      # Flask example with lazy loading
├── requirements.txt          # Required dependencies
└── model.pkl                 # Trained model (not included)
```

---

## Quick Start

### 1. Environment Setup

```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Model Setup

Place your serialized model file `model.pkl` in the root directory, or specify its path via the `MODEL_PATH` environment variable.

### 3. Run Server

**FastAPI (with auto-reload for development)**
```powershell
uvicorn fastapi_api_example:app --reload
```
Swagger UI: http://127.0.0.1:8000/docs

**Flask**
```powershell
python flask_api_example.py
```
Default URL: http://127.0.0.1:5000

---

## Configuration

| Variable | Description | Default Value |
|----------|-------------|---------------|
| `MODEL_PATH` | Path to serialized model file | `model.pkl` |
| `HOST` | Server IP address | `127.0.0.1` |
| `PORT` | Server port | `8000` / `5000` |

---

## API Reference

### Prediction Endpoint

**`POST /predict`**

JSON Request:
```json
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

Responses:
- **200** - `{"prediction": [...]}` (success)
- **400** - Invalid input or prediction error
- **503** - Model unavailable (not loaded or failed to load)

### Health Check Endpoint

**`GET /health`**

Used to check server and model status.

---

## Usage Examples

### PowerShell
```powershell
$body = @{ features = @(5.1, 3.5, 1.4, 0.2) } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/predict" `
  -Body $body -ContentType "application/json"
```

### cURL
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features":[5.1,3.5,1.4,0.2]}'
```

---

## Docker Deployment

### Create Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy and install requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy files
COPY . /app

EXPOSE 8000

# Run server
CMD ["uvicorn", "fastapi_api_example:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Build and Run

```powershell
# Build image
docker build -t ml-api:latest .

# Run container
docker run --rm -p 8000:8000 -e MODEL_PATH=/app/model.pkl ml-api:latest
```

---

## Production Recommendations

### 1. Process Management
- Use `uvicorn` with multiple workers
- Or use `gunicorn` with `uvicorn.workers.UvicornWorker`

### 2. Monitoring & Health
- Wire `/health` to Kubernetes readiness/liveness probes
- Add Prometheus metrics for monitoring
- Use OpenTelemetry for distributed tracing

### 3. Security
- Validate and sanitize inputs
- Enable authentication and rate limiting
- Serve behind HTTPS proxy
- Don't log sensitive data

### 4. Model Management
- Separate code from model artifacts
- Use a Model Registry
- Use CI/CD for deploying new model versions
- Keep models immutable with version references

### 5. Observability
- Implement structured logging
- Monitor response time and resource usage
- Track error rates and prediction success

---

## Generative AI Considerations

When deploying Large Language Models (LLMs) or generative models:

### Performance & Efficiency
- **Optimization**: Use quantization (INT8/INT4) or distillation
- **Runtime**: Use `vLLM`, `text-generation-inference`, or ONNX Runtime
- **Hardware**: Use GPUs or NPUs (AWS Inferentia, GCP A2)

### Streaming Responses
- Implement streaming for long responses
- Use Server-Sent Events or WebSockets
- Design backpressure and timeout mechanisms

### Safety & Security
1. **Input Sanitization**: Apply length limits and content filters
2. **Output Moderation**: Use safety classifiers
3. **Rate Limiting**: Apply per-user/tenant quotas
4. **Privacy**: Be cautious about logging sensitive data

### Monitoring & Costs
- Monitor response time, GPU usage, and memory
- Track generated tokens and costs
- Export metrics to Prometheus/Grafana

### Caching & Optimization
- Cache repeated inputs/outputs
- Use repetition penalties in sampling
- Apply batching to improve throughput

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| 503 response at `/predict` | Check `/health` and server logs for model loading status |
| Prediction errors | Ensure `features` matches expected model shape and dtype |
| Slow response | Review model size and resource usage, consider optimization |

---

## Suggested Extensions

- **Unit Tests**: Add pytest for endpoint testing
- **Docker Compose**: Create multi-service setup
- **Authentication**: Add JWT or OAuth2
- **Caching**: Use Redis for result caching
- **Queue**: Use RabbitMQ or Celery for heavy tasks

---

## Developer

**Mohamed Khaled**  
Email: qq11gharipqq11@gmail.com

---

## License

This project is open source and available for educational and commercial use.

---

**Note**: This is a reference guide. Ensure you review your organization's specific security and performance requirements before deploying to production.

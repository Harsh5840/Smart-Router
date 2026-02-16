# 🚀 Intelligent Multi-Model LLM Chat Router

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-009688.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Production-grade intelligent routing system for multi-model LLM deployments.**

Automatically routes each user query to the optimal Large Language Model based on query complexity, domain, cost, latency, historical performance, and user tier. Designed to **reduce LLM inference costs by 40-60%** while maintaining response quality.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Quick Start](#quick-start)
- [API Documentation](#api-documentation)
- [Configuration](#configuration)
- [Development Phases](#development-phases)
- [Deployment](#deployment)
- [Monitoring & Observability](#monitoring--observability)
- [Performance](#performance)
- [Testing](#testing)
- [Training ML Models](#training-ml-models)
- [Production Checklist](#production-checklist)
- [Contributing](#contributing)
- [Roadmap](#roadmap)

---

## 🎯 Overview

### The Problem

Organizations using multiple LLM providers face challenges:
- **Cost**: Premium models (GPT-4, Claude) are expensive for all queries
- **Latency**: Not all queries need the most powerful (and slowest) models
- **Complexity**: Manual routing is error-prone and doesn't scale
- **Waste**: Using GPT-4 for "Hello, how are you?" is inefficient

### The Solution

This router intelligently selects the right model for each query using:
1. **Query Analysis**: NLP-based feature extraction and complexity scoring
2. **Rule-Based Routing**: Fast heuristics for common patterns
3. **ML Classification**: BERT-based models predict optimal routing
4. **RAG Enhancement**: Historical performance data guides decisions
5. **Multi-Factor Optimization**: Balances quality, cost, latency, and user tier

### Cost Savings Example

```
Before Router:
- 10,000 queries/day × $0.03/1k tokens × 100 tokens avg = $30/day = $900/month

After Router (60% routed to free local models):
- 6,000 queries → Llama-7B (local) = $0
- 4,000 queries → GPT-4/Claude = $12/day = $360/month

Monthly Savings: $540 (60% reduction)
```

---

## 🏗️ Architecture

### High-Level Design

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│           Nginx (Rate Limiting)             │
└──────────────────┬──────────────────────────┘
                   │
       ┌───────────┴───────────┐
       ▼                       ▼
┌─────────────┐         ┌─────────────┐
│  Router API │  ◄──►   │    Redis    │  (Cache)
│   Instance  │         │   Cache     │
└──────┬──────┘         └─────────────┘
       │
       ├──► Feature Extractor (NLP Analysis)
       │
       ├──► ML Classifier (Complexity/Domain)
       │
       ├──► RAG Service (Vector DB Similarity)
       │
       ├──► Decision Engine (Multi-factor Scoring)
       │
       ▼
┌─────────────────────────────────────────────┐
│         Model Selection & Routing           │
├─────────────┬──────────────┬────────────────┤
│   Llama-7B  │   GPT-4      │ Claude Sonnet  │
│   (Local)   │  (OpenAI)    │  (Anthropic)   │
└─────────────┴──────────────┴────────────────┘
       │
       ▼
┌─────────────┐
│  PostgreSQL │  (Logs, Analytics, Training Data)
└─────────────┘
```

### Component Breakdown

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **API Server** | FastAPI | Async REST API with automatic OpenAPI docs |
| **Feature Extractor** | sentence-transformers | NLP-based query analysis and embeddings |
| **Router** | Custom Python | Multi-stage routing logic (rules → ML → RAG) |
| **ML Classifier** | BERT (transformers) | Complexity and domain classification |
| **RAG Service** | Pinecone | Historical performance retrieval |
| **Cache** | Redis | Response caching and deduplication |
| **Database** | PostgreSQL | Routing logs, metrics, training data |
| **Metrics** | Prometheus | Request counters, latency histograms |
| **Model Registry** | Custom | Unified client interface for all LLMs |

---

## ✨ Features

### Core Capabilities

- ✅ **Multi-Model Support**: Llama, GPT-4, Claude (extensible to any LLM)
- ✅ **Intelligent Routing**: 4-stage decision process (cache → rules → ML → RAG)
- ✅ **Cost Optimization**: Automatic selection of cheapest adequate model
- ✅ **Response Caching**: Redis-based exact and semantic caching
- ✅ **User Tiers**: Different routing strategies for free/pro/enterprise
- ✅ **Feature Extraction**: Automatic query analysis (tokens, domain, complexity)
- ✅ **ML Classification**: BERT-based complexity and domain prediction
- ✅ **RAG Enhancement**: Vector similarity search for historical outcomes
- ✅ **Observability**: Structured logging + Prometheus metrics
- ✅ **Fault Tolerance**: Circuit breakers, retries, fallback models
- ✅ **Production-Ready**: Docker, Kubernetes, load balancing, health checks

### Routing Strategies

#### Phase 1-2: Hardcoded + Feature Extraction
```python
# All queries → default model (Llama-7B)
# Features extracted but not used for routing yet
```

#### Phase 3: Rule-Based
```python
if complexity < 0.3 and not has_code_block:
    return "llama-7b"  # Fast, free
elif is_coding and has_code_block:
    return "gpt-4"     # Best for code
elif is_analytical:
    return "claude-sonnet"  # Good at analysis
else:
    return "llama-7b"  # Default
```

#### Phase 5-7: ML + RAG + Multi-Factor
```python
# 1. ML Classifier predicts complexity + domain
# 2. RAG retrieves similar historical queries
# 3. Score each model:
#    - Quality (model tier vs query complexity)
#    - Cost (cheaper is better, weighted by user tier)
#    - Latency (faster is better)
#    - ML confidence
#    - RAG success rate
# 4. Select highest-scoring model
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- PostgreSQL 15+
- Redis 7+
- (Optional) Pinecone account for RAG

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-org/llm-router.git
cd llm-router
```

2. **Set up environment**
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

3. **Install dependencies**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

4. **Start infrastructure with Docker**
```bash
docker-compose up -d postgres redis
```

5. **Run the API**
```bash
python src/main.py
```

The API will be available at `http://localhost:8000`

### Docker Deployment (Recommended)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f router-api

# Stop all services
docker-compose down
```

### First Request

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is Python?",
    "user_id": "user_123",
    "user_tier": "free"
  }'
```

Response:
```json
{
  "response": "[Llama-7B Response to: What is Python?...]",
  "model_used": "llama-7b",
  "latency_ms": 523.4,
  "routing_metadata": {
    "reason": "simple_query",
    "confidence": 0.9,
    "features": {
      "complexity": 0.15,
      "is_coding": false,
      "is_analytical": false,
      "is_creative": false
    }
  }
}
```

---

## 📚 API Documentation

### Endpoints

#### `POST /api/v1/chat`

Route a query to the optimal LLM and return response.

**Request:**
```json
{
  "query": "string",          // User query (required)
  "user_id": "string",        // Unique user ID (required)
  "context": "string",        // Optional conversation context
  "user_tier": "free|pro|enterprise"  // Default: "free"
}
```

**Response:**
```json
{
  "response": "string",       // LLM generated response
  "model_used": "string",     // Model that handled the request
  "latency_ms": 0.0,          // Total request latency
  "routing_metadata": {
    "reason": "string",       // Routing decision reason
    "confidence": 0.0,        // Confidence score (0-1)
    "alternatives": [],       // Alternative model options
    "fallback": false,        // Whether fallback was used
    "request_id": "string",   // Log ID for feedback
    "features": {}            // Extracted query features
  }
}
```

#### `POST /api/v1/feedback`

Submit feedback for a chat response.

**Request:**
```json
{
  "request_id": "string",     // From routing_metadata
  "rating": 1-5,              // Required
  "comment": "string"         // Optional
}
```

#### `GET /api/v1/health`

Health check endpoint.

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2026-02-16T...",
  "services": {
    "models": "3/3 available",
    "cache": "available",
    "database": "available"
  }
}
```

#### `GET /api/v1/metrics`

Prometheus metrics in text format.

#### `GET /api/v1/stats`

Human-readable statistics.

```json
{
  "total_requests": 10000,
  "cache_hit_rate": 0.35,
  "avg_latency_ms": 450.0,
  "model_distribution": {
    "llama-7b": 6000,
    "gpt-4": 2500,
    "claude-sonnet": 1500
  },
  "error_rate": 0.02
}
```

#### `GET /api/v1/models`

List all available models and their configurations.

---

## ⚙️ Configuration

### Environment Variables

```bash
# Application
APP_NAME=llm-router
APP_ENV=production          # development|production
LOG_LEVEL=INFO              # DEBUG|INFO|WARNING|ERROR

# Server
HOST=0.0.0.0
PORT=8000
WORKERS=4

# Database
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/router_db

# Redis Cache
REDIS_URL=redis://localhost:6379/0

# Pinecone (for RAG)
PINECONE_API_KEY=your-key
PINECONE_ENVIRONMENT=us-west1-gcp
PINECONE_INDEX_NAME=llm-router

# Model APIs
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Feature Flags
ENABLE_ML_ROUTING=true      # Use ML classifier
ENABLE_RAG_ROUTING=true     # Use RAG recommendations
ENABLE_CACHING=true         # Enable Redis caching

# Performance
REQUEST_TIMEOUT=30          # Seconds
MAX_RETRIES=3
CIRCUIT_BREAKER_THRESHOLD=5
```

### Model Configuration

Edit [src/services/model_registry.py](src/services/model_registry.py) to add/modify models:

```python
MODEL_CONFIGS = {
    "your-model": ModelConfig(
        name="your-model",
        provider="your-provider",
        cost_per_1k_tokens=0.001,
        max_tokens=4096,
        supports_streaming=True,
        avg_latency_ms=500,
        quality_tier="medium",  # low|medium|high
    ),
}
```

---

## 🔄 Development Phases

The system was built incrementally in 10 phases (all phases are implemented):

### ✅ Phase 1: System Skeleton
- FastAPI application with `/chat` endpoint
- Model client abstraction
- Hardcoded routing (all → default model)
- Structured logging

### ✅ Phase 2: Feature Extraction
- Token counting, sentence analysis
- Domain classification (coding/analytical/creative)
- Embedding generation (sentence-transformers)

### ✅ Phase 3: Rule-Based Router
- Complexity scoring
- Simple heuristics for model selection
- Fallback logic

### ✅ Phase 4: Data Collection
- PostgreSQL schema for routing logs
- Performance tracking per model
- Feedback collection endpoint

### ✅ Phase 5: ML Classifier
- BERT-based multi-task classifier
- Training pipeline: `python training/train_classifier.py`
- Predicts: complexity (simple/medium/complex) + domain

### ✅ Phase 6: RAG Enhancement
- Pinecone vector database integration
- Historical query similarity search
- Success-rate weighted recommendations

### ✅ Phase 7: Decision Engine
- Multi-factor scoring (quality, cost, latency, ML, RAG)
- Confidence scores and alternative models
- User tier consideration

### ✅ Phase 8: Caching & Performance
- Redis response caching
- Cache hit rate tracking
- Semantic deduplication

### ✅ Phase 9: Reliability & Observability
- Prometheus metrics
- Circuit breakers and retries
- Health checks and status endpoints

### ✅ Phase 10: Deployment
- Production Dockerfile
- Docker Compose orchestration
- Kubernetes manifests with autoscaling
- Nginx reverse proxy with rate limiting

---

## 🚢 Deployment

### Docker Compose (Development/Staging)

```bash
# Start all services
docker-compose up -d

# Scale API instances
docker-compose up -d --scale router-api=3

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Kubernetes (Production)

```bash
# Apply manifests
kubectl apply -f deployment/kubernetes/manifests.yaml

# Check deployment
kubectl get pods -n llm-router

# Scale horizontally
kubectl scale deployment router-api -n llm-router --replicas=10

# View logs
kubectl logs -f deployment/router-api -n llm-router
```

### Environment-Specific Deployments

**Development:**
```bash
APP_ENV=development python src/main.py
# Features: auto-reload, verbose logging, CORS allow-all
```

**Production:**
```bash
APP_ENV=production uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4
# Features: minimal logging, trusted hosts, no CORS
```

---

## 📊 Monitoring & Observability

### Structured Logging

All logs are JSON-formatted (production) or human-readable (development):

```json
{
  "event": "routing_decision",
  "model": "llama-7b",
  "reason": "simple_query",
  "confidence": 0.9,
  "complexity": 0.15,
  "timestamp": "2026-02-16T10:30:45Z",
  "level": "info"
}
```

### Prometheus Metrics

Available at `/api/v1/metrics`:

- `llm_router_requests_total{model, endpoint}` - Total requests
- `llm_router_request_latency_seconds{model, endpoint}` - Latency histogram
- `llm_router_errors_total{model, error_type}` - Error counter
- `llm_router_cache_hits_total` - Cache hits
- `llm_router_cache_misses_total` - Cache misses
- `llm_router_routing_decisions_total{routing_type, model}` - Routing decisions
- `llm_router_circuit_breaker_state{model}` - Circuit breaker status

### Grafana Dashboard (Example Queries)

```promql
# Request rate by model
rate(llm_router_requests_total[5m])

# P95 latency
histogram_quantile(0.95, rate(llm_router_request_latency_seconds_bucket[5m]))

# Cache hit rate
rate(llm_router_cache_hits_total[5m]) / 
  (rate(llm_router_cache_hits_total[5m]) + rate(llm_router_cache_misses_total[5m]))

# Error rate
rate(llm_router_errors_total[5m]) / rate(llm_router_requests_total[5m])
```

---

## ⚡ Performance

### Benchmarks (Target: 10k RPS)

| Metric | Value | Notes |
|--------|-------|-------|
| **Throughput** | 10,000 RPS | With 10 instances (1k RPS each) |
| **Latency (p50)** | 450ms | Includes model inference |
| **Latency (p95)** | 1,200ms | Premium models slower |
| **Latency (p99)** | 2,500ms | Edge cases + retries |
| **Cache Hit Rate** | 35-45% | Depends on query diversity |
| **Cold Start** | <5s | Application startup |
| **Memory/Instance** | ~500MB | Base + ML models |

### Optimization Tips

1. **Enable Caching**: `ENABLE_CACHING=true` (35%+ hit rate typical)
2. **Use Local Models**: Route simple queries to self-hosted Llama
3. **Horizontal Scaling**: Add more API instances (stateless design)
4. **Connection Pooling**: Adjust DB pool size for high concurrency
5. **Async Everything**: All I/O operations are async (no blocking)

---

## 🧪 Testing

### Run Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Specific test file
pytest tests/test_api.py -v

# Integration tests only
pytest tests/ -v -m integration
```

### Test Coverage

- ✅ API endpoints (chat, feedback, health, metrics)
- ✅ Feature extraction
- ✅ Routing logic (rule-based, ML, RAG)
- ✅ Model registry
- ✅ Caching service
- ✅ End-to-end integration

### Load Testing (Example)

```bash
# Using Apache Bench
ab -n 10000 -c 100 -p request.json -T application/json \
  http://localhost:8000/api/v1/chat

# Using Locust
locust -f tests/load_test.py --host=http://localhost:8000
```

---

## 🎓 Training ML Models

### Phase 5: Train Complexity/Domain Classifier

1. **Collect Training Data** (need 1000+ routing logs):
```bash
# Use the API for a while to generate logs
# Or import existing data into PostgreSQL
```

2. **Train Models**:
```bash
python training/train_classifier.py
# Models saved to: training/models/complexity & training/models/domain
```

3. **Enable ML Routing**:
```bash
ENABLE_ML_ROUTING=true
```

### Phase 6: Populate Vector Database

```bash
python training/rag_service.py
# Indexes historical routing logs into Pinecone
```

Enable RAG:
```bash
ENABLE_RAG_ROUTING=true
```

---

## 🏢 Production Checklist

Before deploying to production:

- [ ] Set `APP_ENV=production`
- [ ] Configure real API keys (OpenAI, Anthropic, Pinecone)
- [ ] Set strong database passwords
- [ ] Enable SSL/TLS (update Nginx config)
- [ ] Configure trusted hosts
- [ ] Set up monitoring (Prometheus + Grafana)
- [ ] Configure log aggregation (ELK, Datadog, etc.)
- [ ] Set up alerting for errors/latency
- [ ] Enable auto-scaling in Kubernetes
- [ ] Configure backup strategy for PostgreSQL
- [ ] Set Redis persistence (AOF or RDB)
- [ ] Review and adjust rate limits
- [ ] Load test with production-like traffic
- [ ] Set up CI/CD pipeline
- [ ] Document runbooks for incidents

---

## 🤝 Contributing

### Development Setup

```bash
# Fork and clone
git clone https://github.com/your-username/llm-router.git
cd llm-router

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dev dependencies
pip install -r requirements.txt
pip install black ruff mypy pytest-cov

# Install pre-commit hooks (if configured)
# pre-commit install
```

### Code Style

```bash
# Format code
black src/ tests/

# Lint
ruff src/ tests/

# Type check
mypy src/
```

### Pull Request Process

1. Create a feature branch: `git checkout -b feature/amazing-feature`
2. Make your changes
3. Add tests for new functionality
4. Ensure all tests pass: `pytest tests/`
5. Format code: `black src/ tests/`
6. Submit PR with clear description

---

## 🗺️ Roadmap

### Future Enhancements

- [ ] **Streaming Responses**: Support for Server-Sent Events
- [ ] **A/B Testing**: Compare routing strategies
- [ ] **Cost Budgets**: Per-user spending limits
- [ ] **Multi-Region**: Deploy across multiple regions
- [ ] **Auto-Tuning**: Self-optimizing routing weights
- [ ] **More Models**: Gemini, Mistral, Cohere support
- [ ] **Fine-Tuning**: Custom routing models per use case
- [ ] **Analytics Dashboard**: Built-in web UI for metrics

---

**Built with ❤️ for production-grade LLM infrastructure**

*Last Updated: February 16, 2026*

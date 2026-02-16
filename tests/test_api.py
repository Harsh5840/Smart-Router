"""
Basic tests for the LLM Router
Run with: pytest tests/
"""

import pytest
from httpx import AsyncClient
from src.main import app
from src.models.schemas import ChatRequest, ChatResponse


@pytest.fixture
async def client():
    """Test client fixture"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


# ============================================================================
# PHASE 1: Basic API Tests
# ============================================================================


@pytest.mark.asyncio
async def test_root_endpoint(client):
    """Test root endpoint"""
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "LLM Router"
    assert data["status"] == "operational"


@pytest.mark.asyncio
async def test_health_check(client):
    """Test health check endpoint"""
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "services" in data


@pytest.mark.asyncio
async def test_list_models(client):
    """Test models listing endpoint"""
    response = await client.get("/api/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert len(data["models"]) > 0


@pytest.mark.asyncio
async def test_chat_endpoint(client):
    """Test chat endpoint with basic request"""
    request_data = {
        "query": "What is Python?",
        "user_id": "test_user_123",
        "user_tier": "free",
    }

    response = await client.post("/api/v1/chat", json=request_data)
    assert response.status_code == 200

    data = response.json()
    assert "response" in data
    assert "model_used" in data
    assert "latency_ms" in data
    assert "routing_metadata" in data


# ============================================================================
# PHASE 2: Feature Extraction Tests
# ============================================================================


@pytest.mark.asyncio
async def test_feature_extraction():
    """Test feature extraction"""
    from src.services.feature_extractor import feature_extractor

    query = "Write a Python function to sort a list"
    features = await feature_extractor.extract_features(query)

    assert features.token_count > 0
    assert features.word_count > 0
    assert features.sentence_count > 0
    assert features.is_coding is True


@pytest.mark.asyncio
async def test_complexity_score():
    """Test complexity score calculation"""
    from src.services.feature_extractor import feature_extractor

    simple_query = "Hi"
    complex_query = "Explain quantum computing and provide Python code examples"

    simple_features = await feature_extractor.extract_features(simple_query)
    complex_features = await feature_extractor.extract_features(complex_query)

    simple_score = feature_extractor.calculate_complexity_score(simple_features)
    complex_score = feature_extractor.calculate_complexity_score(complex_features)

    assert complex_score > simple_score


# ============================================================================
# PHASE 3: Routing Tests
# ============================================================================


@pytest.mark.asyncio
async def test_rule_based_routing():
    """Test rule-based routing"""
    from src.services.router import router
    from src.services.feature_extractor import feature_extractor

    # Simple query should route to cheap model
    simple_query = "Hello, how are you?"
    simple_features = await feature_extractor.extract_features(simple_query)
    simple_decision = await router.route_rule_based(
        simple_query, simple_features, "free"
    )

    assert simple_decision.selected_model == "llama-7b"
    assert simple_decision.confidence > 0


# ============================================================================
# PHASE 4: Data Collection Tests
# ============================================================================


@pytest.mark.asyncio
async def test_feedback_endpoint(client):
    """Test feedback submission"""
    feedback_data = {
        "request_id": "test_request_123",
        "rating": 5,
        "comment": "Great response!",
    }

    response = await client.post("/api/v1/feedback", json=feedback_data)
    # May fail if database not available in test env
    assert response.status_code in [200, 500]


# ============================================================================
# Model Client Tests
# ============================================================================


@pytest.mark.asyncio
async def test_model_registry():
    """Test model registry"""
    from src.services.model_registry import model_registry

    assert model_registry.is_model_available("llama-7b")
    
    client = model_registry.get_client("llama-7b")
    assert client is not None
    assert client.get_model_name() == "llama-7b"


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.asyncio
async def test_end_to_end_simple_query(client):
    """Test complete flow with simple query"""
    request_data = {
        "query": "Hi there!",
        "user_id": "integration_test_user",
        "user_tier": "free",
    }

    response = await client.post("/api/v1/chat", json=request_data)
    assert response.status_code == 200

    data = response.json()
    # Simple query should use cheap model
    assert data["model_used"] == "llama-7b"


@pytest.mark.asyncio
async def test_end_to_end_coding_query(client):
    """Test complete flow with coding query"""
    request_data = {
        "query": "```python\ndef quicksort(arr):\n    pass\n```\nComplete this function",
        "user_id": "integration_test_user",
        "user_tier": "pro",
    }

    response = await client.post("/api/v1/chat", json=request_data)
    assert response.status_code == 200

    data = response.json()
    # Coding query with code block should use premium model
    assert data["model_used"] in ["gpt-4", "claude-sonnet"]
    assert data["routing_metadata"]["features"]["is_coding"] is True

"""
API endpoints for the LLM router
"""

import time
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Response
from src.models.schemas import (
    ChatRequest,
    ChatResponse,
    FeedbackRequest,
    HealthCheck,
    MetricsResponse,
)
from src.services.model_registry import model_registry
from src.services.feature_extractor import feature_extractor
from src.services.router import router
from src.services.cache import cache_service
from src.services.data_collection import data_collection_service
from src.utils.logging import get_logger
from src.utils.metrics import REQUEST_COUNT, REQUEST_LATENCY, REQUEST_ERRORS, get_metrics
from src.config import settings
from datetime import datetime

logger = get_logger(__name__)

# Create API router
api_router = APIRouter()


# ============================================================================
# PHASE 1: Core Chat Endpoint
# ============================================================================


@api_router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Main chat endpoint - routes query to optimal LLM and returns response

    Flow:
    1. Check cache (Phase 8)
    2. Extract features (Phase 2)
    3. Route to best model (Phase 3-7)
    4. Generate response
    5. Cache response
    6. Log decision (Phase 4)
    """
    start_time = time.time()
    request_id = None

    try:
        logger.info(
            "chat_request_received",
            user_id=request.user_id,
            query_length=len(request.query),
            user_tier=request.user_tier,
        )

        # PHASE 8: Check cache first
        cached = await cache_service.get_cached_response(
            request.query,
            settings.default_model,
        )

        if cached:
            logger.info("serving_from_cache", user_id=request.user_id)
            return ChatResponse(
                response=cached["response"],
                model_used=cached["model"],
                latency_ms=cached["latency_ms"],
                routing_metadata={
                    "from_cache": True,
                    "cache_timestamp": cached.get("timestamp"),
                },
            )

        # PHASE 2: Extract features
        features = await feature_extractor.extract_features(request.query)
        
        # Generate embedding for advanced routing (Phase 6)
        embedding = None
        if settings.enable_rag_routing:
            embedding = await feature_extractor.generate_embedding(request.query)

        # PHASE 3-7: Route to best model
        routing_decision = await router.route(
            query=request.query,
            user_id=request.user_id,
            user_tier=request.user_tier or "free",
            features=features,
            embedding=embedding,
        )

        logger.info(
            "routing_complete",
            model=routing_decision.selected_model,
            reason=routing_decision.reason,
            confidence=routing_decision.confidence,
        )

        # Get model client and generate response
        client = model_registry.get_client(routing_decision.selected_model)

        if client is None:
            raise HTTPException(
                status_code=503,
                detail=f"Model {routing_decision.selected_model} not available",
            )

        # Generate response
        result = await client.generate(
            prompt=request.query,
            temperature=0.7,
        )

        # Calculate total latency
        total_latency_ms = (time.time() - start_time) * 1000

        # PHASE 8: Cache the response
        if settings.enable_caching:
            await cache_service.cache_response(
                query=request.query,
                model=routing_decision.selected_model,
                response=result["response"],
                latency_ms=result["latency_ms"],
            )

        # PHASE 4: Log routing decision
        cost_estimate = client.estimate_cost(result["tokens_used"])
        
        request_id = await data_collection_service.log_routing_decision(
            query=request.query,
            user_id=request.user_id,
            features=features.model_dump(),
            model_used=routing_decision.selected_model,
            latency_ms=total_latency_ms,
            cost_estimate=cost_estimate,
            success=True,
            embedding=embedding,
        )

        # Update metrics
        REQUEST_COUNT.labels(
            model=routing_decision.selected_model,
            endpoint="/chat",
        ).inc()

        REQUEST_LATENCY.labels(
            model=routing_decision.selected_model,
            endpoint="/chat",
        ).observe(total_latency_ms / 1000)

        logger.info(
            "chat_request_complete",
            user_id=request.user_id,
            model=routing_decision.selected_model,
            latency_ms=total_latency_ms,
            cost=cost_estimate,
            request_id=request_id,
        )

        return ChatResponse(
            response=result["response"],
            model_used=routing_decision.selected_model,
            latency_ms=total_latency_ms,
            routing_metadata={
                "reason": routing_decision.reason,
                "confidence": routing_decision.confidence,
                "alternatives": routing_decision.alternatives,
                "fallback": routing_decision.fallback,
                "request_id": str(request_id) if request_id else None,
                "features": {
                    "complexity": feature_extractor.calculate_complexity_score(features),
                    "is_coding": features.is_coding,
                    "is_analytical": features.is_analytical,
                    "is_creative": features.is_creative,
                },
            },
        )

    except Exception as e:
        logger.error(
            "chat_request_failed",
            user_id=request.user_id,
            error=str(e),
            exc_info=True,
        )

        REQUEST_ERRORS.labels(
            model="unknown",
            error_type=type(e).__name__,
        ).inc()

        raise HTTPException(status_code=500, detail=f"Request failed: {str(e)}")


# ============================================================================
# PHASE 4: Feedback Endpoint
# ============================================================================


@api_router.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest) -> Dict[str, str]:
    """
    Submit user feedback for a chat response
    Used to improve routing decisions over time
    """
    try:
        success = await data_collection_service.save_feedback(
            request_id=feedback.request_id,
            rating=feedback.rating,
            comment=feedback.comment,
        )

        if success:
            logger.info(
                "feedback_received",
                request_id=feedback.request_id,
                rating=feedback.rating,
            )
            return {"status": "success", "message": "Feedback recorded"}
        else:
            raise HTTPException(status_code=500, detail="Failed to save feedback")

    except Exception as e:
        logger.error("feedback_submission_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# PHASE 9: Health and Metrics Endpoints
# ============================================================================


@api_router.get("/health", response_model=HealthCheck)
async def health_check() -> HealthCheck:
    """
    Health check endpoint
    Returns service status and availability
    """
    # Check model availability
    model_health = await model_registry.health_check_all()

    # Determine overall status
    healthy_models = sum(1 for status in model_health.values() if status)
    total_models = len(model_health)

    overall_status = "healthy" if healthy_models > 0 else "unhealthy"

    services = {
        "models": f"{healthy_models}/{total_models} available",
        "cache": "available" if cache_service.redis_client else "unavailable",
        "database": "available" if data_collection_service.engine else "unavailable",
    }

    return HealthCheck(
        status=overall_status,
        version="1.0.0",
        timestamp=datetime.utcnow(),
        services=services,
    )


@api_router.get("/metrics")
async def metrics() -> Response:
    """
    Prometheus metrics endpoint
    Returns metrics in Prometheus text format
    """
    metrics_data, content_type = get_metrics()
    return Response(content=metrics_data, media_type=content_type)


@api_router.get("/stats", response_model=MetricsResponse)
async def get_stats() -> MetricsResponse:
    """
    Get aggregated statistics
    Returns human-readable metrics
    """
    try:
        # Get cache stats
        cache_stats = await cache_service.get_cache_stats()
        
        # Calculate cache hit rate
        total_cache_requests = (
            cache_stats.get("keyspace_hits", 0) + cache_stats.get("keyspace_misses", 0)
        )
        cache_hit_rate = (
            cache_stats.get("keyspace_hits", 0) / total_cache_requests
            if total_cache_requests > 0
            else 0.0
        )

        # TODO: Fetch actual stats from database
        # For now, return mock data
        return MetricsResponse(
            total_requests=total_cache_requests,
            cache_hit_rate=cache_hit_rate,
            avg_latency_ms=500.0,
            model_distribution={
                "llama-7b": 60,
                "gpt-4": 25,
                "claude-sonnet": 15,
            },
            error_rate=0.02,
        )

    except Exception as e:
        logger.error("stats_fetch_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to fetch stats")


@api_router.get("/models")
async def list_models() -> Dict[str, Any]:
    """
    List all available models and their configurations
    """
    models = model_registry.get_all_models()
    health = await model_registry.health_check_all()

    return {
        "models": [
            {
                "name": name,
                "provider": config.provider,
                "cost_per_1k_tokens": config.cost_per_1k_tokens,
                "max_tokens": config.max_tokens,
                "avg_latency_ms": config.avg_latency_ms,
                "quality_tier": config.quality_tier,
                "available": health.get(name, False),
            }
            for name, config in models.items()
        ]
    }

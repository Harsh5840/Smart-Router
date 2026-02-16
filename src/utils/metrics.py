"""
Metrics collection using Prometheus
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# ============================================================================
# PHASE 9: Metrics Definitions
# ============================================================================

# Request metrics
REQUEST_COUNT = Counter(
    "llm_router_requests_total",
    "Total number of requests",
    ["model", "endpoint"],
)

REQUEST_LATENCY = Histogram(
    "llm_router_request_latency_seconds",
    "Request latency in seconds",
    ["model", "endpoint"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
)

REQUEST_ERRORS = Counter(
    "llm_router_errors_total",
    "Total number of errors",
    ["model", "error_type"],
)

# Cache metrics
CACHE_HITS = Counter(
    "llm_router_cache_hits_total",
    "Total number of cache hits",
)

CACHE_MISSES = Counter(
    "llm_router_cache_misses_total",
    "Total number of cache misses",
)

# Routing metrics
ROUTING_DECISIONS = Counter(
    "llm_router_routing_decisions_total",
    "Total routing decisions by type",
    ["routing_type", "model"],
)

# Model metrics
ACTIVE_MODELS = Gauge(
    "llm_router_active_models",
    "Number of active models",
)

MODEL_CIRCUIT_BREAKER = Gauge(
    "llm_router_circuit_breaker_state",
    "Circuit breaker state (0=closed, 1=open)",
    ["model"],
)


def get_metrics() -> tuple[bytes, str]:
    """Get Prometheus metrics in text format"""
    return generate_latest(), CONTENT_TYPE_LATEST

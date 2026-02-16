"""
Routing service - decides which model to use for a given query
"""

from typing import Dict, Any, Optional, List
from src.models.schemas import QueryFeatures, RoutingDecision, ModelCandidate
from src.services.model_registry import model_registry, MODEL_CONFIGS
from src.services.feature_extractor import feature_extractor
from src.utils.logging import get_logger
from src.config import settings

logger = get_logger(__name__)


# ============================================================================
# PHASE 3: Rule-Based Router
# ============================================================================


class Router:
    """
    Intelligent router that selects the optimal model for each query
    Evolves from simple rules to ML-based decisions
    """

    def __init__(self):
        self.ml_classifier = None  # PHASE 5: Will be loaded here
        self.rag_service = None    # PHASE 6: Will be loaded here

    # ========================================================================
    # PHASE 1: Hardcoded Routing (Initial Implementation)
    # ========================================================================

    async def route_hardcoded(self, query: str, user_id: str) -> RoutingDecision:
        """
        Initial hardcoded routing - always returns default model
        PHASE 1 ONLY - Replaced in Phase 3
        """
        return RoutingDecision(
            selected_model=settings.default_model,
            reason="hardcoded_default",
            confidence=1.0,
            fallback=False,
        )

    # ========================================================================
    # PHASE 3: Rule-Based Routing
    # ========================================================================

    async def route_rule_based(
        self,
        query: str,
        features: QueryFeatures,
        user_tier: str = "free"
    ) -> RoutingDecision:
        """
        Rule-based routing using extracted features
        Simple heuristics for model selection
        """
        complexity_score = feature_extractor.calculate_complexity_score(features)

        # Rule 1: Simple queries -> cheap model
        if complexity_score < 0.3 and not features.has_code_block:
            selected = "llama-7b"
            reason = "simple_query"
            confidence = 0.9

        # Rule 2: Coding queries with code blocks -> premium model
        elif features.is_coding and features.has_code_block:
            selected = "gpt-4" if user_tier in ["pro", "enterprise"] else "claude-sonnet"
            reason = "coding_query_with_code"
            confidence = 0.85

        # Rule 3: Long analytical queries -> premium model
        elif features.is_analytical and features.token_count > 100:
            selected = "claude-sonnet"
            reason = "analytical_query"
            confidence = 0.8

        # Rule 4: Creative queries -> medium model
        elif features.is_creative:
            selected = "claude-sonnet"
            reason = "creative_query"
            confidence = 0.75

        # Rule 5: Complex queries -> premium model
        elif complexity_score > 0.6:
            selected = "gpt-4" if user_tier == "enterprise" else "claude-sonnet"
            reason = "high_complexity"
            confidence = 0.8

        # Default: Use cheap model
        else:
            selected = "llama-7b"
            reason = "default_fallback"
            confidence = 0.7

        # Fallback if selected model not available
        if not model_registry.is_model_available(selected):
            logger.warning("model_unavailable", requested=selected)
            selected = settings.fallback_model
            reason = f"fallback_from_{selected}"
            confidence *= 0.8
            fallback = True
        else:
            fallback = False

        logger.info(
            "routing_decision",
            model=selected,
            reason=reason,
            confidence=confidence,
            complexity=complexity_score,
        )

        return RoutingDecision(
            selected_model=selected,
            reason=reason,
            confidence=confidence,
            fallback=fallback,
        )

    # ========================================================================
    # PHASE 7: Advanced Decision Engine
    # ========================================================================

    async def route_optimized(
        self,
        query: str,
        features: QueryFeatures,
        user_tier: str = "free",
        embedding: Optional[List[float]] = None,
    ) -> RoutingDecision:
        """
        Advanced routing using multi-factor optimization
        Combines ML predictions, RAG, and scoring
        """
        candidates: List[ModelCandidate] = []

        # Score each available model
        for model_name, config in MODEL_CONFIGS.items():
            if not model_registry.is_model_available(model_name):
                continue

            # Calculate individual scores
            quality_score = self._calculate_quality_score(config, features)
            cost_score = self._calculate_cost_score(config, user_tier)
            latency_score = self._calculate_latency_score(config)

            # PHASE 5: Add ML prediction score
            ml_score = 0.0
            if settings.enable_ml_routing and self.ml_classifier:
                ml_score = await self._get_ml_score(query, model_name)

            # PHASE 6: Add RAG recommendation score
            rag_score = 0.0
            if settings.enable_rag_routing and self.rag_service and embedding:
                rag_score = await self._get_rag_score(embedding, model_name)

            # Weighted overall score
            overall_score = (
                quality_score * 0.3 +
                cost_score * 0.25 +
                latency_score * 0.15 +
                ml_score * 0.15 +
                rag_score * 0.15
            )

            candidates.append(
                ModelCandidate(
                    model_name=model_name,
                    quality_score=quality_score,
                    cost_score=cost_score,
                    latency_score=latency_score,
                    overall_score=overall_score,
                    metadata={
                        "ml_score": ml_score,
                        "rag_score": rag_score,
                    }
                )
            )

        # Sort by overall score
        candidates.sort(key=lambda x: x.overall_score, reverse=True)

        if not candidates:
            return await self.route_rule_based(query, features, user_tier)

        best = candidates[0]
        alternatives = [
            {"model": c.model_name, "score": c.overall_score}
            for c in candidates[1:3]
        ]

        logger.info(
            "optimized_routing",
            model=best.model_name,
            score=best.overall_score,
            quality=best.quality_score,
            cost=best.cost_score,
        )

        return RoutingDecision(
            selected_model=best.model_name,
            reason="multi_factor_optimization",
            confidence=best.overall_score,
            alternatives=alternatives,
            fallback=False,
        )

    def _calculate_quality_score(self, config, features: QueryFeatures) -> float:
        """Calculate quality score for a model"""
        base_score = {"low": 0.3, "medium": 0.6, "high": 1.0}[config.quality_tier]
        
        # Boost for complex queries
        complexity = feature_extractor.calculate_complexity_score(features)
        if complexity > 0.6 and config.quality_tier == "high":
            base_score *= 1.2
        
        return min(base_score, 1.0)

    def _calculate_cost_score(self, config, user_tier: str) -> float:
        """Calculate cost score (higher is better = cheaper)"""
        if config.cost_per_1k_tokens == 0:
            return 1.0
        
        # Normalize cost (inverse - cheaper is better)
        max_cost = 0.03
        cost_score = 1.0 - (config.cost_per_1k_tokens / max_cost)
        
        # Enterprise users care less about cost
        if user_tier == "enterprise":
            cost_score = 0.5 + (cost_score * 0.5)
        
        return max(cost_score, 0.0)

    def _calculate_latency_score(self, config) -> float:
        """Calculate latency score (higher is better = faster)"""
        max_latency = 2000
        return 1.0 - min(config.avg_latency_ms / max_latency, 1.0)

    async def _get_ml_score(self, query: str, model_name: str) -> float:
        """
        PHASE 5: Get ML classifier score
        TODO: Implement in Phase 5
        """
        return 0.0

    async def _get_rag_score(self, embedding: List[float], model_name: str) -> float:
        """
        PHASE 6: Get RAG recommendation score
        TODO: Implement in Phase 6
        """
        return 0.0

    # ========================================================================
    # Main Routing Entry Point
    # ========================================================================

    async def route(
        self,
        query: str,
        user_id: str,
        user_tier: str = "free",
        features: Optional[QueryFeatures] = None,
        embedding: Optional[List[float]] = None,
    ) -> RoutingDecision:
        """
        Main routing method - dispatches to appropriate routing strategy
        """
        # Extract features if not provided
        if features is None:
            features = await feature_extractor.extract_features(query)

        # PHASE 7+: Use optimized routing if advanced features enabled
        if settings.enable_ml_routing or settings.enable_rag_routing:
            return await self.route_optimized(query, features, user_tier, embedding)
        
        # PHASE 3+: Use rule-based routing
        return await self.route_rule_based(query, features, user_tier)


# Global router instance
router = Router()

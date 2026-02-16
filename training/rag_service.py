"""
RAG Service for historical routing recommendations
PHASE 6: Vector-based similarity search for routing decisions
"""

import asyncio
from typing import List, Dict, Any, Optional
import numpy as np

try:
    import pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import settings
from src.services.feature_extractor import feature_extractor
from src.services.data_collection import data_collection_service
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# PHASE 6: RAG-Enhanced Routing Service
# ============================================================================


class RAGService:
    """
    Retrieval-Augmented Generation service for routing
    Uses historical routing outcomes to recommend models
    """

    def __init__(self):
        self.index = None
        self.initialized = False
        
        if PINECONE_AVAILABLE and settings.pinecone_api_key:
            self._initialize_pinecone()
        else:
            logger.warning("pinecone_not_available")

    def _initialize_pinecone(self) -> None:
        """Initialize Pinecone vector database"""
        try:
            pinecone.init(
                api_key=settings.pinecone_api_key,
                environment=settings.pinecone_environment,
            )

            # Create index if doesn't exist
            index_name = settings.pinecone_index_name
            
            if index_name not in pinecone.list_indexes():
                logger.info("creating_pinecone_index", name=index_name)
                pinecone.create_index(
                    name=index_name,
                    dimension=384,  # all-MiniLM-L6-v2 embedding dimension
                    metric="cosine",
                )

            self.index = pinecone.Index(index_name)
            self.initialized = True
            logger.info("pinecone_initialized", index=index_name)

        except Exception as e:
            logger.error("pinecone_init_failed", error=str(e))
            self.initialized = False

    async def index_routing_log(
        self,
        log_id: str,
        query: str,
        embedding: List[float],
        model_used: str,
        latency_ms: float,
        success: bool,
        rating: Optional[float] = None,
    ) -> bool:
        """
        Index a routing decision in the vector database

        Args:
            log_id: Unique log ID
            query: Original query
            embedding: Query embedding vector
            model_used: Model that was used
            latency_ms: Response latency
            success: Whether request succeeded
            rating: User rating if available

        Returns:
            True if indexed successfully
        """
        if not self.initialized:
            return False

        try:
            metadata = {
                "query": query[:500],  # Truncate for storage
                "model_used": model_used,
                "latency_ms": latency_ms,
                "success": success,
                "rating": rating or 0.0,
            }

            self.index.upsert(
                vectors=[(log_id, embedding, metadata)]
            )

            logger.info(
                "routing_log_indexed",
                log_id=log_id,
                model=model_used,
            )

            return True

        except Exception as e:
            logger.error("indexing_failed", error=str(e))
            return False

    async def get_similar_queries(
        self,
        embedding: List[float],
        top_k: int = 5,
        min_similarity: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """
        Find similar historical queries

        Args:
            embedding: Query embedding to search for
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold

        Returns:
            List of similar queries with metadata
        """
        if not self.initialized:
            return []

        try:
            results = self.index.query(
                vector=embedding,
                top_k=top_k,
                include_metadata=True,
            )

            similar_queries = []

            for match in results["matches"]:
                if match["score"] >= min_similarity:
                    similar_queries.append({
                        "id": match["id"],
                        "similarity": match["score"],
                        "query": match["metadata"]["query"],
                        "model_used": match["metadata"]["model_used"],
                        "latency_ms": match["metadata"]["latency_ms"],
                        "success": match["metadata"]["success"],
                        "rating": match["metadata"].get("rating", 0.0),
                    })

            logger.info(
                "similar_queries_found",
                count=len(similar_queries),
                top_similarity=similar_queries[0]["similarity"] if similar_queries else 0,
            )

            return similar_queries

        except Exception as e:
            logger.error("similarity_search_failed", error=str(e))
            return []

    async def recommend_model(
        self,
        query_embedding: List[float],
        top_k: int = 10,
    ) -> Dict[str, float]:
        """
        Recommend models based on similar historical queries

        Args:
            query_embedding: Embedding of current query
            top_k: Number of similar queries to consider

        Returns:
            Dict mapping model names to recommendation scores
        """
        similar_queries = await self.get_similar_queries(query_embedding, top_k)

        if not similar_queries:
            return {}

        # Aggregate recommendations by model
        model_scores: Dict[str, List[float]] = {}

        for query in similar_queries:
            model = query["model_used"]
            
            # Score based on similarity, success, and rating
            score = (
                query["similarity"] * 0.5 +
                (1.0 if query["success"] else 0.0) * 0.3 +
                (query["rating"] / 5.0) * 0.2
            )

            if model not in model_scores:
                model_scores[model] = []
            
            model_scores[model].append(score)

        # Calculate average scores
        recommendations = {
            model: np.mean(scores)
            for model, scores in model_scores.items()
        }

        logger.info(
            "rag_recommendations_generated",
            models=list(recommendations.keys()),
            top_model=max(recommendations, key=recommendations.get) if recommendations else None,
        )

        return recommendations


async def populate_vector_db():
    """
    Utility script to populate vector database from existing routing logs
    Run this after Phase 4 to enable RAG routing
    """
    logger.info("populating_vector_db_started")

    rag_service = RAGService()

    if not rag_service.initialized:
        logger.error("rag_service_not_initialized")
        return

    # Fetch all routing logs with embeddings
    training_data = await data_collection_service.get_training_data(limit=50000)

    indexed_count = 0

    for i, entry in enumerate(training_data):
        embedding = entry.get("embedding")
        
        if embedding is None:
            # Generate embedding if not present
            embedding = await feature_extractor.generate_embedding(entry["query"])
            
            if embedding is None:
                continue

        success = await rag_service.index_routing_log(
            log_id=f"log_{i}",
            query=entry["query"],
            embedding=embedding,
            model_used=entry["model_used"],
            latency_ms=entry["latency_ms"],
            success=True,
        )

        if success:
            indexed_count += 1

        if (i + 1) % 100 == 0:
            logger.info("indexing_progress", processed=i + 1, indexed=indexed_count)

    logger.info(
        "vector_db_population_complete",
        total=len(training_data),
        indexed=indexed_count,
    )

    print(f"\n✅ Indexed {indexed_count} routing logs to vector database")


if __name__ == "__main__":
    asyncio.run(populate_vector_db())

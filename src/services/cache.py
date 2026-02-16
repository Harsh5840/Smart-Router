"""
Caching service using Redis
"""

import hashlib
import json
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import redis.asyncio as redis
from src.utils.logging import get_logger
from src.config import settings
from src.utils.metrics import CACHE_HITS, CACHE_MISSES

logger = get_logger(__name__)


# ============================================================================
# PHASE 8: Caching Service
# ============================================================================


class CacheService:
    """
    Redis-based caching for query responses
    Supports exact and semantic similarity matching
    """

    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self._initialize_redis()

    def _initialize_redis(self) -> None:
        """Initialize Redis connection"""
        if not settings.enable_caching:
            logger.info("caching_disabled")
            return

        try:
            self.redis_client = redis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            logger.info("redis_initialized", url=settings.redis_url.split("@")[0])
        except Exception as e:
            logger.error("redis_init_failed", error=str(e))
            self.redis_client = None

    def _generate_cache_key(self, query: str, model: str) -> str:
        """Generate cache key from query and model"""
        # Use hash to handle long queries
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        return f"cache:query:{model}:{query_hash}"

    async def get_cached_response(
        self,
        query: str,
        model: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached response for exact query match

        Args:
            query: User query
            model: Model name

        Returns:
            Cached response dict or None
        """
        if self.redis_client is None:
            return None

        try:
            cache_key = self._generate_cache_key(query, model)
            cached_data = await self.redis_client.get(cache_key)

            if cached_data:
                CACHE_HITS.inc()
                result = json.loads(cached_data)
                result["from_cache"] = True
                result["cache_timestamp"] = result.get("timestamp")

                logger.info(
                    "cache_hit",
                    model=model,
                    query_length=len(query),
                )

                # Increment hit count
                await self.redis_client.incr(f"{cache_key}:hits")

                return result
            else:
                CACHE_MISSES.inc()
                logger.debug("cache_miss", model=model)
                return None

        except Exception as e:
            logger.error("cache_get_failed", error=str(e))
            return None

    async def cache_response(
        self,
        query: str,
        model: str,
        response: str,
        latency_ms: float,
        ttl_hours: int = 24,
    ) -> bool:
        """
        Cache a response

        Args:
            query: User query
            model: Model used
            response: Model response
            latency_ms: Response latency
            ttl_hours: Time to live in hours

        Returns:
            True if cached successfully
        """
        if self.redis_client is None:
            return False

        try:
            cache_key = self._generate_cache_key(query, model)
            cache_data = {
                "query": query,
                "response": response,
                "model": model,
                "latency_ms": latency_ms,
                "timestamp": datetime.utcnow().isoformat(),
            }

            ttl_seconds = ttl_hours * 3600

            await self.redis_client.setex(
                cache_key,
                ttl_seconds,
                json.dumps(cache_data),
            )

            # Initialize hit counter
            await self.redis_client.setex(
                f"{cache_key}:hits",
                ttl_seconds,
                "0",
            )

            logger.info(
                "response_cached",
                model=model,
                ttl_hours=ttl_hours,
            )

            return True

        except Exception as e:
            logger.error("cache_set_failed", error=str(e))
            return False

    async def invalidate_cache(self, query: str, model: Optional[str] = None) -> int:
        """
        Invalidate cache entries

        Args:
            query: Query to invalidate
            model: Specific model or None for all models

        Returns:
            Number of keys deleted
        """
        if self.redis_client is None:
            return 0

        try:
            if model:
                cache_key = self._generate_cache_key(query, model)
                deleted = await self.redis_client.delete(cache_key)
            else:
                # Invalidate for all models
                query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
                pattern = f"cache:query:*:{query_hash}"
                keys = await self.redis_client.keys(pattern)
                deleted = await self.redis_client.delete(*keys) if keys else 0

            logger.info("cache_invalidated", query_hash=query_hash, count=deleted)
            return deleted

        except Exception as e:
            logger.error("cache_invalidation_failed", error=str(e))
            return 0

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if self.redis_client is None:
            return {"enabled": False}

        try:
            info = await self.redis_client.info("stats")
            
            # Count total cached queries
            pattern = "cache:query:*"
            cursor = 0
            total_keys = 0
            
            # Use scan for efficient counting
            while True:
                cursor, keys = await self.redis_client.scan(
                    cursor,
                    match=pattern,
                    count=100,
                )
                total_keys += len(keys)
                if cursor == 0:
                    break

            return {
                "enabled": True,
                "total_cached_queries": total_keys,
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
            }

        except Exception as e:
            logger.error("cache_stats_failed", error=str(e))
            return {"enabled": True, "error": str(e)}

    async def close(self) -> None:
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("redis_connection_closed")


# Global cache service instance
cache_service = CacheService()

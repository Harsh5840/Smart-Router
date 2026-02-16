"""
Data collection service for routing logs and feedback
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, func
from src.models.database import Base, RoutingLogDB, FeedbackDB, ModelPerformanceDB
from src.models.schemas import RoutingLog, FeedbackRequest
from src.config import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# PHASE 4: Data Collection Service
# ============================================================================


class DataCollectionService:
    """
    Collects and persists routing data for analytics and ML training
    Stores: routing decisions, performance metrics, user feedback
    """

    def __init__(self):
        self.engine = None
        self.async_session_factory = None
        self._initialize_db()

    def _initialize_db(self) -> None:
        """Initialize async database connection"""
        try:
            self.engine = create_async_engine(
                settings.database_url,
                echo=settings.app_env == "development",
                pool_pre_ping=True,
                pool_size=10,
                max_overflow=20,
            )

            self.async_session_factory = sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )

            logger.info("database_initialized", url=settings.database_url.split("@")[0])
        except Exception as e:
            logger.error("database_init_failed", error=str(e))
            # In production, this should be a critical error
            # For now, we'll continue without persistence

    async def create_tables(self) -> None:
        """Create all database tables"""
        if self.engine is None:
            logger.warning("database_not_initialized")
            return

        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("database_tables_created")
        except Exception as e:
            logger.error("table_creation_failed", error=str(e))

    async def log_routing_decision(
        self,
        query: str,
        user_id: str,
        features: Dict[str, Any],
        model_used: str,
        latency_ms: float,
        cost_estimate: float,
        success: bool = True,
        embedding: Optional[List[float]] = None,
    ) -> Optional[int]:
        """
        Log a routing decision to the database

        Returns:
            ID of the created log entry
        """
        if self.async_session_factory is None:
            logger.warning("database_not_available")
            return None

        try:
            async with self.async_session_factory() as session:
                log_entry = RoutingLogDB(
                    query=query,
                    user_id=user_id,
                    features=features,
                    model_used=model_used,
                    latency_ms=latency_ms,
                    cost_estimate=cost_estimate,
                    success=success,
                    timestamp=datetime.utcnow(),
                    embedding=embedding,
                )

                session.add(log_entry)
                await session.commit()
                await session.refresh(log_entry)

                logger.info(
                    "routing_logged",
                    log_id=log_entry.id,
                    model=model_used,
                    user=user_id,
                )

                # Update model performance stats
                await self._update_model_performance(
                    session, model_used, latency_ms, cost_estimate, success
                )

                return log_entry.id

        except Exception as e:
            logger.error("routing_log_failed", error=str(e))
            return None

    async def _update_model_performance(
        self,
        session: AsyncSession,
        model_name: str,
        latency_ms: float,
        cost: float,
        success: bool,
    ) -> None:
        """Update aggregate model performance metrics"""
        try:
            # Get or create performance record
            result = await session.execute(
                select(ModelPerformanceDB).where(
                    ModelPerformanceDB.model_name == model_name
                )
            )
            perf = result.scalar_one_or_none()

            if perf is None:
                perf = ModelPerformanceDB(
                    model_name=model_name,
                    total_requests=0,
                    success_count=0,
                    total_latency_ms=0.0,
                    total_cost=0.0,
                )
                session.add(perf)

            # Update metrics
            perf.total_requests += 1
            perf.success_count += 1 if success else 0
            perf.total_latency_ms += latency_ms
            perf.total_cost += cost
            perf.last_updated = datetime.utcnow()

            await session.commit()

        except Exception as e:
            logger.error("performance_update_failed", model=model_name, error=str(e))

    async def save_feedback(
        self,
        request_id: str,
        rating: int,
        comment: Optional[str] = None,
    ) -> bool:
        """
        Save user feedback for a request

        Args:
            request_id: ID of the routing log entry
            rating: Rating from 1-5
            comment: Optional text feedback

        Returns:
            True if saved successfully
        """
        if self.async_session_factory is None:
            logger.warning("database_not_available")
            return False

        try:
            async with self.async_session_factory() as session:
                feedback = FeedbackDB(
                    request_id=request_id,
                    rating=rating,
                    comment=comment,
                    timestamp=datetime.utcnow(),
                )

                session.add(feedback)
                await session.commit()

                logger.info("feedback_saved", request_id=request_id, rating=rating)
                return True

        except Exception as e:
            logger.error("feedback_save_failed", error=str(e))
            return False

    async def get_model_performance(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get performance stats for a specific model"""
        if self.async_session_factory is None:
            return None

        try:
            async with self.async_session_factory() as session:
                result = await session.execute(
                    select(ModelPerformanceDB).where(
                        ModelPerformanceDB.model_name == model_name
                    )
                )
                perf = result.scalar_one_or_none()

                if perf is None:
                    return None

                avg_latency = (
                    perf.total_latency_ms / perf.total_requests
                    if perf.total_requests > 0
                    else 0.0
                )

                success_rate = (
                    perf.success_count / perf.total_requests
                    if perf.total_requests > 0
                    else 0.0
                )

                return {
                    "model_name": model_name,
                    "total_requests": perf.total_requests,
                    "success_rate": success_rate,
                    "avg_latency_ms": avg_latency,
                    "total_cost": perf.total_cost,
                    "avg_rating": perf.avg_rating,
                }

        except Exception as e:
            logger.error("performance_fetch_failed", model=model_name, error=str(e))
            return None

    async def get_training_data(
        self,
        limit: int = 10000,
        min_rating: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve data for ML training
        
        PHASE 5: Used for training the ML classifier

        Returns:
            List of routing logs suitable for training
        """
        if self.async_session_factory is None:
            return []

        try:
            async with self.async_session_factory() as session:
                query = select(RoutingLogDB).where(RoutingLogDB.success == True)

                if min_rating:
                    # Join with feedback if rating filter provided
                    pass  # TODO: Implement join query

                query = query.limit(limit)
                result = await session.execute(query)
                logs = result.scalars().all()

                training_data = [
                    {
                        "query": log.query,
                        "features": log.features,
                        "model_used": log.model_used,
                        "latency_ms": log.latency_ms,
                        "embedding": log.embedding,
                    }
                    for log in logs
                ]

                logger.info("training_data_retrieved", count=len(training_data))
                return training_data

        except Exception as e:
            logger.error("training_data_fetch_failed", error=str(e))
            return []

    async def close(self) -> None:
        """Close database connection"""
        if self.engine:
            await self.engine.dispose()
            logger.info("database_connection_closed")


# Global data collection service instance
data_collection_service = DataCollectionService()

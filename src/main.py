"""
FastAPI application entry point
Production-grade LLM Router with intelligent request routing
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from src.api.endpoints import api_router
from src.services.data_collection import data_collection_service
from src.services.cache import cache_service
from src.utils.logging import setup_logging, get_logger
from src.config import settings

# Initialize logging
setup_logging()
logger = get_logger(__name__)


# ============================================================================
# Application Lifecycle Management
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events
    """
    # Startup
    logger.info(
        "application_starting",
        env=settings.app_env,
        version="1.0.0",
    )

    # Initialize database tables
    await data_collection_service.create_tables()

    logger.info("application_ready")

    yield

    # Shutdown
    logger.info("application_shutting_down")

    # Close connections
    await data_collection_service.close()
    await cache_service.close()

    logger.info("application_stopped")


# ============================================================================
# Application Factory
# ============================================================================


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application
    """
    app = FastAPI(
        title="Intelligent LLM Router",
        description="Production-grade multi-model LLM chat router with intelligent routing",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ========================================================================
    # Middleware Configuration
    # ========================================================================

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.app_env == "development" else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Trusted host middleware (production only)
    if settings.app_env == "production":
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*"],  # Configure based on deployment
        )

    # ========================================================================
    # Route Registration
    # ========================================================================

    app.include_router(
        api_router,
        prefix="/api/v1",
        tags=["chat"],
    )

    # ========================================================================
    # Root Endpoint
    # ========================================================================

    @app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "service": "LLM Router",
            "version": "1.0.0",
            "status": "operational",
            "docs": "/docs",
        }

    return app


# ============================================================================
# Application Instance
# ============================================================================

app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.app_env == "development",
        log_level=settings.log_level.lower(),
    )

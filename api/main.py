"""
FastAPI Main Application
========================
Entry point for the Enterprise RAG Ingestion Pipeline API.

Features:
- Document ingestion endpoints
- Health checks
- CORS configuration
- OpenAPI documentation
"""

from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import documents
from config.settings import get_settings
from storage.postgres import get_db_manager
from storage.s3 import get_s3_storage


# ============================================================================
# Lifespan Management
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events.
    """
    # Startup
    settings = get_settings()
    
    # Initialize database
    db_manager = get_db_manager()
    if settings.is_development:
        # Auto-create tables in development
        await db_manager.create_tables()
    
    # Initialize S3
    storage = get_s3_storage()
    try:
        await storage.ensure_bucket_exists()
    except Exception as e:
        # Log but don't fail startup - S3 might not be available locally
        print(f"Warning: Could not ensure S3 bucket exists: {e}")
    
    yield
    
    # Shutdown
    await db_manager.close()


# ============================================================================
# Application Factory
# ============================================================================

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application
    """
    settings = get_settings()
    
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="""
        ## Enterprise RAG Ingestion Pipeline API
        
        Multi-modal document ingestion for enterprise RAG applications.
        
        ### Features
        - **PDF, Text, and Image Support** - V1 scope covers text, images, and PDFs
        - **RBAC Enforcement** - Role and user-based access control
        - **Document Versioning** - Automatic versioning with soft delete
        - **Async Processing** - Background ingestion via Celery
        
        ### Authentication
        
        Pass the following headers with each request:
        - `X-Tenant-ID`: Your tenant/organization identifier
        - `X-User-ID`: Your user identifier
        - `X-User-Roles`: Comma-separated list of roles (optional)
        - `X-Is-Admin`: Set to "true" for admin access (optional)
        """,
        openapi_tags=[
            {
                "name": "Documents",
                "description": "Document ingestion and management operations"
            },
            {
                "name": "Health",
                "description": "Health check endpoints"
            }
        ],
        lifespan=lifespan,
        debug=settings.debug
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.is_development else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Register routes
    app.include_router(documents.router, prefix="/api/v1")
    
    # Register health check routes
    register_health_routes(app)
    
    # Register exception handlers
    register_exception_handlers(app)
    
    return app


# ============================================================================
# Health Check Routes
# ============================================================================

def register_health_routes(app: FastAPI) -> None:
    """Register health check endpoints."""
    
    @app.get(
        "/health",
        tags=["Health"],
        summary="Health Check",
        description="Basic health check endpoint"
    )
    async def health_check() -> Dict[str, Any]:
        """Basic health check."""
        return {
            "status": "healthy",
            "service": "enterprise-rag-ingestion"
        }
    
    @app.get(
        "/health/ready",
        tags=["Health"],
        summary="Readiness Check",
        description="Check if the service is ready to accept requests"
    )
    async def readiness_check() -> Dict[str, Any]:
        """
        Readiness check - verifies dependencies are available.
        """
        checks = {
            "database": False,
            "s3": False
        }
        
        # Check database
        try:
            db_manager = get_db_manager()
            async with db_manager.async_session() as session:
                await session.execute("SELECT 1")
            checks["database"] = True
        except Exception as e:
            checks["database_error"] = str(e)
        
        # Check S3
        try:
            storage = get_s3_storage()
            # Just check if we can create a client
            async with storage._get_client() as client:
                await client.list_buckets()
            checks["s3"] = True
        except Exception as e:
            checks["s3_error"] = str(e)
        
        all_healthy = all(v for k, v in checks.items() if not k.endswith("_error"))
        
        return {
            "status": "ready" if all_healthy else "degraded",
            "checks": checks
        }
    
    @app.get(
        "/health/live",
        tags=["Health"],
        summary="Liveness Check",
        description="Check if the service is alive"
    )
    async def liveness_check() -> Dict[str, str]:
        """Liveness check - just confirms the service is running."""
        return {"status": "alive"}


# ============================================================================
# Exception Handlers
# ============================================================================

def register_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers."""
    
    @app.exception_handler(Exception)
    async def global_exception_handler(
        request: Request, 
        exc: Exception
    ) -> JSONResponse:
        """Handle unexpected exceptions."""
        settings = get_settings()
        
        # Log the error
        print(f"Unhandled exception: {exc}")
        
        # Return appropriate response
        if settings.debug:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "detail": str(exc),
                    "type": type(exc).__name__
                }
            )
        else:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Internal server error"}
            )


# ============================================================================
# Application Instance
# ============================================================================

app = create_app()


# ============================================================================
# Development Runner
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.is_development,
        workers=1 if settings.is_development else settings.api_workers
    )

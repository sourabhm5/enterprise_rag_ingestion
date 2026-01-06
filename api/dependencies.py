"""
API Dependencies
================
FastAPI dependency injection for the Enterprise RAG Ingestion Pipeline.

Provides:
- Database session injection
- S3 storage injection
- Settings injection
- Authentication/RBAC context
"""

from typing import Annotated, Optional
from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import Settings, get_settings
from storage.postgres import get_db_session, get_db_manager, DatabaseManager
from storage.s3 import S3Storage, get_s3_storage


# ============================================================================
# Settings Dependency
# ============================================================================

def get_settings_dep() -> Settings:
    """Get application settings."""
    return get_settings()


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]


# ============================================================================
# Database Dependencies
# ============================================================================

async def get_db() -> AsyncSession:
    """Get database session for request."""
    db_manager = get_db_manager()
    async with db_manager.async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


DBSessionDep = Annotated[AsyncSession, Depends(get_db)]


# ============================================================================
# S3 Storage Dependency
# ============================================================================

def get_storage() -> S3Storage:
    """Get S3 storage instance."""
    return get_s3_storage()


S3StorageDep = Annotated[S3Storage, Depends(get_storage)]


# ============================================================================
# Authentication Context
# ============================================================================

@dataclass
class AuthContext:
    """
    Authentication context for the current request.
    
    In a production system, this would be populated from JWT/OAuth tokens.
    For now, we extract from headers for simplicity.
    """
    tenant_id: str
    user_id: str
    roles: list[str]
    is_admin: bool = False
    
    def has_role(self, role: str) -> bool:
        """Check if user has a specific role."""
        return role in self.roles or self.is_admin
    
    def can_access_classification(self, classification: str) -> bool:
        """Check if user can access a classification level."""
        # Simple hierarchy: RESTRICTED > CONFIDENTIAL > INTERNAL > PUBLIC
        hierarchy = ["PUBLIC", "INTERNAL", "CONFIDENTIAL", "RESTRICTED"]
        
        if self.is_admin:
            return True
        
        # User's max classification based on roles
        user_max = "PUBLIC"
        if "restricted_access" in self.roles:
            user_max = "RESTRICTED"
        elif "confidential_access" in self.roles:
            user_max = "CONFIDENTIAL"
        elif "internal_access" in self.roles:
            user_max = "INTERNAL"
        
        try:
            user_level = hierarchy.index(user_max)
            doc_level = hierarchy.index(classification)
            return user_level >= doc_level
        except ValueError:
            return False


async def get_auth_context(
    x_tenant_id: Annotated[str, Header(description="Tenant identifier")],
    x_user_id: Annotated[str, Header(description="User identifier")],
    x_user_roles: Annotated[
        Optional[str], 
        Header(description="Comma-separated list of user roles")
    ] = None,
    x_is_admin: Annotated[
        Optional[str], 
        Header(description="Whether user is admin")
    ] = "false",
) -> AuthContext:
    """
    Extract authentication context from request headers.
    
    In production, this would validate JWT tokens and extract claims.
    """
    if not x_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-Tenant-ID header"
        )
    
    if not x_user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-User-ID header"
        )
    
    roles = []
    if x_user_roles:
        roles = [r.strip() for r in x_user_roles.split(",") if r.strip()]
    
    is_admin = x_is_admin.lower() in ("true", "1", "yes")
    
    return AuthContext(
        tenant_id=x_tenant_id,
        user_id=x_user_id,
        roles=roles,
        is_admin=is_admin
    )


AuthContextDep = Annotated[AuthContext, Depends(get_auth_context)]


# ============================================================================
# Combined Dependencies
# ============================================================================

@dataclass
class RequestContext:
    """Combined request context with all dependencies."""
    db: AsyncSession
    storage: S3Storage
    settings: Settings
    auth: AuthContext


async def get_request_context(
    db: DBSessionDep,
    storage: S3StorageDep,
    settings: SettingsDep,
    auth: AuthContextDep,
) -> RequestContext:
    """Get combined request context."""
    return RequestContext(
        db=db,
        storage=storage,
        settings=settings,
        auth=auth
    )


RequestContextDep = Annotated[RequestContext, Depends(get_request_context)]


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "SettingsDep",
    "DBSessionDep",
    "S3StorageDep",
    "AuthContext",
    "AuthContextDep",
    "RequestContext",
    "RequestContextDep",
    "get_settings_dep",
    "get_db",
    "get_storage",
    "get_auth_context",
    "get_request_context",
]

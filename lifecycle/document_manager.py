"""
MODULE 17 — DELETE & UPDATE HANDLERS
=====================================
Document lifecycle management for the Enterprise RAG Ingestion Pipeline.

DELETE:
- Soft delete Postgres rows
- Delete vectors
- Optionally delete S3 objects

UPDATE:
- Increment version
- Re-ingest
- Retain audit trail

This module provides comprehensive lifecycle management for documents,
ensuring consistency across all storage systems (PostgreSQL, S3, Vector DB).
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import get_settings
from storage.postgres import (
    Document,
    IngestionJob,
    get_active_document,
    soft_delete_old_versions,
)
from storage.s3 import S3Storage, get_s3_storage
from storage.vector_store import (
    VectorStoreService,
    SearchFilter,
    get_vector_store_service,
)


# ============================================================================
# Lifecycle Types
# ============================================================================

class LifecycleAction(str, Enum):
    """Document lifecycle actions."""
    CREATE = "create"
    UPDATE = "update"
    SOFT_DELETE = "soft_delete"
    HARD_DELETE = "hard_delete"
    RESTORE = "restore"
    VERSION_CLEANUP = "version_cleanup"


class LifecycleStatus(str, Enum):
    """Status of lifecycle operation."""
    SUCCESS = "success"
    PARTIAL = "partial"  # Some operations failed
    FAILED = "failed"


@dataclass
class LifecycleResult:
    """Result of a lifecycle operation."""
    action: LifecycleAction
    status: LifecycleStatus
    document_id: str
    tenant_id: str
    version: Optional[int] = None
    
    # Detailed results
    postgres_affected: int = 0
    vectors_deleted: int = 0
    s3_objects_deleted: int = 0
    
    # Errors
    errors: List[Dict[str, Any]] = field(default_factory=list)
    
    # Timing
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "action": self.action.value,
            "status": self.status.value,
            "document_id": self.document_id,
            "tenant_id": self.tenant_id,
            "version": self.version,
            "postgres_affected": self.postgres_affected,
            "vectors_deleted": self.vectors_deleted,
            "s3_objects_deleted": self.s3_objects_deleted,
            "errors": self.errors,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_ms": self.duration_ms,
        }


@dataclass
class AuditEntry:
    """Audit log entry for lifecycle actions."""
    action: LifecycleAction
    document_id: str
    tenant_id: str
    user_id: str
    version: Optional[int]
    timestamp: str
    details: Dict[str, Any] = field(default_factory=dict)
    result: Optional[LifecycleResult] = None


# ============================================================================
# Document Manager
# ============================================================================

class DocumentManager:
    """
    Manages document lifecycle operations.
    
    Provides:
    - Soft delete with cascade to vectors
    - Hard delete with optional S3 cleanup
    - Version management
    - Update with re-ingestion
    - Audit trail
    """
    
    def __init__(
        self,
        db_session: AsyncSession,
        s3_storage: Optional[S3Storage] = None,
        vector_service: Optional[VectorStoreService] = None
    ):
        """
        Initialize document manager.
        
        Args:
            db_session: Database session
            s3_storage: S3 storage instance
            vector_service: Vector store service
        """
        self.db = db_session
        self.s3 = s3_storage or get_s3_storage()
        self.vector_service = vector_service or get_vector_store_service()
        self.settings = get_settings()
        
        # Audit log (in production, this would go to a proper audit system)
        self._audit_log: List[AuditEntry] = []
    
    # =========================================================================
    # Delete Operations
    # =========================================================================
    
    async def soft_delete(
        self,
        document_id: str,
        tenant_id: str,
        user_id: str,
        delete_vectors: bool = True,
        version: Optional[int] = None
    ) -> LifecycleResult:
        """
        Soft delete a document.
        
        Marks the document as SOFT_DELETED in PostgreSQL and
        optionally deletes vectors from the vector store.
        
        Args:
            document_id: Document ID
            tenant_id: Tenant ID
            user_id: User performing the action
            delete_vectors: Whether to delete vectors
            version: Specific version to delete (None = all versions)
            
        Returns:
            LifecycleResult
        """
        import time
        start_time = time.time()
        
        result = LifecycleResult(
            action=LifecycleAction.SOFT_DELETE,
            status=LifecycleStatus.SUCCESS,
            document_id=document_id,
            tenant_id=tenant_id,
            version=version,
            started_at=datetime.utcnow().isoformat(),
        )
        
        try:
            # Soft delete in PostgreSQL
            if version is not None:
                # Delete specific version
                stmt = (
                    update(Document)
                    .where(
                        Document.document_id == document_id,
                        Document.tenant_id == tenant_id,
                        Document.document_version == version,
                        Document.status == "ACTIVE"
                    )
                    .values(status="SOFT_DELETED")
                )
            else:
                # Delete all versions
                stmt = (
                    update(Document)
                    .where(
                        Document.document_id == document_id,
                        Document.tenant_id == tenant_id,
                        Document.status == "ACTIVE"
                    )
                    .values(status="SOFT_DELETED")
                )
            
            db_result = await self.db.execute(stmt)
            result.postgres_affected = db_result.rowcount
            await self.db.commit()
            
            # Delete vectors
            if delete_vectors:
                try:
                    vectors_deleted = await self.vector_service.delete_document(
                        document_id=document_id,
                        tenant_id=tenant_id,
                        version=version
                    )
                    result.vectors_deleted = vectors_deleted
                except Exception as e:
                    result.errors.append({
                        "component": "vector_store",
                        "error": str(e)
                    })
                    result.status = LifecycleStatus.PARTIAL
            
        except Exception as e:
            result.status = LifecycleStatus.FAILED
            result.errors.append({
                "component": "postgres",
                "error": str(e)
            })
            await self.db.rollback()
        
        # Finalize
        result.completed_at = datetime.utcnow().isoformat()
        result.duration_ms = (time.time() - start_time) * 1000
        
        # Audit
        self._log_audit(
            action=LifecycleAction.SOFT_DELETE,
            document_id=document_id,
            tenant_id=tenant_id,
            user_id=user_id,
            version=version,
            result=result
        )
        
        return result
    
    async def hard_delete(
        self,
        document_id: str,
        tenant_id: str,
        user_id: str,
        delete_s3: bool = False,
        version: Optional[int] = None
    ) -> LifecycleResult:
        """
        Hard delete a document.
        
        Permanently removes document from PostgreSQL.
        Optionally deletes S3 objects.
        
        Args:
            document_id: Document ID
            tenant_id: Tenant ID
            user_id: User performing the action
            delete_s3: Whether to delete S3 objects
            version: Specific version to delete (None = all versions)
            
        Returns:
            LifecycleResult
        """
        import time
        start_time = time.time()
        
        result = LifecycleResult(
            action=LifecycleAction.HARD_DELETE,
            status=LifecycleStatus.SUCCESS,
            document_id=document_id,
            tenant_id=tenant_id,
            version=version,
            started_at=datetime.utcnow().isoformat(),
        )
        
        try:
            # Get documents to delete (for S3 paths)
            if version is not None:
                query = select(Document).where(
                    Document.document_id == document_id,
                    Document.tenant_id == tenant_id,
                    Document.document_version == version
                )
            else:
                query = select(Document).where(
                    Document.document_id == document_id,
                    Document.tenant_id == tenant_id
                )
            
            docs_result = await self.db.execute(query)
            documents = docs_result.scalars().all()
            
            # Delete vectors first
            try:
                vectors_deleted = await self.vector_service.delete_document(
                    document_id=document_id,
                    tenant_id=tenant_id,
                    version=version
                )
                result.vectors_deleted = vectors_deleted
            except Exception as e:
                result.errors.append({
                    "component": "vector_store",
                    "error": str(e)
                })
            
            # Delete S3 objects if requested
            if delete_s3:
                for doc in documents:
                    try:
                        deleted = await self.s3.delete_document_version(
                            tenant_id=tenant_id,
                            document_id=document_id,
                            version=doc.document_version
                        )
                        result.s3_objects_deleted += deleted
                    except Exception as e:
                        result.errors.append({
                            "component": "s3",
                            "error": str(e),
                            "version": doc.document_version
                        })
            
            # Delete from PostgreSQL
            for doc in documents:
                await self.db.delete(doc)
            
            result.postgres_affected = len(documents)
            await self.db.commit()
            
            if result.errors:
                result.status = LifecycleStatus.PARTIAL
                
        except Exception as e:
            result.status = LifecycleStatus.FAILED
            result.errors.append({
                "component": "postgres",
                "error": str(e)
            })
            await self.db.rollback()
        
        # Finalize
        result.completed_at = datetime.utcnow().isoformat()
        result.duration_ms = (time.time() - start_time) * 1000
        
        # Audit
        self._log_audit(
            action=LifecycleAction.HARD_DELETE,
            document_id=document_id,
            tenant_id=tenant_id,
            user_id=user_id,
            version=version,
            result=result
        )
        
        return result
    
    # =========================================================================
    # Restore Operations
    # =========================================================================
    
    async def restore(
        self,
        document_id: str,
        tenant_id: str,
        user_id: str,
        version: Optional[int] = None
    ) -> LifecycleResult:
        """
        Restore a soft-deleted document.
        
        Args:
            document_id: Document ID
            tenant_id: Tenant ID
            user_id: User performing the action
            version: Specific version to restore (None = latest)
            
        Returns:
            LifecycleResult
        """
        import time
        start_time = time.time()
        
        result = LifecycleResult(
            action=LifecycleAction.RESTORE,
            status=LifecycleStatus.SUCCESS,
            document_id=document_id,
            tenant_id=tenant_id,
            version=version,
            started_at=datetime.utcnow().isoformat(),
        )
        
        try:
            if version is not None:
                stmt = (
                    update(Document)
                    .where(
                        Document.document_id == document_id,
                        Document.tenant_id == tenant_id,
                        Document.document_version == version,
                        Document.status == "SOFT_DELETED"
                    )
                    .values(status="ACTIVE")
                )
            else:
                # Restore only the latest soft-deleted version
                subquery = (
                    select(Document.document_version)
                    .where(
                        Document.document_id == document_id,
                        Document.tenant_id == tenant_id,
                        Document.status == "SOFT_DELETED"
                    )
                    .order_by(Document.document_version.desc())
                    .limit(1)
                    .scalar_subquery()
                )
                
                stmt = (
                    update(Document)
                    .where(
                        Document.document_id == document_id,
                        Document.tenant_id == tenant_id,
                        Document.document_version == subquery
                    )
                    .values(status="ACTIVE")
                )
            
            db_result = await self.db.execute(stmt)
            result.postgres_affected = db_result.rowcount
            await self.db.commit()
            
            if result.postgres_affected == 0:
                result.status = LifecycleStatus.FAILED
                result.errors.append({
                    "component": "postgres",
                    "error": "No soft-deleted document found to restore"
                })
                
        except Exception as e:
            result.status = LifecycleStatus.FAILED
            result.errors.append({
                "component": "postgres",
                "error": str(e)
            })
            await self.db.rollback()
        
        # Finalize
        result.completed_at = datetime.utcnow().isoformat()
        result.duration_ms = (time.time() - start_time) * 1000
        
        # Audit
        self._log_audit(
            action=LifecycleAction.RESTORE,
            document_id=document_id,
            tenant_id=tenant_id,
            user_id=user_id,
            version=version,
            result=result
        )
        
        return result
    
    # =========================================================================
    # Version Management
    # =========================================================================
    
    async def cleanup_old_versions(
        self,
        document_id: str,
        tenant_id: str,
        user_id: str,
        keep_versions: int = 5,
        delete_s3: bool = False
    ) -> LifecycleResult:
        """
        Clean up old document versions.
        
        Keeps the specified number of most recent versions and
        soft-deletes or hard-deletes older versions.
        
        Args:
            document_id: Document ID
            tenant_id: Tenant ID
            user_id: User performing the action
            keep_versions: Number of versions to keep
            delete_s3: Whether to delete S3 objects for old versions
            
        Returns:
            LifecycleResult
        """
        import time
        start_time = time.time()
        
        result = LifecycleResult(
            action=LifecycleAction.VERSION_CLEANUP,
            status=LifecycleStatus.SUCCESS,
            document_id=document_id,
            tenant_id=tenant_id,
            started_at=datetime.utcnow().isoformat(),
        )
        
        try:
            # Get all versions
            query = (
                select(Document)
                .where(
                    Document.document_id == document_id,
                    Document.tenant_id == tenant_id
                )
                .order_by(Document.document_version.desc())
            )
            
            docs_result = await self.db.execute(query)
            documents = docs_result.scalars().all()
            
            # Identify versions to delete
            versions_to_delete = documents[keep_versions:]
            
            for doc in versions_to_delete:
                # Soft delete in PostgreSQL
                doc.status = "SOFT_DELETED"
                result.postgres_affected += 1
                
                # Delete vectors
                try:
                    deleted = await self.vector_service.delete_document(
                        document_id=document_id,
                        tenant_id=tenant_id,
                        version=doc.document_version
                    )
                    result.vectors_deleted += deleted
                except Exception as e:
                    result.errors.append({
                        "component": "vector_store",
                        "error": str(e),
                        "version": doc.document_version
                    })
                
                # Delete S3 if requested
                if delete_s3:
                    try:
                        deleted = await self.s3.delete_document_version(
                            tenant_id=tenant_id,
                            document_id=document_id,
                            version=doc.document_version
                        )
                        result.s3_objects_deleted += deleted
                    except Exception as e:
                        result.errors.append({
                            "component": "s3",
                            "error": str(e),
                            "version": doc.document_version
                        })
            
            await self.db.commit()
            
            if result.errors:
                result.status = LifecycleStatus.PARTIAL
                
        except Exception as e:
            result.status = LifecycleStatus.FAILED
            result.errors.append({
                "component": "postgres",
                "error": str(e)
            })
            await self.db.rollback()
        
        # Finalize
        result.completed_at = datetime.utcnow().isoformat()
        result.duration_ms = (time.time() - start_time) * 1000
        
        # Audit
        self._log_audit(
            action=LifecycleAction.VERSION_CLEANUP,
            document_id=document_id,
            tenant_id=tenant_id,
            user_id=user_id,
            version=None,
            result=result,
            details={"keep_versions": keep_versions}
        )
        
        return result
    
    async def get_document_versions(
        self,
        document_id: str,
        tenant_id: str,
        include_deleted: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get all versions of a document.
        
        Args:
            document_id: Document ID
            tenant_id: Tenant ID
            include_deleted: Include soft-deleted versions
            
        Returns:
            List of version information
        """
        query = (
            select(Document)
            .where(
                Document.document_id == document_id,
                Document.tenant_id == tenant_id
            )
        )
        
        if not include_deleted:
            query = query.where(Document.status == "ACTIVE")
        
        query = query.order_by(Document.document_version.desc())
        
        result = await self.db.execute(query)
        documents = result.scalars().all()
        
        return [
            {
                "version": doc.document_version,
                "status": doc.status,
                "created_at": doc.created_at.isoformat() if doc.created_at else None,
                "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
                "content_hash": doc.content_hash,
                "file_size": doc.file_size_bytes,
            }
            for doc in documents
        ]
    
    # =========================================================================
    # Audit Trail
    # =========================================================================
    
    def _log_audit(
        self,
        action: LifecycleAction,
        document_id: str,
        tenant_id: str,
        user_id: str,
        version: Optional[int],
        result: LifecycleResult,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log an audit entry."""
        entry = AuditEntry(
            action=action,
            document_id=document_id,
            tenant_id=tenant_id,
            user_id=user_id,
            version=version,
            timestamp=datetime.utcnow().isoformat(),
            details=details or {},
            result=result
        )
        
        self._audit_log.append(entry)
        
        # In production, this would:
        # 1. Write to a dedicated audit table
        # 2. Send to an audit logging service
        # 3. Emit events for compliance systems
    
    def get_audit_log(
        self,
        document_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get audit log entries.
        
        Args:
            document_id: Filter by document ID
            limit: Maximum entries to return
            
        Returns:
            List of audit entries
        """
        entries = self._audit_log
        
        if document_id:
            entries = [e for e in entries if e.document_id == document_id]
        
        # Sort by timestamp descending
        entries = sorted(entries, key=lambda e: e.timestamp, reverse=True)
        
        return [
            {
                "action": e.action.value,
                "document_id": e.document_id,
                "tenant_id": e.tenant_id,
                "user_id": e.user_id,
                "version": e.version,
                "timestamp": e.timestamp,
                "details": e.details,
                "result": e.result.to_dict() if e.result else None,
            }
            for e in entries[:limit]
        ]


# ============================================================================
# Batch Operations
# ============================================================================

class BatchDocumentManager:
    """
    Batch operations for document lifecycle management.
    """
    
    def __init__(self, document_manager: DocumentManager):
        """
        Initialize batch manager.
        
        Args:
            document_manager: DocumentManager instance
        """
        self.manager = document_manager
    
    async def batch_soft_delete(
        self,
        document_ids: List[str],
        tenant_id: str,
        user_id: str,
        delete_vectors: bool = True
    ) -> List[LifecycleResult]:
        """
        Soft delete multiple documents.
        
        Args:
            document_ids: List of document IDs
            tenant_id: Tenant ID
            user_id: User performing the action
            delete_vectors: Whether to delete vectors
            
        Returns:
            List of LifecycleResult
        """
        results = []
        
        for doc_id in document_ids:
            result = await self.manager.soft_delete(
                document_id=doc_id,
                tenant_id=tenant_id,
                user_id=user_id,
                delete_vectors=delete_vectors
            )
            results.append(result)
        
        return results
    
    async def batch_hard_delete(
        self,
        document_ids: List[str],
        tenant_id: str,
        user_id: str,
        delete_s3: bool = False
    ) -> List[LifecycleResult]:
        """
        Hard delete multiple documents.
        
        Args:
            document_ids: List of document IDs
            tenant_id: Tenant ID
            user_id: User performing the action
            delete_s3: Whether to delete S3 objects
            
        Returns:
            List of LifecycleResult
        """
        results = []
        
        for doc_id in document_ids:
            result = await self.manager.hard_delete(
                document_id=doc_id,
                tenant_id=tenant_id,
                user_id=user_id,
                delete_s3=delete_s3
            )
            results.append(result)
        
        return results


# ============================================================================
# Factory Functions
# ============================================================================

async def get_document_manager(
    db_session: AsyncSession
) -> DocumentManager:
    """Get a DocumentManager instance."""
    return DocumentManager(db_session=db_session)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "LifecycleAction",
    "LifecycleStatus",
    "LifecycleResult",
    "AuditEntry",
    "DocumentManager",
    "BatchDocumentManager",
    "get_document_manager",
]

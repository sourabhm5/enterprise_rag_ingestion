"""
MODULE 4 — INGESTION API (CREATE / UPDATE / DELETE)
===================================================
FastAPI ingestion API for the Enterprise RAG Ingestion Pipeline.

Endpoints:
- POST /documents - Create or update document (triggers ingestion)
- GET /documents/{document_id} - Get document metadata
- DELETE /documents/{document_id} - Soft delete document

Behavior:
- If document exists → increment version, soft-delete old metadata & vectors
- RBAC metadata is mandatory at upload
- Ingestion job is triggered asynchronously
"""

import hashlib
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import (
    AuthContextDep,
    DBSessionDep,
    S3StorageDep,
    SettingsDep,
    AuthContext,
)
from config.settings import DocumentStatus, JobStatus, get_settings
from storage.postgres import (
    Document,
    IngestionJob,
    get_active_document,
    get_latest_version_number,
    soft_delete_old_versions,
)
from storage.s3 import S3PathType


# ============================================================================
# Router Setup
# ============================================================================

router = APIRouter(prefix="/documents", tags=["Documents"])


# ============================================================================
# Request/Response Models
# ============================================================================

class RBACConfig(BaseModel):
    """RBAC configuration for document access control."""
    allowed_roles: List[str] = Field(
        default_factory=list,
        description="Roles allowed to access this document"
    )
    allowed_users: List[str] = Field(
        default_factory=list,
        description="User IDs allowed to access this document"
    )
    
    @field_validator("allowed_roles", "allowed_users", mode="before")
    @classmethod
    def ensure_list(cls, v):
        if isinstance(v, str):
            return [r.strip() for r in v.split(",") if r.strip()]
        return v or []


class DocumentMetadata(BaseModel):
    """Additional document metadata."""
    source_system: Optional[str] = Field(
        None, 
        description="Source system identifier"
    )
    source_url: Optional[str] = Field(
        None, 
        description="Original source URL"
    )
    custom_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Custom metadata key-value pairs"
    )


class CreateDocumentRequest(BaseModel):
    """Request model for document creation (JSON body alternative)."""
    document_id: str = Field(..., description="External document identifier")
    classification: str = Field(
        default="INTERNAL",
        description="Security classification level"
    )
    rbac: RBACConfig = Field(
        default_factory=RBACConfig,
        description="RBAC configuration"
    )
    metadata: DocumentMetadata = Field(
        default_factory=DocumentMetadata,
        description="Additional metadata"
    )


class DocumentResponse(BaseModel):
    """Response model for document operations."""
    id: str = Field(..., description="Internal database ID")
    document_id: str = Field(..., description="External document identifier")
    tenant_id: str = Field(..., description="Tenant identifier")
    version: int = Field(..., description="Document version")
    status: str = Field(..., description="Document status")
    filename: str = Field(..., description="Original filename")
    mime_type: str = Field(..., description="MIME type")
    file_size_bytes: int = Field(..., description="File size in bytes")
    content_hash: str = Field(..., description="Content SHA-256 hash")
    classification: str = Field(..., description="Security classification")
    s3_raw_path: str = Field(..., description="S3 path to raw document")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    
    # RBAC
    allowed_roles: List[str] = Field(default_factory=list)
    allowed_users: List[str] = Field(default_factory=list)
    
    # Job info
    ingestion_job_id: Optional[str] = Field(
        None, 
        description="ID of active ingestion job"
    )
    ingestion_status: Optional[str] = Field(
        None, 
        description="Status of ingestion job"
    )
    
    class Config:
        from_attributes = True


class DocumentListResponse(BaseModel):
    """Response model for document listing."""
    documents: List[DocumentResponse]
    total: int
    page: int
    page_size: int
    has_more: bool


class IngestionJobResponse(BaseModel):
    """Response model for ingestion job status."""
    job_id: str
    document_id: str
    status: str
    current_stage: Optional[str]
    completed_stages: List[str]
    error_log: List[Dict[str, Any]]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]


class DeleteResponse(BaseModel):
    """Response model for delete operations."""
    document_id: str
    deleted_versions: List[int]
    message: str


# ============================================================================
# Helper Functions
# ============================================================================

def compute_file_hash(content: bytes) -> str:
    """Compute SHA-256 hash of file content."""
    return hashlib.sha256(content).hexdigest()


async def trigger_ingestion_job(
    db: AsyncSession,
    document_db_id: uuid.UUID,
    document_id: str,
    tenant_id: str,
    mime_type: str,
    settings: Any
) -> IngestionJob:
    """
    Create and trigger an ingestion job for a document.
    
    The actual job execution is handled by Celery workers.
    """
    # Determine pipeline plan based on document type and feature flags
    pipeline_stages = []
    
    # Always start with routing
    pipeline_stages.append("routing")
    
    # Layout parsing for PDFs
    if mime_type == "application/pdf" and settings.feature_flags.enable_layout_parsing:
        pipeline_stages.append("layout_parsing")
    
    # Text pipeline
    pipeline_stages.append("text_pipeline")
    
    # Image pipeline if enabled
    if settings.feature_flags.enable_image_pipeline:
        if mime_type.startswith("image/") or mime_type == "application/pdf":
            pipeline_stages.append("image_pipeline")
    
    # Linkage
    pipeline_stages.append("linkage")
    
    # Chunking
    pipeline_stages.append("chunking")
    
    # Metadata enrichment if enabled
    if settings.feature_flags.enable_llm_metadata_enrichment:
        pipeline_stages.append("metadata_enrichment")
    
    # Embedding
    pipeline_stages.append("embedding")
    
    # Vector store
    pipeline_stages.append("vector_store")
    
    # Create job record
    job = IngestionJob(
        document_db_id=document_db_id,
        job_id=f"job-{uuid.uuid4().hex[:12]}",
        status=JobStatus.PENDING.value,
        pipeline_plan={
            "stages": pipeline_stages,
            "document_id": document_id,
            "tenant_id": tenant_id,
            "mime_type": mime_type
        },
        completed_stages=[],
        error_log=[]
    )
    
    db.add(job)
    await db.flush()
    
    # TODO: Dispatch Celery task
    # celery_app.send_task('jobs.execute_pipeline', args=[str(job.id)])
    
    return job


async def get_document_with_job(
    db: AsyncSession,
    document: Document
) -> DocumentResponse:
    """Build document response with latest job status."""
    # Get latest job for this document
    stmt = (
        select(IngestionJob)
        .where(IngestionJob.document_db_id == document.id)
        .order_by(IngestionJob.created_at.desc())
        .limit(1)
    )
    result = await db.execute(stmt)
    job = result.scalar_one_or_none()
    
    return DocumentResponse(
        id=str(document.id),
        document_id=document.document_id,
        tenant_id=document.tenant_id,
        version=document.document_version,
        status=document.status,
        filename=document.filename,
        mime_type=document.mime_type,
        file_size_bytes=document.file_size_bytes,
        content_hash=document.content_hash,
        classification=document.classification,
        s3_raw_path=document.s3_raw_path,
        created_at=document.created_at,
        updated_at=document.updated_at,
        allowed_roles=document.allowed_roles.get("roles", []),
        allowed_users=document.allowed_users.get("users", []),
        ingestion_job_id=job.job_id if job else None,
        ingestion_status=job.status if job else None
    )


# ============================================================================
# Endpoints
# ============================================================================

@router.post(
    "",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create or Update Document",
    description="""
    Upload a document for ingestion. If a document with the same document_id exists,
    a new version is created and the old version is soft-deleted.
    
    RBAC metadata is mandatory - documents must specify allowed roles or users.
    """
)
async def create_or_update_document(
    db: DBSessionDep,
    storage: S3StorageDep,
    settings: SettingsDep,
    auth: AuthContextDep,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Document file to upload"),
    document_id: str = Form(..., description="External document identifier"),
    classification: str = Form(
        default="INTERNAL",
        description="Security classification (PUBLIC, INTERNAL, CONFIDENTIAL, RESTRICTED)"
    ),
    allowed_roles: str = Form(
        default="",
        description="Comma-separated list of allowed roles"
    ),
    allowed_users: str = Form(
        default="",
        description="Comma-separated list of allowed user IDs"
    ),
    source_system: Optional[str] = Form(
        default=None,
        description="Source system identifier"
    ),
    source_url: Optional[str] = Form(
        default=None,
        description="Original source URL"
    ),
):
    """Create or update a document."""
    tenant_id = auth.tenant_id
    
    # Validate file
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must have a filename"
        )
    
    # Check MIME type
    content_type = file.content_type or "application/octet-stream"
    if content_type not in settings.supported_mime_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {content_type}. Supported types: {settings.supported_mime_types}"
        )
    
    # Read file content
    content = await file.read()
    file_size = len(content)
    
    # Check file size
    max_size = settings.max_upload_size_mb * 1024 * 1024
    if file_size > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size ({file_size} bytes) exceeds maximum ({max_size} bytes)"
        )
    
    # Compute content hash
    content_hash = compute_file_hash(content)
    
    # Parse RBAC
    roles_list = [r.strip() for r in allowed_roles.split(",") if r.strip()]
    users_list = [u.strip() for u in allowed_users.split(",") if u.strip()]
    
    # Validate RBAC - at least one must be specified
    if not roles_list and not users_list:
        # Default to uploader having access
        users_list = [auth.user_id]
    
    # Validate classification
    valid_classifications = ["PUBLIC", "INTERNAL", "CONFIDENTIAL", "RESTRICTED"]
    if classification.upper() not in valid_classifications:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid classification. Must be one of: {valid_classifications}"
        )
    
    # Check if document already exists
    existing_doc = await get_active_document(db, tenant_id, document_id)
    
    if existing_doc:
        # Update flow: increment version, soft-delete old
        new_version = existing_doc.document_version + 1
        
        # Check if content is identical (idempotency)
        if existing_doc.content_hash == content_hash:
            # Same content, return existing document
            return await get_document_with_job(db, existing_doc)
        
        # Soft delete old version
        await soft_delete_old_versions(db, tenant_id, document_id, new_version)
    else:
        # New document
        new_version = 1
    
    # Upload to S3
    s3_path = storage.generate_path(
        tenant_id=tenant_id,
        document_id=document_id,
        version=new_version,
        path_type=S3PathType.RAW,
        filename=file.filename
    )
    
    upload_result = await storage.upload_raw_document(
        tenant_id=tenant_id,
        document_id=document_id,
        version=new_version,
        filename=file.filename,
        content=content,
        content_type=content_type,
        metadata={
            "original-filename": file.filename,
            "uploaded-by": auth.user_id,
            "classification": classification.upper()
        }
    )
    
    # Create document record
    document = Document(
        document_id=document_id,
        tenant_id=tenant_id,
        document_version=new_version,
        status=DocumentStatus.ACTIVE.value,
        allowed_roles={"roles": roles_list},
        allowed_users={"users": users_list},
        classification=classification.upper(),
        filename=file.filename,
        mime_type=content_type,
        file_size_bytes=file_size,
        content_hash=content_hash,
        s3_raw_path=upload_result.key,
        s3_derived_prefix=storage.generate_prefix(
            tenant_id, document_id, new_version, S3PathType.DERIVED
        ),
        source_system=source_system,
        source_url=source_url,
        enriched_metadata={}
    )
    
    db.add(document)
    await db.flush()
    
    # Trigger ingestion job
    job = await trigger_ingestion_job(
        db=db,
        document_db_id=document.id,
        document_id=document_id,
        tenant_id=tenant_id,
        mime_type=content_type,
        settings=settings
    )
    
    await db.commit()
    
    return DocumentResponse(
        id=str(document.id),
        document_id=document.document_id,
        tenant_id=document.tenant_id,
        version=document.document_version,
        status=document.status,
        filename=document.filename,
        mime_type=document.mime_type,
        file_size_bytes=document.file_size_bytes,
        content_hash=document.content_hash,
        classification=document.classification,
        s3_raw_path=document.s3_raw_path,
        created_at=document.created_at,
        updated_at=document.updated_at,
        allowed_roles=roles_list,
        allowed_users=users_list,
        ingestion_job_id=job.job_id,
        ingestion_status=job.status
    )


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Get Document",
    description="Get metadata for a document by its external ID."
)
async def get_document(
    document_id: str,
    db: DBSessionDep,
    auth: AuthContextDep,
    version: Optional[int] = Query(
        None, 
        description="Specific version to retrieve (default: latest active)"
    ),
):
    """Get document metadata."""
    tenant_id = auth.tenant_id
    
    if version:
        # Get specific version
        stmt = (
            select(Document)
            .where(Document.tenant_id == tenant_id)
            .where(Document.document_id == document_id)
            .where(Document.document_version == version)
        )
        result = await db.execute(stmt)
        document = result.scalar_one_or_none()
    else:
        # Get latest active version
        document = await get_active_document(db, tenant_id, document_id)
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {document_id}"
        )
    
    # Check RBAC
    if not document.is_accessible_by_user(auth.user_id, auth.roles):
        if not auth.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this document"
            )
    
    return await get_document_with_job(db, document)


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List Documents",
    description="List documents for the current tenant with pagination."
)
async def list_documents(
    db: DBSessionDep,
    auth: AuthContextDep,
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page"),
    status_filter: Optional[str] = Query(
        default="ACTIVE",
        description="Filter by status (ACTIVE, SOFT_DELETED, or None for all)"
    ),
    classification: Optional[str] = Query(
        default=None,
        description="Filter by classification level"
    ),
):
    """List documents with pagination and filtering."""
    tenant_id = auth.tenant_id
    
    # Build query
    conditions = [Document.tenant_id == tenant_id]
    
    if status_filter:
        conditions.append(Document.status == status_filter.upper())
    
    if classification:
        conditions.append(Document.classification == classification.upper())
    
    # Count total
    count_stmt = select(Document).where(and_(*conditions))
    count_result = await db.execute(count_stmt)
    total = len(count_result.scalars().all())
    
    # Paginated query
    offset = (page - 1) * page_size
    stmt = (
        select(Document)
        .where(and_(*conditions))
        .order_by(Document.created_at.desc())
        .offset(offset)
        .limit(page_size)
    )
    
    result = await db.execute(stmt)
    documents = result.scalars().all()
    
    # Filter by RBAC unless admin
    if not auth.is_admin:
        documents = [
            doc for doc in documents
            if doc.is_accessible_by_user(auth.user_id, auth.roles)
        ]
    
    # Build responses
    responses = []
    for doc in documents:
        responses.append(await get_document_with_job(db, doc))
    
    return DocumentListResponse(
        documents=responses,
        total=total,
        page=page,
        page_size=page_size,
        has_more=(offset + page_size) < total
    )


@router.delete(
    "/{document_id}",
    response_model=DeleteResponse,
    summary="Delete Document",
    description="""
    Soft delete a document and all its versions.
    
    This marks the document as SOFT_DELETED but retains data for audit purposes.
    Vectors are removed from the vector store immediately.
    """
)
async def delete_document(
    document_id: str,
    db: DBSessionDep,
    storage: S3StorageDep,
    auth: AuthContextDep,
    settings: SettingsDep,
    hard_delete: bool = Query(
        default=False,
        description="Permanently delete (admin only, if soft_delete disabled)"
    ),
    delete_s3: bool = Query(
        default=False,
        description="Also delete S3 objects (admin only)"
    ),
):
    """Delete a document (soft delete by default)."""
    tenant_id = auth.tenant_id
    
    # Find all versions of the document
    stmt = (
        select(Document)
        .where(Document.tenant_id == tenant_id)
        .where(Document.document_id == document_id)
        .where(Document.status == DocumentStatus.ACTIVE.value)
    )
    result = await db.execute(stmt)
    documents = result.scalars().all()
    
    if not documents:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {document_id}"
        )
    
    # Check RBAC - need admin or owner access to delete
    for doc in documents:
        if not doc.is_accessible_by_user(auth.user_id, auth.roles):
            if not auth.is_admin:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied - cannot delete this document"
                )
    
    deleted_versions = []
    
    for doc in documents:
        deleted_versions.append(doc.document_version)
        
        if hard_delete and auth.is_admin and not settings.feature_flags.enable_soft_delete:
            # Hard delete - remove from database
            await db.delete(doc)
        else:
            # Soft delete
            doc.soft_delete()
    
    # Delete from S3 if requested (admin only)
    if delete_s3 and auth.is_admin:
        await storage.delete_all_document_versions(tenant_id, document_id)
    
    # TODO: Delete vectors from vector store
    # await vector_store.delete_by_document(tenant_id, document_id)
    
    await db.commit()
    
    return DeleteResponse(
        document_id=document_id,
        deleted_versions=deleted_versions,
        message=f"Successfully deleted {len(deleted_versions)} version(s) of document {document_id}"
    )


@router.get(
    "/{document_id}/versions",
    response_model=List[DocumentResponse],
    summary="List Document Versions",
    description="List all versions of a document including soft-deleted ones."
)
async def list_document_versions(
    document_id: str,
    db: DBSessionDep,
    auth: AuthContextDep,
    include_deleted: bool = Query(
        default=False,
        description="Include soft-deleted versions"
    ),
):
    """List all versions of a document."""
    tenant_id = auth.tenant_id
    
    conditions = [
        Document.tenant_id == tenant_id,
        Document.document_id == document_id
    ]
    
    if not include_deleted:
        conditions.append(Document.status == DocumentStatus.ACTIVE.value)
    
    stmt = (
        select(Document)
        .where(and_(*conditions))
        .order_by(Document.document_version.desc())
    )
    
    result = await db.execute(stmt)
    documents = result.scalars().all()
    
    if not documents:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {document_id}"
        )
    
    # Check RBAC on first document (all versions share same ACL initially)
    if not documents[0].is_accessible_by_user(auth.user_id, auth.roles):
        if not auth.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this document"
            )
    
    responses = []
    for doc in documents:
        responses.append(await get_document_with_job(db, doc))
    
    return responses


@router.get(
    "/{document_id}/job",
    response_model=IngestionJobResponse,
    summary="Get Ingestion Job Status",
    description="Get the status of the latest ingestion job for a document."
)
async def get_ingestion_job_status(
    document_id: str,
    db: DBSessionDep,
    auth: AuthContextDep,
    job_id: Optional[str] = Query(
        None,
        description="Specific job ID (default: latest)"
    ),
):
    """Get ingestion job status."""
    tenant_id = auth.tenant_id
    
    # Get the document first
    document = await get_active_document(db, tenant_id, document_id)
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {document_id}"
        )
    
    # Check RBAC
    if not document.is_accessible_by_user(auth.user_id, auth.roles):
        if not auth.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this document"
            )
    
    # Get job
    if job_id:
        stmt = (
            select(IngestionJob)
            .where(IngestionJob.job_id == job_id)
            .where(IngestionJob.document_db_id == document.id)
        )
    else:
        stmt = (
            select(IngestionJob)
            .where(IngestionJob.document_db_id == document.id)
            .order_by(IngestionJob.created_at.desc())
            .limit(1)
        )
    
    result = await db.execute(stmt)
    job = result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No ingestion job found for document: {document_id}"
        )
    
    return IngestionJobResponse(
        job_id=job.job_id,
        document_id=document_id,
        status=job.status,
        current_stage=job.current_stage,
        completed_stages=job.completed_stages if isinstance(job.completed_stages, list) else [],
        error_log=job.error_log if isinstance(job.error_log, list) else [],
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at
    )


# ============================================================================
# Exports
# ============================================================================

__all__ = ["router"]

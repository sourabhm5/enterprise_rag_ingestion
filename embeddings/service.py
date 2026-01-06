"""
MODULE 14 — EMBEDDING SERVICE
=============================
Embedding generation for the Enterprise RAG Ingestion Pipeline.

Requirements:
- Validate embedding dimension
- Cache by content hash
- Generate:
    - Text embeddings
    - Image caption embeddings

Supports multiple embedding providers:
- OpenAI
- Cohere (future)
- Local models (future)
"""

import asyncio
import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

from pipelines.base import BasePipeline
from schema.intermediate_representation import (
    Chunk,
    IngestionDocument,
    ProcessingStatus,
)
from config.settings import get_settings, EmbeddingModelConfig
from jobs.pipeline_executor import register_pipeline


# ============================================================================
# Embedding Result
# ============================================================================

@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    embedding: List[float]
    model: str
    dimensions: int
    content_hash: str
    cached: bool = False
    latency_ms: float = 0.0
    
    def validate_dimensions(self, expected: int) -> bool:
        """Validate embedding dimensions."""
        return len(self.embedding) == expected


# ============================================================================
# Embedding Cache
# ============================================================================

class EmbeddingCache:
    """
    Cache for embeddings by content hash.
    
    Uses Redis if available, falls back to in-memory cache.
    """
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        ttl_seconds: int = 86400 * 7,  # 7 days
        max_memory_items: int = 10000
    ):
        """
        Initialize embedding cache.
        
        Args:
            redis_url: Redis connection URL
            ttl_seconds: Cache TTL in seconds
            max_memory_items: Max items for in-memory fallback
        """
        self.redis_url = redis_url
        self.ttl_seconds = ttl_seconds
        self.max_memory_items = max_memory_items
        
        self._redis_client = None
        self._memory_cache: Dict[str, Tuple[List[float], datetime]] = {}
        self._redis_available = False
        
        if redis_url:
            self._init_redis()
    
    def _init_redis(self) -> None:
        """Initialize Redis connection."""
        try:
            import redis.asyncio as redis
            self._redis_client = redis.from_url(self.redis_url)
            self._redis_available = True
        except Exception:
            self._redis_available = False
    
    def _get_cache_key(self, content_hash: str, model: str) -> str:
        """Generate cache key."""
        return f"emb:{model}:{content_hash}"
    
    async def get(
        self,
        content_hash: str,
        model: str
    ) -> Optional[List[float]]:
        """
        Get embedding from cache.
        
        Args:
            content_hash: Content hash
            model: Model name
            
        Returns:
            Cached embedding or None
        """
        key = self._get_cache_key(content_hash, model)
        
        # Try Redis first
        if self._redis_available and self._redis_client:
            try:
                data = await self._redis_client.get(key)
                if data:
                    return json.loads(data)
            except Exception:
                pass
        
        # Fall back to memory cache
        if key in self._memory_cache:
            embedding, timestamp = self._memory_cache[key]
            if datetime.utcnow() - timestamp < timedelta(seconds=self.ttl_seconds):
                return embedding
            else:
                del self._memory_cache[key]
        
        return None
    
    async def set(
        self,
        content_hash: str,
        model: str,
        embedding: List[float]
    ) -> None:
        """
        Store embedding in cache.
        
        Args:
            content_hash: Content hash
            model: Model name
            embedding: Embedding vector
        """
        key = self._get_cache_key(content_hash, model)
        
        # Try Redis first
        if self._redis_available and self._redis_client:
            try:
                await self._redis_client.setex(
                    key,
                    self.ttl_seconds,
                    json.dumps(embedding)
                )
                return
            except Exception:
                pass
        
        # Fall back to memory cache
        if len(self._memory_cache) >= self.max_memory_items:
            # Evict oldest entries
            sorted_keys = sorted(
                self._memory_cache.keys(),
                key=lambda k: self._memory_cache[k][1]
            )
            for old_key in sorted_keys[:len(sorted_keys) // 2]:
                del self._memory_cache[old_key]
        
        self._memory_cache[key] = (embedding, datetime.utcnow())
    
    async def close(self) -> None:
        """Close cache connections."""
        if self._redis_client:
            await self._redis_client.close()


# ============================================================================
# Embedding Provider Interface
# ============================================================================

class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get model name."""
        pass
    
    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Get embedding dimensions."""
        pass
    
    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        pass
    
    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for batch of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        pass


# ============================================================================
# OpenAI Embedding Provider
# ============================================================================

class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    OpenAI embedding provider.
    
    Supports:
    - text-embedding-3-small (1536 dims)
    - text-embedding-3-large (3072 dims)
    - text-embedding-ada-002 (1536 dims)
    """
    
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        batch_size: int = 100
    ):
        """
        Initialize OpenAI embedding provider.
        
        Args:
            model: Model name
            api_key: OpenAI API key (optional, uses env var)
            batch_size: Max batch size for requests
        """
        self.model = model
        self.api_key = api_key
        self.batch_size = batch_size
        self._client = None
        self._dimensions = self.MODEL_DIMENSIONS.get(model, 1536)
    
    @property
    def model_name(self) -> str:
        return self.model
    
    @property
    def dimensions(self) -> int:
        return self._dimensions
    
    def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            import openai
            if self.api_key:
                self._client = openai.AsyncOpenAI(api_key=self.api_key)
            else:
                self._client = openai.AsyncOpenAI()
        return self._client
    
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for single text."""
        client = self._get_client()
        
        response = await client.embeddings.create(
            model=self.model,
            input=text
        )
        
        return response.data[0].embedding
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for batch of texts."""
        if not texts:
            return []
        
        client = self._get_client()
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            response = await client.embeddings.create(
                model=self.model,
                input=batch
            )
            
            # Sort by index to maintain order
            batch_embeddings = sorted(response.data, key=lambda x: x.index)
            embeddings.extend([e.embedding for e in batch_embeddings])
        
        return embeddings


# ============================================================================
# Mock Embedding Provider (for testing)
# ============================================================================

class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing."""
    
    def __init__(self, dimensions: int = 1536):
        """Initialize mock provider."""
        self._dimensions = dimensions
        self._model = "mock-embedding-model"
    
    @property
    def model_name(self) -> str:
        return self._model
    
    @property
    def dimensions(self) -> int:
        return self._dimensions
    
    async def embed_text(self, text: str) -> List[float]:
        """Generate deterministic mock embedding."""
        # Create deterministic embedding from text hash
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        
        embedding = []
        for i in range(self._dimensions):
            # Use hash bytes to generate float values
            byte_idx = i % 32
            value = int(text_hash[byte_idx * 2:(byte_idx * 2) + 2], 16) / 255.0
            # Normalize to [-1, 1]
            embedding.append((value * 2) - 1)
        
        return embedding
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate mock embeddings for batch."""
        return [await self.embed_text(t) for t in texts]


# ============================================================================
# Embedding Service
# ============================================================================

class EmbeddingService:
    """
    Main embedding service.
    
    Features:
    - Multiple provider support
    - Caching by content hash
    - Dimension validation
    - Batch processing
    """
    
    def __init__(
        self,
        provider: Optional[EmbeddingProvider] = None,
        cache: Optional[EmbeddingCache] = None,
        enable_cache: bool = True
    ):
        """
        Initialize embedding service.
        
        Args:
            provider: Embedding provider
            cache: Embedding cache
            enable_cache: Whether to use caching
        """
        settings = get_settings()
        
        # Initialize provider
        if provider:
            self.provider = provider
        else:
            # Use default from settings
            model_config = settings.get_embedding_model()
            self.provider = OpenAIEmbeddingProvider(
                model=model_config.name,
                batch_size=100
            )
        
        # Initialize cache
        self.enable_cache = enable_cache and settings.feature_flags.enable_embedding_cache
        if self.enable_cache:
            self.cache = cache or EmbeddingCache(
                redis_url=str(settings.redis.url) if settings.redis else None
            )
        else:
            self.cache = None
    
    def _compute_content_hash(self, text: str) -> str:
        """Compute content hash for caching."""
        return hashlib.sha256(text.encode()).hexdigest()
    
    async def embed_text(
        self,
        text: str,
        skip_cache: bool = False
    ) -> EmbeddingResult:
        """
        Generate embedding for text.
        
        Args:
            text: Text to embed
            skip_cache: Skip cache lookup
            
        Returns:
            EmbeddingResult
        """
        import time
        start_time = time.time()
        
        content_hash = self._compute_content_hash(text)
        
        # Try cache
        if self.cache and not skip_cache:
            cached = await self.cache.get(content_hash, self.provider.model_name)
            if cached:
                return EmbeddingResult(
                    embedding=cached,
                    model=self.provider.model_name,
                    dimensions=len(cached),
                    content_hash=content_hash,
                    cached=True,
                    latency_ms=0
                )
        
        # Generate embedding
        embedding = await self.provider.embed_text(text)
        latency_ms = (time.time() - start_time) * 1000
        
        # Validate dimensions
        if len(embedding) != self.provider.dimensions:
            raise ValueError(
                f"Dimension mismatch: expected {self.provider.dimensions}, "
                f"got {len(embedding)}"
            )
        
        # Cache result
        if self.cache:
            await self.cache.set(content_hash, self.provider.model_name, embedding)
        
        return EmbeddingResult(
            embedding=embedding,
            model=self.provider.model_name,
            dimensions=len(embedding),
            content_hash=content_hash,
            cached=False,
            latency_ms=latency_ms
        )
    
    async def embed_batch(
        self,
        texts: List[str],
        skip_cache: bool = False
    ) -> List[EmbeddingResult]:
        """
        Generate embeddings for batch of texts.
        
        Args:
            texts: Texts to embed
            skip_cache: Skip cache lookup
            
        Returns:
            List of EmbeddingResult
        """
        import time
        
        if not texts:
            return []
        
        results = [None] * len(texts)
        texts_to_embed = []
        indices_to_embed = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            content_hash = self._compute_content_hash(text)
            
            if self.cache and not skip_cache:
                cached = await self.cache.get(content_hash, self.provider.model_name)
                if cached:
                    results[i] = EmbeddingResult(
                        embedding=cached,
                        model=self.provider.model_name,
                        dimensions=len(cached),
                        content_hash=content_hash,
                        cached=True,
                        latency_ms=0
                    )
                    continue
            
            texts_to_embed.append(text)
            indices_to_embed.append(i)
        
        # Embed uncached texts
        if texts_to_embed:
            start_time = time.time()
            embeddings = await self.provider.embed_batch(texts_to_embed)
            latency_ms = (time.time() - start_time) * 1000 / len(texts_to_embed)
            
            for j, (text, embedding) in enumerate(zip(texts_to_embed, embeddings)):
                content_hash = self._compute_content_hash(text)
                idx = indices_to_embed[j]
                
                # Validate dimensions
                if len(embedding) != self.provider.dimensions:
                    raise ValueError(
                        f"Dimension mismatch: expected {self.provider.dimensions}, "
                        f"got {len(embedding)}"
                    )
                
                # Cache result
                if self.cache:
                    await self.cache.set(content_hash, self.provider.model_name, embedding)
                
                results[idx] = EmbeddingResult(
                    embedding=embedding,
                    model=self.provider.model_name,
                    dimensions=len(embedding),
                    content_hash=content_hash,
                    cached=False,
                    latency_ms=latency_ms
                )
        
        return results
    
    async def embed_chunks(
        self,
        chunks: List[Chunk]
    ) -> List[Chunk]:
        """
        Generate embeddings for chunks.
        
        Args:
            chunks: List of chunks to embed
            
        Returns:
            Chunks with embeddings populated
        """
        # Extract texts
        texts = [c.text for c in chunks]
        
        # Generate embeddings
        results = await self.embed_batch(texts)
        
        # Populate chunks
        for chunk, result in zip(chunks, results):
            chunk.embedding = result.embedding
            chunk.embedding_model = result.model
        
        return chunks
    
    @property
    def model_name(self) -> str:
        """Get current model name."""
        return self.provider.model_name
    
    @property
    def dimensions(self) -> int:
        """Get embedding dimensions."""
        return self.provider.dimensions


# ============================================================================
# Embedding Pipeline
# ============================================================================

@register_pipeline("embedding")
class EmbeddingPipeline(BasePipeline):
    """
    Pipeline stage for generating embeddings.
    
    Generates embeddings for:
    - Chunk text content
    - Image captions (combined with text)
    """
    
    stage_name = "embedding"
    
    def __init__(
        self,
        provider: Optional[EmbeddingProvider] = None,
        enable_cache: bool = True
    ):
        """
        Initialize embedding pipeline.
        
        Args:
            provider: Custom embedding provider
            enable_cache: Enable embedding cache
        """
        super().__init__()
        self.service = EmbeddingService(
            provider=provider,
            enable_cache=enable_cache
        )
    
    async def process(self, document: IngestionDocument) -> IngestionDocument:
        """
        Process document to generate embeddings.
        
        Args:
            document: IngestionDocument with chunks
            
        Returns:
            Document with embeddings in chunks
        """
        if not document.chunks:
            document.add_error(
                self.stage_name,
                "No chunks to embed"
            )
            return document
        
        # Embed chunks
        document.chunks = await self.service.embed_chunks(document.chunks)
        
        # Statistics
        cached_count = sum(
            1 for c in document.chunks 
            if c.embedding and c.metadata.get("cached", False)
        )
        
        document.metadata["embedding_stats"] = {
            "total_chunks_embedded": len(document.chunks),
            "model": self.service.model_name,
            "dimensions": self.service.dimensions,
            "cached_embeddings": cached_count,
        }
        
        return document
    
    def can_skip(self, document: IngestionDocument) -> bool:
        """Check if embedding can be skipped."""
        # Skip if all chunks already have embeddings
        if document.chunks and all(c.embedding for c in document.chunks):
            return True
        
        return document.is_stage_completed(self.stage_name)


# ============================================================================
# Factory Functions
# ============================================================================

def get_embedding_service(
    model: Optional[str] = None,
    enable_cache: bool = True
) -> EmbeddingService:
    """
    Get an embedding service instance.
    
    Args:
        model: Model name override
        enable_cache: Enable caching
        
    Returns:
        EmbeddingService instance
    """
    provider = None
    if model:
        provider = OpenAIEmbeddingProvider(model=model)
    
    return EmbeddingService(provider=provider, enable_cache=enable_cache)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "EmbeddingResult",
    "EmbeddingCache",
    "EmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "MockEmbeddingProvider",
    "EmbeddingService",
    "EmbeddingPipeline",
    "get_embedding_service",
]

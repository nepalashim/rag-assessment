from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, 
    VectorParams, 
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)
from app.config import settings
import uuid


#Vector store service using Qdrant for similarity search.


class VectorStoreService:
    """Service for managing vector storage and retrieval using Qdrant."""
    
    def __init__(
        self,
        collection_name: Optional[str] = None,
        embedding_dimension: int = None
    ):
        """
        Initialize vector store service.
        
        Args:
            collection_name: Name of the Qdrant collection
            embedding_dimension: Dimension of embedding vectors
        """
        self.collection_name = collection_name or settings.vector_collection_name
        self.embedding_dimension = embedding_dimension or settings.embedding_dimension
        
        # Initialize Qdrant client
        # For local development, use HTTP (not HTTPS) and disable gRPC
        self.client = QdrantClient(
            url=f"http://{settings.qdrant_host}:{settings.qdrant_port}",
            api_key=settings.qdrant_api_key,
            prefer_grpc=False
        )
        
        # Create collection if it doesn't exist
        self._ensure_collection_exists()
    
    def _ensure_collection_exists(self) -> None:
        """Create collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        collection_names = [col.name for col in collections]
        
        if self.collection_name not in collection_names:
            print(f"Creating Qdrant collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dimension,
                    distance=Distance.COSINE
                )
            )
            print("Collection created successfully!")
        else:
            print(f"Collection '{self.collection_name}' already exists")
    
    def add_vectors(
        self,
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add vectors to the collection.
        
        Args:
            embeddings: List of embedding vectors
            metadatas: List of metadata dictionaries for each vector
            ids: Optional list of IDs (generated if not provided)
            
        Returns:
            List of vector IDs
        """
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(embeddings))]
        
        # Create points
        points = [
            PointStruct(
                id=point_id,
                vector=embedding,
                payload=metadata
            )
            for point_id, embedding, metadata in zip(ids, embeddings, metadatas)
        ]
        
        # Upload points
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        print(f"Added {len(points)} vectors to collection")
        return ids
    
    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            filter_conditions: Optional metadata filters
            
        Returns:
            List of search results with scores and metadata
        """
        # Build filter if provided
        query_filter = None
        if filter_conditions:
            query_filter = self._build_filter(filter_conditions)
        
        # Search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter
        )
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "id": result.id,
                "score": result.score,
                "metadata": result.payload
            })
        
        return formatted_results
    
    def _build_filter(self, conditions: Dict[str, Any]) -> Filter:
        """
        Build Qdrant filter from conditions dictionary.
        
        Args:
            conditions: Dictionary of field:value conditions
            
        Returns:
            Qdrant Filter object
        """
        field_conditions = []
        for field, value in conditions.items():
            field_conditions.append(
                FieldCondition(
                    key=field,
                    match=MatchValue(value=value)
                )
            )
        
        return Filter(must=field_conditions)
    
    def delete_vectors(
        self,
        ids: Optional[List[str]] = None,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Delete vectors from collection.
        
        Args:
            ids: List of vector IDs to delete
            filter_conditions: Delete vectors matching filter
            
        Returns:
            Success status
        """
        if ids:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=ids
            )
            print(f"Deleted {len(ids)} vectors")
            return True
        elif filter_conditions:
            query_filter = self._build_filter(filter_conditions)
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=query_filter
            )
            print("Deleted vectors matching filter")
            return True
        else:
            print("No deletion criteria provided")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.
        
        Returns:
            Collection information dictionary
        """
        info = self.client.get_collection(collection_name=self.collection_name)
        return {
            "name": self.collection_name,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status
        }


# Global vector store instance
_vector_store: Optional[VectorStoreService] = None


def get_vector_store(embedding_dimension: Optional[int] = None) -> VectorStoreService:
    """
    Get or create global vector store instance.
    
    Args:
        embedding_dimension: Dimension of embeddings (required on first call)
        
    Returns:
        VectorStoreService instance
    """
    global _vector_store
    
    if _vector_store is None:
        if embedding_dimension is None:
            embedding_dimension = settings.embedding_dimension
        _vector_store = VectorStoreService(embedding_dimension=embedding_dimension)
    
    return _vector_store

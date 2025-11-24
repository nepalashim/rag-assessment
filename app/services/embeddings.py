import ssl
ssl._create_default_https_context = ssl._create_unverified_context


from typing import List, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
from app.config import settings




class EmbeddingService:
    """Service for generating text embeddings."""
    
    def __init__(
        self, 
        model_name: Optional[str] = None,
        use_openai: bool = False
    ):
        """
        Initialize embedding service.
        
        Args:
            model_name: Model name to use for embeddings
            use_openai: Whether to use OpenAI embeddings
        """
        self.model_name = model_name or settings.embedding_model
        self.use_openai = use_openai
        
        if self.use_openai:
            self._init_openai()
        else:
            self._init_sentence_transformer()
    
    def _init_sentence_transformer(self) -> None:
        """Initialize sentence-transformers model."""
        print(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        print("Embedding model loaded successfully!")
    
    def _init_openai(self) -> None:
        """Initialize OpenAI client."""
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key not found in settings")
        
        try:
            from openai import OpenAI
            self.openai_client = OpenAI(api_key=settings.openai_api_key)
            print("OpenAI client initialized successfully!")
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as list of floats
        """
        if self.use_openai:
            return self._generate_openai_embedding(text)
        else:
            return self._generate_st_embedding(text)
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (batch processing).
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
        """
        if self.use_openai:
            return [self._generate_openai_embedding(text) for text in texts]
        else:
            return self._generate_st_embeddings(texts)
    
    def _generate_st_embedding(self, text: str) -> List[float]:
        """
        Generate embedding using sentence-transformers.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        embedding = self.model.encode(
            text, 
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embedding.tolist()
    
    def _generate_st_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts using sentence-transformers.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=32
        )
        return embeddings.tolist()
    
    def _generate_openai_embedding(self, text: str) -> List[float]:
        """
        Generate embedding using OpenAI API.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        response = self.openai_client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.
        
        Returns:
            Embedding dimension
        """
        if self.use_openai:
            # OpenAI text-embedding-3-small has 1536 dimensions
            return 1536
        else:
            # Get dimension from model
            test_embedding = self.generate_embedding("test")
            return len(test_embedding)


# Global embedding service instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """
    Get or create global embedding service instance.
    
    Returns:
        EmbeddingService instance
    """
    global _embedding_service
    
    if _embedding_service is None:
        # Use sentence-transformers by default (free, no API key needed)
        _embedding_service = EmbeddingService(use_openai=False)
    
    return _embedding_service
from typing import List, Dict, Any
import re
from app.config import settings


class ChunkingStrategy:
    """Base class for chunking strategies."""
    
    def chunk(self, text: str) -> List[Dict[str, Any]]:
        """
        Chunk text into smaller pieces.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of chunks with metadata
        """
        raise NotImplementedError


class FixedSizeChunking(ChunkingStrategy):
    """
    Fixed-size chunking with overlap.
    Splits text by character count with configurable overlap.
    """
    
    def __init__(
        self, 
        chunk_size: int = None, 
        chunk_overlap: int = None
    ):
        """
        Initialize fixed-size chunking.
        
        Args:
            chunk_size: Size of each chunk in characters
            chunk_overlap: Number of overlapping characters between chunks
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
    
    def chunk(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text into fixed-size chunks with overlap.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        chunks = []
        start = 0
        chunk_index = 0
        
        # Clean text
        text = self._clean_text(text)
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            # Don't create empty chunks
            if chunk_text.strip():
                chunks.append({
                    "chunk_index": chunk_index,
                    "chunk_text": chunk_text.strip(),
                    "chunk_size": len(chunk_text),
                    "start_position": start,
                    "end_position": end
                })
                chunk_index += 1
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            
            # Prevent infinite loop
            if start <= 0 and end >= len(text):
                break
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        # Remove multiple newlines
        text = re.sub(r'\n+', '\n', text)
        return text.strip()


class SemanticChunking(ChunkingStrategy):
    """
    Semantic chunking based on sentence boundaries.
    Groups sentences together until reaching target size.
    """
    
    def __init__(
        self, 
        chunk_size: int = None,
        min_chunk_size: int = 100
    ):
        """
        Initialize semantic chunking.
        
        Args:
            chunk_size: Target size for each chunk
            min_chunk_size: Minimum chunk size to avoid tiny chunks
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.min_chunk_size = min_chunk_size
    
    def chunk(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text into semantic chunks based on sentences.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        # Clean text
        text = self._clean_text(text)
        
        # Split into sentences
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence exceeds chunk_size and we have content
            if current_size + sentence_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append({
                        "chunk_index": chunk_index,
                        "chunk_text": chunk_text,
                        "chunk_size": len(chunk_text),
                        "sentence_count": len(current_chunk)
                    })
                    chunk_index += 1
                
                # Start new chunk with current sentence
                current_chunk = [sentence]
                current_size = sentence_length
            else:
                # Add sentence to current chunk
                current_chunk.append(sentence)
                current_size += sentence_length
        
        # Add remaining chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append({
                    "chunk_index": chunk_index,
                    "chunk_text": chunk_text,
                    "chunk_size": len(chunk_text),
                    "sentence_count": len(current_chunk)
                })
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        # Remove multiple newlines
        text = re.sub(r'\n+', ' ', text)
        return text.strip()
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using regex.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Split on sentence boundaries (., !, ?)
        # Keep the punctuation with the sentence
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filter out empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences


def get_chunking_strategy(strategy: str) -> ChunkingStrategy:
    """
    Factory function to get chunking strategy.
    
    Args:
        strategy: Strategy name ('fixed' or 'semantic')
        
    Returns:
        ChunkingStrategy instance
        
    Raises:
        ValueError: If strategy is not supported
    """
    strategies = {
        "fixed": FixedSizeChunking,
        "semantic": SemanticChunking
    }
    
    if strategy not in strategies:
        raise ValueError(
            f"Unknown chunking strategy: {strategy}. "
            f"Available strategies: {list(strategies.keys())}"
        )
    
    return strategies[strategy]()
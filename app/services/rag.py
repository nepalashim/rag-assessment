import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from typing import List, Dict, Any, Optional, Tuple
import re
from datetime import datetime
from openai import OpenAI
from groq import Groq
from app.config import settings
from app.services.embeddings import get_embedding_service
from app.services.vector_store import get_vector_store
from app.services.redis_service import get_redis_service


class RAGService:
   
    
    def __init__(self):
        """Initialize RAG service with required components."""
        self.embedding_service = get_embedding_service()
        self.vector_store = get_vector_store()
        self.redis_service = get_redis_service()
        
        # Select LLM provider
        if settings.llm_provider == "groq":
            if not settings.groq_api_key:
                raise ValueError("Groq API key is required for Groq LLM provider")

            self.llm_client = Groq(api_key=settings.groq_api_key)
        # # Initialize LLM client
        # if settings.openai_api_key:
        #     self.llm_client = OpenAI(api_key=settings.openai_api_key)
        # else:
        #     raise ValueError("OpenAI API key is required for RAG service")
    
    def query(
        self,
        user_query: str,
        session_id: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Process a user query using custom RAG logic.
        
        Args:
            user_query: User's question
            session_id: Session identifier for conversation context
            top_k: Number of relevant chunks to retrieve
            
        Returns:
            Dictionary with answer and sources
        """
        # Check if this is a booking request
        if self._is_booking_intent(user_query):
            return {
                "type": "booking_intent",
                "message": "I detected you want to book an interview. Please provide: name, email, date (YYYY-MM-DD), and time (HH:MM)",
                "sources": []
            }
        
        # Step 1: Generate query embedding
        query_embedding = self.embedding_service.generate_embedding(user_query)
        
        # Step 2: Retrieve relevant context from vector database
        search_results = self.vector_store.search(
            query_vector=query_embedding,
            top_k=top_k
        )
        
        # Step 3: Get conversation history from Redis
        conversation_history = self.redis_service.get_conversation_history(
            session_id=session_id,
            limit=10  # Last 10 messages
        )
        
        # Step 4: Generate answer using LLM
        answer = self._generate_answer(
            query=user_query,
            context_results=search_results,
            conversation_history=conversation_history
        )
        
        # Step 5: Save to Redis
        self.redis_service.save_message(
            session_id=session_id,
            role="user",
            content=user_query
        )
        self.redis_service.save_message(
            session_id=session_id,
            role="assistant",
            content=answer
        )
        
        # Format sources
        sources = self._format_sources(search_results)
        
        return {
            "type": "answer",
            "answer": answer,
            "sources": sources
        }
    
    def _retrieve_context(
        self,
        query_embedding: List[float],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Custom retrieval logic - search vector database.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results
            
        Returns:
            List of relevant chunks with metadata
        """
        results = self.vector_store.search(
            query_vector=query_embedding,
            top_k=top_k
        )
        return results
    
    def _generate_answer(
        self,
        query: str,
        context_results: List[Dict[str, Any]],
        conversation_history: List[Dict[str, Any]]
    ) -> str:
        """
        Generate answer using LLM with custom prompt construction.
        
        Args:
            query: User query
            context_results: Retrieved context from vector DB
            conversation_history: Previous messages
            
        Returns:
            Generated answer
        """
        # Build context string from retrieved chunks
        context_parts = []
        for idx, result in enumerate(context_results, 1):
            metadata = result.get("metadata", {})
            chunk_text = metadata.get("chunk_text", "")
            source = metadata.get("filename", "Unknown")
            
            context_parts.append(
                f"[Source {idx}: {source}]\n{chunk_text}\n"
            )
        
        context_text = "\n".join(context_parts)
        
        # Build conversation history string
        history_parts = []
        for msg in conversation_history[-6:]:  # Last 6 messages (3 turns)
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role and content:
                history_parts.append(f"{role.capitalize()}: {content}")
        
        history_text = "\n".join(history_parts)
        
        # Construct prompt
        system_prompt = """You are a helpful AI assistant. Answer questions based on the provided context.
If the context doesn't contain relevant information, say so politely.
Be concise and accurate. Use the conversation history for context."""
        
        user_prompt = f"""Previous conversation:
{history_text if history_text else "No previous conversation"}

Context from documents:
{context_text if context_text else "No relevant context found"}

Question: {query}

Please provide a clear and concise answer based on the context above."""
        
        # Call LLM
        try:
            response = self.llm_client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content.strip()
            return answer
            
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def _format_sources(
        self,
        search_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Format search results into source objects.
        
        Args:
            search_results: Raw search results from vector DB
            
        Returns:
            Formatted source list
        """
        sources = []
        for result in search_results:
            metadata = result.get("metadata", {})
            sources.append({
                "document_id": metadata.get("document_id", ""),
                "filename": metadata.get("filename", "Unknown"),
                "chunk_text": metadata.get("chunk_text", "")[:200] + "...",  # Truncate
                "relevance_score": round(result.get("score", 0.0), 4)
            })
        return sources
    
    def _is_booking_intent(self, query: str) -> bool:
        """
        Detect if query is about booking an interview.
        
        Args:
            query: User query
            
        Returns:
            True if booking intent detected
        """
        booking_keywords = [
            "book", "schedule", "interview", "appointment",
            "meeting", "set up", "arrange", "reserve"
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in booking_keywords)
    
    def extract_booking_info(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Extract booking information from query using LLM.
        
        Args:
            query: User query potentially containing booking info
            
        Returns:
            Dictionary with booking details or None
        """
        prompt = f"""Extract booking information from this text. 
Return ONLY a JSON object with these fields: name, email, date, time.
If a field is not found, set it to null.
Date format: YYYY-MM-DD
Time format: HH:MM

Text: {query}

JSON:"""
        
        try:
            response = self.llm_client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": "You extract structured data and return only JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            result_text = re.sub(r'```json\s*|\s*```', '', result_text)
            
            # Parse JSON
            import json
            booking_data = json.loads(result_text)
            
            # Validate at least one field is present
            if any(booking_data.values()):
                return booking_data
            
            return None
            
        except Exception as e:
            print(f"Error extracting booking info: {str(e)}")
            return None


# Global RAG service instance
_rag_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """
    Get or create global RAG service instance.
    
    Returns:
        RAGService instance
    """
    global _rag_service
    
    if _rag_service is None:
        _rag_service = RAGService()
    
    return _rag_service
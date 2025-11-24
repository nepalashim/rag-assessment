"""
Redis service for managing chat memory and conversation history.
"""
from typing import List, Dict, Any, Optional
import json
import redis
from app.config import settings


class RedisService:
    """Service for managing conversation memory in Redis."""
    
    def __init__(self):
        """Initialize Redis connection."""
        self.client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            password=settings.redis_password,
            db=settings.redis_db,
            decode_responses=True
        )
        
        # Test connection
        try:
            self.client.ping()
            print("Redis connection established successfully!")
        except redis.ConnectionError as e:
            raise ConnectionError(f"Failed to connect to Redis: {str(e)}")
    
    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Save a message to conversation history.
        
        Args:
            session_id: Session identifier
            role: Message role (user, assistant, system)
            content: Message content
            metadata: Optional metadata
            
        Returns:
            Success status
        """
        message = {
            "role": role,
            "content": content,
            "metadata": metadata or {}
        }
        
        key = self._get_session_key(session_id)
        
        # Add message to list
        self.client.rpush(key, json.dumps(message))
        
        # Set expiration (24 hours)
        self.client.expire(key, 86400)
        
        return True
    
    def get_conversation_history(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history for a session.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of messages to retrieve (most recent)
            
        Returns:
            List of messages in chronological order
        """
        key = self._get_session_key(session_id)
        
        if limit:
            # Get last N messages
            messages = self.client.lrange(key, -limit, -1)
        else:
            # Get all messages
            messages = self.client.lrange(key, 0, -1)
        
        # Parse JSON messages
        parsed_messages = []
        for msg in messages:
            try:
                parsed_messages.append(json.loads(msg))
            except json.JSONDecodeError:
                continue
        
        return parsed_messages
    
    def clear_conversation(self, session_id: str) -> bool:
        """
        Clear conversation history for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Success status
        """
        key = self._get_session_key(session_id)
        self.client.delete(key)
        return True
    
    def get_conversation_length(self, session_id: str) -> int:
        """
        Get number of messages in conversation.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Number of messages
        """
        key = self._get_session_key(session_id)
        return self.client.llen(key)
    
    def session_exists(self, session_id: str) -> bool:
        """
        Check if session exists.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session exists
        """
        key = self._get_session_key(session_id)
        return self.client.exists(key) > 0
    
    def _get_session_key(self, session_id: str) -> str:
        """
        Generate Redis key for session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Redis key
        """
        return f"chat:session:{session_id}"
    
    def save_chat_context(
        self,
        session_id: str,
        context: Dict[str, Any],
        ttl: int = 3600
    ) -> bool:
        """
        Save additional context for a session.
        
        Args:
            session_id: Session identifier
            context: Context dictionary
            ttl: Time to live in seconds
            
        Returns:
            Success status
        """
        key = f"chat:context:{session_id}"
        self.client.setex(key, ttl, json.dumps(context))
        return True
    
    def get_chat_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get additional context for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Context dictionary or None
        """
        key = f"chat:context:{session_id}"
        data = self.client.get(key)
        
        if data:
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                return None
        
        return None


# Global Redis service instance
_redis_service: Optional[RedisService] = None


def get_redis_service() -> RedisService:
    """
    Get or create global Redis service instance.
    
    Returns:
        RedisService instance
    """
    global _redis_service
    
    if _redis_service is None:
        _redis_service = RedisService()
    
    return _redis_service
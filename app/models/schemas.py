"""
Pydantic models for request/response validation.
"""
from typing import List, Optional, Literal
from datetime import datetime
from pydantic import BaseModel, EmailStr, Field


#  Document Ingestion Schemas 

class DocumentIngestionRequest(BaseModel):
    """Request model for document ingestion."""
    chunking_strategy: Literal["fixed", "semantic"] = Field(
        default="fixed",
        description="Chunking strategy to use: 'fixed' or 'semantic'"
    )


class DocumentIngestionResponse(BaseModel):
    """Response model for document ingestion."""
    document_id: str
    filename: str
    chunks_count: int
    status: str
    message: str
    chunking_strategy: str
    
    model_config = {"from_attributes": True}


# Chat/RAG Schemas

class ChatRequest(BaseModel):
    """Request model for chat/RAG endpoint."""
    query: str = Field(..., min_length=1, description="User query")
    session_id: str = Field(..., description="Session ID for conversation context")
    user_id: Optional[str] = Field(None, description="Optional user identifier")


class Source(BaseModel):
    """Source document information."""
    document_id: str
    filename: str
    chunk_text: str
    relevance_score: float


class ChatResponse(BaseModel):
    """Response model for chat/RAG endpoint."""
    answer: str
    sources: List[Source]
    session_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Interview Booking Schemas 

class BookingRequest(BaseModel):
    """Request model for interview booking."""
    name: str = Field(..., min_length=1, description="Candidate name")
    email: EmailStr = Field(..., description="Candidate email")
    date: str = Field(..., description="Interview date (YYYY-MM-DD)")
    time: str = Field(..., description="Interview time (HH:MM)")
    session_id: str = Field(..., description="Session ID")


class BookingResponse(BaseModel):
    """Response model for interview booking."""
    booking_id: str
    name: str
    email: str
    date: str
    time: str
    status: str
    message: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    model_config = {"from_attributes": True}


# Generic Responses

class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
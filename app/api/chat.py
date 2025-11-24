from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import uuid
from datetime import datetime

from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    BookingRequest,
    BookingResponse,
    ErrorResponse
)
from app.db.database import get_db
from app.db.models import InterviewBooking, ChatHistory
from app.services.rag import get_rag_service


router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}
)
async def chat(
    request: ChatRequest,
    db: Session = Depends(get_db)
) -> ChatResponse:
    """
    Conversational RAG endpoint with multi-turn support.
    
    Features:
    - Custom RAG implementation (no RetrievalQAChain)
    - Redis-backed conversation memory
    - Multi-turn query handling
    - Context-aware responses
    
    Args:
        request: Chat request with query and session_id
        db: Database session
        
    Returns:
        ChatResponse with answer and sources
        
    Raises:
        HTTPException: If processing fails
    """
    try:
        # Get RAG service
        rag_service = get_rag_service()
        
        # Process query
        result = rag_service.query(
            user_query=request.query,
            session_id=request.session_id,
            top_k=5
        )
        
        # Check if it's a booking intent
        if result.get("type") == "booking_intent":
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "booking_intent_detected",
                    "message": result.get("message"),
                    "suggestion": "Please use the /book-interview endpoint with complete booking details"
                }
            )
        
        # Save to database (backup)
        chat_record = ChatHistory(
            session_id=request.session_id,
            user_id=request.user_id,
            query=request.query,
            response=result["answer"]
        )
        db.add(chat_record)
        db.commit()
        
        # Return response
        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            session_id=request.session_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat request: {str(e)}"
        )


@router.post(
    "/book-interview",
    response_model=BookingResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}
)
async def book_interview(
    request: BookingRequest,
    db: Session = Depends(get_db)
) -> BookingResponse:
    """
    Book an interview with candidate details.
    
    Args:
        request: Booking request with name, email, date, time
        db: Database session
        
    Returns:
        BookingResponse with confirmation
        
    Raises:
        HTTPException: If booking fails
    """
    try:
        # Validate date format
        try:
            datetime.strptime(request.date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid date format. Use YYYY-MM-DD"
            )
        
        # Validate time format
        try:
            datetime.strptime(request.time, "%H:%M")
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid time format. Use HH:MM (24-hour format)"
            )
        
        # Check for duplicate booking
        existing_booking = db.query(InterviewBooking).filter(
            InterviewBooking.email == request.email,
            InterviewBooking.date == request.date,
            InterviewBooking.time == request.time,
            InterviewBooking.status == "scheduled"
        ).first()
        
        if existing_booking:
            raise HTTPException(
                status_code=400,
                detail="An interview is already scheduled for this email at this time"
            )
        
        # Create booking
        booking_id = str(uuid.uuid4())
        booking = InterviewBooking(
            id=booking_id,
            name=request.name,
            email=request.email,
            date=request.date,
            time=request.time,
            session_id=request.session_id,
            status="scheduled"
        )
        
        db.add(booking)
        db.commit()
        db.refresh(booking)
        
        return BookingResponse(
            booking_id=booking_id,
            name=booking.name,
            email=booking.email,
            date=booking.date,
            time=booking.time,
            status=booking.status,
            message=f"Interview successfully scheduled for {request.name} on {request.date} at {request.time}",
            created_at=booking.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Error creating booking: {str(e)}"
        )


@router.get("/bookings", tags=["chat"])
async def list_bookings(
    skip: int = 0,
    limit: int = 10,
    status: str = None,
    db: Session = Depends(get_db)
) -> list:
    """
    List all interview bookings.
    
    Args:
        skip: Number of records to skip
        limit: Maximum number of records to return
        status: Filter by status (optional)
        db: Database session
        
    Returns:
        List of bookings
    """
    query = db.query(InterviewBooking)
    
    if status:
        query = query.filter(InterviewBooking.status == status)
    
    bookings = query.order_by(
        InterviewBooking.created_at.desc()
    ).offset(skip).limit(limit).all()
    
    return [
        {
            "booking_id": b.id,
            "name": b.name,
            "email": b.email,
            "date": b.date,
            "time": b.time,
            "status": b.status,
            "created_at": b.created_at.isoformat() if b.created_at else None
        }
        for b in bookings
    ]


@router.delete("/bookings/{booking_id}", tags=["chat"])
async def cancel_booking(
    booking_id: str,
    db: Session = Depends(get_db)
) -> dict:
    """
    Cancel an interview booking.
    
    Args:
        booking_id: Booking ID to cancel
        db: Database session
        
    Returns:
        Cancellation confirmation
    """
    booking = db.query(InterviewBooking).filter(
        InterviewBooking.id == booking_id
    ).first()
    
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")
    
    booking.status = "cancelled"
    db.commit()
    
    return {
        "status": "success",
        "message": f"Booking {booking_id} cancelled successfully"
    }


@router.get("/chat-history/{session_id}", tags=["chat"])
async def get_chat_history(
    session_id: str,
    limit: int = 20,
    db: Session = Depends(get_db)
) -> list:
    """
    Get chat history for a session.
    
    Args:
        session_id: Session ID
        limit: Maximum number of messages
        db: Database session
        
    Returns:
        List of chat messages
    """
    messages = db.query(ChatHistory).filter(
        ChatHistory.session_id == session_id
    ).order_by(ChatHistory.timestamp.desc()).limit(limit).all()
    
    return [
        {
            "query": msg.query,
            "response": msg.response,
            "timestamp": msg.timestamp.isoformat() if msg.timestamp else None
        }
        for msg in reversed(messages)  # Return in chronological order
    ]
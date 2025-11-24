from typing import List
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.orm import Session
import uuid

from app.models.schemas import DocumentIngestionResponse, ErrorResponse
from app.db.database import get_db
from app.db.models import Document, DocumentChunk
from app.services.document_processor import DocumentProcessor
from app.services.chunking import get_chunking_strategy
from app.services.embeddings import get_embedding_service
from app.services.vector_store import get_vector_store


router = APIRouter(prefix="/api/v1", tags=["ingestion"])


@router.post(
    "/ingest",
    response_model=DocumentIngestionResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}
)
async def ingest_document(
    file: UploadFile = File(..., description="PDF or TXT file to ingest"),
    chunking_strategy: str = Form(
        default="fixed",
        description="Chunking strategy: 'fixed' or 'semantic'"
    ),
    db: Session = Depends(get_db)
) -> DocumentIngestionResponse:
    """
    Ingest a document: extract text, chunk, generate embeddings, and store.
    
    Process:
    1. Validate and extract text from uploaded file
    2. Apply selected chunking strategy
    3. Generate embeddings for each chunk
    4. Store embeddings in vector database (Qdrant)
    5. Save metadata in SQL database
    
    Args:
        file: Uploaded PDF or TXT file
        chunking_strategy: Strategy to use ('fixed' or 'semantic')
        db: Database session
        
    Returns:
        DocumentIngestionResponse with document ID and metadata
        
    Raises:
        HTTPException: If validation or processing fails
    """
    try:
        # Step 1: Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        file_content = await file.read()
        file_size = len(file_content)
        
        try:
            DocumentProcessor.validate_file(file.filename, file_size)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"File validation failed: {str(e)}")
        
        # Step 2: Extract text
        try:
            await file.seek(0)  # Reset file pointer
            text, file_type = DocumentProcessor.extract_text(file.file, file.filename)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Text extraction failed: {str(e)}")
        
        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail="No text could be extracted from the file"
            )
        
        # Step 3: Apply chunking strategy
        try:
            chunker = get_chunking_strategy(chunking_strategy)
            chunks = chunker.chunk(text)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Chunking failed: {str(e)}")
        
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="Chunking produced no results"
            )
        
        # Step 4: Generate embeddings
        try:
            embedding_service = get_embedding_service()
            chunk_texts = [chunk["chunk_text"] for chunk in chunks]
            embeddings = embedding_service.generate_embeddings(chunk_texts)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")
        
        # Step 5: Generate document ID
        document_id = str(uuid.uuid4())
        
        # Step 6: Prepare metadata for vector storage
        metadatas = []
        for chunk in chunks:
            metadatas.append({
                "document_id": document_id,
                "filename": file.filename,
                "chunk_index": chunk["chunk_index"],
                "chunk_text": chunk["chunk_text"],
                "chunk_size": chunk["chunk_size"]
            })
        
        # Step 7: Store embeddings in vector database
        try:
            vector_store = get_vector_store()
            vector_ids = vector_store.add_vectors(
                embeddings=embeddings,
                metadatas=metadatas
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Vector storage failed: {str(e)}")
        
        # Step 8: Save document metadata to SQL
        document = Document(
            id=document_id,
            filename=file.filename,
            file_type=file_type,
            chunking_strategy=chunking_strategy,
            chunks_count=len(chunks),
            file_size=file_size
        )
        db.add(document)
        
        # Step 9: Save chunk metadata to SQL
        for idx, (chunk, vector_id) in enumerate(zip(chunks, vector_ids)):
            chunk_record = DocumentChunk(
                id=str(uuid.uuid4()),
                document_id=document_id,
                chunk_index=idx,
                chunk_text=chunk["chunk_text"],
                chunk_size=chunk["chunk_size"],
                embedding_id=vector_id
            )
            db.add(chunk_record)
        
        db.commit()
        
        # Step 10: Return response
        return DocumentIngestionResponse(
            document_id=document_id,
            filename=file.filename,
            chunks_count=len(chunks),
            status="success",
            message=f"Document ingested successfully with {len(chunks)} chunks",
            chunking_strategy=chunking_strategy
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )


@router.get("/documents", tags=["ingestion"])
async def list_documents(
    skip: int = 0,
    limit: int = 10,
    db: Session = Depends(get_db)
) -> List[dict]:
    """
    List all ingested documents.
    
    Args:
        skip: Number of records to skip
        limit: Maximum number of records to return
        db: Database session
        
    Returns:
        List of document metadata
    """
    documents = db.query(Document).offset(skip).limit(limit).all()
    
    return [
        {
            "id": doc.id,
            "filename": doc.filename,
            "file_type": doc.file_type,
            "chunking_strategy": doc.chunking_strategy,
            "chunks_count": doc.chunks_count,
            "upload_date": doc.upload_date.isoformat() if doc.upload_date else None
        }
        for doc in documents
    ]


@router.delete("/documents/{document_id}", tags=["ingestion"])
async def delete_document(
    document_id: str,
    db: Session = Depends(get_db)
) -> dict:
    """
    Delete a document and its chunks.
    
    Args:
        document_id: Document ID to delete
        db: Database session
        
    Returns:
        Deletion confirmation
    """
    # Find document
    document = db.query(Document).filter(Document.id == document_id).first()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Delete from vector database
    vector_store = get_vector_store()
    vector_store.delete_vectors(filter_conditions={"document_id": document_id})
    
    # Delete chunks from SQL
    db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).delete()
    
    # Delete document from SQL
    db.delete(document)
    db.commit()
    
    return {
        "status": "success",
        "message": f"Document {document_id} deleted successfully"
    }
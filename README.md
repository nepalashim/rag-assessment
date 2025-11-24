# RAG Assessment API

A production-ready **Retrieval-Augmented Generation (RAG)** system built with FastAPI. This system allows you to upload documents (PDF/TXT), ask questions about them using AI, and book interviews - all powered by modern AI technologies.

##  Features

- ✅ **Document Ingestion**: Upload PDF and TXT files
- ✅ **Two Chunking Strategies**: Fixed-size and semantic chunking
- ✅ **Vector Embeddings**: Using sentence-transformers (runs locally)
- ✅ **Vector Search**: Qdrant for similarity search
- ✅ **Conversational AI**: Groq LLM (free API)
- ✅ **Chat Memory**: Redis for conversation history
- ✅ **Multi-turn Conversations**: Contextual follow-up questions
- ✅ **Interview Booking**: Schedule interviews with email validation
- ✅ **RESTful API**: Complete API with Swagger documentation
- ✅ **No Pre-built Chains**: Custom RAG implementation from scratch

##  Tech Stack

| Component | Technology |
|-----------|-----------|
| **Framework** | FastAPI |
| **Vector Database** | Qdrant |
| **SQL Database** | PostgreSQL |
| **Cache/Memory** | Redis |
| **Embeddings** | sentence-transformers (local, free) |
| **LLM** | Groq (cloud, free API) |
| **ORM** | SQLAlchemy |

##  Prerequisites

- **Python**: 3.9 or higher
- **Docker & Docker Compose**: For running PostgreSQL, Redis, and Qdrant
- **Groq API Key**: Free account at [console.groq.com](https://console.groq.com)

##  Quick Start

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd rag-assessment
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt
```

### 4. Configure Environment Variables( If any problem comes with API keys I will expose my .env original file from gitignore and provide with the API keys)

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your Groq API key
nano .env  # or use any text editor
```

**Get your free Groq API key:**
1. Sign up at [console.groq.com](https://console.groq.com)
2. Go to [console.groq.com/keys](https://console.groq.com/keys)
3. Click "Create API Key"
4. Copy and paste it into your `.env` file

**Your `.env` should look like:**
```env
# LLM Configuration
LLM_PROVIDER=groq
LLM_MODEL=llama-3.1-8b-instant
GROQ_API_KEY=gsk_your_actual_groq_api_key_here

# Database (default values work with Docker Compose)
DATABASE_URL=postgresql://user:password@localhost:5432/rag_db
REDIS_HOST=localhost
REDIS_PORT=6379
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

### 5. Download Embedding Model (Optional but Recommended)

```bash
# Pre-download the embedding model (~80 MB)
# This avoids delay on first document upload
python download_model.py
```

### 6. Start Docker Services

```bash
# Start PostgreSQL, Redis, and Qdrant
docker-compose up -d

#if shows error remove hyphen(-) and try
docker compose up -d 


# Verify all services are running
docker-compose ps

# You should see 3 services with status "Up":
# - rag_postgres
# - rag_redis  
# - rag_qdrant
```

### 7. Wait for Services to be Ready

```bash
# Wait 10 seconds for PostgreSQL to fully initialize
sleep 10
```

### 8. Start the Application

```bash
# Start FastAPI server
python -m app.main

# Or with auto-reload for development:
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Expected output:**
```
INFO:     Started server process [xxxxx]
INFO:     Waiting for application startup.
Starting application...
Database tables created successfully!
Application started successfully!
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 9. Access the API

Open your browser and visit:

- **Swagger UI (Interactive API Docs)**: http://localhost:8000/docs
- **ReDoc (Alternative Docs)**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

---

##  Testing the API with Swagger UI

Access the interactive API documentation at **http://localhost:8000/docs** and test all endpoints visually.

### Test 1: Upload a Document

1. **Scroll to** the green `POST /api/v1/ingest` endpoint
2. **Click** on it to expand
3. **Click** "Try it out" button
4. **Click** "Choose File" and select a `.txt` or `.pdf` file
5. **Select** chunking strategy: `fixed` or `semantic`
6. **Click** "Execute"

**Expected Response (200):**
```json
{
  "document_id": "2c136257-7c4d-42eb-be5c-3d75316d91df",
  "filename": "your_document.txt",
  "chunks_count": 9,
  "status": "success",
  "message": "Document ingested successfully with 9 chunks",
  "chunking_strategy": "fixed"
}
```

**Save the `document_id` for reference!**

---

### Test 2: List All Documents

1. **Find** the blue `GET /api/v1/documents` endpoint
2. **Click** "Try it out"
3. **Click** "Execute"

**Expected Response:**
```json
[
  {
    "id": "2c136257-7c4d-42eb-be5c-3d75316d91df",
    "filename": "your_document.txt",
    "file_type": "txt",
    "chunking_strategy": "fixed",
    "chunks_count": 9,
    "upload_date": "2025-11-24T04:16:41"
  }
]
```

---

### Test 3: Ask Questions About Your Document (RAG)

1. **Find** the green `POST /api/v1/chat` endpoint
2. **Click** "Try it out"
3. **Delete** the example JSON and paste:

```json
{
  "query": "What is this document about? Give me a brief summary.",
  "session_id": "test-session-001"
}
```

4. **Click** "Execute"

**Expected Response:**
```json
{
  "answer": "Based on the document, it discusses...",
  "sources": [
    {
      "document_id": "2c136257-7c4d-42eb-be5c-3d75316d91df",
      "filename": "your_document.txt",
      "chunk_text": "...",
      "relevance_score": 0.85
    }
  ],
  "session_id": "test-session-001",
  "timestamp": "2025-11-24T04:20:00"
}
```

**Your RAG system just answered a question using your document!** 

---

### Test 4: Multi-Turn Conversation (Memory Test)

1. **Stay in** `POST /api/v1/chat` endpoint
2. **Ask a follow-up** with the **same session_id**:

```json
{
  "query": "Can you give me more details about that?",
  "session_id": "test-session-001"
}
```

3. **Click** "Execute"

**The system remembers your previous question and provides context-aware answers!** 

---

### Test 5: Try Different Questions

Ask specific questions about your document:

**Question 1:**
```json
{
  "query": "What are the key points mentioned?",
  "session_id": "test-session-002"
}
```

**Question 2:**
```json
{
  "query": "Summarize the main topics",
  "session_id": "test-session-003"
}
```

---

### Test 6: Book an Interview

1. **Find** the green `POST /api/v1/book-interview` endpoint
2. **Click** "Try it out"
3. **Paste this JSON:**

```json
{
  "name": "John Doe",
  "email": "john@example.com",
  "date": "2024-12-20",
  "time": "15:00",
  "session_id": "test-session-001"
}
```

4. **Click** "Execute"

**Expected Response:**
```json
{
  "booking_id": "abc-xyz-123",
  "name": "John Doe",
  "email": "john@example.com",
  "date": "2024-12-20",
  "time": "15:00",
  "status": "scheduled",
  "message": "Interview successfully scheduled for John Doe on 2024-12-20 at 15:00",
  "created_at": "2025-11-24T04:25:00"
}
```

---

### Test 7: View All Bookings

1. **Find** the blue `GET /api/v1/bookings` endpoint
2. **Click** "Try it out"
3. **Click** "Execute"

**You'll see all scheduled interviews!**

---

### Test 8: View Chat History

1. **Find** the blue `GET /api/v1/chat-history/{session_id}` endpoint
2. **Click** "Try it out"
3. **Enter** `test-session-001` in the `session_id` field
4. **Click** "Execute"

**You'll see your entire conversation history!** 

---

### Test 9: Try Semantic Chunking

1. **Upload the same document again** using `POST /api/v1/ingest`
2. **Select** `semantic` for chunking_strategy
3. **Compare** the chunk count with the fixed strategy

**Different chunking strategies produce different numbers of chunks!**

---

### Test 10: Delete a Document

1. **Find** the red `DELETE /api/v1/documents/{document_id}` endpoint
2. **Click** "Try it out"
3. **Paste** your document ID (e.g., `2c136257-7c4d-42eb-be5c-3d75316d91df`)
4. **Click** "Execute"

**Expected Response:**
```json
{
  "status": "success",
  "message": "Document deleted successfully"
}
```

---

### Test 11: Health Check

1. **Scroll to bottom** and find `GET /health`
2. **Click** "Try it out"
3. **Click** "Execute"

**Expected Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-11-24T04:30:00"
}
```

---

##  Complete Test Checklist

- [ ] ✅ Upload document (fixed chunking)
- [ ] ✅ List all documents
- [ ] ✅ Ask question about document
- [ ] ✅ Ask follow-up question (same session)
- [ ] ✅ Ask different specific questions
- [ ] ✅ Book an interview
- [ ] ✅ View all bookings
- [ ] ✅ View chat history
- [ ] ✅ Upload with semantic chunking
- [ ] ✅ Delete a document
- [ ] ✅ Health check

---

##  Project Structure

```
rag-assessment/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application entry point
│   ├── config.py            # Configuration management
│   ├── api/
│   │   ├── __init__.py
│   │   ├── ingestion.py     # Document upload endpoints
│   │   └── chat.py          # Chat and booking endpoints
│   ├── services/
│   │   ├── __init__.py
│   │   ├── chunking.py      # Chunking strategies
│   │   ├── embeddings.py    # Embedding generation
│   │   ├── vector_store.py  # Qdrant integration
│   │   ├── document_processor.py  # PDF/TXT extraction
│   │   ├── redis_service.py # Redis memory management
│   │   └── rag.py           # Custom RAG logic
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py       # Pydantic models
│   └── db/
│       ├── __init__.py
│       ├── database.py      # Database connection
│       └── models.py        # SQLAlchemy models
├── requirements.txt         # Python dependencies
├── docker-compose.yml       # Docker services configuration
├── download_model.py        # Script to pre-download embedding model
├── .env.example             # Environment variables template
├── .env                     # Your actual environment variables (not in git)
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

##  Development

### View Logs

```bash
# Application logs (in the terminal where you ran python -m app.main)

# Docker service logs
docker-compose logs postgres
docker-compose logs redis
docker-compose logs qdrant

# All logs with follow mode
docker-compose logs -f
```

### Restart Services

```bash
# Restart all Docker services
docker-compose restart

# Restart specific service
docker-compose restart postgres

# Stop all services
docker-compose down

# Stop and remove all data (WARNING: deletes all data!)
docker-compose down -v
```

### Restarting After Computer Restart

```bash
# Navigate to project
cd /path/to/rag-assessment

# Activate virtual environment
source venv/bin/activate

# Start Docker services
docker-compose up -d

# Wait for services to initialize
sleep 10

# Start application
python -m app.main
```

---

##  Key Features Explained

### Chunking Strategies

1. **Fixed-size Chunking**: Splits text by character count with overlap
   - Fast and predictable
   - Default chunk size: 500 characters
   - Overlap: 50 characters

2. **Semantic Chunking**: Splits text by sentence boundaries
   - Preserves semantic meaning
   - Groups sentences until target size
   - Better for question answering

### Custom RAG Implementation

This project implements RAG from scratch without using pre-built chains:

1. **Query Processing**: Convert user query to embedding
2. **Retrieval**: Search vector database for relevant chunks
3. **Context Building**: Combine retrieved chunks with conversation history
4. **Generation**: Use Groq LLM to generate contextual answer
5. **Memory**: Store conversation in Redis for multi-turn chat

### Multi-turn Conversations

Use the same `session_id` for related questions. The system maintains conversation context in Redis, allowing you to ask follow-up questions naturally.

---

##  Troubleshooting

### Services Not Starting

```bash
# Check Docker is running
docker --version
docker-compose --version

# Restart Docker services
docker-compose down
docker-compose up -d
docker-compose ps
```

### Database Connection Errors

```bash
# Check PostgreSQL is running
docker-compose ps postgres

# View logs
docker-compose logs postgres

# Restart PostgreSQL
docker-compose restart postgres
```

### Port Already in Use

```bash
# Check what's using port 8000
sudo lsof -i :8000

# Kill the process
sudo kill -9 <PID>

# Or use a different port
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

### Groq API Errors

1. Verify API key in `.env` file (no spaces, quotes, or extra characters)
2. Check you have credits at [console.groq.com](https://console.groq.com)
3. Groq free tier: 30 requests/minute, 14,400 requests/day

### Embedding Model Download Issues

```bash
# Manually download the model
python download_model.py

# Or download directly in Python:
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

### SSL Certificate Errors

If you encounter SSL errors during model download or API calls, the code includes SSL workarounds for development environments. For production, update your system's SSL certificates:

```bash
# Ubuntu/Debian
sudo apt-get update && sudo apt-get install ca-certificates

# Update Python certificates
pip install --upgrade certifi
```

---

##  Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://user:password@localhost:5432/rag_db` |
| `REDIS_HOST` | Redis hostname | `localhost` |
| `REDIS_PORT` | Redis port | `6379` |
| `QDRANT_HOST` | Qdrant hostname | `localhost` |
| `QDRANT_PORT` | Qdrant port | `6333` |
| `GROQ_API_KEY` | Groq API key | **Required** |
| `LLM_PROVIDER` | LLM provider | `groq` |
| `LLM_MODEL` | LLM model name | `llama-3.1-8b-instant` |
| `EMBEDDING_MODEL` | Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| `CHUNK_SIZE` | Default chunk size | `500` |
| `CHUNK_OVERLAP` | Chunk overlap | `50` |

---

##  Docker Services

The `docker-compose.yml` file sets up three services:

1. **PostgreSQL** (Port 5432): Stores document metadata and bookings
2. **Redis** (Port 6379): Caches conversation history
3. **Qdrant** (Port 6333): Vector database for similarity search

All services use persistent volumes to retain data across restarts.

---

##  Acknowledgments

- **Groq**: For free, fast LLM API
- **Qdrant**: For excellent vector database
- **Sentence Transformers**: For quality embeddings
- **FastAPI**: For awesome framework
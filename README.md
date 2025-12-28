# Internal Knowledge Base RAG Chatbot

A production-ready RAG (Retrieval-Augmented Generation) chatbot for internal knowledge base queries using OpenAI, FAISS, and Streamlit.

## 🌟 Features

- **Document Ingestion**: Support for PDF, DOCX, TXT, MD files
- **Vector Search**: Fast similarity search with FAISS
- **OpenAI Integration**: GPT-4.1 mini for answers, Ada-002 for embeddings
- **REST API**: FastAPI backend with async support
- **Modern UI**: Streamlit chat interface
- **Source Citations**: Automatic source tracking and display

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <your-repo-url>
cd internal-know-qa-rag-bot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=your_api_key_here
```

### 3. Ingest Documents

```bash
# Option 1: Download from website
python main.py ingest --url https://handbook.gitlab.com/handbook/engineering --max-pages 50

# Option 2: Place documents in data/raw/ manually
# Then skip to building the index
```

### 4. Build Vector Index

```bash
python main.py build-index --data-dir data/raw
```

### 5. Run Application

```bash
# Terminal 1: Start API server
python main.py api

# Terminal 2: Start Streamlit UI
python main.py ui
```

Visit http://localhost:8501 to use the chatbot!

## 📋 Architecture

```
📄 Documents → 📦 Chunks → 🔢 Embeddings → 💾 FAISS Index
                                                    ↓
❓ User Question → 🔢 Query Embedding → 🔍 Similarity Search
                                                    ↓
📋 Top-K Chunks → 📝 Prompt + Context → 🤖 LLM → 💬 Answer
```

## 🛠️ Technology Stack

- **LLM**: OpenAI GPT-4.1 mini
- **Embeddings**: OpenAI Ada-002
- **Vector Store**: FAISS
- **API**: FastAPI
- **UI**: Streamlit
- **Document Processing**: PyPDF2, python-docx, LangChain

## 📂 Project Structure

```
internal-know-qa-rag-bot/
├── src/
│   ├── config/           # Configuration management
│   ├── ingestion/        # Document loading
│   ├── embeddings/       # Embedding generation & indexing
│   ├── retrieval/        # Document retrieval & RAG pipeline
│   ├── generation/       # LLM integration
│   └── utils/            # Helper functions
├── api/                  # FastAPI endpoints
├── ui/                   # Streamlit interface
├── data/
│   ├── raw/             # Raw documents
│   └── processed/       # Processed chunks
├── vector_store/        # FAISS index
├── scripts/             # Utility scripts
└── config.yaml          # Application configuration
```

## 🔧 Configuration

Edit `config.yaml` to customize:

- Model settings (GPT-4, temperature, max tokens)
- Chunk size and overlap
- Retrieval parameters (top-k, threshold)
- API and UI ports

## 📖 Usage Examples

### API

```bash
# Health check
curl http://localhost:8000/api/health

# Ask a question
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is our code review process?",
    "top_k": 5
  }'
```

### Python

```python
from src.retrieval.rag_pipeline import RAGPipeline

# Initialize pipeline
pipeline = RAGPipeline()

# Ask question
result = pipeline.query("What is our code review process?")

print(result["answer"])
print(result["sources"])
```

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Run specific test
pytest tests/test_retrieval.py
```

## 📝 License

MIT License - see LICENSE file for details

## 👤 Author

Srikanth Gali (srikanthgali137@gmail.com)

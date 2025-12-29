# Internal Knowledge Base RAG Chatbot

A production-ready RAG (Retrieval-Augmented Generation) chatbot for internal knowledge base queries using OpenAI, FAISS, and Streamlit.

## 🌟 Features

- **Document Ingestion**: Support for PDF, DOCX, TXT, MD files
- **Vector Search**: Fast similarity search with FAISS
- **OpenAI Integration**: GPT-4.1 mini for answers, Ada-002 for embeddings
- **REST API**: FastAPI backend with async support
- **Modern UI**: Streamlit chat interface
- **Source Citations**: Automatic source tracking and display
- **High Performance**: 85.7% overall accuracy with strong retrieval and relevance scores

## 📊 Performance Metrics

The RAG system has been thoroughly evaluated with the following results:

| Metric | Score | Description |
|--------|-------|-------------|
| **Retrieval Score** | 91.9% | Accuracy in finding relevant documents |
| **Faithfulness** | 71.8% | Factual accuracy of generated answers |
| **Relevance** | 83.0% | Answer relevance to the question |
| **Completeness** | 100% | Coverage of question aspects |
| **Overall Score** | 85.7% | Weighted average performance |

### Evaluation Details

Tested on 5 diverse queries covering:
- ✅ Factual questions (GitLab's mission and purpose)
- ✅ Technical questions (training and knowledge sharing)
- ✅ Procedural questions (time off requests)
- ✅ Policy questions (customer acceptance)

**Key Strengths:**
- Perfect completeness (100%) - all questions fully answered
- Excellent retrieval (91.9%) - highly accurate document matching
- Strong relevance (83.0%) - answers well-aligned with questions

**Continuous Improvement:**
- Faithfulness score (71.8%) indicates room for reducing hallucinations
- Ongoing monitoring and refinement of prompt engineering

*Last evaluation: December 28, 2025*

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
├── tests/               # Test suite & evaluation
│   ├── test_questions.json  # Evaluation test cases
│   └── test_*.py        # Unit tests
├── evaluation_report.json   # Latest evaluation results
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

## 🧪 Testing & Evaluation

```bash
# Run unit tests
pytest tests/

# Run specific test
pytest tests/test_retrieval.py

# View latest evaluation results
cat evaluation_report.json
```

### Evaluation Framework

The system uses a comprehensive evaluation framework that measures:

1. **Retrieval Quality**: Relevance of retrieved documents
2. **Faithfulness**: Factual accuracy without hallucinations
3. **Relevance**: Answer alignment with the question
4. **Completeness**: Coverage of all question aspects

Test questions span multiple categories:
- Factual (company mission, purpose)
- Technical (training, knowledge sharing)
- Procedural (time off requests)
- Policy (customer acceptance)
- Edge cases (out-of-scope queries)

## 📝 License

MIT License - see LICENSE file for details

## 👤 Author

Srikanth Gali (srikanthgali137@gmail.com)

# Internal Knowledge Base RAG Chatbot

A production-ready RAG (Retrieval-Augmented Generation) chatbot for internal knowledge base queries using OpenAI, FAISS, and Streamlit.

## 🌟 Features

- **Document Ingestion**: Support for PDF, DOCX, TXT, MD files
- **Vector Search**: Fast similarity search with FAISS
- **OpenAI Integration**: GPT-4.1 mini for answers, Ada-002 for embeddings
- **REST API**: FastAPI backend with async support
- **Modern UI**: Streamlit chat interface
- **Source Citations**: Automatic source tracking and display
- **Conversational AI**: Handles greetings, introductions, and natural conversations
- **Edge Case Handling**: Smart detection of out-of-scope queries
- **High Performance**: 89.1% overall accuracy with excellent retrieval and relevance scores

## 📊 Performance Metrics

The RAG system has been thoroughly evaluated with the following results:

| Metric | Score | Description |
|--------|-------|-------------|
| **Retrieval Score** | 90.4% | Accuracy in finding relevant documents |
| **Faithfulness** | 88.9% | Factual accuracy of generated answers |
| **Relevance** | 89.0% | Answer relevance to the question |
| **Completeness** | 87.2% | Coverage of question aspects |
| **Overall Score** | 89.1% | Weighted average performance |

### Evaluation Details

**Test Coverage:**
- ✅ **14 total queries** evaluated across multiple categories
- ✅ **8 factual/procedural queries** - Core knowledge base questions
- ✅ **4 greeting/intro queries** - Conversational handling (100% success rate)
- ✅ **2 edge case queries** - Out-of-scope detection (100% success rate)

**Query Categories Tested:**
- **Conversational** (Greetings, introductions) - Perfect handling
- **Factual** (GitLab's mission, purpose, customer acceptance)
- **Technical** (Training, knowledge sharing, remote work)
- **Procedural** (Time off requests, contribution processes)
- **Edge Cases** (Weather, sports) - Correctly rejected

**Key Strengths:**
- 🎯 **Perfect edge case handling** (100%) - Correctly identifies out-of-scope questions
- 🎯 **Perfect greeting handling** (100%) - Natural conversational responses
- 🎯 **Excellent retrieval** (90.4%) - Highly accurate document matching
- 🎯 **Strong relevance** (89.0%) - Answers well-aligned with questions
- 🎯 **High faithfulness** (88.9%) - Factual accuracy with minimal hallucinations
- 🎯 **Good completeness** (87.2%) - Comprehensive answer coverage

**Quality Consistency:**
- Faithfulness standard deviation: 5.0% (consistent quality)
- Completeness standard deviation: 9.2% (reliable coverage)
- Relevance standard deviation: 8.0% (stable relevance)

**Continuous Improvement:**
- System demonstrates robust performance across diverse query types
- Edge case detection prevents hallucinations on out-of-scope questions
- Conversational abilities enhance user experience
- Ongoing monitoring and refinement of retrieval and generation quality

*Last evaluation: December 29, 2025*
*Evaluation framework: 14 test cases across 5 categories*

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
│   └── streamlit_app/
│       ├── app.py        # Main UI application
│       ├── components/   # UI components (chat, sidebar)
│       └── styles/       # Custom CSS styling
├── data/
│   ├── raw/             # Raw documents
│   └── processed/       # Processed chunks
├── vector_store/        # FAISS index
├── scripts/             # Utility scripts
│   └── demo_evaluation.py  # Evaluation script
├── tests/               # Test suite & evaluation
│   ├── test_questions.json  # Evaluation test cases
│   └── test_*.py        # Unit tests
├── evaluation_report.json   # Latest evaluation results
└── config.yaml          # Application configuration
└── assets/              # Static assets
    └── demo.gif         # Demo gif video
```

## 🔧 Configuration

Edit `config.yaml` to customize:

- Model settings (GPT-4, temperature, max tokens)
- Chunk size and overlap
- Retrieval parameters (top-k, threshold)
- API and UI ports

## 📖 Usage Examples

### Conversational Queries

```bash
# Greeting
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Hi!", "max_sources": 5}'

# Introduction request
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What can you do?", "max_sources": 5}'
```

### Knowledge Base Queries

```bash
# Ask a factual question
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is GitLab'\''s mission?", "max_sources": 5}'

# Ask a procedural question
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I request time off?", "max_sources": 5}'
```

### API Health Check

```bash
# Health check
curl http://localhost:8000/health
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

# Handle greetings
greeting_result = pipeline.query("Hello!")
print(greeting_result["answer"])  # Returns friendly greeting
print(greeting_result.get("is_greeting"))  # True
```
## 📺 Demo

Here's a quick demo of the GitLab KnowledgeBase QA chatbot in action:
> **App Demo:**
>
> [![Watch the demo video](assets/demo.gif)]

## 🧪 Testing & Evaluation

```bash
# Run comprehensive evaluation
python scripts/demo_evaluation.py

# Run unit tests
pytest tests/

# Run specific test
pytest tests/test_retrieval.py

# View latest evaluation results
cat evaluation_report.json
```

### Evaluation Framework

The system uses a comprehensive evaluation framework that measures:

1. **Retrieval Quality**: Relevance of retrieved documents (90.4%)
2. **Faithfulness**: Factual accuracy without hallucinations (88.9%)
3. **Relevance**: Answer alignment with the question (89.0%)
4. **Completeness**: Coverage of all question aspects (87.2%)
5. **Edge Case Handling**: Out-of-scope query detection (100%)
6. **Conversational Handling**: Greeting and introduction responses (100%)

**Test Categories:**
- ✅ **Conversational**: Greetings, introductions, help requests
- ✅ **Factual**: Company mission, purpose, values, policies
- ✅ **Technical**: Training, knowledge sharing, platform features
- ✅ **Procedural**: Time off requests, contribution processes
- ✅ **Policy**: Customer acceptance, remote work guidelines
- ✅ **Edge Cases**: Out-of-scope queries (weather, sports, etc.)

**Special Features:**
- **Greeting Detection**: Automatically recognizes conversational openings
- **Introduction Requests**: Explains capabilities and usage
- **Out-of-Scope Detection**: Politely declines irrelevant questions
- **Source Attribution**: Tracks and displays relevant sources (when applicable)

## 🎨 UI Features

- **Fixed Header**: GitLab-branded orange gradient header stays visible while scrolling
- **Collapsible Sidebar**: Settings and controls in a collapsible side panel
- **Chat Interface**: Streamlit's native chat UI with source expandables
- **Responsive Design**: Works on desktop and mobile devices
- **Custom Styling**: GitLab color scheme

## 📝 License

MIT License - see LICENSE file for details

## 👤 Author

Srikanth Gali (srikanthgali137@gmail.com)

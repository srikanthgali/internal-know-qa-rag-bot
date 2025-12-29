# RAG Chatbot Evaluation Guide

## Quick Quality Check (2 minutes)

### 1. Start the system

```bash
python main.py api  # Terminal 1
python main.py ui   # Terminal 2
```

### 2. Test these 3 questions

- ✓ **Factual**: "What is [specific topic from your docs]?"
- ✓ **Edge case**: "What is the weather today?" (should say no info)
- ✓ **Complex**: "[Multi-part question about your domain]"

### 3. Check for each

- [ ] Answer makes sense
- [ ] Sources are cited
- [ ] No hallucinations
- [ ] Appropriate "I don't know" when needed

## Full Evaluation (10 minutes)

### Run the automated evaluation

```bash
python scripts/demo_evaluation.py
```

### Expected output

- Overall score > 0.80
- Retrieval score > 0.90
- Faithfulness score > 0.70
- No crashes or errors

## Key Metrics Explained

### Retrieval Score (0-1)

- **> 0.8**: Excellent - finds highly relevant documents
- **0.6-0.8**: Good - finds relevant documents most of the time
- **< 0.6**: Needs improvement - missing relevant documents

### Faithfulness Score (0-1)

- **> 0.8**: Excellent - answers stick to retrieved context
- **0.6-0.8**: Good - mostly based on context, minor extrapolation
- **< 0.6**: Concerning - potential hallucinations

### Relevance Score (0-1)

- **> 0.7**: Excellent - directly answers the question
- **0.5-0.7**: Good - answers related but could be more focused
- **< 0.5**: Poor - answer doesn't address the question

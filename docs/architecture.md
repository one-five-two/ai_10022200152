# System Architecture

## Data Flow

```text
User Query
   ↓
Optional Conversation Memory
   ↓
Query Expansion
   ↓
Query Embedding
   ↓
FAISS / Cosine Similarity Retrieval
   ↓
Top-k Retrieved Chunks
   ↓
Context Selection and Budget Control
   ↓
Prompt Construction
   ↓
Claude LLM
   ↓
Grounded Final Answer + Retrieved Evidence
```

## Component Explanation

1. **Data Loader** cleans the Ghana Election CSV and Ghana 2025 Budget PDF.
2. **Chunker** splits source documents into searchable passages.
3. **Embedder** converts text chunks and user queries into vector embeddings.
4. **Vector Store** stores embeddings and performs top-k similarity retrieval.
5. **Query Expansion** improves recall by creating alternative query phrasings.
6. **Prompt Builder** injects retrieved context into an anti-hallucination prompt.
7. **RAG Pipeline** coordinates retrieval, context selection, prompt construction, LLM generation, and logging.
8. **Streamlit UI** lets users ask questions and inspect retrieved chunks and similarity scores.
9. **Evaluation Scripts** compare RAG against pure LLM responses and run adversarial tests.
10. **Memory RAG** adds follow-up question handling as the innovation component.

## Why This Design Fits the Domain

The Ghana election dataset contains structured factual rows, while the budget PDF contains long policy text. A RAG design is suitable because it allows the system to retrieve exact evidence before generating an answer. This reduces hallucination and makes the chatbot more trustworthy for political and public-finance questions.

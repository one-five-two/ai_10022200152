# Ghana RAG Chatbot — CS4241 Artificial Intelligence Project

**Name:** NWAOGU SOMTOCHUKWU SHARON
**Index Number:** 10022200152

---

## Project Overview

This project is a Retrieval-Augmented Generation (RAG) chatbot developed as part of my CS4241 Artificial Intelligence course. The system answers questions about Ghana’s elections and the 2025 national budget using real datasets.

The chatbot works by retrieving relevant information from:

* A Ghana Election Results dataset (CSV)
* The Ghana 2025 Budget Statement (PDF)

Instead of using frameworks like LangChain or LlamaIndex, I implemented the full RAG pipeline manually to demonstrate understanding of how modern AI systems retrieve and generate responses.

---

## How the System Works

The chatbot follows a simple pipeline:

1. The user enters a question
2. The system converts the question into embeddings
3. It retrieves the most relevant chunks from the dataset
4. These chunks are inserted into a prompt
5. A language model generates a response based only on the retrieved context

This approach ensures that answers are grounded in real data and reduces hallucination.

---

## Key Features

* Manual implementation of RAG (no LangChain or LlamaIndex)
* Data cleaning and chunking for both CSV and PDF
* Embedding generation using sentence-transformers
* Vector storage using FAISS
* Top-k similarity-based retrieval
* Query expansion to improve search results
* Prompt engineering to reduce hallucination
* Debug tools:

  * Retrieved context
  * Similarity scores
  * Final prompt
* Performance tracking (latency)
* Conversation memory (innovation feature)
* Streamlit web interface

---

## Exam Requirement Checklist

| Exam Part | Requirement                | Location               |
| --------- | -------------------------- | ---------------------- |
| Part A    | Data cleaning and chunking | `src/ingestion/`       |
| Part B    | Embeddings and retrieval   | `src/retrieval/`       |
| Part B    | Query expansion            | `query_expansion.py`   |
| Part C    | Prompt engineering         | `src/prompts/`         |
| Part D    | Full RAG pipeline          | `src/pipeline/`        |
| Part D    | Logging                    | `logs/`                |
| Part E    | Evaluation                 | README (below)         |
| Part F    | Architecture               | `docs/architecture.md` |
| Part G    | Innovation (memory)        | `src/innovation/`      |
| Final     | UI                         | `ui/app.py`            |

---

## Project Structure

```text
ghana_rag_chatbot_project/
├── src/
├── ui/
├── data/
├── vector_store/
├── logs/
├── experiments/
├── docs/
├── tests/
├── config.py
├── requirements.txt
└── run_eval.py
```

---

## Setup

### 1. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add datasets

Place the following files in the `data/` folder:

```text
ghana_elections.csv
ghana_budget_2025.pdf
```

### 4. Run ingestion

```bash
python -m src.pipeline.ingest --csv data/ghana_elections.csv --pdf data/ghana_budget_2025.pdf
```

### 5. Run the app

```bash
streamlit run ui/app.py
```

---

## Part E — Evaluation

### Evaluation Approach

I evaluated the system based on:

* Accuracy (correctness of answers)
* Hallucination (whether the system invents information)
* Consistency (stability across similar queries)

---

### Adversarial Queries

**Query 1 (Ambiguous):**
"What did they say about education?"

* Without memory, the system struggled
* With memory enabled, it correctly inferred context

**Conclusion:** Memory improves multi-turn understanding.

---

**Query 2 (Misleading):**
"What is the election code for NPP?"

* No relevant data was retrieved
* System correctly responded that information was not available

**Conclusion:** Prompt design successfully prevents hallucination.

---

### RAG vs Non-RAG

**Without RAG:**

* Generic answers
* Occasional incorrect assumptions

**With RAG:**

* Answers grounded in dataset
* References to sources
* More accurate and reliable

---

### Failure Case

**Query:**
"Summarize the education section of the budget"

**Issue:**

* Output was mostly numbers and hard to read

**Reason:**

* Retrieved chunk contained structured numerical data
* Model struggled with summarization

**Fix:**

* Modified prompt to enforce structured summaries

**Result:**

* Output became clearer and easier to understand

---

### Overall Performance

* Retrieval accuracy was high
* Similarity scores showed relevant chunk selection
* Latency was acceptable (LLM step was slowest)

---

### Key Takeaways

* Retrieval quality is critical in RAG systems
* Prompt design strongly affects output quality
* Even small models perform well with good retrieval
* Memory improves user experience

---

## Deployment

The application can be deployed using Streamlit Cloud:

1. Push project to GitHub
2. Use repository name: `ai_10022200152`
3. Deploy `ui/app.py`
4. No API key required (free local model used)

---

## Final Notes

This project demonstrates a complete, manual implementation of a RAG system, including retrieval, generation, evaluation, and deployment. It highlights how combining structured data with language models can produce more reliable and grounded AI systems.

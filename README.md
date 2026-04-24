# Ghana RAG Chatbot — CS4241 Introduction to Artificial Intelligence

This project implements a manual Retrieval-Augmented Generation (RAG) chatbot for Academic City using:

1. Ghana Election Result CSV
2. Ghana 2025 Budget Statement PDF

The project follows the exam constraint: **no LangChain, no LlamaIndex, and no pre-built RAG pipeline**. The core RAG pieces are implemented manually: data cleaning, chunking, embeddings, vector storage, top-k retrieval, similarity scoring, prompt construction, context management, logging, and evaluation.

> Replace this section before submission:
>
> **Name:** YOUR NAME  
> **Index Number:** YOUR INDEX NUMBER  
> **Repository Name:** ai_YOUR_INDEX_NUMBER

---

## Exam Requirement Checklist

| Exam Part | Requirement | Where Implemented |
|---|---|---|
| Part A | Data cleaning and chunking | `src/ingestion/data_loader.py`, `src/ingestion/chunker.py` |
| Part A | Chunking comparison | `experiments/chunking_comparison.py` |
| Part B | Embeddings | `src/retrieval/embedder.py` |
| Part B | Vector store and top-k retrieval | `src/retrieval/vector_store.py` |
| Part B | Query expansion | `src/retrieval/query_expansion.py` |
| Part B | Failure case and fix | `experiments/experiment_logs.txt` |
| Part C | Prompt template and hallucination control | `src/prompts/prompt_builder.py` |
| Part C | Prompt experiments | `experiments/prompt_ablation.py` |
| Part D | Full RAG pipeline | `src/pipeline/rag_pipeline.py` |
| Part D | Logging | `logs/pipeline_runs.jsonl` generated at runtime |
| Part E | Adversarial testing | `src/evaluation/evaluator.py`, `run_eval.py` |
| Part F | Architecture | `docs/architecture.md` |
| Part G | Innovation component | `src/innovation/memory_rag.py` |
| Final | Streamlit UI | `ui/app.py` |

---

## Project Structure

```text
ghana_rag_chatbot_project/
├── config.py
├── requirements.txt
├── run_eval.py
├── ui/
│   └── app.py
├── src/
│   ├── ingestion/
│   ├── retrieval/
│   ├── prompts/
│   ├── pipeline/
│   ├── evaluation/
│   └── innovation/
├── experiments/
├── tests/
├── docs/
├── data/
├── vector_store/
└── logs/
```

---

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. No API key needed

This free version uses a local HuggingFace model (`google/flan-t5-small`) instead of Claude/Anthropic. You do not need to buy credits or create an API key.

### 4. Add the datasets

Download the exam datasets and place them here:

```text
data/ghana_elections.csv
data/ghana_budget_2025.pdf
```

### 5. Build the vector index

```bash
python -m src.pipeline.ingest --csv data/ghana_elections.csv --pdf data/ghana_budget_2025.pdf --output vector_store
```

### 6. Run the application

```bash
streamlit run ui/app.py
```

### 7. Run tests

```bash
python -m pytest tests/ -v
```

### 8. Run evaluation

```bash
python run_eval.py --mode all
```

---

## Deployment

Recommended: Streamlit Cloud.

1. Push this project to GitHub.
2. Set repository name as `ai_YOUR_INDEX_NUMBER`.
3. No secrets are required for this free version.
4. Deploy `ui/app.py`.
5. Submit both GitHub and deployed URL.

---

## Important Submission Notes

- Add your name and index number in this README and in major source files before submission.
- Invite `godwin.danso@acity.edu.gh` or `GodwinDansoAcity` as GitHub collaborator.
- Record a 2-minute walkthrough explaining architecture, chunking, retrieval, prompt design, evaluation, and innovation.
- Update `experiments/experiment_logs.txt` with your own manual experiment observations.

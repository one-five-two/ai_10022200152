# Ghana RAG Chatbot — CS4241 AI Project

**Name:** NWAOGU SOMTOCHUKWU SHARON  
**Index Number:** 10022200152  

---

## Project Overview

This project is a Retrieval-Augmented Generation (RAG) chatbot built to answer questions about Ghana’s elections and the 2025 national budget.

The system uses two datasets:
- Ghana Election Results (CSV)
- Ghana 2025 Budget Statement (PDF)

Instead of relying on prebuilt frameworks like LangChain or LlamaIndex, I implemented the full RAG pipeline manually. This includes data cleaning, chunking, embedding, similarity search, and response generation.

The goal of this project is to demonstrate a clear understanding of how modern AI systems retrieve and generate answers using external knowledge sources.

---

## How the System Works

The chatbot follows a simple pipeline:

1. The user enters a question  
2. The system searches for the most relevant chunks from the dataset  
3. These chunks are passed into a prompt  
4. The language model generates a response using only the retrieved context  

This ensures that answers are grounded in real data rather than hallucinated.

---

## Key Features

- Manual implementation of RAG (no LangChain)
- Vector search using FAISS
- Query expansion for better retrieval
- Prompt engineering to reduce hallucination
- Debug tools (retrieved chunks, similarity scores, final prompt)
- Conversation memory (innovation component)
- Streamlit interface for interaction

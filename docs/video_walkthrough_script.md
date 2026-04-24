# 2-Minute Video Walkthrough Script

Hello, my name is [YOUR NAME], index number [YOUR INDEX NUMBER]. This is my CS4241 Introduction to Artificial Intelligence RAG chatbot project.

The goal of the system is to answer questions about Ghana elections and the 2025 Budget Statement using retrieval-augmented generation.

First, I clean the two datasets: the Ghana Election CSV and the Ghana Budget PDF. The CSV is converted into structured text rows, while the PDF is extracted page by page and cleaned to remove noise.

Next, I chunk the data. Election rows are kept precise because each row represents a factual record. The budget PDF uses sliding-window chunking with overlap so that policy explanations are not cut off mid-context.

For retrieval, I use sentence-transformers to create embeddings and FAISS to store vectors. I implemented cosine similarity manually through normalized embeddings and top-k retrieval. I also added query expansion so the system can retrieve better results when users ask questions in casual language.

The RAG pipeline follows this flow: user query, retrieval, context selection, prompt construction, LLM generation, and final answer. The app displays retrieved chunks, similarity scores, and the final prompt, which makes the system transparent.

For prompt engineering, I tested different prompt templates. The final version instructs the model to only answer from retrieved context and say when the documents are insufficient.

For evaluation, I tested adversarial questions such as ambiguous and misleading queries. I compared the RAG system with a pure LLM and found that RAG gives more grounded answers.

My innovation is memory-based RAG, which helps the chatbot understand follow-up questions during a conversation.

Finally, I built a Streamlit interface and prepared the project for GitHub and cloud deployment.

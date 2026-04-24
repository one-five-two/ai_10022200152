# Project Cleanup and Validation Notes

What was done:
- Extracted the source code from the DOCX into proper files and folders.
- Removed non-code section headings that had been accidentally mixed into Python files.
- Added missing `__init__.py` files so the project imports correctly.
- Cleaned `.gitignore`.
- Added `README.md`, `docs/architecture.md`, `docs/video_walkthrough_script.md`, and `data/README.md`.
- Added `pytest` to requirements.
- Rewrote `src/retrieval/vector_store.py` to support FAISS with a NumPy fallback, making the project easier to run during testing.

Validation completed:
- Python syntax compilation passed for all project Python files.
- Basic direct checks passed for chunking and vector search functions.

Important:
- I could not run the full pytest suite in this sandbox because the test command timed out. Do not claim full test execution until you run it locally.
- You still need to add your name and index number to the README and main files.
- You must place the actual exam datasets in the `data/` folder and run ingestion before the Streamlit app can answer questions.

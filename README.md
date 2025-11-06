### Resume Summarization AI Agent (RAG + LLM):

An end-to-end pipeline that uses Retrieval-Augmented Generation (RAG), Sentence Transformers, Re-ranking and an LLM to generate concise, role-aware candidate resume summaries.

### Problem

Recruiters and hiring managers often review hundreds of resumes manually—slow, inconsistent, and error-prone. This agent extracts the information that actually matters to hiring teams and produces fast, consistent summaries.

## Goals

Serve Hiring Managers and Recruiters across functions (Engineering, Global Functions, Professional Services, etc.).
Let users query large resume sets by skills, experience, and team fit.
Reduce time-to-screen by surfacing the most relevant candidates and summaries.

---

### How It Works (Architecture)

1. **Data Loading:** Read resumes stored in Unity Catalog Volumes / Delta.

2. **Chunking & Embeddings:** Split resumes into ~800-character chunks.

3. **Embed with all-MiniLM-L6-v2 (384-dim) via SentenceTransformer.**

4. **Semantic Retrieval (Recall):** Cosine similarity over embeddings to fetch top-k relevant chunks (FAISS or equivalent index recommended).

5. **Re-ranking (Precision):** Cross-encoder scores (query, chunk) pairs and reorders the retrieved set to keep the most relevant passages.

6. **Prompt Construction:** Insert top chunks into a concise, instruction-driven prompt with rules/constraints (focus on role fit, impact, skills, recency).

7. **LLM Generation:** Call Databricks-hosted Llama endpoint to produce the final summary/answer.

8. **Evaluation & Observability:** Track latency, error rate, retrieval quality, and token/cost.
 
---

#### Project Structure:

1. Read & Chunking: read_chunk_resume() splits text → list of chunks.

2. Embeddings: SentenceTransformer("all-MiniLM-L6-v2").encode(chunks) → vectors.

3. Retrieve: cosine similarity to get top-k chunks.

4. Re-rank: using cross-encoder (cross-encoder/ms-marco-MiniLM-L-6-v2) for re-scores.

5. Prompt: build_prompt(context, query) adds rules/indtructions + top chunks.

6. Generate: call Databricks Llama endpoint with max_tokens limit.

7. Log: latency, errors, rank metrics, token usage.

   
---

**Model Output** and **Evaluation Metrics:**::

<img width="605" height="162" alt="Screenshot 2025-11-06 at 15 15 31" src="https://github.com/user-attachments/assets/09998a6a-57b8-479f-9bed-cdcc1aa86268" />

---

**Instructions to Run:**

**Backend (Databricks + Python)** Create a Databricks token and Llama endpoint (model serving).

**Install deps:**

pip install sentence-transformers torch requests faiss-cpu

**Configure environment variables:**

DATABRICKS_HOST, DATABRICKS_TOKEN, LLM_ENDPOINT (your model serving URL), Load resumes (Change path your specified path location, run embedding + index build, then start the API script that calls the Llama endpoint.

---

📜 **License**

This project is intended for educational and internal enterprise use.

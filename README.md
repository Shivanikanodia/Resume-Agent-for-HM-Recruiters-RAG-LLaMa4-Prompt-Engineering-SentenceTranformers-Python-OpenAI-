### Resume Summarization AI Agent (RAG + LLM)

An end-to-end AI pipeline built on **Databricks** that leverages **Retrieval-Augmented Generation (RAG)** and **Large Language Models (LLMs)** to automatically generate **professional resume summaries**.

---

###  Problem Statement
Recruiters and hiring managers often have to manually review hundreds of resumes — a time-consuming and error-prone process.  
This project aims to **streamline recruitment** by automatically creating **concise, professional summaries** from resumes, ensuring only the **most relevant and important content** is highlighted.

---

###  Project Goals
- Reduce manual effort in resume screening.
- Generate **clear, concise, and relevant** summaries.
- Retrieve **only the most important sections** from resumes using semantic similarity before LLM processing.

---

### How It Works
1. **Data Loading** – Resumes are read from Unity Catalog Volumes using Apache Spark.
2. **Chunking & Embedding** – Each resume is split into chunks, embedded using Sentence Transformers.
3. **Semantic Retrieval** – Relevant chunks are retrieved based on query similarity.
4. **Prompt Construction** – Retrieved content is passed into a carefully engineered LLM prompt.
5. **Summary Generation** – LLaMA 4 (Databricks-hosted) generates the final professional summary.
6. **Output Storage** – Summaries are saved back to Databricks for recruiter access.
 
---

#### Project Structure:

---

#### Installation:
Make sure you are running inside a **Databricks Notebook** with the following installed:
```bash
%pip install sentence-transformers


#### Usage:

Upload your resumes to a Unity Catalog Volume.

Open the notebook in Databricks.

#### Configure:

LLM endpoint URL & token

Resume data path from your database or connect with ATS to retrive resumes. 

Run all cells — summaries will be generated and stored.


📜 License

This project is intended for educational and internal enterprise use.

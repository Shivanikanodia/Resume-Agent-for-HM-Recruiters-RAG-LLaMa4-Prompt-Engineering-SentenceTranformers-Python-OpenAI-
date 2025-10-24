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

**1.Reading and Chunking Resume:**
LLMs perform better with concise inputs. We are chunking resumes into small parts (~800 chars) to be later searched semantically using read_chunk_resume function.

**2.Generating Embeddings:**
Embeddings converts chunks into high-dimensional vectors so we can find semantically similar text based on a query.
A pretrained transformer model is loaded using SentenceTransformer(all-MiniLM-L6-v2). It is an efficient choice that generates 384-dimensional embeddings. The model.encode() function takes a list of text chunks.

**3.Semantic Retrieval:**
Finds the most relevant text chunks from a list based on semantic similarity to a given query using cosine similarity, saving tokens and improving accuracy.

**4.Calling Datbricks LLAMA-4 Endpoint API to generate outputs:**

<img width="770" height="624" alt="Screenshot 2025-10-24 at 12 17 33" src="https://github.com/user-attachments/assets/82a6dd28-e614-428a-95ce-779080977638" />

This script interacts with a Databricks-hosted large language model (LLM) by sending a prompt and receiving a summarized response. The API URL is constructed dynamically using the model name to ensure the request reaches the correct endpoint.

The max_tokens parameter is used to limit the length of the model's response, ensuring it remains concise and controlled. Finally, the requests.post() function is used to make the HTTP request, sending the payload and headers to the LLM endpoint and retrieving the generated output.

**5.Prompt Engineering and Fine Tuning**

<img width="784" height="637" alt="Screenshot 2025-10-24 at 12 18 52" src="https://github.com/user-attachments/assets/da7cca00-c37a-46e3-8dfb-39923c0558ec" />

This function is designed to build a well-structured prompt that will be sent to a Large Language Model (LLM) for resume summarization.

It takes two main inputs: Context, which consists of relevant resume chunks retrieved through semantic search, and query, which is the specific instruction or question for the model to answer (e.g., "Summarize this candidate’s experience").

The goal is to format these inputs into a cohesive and clear prompt that guides the LLM to produce accurate, concise, and relevant summaries. The structured format helps ensure the model focuses only on the important information from the resume while following the desired task.

**Model Output**::

<img width="793" height="274" alt="Screenshot 2025-10-24 at 12 19 23" src="https://github.com/user-attachments/assets/368d9023-024c-4c05-83d8-c928ebca5138" />

---

## Installation and Usage:


---

📜 License

This project is intended for educational and internal enterprise use.

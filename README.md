### Resume Summarization AI Agent (RAG + LLM)

An end-to-end AI pipeline built on **Databricks** that leverages **Retrieval-Augmented Generation (RAG)** and **Large Language Models (LLMs)** to automatically generate **professional resume summaries**.

---

###  Problem Statement
Recruiters and hiring managers often have to manually review hundreds of resumes — a time-consuming and error-prone process.  
This project aims to **streamline recruitment** by extracting information from resumes matters most to Hiring Managers and Recruiters.  

---

###  Project Goals
Automates resume screening by retrieving key sections, using semantic similarity, Re-Ranking and generating clear, concise, and relevant summaries.

---

### How It Works:

1. Data Loading – Read resumes from Unity Catalog Volumes.

2. Chunking & Embedding – Split each resume and embed with all-MiniLM-L6-v2 (384-D vectors).

3. Semantic Retrieval (Bi-encoder, recall-focused) – Use ANN/cosine search to pull the most likely relevant chunks.

4. Re-ranking (Cross-encoder, precision-focused) – Score query–chunk pairs and sort to keep the best matches.

5. Prompt Construction – Insert top chunks with instructions/formatting/constraints (and cite sources if needed).

6. Summary/Answer Generation – Use the LLM (e.g., Databricks-hosted Llama) to produce the final response for hiring managers.

7. Evaluation & Observability – Track latency, error rate, and relevance/quality, plus token/cost.
 
---

#### Project Structure:

**1.Reading and Chunking Resume:**
LLMs perform better with concise inputs. We are chunking resumes into small parts (~800 chars) to be later searched semantically using read_chunk_resume function.

**2.Generating Embeddings:**
Embeddings converts chunks into high-dimensional vectors so we can find semantically similar text based on a query.
A pretrained transformer model is loaded using SentenceTransformer(all-MiniLM-L6-v2). It is an efficient choice that generates 384-dimensional embeddings. The model.encode() function takes a list of text chunks.

**3.Semantic Retrieval:**
Finds the most relevant text chunks from a list based on semantic similarity to a given query using cosine similarity, saving tokens and improving accuracy.

**Chunks Re-ranking**

<img width="2096" height="652" alt="image" src="https://github.com/user-attachments/assets/5e96be2b-e314-4449-8424-86367c19d1ad" />


**4.Calling Datbricks LLAMA-4 Endpoint API to generate outputs:**

<img width="770" height="624" alt="Screenshot 2025-10-24 at 12 17 33" src="https://github.com/user-attachments/assets/82a6dd28-e614-428a-95ce-779080977638" />

This script interacts with a Databricks-hosted large language model (LLM) by sending a prompt and receiving a summarized response. The API URL is constructed dynamically using the model name to ensure the request reaches the correct endpoint.

The max_tokens parameter is used to limit the length of the model's response, ensuring it remains concise and controlled. Finally, the requests.post() function is used to make the HTTP request, sending the payload and headers to the LLM endpoint and retrieving the generated output.

**5.Prompt Engineering and Fine Tuning**

<img width="784" height="637" alt="Screenshot 2025-10-24 at 12 18 52" src="https://github.com/user-attachments/assets/da7cca00-c37a-46e3-8dfb-39923c0558ec" />

This function is designed to build a well-structured prompt that will be sent to a Large Language Model (LLM) for resume summarization.

It takes two main inputs: Context, which consists of relevant resume chunks retrieved through semantic search, and query, which is the specific instruction or question for the model to answer (e.g., "Summarize this candidate’s experience").

The goal is to format these inputs into a cohesive and clear prompt that guides the LLM to produce accurate, concise, and relevant summaries. The structured format helps ensure the model focuses only on the important information from the resume while following the desired task.

---

**Model Output**::

<img width="2096" height="578" alt="image" src="https://github.com/user-attachments/assets/be61dc00-b4c7-4bd2-9d77-90ca7301cb7d" />


--- 

**Evaluation Metrics:**


<img width="758" height="166" alt="image" src="https://github.com/user-attachments/assets/ef695a10-79b8-42bc-be70-f4e40634383d" />

--- 

**Streamlit UI**

<img width="930" height="602" alt="Screenshot 2025-11-02 at 11 20 31" src="https://github.com/user-attachments/assets/8d7cfd70-9b08-499f-8dfc-f6f2e01ae84b" />


---

**Instructions to Run:**

To Run Resume Agent file.ipynp with LLAMA-4 and Sentence Transformers create databricks token and use LLAMA-4 Endpoint URL, host it on databricks platform. 
Install Sentence Transformers, Cross Encoder, import requests and Pathlib. 

To Run test_file you need Groq API Key and run pip install streamlit and PyPDF2 for UI in your terminal. 

📜 **License**

This project is intended for educational and internal enterprise use.

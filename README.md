### Resume Summarization AI Agent (RAG + LLM)

An end-to-end AI pipeline built on Databricks that leverages Retrieval-Augmented Generation (RAG) and Large Language Models (LLMs) to automatically generate professional resume summaries.

#### 📌 Problem Statement:

Recruiters and hiring managers often have to manually review hundreds of resumes — a time-consuming and error-prone process. This project aims to streamline recruitment by automatically creating concise, professional summaries from resumes, ensuring only the most relevant and important content is highlighted.

#### 🎯 Project Goals:

Reduce manual effort in resume screening.

Generate clear, concise, and relevant summaries.

Retrieve only the most important sections from resumes using semantic similarity before LLM processing.

#### 🛠️ Technologies Used

- Python – Core language for building and orchestrating the pipeline.

- Sentence Transformers (all-MiniLM-L6-v2) – For semantic search and intelligent chunk retrieval.

- Databricks LLM Endpoint (LLaMA 4) – Hosted Large Language Model for high-quality summarization.

- Apache Spark + Unity Catalog – Distributed processing and secure access to resume data.

- Prompt Engineering – Optimized instructions for accurate and context-aware model output.

- REST API Calls – Integration with Databricks LLM endpoints.

- Databricks Notebooks – For pipeline execution and orchestration.

#### ⚙️ How It Works

1. Data Loading – Resumes are read from Unity Catalog Volumes using Apache Spark.

2. Chunking & Embedding – Each resume is split into chunks, embedded using Sentence Transformers.

3. Semantic Retrieval – Relevant chunks are retrieved based on query similarity.

4. Prompt Construction – Retrieved content is passed into a carefully engineered LLM prompt.

5. Summary Generation – LLaMA 4 (Databricks-hosted) generates the final professional summary.

6. Output Storage – Summaries are saved back to Databricks for recruiter access.

#### 📂 Project Structure
Resume Summarizer Agent (3).py   # Main Databricks notebook script

#### 🚀 Installation

Make sure you are running inside a Databricks Notebook with the following installed:

%pip install sentence-transformers

▶️ Usage

Upload your resumes to a Unity Catalog Volume.

Open the notebook in Databricks.

Configure:

LLM endpoint URL & token

Resume data path in Unity Catalog

Run all cells — summaries will be generated and stored.

🔑 Example Output

Input:

[Resume text...]


Output:

"Results-driven data analyst with 5 years of experience in..."

📈 Benefits

Cuts resume review time drastically.

Improves accuracy & relevance of summaries.

Can be scaled to thousands of resumes.

📜 License

This project is intended for educational and internal enterprise use.

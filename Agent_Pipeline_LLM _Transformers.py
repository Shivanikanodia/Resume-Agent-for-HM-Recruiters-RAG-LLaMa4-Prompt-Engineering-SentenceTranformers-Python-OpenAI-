# Databricks notebook source
# MAGIC %md
# MAGIC ####  Resume Summarization AI Agent (RAG + LLM):
# MAGIC An end-to-end pipeline that uses Retrieval-Augmented Generation (RAG), Sentence Transformers, Re-ranking and an LLM to generate concise, role-aware candidate resume summaries.
# MAGIC
# MAGIC ####  Problem:
# MAGIC Recruiters and hiring managers often review hundreds of resumes manually—slow, inconsistent, and error-prone. This agent extracts the information that actually matters to hiring teams and produces fast, consistent summaries.
# MAGIC
# MAGIC ####  Goal:
# MAGIC Serve Hiring Managers and Recruiters across functions (Engineering, Global Functions, Professional Services, etc.). Let users query large resume sets by skills, experience, and team fit. Reduce time-to-screen by surfacing the most relevant candidates and summaries.  

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## How It Works (Architecture):
# MAGIC
# MAGIC  - Data Loading: Read resumes stored in Unity Catalog Volumes / Delta.
# MAGIC
# MAGIC - Chunking & Embeddings: Split resumes into ~800-character chunks.
# MAGIC
# MAGIC - Embed with all-MiniLM-L6-v2 (384-dim) via SentenceTransformer.
# MAGIC
# MAGIC - Semantic Retrieval (Recall): Cosine similarity over embeddings to fetch top-k relevant chunks (FAISS or equivalent index recommended).
# MAGIC
# MAGIC - Re-ranking (Precision): Cross-encoder scores (query, chunk) pairs and reorders the retrieved set to keep the most relevant passages.
# MAGIC
# MAGIC - Prompt Construction: Insert top chunks into a concise, instruction-driven prompt with rules/constraints (focus on role fit, impact, skills, recency).
# MAGIC
# MAGIC - LLM Generation: Call Databricks-hosted Llama endpoint to produce the final summary/answer, REST API Calls (via requests) – To communicate with the Databricks LLM endpoint for inference 
# MAGIC
# MAGIC - Evaluation & Observability: Track latency, error rate, retrieval quality, and token/cost.  

# COMMAND ----------

# resume_summary_pipeline.py

%pip install sentence-transformers

from pathlib import Path
import textwrap
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import requests
from pyspark.sql import SparkSession

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 1. Reading and Chunking Resume: 
# MAGIC
# MAGIC LLMs perform better with concise inputs. We are chunking long resumes into small parts (~800 chars) to be later searched semantically using **read_chunk_resume** function. 

# COMMAND ----------


from pathlib import Path
import textwrap

def read_and_chunk_resume(path, width=800):
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    return textwrap.wrap(text, width=width)



# COMMAND ----------

# MAGIC %md
# MAGIC Governance and Privacy - PII redaction: Masking name, organisation/emailaddress before retrieval is crucial for securing sensitive employee information. 

# COMMAND ----------

import re

def anonymize_text(text: str) -> str:
    """
    Redacts candidate names and company names using simple regex patterns.
    """
    # Common company suffixes
    company_pattern = re.compile(
        r"\b[A-Z][A-Za-z&\.\- ]{1,30}\s+(Inc|Ltd|LLC|Technologies|Solutions|Consulting|Corporation|Corp|Systems|Group|Company|Enterprises)\b",
        flags=re.I
    )

    # Detect names (2 capitalized words like "John Smith")
    name_pattern = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")

    text = company_pattern.sub("[COMPANY]", text)
    text = name_pattern.sub("[CANDIDATE]", text)
    return text


# COMMAND ----------

# MAGIC %md
# MAGIC **Path(path).read_text(),** Reads the content of the file at the specified path. 
# MAGIC
# MAGIC **textwrap.wrap(text, width=800),** Splits the long resume string into chunks of **~800 characters** and which avoids cutting off, which improves token management for LLMs. 

# COMMAND ----------

# MAGIC %md
# MAGIC #####2.  Generating Embeddings:
# MAGIC Embeddings converts chunks into high-dimensional vectors so we can find semantically similar text based on a query.  
# MAGIC

# COMMAND ----------

from sentence_transformers import SentenceTransformer

def get_embeddings(chunks, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, normalize_embeddings=True)
    return embeddings, model


# COMMAND ----------

# MAGIC %md
# MAGIC The SentenceTransformer('all-MiniLM-L6-v2') model converts text chunks into dense numeric vectors (embeddings) that represent their meaning. The model.encode() function takes a list of text chunks and creates a 384-dimensional embedding for each one. This process returns two things:
# MAGIC
# MAGIC Embeddings – a list of vector representations for each text chunk.
# MAGIC
# MAGIC The model – which can be reused later to embed new text or queries. 

# COMMAND ----------

# MAGIC %md
# MAGIC #####  3.Re-ranking using Cross Encoder to score and order the relevant chunks. 

# COMMAND ----------

import numpy as np
from sentence_transformers import CrossEncoder

def rerank_with_cross_encoder(query, candidate_pairs, cross_encoder_name="cross-encoder/ms-marco-MiniLM-L-6-v2", top_k=3):
    ce = CrossEncoder(cross_encoder_name)
    pairs = [(query, txt) for _, txt in candidate_pairs]
    scores = ce.predict(pairs)
    order = np.argsort(-scores)[:top_k]
    return [candidate_pairs[i][0] for i in order]



# COMMAND ----------

# MAGIC %md
# MAGIC This function reranks text chunks based on their relevance to different queries  using a pre-trained CrossEncoder model.
# MAGIC It pairs the query with each text chunk, predicts relevance scores, sorts them in descending order, and returns the indices of the top_k most relevant chunks. 

# COMMAND ----------

from sklearn.metrics.pairwise import cosine_similarity

def get_top_k_chunks(query, chunks, embeddings, model, top_k=5, use_cross_encoder=True, bi_encoder_candidates=10):
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    sims = cosine_similarity([q_emb], embeddings)[0]
    cand_ids = np.argsort(-sims)[:max(top_k, bi_encoder_candidates)]
    candidates = [(i, chunks[i]) for i in cand_ids]
    if use_cross_encoder:
        top_ids = rerank_with_cross_encoder(query, candidates, top_k=top_k)
    else:
        top_ids = list(cand_ids[:top_k])
    return [chunks[i] for i in top_ids]


# COMMAND ----------

# MAGIC %md
# MAGIC The function first uses a bi-encoder to quickly find the most similar chunks to a query based on cosine similarity.
# MAGIC Then, if enabled, it uses a cross-encoder to rerank those top candidates for higher accuracy before returning the best top_k chunks.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ##### Calling Databricks LLM endpoint LLAMA-4. 

# COMMAND ----------

import os
import requests

def call_databricks_llm(prompt, model="databricks-llama-4-maverick", max_tokens=500):
    os.environ["DATABRICKS_HOST"] = DATABRICKS_HOST
    os.environ["DATABRICKS_TOKEN"] = DATABRICKS_TOKEN

    api_url = f"{DATABRICKS_HOST}/serving-endpoints/{model}/invocations"
    headers = {
        "Authorization": f"Bearer {os.environ['DATABRICKS_TOKEN']}",
        "Content-Type": "application/json"
    }

    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that summarizes resumes."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens
    }

    response = requests.post(api_url, headers=headers, json=payload)
    if response.status_code == 200:
        r = response.json()
        if "predictions" in r:
            return r["predictions"][0]["message"]["content"]
        elif "choices" in r:
            return r["choices"][0]["message"]["content"]
        elif "data" in r:
            return r["data"]["message"]
        else:
            raise Exception(f"Unknown response structure: {r}")
    else:
        raise Exception(f"Request failed: {response.status_code} — {response.text}")



# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC This script interacts with a Databricks-hosted large language model (LLM) by sending a prompt and receiving a summarized response. The API URL is constructed dynamically using the model name to ensure the request reaches the correct endpoint.
# MAGIC
# MAGIC The payload follows the OpenAI-style chat format, which includes a system message to set the assistant's behavior and a user message containing the actual prompt. 
# MAGIC
# MAGIC The max_tokens parameter is used to limit the length of the model's response, ensuring it remains concise and controlled. Finally, the requests.post() function is used to make the HTTP request, sending the payload and headers to the LLM endpoint and retrieving the generated output.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #### Format Prompt with Prompt Engineering: 

# COMMAND ----------


def format_prompt(context, query):
    return f"""
[System]
You are a Resume Summarization Agent for hiring managers and recruiters.
Your goal is to generate a relevance candidate summary aligned with Hiring Managers Expectations. 

[Rules]
- Use ONLY the provided resume text. Do not hallucinate or make up content.
- Output 5–6 bullets max. 
- Prioritize dated experience and measurable impact over skill lists.
- Prefer recent (last 24 months) items when choosing examples.
- No filler, no duplication, no assumptions.
- Focus on relevance Job role (skills, experience, and Impact alignment).
- If multiple roles or domains are present, highlight the most relevant one to the query.
- Apply section weighting when selecting evidence and examples. Order of priority:
  1) Experience (1.30x)
  2) Education (1.10x)
  3) Skills (0.90x)
  4) Other (0.70x)
- When in doubt, pick items from higher-weight sections; include lower-weight sections only if they clearly strengthen relevance to the query and Job Desciption. 

[Output format]
• Experience & Domain: <Years> across <key domains>; recent roles at <2 most recent companies/titles>.
• Tech Stack: Mention only what appears in the resume.
• Responsibilities/Projects:  4 aligned tasks, if present. 
  Impact Highlights:   2–3 short clauses with numbers.
• Risks/Notes:   gaps >6mo and roles <12months
• Relevance Summary: Overall fit to the query/Job (High / Medium / Low).

[Resume]
{context}

[Task]
"{query}"
"""


# COMMAND ----------

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

_relevance_model = SentenceTransformer("all-MiniLM-L6-v2")

def relevance_score(summary, context_chunks):
    summary_emb = _relevance_model.encode([summary], normalize_embeddings=True)
    context_emb = _relevance_model.encode([" ".join(context_chunks)], normalize_embeddings=True)
    cos = float(cosine_similarity(summary_emb, context_emb)[0][0])


    score = max(0.0, min(1.0, cos))
    return score


# COMMAND ----------

def summarize_resume(file_path_txt, query):
    chunks = read_and_chunk_resume(file_path_txt)
    embeddings, model = get_embeddings(chunks)
    top_chunks = get_top_k_chunks(query, chunks, embeddings, model)
    context = "\n\n".join(top_chunks)
    prompt = format_prompt(context, query)
    summary = call_databricks_llm(prompt, max_tokens=600)

    return summary


# COMMAND ----------

# MAGIC %md
# MAGIC This function computes how relevant a generated summary is to the original context using a semantic similarity score.

# COMMAND ----------

def summarize_all_resumes(file_paths, query):
    results = []
    errors = 0
    total_time = 0
    relevance_scores = []

    for path in file_paths:
        start = time.time()
        try:
            result = summarize_resume(path, query)
            result_text = result if isinstance(result, str) else result[0]
            relevance = relevance_score(result_text, read_and_chunk_resume(path))
            results.append((path, result_text, relevance))
            relevance_scores.append(relevance)
        except Exception as e:
            print(f"❌ Error on {path}: {e}")
            errors += 1
        total_time += time.time() - start

    avg_latency = total_time / len(file_paths)
    avg_rel = np.mean(relevance_scores)
    err_rate = errors / len(file_paths)

    print("\n### **Evaluation Summary**")
    print("| Metrics | ")
    print(f"| Average Latency: | {avg_latency:.2f} sec/resume |")
    print(f"| Error Rate: | {err_rate*100:.1f}% |")
    print(f"| Average Relevance Score: | {avg_rel:.2f} |")


    # Sort by relevance so best ones come first
    results.sort(key=lambda x: x[2], reverse=True)

    for path, summary, rel in results:
        name = Path(path).stem
        print(f"\n---\n#### {name} — Relevance: {rel:.2f}")
        print(summary.strip())


# COMMAND ----------

# MAGIC %md
# MAGIC This function loops through multiple resumes, summarizes each one using your summarize_resume() function, then measures:
# MAGIC
# MAGIC - Latency (how long each took)
# MAGIC - Error rate, and
# MAGIC - Average relevance (how semantically similar each summary is to its original resume, via your relevance_score() function). 

# COMMAND ----------

import time

if __name__ == "__main__":
    resumes = [
        "/Volumes/workspace/default/default_resume/Haiming Wang.txt",
        "/Volumes/workspace/default/default_resume/Kayuri Shah.txt",
        "/Volumes/workspace/default/default_resume/Raja_Agarwal.txt",
        "/Volumes/workspace/default/default_resume/Sakshi_Gundawar.txt",
    ]

    query = " Hide candidate organization and Name, summaries resume"
    summarize_all_resumes(resumes, query)



# COMMAND ----------

# MAGIC %md
# MAGIC #### Future Work: 
# MAGIC
# MAGIC 2. Creating a Front End UI for Hiring Managers which will integrate with ATS and have options for them to write query, see summaries and download only relevant resumes. 
# MAGIC
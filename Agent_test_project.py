import streamlit as st
import PyPDF2
import requests
import os

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text


from groq import Groq

client = Groq(
    api_key="",
)

def score_resume_with_groq(resume_text, job_description, api_key):
    from groq import Groq
    import re

    # Prepare prompt
    prompt = f"""You are a smart hiring assistant. 
                Given the following resume and job description, provide only a single integer score from 0 to 100 indicating the suitability of the resume for this job, where 100 means a perfect match and 0 means no match at all. Do not provide any explanation. Output just the number.

                Resume:
                {resume_text}

                Job Description:
                {job_description}
                """

    try:
        client = Groq(api_key=api_key)
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an expert job matching AI."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.1-8b-instant",
            max_tokens=8,
            temperature=0.0,
        )
        model_reply = chat_completion.choices[0].message.content
        # Extract score (strip everything except digits, take the first found number)
        possible_nums = re.findall(r"\d+", model_reply)
        score = int(possible_nums[0]) if possible_nums else 0
        score = min(max(score, 0), 100)
        return score
    except Exception as e:
        return f"Error: {str(e)}"


st.title("Resume Hire-ability Scorer (via Groq Llama 8B)")

uploaded_file = st.file_uploader("Upload your resume (PDF)", type="pdf")
job_description = st.text_area("Paste the Job Description here")

GROQ_API_KEY = "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"

if uploaded_file is not None and job_description.strip():
    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY not found in Streamlit Secrets or Environment. Please set it to use Groq API.")
    else:
        resume_text = extract_text_from_pdf(uploaded_file)
        if not resume_text.strip():
            st.warning("Could not extract text from the PDF. Please try another file.")
        else:
            with st.spinner("Scoring using Llama 8B via Groq..."):
                score = score_resume_with_groq(resume_text, job_description, GROQ_API_KEY)
            st.success(f"Hire-ability Score based on Job Description: {score}/100")

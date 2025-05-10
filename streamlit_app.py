import os
import re
import PyPDF2
import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# === Load API Key from Secrets ===
load_dotenv()
openai_key = st.secrets["OPENAI_API_KEY"]

# === Streamlit Configuration ===
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

# === Page Styling ===
st.markdown("""
    <style>
        .title {
            font-size: 3em;
            text-align: center;
            color: #4CAF50;
            font-family: 'Segoe UI', sans-serif;
            margin-bottom: 0.2em;
        }
        .subtitle {
            font-size: 1.4em;
            text-align: center;
            color: #555;
            font-family: 'Segoe UI', sans-serif;
        }
        .section-title {
            font-size: 1.6em;
            font-weight: bold;
            margin-top: 2em;
            color: #2E8B57;
        }
        .box {
            background-color: #f7f7f7;
            border-left: 5px solid #4CAF50;
            padding: 1em;
            margin-top: 1em;
            font-family: 'Courier New', monospace;
            white-space: pre-wrap;
        }
    </style>
""", unsafe_allow_html=True)

# === Title & Subtitle ===
st.markdown('<div class="title">🚀 AI-Powered Resume Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload your resume (PDF) to get your ATS Score and expert feedback</div>', unsafe_allow_html=True)

# === File Upload ===
uploaded_file = st.file_uploader("📄 Upload Resume", type=["pdf"])

if uploaded_file:
    with st.spinner("🔍 Analyzing your resume..."):

        # --- Read PDF ---
        def read_pdf(file_obj):
            text = ""
            reader = PyPDF2.PdfReader(file_obj)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text

        pdf_text = read_pdf(uploaded_file)
        documents = [Document(page_content=pdf_text)]

        # --- Text Splitting ---
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_docs = splitter.split_documents(documents)

        # --- Embedding and Vector Store ---
        embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
        vectorstore = FAISS.from_documents(split_docs, embedding=embeddings)

        # --- Prompt Template ---
        prompt = PromptTemplate.from_template("""
        You are an ATS (Applicant Tracking System) Resume Evaluator.
        Given the resume content below, please provide:
        1. An estimated ATS Score out of 100 (clearly state it).
        2. Professional feedback and suggestions for improvement.
        
        Resume:
        {context}
        """)

        formatted_question = prompt.format(context=pdf_text)

        # --- LLM QA Chain ---
        retriever = vectorstore.as_retriever()
        llm = OpenAI(openai_api_key=openai_key)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        response = qa_chain.run(formatted_question)

        # --- Extract ATS Score ---
        ats_score_match = re.search(r'(?i)ATS Score.*?(\d{1,3})', response)
        ats_score = int(ats_score_match.group(1)) if ats_score_match else None

        # === Display Result ===
        st.success("✅ Resume Analysis Complete!")

        # === ATS Score Section ===
        st.markdown('<div class="section-title">📈 ATS Score</div>', unsafe_allow_html=True)
        if ats_score is not None:
            st.markdown(f"""
                <div class="box">
                Your estimated ATS Score is: **{ats_score} / 100**
                </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ ATS Score could not be extracted from the model's response.")

        # === Feedback Section ===
        st.markdown('<div class="section-title">💡 Feedback & Advice</div>', unsafe_allow_html=True)
        st.markdown(f"<div class='box'>{response}</div>", unsafe_allow_html=True)

else:
    st.info("📎 Please upload a resume to begin the analysis.")

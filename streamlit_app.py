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

# === Load Environment Variables ===
load_dotenv()
openai_key = st.secrets["OPENAI_API_KEY"]

# === Streamlit Page Config ===
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

# === Custom Styling ===
st.markdown("""
    <style>
        body {
            background-color: #f0f2f6;
        }
        .title {
            font-size: 3em;
            text-align: center;
            color: #4CAF50;
            font-family: 'Segoe UI', sans-serif;
            margin-bottom: 0.2em;
        }
        .subtitle {
            font-size: 1.5em;
            text-align: center;
            color: #777;
            font-family: 'Segoe UI', sans-serif;
        }
        .section {
            background-color: #ffffff;
            padding: 2em;
            margin: 2em 0;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .section h2 {
            color: #4CAF50;
            font-family: 'Segoe UI', sans-serif;
        }
        .score {
            font-size: 2em;
            color: white;  /* White text for ATS score */
            text-align: center;
            background-color: #4CAF50;  /* Green background for contrast */
            padding: 0.5em;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# === Title & Subtitle ===
st.markdown('<div class="title">🚀 AI-Powered Resume Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload your resume to get ATS score, feedback, and advice</div>', unsafe_allow_html=True)

# === File Upload ===
uploaded_file = st.file_uploader("📄 Upload your Resume (PDF)", type=["pdf"])

# === Resume Processing ===
if uploaded_file:
    with st.spinner("🔍 Reading and analyzing your resume..."):

        # --- Extract Text from PDF ---
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

        # --- Text Splitting & Embedding ---
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_docs = splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
        try:
            vectorstore = FAISS.from_documents(split_docs, embedding=embeddings)
        except ImportError as e:
            st.error(f"ImportError: {e}")
            raise

        # --- LLM QA Chain ---
        llm = OpenAI(openai_api_key=openai_key)
        prompt = PromptTemplate.from_template("""
        You are a highly professional ATS Resume Analyzer. Based on the context below (a resume), provide:

        1. ATS Score (out of 100): Provide only the numeric score.
        2. Feedback: Detailed and constructive feedback on how to improve the resume.
        3. Advice: Specific suggestions to enhance the resume's effectiveness.

        Format your response with clear section headings: "ATS Score", "Feedback", and "Advice".

        Context:
        {context}
        """)
        formatted_question = prompt.format(context=pdf_text)

        retriever = vectorstore.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        response = qa_chain.run(formatted_question)

        # --- Extract ATS Score using regex ---
        ats_score_match = re.search(r'ATS Score.*?(\d{1,3})', response, re.IGNORECASE)
        ats_score = int(ats_score_match.group(1)) if ats_score_match else None

    # === Display Results ===
    st.success("✅ Analysis Complete!")

    # --- ATS Score Section ---
    if ats_score is not None:
        st.markdown('<div class="section"><h2>📊 ATS Score</h2>', unsafe_allow_html=True)
        st.markdown(f'<div class="score">{ats_score}/100</div></div>', unsafe_allow_html=True)
    else:
        st.warning("Could not extract ATS score from the response.")

    # --- Feedback Section ---
    feedback_match = re.search(r'Feedback:\s*(.*?)\s*Advice:', response, re.DOTALL | re.IGNORECASE)
    feedback = feedback_match.group(1).strip() if feedback_match else "No feedback available."

    st.markdown('<div class="section"><h2>📝 Feedback</h2>', unsafe_allow_html=True)
    st.markdown(f"<p>{feedback}</p></div>", unsafe_allow_html=True)

    # --- Advice Section ---
    advice_match = re.search(r'Advice:\s*(.*)', response, re.DOTALL | re.IGNORECASE)
    advice = advice_match.group(1).strip() if advice_match else "No advice available."

    st.markdown('<div class="section"><h2>💡 Advice</h2>', unsafe_allow_html=True)
    st.markdown(f"<p>{advice}</p></div>", unsafe_allow_html=True)

else:
    st.info("Please upload a resume to begin the analysis.")

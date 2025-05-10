import os
import re
import PyPDF2
import streamlit as st
import plotly.graph_objects as go
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
    </style>
""", unsafe_allow_html=True)

# === Title & Subtitle ===
st.markdown('<div class="title">🚀 AI-Powered Resume Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload your resume to get ATS score and expert feedback</div>', unsafe_allow_html=True)

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
        1. An estimated ATS Score (out of 100).
        2. Detailed and constructive feedback on how to improve the resume.

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

        # Add dynamic ATS score line to the response
        if ats_score is not None:
            response = f"An estimated ATS Score (out of 100): {ats_score}/100\n\n" + response

    # === Display Results ===
    st.success("✅ Analysis Complete!")

    # --- ATS Score Gauge ---
    if ats_score is not None:
        st.subheader("📊 ATS Score")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=ats_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "ATS Compatibility", 'font': {'size': 24}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#4CAF50"},
                'bgcolor': "white",
                'steps': [
                    {'range': [0, 50], 'color': '#ffcccc'},
                    {'range': [50, 75], 'color': '#fff0b3'},
                    {'range': [75, 100], 'color': '#d4edda'}
                ],
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Could not extract ATS score from the response.")

    # --- Resume Feedback ---
    st.subheader("📝 Resume Feedback")
    st.markdown(f"```markdown\n{response}\n```")

else:
    st.info("Please upload a resume to begin the analysis.")

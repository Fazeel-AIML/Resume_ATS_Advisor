# 🚀 AI-Powered Resume Analyzer

This is an intelligent web application built with **Streamlit**, **LangChain**, and **OpenAI GPT** that analyzes resumes (PDF) for **ATS (Applicant Tracking System) compatibility** and provides **expert feedback** to improve the chances of landing interviews.

## 🔍 Features

- 📄 Upload and read resumes in PDF format
- 🤖 Uses OpenAI's LLM to analyze resume content
- 📊 Displays ATS compatibility score (0–100) using an interactive gauge chart
- 🧠 Provides professional, constructive feedback
- ⚙️ Built with LangChain, ChromaDB, and OpenAI embeddings
- 💬 Easy-to-use, clean UI using Streamlit

---

## 🧰 Tech Stack

| Tech            | Use Case                          |
|-----------------|-----------------------------------|
| **Python**      | Backend logic and integrations    |
| **Streamlit**   | Frontend interface                |
| **LangChain**   | LLM orchestration and prompt engineering |
| **OpenAI API**  | Resume analysis and feedback      |
| **Chroma DB**   | Document vector storage           |
| **Plotly**      | Dial-based ATS score visualization |

---

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/resume-analyzer.git
   cd resume-analyzer
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
3. Set up environment variables:
   ```bash
   OPENAI_API_KEY=your_openai_key_here

## 🚀 Run the App
```bash
stremlit run app.py

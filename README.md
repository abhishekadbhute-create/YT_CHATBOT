# 🎥 YouTube Video Chatbot (RAG AI Assistant)

An interactive **Retrieval-Augmented Generation (RAG)** web application built with **Streamlit**, **LangChain**, **FAISS**, and **Hugging Face LLMs**. This application enables users to paste any YouTube video URL, automatically extracts and translates its transcript into English, indexes the text into a vector store, and allows users to ask questions directly about the video's content.

---

## ✨ Features

- **🎥 YouTube Transcript Ingestion**: Automatically fetches transcripts for YouTube videos (supports native English transcripts as well as auto-translating non-English transcripts like Hindi into English).
- **⚡ RAG Vector Store**: Chunks transcripts using `RecursiveCharacterTextSplitter` and indexes them into a `FAISS` vector database using `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` embeddings.
- **🧠 Query Rewriting & Answering**: Uses **Llama-3.1-8B-Instruct** via `HuggingFaceEndpoint` and `ChatHuggingFace` to rewrite search queries for high-precision retrieval and answer questions accurately.
- **🌐 English Enforced Answers**: Consistently returns answers in English regardless of the original video transcript's language.
- **🔄 Session & Video Management**: Automatically resets chat history and vector stores when a new video URL is loaded to prevent context contamination between videos.

---

## 🛠️ Tech Stack

- **Frontend UI**: [Streamlit](https://streamlit.io/)
- **Orchestration**: [LangChain](https://www.langchain.com/)
- **Vector Database**: [FAISS](https://github.com/facebookresearch/faiss)
- **Embeddings**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **LLM**: `meta-llama/Llama-3.1-8B-Instruct` (Hugging Face Inference API)
- **Transcript Extraction**: `youtube-transcript-api`
- **Environment Management**: `python-dotenv`

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9+ installed.
- A free **Hugging Face Access Token** (get yours at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)).

### 2. Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/abhishekadbhute-create/YT_CHATBOT.git
   cd YT_CHATBOT
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows PowerShell:
   .\venv\Scripts\Activate.ps1
   # On Mac/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install streamlit langchain langchain-community langchain-huggingface langchain-text-splitters faiss-cpu youtube-transcript-api python-dotenv
   ```

4. **Configure Environment Variables:**
   Create a `.env` file in the root directory:
   ```env
   HUGGINGFACEHUB_API_TOKEN=hf_your_actual_huggingface_token_here
   ```

---

## 💻 Running the Application

Launch the Streamlit app:
```bash
streamlit run app.py
```

Open your browser and navigate to: **`http://localhost:8501`**

---

## 📖 Usage Guide

1. **Load Video**: Paste a YouTube URL (e.g., `https://www.youtube.com/watch?v=...` or `https://youtu.be/...`) into the input box and click **Load Video**.
2. **Ask Questions**: Type any question about the video in the query box and click **Ask**.
3. **Switching Videos**: Paste a new video URL and click **Load Video**. The app will automatically clear the old chat history and load the new video's transcript.

---

## 🛡️ License

Distributed under the MIT License. Feel free to use and modify!
import os
from pathlib import Path
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Load .env from the same folder as this script, regardless of cwd
load_dotenv(dotenv_path=Path(__file__).parent / ".env")
hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

st.title("🎥 YouTube Chatbot")

youtube_url = st.text_input("Paste YouTube Video URL")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "current_video_id" not in st.session_state:
    st.session_state.current_video_id = None

if st.button("Load Video"):

    # Get video ID
    if "v=" in youtube_url:
        video_id = youtube_url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in youtube_url:
        video_id = youtube_url.split("youtu.be/")[1].split("?")[0]
    else:
        st.error("Invalid YouTube URL")
        st.stop()

    try:
        api = YouTubeTranscriptApi()
        
        # Try fetching English transcript first, fallback to translation if needed
        try:
            transcript_list = api.fetch(video_id, languages=["en"])
        except Exception:
            try:
                transcript_list_obj = api.list(video_id)
                try:
                    transcript_obj = transcript_list_obj.find_transcript(["en"])
                except Exception:
                    first_transcript = next(iter(transcript_list_obj))
                    transcript_obj = first_transcript.translate("en")
                transcript_list = transcript_obj.fetch()
            except Exception:
                # Fallback to general fetch
                transcript_list = api.fetch(video_id, languages=["en", "hi"])

        transcript = " ".join(
            chunk.text if hasattr(chunk, "text") else chunk["text"] for chunk in transcript_list
        )

        st.success("Video loaded successfully!")

        # Just for testing
        st.write(transcript[:1000])

        # TEXT SPLITTER
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len
        )

        chunks = splitter.create_documents([transcript])

        # vector store creation
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        st.session_state.vector_store = FAISS.from_documents(
            chunks,
            embeddings
        )
        
        # Reset chat history and save active video ID
        st.session_state.chat_history = []
        st.session_state.current_video_id = video_id

    except TranscriptsDisabled:
        st.error("No captions available for this video.")
        st.stop()

    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

# QUERY INPUT
query = st.text_input(
    "Ask a question",
    placeholder="What is this video about?"
)

if st.button("Ask"):

    # Get current video ID from input box
    current_input_id = None
    if "v=" in youtube_url:
        current_input_id = youtube_url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in youtube_url:
        current_input_id = youtube_url.split("youtu.be/")[1].split("?")[0]

    if st.session_state.vector_store is None:
        st.warning("Please load a video first by clicking 'Load Video'.")
        st.stop()

    if current_input_id and current_input_id != st.session_state.current_video_id:
        st.warning("You changed the video URL! Please click 'Load Video' to process the new video before asking questions.")
        st.stop()

    if not hf_token:
        st.error(
            "No Hugging Face token found. Make sure your .env file (next to app.py) "
            "contains a line like: HUGGINGFACEHUB_API_TOKEN=hf_xxx  (no quotes around the key or value)."
        )
        st.stop()

    st.write("Question:", query)

    # PROMPT TEMPLATE FOR REWRITING THE QUERY
    context_query_prompt = PromptTemplate(
        template="""
Rewrite the given query into a clear and detailed search query in ENGLISH
that will help retrieve all relevant information from the vector store.
Return ONLY the rewritten query in English, nothing else.

Query: {query}

Search query:
""",
        input_variables=["query"]
    )

    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        task="text-generation",
        max_new_tokens=512,
        temperature=0.7,
        huggingfacehub_api_token=hf_token,
    )
    model = ChatHuggingFace(llm=llm)
    parser = StrOutputParser()

    # Actually invoke the LLM to rewrite the query
    rewrite_chain = context_query_prompt | model | parser
    search_query = rewrite_chain.invoke({"query": query}).strip()

    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 6})
    context_docs = retriever.invoke(search_query)

    context_text = "\n\n".join(doc.page_content for doc in context_docs)

    # PROMPT TEMPLATE FOR ANSWERING QUESTIONS
    answer_prompt = PromptTemplate(
        template="""
        You are a helpful assistant. Answer the user's question using ONLY the provided
        context and chat history.

        First, understand the relevant chat history and the user's current intent.
        Then use the context to give a clear, direct, and accurate answer.

        - Do not copy the context verbatim.
        - Do not add information from outside the context or chat history.
        - If the information is insufficient to answer, say: "I don't know."
        - For follow-up questions, use the chat history to maintain context.
        - ALWAYS answer ONLY in English, regardless of the language of the question or context.

        Context:
        {context}

        Chat History:
        {chat_history}

        Question:
        {query}

        Answer (in English):
        """,
        input_variables=["context", "query", "chat_history"]
    )

    answer_chain = answer_prompt | model | parser

    answer = answer_chain.invoke({
        "context": context_text,
        "query": query,
        "chat_history": st.session_state.chat_history
    })

    st.session_state.chat_history.append({
        "query": query,
        "answer": answer
    })

    st.write("### Answer")
    st.write(answer)
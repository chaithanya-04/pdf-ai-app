import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import time
from dotenv import load_dotenv
import os

# CONFIG

st.set_page_config(page_title="PDF AI", layout="wide")
st.title("PDF AI Assistant")

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# MODELS

fast_llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant"
)

smart_llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile"
)


# EMBEDDINGS

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# PDF LOADER

@st.cache_data
def load_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""
    return text

# CHUNKING

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=250
    )
    return splitter.split_text(text)

# VECTOR STORE (CACHED)

@st.cache_resource
def build_vectorstore(chunks):
    return FAISS.from_texts(chunks, embeddings)

# LIGHT DOCUMENT UNDERSTANDING

def build_doc_memory(text):
    prompt = f"""
    Understand this document in a GENERAL way:

    - What is it about?
    - What type of document is it?
    - What are key ideas?

    TEXT:
    {text[:8000]}
    """
    return fast_llm.invoke(prompt).content

# SUMMARY 

def generate_summary(text):
    prompt = f"""
    Summarize clearly:

    - Main idea
    - Key points

    TEXT:
    {text[:10000]}
    """
    return fast_llm.invoke(prompt).content

# RETRIEVAL

def retrieve_context(vs, query):
    docs = vs.similarity_search(query, k=8)
    return "\n\n".join([d.page_content for d in docs])


# STREAM OUTPUT

def stream_text(text):
    placeholder = st.empty()
    out = ""
    for w in text.split():
        out += w + " "
        placeholder.markdown(out)
        time.sleep(0.006)

# SAFETY CONTROL

if "api_count" not in st.session_state:
    st.session_state.api_count = 0

if "last_query" not in st.session_state:
    st.session_state.last_query = None

if "last_answer" not in st.session_state:
    st.session_state.last_answer = None

def safety_guard():
    st.session_state.api_count += 1

    if st.session_state.api_count > 10:
        st.error("Too many requests. Refresh app.")
        return False

    return True

def safe_call(model, prompt):
    time.sleep(1.5)
    return model.invoke(prompt)

# SESSION STATE

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "doc_memory" not in st.session_state:
    st.session_state.doc_memory = None

if "full_text" not in st.session_state:
    st.session_state.full_text = None

if "file_id" not in st.session_state:
    st.session_state.file_id = None

# FILE UPLOAD

uploaded_file = st.file_uploader("Upload ANY PDF", type="pdf")

if uploaded_file:

    if st.session_state.file_id != uploaded_file.name:
        st.session_state.vectorstore = None
        st.session_state.doc_memory = None
        st.session_state.full_text = None
        st.session_state.file_id = uploaded_file.name

    if st.session_state.vectorstore is None:

        with st.spinner("Reading PDF..."):
            text = load_pdf(uploaded_file)
            st.session_state.full_text = text

        with st.spinner("Processing..."):
            chunks = chunk_text(text)
            st.session_state.vectorstore = build_vectorstore(chunks)

        with st.spinner("Understanding document..."):
            st.session_state.doc_memory = build_doc_memory(text)

        st.success("PDF Loaded and Processed!")

# DOCUMENT VIEW

if st.session_state.doc_memory:
    st.subheader("Document Understanding")
    st.write(st.session_state.doc_memory)

# SUMMARY

if st.session_state.full_text:
    if st.button("Generate Summary"):
        summary = generate_summary(st.session_state.full_text)
        st.subheader("Summary")
        stream_text(summary)

# Q&A ENGINE

query = st.text_input("Ask anything:")

if query and st.session_state.vectorstore:

    if query == st.session_state.last_query:
        st.info("Using cached answer")
        st.write(st.session_state.last_answer)

    else:

        if safety_guard():

            context = retrieve_context(st.session_state.vectorstore, query)

            prompt = f""" You are a universal document assistant. 
            Document understanding: {st.session_state.doc_memory} 
            Context: {context} 
            Question: {query} 
            Rules: - Answer naturally 
                   - Do not assume research-paper structure
                   - Use context intelligently
                    """

            model = fast_llm if len(query) < 100 else smart_llm
            response = safe_call(model, prompt)

            st.session_state.last_query = query
            st.session_state.last_answer = response.content

            st.subheader("Answer")
            stream_text(response.content)
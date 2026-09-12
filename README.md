# PDF AI Assistant

A Retrieval-Augmented Generation (RAG) powered chatbot that lets you upload any PDF and instantly ask questions about it in natural language, no more manually scrolling through pages to find what you need.
Built with Streamlit, LangChain, FAISS, and Groq LLMs.

---

## 🚀 Why This Project?

Reading long PDFs — research papers, reports, notes — just to find one specific piece of information is slow and inefficient. PDF AI Assistant solves this by letting you upload a document once and then interact with it conversationally, getting instant, context-aware answers instead of reading the whole thing manually.

---

## Features

- Upload any PDF and get it processed instantly
- Ask questions in natural language and get accurate, context-grounded answers
- Automatic document understanding — get a general overview of what the document is about right after upload
- One-click AI summary generation for the entire document
- Fast retrieval using FAISS vector similarity search — only relevant chunks are sent to the LLM, not the whole document
- Dynamic model routing — automatically uses a lightweight LLM for short/simple queries and a larger LLM for complex ones, balancing speed and quality
- Response caching — repeated questions are answered instantly without hitting the API again
- Session-based rate limiting to prevent excessive API usage
- Streaming responses for a real-time, chat-like experience

---

## How It Works

1. **Text Extraction** — The uploaded PDF is parsed and its text extracted using `PyPDF2`.
2. **Chunking** — The extracted text is split into overlapping chunks (`RecursiveCharacterTextSplitter`) to preserve context across boundaries.
3. **Embedding & Indexing** — Each chunk is converted into a vector using HuggingFace's `all-MiniLM-L6-v2` sentence-transformer model and stored in a FAISS vector store.
4. **Document Understanding** — A lightweight LLM call generates a quick summary of what the document is and its key ideas, shown immediately after upload.
5. **Retrieval** — When a question is asked, the top-k most semantically relevant chunks are retrieved from FAISS instead of passing the entire document to the LLM.
6. **Answer Generation** — The retrieved context + question is passed to a Groq-hosted LLM (model chosen dynamically based on query complexity) to generate a grounded answer.
7. **Streaming Output** — The answer is streamed word-by-word to the UI for a smoother, chat-like feel.

```
PDF Upload → Text Extraction → Chunking → Embeddings → FAISS Vector Store
                                                               │
                                  User Question ───────────────┘
                                          │
                              Similarity Search (top-k chunks)
                                          │
                              LLM (Groq) + Retrieved Context
                                          │
                                    Streamed Answer
```

---

## Tech Stack

| Layer               | Technology                                           |
| ------------------- | ---------------------------------------------------- |
| UI                  | Streamlit                                            |
| LLM Inference       | Groq API (`gpt-oss-20b`, `gpt-oss-120b`)             |
| Orchestration / RAG | LangChain                                            |
| Vector Store        | FAISS                                                |
| Embeddings          | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| PDF Parsing         | PyPDF2                                               |
| Config Management   | python-dotenv                                        |

---

## Installation & Setup

### Prerequisites

- Python 3.9+
- A [Groq API key](https://console.groq.com/)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/chaithanya-04/pdf-ai-app.git
cd pdf-ai-app

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment variables
# Create a .env file in the root directory and add:
GROQ_API_KEY=your_groq_api_key_here

# 4. Run the app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## Usage

1. Launch the app and upload a PDF using the file uploader.
2. Wait for the app to read, chunk, and index the document (a progress spinner will show status).
3. View the auto-generated **Document Understanding** section for a quick overview.
4. Click **Generate Summary** for a full document summary.
5. Type any question into the **Ask anything** box and get an instant, context-aware answer.

---

## Project Structure

```
pdf-ai-app/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── runtime.txt         # Python runtime version
├── .devcontainer/      # Dev container configuration
└── README.md
```

---

## Future Improvements

- Support for multiple PDFs in a single session (multi-document Q&A)
- Persistent vector storage across sessions instead of in-memory FAISS
- Source citation — highlighting which page/section an answer came from
- User authentication and per-user usage limits
- Export chat history and summaries as a downloadable file

---

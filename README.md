# 📄 PDF Question Answering with RAG

A **Retrieval-Augmented Generation (RAG)** application that allows users to upload a PDF file and ask questions about its content using **LangChain**, **Groq LLaMA**, and **Streamlit**.

---

## 🚀 Features

- Upload any PDF file
- Ask questions based only on the PDF content
- Uses semantic search with vector embeddings
- Powered by LLaMA 3 via Groq API
- Simple and interactive Streamlit UI

---

## 🧠 How It Works (RAG Pipeline)

1. PDF is uploaded by the user
2. Text is extracted from the PDF
3. Text is split into chunks
4. Chunks are converted into embeddings
5. Embeddings are stored in a vector database (Chroma)
6. User question retrieves relevant chunks
7. LLM generates an answer based on retrieved context

---

## 🛠 Tech Stack

- Python
- Streamlit
- LangChain
- Chroma Vector Database
- FastEmbed Embeddings
- Groq API (LLaMA 3)
- PyPDF



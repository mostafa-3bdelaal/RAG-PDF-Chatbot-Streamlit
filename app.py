
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
import os

st.set_page_config(page_title="PDF Question Answering", layout="wide")
st.title("📄 PDF Question Answering with RAG")

# --- Upload PDF ---
pdf_file = st.file_uploader("Upload your PDF file", type="pdf")
    

if pdf_file is not None:
    # حفظ الملف مؤقتاً
    with open("temp.pdf", "wb") as f:
        f.write(pdf_file.read())
    
    st.success("PDF uploaded successfully!")
    
    # Load PDF
    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()
    
    # Split PDF
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = splitter.split_documents(docs)
    
    # Embedding
    embedding = FastEmbedEmbeddings()
    vectorstore = Chroma.from_documents(splits, embedding=embedding)
    retriever = vectorstore.as_retriever()
    
    # Initialize LLM
    os.environ["GROQ_API_KEY"] = "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"
    llm = ChatOpenAI(
        model="llama-3.3-70b-versatile",
        openai_api_base="https://api.groq.com/openai/v1",
        openai_api_key=os.environ["GROQ_API_KEY"],
        temperature=0.7,
        max_tokens=512
    )
    
    # Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
Use the following context to answer the question.
If you don't know the answer, just say "I don't know."

Context:
{context}
        """),
        ("human", "{input}"),
    ])
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    # --- Question Input ---
    query = st.text_input("Ask a question about your PDF:")
    
    if st.button("Get Answer") and query:
        response = retrieval_chain.invoke({"input": query})
        st.markdown("**Answer:**")
        st.write(response["answer"])




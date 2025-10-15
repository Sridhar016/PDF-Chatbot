import os
import streamlit as st
from dotenv import load_dotenv  # For loading .env file
from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq  # Groq API integration
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document

# Load environment variables from .env file
load_dotenv()

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def create_faiss_vector_store(text, path="faiss_index"):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    documents = [Document(page_content=chunk) for chunk in chunks]
    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(documents, embedding=embeddings)
    vector_store.save_local(path)

def load_faiss_vectore_store(path="faiss_index"):
    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    return vector_store

def build_qa_chain(vector_store_path="faiss_index"):
    vector_store = load_faiss_vectore_store(vector_store_path)
    retriever = vector_store.as_retriever()

    # Load Groq API key from .env
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("Groq API key not found in .env file.")

    # Use a currently supported model (e.g., 'llama-3.1-8b-instant')
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

    qa_chain = load_qa_chain(llm, chain_type="stuff")
    qa_chain = RetrievalQA(retriever=retriever, combine_documents_chain=qa_chain)
    return qa_chain






# Streamlit UI
st.header("Chat with PDFs")
with st.sidebar:
    st.title("Menu:")
    uploaded_file = st.file_uploader("Upload your PDF file and click Submit & Process Button", type="pdf")
    if st.button("Submit & Process"):
        if uploaded_file:
            os.makedirs("uploaded", exist_ok=True)
            pdf_path = os.path.join("uploaded", uploaded_file.name)
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            with st.spinner("Extracting text from PDF..."):
                text = extract_text_from_pdf(pdf_path)
            if not text.strip():
                st.error("Failed to extract text from PDF. Try a different file.")
            else:
                with st.spinner("Creating FAISS vector store..."):
                    create_faiss_vector_store(text)
                st.info("Initializing chatbot...")
                qa_chain = build_qa_chain()
                st.session_state['qa_chain'] = qa_chain
                st.success("Chatbot is ready!")

if 'qa_chain' in st.session_state:
    question = st.text_input("Ask a question about the uploaded PDF:")
    if question:
        with st.spinner("Querying the document..."):
            answer = st.session_state.qa_chain.run(question)
        st.success(f"Answer: {answer}")
else:
    st.info("Upload and process a PDF to start chatting.")

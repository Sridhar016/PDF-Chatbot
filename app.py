import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

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
    groq_api_key = st.secrets["GROQ_API_KEY"]
    if not groq_api_key:
        raise ValueError("Groq API key not found in Streamlit secrets.")
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

    # Define the prompt template
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Create the chain using LCEL
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Streamlit UI
st.header("Chat with PDFs")
with st.sidebar:
    st.title("Menu:")
    uploaded_files = st.file_uploader(
        "Upload your PDF files and click Submit & Process Button",
        type="pdf",
        accept_multiple_files=True
    )
    if st.button("Submit & Process"):
        if uploaded_files:
            os.makedirs("uploaded", exist_ok=True)
            combined_text = ""
            for uploaded_file in uploaded_files:
                pdf_path = os.path.join("uploaded", uploaded_file.name)
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                with st.spinner(f"Extracting text from {uploaded_file.name}..."):
                    text = extract_text_from_pdf(pdf_path)
                    combined_text += text + "\n\n"
            if not combined_text.strip():
                st.error("Failed to extract text from PDFs. Try different files.")
            else:
                with st.spinner("Creating FAISS vector store..."):
                    create_faiss_vector_store(combined_text)
                st.info("Initializing chatbot...")
                qa_chain = build_qa_chain()
                st.session_state['qa_chain'] = qa_chain
                st.success("Chatbot is ready!")

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if 'qa_chain' in st.session_state:
    if prompt := st.chat_input("Ask a question about the uploaded PDFs:"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        # Get answer from QA chain
        with st.spinner("Querying the documents..."):
            answer = st.session_state.qa_chain.invoke(prompt)
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(answer)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("Upload and process PDFs to start chatting.")

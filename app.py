import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(page_title="RAG System", page_icon="🤖")

st.title("LLM-Powered RAG System")
st.write("Upload a text file to ask questions based on its content.")

# Sidebar for API Key input (optional, if not in .env)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")

if not api_key:
    st.warning("Please provide an OpenAI API Key to proceed.")
    st.stop()

# File uploader
uploaded_file = st.file_uploader("Upload a txt file", type=["txt"])

if uploaded_file is not None:
    # Read the file
    text = uploaded_file.read().decode("utf-8")
    st.write("File uploaded successfully!")

    # Split text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(text)
    
    # Create documents
    documents = [Document(page_content=t) for t in texts]

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    try:
        vectorstore = FAISS.from_documents(documents, embeddings)
        st.success("Embeddings created successfully!")
    except Exception as e:
        st.error(f"Error creating embeddings: {e}")
        st.stop()

    # Create retrieval chain
    llm = ChatOpenAI(temperature=0, openai_api_key=api_key)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

    # User query input
    query = st.text_input("Ask a question about the document:")

    if query:
        with st.spinner("Thinking..."):
            try:
                response = qa_chain.run(query)
                st.write("### Answer:")
                st.write(response)
            except Exception as e:
                st.error(f"Error generating answer: {e}")

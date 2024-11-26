import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Initialize Groq API Key and Embedding Model
groq_api_key = "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"
embeddings = HuggingFaceEmbeddings()

# Set Streamlit Page Configuration (Title and Icon)
st.set_page_config(page_title="RAG Chat", page_icon="💬", layout="centered")

# Add a custom title and description
st.title("RAG Chat - Ask Anything from the Web")
st.markdown(
    """
    <style>
        .big-font {
            font-size: 40px !important;
            color: #4CAF50;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)
st.markdown('<p class="big-font">Enter a website URL to begin.</p>', unsafe_allow_html=True)

# Add a sidebar for a user-friendly interface
st.sidebar.header("RAG Chat - Setup")
st.sidebar.markdown("Welcome! Please enter the URL of the website you want to query.")

# Input: Website URL
url_input = st.sidebar.text_input("Enter Website URL:")

# Decorative elements (Optional: Background color or image)
st.markdown("""
    <style>
        body {
            background-color: #f0f4f8;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            font-weight: bold;
            border-radius: 12px;
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Only proceed with chat if a valid URL is entered
if url_input:
    st.write("Loading content from the website...")

    # Load and split documents from the provided URL
    loader = WebBaseLoader(url_input)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs[:50])

    # Create FAISS vector store
    vectors = FAISS.from_documents(final_documents, embeddings)

    # Initialize Groq LLM
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")

    # Create prompt template
    prompt_template = ChatPromptTemplate.from_template("""
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    </context>
    Question: {input}
    """)

    # Create document chain and retrieval chain
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    retriever = vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Interactive query input field
    st.markdown("### Ask your question below:")
    user_prompt = st.text_input("Enter your question:")

    if user_prompt:
        st.write("Processing your query...")

        try:
            start_time = time.process_time()
            response = retrieval_chain.invoke({"input": user_prompt})
            elapsed_time = time.process_time() - start_time

            # Display response time and answer only
            st.write(f"Response time: {elapsed_time:.2f} seconds")
            st.write("\n*Answer:*")
            st.write(response['answer'])

        except Exception as e:
            st.write(f"Error querying model: {e}")

else:
    st.sidebar.warning("Please enter a valid URL to start the chat.")

# Optionally, add an "Exit" button
if st.sidebar.button('Exit'):
    st.write("Exiting...")
    st.stop()

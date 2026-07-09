import os
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# --- 1. CONFIGURATION & PAGE SETUP ---
st.set_page_config(page_title="Star Health Insurance Assistant", page_icon="🏥", layout="centered")
st.title("🏥 Star Health Insurance Chatbot")
st.write("Ask questions about different health insurance plans and benefits.")

# Secure your API key (Uses streamlit secrets or local environment variable)
# For local testing, you can paste your key here, but using Streamlit Secrets is recommended for deployment.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "Your API Key")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Path to your local HTML documentation file
FILE_PATH = "Types of Health Insurance Plans.html" 

# --- 2. CACHED RAG PIPELINE INITIALIZATION ---
# We use st.cache_resource so the data is only loaded and indexed ONCE when the app starts.
@st.cache_resource
def initialize_rag_chain(file_path):
    if not os.path.exists(file_path):
        st.error(f"Required file not found: `{file_path}`. Please ensure it is in the same directory as app.py.")
        st.stop()
        
    # Load document
    loader = UnstructuredHTMLLoader(file_path=file_path)
    machine_docs = loader.load()
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(machine_docs)
    
    # Generate embeddings and build vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    
    # Define models and prompt templates
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_template(
        "You are an assistant for question-answering tasks.\n"
        "Use the following pieces of retrieved context to answer the question.\n"
        "If you don't know the answer, just say that you don't know.\n"
        "Use three sentences maximum and keep the answer concise.\n\n"
        "Question: {question} \n"
        "Context: {context} \n"
        "Answer:"
    )
    
    # Create the LCEL RAG Chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    return chain

# Setup the chain
with st.spinner("Indexing documentation... Please wait."):
    rag_chain = initialize_rag_chain(FILE_PATH)

# --- 3. CHAT SESSION STATE MANAGEMENT ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your Star Health assistant. How can I help you today?"}
    ]

# Render historical messages from session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# --- 4. USER CHAT INTERACTION ---
if user_query := st.chat_input("Type your question here..."):
    # Display user message
    with st.chat_message("user"):
        st.write(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Generate response from RAG Chain
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = rag_chain.invoke(user_query)
                answer_text = response.content
                st.write(answer_text)
                st.session_state.messages.append({"role": "assistant", "content": answer_text})
            except Exception as e:
                st.error(f"An error occurred: {e}")
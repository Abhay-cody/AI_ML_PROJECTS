import os
import requests
import streamlit as st
from bs4 import BeautifulSoup

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)

# ---------------------------
# Streamlit Page Config
# ---------------------------
st.set_page_config(
    page_title="Health Insurance RAG Chatbot",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Health Insurance RAG Chatbot")
st.write("Ask questions about Health Insurance Policies.")

# ---------------------------
# API Key
# ---------------------------
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    api_key = st.sidebar.text_input(
        "Enter your Gemini API Key",
        type="password"
    )

if not api_key:
    st.warning("Please provide your Gemini API Key.")
    st.stop()

# ---------------------------
# Website URL
# ---------------------------
URL = "https://www.starhealth.in/health-insurance/types-of-health-insurance/"


# ---------------------------
# Load Website
# ---------------------------
@st.cache_data(show_spinner=True)
def load_website():

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(URL, headers=headers, timeout=30)

    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    text = soup.get_text(separator="\n")

    return text


# ---------------------------
# Build Vector Store
# ---------------------------
@st.cache_resource(show_spinner=True)
def create_vector_store(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    documents = splitter.create_documents([text])

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )

    db = FAISS.from_documents(documents, embeddings)

    return db


try:

    website_text = load_website()

    db = create_vector_store(website_text)

except Exception as e:
    st.error(f"Error loading website:\n\n{e}")
    st.stop()

# ---------------------------
# User Question
# ---------------------------
question = st.text_input("Ask your question")

if question:

    with st.spinner("Searching..."):

        docs = db.similarity_search(question, k=4)

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        prompt = f"""
You are a Health Insurance assistant.

Answer ONLY from the context below.

If the answer is unavailable, say:
"I couldn't find that information in the provided document."

Context:
{context}

Question:
{question}
"""

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.2
        )

        response = llm.invoke(prompt)

        st.subheader("Answer")

        st.write(response.content)

st.set_page_config(
    page_title="Health Insurance RAG Chatbot",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Health Insurance RAG Chatbot")
st.write("Ask questions about Health Insurance Policies.")

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.title("👨‍💻 Developer")

st.sidebar.markdown(
    """
### 🔗 Connect with Me

📂 **GitHub Repository**

https://github.com/Abhay-cody/AI_ML_PROJECTS/tree/main/Project_10_RAG_Chatbots_for_Health_Insurance

💼 **LinkedIn**

https://www.linkedin.com/in/abhay-kumar-gupta-104a18397
"""
)

st.sidebar.divider()

st.sidebar.link_button(
    "📂 View GitHub Repository",
    "https://github.com/Abhay-cody/AI_ML_PROJECTS/tree/main/Project_10_RAG_Chatbots_for_Health_Insurance"
)

st.sidebar.link_button(
    "💼 Connect on LinkedIn",
    "https://www.linkedin.com/in/abhay-kumar-gupta-104a18397"
)

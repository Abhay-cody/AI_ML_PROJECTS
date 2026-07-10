import streamlit as st
import requests
from bs4 import BeautifulSoup

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain

st.set_page_config(page_title="Health Insurance RAG")

st.title("🏥 Health Insurance Chatbot")

api_key = st.sidebar.text_input("Google API Key", type="password")

URL = "https://www.starhealth.in/health-insurance/types-of-health-insurance/"


@st.cache_resource
def build_vectorstore(api_key):

    response = requests.get(URL)

    soup = BeautifulSoup(response.text, "html.parser")

    text = soup.get_text(separator="\n")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.create_documents([text])

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )

    db = FAISS.from_documents(docs, embeddings)

    return db


if api_key:

    db = build_vectorstore(api_key)

    question = st.text_input("Ask a question")

    if question:

        docs = db.similarity_search(question)

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.2
        )

        chain = load_qa_chain(llm, chain_type="stuff")

        response = chain.run(
            input_documents=docs,
            question=question
        )

        st.write(response)

else:
    st.info("Enter your Google API Key.")

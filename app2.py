from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_core.pydantic_v1 import BaseModel, Field
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Dict
import streamlit as st
import cassio
import json
import os
import docx

# Load environment variables from .env file
load_dotenv(dotenv_path='../.env')

# Cache the initialization of the database, LLM, and embeddings
@st.cache_resource
def initialize_resources():
    # Fetch environment variables
    ASTRA_DB_APPLICATION_TOKEN = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
    ASTRA_DB_ID = os.getenv('ASTRA_DB_ID')
    groq_api_key = os.getenv('GROQ_API_KEY')
    os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

    # Initialize the database, LLM, and embeddings
    cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)
    llm = ChatGroq(model="llama-3.1-70b-versatile", groq_api_key=groq_api_key)
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Connect to the Cassandra table where vectors are stored
    astra_vector_store = Cassandra(
        embedding=embedding,
        table_name="qa_mini_demo_1024",
        session=None,
        keyspace=None,
    )

    astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)
    
    retrieval_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="map_reduce",  # or use "map_reduce" if the dataset is large
    retriever=astra_vector_store.as_retriever(),
)
    
    return astra_vector_index, llm, embedding, retrieval_chain, astra_vector_store

# Call the function to initialize resources
astra_vector_index, llm, embedding, retrieval_chain, astra_vector_store = initialize_resources()


# Initialize the first question flag
first_question = True

# Streamlit app interface
st.title("Document Query App")

# Input widget for user query
query_text = st.text_input("Enter your question (or type 'quit' to exit):")

# Check if user entered a query
if query_text.lower() == "quit":
    st.stop()

# If there's a query text, process it
if query_text:
    first_question = False
    
    st.write(f"**QUESTION:** \"{query_text}\"")
    
    # Querying the AstraDB vector index and displaying the answer
    answer = astra_vector_index.query(query_text, llm=llm).strip()
    st.write(f"**ANSWER:** \"{answer}\"")
    
    # Display the first documents by relevance
    st.write("**FIRST DOCUMENTS BY RELEVANCE:**")
    docs_scores = astra_vector_store.similarity_search_with_score(query_text, k=4)
    
    for doc, score in docs_scores:
        st.write(f"[{score:.4f}] \"{doc.page_content[:84]} ...\"")

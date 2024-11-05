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

class TechnicalSpecification(BaseModel):
    specifications: List[Dict[str, str]] = Field(description="List of parameter-specification pairs")

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
    
    return astra_vector_index, llm, embedding, retrieval_chain

# Call the function to initialize resources
astra_vector_index, llm, embedding, retrieval_chain = initialize_resources()

# Define the JsonOutputParser and PromptTemplate
parser = JsonOutputParser(pydantic_object=TechnicalSpecification)

prompt_template = PromptTemplate(
    template="You are an AI assistant specialized in analyzing and answering questions about automotive headlamp assembly. You have access to the full SOR document and can reference specific sections when answering questions.\n{query}\nProject Title: {project_title}\nProject Overview: {project_overview}\nProject Background: {project_background}\nPlease provide a detailed and comprehensive response, including references to specific sections of the document. Make sure the output is thorough and covers all aspects of the project requirements.\n{format_instructions}",
    input_variables=["query", "project_title", "project_overview", "project_background"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

def main():
    st.title("Technical Specification Generator")

    # User input for query and project details
    query = st.text_input("Enter your query", value="What are the headlamp specifications?")
    project_title = st.text_input("Project Title", value="Automotive Headlamp Assembly")
    project_overview = st.text_area("Project Overview", value="Overview of the headlamp assembly project.")
    project_background = st.text_area("Project Background", value="Background details about the automotive headlamp assembly project.")

    # If the user clicks the button, generate the technical specification
    if st.button("Generate Technical Specification"):
        formatted_prompt = prompt_template.format(
            query=query,
            project_title=project_title,
            project_overview=project_overview,
            project_background=project_background
        )
        
        response = retrieval_chain.run(formatted_prompt)
        try:
            # Parse the response as JSON
            technical_spec = json.loads(response) if isinstance(response, str) else response
            
            # Display the technical specification
            st.write("### Technical Specification")
            if isinstance(technical_spec, dict):
                st.write(f"**{technical_spec['parameter']}**: {technical_spec['specification']}")
            elif isinstance(technical_spec, list):
                for spec in technical_spec:
                    st.write(f"**{spec['parameter']}**: {spec['specification']}")
        except json.JSONDecodeError:
            st.error("Could not parse the response as JSON. Raw response:")
            st.write(response)
        except KeyError as e:
            st.error(f"Missing expected key in response: {e}")
            st.write("Raw response:", response)
        
if __name__ == "__main__":
    main()
# query: Using the following project title, project overview and background, create a Statement of Requirements or SoR of 4 pages long
# title: Development and Supply of Automotive Headlamp Assembly
# overview: We're designing and making advanced headlamps for cars and trucks. They'll use LED or laser lights. The lamps must meet safety rules and look good while improving visibility.
# background: We're launching a new vehicle with premium headlamps. They need to work well in all conditions and match the car's style. The lamps will connect to the car's electronics and ADAS systems.
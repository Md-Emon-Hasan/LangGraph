import streamlit as st
import cassio
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Literal
from langchain.vectorstores.cassandra import Cassandra
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langgraph.graph import StateGraph, START, END
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings

# Load environment variables from the .env file
load_dotenv()

# Fetch the environment variables for the API keys and tokens
# These values are securely stored in the .env file
groq_api_key = os.getenv('GROQ_API_KEY')  # Your Groq API key for interacting with the Groq LLM
ASTRA_DB_APPLICATION_TOKEN = os.getenv('ASTRA_DB_APPLICATION_TOKEN')  # Token for Astra DB access
ASTRA_DB_ID = os.getenv('ASTRA_DB_ID')  # Astra DB ID to connect to your database

# Initialize Cassio for Astra DB connection using the provided token and DB ID
cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

# Initialize embedding model using HuggingFace's 'all-MiniLM-L6-v2' model for text embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Set up a connection to the Cassandra vector store using the embeddings
# The 'qa_mini_demo' table is assumed to be used for storing and querying embeddings
astra_vector_store = Cassandra(
    embedding=embeddings, table_name="qa_mini_demo", session=None, keyspace=None
)

# Create a retriever from the vector store
retriever = astra_vector_store.as_retriever()

# Set up a Wikipedia search tool with some settings for the maximum number of results and content length
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

# Routing logic: This will determine whether to fetch data from the vector store or Wikipedia based on the question type
class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "wiki_search"] = Field(...)  # Define two possible data sources

# Initialize the Groq model using the provided API key and model name
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")

# Enable structured output for routing based on the question
structured_llm_router = llm.with_structured_output(RouteQuery)

# Define a system prompt for routing: it directs which data source to use based on the question content
system_prompt = """You are an expert router. Use the vectorstore for topics related to agents, prompt engineering, and adversarial attacks. Otherwise, use wiki-search."""

# Create a template for the prompt, combining the system and human parts (question from the user)
route_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}"),
])

# Create a question router that will use the Groq model to decide whether to use the vectorstore or wiki-search
question_router = route_prompt | structured_llm_router

# Function to retrieve documents from the vector store based on the question
def retrieve(state):
    question = state["question"]  # Extract the question from the state
    documents = retriever.invoke(question)  # Fetch relevant documents from the vector store
    return {"documents": documents, "question": question}  # Return the results

# Function to perform a Wikipedia search based on the question
def wiki_search(state):
    question = state["question"]  # Extract the question from the state
    docs = wiki.invoke({"query": question})  # Perform the Wikipedia search
    wiki_results = Document(page_content=docs)  # Wrap the result in a Document object
    return {"documents": wiki_results, "question": question}  # Return the results

# Function to route the question to either the vector store or Wikipedia search based on the answer from the model
def route_question(state):
    question = state["question"]  # Extract the question from the state
    source = question_router.invoke({"question": question})  # Use the router to decide the data source
    return "wiki_search" if source.datasource == "wiki_search" else "vectorstore"  # Return the appropriate route

# Define the stateful workflow with nodes for both Wikipedia search and vectorstore retrieval
workflow = StateGraph(dict)

# Add the nodes (tasks) to the workflow
workflow.add_node("wiki_search", wiki_search)  # Node for Wikipedia search
workflow.add_node("retrieve", retrieve)  # Node for vector store retrieval

# Define conditional edges based on the routing decision
workflow.add_conditional_edges(START, route_question, {"wiki_search": "wiki_search", "vectorstore": "retrieve"})

# Add the edges for ending the workflow after retrieval or Wikipedia search
workflow.add_edge("retrieve", END)
workflow.add_edge("wiki_search", END)

# Compile the workflow into an executable form
app = workflow.compile()

# Streamlit UI
st.title("RAG and Wikipedia Search App")  # Set the title for the web app
user_query = st.text_input("Enter your question:")  # Input field for the user to type their question

# When the "Search" button is clicked
if st.button("Search"):
    if user_query:  # Ensure the user has entered a question
        inputs = {"question": user_query}  # Create an input dictionary for the workflow
        for output in app.stream(inputs):  # Stream the workflow output
            for key, value in output.items():  # Loop through each key and value in the output
                st.subheader(f"Node '{key}':")  # Display the name of the node (task)
                st.write(value["documents"])  # Display the retrieved documents (results)
    else:
        st.warning("Please enter a question.")  # Display a warning if no question is entered
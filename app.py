# app.py
import os
from dotenv import load_dotenv
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools.retriever import create_retriever_tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv()

# --- RAG PIPELINE ---
def get_retriever():
    """
    Creates and returns a retriever for the cognitive bias documents.
    This function encapsulates the entire RAG setup process.
    """
    # 1. Load Documents
    loader = DirectoryLoader('./data', glob="**/*.txt")
    documents = loader.load()

    # 2. Split Documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    # 3. Create Embeddings & Store in ChromaDB
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Persist directory is set to "db" to ensure embeddings are saved locally.
    # It will save the embeddings in the 'db' directory.
    vector_store = Chroma.from_documents(texts, embeddings, persist_directory="db")
    
    # 4. Create and return the retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 1}) # Retrieve top 2 chunks
    return retriever

# --- PHASE 3: AGENT LOGIC ---

def create_agent_executor():
    """Creates the complete agent logic."""
    # 1. Initialize the LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    
    # 2. Get the Retriever from Phase 2
    retriever = get_retriever()

    # 3. Create the Retriever Tool
    # The tool description is CRITICAL. It tells the agent WHEN to use this tool.
    retriever_tool = create_retriever_tool(
        retriever,
        "cognitive_bias_retriever",
        "Searches for definitions and coping strategies for cognitive biases in behavioral finance. Use this to find factual information."
    )
    tools = [retriever_tool]

    # 4. Create the Agent Prompt
    # This prompt guides the agent's reasoning process.
    prompt_template = """You are a helpful and empathetic Behavioral Finance Coach. 
    Your goal is to help users identify their cognitive biases and provide actionable advice based on proven strategies.

    Follow these steps:
    1.  Listen to the user's statement and identify the primary cognitive bias they are exhibiting.
    2.  Use the 'cognitive_bias_retriever' tool to get the definition and coping strategies for that specific bias.
    3.  Synthesize the retrieved information with your empathetic understanding to provide a supportive and guiding response.
    
    Rely on the information from the tool. Combine it with empathy to help the user manage their bias effectively. Complement with your own knowledge only if necessary.

    {chat_history}
    User input: {input}
    
    {agent_scratchpad}
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # 5. Create the Agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True) # Set verbose=True for debugging
    
    return agent_executor

# --- PHASE 4: STREAMLIT UI ---

# App title
st.set_page_config(page_title=" Finance Cognitive RAG Agent")
st.title("Behavioral Finance Coaching Agent")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How are you feeling about your finances today?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Use st.cache_resource to initialize the agent executor only once
@st.cache_resource
def get_agent():
    return create_agent_executor()

agent_executor = get_agent()

# User input
if prompt := st.chat_input("Describe your financial situation or feelings..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Prepare chat history for the agent
            chat_history = [
                HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"])
                for msg in st.session_state.messages[:-1] # Exclude the current user prompt for this turn
            ]

            response = agent_executor.invoke({
                "input": prompt,
                "chat_history": chat_history
            })
            
            response_text = response["output"]
            st.write(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
-----

# 🧠 Cognitive RAG Agent: A Behavioral Finance Coach

[](https://www.python.org/downloads/release/python-390/)
[](https://www.langchain.com/)
[](https://streamlit.io/)

A proof-of-concept AI agent developed for an Applied NLP course. This project uses a **Retrieval-Augmented Generation (RAG)** architecture to function as a behavioral finance coach. It identifies users' potential cognitive biases based on their stated financial anxieties and provides empathetic, evidence-based coping strategies.

## ✨ Project Description

In moments of financial stress, investors often make irrational decisions driven by cognitive biases like **Loss Aversion** or **Anchoring**. This project explores how Large Language Models (LLMs) can be used to build a supportive tool that helps users navigate these challenges.

The agent listens to a user's problem, uses an LLM to diagnose a likely cognitive bias, retrieves factual information and coping strategies from a specialized knowledge base, and synthesizes this information into a helpful, guiding response.

The core of this project is to demonstrate a modern, agentic RAG pipeline using state-of-the-art NLP tools.

-----

## 🛠️ Core Technologies & Architecture

This project is built on a modern NLP stack, demonstrating key concepts in applied AI:

  * **Large Language Models (LLM):** Google's **Gemini 1.5 Flash** is used as the "brain" or orchestrator of the agent.
  * **Retrieval-Augmented Generation (RAG):** The agent's knowledge is not just in its pre-trained weights. It retrieves relevant, up-to-date information from a custom knowledge base to ground its responses in facts.
  * **Embeddings:** Text from our knowledge base is converted into numerical vectors. We use local, open-source embeddings (`HuggingFaceInstructEmbeddings`) to ensure privacy and avoid API rate limits.
  * **Vector Store:** **ChromaDB** is used as a lightweight, local vector database to store and efficiently query the embeddings.
  * **Agent Framework:** **LangChain** orchestrates the entire process, defining the agent, its tools (like the RAG retriever), and the logic flow.
  * **User Interface:** **Streamlit** provides a simple, fast, and interactive web interface for the user to chat with the agent.

### ⚙️ Logical Flow

The agent operates in a clear, multi-step process:

1.  **User Input:** The user describes their financial anxiety in a simple chat interface.
2.  **Bias Identification:** The primary LLM (the Agent) analyzes the user's text to hypothesize the dominant cognitive bias at play.
3.  **RAG Tool Execution:** The agent autonomously decides to use its `cognitive_bias_retriever` tool. This tool queries the ChromaDB vector store to find the most relevant documents (definitions and coping strategies) related to the identified bias.
4.  **Synthesis & Response:** The agent receives the factual data from the RAG tool. It then synthesizes the user's emotional state, the identified bias, and the retrieved knowledge to generate an empathetic, factual, and guiding response.

-----

## 🚀 Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites

  * Python 3.9 or higher
  * A Google API Key for the Gemini model. You can get one from [Google AI Studio](https://aistudio.google.com/app/apikey).

### 1\. Clone the Repository

```bash
git clone https://github.com/jherronr/cognitive-rag-agent.git
cd cognitive-rag-agent
```

### 2\. Set Up a Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies.

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3\. Install Dependencies

Install all the required Python packages.

```bash
pip install -r requirements.txt
```

### 4\. Configure Environment Variables

Create a file named `.env` in the root of the project directory. This file will hold your secret API key.

```
GOOGLE_API_KEY="YOUR_API_KEY_HERE"
```

### 5\. Add Your Knowledge Base

The agent's knowledge comes from the text files in the `/data` directory. You can add or edit `.txt` files here. For the best results:

  * Create one `.txt` file per cognitive bias.
  * In each file, include a clear **definition** of the bias and a list of actionable **coping strategies**. The agent is specifically prompted to look for this information.

-----

## ▶️ How to Run the Application

Once the setup is complete, you can start the Streamlit web server with a single command:

```bash
streamlit run app.py
```

Your web browser should automatically open to the application's chat interface. The first time you run it, it may take a few minutes to download the local embedding model (`instructor-large`).

### 💡 Example Interaction

**User:** "My stocks are down 20% and I'm panicking. I feel like selling everything just to stop the bleeding\!"

**Cognitive RAG Agent:** "It sounds like you're going through a really stressful time, and it's completely understandable to feel that way when your portfolio is down. That intense urge to sell to avoid further losses is a classic sign of a cognitive bias called **Loss Aversion**.

From what I can access, Loss Aversion is the principle where the pain of losing is psychologically about twice as powerful as the pleasure of gaining. This can lead to making reactive decisions based on emotion rather than a long-term strategy.

Here are a couple of proven strategies to help manage this feeling:

  * **Reframe Your Perspective:** Instead of viewing a 20% drop as a 'loss,' try to see it as the market being 'on sale.' This can shift your mindset from panic to opportunity, assuming your long-term belief in the investments hasn't changed.
  * **Review Your Original Plan:** Go back to the reasons you invested in the first place. Was it for long-term growth? If your original thesis is still valid, acting on short-term fear might go against your own goals." 

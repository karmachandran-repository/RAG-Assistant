import os
import glob
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from vectordb import VectorDB
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()


def load_documents() -> List[Dict[str, Any]]:
    """
    Loads documents out of the data directory automatically parsing file data.

    Returns:
        List of dictionaries containing document contents and operational paths
    """
    results = []
    
    # This finds the directory of app.py, goes up one level, and targets 'data'
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "..", "data")
    
    if not os.path.exists(data_dir):
        print(f"[*] Creating target context data directory: {data_dir}")
        os.makedirs(data_dir)
        return results

    # Track structural txt, md, and log files in the folder path
    supported_extensions = ["*.txt", "*.md", "*.rst"]
    found_files = []
    for ext in supported_extensions:
        found_files.extend(glob.glob(os.path.join(data_dir, ext)))

    for file_path in found_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                if content.strip():
                    results.append({
                        "content": content,
                        "metadata": {"source": os.path.basename(file_path)}
                    })
        except Exception as e:
            print(f"[!] Warning: Could not read file {file_path}. Error: {e}")

    return results


class RAGAssistant:
    """
    A simple RAG-based AI assistant using ChromaDB and multiple LLM providers.
    Supports OpenAI, Groq, and Google Gemini APIs.
    """

    def __init__(self):
        """Initialize the RAG assistant."""
        # Initialize LLM - check for available API keys in order of preference
        self.llm = self._initialize_llm()
        if not self.llm:
            raise ValueError(
                "No valid API key found. Please set one of: "
                "OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY in your .env file"
            )

        # Initialize vector database
        self.vector_db = VectorDB()

        # Create RAG prompt template to enforce grounding rules
        self.prompt_template = ChatPromptTemplate.from_template("""
You are a strict, focused technical assistant. Your task is to answer the user's question using ONLY the provided context below.

CRITICAL RULES:
1. Rely only on the clear facts directly mentioned in the context.
2. If the context does not contain the answer to the question, or if the question is out of scope for these documents, you must reply exactly with: "I am sorry, but I do not have enough specific information in my local knowledge base to answer that query."
3. Do not use your own external pre-trained knowledge to answer general questions (such as general science, broad programming definitions, or unrelated topics).

Context:
{context}

Question: 
{question}

Answer:
""")

        # Create the execution chain
        self.chain = self.prompt_template | self.llm | StrOutputParser()

        print("RAG Assistant initialized successfully")

    def _initialize_llm(self):
        """
        Initialize the LLM by checking for available API keys.
        Tries OpenAI, Groq, and Google Gemini in that order.
        """
        if os.getenv("OPENAI_API_KEY"):
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            print(f"Using OpenAI model: {model_name}")
            return ChatOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"), model=model_name, temperature=0.0
            )

        elif os.getenv("GROQ_API_KEY"):
            model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            print(f"Using Groq model: {model_name}")
            return ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"), model=model_name, temperature=0.0
            )

        elif os.getenv("GOOGLE_API_KEY"):
            model_name = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
            print(f"Using Google Gemini model: {model_name}")
            return ChatGoogleGenerativeAI(
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                model=model_name,
                temperature=0.0,
            )

        else:
            return None

    def add_documents(self, documents: List) -> None:
        """
        Add documents to the knowledge base.

        Args:
            documents: List of documents
        """
        self.vector_db.add_documents(documents)

    def invoke(self, input_query: str, n_results: int = 3) -> str:
        """
        Query the RAG assistant pipeline.

        Args:
            input_query: User's input
            n_results: Number of relevant chunks to retrieve

        Returns:
            String response content output generated from LLM Chain
        """
        # Step 1: Query vector store for related text segments
        search_results = self.vector_db.search(query=input_query, n_results=n_results)
        retrieved_docs = search_results.get("documents", [])

        # Step 2: Combine context blocks dynamically
        if retrieved_docs:
            context_string = "\n\n".join(
                [f"[Document Source Chunk {i+1}]: {doc}" for i, doc in enumerate(retrieved_docs)]
            )
        else:
            context_string = "No matching contextual documents located inside knowledge bases."

        # Step 3: Map input objects cleanly through standard sequence stream execution invoke
        llm_answer = self.chain.invoke({
            "context": context_string,
            "question": input_query
        })

        return llm_answer


def main():
    """Main function to demonstrate the RAG assistant."""
    try:
        # Initialize the RAG assistant
        print("Initializing RAG Assistant...")
        assistant = RAGAssistant()

        # Load sample documents
        print("\nLoading documents from ./data folder...")
        sample_docs = load_documents()
        print(f"Loaded {len(sample_docs)} sample source document data files.")

        if sample_docs:
            # OPTIMIZATION: Prevent duplicate database scaling bloat between sessions
            existing_count = assistant.vector_db.collection.count()
            if existing_count > 0:
                print(f"[*] Vector database already contains {existing_count} chunks. Skipping re-ingestion phase.")
            else:
                print("[*] Vector store empty. Beginning extraction and ingestion...")
                # FIXED: Restored original structural direct structure tracking passing
                assistant.add_documents(sample_docs)
        else:
            print("[*] Notice: No initial text file materials found in data folder directory. "
                  "Please populate './data' with .txt or .md files.")

        done = False
        print("\n" + "="*50)
        print(" Interactive RAG Session Initialized. Ready for inquiries.")
        print("="*50 + "\n")

        while not done:
            question = input("Enter a question or 'quit' to exit: ").strip()
            if not question:
                continue
            if question.lower() == "quit":
                done = True
            else:
                result = assistant.invoke(question)
                print(f"\nAnswer:\n{result}\n" + "-"*50 + "\n")

    except Exception as e:
        print(f"Error running RAG assistant: {e}")
        print("Make sure you have set up your .env file with at least one API key:")
        print("- OPENAI_API_KEY (OpenAI GPT models)")
        print("- GROQ_API_KEY (Groq Llama models)")
        print("- GOOGLE_API_KEY (Google Gemini models)")


if __name__ == "__main__":
    main()
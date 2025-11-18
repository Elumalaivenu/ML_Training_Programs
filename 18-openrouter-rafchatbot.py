# rag_chatbot_openrouter.py
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from openai import OpenAI

# Load environment variables
load_dotenv()

# Get API key from environment
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    print("❌ Error: OPENROUTER_API_KEY not found in environment variables.")
    print("Please create a .env file and add: OPENROUTER_API_KEY=your_api_key_here")
    exit(1)

# 1️⃣ Initialize OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

# 2️⃣ Sample knowledge base
docs = [
    Document(page_content="RAG stands for Retrieval-Augmented Generation."),
    Document(page_content="LangChain helps build LLM-powered applications."),
    Document(page_content="OpenRouter provides unified access to many open-source LLMs."),
    Document(page_content="FAISS is a vector store library for similarity search."),
]

# 3️⃣ Split and embed documents
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
split_docs = splitter.split_documents(docs)

try:
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=OPENROUTER_API_KEY
    )
    
    vector_db = FAISS.from_documents(split_docs, embedding_model)
    print("✅ Vector database initialized successfully!")
    
except Exception as e:
    print(f"❌ Error initializing embeddings: {str(e)}")
    print("💡 Make sure your OpenRouter API key supports embeddings or use a different embedding model.")
    exit(1)

# 4️⃣ RAG query
def rag_query(question):
    # Retrieve top similar chunks
    results = vector_db.similarity_search(question, k=2)
    context = "\n".join([r.page_content for r in results])

    # Prompt
    prompt = f"""
    You are a helpful AI assistant. 
    Use the following context to answer the user's question.

    Context:
    {context}

    Question: {question}
    Answer:
    """

    # 5️⃣ Generate response using OpenRouter LLM (e.g. Mistral or Llama3)
    response = client.chat.completions.create(
        model="mistralai/mistral-7b-instruct",  # you can change to 'meta-llama/llama-3-8b-instruct'
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()

# 6️⃣ Run chatbot
if __name__ == "__main__":
    print("🧠 RAG Chatbot Ready! Type 'exit' to quit.")
    print("💡 Ask questions about RAG, LangChain, OpenRouter, or FAISS")
    print("-" * 50)
    
    while True:
        try:
            q = input("\nYou: ")
            if q.lower() in ["exit", "quit", "q"]:
                print("👋 Goodbye!")
                break
            
            if not q.strip():
                continue
                
            ans = rag_query(q)
            print(f"Bot: {ans}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            print("💡 Please try again or type 'exit' to quit.")

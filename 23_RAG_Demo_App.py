# ------------------------------------------------------------
# Streamlit App: RAG vs Normal LLM Comparison
# ------------------------------------------------------------
# Demonstrates how Retrieval-Augmented Generation (RAG)
# improves factual accuracy using a local knowledge base.
# Uses: LangChain + FAISS + Sentence Transformers + OpenRouter
# ------------------------------------------------------------

import streamlit as st
import numpy as np
import requests
import json
import os
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(page_title="🧠 RAG vs Normal LLM Demo", layout="wide")
st.title("🤖 Retrieval-Augmented Generation (RAG) Demo")
st.markdown("""
This demo compares how a **Normal LLM** and a **RAG-enhanced LLM** respond to the same query.  
RAG retrieves facts from a local **knowledge base**, improving accuracy and grounding responses.
""")

# -----------------------------
# Knowledge Base (Example Docs)
# -----------------------------
documents = [
    "The Eiffel Tower is located in Paris, France. It was built in 1889.",
    "The Great Wall of China was built to protect against invasions and spans over 13,000 miles.",
    "Mount Everest is the highest mountain in the world, standing at 8,848 meters.",
    "The Taj Mahal in India was built by Mughal Emperor Shah Jahan in memory of his wife Mumtaz Mahal.",
    "The Pyramids of Giza are one of the Seven Wonders of the Ancient World located in Egypt."
]

# -----------------------------
# Create Vector Index using TF-IDF
# -----------------------------
@st.cache_resource
def create_vector_index(docs):
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    doc_vectors = vectorizer.fit_transform(docs)
    return vectorizer, doc_vectors

vectorizer, doc_vectors = create_vector_index(documents)

# -----------------------------
# OpenRouter API Function
# -----------------------------
def query_openrouter(prompt, api_key, model="meta-llama/llama-3-8b-instruct", temperature=0.7):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": temperature}
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, data=json.dumps(payload))
    result = response.json()
    if "choices" in result:
        return result["choices"][0]["message"]["content"].strip()
    return f"❌ Error: {result}"

# -----------------------------
# RAG Query Function using TF-IDF
# -----------------------------
def rag_query(user_query, api_key, top_k=2):
    # Transform the user query using the same vectorizer
    query_vector = vectorizer.transform([user_query])
    
    # Calculate cosine similarity between query and documents
    similarities = cosine_similarity(query_vector, doc_vectors)[0]
    
    # Get top-k most similar documents
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    retrieved_docs = [documents[i] for i in top_indices]

    context = "\n".join(retrieved_docs)
    prompt = f"""
    You are a helpful assistant. Use the following context to answer accurately.

    Context:
    {context}

    Question:
    {user_query}

    Answer:
    """
    return query_openrouter(prompt, api_key)

# -----------------------------
# Streamlit Inputs
# -----------------------------
st.sidebar.header("⚙️ Settings")

# Get API key from environment file
api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    st.sidebar.error("❌ OpenRouter API key not found!")
    st.sidebar.markdown("""
    **Setup Instructions:**
    1. Create a `.env` file
    2. Add: `OPENROUTER_API_KEY=your_key`
    3. Restart the app
    """)
    st.stop()
else:
    st.sidebar.success("✅ API Key loaded from config")
model = st.sidebar.selectbox(
    "Select Model:",
    ["meta-llama/llama-3-8b-instruct", "mistralai/mistral-7b-instruct-v0.1", "meta-llama/llama-3-70b-instruct"],
    index=0
)

st.divider()
user_query = st.text_input("💬 Enter your question:", "Where is the Eiffel Tower located?")

if st.button("🚀 Generate Responses"):
    col1, col2 = st.columns(2)

    # Normal LLM
    with col1:
        st.subheader("🤖 Normal LLM Response (No Retrieval)")
        normal_prompt = f"Answer this question as best as you can: {user_query}"
        with st.spinner("Generating..."):
            normal_answer = query_openrouter(normal_prompt, api_key, model)
        st.write(normal_answer)

    # RAG
    with col2:
        st.subheader("📚 RAG-Enhanced Response (With Retrieval)")
        with st.spinner("Retrieving and generating..."):
            rag_answer = rag_query(user_query, api_key)
        st.write(rag_answer)

    st.markdown("---")
    st.markdown("### 📄 Retrieved Documents (for RAG)")
    for doc in documents:
        st.markdown(f"- {doc}")

# -----------------------------
# Info Section
# -----------------------------
st.markdown("---")
st.markdown("""
### 🧠 How RAG Works
1. **Vectorize** local documents using TF-IDF (Term Frequency-Inverse Document Frequency).  
2. **Retrieve** top relevant documents for a query using cosine similarity.  
3. **Augment** the LLM prompt with these facts.  
4. **Generate** an answer that's factual and grounded.

✅ **RAG Advantage:** Reduces hallucinations, ensures context-aware accuracy.  
❌ **Normal LLM:** May produce creative but unreliable responses.
""")

st.caption("Built with ❤️ using Streamlit + TF-IDF + OpenRouter • For Training Demos")

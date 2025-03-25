import streamlit as st
import google.generativeai as genai
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# Configure Gemini API
genai.configure(api_key="AIzaSyCM3Uz8OXw8a-5_ruVqLCKq9X2-05Ve-I8")

# Load vector database and embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_db = Chroma(persist_directory="./new_chroma_db", embedding_function=embedding_model)

# Function to retrieve relevant chunks
def retrieve_relevant_chunks(query):
    query_embedding = embedding_model.embed_query(query)
    return vector_db.similarity_search_by_vector(query_embedding, k=5)

# Generate response using Gemini
def generate_response(query):
    retrieved_chunks = retrieve_relevant_chunks(query)
    context = "\n".join([chunk.page_content for chunk in retrieved_chunks])

    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    response = model.generate_content(f"Context:\n{context}\n\nQuery: {query}")
    
    return response.text

# Streamlit UI
st.set_page_config(page_title="AI Chatbot", page_icon="💬")
st.title("💬 AI Chatbot with RAG")

user_query = st.text_input("Ask me anything:")
if user_query:
    with st.spinner("Thinking..."):
        bot_response = generate_response(user_query)
    st.write("### 🤖 Chatbot Response")
    st.write(bot_response)

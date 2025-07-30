import streamlit as st
from embedder import load_pdf, embed_texts, build_faiss_index
from rag_pipeline import generate_answer
import openai
import faiss
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

st.title("🔎 RAG: Ask Questions from a PDF")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
question = st.text_input("Ask a question")

if uploaded_file and question:
    # Extract and embed
    texts = load_pdf(uploaded_file)
    from openai.embeddings_utils import get_embedding  # Optional
    embeddings = embed_texts(texts, model=openai.Embedding)
    
    # Build index
    index = build_faiss_index(embeddings)
    
    # Search
    query_embedding = get_embedding(question, engine="text-embedding-ada-002")
    D, I = index.search(np.array([query_embedding]).astype('float32'), k=3)
    relevant_texts = [texts[i] for i in I[0]]
    
    # Generate answer
    answer = generate_answer(question, relevant_texts)
    st.subheader("Answer")
    st.write(answer)

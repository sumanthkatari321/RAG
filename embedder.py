import faiss
import numpy as np
import fitz  # PyMuPDF
from openai import OpenAIEmbeddings

def load_pdf(file_path):
    doc = fitz.open(file_path)
    texts = [page.get_text() for page in doc]
    return texts

def embed_texts(texts, model):
    return [model.embed(text) for text in texts]

def build_faiss_index(embeddings):
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype('float32'))
    return index

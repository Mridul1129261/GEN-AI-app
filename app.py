import streamlit as st
import re
import numpy as np
import requests
import faiss
from tqdm import tqdm
from pypdf import PdfReader

# --- CONFIG ---
deploymentName = "gpt-4.1"
apiVersion = "2024-12-01-preview"
CHAT_MODEL_API = f"https://aicafe.hcl.com/AICafeService/api/v1/subscription/openai/deployments/{deploymentName}/chat/completions?api-version={apiVersion}"
EMBEDDING_MODEL_API = f"https://aicafe.hcl.com/AICafeService/api/v1/subscription/openai/deployments/ada/embeddings?api-version=2023-05-15"

SUBSCRIPTION_KEY = st.secrets["API_KEY"]
# SUBSCRIPTION_KEY = "59f971fd-48fb-48cf-adae-d3e1e584c365"

HEADERS = {
    "Content-Type": "application/json",
    "api-key": SUBSCRIPTION_KEY
}

# --- FUNCTIONS ---

def chunk_text(text, max_words=120, overlap=30):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = []
    word_count = 0
    for sentence in sentences:
        words = sentence.split()
        if word_count + len(words) > max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap:]
            word_count = len(current_chunk)
        current_chunk.extend(words)
        word_count += len(words)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def get_embeddings(texts):
    payload = {"input": texts, "model": "ADA"}
    response = requests.post(EMBEDDING_MODEL_API, headers=HEADERS, json=payload)
    if response.status_code != 200:
        st.error(f"Embedding API error: {response.text}")
        return None
    data = response.json()["data"]
    return [np.array(item["embedding"]) for item in data]

def build_faiss_index(chunks, embeddings):
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    return index

def retrieve(query, index, chunks, top_k=3):
    query_emb = get_embeddings([query])[0]
    distances, indices = index.search(np.array([query_emb]).astype("float32"), top_k)
    return [chunks[i] for i in indices[0]]

def rerank(query, docs):
    scores = []
    query_words = set(query.lower().split())
    for doc in docs:
        overlap = len(query_words.intersection(doc.lower().split()))
        scores.append((overlap, doc))
    scores.sort(reverse=True)
    return [doc for _, doc in scores]

def generate_answer(context, question):
    prompt = f"""
You are a factual AI assistant.

Answer ONLY using the provided context.

Rules:
1. Do not use outside knowledge
2. If answer is missing say:
   "The answer is not available in the provided context."
3. Keep answers concise.

Context:
{context}

Question:
{question}

Answer:
"""
    payload = {
        "model": "gpt-4.1",
        "messages": [{"role": "user", "content": prompt}],
        "maxTokens": 150,
        "temperature": 0
    }
    response = requests.post(CHAT_MODEL_API, headers=HEADERS, json=payload)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        st.error(f"Chat API error: {response.text}")
        return "Error generating answer"

# --- STREAMLIT UI ---

st.title("Advanced RAG QA Demo")
st.write("Upload a TXT or PDF document and ask questions!")

uploaded_file = st.file_uploader("Upload TXT or PDF", type=["txt", "pdf"])

if uploaded_file:
    if uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        document_text = ""
        for page in reader.pages:
            document_text += page.extract_text()
    else:
        document_text = uploaded_file.read().decode()

    st.success("Document loaded!")

    chunks = chunk_text(document_text, max_words=120, overlap=30)
    st.write(f"Created {len(chunks)} chunks for embedding.")

    embeddings = get_embeddings(chunks)
    if not embeddings:
        st.stop()
    
    index = build_faiss_index(chunks, embeddings)
    st.success("Vector index built!")

    question = st.text_input("Ask a question:")

    if question:
        top_chunks = retrieve(question, index, chunks)
        top_chunks = rerank(question, top_chunks)
        context = "\n\n".join(top_chunks)
        answer = generate_answer(context, question)
        
        st.subheader("Answer")
        st.write(answer)

        with st.expander("Retrieved Context"):
            for i, c in enumerate(top_chunks):
                st.markdown(f"**Chunk {i+1}:** {c}")
import streamlit as st
import os
import fitz  
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env file
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
model = os.getenv("MODEL")  # e.g., LLaMA3 70B model
tokenizer_model = os.getenv('TOKENIZER')

# Function to get response from LLM using Groq
def get_response_from_llm(prompt, context):
    # Prompt the model, restricting it to only use the provided context
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are an assistant that provides answers only based on the provided context. Do not generate any information that is not in the context."
            },
            {
                "role": "user",
                "content": f"Context: {context}\n\nQuestion: {prompt}\nAnswer based strictly on the context:"
            }
        ]
    )
    return completion.choices[0].message.content

# Function to extract text from PDF
def extract_text_from_pdf(pdf):
    pdf_text = ""
    pdf_reader = fitz.open(stream=pdf.read(), filetype="pdf")
    for page_num in range(pdf_reader.page_count):
        page = pdf_reader.load_page(page_num)
        pdf_text += page.get_text()
    return pdf_text

# Function to split text into chunks
def split_text(text, max_length=512):
    words = text.split()
    chunks = [' '.join(words[i:i + max_length]) for i in range(0, len(words), max_length)]
    return chunks

# Embed text using a pre-trained sentence transformer model
def embed_text(text_chunks):
    model = SentenceTransformer(tokenizer_model)
    embeddings = model.encode(text_chunks, convert_to_numpy=True)
    return embeddings

# Function to build FAISS index
def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]  # Size of the embedding vector
    index = faiss.IndexFlatL2(dimension)  # L2 distance index
    index.add(embeddings)
    return index

# Function to search FAISS index
def search_faiss_index(index, query_embedding, top_k=5):
    distances, indices = index.search(query_embedding, top_k)
    return indices, distances

# Streamlit app main function
def main():
    st.set_page_config(layout="wide", page_title="RAG Document Search")
    st.sidebar.title("Document Upload")

    uploaded_file = st.sidebar.file_uploader("Upload Document (PDF)", type=["pdf"])

    if "document_text" not in st.session_state:
        st.session_state.document_text = None

    if uploaded_file:
        if st.session_state.document_text is None:
            st.session_state.document_text = extract_text_from_pdf(uploaded_file)
            
            if not st.session_state.document_text.strip():
                st.error("No text found in the uploaded document.")
                return

        st.sidebar.success("Document uploaded & text extracted!")

        text_chunks = split_text(st.session_state.document_text)
        embeddings = embed_text(text_chunks)
        faiss_index = build_faiss_index(embeddings)

        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello, I am RAG Search Bot. Please ask your question!"}
            ]

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        user_input = st.chat_input("Ask a question about the document...")
        
        if user_input:
        
            query_embedding = embed_text([user_input])
            indices, distances = search_faiss_index(faiss_index, query_embedding)
            
    
            relevant_chunks = [text_chunks[i] for i in indices[0]]
            context = ' '.join(relevant_chunks)

            response = get_response_from_llm(user_input, context)

            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.messages.append({"role": "assistant", "content": response})

           
    else:
        st.sidebar.info("Please upload a document to start chatting.")

if __name__ == "__main__":
    main()

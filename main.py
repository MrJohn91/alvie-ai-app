import os
import uuid
import io
import faiss
import json
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pymongo
import boto3
import fitz  # PyMuPDF for PDFs
import requests
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.docstore import InMemoryDocstore

# Load environment variables (useful for local testing)
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Enable CORS Middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# MongoDB Connection (using environment variable)
client = pymongo.MongoClient(os.getenv("MONGO_URL"))
db = client["chat_with_doc"]
conversationcol = db["chat-history"]
feedback_col = db["feedback"]

# AWS S3 Client Initialization
s3 = boto3.client("s3")

# Global FAISS database variable for vector storage
faiss_db = None

# Chat Message Model for API requests
class ChatMessage(BaseModel):
    session_id: str = None
    user_input: str
    data_source: str

# Ollama URL (strictly use environment variable; raise error if not set)
OLLAMA_URL = os.getenv("OLLAMA_URL")
if not OLLAMA_URL:
    raise EnvironmentError("❌ OLLAMA_URL is not set in the environment variables.")

def load_faiss_index():
    """Loads the FAISS index from a file."""
    global faiss_db  # Use the global faiss_db variable to store the index in memory

    try:
        if not os.path.exists("faiss_index.bin"):
            print("❌ FAISS index file not found. Process a PDF first.")
            return False
        
        # Load the FAISS index from the file
        index = faiss.read_index("faiss_index.bin")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Initialize the docstore and index_to_docstore_id with dummy documents (replace as needed)
        documents = [Document(page_content="dummy")]
        docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
        index_to_docstore_id = {str(i): str(i) for i in range(len(documents))}
        
        # Reinitialize the FAISS database with the updated index and embeddings function
        faiss_db = FAISS(
            embedding_function=embeddings.embed_query,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
        )

        print("✅ FAISS index loaded successfully!")
        return True

    except Exception as e:
        print(f"❌ Failed to load FAISS index: {e}")
        return False

@app.post("/chat")
async def chat(chat: ChatMessage):
    """Retrieves answers from FAISS-stored PDF data and generates a response using Ollama."""
    global faiss_db

    # Ensure FAISS index is loaded or return an error if not available
    if not faiss_db:
        if not load_faiss_index():  # Ensure this function is defined and accessible
            return JSONResponse({"error": "FAISS index not loaded. Process a PDF first."}, status_code=500)

    session_id = chat.session_id or str(uuid.uuid4())
    chat_history = conversationcol.find_one({"session_id": session_id}) or {"conversation": []}

    retrieved_docs = faiss_db.similarity_search(chat.user_input, k=5)
    context = "\n".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else "No relevant context found."

    try:
        if not retrieved_docs:
            ai_response = "I don't have enough information to answer that question."
        else:
            data = {
                "model": "llama3",
                "prompt": f"""You are an AI assistant for Psymeon's ALVIE app. 
                Answer questions **only** based on the provided context. If the context doesn't contain enough 
                information to answer accurately, say, "I don't have enough information to answer that question." 
                Use markdown formatting for better readability.

                Context:\n{context}\n\nQuestion: {chat.user_input}""",
                "stream": False,
            }

            response = requests.post(OLLAMA_URL, json=data)
            if response.status_code == 200:
                ai_response = response.json().get("response", "No response generated.")
            else:
                print(f"❌ Ollama API Error: {response.status_code} - {response.text}")
                return JSONResponse({"error": "Failed to generate AI response"}, status_code=500)
    except Exception as e:
        print(f"❌ Ollama API Error: {e}")
        return JSONResponse({"error": "Failed to generate AI response"}, status_code=500)

    conversationcol.update_one(
        {"session_id": session_id}, {"$push": {"conversation": [chat.user_input, ai_response]}}, upsert=True,
    )

    return JSONResponse({"response": ai_response, "session_id": session_id, "chat_history": chat_history["conversation"]})

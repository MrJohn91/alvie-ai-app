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
from langchain.embeddings import HuggingFaceEmbeddings  # Replace OpenAI Embeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.docstore import InMemoryDocstore
import requests  # For making HTTP requests to Ollama API

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Enable CORS Middleware
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# Connect to MongoDB
client = pymongo.MongoClient(os.getenv("MONGO_URL"))
db = client["chat_with_doc"]
conversationcol = db["chat-history"]
feedback_col = db["feedback"]

# AWS S3 Client
s3 = boto3.client("s3")

# Global FAISS database
faiss_db = None

# Chat Message Model
class ChatMessage(BaseModel):
    session_id: str = None
    user_input: str
    data_source: str

# Function to Extract Text from PDF in S3
def extract_text_from_s3(bucket_name, s3_key):
    """Reads a PDF from S3 and extracts text."""
    try:
        obj = s3.get_object(Bucket=bucket_name, Key=s3_key)
        print("✅ Successfully downloaded PDF from S3.")
        doc = fitz.open(stream=io.BytesIO(obj["Body"].read()), filetype="pdf")
        text_data = [page.get_text("text") for page in doc]
        print(f"✅ Extracted {len(text_data)} pages of text from the PDF.")
        return text_data
    except Exception as e:
        print(f"❌ PDF Processing Error: {e}")
        return None

# Function to Store Extracted Text in FAISS
def store_text_in_faiss(text_data):
    """Splits text into chunks, converts them into embeddings, and stores in FAISS."""
    global faiss_db
    try:
        if not text_data:
            print("❌ No text extracted from the PDF. Check S3 and PDF contents.")
            return False

        # Enhanced text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  
            chunk_overlap=200,  
            length_function=len,
            is_separator_regex=False
        )

        if isinstance(text_data, list):
            text_data = "\n\n".join(text_data)

        chunks = text_splitter.create_documents([text_data])
        print(f"✅ Created {len(chunks)} text chunks")

        # Use Sentence Transformer Embeddings (Replaces OpenAI)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Store in FAISS
        faiss_db = FAISS.from_documents(chunks, embeddings)

        # Save FAISS index
        faiss.write_index(faiss_db.index, "faiss_index.bin")

        print("✅ FAISS storage successful!")
        return True

    except Exception as e:
        print(f"❌ FAISS Storage Error: {e}")
        return False

# Function to Load FAISS Index
def load_faiss_index():
    """Loads the FAISS index from a file."""
    global faiss_db
    try:
        if not os.path.exists("faiss_index.bin"):
            print("❌ FAISS index file not found. Process a PDF first.")
            return False
        
        index = faiss.read_index("faiss_index.bin")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        documents = [Document(page_content="dummy")]
        docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
        index_to_docstore_id = {str(i): str(i) for i in range(len(documents))}

        # Initialize FAISS database
        faiss_db = FAISS(
            index=index,
            embedding_function=embeddings.embed_query,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id
        )

        print("✅ FAISS index loaded successfully!")
        return True

    except Exception as e:
        print(f"❌ Failed to load FAISS index: {e}")
        return False

# Lifespan Event Handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Setup and cleanup for the app."""
    print("Starting up the application and loading FAISS index...")
    if not load_faiss_index():
        print("❌ Failed to load FAISS index during startup.")
    yield  
    print("Shutting down the application...")

app = FastAPI(lifespan=lifespan)

# API Endpoint: Store PDF Text in FAISS
@app.post("/process-pdf")
async def process_pdf():
    """Extracts text from the S3 PDF, stores it in FAISS, and rebuilds the index."""
    global faiss_db
    bucket_name = "ai-document-storage"  
    s3_key = "ai_document.pdf"  

    pdf_text = extract_text_from_s3(bucket_name, s3_key)
    if not pdf_text:
        return JSONResponse({"error": "Failed to extract text from PDF"}, status_code=500)

    if not store_text_in_faiss(pdf_text):
        return JSONResponse({"error": "Failed to store text in FAISS"}, status_code=500)
    
    return JSONResponse({"message": "✅ PDF processed and FAISS index updated successfully!"})

# API Endpoint: Chat Using FAISS & Ollama API
@app.post("/chat")
async def chat(chat: ChatMessage):
    """Retrieves answers from FAISS-stored PDF data using Ollama API (Llama3)."""
    global faiss_db
    if not faiss_db:
        if not load_faiss_index():
            return JSONResponse({"error": "FAISS index not loaded. Process a PDF first."}, status_code=500)
    
    session_id = chat.session_id or str(uuid.uuid4())
    chat_history = conversationcol.find_one({"session_id": session_id}) or {"conversation": []}
    
    # Search FAISS for relevant text
    retrieved_docs = faiss_db.similarity_search(chat.user_input, k=5)  
    context = "\n".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else "No relevant context found."

    # Prepare prompt for Ollama
    system_message = (
        "You are an AI assistant for Psymeon's ALVIE app. "
        "Answer questions **only** based on the provided context. "
        "If the context doesn't contain enough information, say, "
        "'I don't have enough information to answer that question.' "
        "Use markdown formatting for readability."
    )
    
    prompt = f"{system_message}\n\nContext:\n{context}\n\nQuestion: {chat.user_input}"

    ollama_url = os.getenv("OLLAMA_URL")
    if not ollama_url:
        return JSONResponse({"error": "Ollama URL not set"}, status_code=500)

    payload = {
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(ollama_url, json=payload)
        response.raise_for_status()
        ai_response = response.json()["response"]
    except requests.RequestException as e:
        return JSONResponse({"error": f"Failed to generate AI response: {str(e)}"}, status_code=500)

    conversationcol.update_one(
        {"session_id": session_id}, {"$push": {"conversation": [chat.user_input, ai_response]}}, upsert=True
    )

    return JSONResponse({"response": ai_response, "session_id": session_id, "chat_history": chat_history["conversation"]})

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
        print("Sample text from the first page:", text_data[0][:500])  # Print first 500 characters
        return text_data
    except Exception as e:
        print(f"❌ PDF Processing Error: {e}")
        return None

# Function to Store Extracted Text in FAISS
def store_text_in_faiss(text_data):
    """Splits text into chunks, converts them into embeddings, and stores in FAISS."""
    global faiss_db
    try:
        # Check if text_data is empty
        if not text_data:
            print("❌ No text extracted from the PDF. Check S3 and PDF contents.")
            return False

        # Print a sample of the extracted text
        print("✅ Extracted text sample:", text_data[:500])  # Print first 500 characters

        # Enhanced text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n# ", "\n## ", "\n### ", "\n\n", "\n", " ", ""],
            chunk_size=1000,  # Increased chunk size for better context
            chunk_overlap=200,  # Increased overlap for better context
            length_function=len,
            is_separator_regex=False
        )

        # Join all pages into one text if it's a list
        if isinstance(text_data, list):
            text_data = "\n\n".join(text_data)

        # Print how many characters we are processing
        print(f"Processing {len(text_data)} characters from the PDF")

        # Create chunks
        chunks = text_splitter.create_documents([text_data])
        print(f"✅ Created {len(chunks)} text chunks")

        # Use HuggingFace embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Print FAISS storage details
        print(f"✅ Storing {len(chunks)} text chunks into FAISS")

        # Create FAISS index
        faiss_db = FAISS.from_documents(chunks, embeddings)

        # Save the FAISS index
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
        
        # Load the FAISS index from the file
        index = faiss.read_index("faiss_index.bin")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Initialize the docstore and index_to_docstore_id
        documents = [Document(page_content="dummy")]  # Dummy document for mock example
        docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
        index_to_docstore_id = {str(i): str(i) for i in range(len(documents))}
        
        # Reinitialize the FAISS database with the updated index
        faiss_db = FAISS(
            embedding_function=embeddings.embed_query,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id
        )

        print("✅ FAISS index loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to load FAISS index: {e}")
        return False

# Lifespan Event handler (new method)
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Setup and cleanup for the app."""
    print("Starting up the application and loading FAISS index...")
    if not load_faiss_index():
        print("❌ Failed to load FAISS index during startup.")
    yield  # When app is running, this is active
    print("Shutting down the application...")

# Add lifespan to the FastAPI app
app = FastAPI(lifespan=lifespan)

# API Endpoint: Store PDF Text in FAISS
@app.post("/process-pdf")
async def process_pdf():
    """Extracts text from the S3 PDF, stores it in FAISS, and rebuilds the index."""
    global faiss_db
    bucket_name = "ai-document-storage"  # S3 bucket name
    s3_key = "ai_document.pdf"  # Updated document path

    # Fetch the document from S3 and extract its text
    pdf_text = extract_text_from_s3(bucket_name, s3_key)
    if not pdf_text:
        return JSONResponse({"error": "Failed to extract text from PDF"}, status_code=500)

    # Rebuild FAISS index with the updated content
    if not store_text_in_faiss(pdf_text):
        return JSONResponse({"error": "Failed to store text in FAISS"}, status_code=500)
    
    # Notify the user that the process was successful
    return JSONResponse({"message": "✅ PDF processed and FAISS index updated successfully!"})

# Ollama URL
OLLAMA_URL = "http://localhost:11434/api/generate"

# API Endpoint: Chat Using FAISS Search
@app.post("/chat")
async def chat(chat: ChatMessage):
    """Retrieves answers from FAISS-stored PDF data."""
    global faiss_db
    if not faiss_db:
        # If FAISS database is empty, load it from the file
        if not load_faiss_index():
            return JSONResponse({"error": "FAISS index not loaded. Process a PDF first."}, status_code=500)
    
    session_id = chat.session_id or str(uuid.uuid4())
    chat_history = conversationcol.find_one({"session_id": session_id}) or {"conversation": []}
    
    # Search FAISS for relevant text
    retrieved_docs = faiss_db.similarity_search(chat.user_input, k=5)  # Increase k to 5 or higher
    context = "\n".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else "No relevant context found."

    # Generate AI Response via Ollama API Call
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
                "stream": False
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

    # Store chat in MongoDB
    conversationcol.update_one(
        {"session_id": session_id}, {"$push": {"conversation": [chat.user_input, ai_response]}}, upsert=True
    )

    return JSONResponse({"response": ai_response, "session_id": session_id, "chat_history": chat_history["conversation"]})

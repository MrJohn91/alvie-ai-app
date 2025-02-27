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
from langchain.embeddings import HuggingFaceEmbeddings  # Replacing OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.docstore import InMemoryDocstore

# Suppress HuggingFace tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Enable CORS Middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# MongoDB connection setup
client = pymongo.MongoClient(os.getenv("MONGO_URL"))
db = client["chat_with_doc"]
conversationcol = db["chat-history"]
feedback_col = db["feedback"]

# AWS S3 Client setup for PDF retrieval
s3 = boto3.client("s3")

# Global FAISS database variable
faiss_db = None

# Chat Message Model definition for API requests
class ChatMessage(BaseModel):
    session_id: str = None
    user_input: str
    data_source: str

# Function to extract text from a PDF stored in S3 bucket
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

# Function to store extracted text in FAISS index using HuggingFace embeddings and save as .bin file.
def store_text_in_faiss(text_data):
    """Splits text into chunks, converts them into embeddings, and stores in FAISS."""
    global faiss_db
    try:
        if not text_data:
            print("❌ No text extracted from the PDF. Check S3 and PDF contents.")
            return False

        # Enhanced text splitter for better chunking and overlap handling.
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n# ", "\n## ", "\n### ", "\n\n", "\n", " ", ""],
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )

        # Join all pages into one text if it's a list.
        if isinstance(text_data, list):
            text_data = "\n\n".join(text_data)

        chunks = text_splitter.create_documents([text_data])
        print(f"✅ Created {len(chunks)} text chunks")

        # Create FAISS index using HuggingFace embeddings object.
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        faiss_db = FAISS.from_documents(chunks, embedding_model)

        # Save only the raw FAISS index as .bin file.
        faiss.write_index(faiss_db.index, "faiss_index.bin")

        # Save index_to_docstore_id mapping to a JSON file.
        with open("index_to_docstore_id.json", "w") as f:
            json.dump(faiss_db.index_to_docstore_id, f)

        print("✅ FAISS storage successful!")
        return True

    except Exception as e:
        print(f"❌ FAISS Storage Error: {e}")
        return False

# Function to load an existing FAISS index from a .bin file using HuggingFace embeddings object.
def load_faiss_index():
    """Loads the FAISS index from a .bin file."""
    global faiss_db
    try:
        if not os.path.exists("faiss_index.bin"):
            print("❌ FAISS index file not found. Process a PDF first.")
            return False

        # Load only the raw FAISS index from .bin file.
        raw_faiss_index = faiss.read_index("faiss_index.bin")

        # Load index_to_docstore_id mapping from JSON file.
        if not os.path.exists("index_to_docstore_id.json"):
            print("❌ Mapping file 'index_to_docstore_id.json' not found.")
            return False

        with open("index_to_docstore_id.json", "r") as f:
            index_to_docstore_id = json.load(f)

        # Recreate LangChain-compatible FAISS object.
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        documents = [Document(page_content="dummy") for _ in range(len(index_to_docstore_id))]
        docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})

        faiss_db = FAISS(
            embedding_function=embedding_model.embed_query,
            index=raw_faiss_index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
        )

        print("✅ FAISS index loaded successfully!")
        return True

    except Exception as e:
        print(f"❌ Failed to load FAISS index: {e}")
        return False

# Lifespan event handler to load FAISS on startup and cleanup on shutdown.
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Setup and cleanup for the app."""
    print("Starting up the application and loading FAISS index...")
    if not load_faiss_index():
        print("❌ Failed to load FAISS index during startup.")
    yield  # When app is running, this is active.
    print("Shutting down the application...")

app = FastAPI(lifespan=lifespan)

# API Endpoint: Process PDF and store its content in FAISS index.
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

# API Endpoint: Chat using Llama 3 via Ollama API with retrieved context from FAISS index.
@app.post("/chat")
async def chat(chat: ChatMessage):
    """Retrieves answers from FAISS-stored PDF data using Llama 3."""
    global faiss_db

    if not faiss_db:
        if not load_faiss_index():
            return JSONResponse({"error": "FAISS index not loaded. Process a PDF first."}, status_code=500)

    session_id = chat.session_id or str(uuid.uuid4())
    chat_history = conversationcol.find_one({"session_id": session_id}) or {"conversation": []}

    # Search FAISS for relevant context.
    retrieved_docs = faiss_db.similarity_search(chat.user_input, k=5)
    context = "\n".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else "No relevant context found."

    try:
        ai_response = f"Simulated response based on context:\n{context}"  # Replace with actual Llama 3 API call.

    except Exception as e:
        print(f"❌ General Error: {e}")
        return JSONResponse({"error": "Failed to generate AI response"}, status_code=500)

    conversationcol.update_one(
        {"session_id": session_id}, {"$push": {"conversation": [chat.user_input, ai_response]}}, upsert=True,
    )

    return JSONResponse({"response": ai_response, "session_id": session_id, "chat_history": chat_history["conversation"]})

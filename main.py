import os
import uuid
import time
import requests
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import pymongo
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
import ollama

# Load environment variables
load_dotenv()

# Set Ollama API URL explicitly to localhost
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
os.environ["OLLAMA_HOST"] = OLLAMA_API_URL
print(f"Initializing with Ollama API URL: {OLLAMA_API_URL}")

# Wait for Ollama to be available before starting FastAPI
def wait_for_ollama(max_retries=30, retry_delay=2):
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt+1}/{max_retries} to connect to Ollama at {OLLAMA_API_URL}...")
            response = requests.get(f"{OLLAMA_API_URL}/api/version", timeout=5)
            if response.status_code == 200:
                print(f"✅ Successfully connected to Ollama: {response.text}")
                return True
        except Exception as e:
            print(f"❌ Connection attempt failed: {e}")
        
        print(f"Retrying in {retry_delay} seconds...")
        time.sleep(retry_delay)
    
    print("⚠️ Failed to connect to Ollama after multiple attempts. Continuing anyway...")
    return False

# Initialize FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up the application...")
    print("Waiting for Ollama service to be available...")
    wait_for_ollama()
    load_pdf_text()
    yield
    print("Shutting down the application...")

app = FastAPI(lifespan=lifespan)

# Enable CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# Connect to MongoDB
client = pymongo.MongoClient(os.getenv("MONGO_URL"))
db = client["chat_with_doc"]
conversationcol = db["chat-history"]
feedback_col = db["feedback"]
metadata_col = db["metadata"]

# Chat Message Model
class ChatMessage(BaseModel):
    session_id: str = Field(default="")  # Default empty string instead of None
    user_input: str
    data_source: str

# Ollama Embeddings 
class OllamaEmbeddings(Embeddings):
    def embed_documents(self, texts):
        embeddings = []
        try:
            for text in texts:
                try:
                    # Try using the ollama client first
                    response = ollama.embeddings(model="llama3", prompt=text)
                    embeddings.append(response["embedding"])
                except Exception as e:
                    print(f"❌ Ollama client error: {e}")
                    # Fall back to direct HTTP request
                    response = requests.post(
                        f"{OLLAMA_API_URL}/api/embeddings",
                        json={"model": "llama3", "prompt": text},
                        timeout=10
                    )
                    if response.status_code == 200:
                        embeddings.append(response.json()["embedding"])
                    else:
                        raise Exception(f"HTTP request failed: {response.status_code}")
            return embeddings
        except Exception as e:
            print(f"❌ Overall embedding error: {e}")
            # Return empty embeddings as fallback
            return [[0.0] * 4096 for _ in texts]

    def embed_query(self, text):
        try:
            # Try using the ollama client first
            try:
                response = ollama.embeddings(model="llama3", prompt=text)
                return response["embedding"]
            except Exception as e:
                print(f"❌ Ollama client error: {e}")
                # Fall back to direct HTTP request
                response = requests.post(
                    f"{OLLAMA_API_URL}/api/embeddings",
                    json={"model": "llama3", "prompt": text},
                    timeout=10
                )
                if response.status_code == 200:
                    return response.json()["embedding"]
                else:
                    raise Exception(f"HTTP request failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Query embedding error: {e}")
            # Return empty embedding as fallback
            return [0.0] * 4096

# Function: Extract Text from PDF
def extract_text_from_pdf(filepath):
    try:
        with fitz.open(filepath) as doc:
            return [page.get_text("text") for page in doc]
    except Exception as e:
        print(f"❌ PDF Processing Error: {e}")
        return None

# Function: Process and Chunk PDF Text
def process_pdf(pdf_path):
    try:
        text_data = extract_text_from_pdf(pdf_path)
        if not text_data:
            print("❌ No text extracted from the PDF.")
            return False

        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n# ", "\n## ", "\n### ", "\n\n", "\n", " ", ""],
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False
        )
        
        # Join all pages into one text
        full_text = "\n\n".join(text_data)
        print(f"Processing {len(full_text)} characters from the PDF")
        
        # Create chunks
        chunks = text_splitter.create_documents([full_text])
        print(f"✅ Created {len(chunks)} text chunks.")

        try:
            embeddings = OllamaEmbeddings()
            chunk_texts = [chunk.page_content for chunk in chunks]
            chunk_embeddings = embeddings.embed_documents(chunk_texts)

            # Clear existing chunks before adding new ones
            db["document_chunks"].delete_many({})
            
            for i, chunk in enumerate(chunks):
                db["document_chunks"].insert_one({
                    "text": chunk.page_content,
                    "embedding": chunk_embeddings[i]
                })
            
            print("✅ Created embeddings and stored in MongoDB!")
        except Exception as e:
            print(f"❌ Embedding error: {e}")
            # Store chunks without embeddings as fallback
            db["document_chunks"].delete_many({})
            for chunk in chunks:
                db["document_chunks"].insert_one({
                    "text": chunk.page_content,
                    "embedding": None
                })
            print("⚠️ Stored text chunks without embeddings due to Ollama error")

        metadata_col.update_one(
            {"file_name": os.path.basename(pdf_path)},
            {"$set": {"last_processed": int(time.time())}},
            upsert=True
        )
        print("✅ Processed PDF and stored in MongoDB!")
        return True

    except Exception as e:
        print(f"❌ PDF Processing Error: {e}")
        return False

# Function: Load PDF Text and Check Metadata
def load_pdf_text():
    pdf_path = os.path.join(os.getcwd(), "metadata", "ai_document.pdf")
    if not os.path.exists(pdf_path):
        print("❌ PDF file not found!")
        return

    file_metadata = metadata_col.find_one({"file_name": os.path.basename(pdf_path)})
    file_mod_time = int(os.path.getmtime(pdf_path))

    if file_metadata:
        last_processed_time = file_metadata.get("last_processed")
        if last_processed_time:
            readable = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(last_processed_time))
            print(f"📄 Last processed at: {readable}")

        if last_processed_time is None or file_mod_time > last_processed_time:
            print("🔁 PDF has been updated or not processed before. Reprocessing...")
            process_pdf(pdf_path)
        else:
            print("✅ PDF is up to date. No processing required.")
    else:
        print("📥 No metadata found. Processing PDF for the first time...")
        process_pdf(pdf_path)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint that verifies connection to Ollama"""
    try:
        response = requests.get(f"{OLLAMA_API_URL}/api/version", timeout=5)
        if response.status_code == 200:
            return {
                "status": "ok",
                "ollama_status": "connected",
                "ollama_url": OLLAMA_API_URL,
                "ollama_version": response.json().get("version")
            }
        else:
            return {
                "status": "warning",
                "ollama_status": "error",
                "details": f"Status code: {response.status_code}"
            }
    except Exception as e:
        return {
            "status": "error",
            "ollama_status": "disconnected",
            "details": str(e)
        }

# API: Chat Endpoint with similarity search
@app.post("/chat")
async def chat(chat: ChatMessage):
    session_id = chat.session_id or str(uuid.uuid4())
    chat_history = conversationcol.find_one({"session_id": session_id}) or {"conversation": []}

    # Get user query embedding
    try:
        embeddings = OllamaEmbeddings()
        query_embedding = embeddings.embed_query(chat.user_input)
    except Exception as e:
        print(f"❌ Query embedding error: {e}")
        return JSONResponse({
            "response": "I'm currently having trouble processing your question. Please try again later.",
            "session_id": session_id,
            "chat_history": chat_history.get("conversation", [])
        }, status_code=500)

    # Get all documents from MongoDB
    chunks = list(db["document_chunks"].find({}))
    
    # If no chunks are found, return an error
    if not chunks:
        ai_response = "I don't have any document data to search through. Please make sure a document has been processed."
        conversationcol.update_one(
            {"session_id": session_id},
            {"$push": {"conversation": [chat.user_input, ai_response]}},
            upsert=True
        )
        return JSONResponse({
            "response": ai_response,
            "session_id": session_id,
            "chat_history": chat_history.get("conversation", [])
        })

    # If embeddings exist, perform similarity search
    if chunks[0].get("embedding") is not None:
        # Compute similarity scores for each chunk
        similarities = []
        for chunk in chunks:
            # Calculate cosine similarity
            chunk_embedding = chunk.get("embedding")
            if chunk_embedding:
                # Calculate dot product
                similarity = sum(a*b for a, b in zip(query_embedding, chunk_embedding))
                # Normalize (approximate cosine similarity)
                similarities.append((chunk, similarity))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Take top 5 most relevant chunks
        relevant_chunks = [item[0] for item in similarities[:5]]
    else:
        # If no embeddings, just use all chunks (not ideal but better than nothing)
        relevant_chunks = chunks[:5]

    # Extract text from relevant chunks
    context = "\n".join([chunk["text"] for chunk in relevant_chunks]) if relevant_chunks else "No relevant context found."

    # Try to generate a response
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt+1}/{max_retries} to get response from Ollama at {OLLAMA_API_URL}")
            
            if not context:
                ai_response = "I don't have enough information to answer that question."
            else:
                # Try using the ollama client first
                try:
                    response = ollama.generate(
                        model="llama3",
                        prompt=f"""You are an AI assistant for Psymeon's ALVIE app. 
                        Answer questions **only** based on the provided context. If the context doesn't contain enough 
                        information to answer accurately, say: "I don't have enough information to answer that question." 
                        Use markdown formatting for readability.

                        Context:\n{context}\n\nQuestion: {chat.user_input}"""
                    )
                    ai_response = response["response"]
                except Exception as e:
                    print(f"❌ Ollama client error: {e}")
                    # Fall back to direct HTTP request
                    response = requests.post(
                        f"{OLLAMA_API_URL}/api/generate",
                        json={
                            "model": "llama3",
                            "prompt": f"""You are an AI assistant for Psymeon's ALVIE app. 
                            Answer questions **only** based on the provided context. If the context doesn't contain enough 
                            information to answer accurately, say: "I don't have enough information to answer that question." 
                            Use markdown formatting for readability.

                            Context:\n{context}\n\nQuestion: {chat.user_input}"""
                        },
                        timeout=30
                    )
                    if response.status_code != 200:
                        raise Exception(f"HTTP request failed: {response.status_code} - {response.text}")
                    
                    ai_response = response.json().get("response", "")
            
            # Store the conversation
            conversationcol.update_one(
                {"session_id": session_id},
                {"$push": {"conversation": [chat.user_input, ai_response]}},
                upsert=True
            )

            return JSONResponse({
                "response": ai_response,
                "session_id": session_id,
                "chat_history": chat_history.get("conversation", [])
            })
            
        except Exception as e:
            print(f"❌ Ollama API Error (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  
    
    # If we reach here, all retries failed
    fallback_response = (
        "I'm currently having trouble connecting to the AI service. "
        "Please try again later or contact support if the issue persists."
    )
    
    # Store the conversation with the fallback response
    conversationcol.update_one(
        {"session_id": session_id},
        {"$push": {"conversation": [chat.user_input, fallback_response]}},
        upsert=True
    )
    
    return JSONResponse({
        "response": fallback_response,
        "session_id": session_id,
        "chat_history": chat_history.get("conversation", []),
        "error": "Failed to connect to Ollama service after multiple attempts"
    }, status_code=500)
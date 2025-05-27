import os, time, uuid
import requests
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import ollama
from models import load_pdf_text, OllamaEmbeddings
from mongodb import conversationcol, chunk_col

load_dotenv()
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
os.environ["OLLAMA_HOST"] = OLLAMA_API_URL

class ChatMessage(BaseModel):
    session_id: str = ""
    user_input: str
    data_source: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    wait_for_ollama()
    load_pdf_text()
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

def wait_for_ollama(max_retries=30, retry_delay=2):
    for attempt in range(max_retries):
        try:
            response = requests.get(f"{OLLAMA_API_URL}/api/version", timeout=5)
            if response.status_code == 200:
                print("Ollama ready.")
                return
        except Exception as e:
            print(f"Ollama wait error: {e}")
        time.sleep(retry_delay)

@app.get("/health")
async def health_check():
    try:
        response = requests.get(f"{OLLAMA_API_URL}/api/version", timeout=5)
        return {"status": "ok", "ollama_status": "connected", "ollama_version": response.json().get("version")}
    except Exception as e:
        return {"status": "error", "ollama_status": "disconnected", "details": str(e)}

@app.post("/chat")
async def chat(chat: ChatMessage):
    session_id = chat.session_id or str(uuid.uuid4())
    chat_history = conversationcol.find_one({"session_id": session_id}) or {"conversation": []}

    try:
        query_embedding = OllamaEmbeddings().embed_query(chat.user_input)
    except:
        return JSONResponse({"response": "Embedding failed", "session_id": session_id}, status_code=500)

    chunks = list(chunk_col.find({}))
    if not chunks:
        response = "No document chunks found."
        conversationcol.update_one({"session_id": session_id}, {"$push": {"conversation": [chat.user_input, response]}}, upsert=True)
        return {"response": response, "session_id": session_id, "chat_history": chat_history["conversation"]}

    if chunks[0].get("embedding"):
        similarities = []
        for chunk in chunks:
            sim = sum(a * b for a, b in zip(query_embedding, chunk["embedding"]))
            similarities.append((chunk, sim))
        similarities.sort(key=lambda x: x[1], reverse=True)
        relevant_chunks = [c[0] for c in similarities[:5]]
    else:
        relevant_chunks = chunks[:5]

    context = "\n".join([c["text"] for c in relevant_chunks])

    for _ in range(3):
        try:
            response = ollama.generate(
                model="llama3",
                prompt=f"""You are an AI assistant for Psymeon's ALVIE app. 
Answer **only** based on the provided context. If the context lacks the answer, reply: "I don't have enough information to answer that question."

Context:
{context}

Question: {chat.user_input}"""
            )
            ai_response = response["response"]
            conversationcol.update_one({"session_id": session_id}, {"$push": {"conversation": [chat.user_input, ai_response]}}, upsert=True)
            return {"response": ai_response, "session_id": session_id, "chat_history": chat_history["conversation"]}
        except Exception as e:
            print(f"Retrying after error: {e}")
            time.sleep(2)

    fallback = "AI service unavailable."
    conversationcol.update_one({"session_id": session_id}, {"$push": {"conversation": [chat.user_input, fallback]}}, upsert=True)
    return JSONResponse({"response": fallback, "session_id": session_id, "chat_history": chat_history["conversation"]}, status_code=500)

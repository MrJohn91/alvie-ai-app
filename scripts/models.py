import os, time
import fitz  # PyMuPDF
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
import ollama
from mongodb import metadata_col, chunk_col

OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")

class OllamaEmbeddings(Embeddings):
    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            try:
                response = ollama.embeddings(model="llama3", prompt=text)
                embeddings.append(response["embedding"])
            except Exception:
                try:
                    response = requests.post(
                        f"{OLLAMA_API_URL}/api/embeddings",
                        json={"model": "llama3", "prompt": text}, timeout=10)
                    embeddings.append(response.json()["embedding"])
                except Exception as e:
                    print(f"Embedding error: {e}")
                    embeddings.append([0.0] * 4096)
        return embeddings

    def embed_query(self, text):
        try:
            response = ollama.embeddings(model="llama3", prompt=text)
            return response["embedding"]
        except Exception:
            try:
                response = requests.post(
                    f"{OLLAMA_API_URL}/api/embeddings",
                    json={"model": "llama3", "prompt": text}, timeout=10)
                return response.json()["embedding"]
            except Exception as e:
                print(f"Query embedding error: {e}")
                return [0.0] * 4096

def extract_text_from_pdf(filepath):
    try:
        with fitz.open(filepath) as doc:
            return [page.get_text("text") for page in doc]
    except Exception as e:
        print(f"PDF read error: {e}")
        return None

def process_pdf(pdf_path):
    text_data = extract_text_from_pdf(pdf_path)
    if not text_data:
        print("No text extracted.")
        return False

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n# ", "\n## ", "\n### ", "\n\n", "\n", " ", ""],
        chunk_size=1000, chunk_overlap=200)
    full_text = "\n\n".join(text_data)
    chunks = text_splitter.create_documents([full_text])
    chunk_texts = [chunk.page_content for chunk in chunks]

    try:
        embeddings = OllamaEmbeddings().embed_documents(chunk_texts)
        chunk_col.delete_many({})
        for i, chunk in enumerate(chunks):
            chunk_col.insert_one({
                "text": chunk.page_content,
                "embedding": embeddings[i]
            })
    except Exception as e:
        print(f"Embedding failure: {e}")
        chunk_col.delete_many({})
        for chunk in chunks:
            chunk_col.insert_one({"text": chunk.page_content, "embedding": None})

    metadata_col.update_one(
        {"file_name": os.path.basename(pdf_path)},
        {"$set": {"last_processed": int(time.time())}}, upsert=True)
    return True

def load_pdf_text():
    pdf_path = os.path.join(os.getcwd(), "metadata", "ai_document.pdf")
    if not os.path.exists(pdf_path):
        print("PDF not found.")
        return

    metadata = metadata_col.find_one({"file_name": os.path.basename(pdf_path)})
    file_mod_time = int(os.path.getmtime(pdf_path))
    last_proc_time = metadata.get("last_processed") if metadata else None

    if not metadata or last_proc_time is None or file_mod_time > last_proc_time:
        process_pdf(pdf_path)
    else:
        print("PDF up-to-date.")

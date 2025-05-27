
# ALVIE AI Agent

This project provides an AI-powered chat interface for Psymeon's ALVIE app. It consists of a **FastAPI backend** that handles document-based Q&A using **Ollama**, **MongoDB**, and **PyMuPDF**, with an optional **Streamlit frontend**.

---

## 🧩 Tech Stack

- **FastAPI** – API backend for AI chat
- **Ollama** – Local LLM (e.g., llama3)
- **MongoDB** – Stores chat history and document chunks
- **PyMuPDF** – Parses PDF documents
- **LangChain + Embeddings** – Handles text vectorization and similarity search
- **Streamlit** *(optional)* – Frontend UI

---

## ⚙️ Prerequisites

- Python 3.8+
- [Ollama installed](https://github.com/jmorganca/ollama) and running locally
- MongoDB (local or remote)
- `pip` for dependency management

---

## 🚀 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/MrJohn91/alvie-ai-app.git
cd alvie-ai-app
````

### 2. Add environment variables

Create a `.env` file in the root:

```
OLLAMA_API_URL=http://localhost:11434
MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start Ollama

```bash
ollama serve
ollama pull llama3
```

### 5. Run the FastAPI backend

```bash
uvicorn main:app --reload
```

* Access the docs at: [http://localhost:8000/docs](http://localhost:8000/docs)
* Health check: [http://localhost:8000/health](http://localhost:8000/health)

---

---

## 🧪 Test the API (optional)

Send a chat request manually:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test_session", "user_input": "What is Psymeon and its main product Alvie?", "data_source": "ai_document.pdf"}'
```

---

## 💬 Run the frontend (optional)

If using the Streamlit frontend:

```bash
streamlit run app.py
```

---

## 📝 Notes

* The system computes vector embeddings for each PDF chunk and stores them in MongoDB.
* User queries are also embedded and compared against document vectors to find relevant context.
* Ollama answers are based only on relevant chunks.

---

##  Next Steps

1. Customize the prompt for business use case.
2. Deploy FastAPI + MongoDB in Docker for production.
3. Secure the API with authentication and origin restrictions.

---

``

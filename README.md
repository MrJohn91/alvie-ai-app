# ALVIE AI Agent

This is a Streamlit-based frontend application that interacts with a FastAPI backend to provide a chat interface powered by AI. The app uses MongoDB for chat history storage, AWS S3 for document processing, and Ollama for generating AI responses.

---

## Prerequisites

- Python 3.8 or higher
- Pip (Python package manager)
- Ollama (for running the AI model locally)

---

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MrJohn91/alvie-ai-app.git
   cd alvie-ai-app
   ```

2. **Place the `.env` file in the root directory of the project.**

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install and run Ollama:**
   - Download and install Ollama from the [official GitHub repository](https://github.com/jmorganca/ollama).
   - Start the Ollama service:
     ```bash
     ollama serve
     ```
   - Pull the required model (e.g., `llama3`):
     ```bash
     ollama pull llama3
     ```

5. **Start the FastAPI backend server:**
   ```bash
   uvicorn main:app --reload
   ```

6. **Process the PDF (optional):**
   - In a new terminal, run the following command to process the PDF:
     ```bash
     curl -X POST http://localhost:8000/process-pdf
     ```

7. **Test the backend API (optional):**
   - You can test the backend API directly using `curl`. For example:
     ```bash
     curl -X POST http://localhost:8000/chat \
          -H "Content-Type: application/json" \
          -d '{"session_id": "test_session", "user_input": "What is Psymeon and its main product Alvie?", "data_source": "ai_document.pdf"}'
     ```

8. **Run the frontend:**
   ```bash
   streamlit run app.py
   ```

---

### How to Use:
1. Copy the updated content into your `README.md` file.
2. Commit and push the changes to your repository.

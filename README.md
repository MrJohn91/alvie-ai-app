
# ALVIE AI Agent

This is a Streamlit-based frontend application that interacts with a FastAPI backend to provide a chat interface powered by AI. The app uses MongoDB for chat history storage and AWS S3 for document processing.

---

## Prerequisites

- Python 3.8 or higher
- Pip (Python package manager)

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

4. **Start the FastAPI backend server:**
   ```bash
   uvicorn main:app --reload
   ```

5. **Process the PDF (optional):**
   - In a new terminal, run the following command to process the PDF:
     ```bash
     curl -X POST http://localhost:8000/process-pdf
     ```

6. **Test the backend API (optional):**
   - You can test the backend API directly using `curl`. For example:
     ```bash
     curl -X POST http://localhost:8000/chat \
          -H "Content-Type: application/json" \
          -d '{"session_id": "test_session", "user_input": "What is Psymeon and its main product Alvie?", "data_source": "ai_document.pdf"}'
     ```

7. **Run the frontend:**
   ```bash
   streamlit run app.py
   ```

---

```

---

### Key Features:
1. **Everything in One File**: No separation or extra sections.
2. **Includes PDF Processing**: Added the `curl` command to process the PDF.
3. **Includes API Testing**: Added the `curl` command to test the `/chat` endpoint.
4. **Clear Instructions**: Follows your exact formatting and structure.

---

### How to Use:
1. Copy the above content into a new file named `README.md` in the root of your repository.
2. Commit and push the changes to your repository.

---

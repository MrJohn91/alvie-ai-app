import streamlit as st
import os
import uuid
import pymongo
import faiss
import requests
import datetime
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from openai import OpenAI

# ✅ Load API Key from Streamlit Secrets
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ✅ MongoDB Setup
try:
    client = pymongo.MongoClient(st.secrets["MONGO_URL"], tls=True, tlsAllowInvalidCertificates=True)
    db = client["chat_with_doc"]
    conversationcol = db["chat-history"]
    feedback_col = db["feedback"]
except pymongo.errors.ServerSelectionTimeoutError:
    st.error("❌ Could not connect to MongoDB.")

# ✅ FAISS Database
faiss_db = None

# ✅ Session Handling (Ensure session attributes exist)
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ✅ Function to Download FAISS Index from GitHub
def download_faiss_index():
    """Downloads FAISS index from GitHub if not available locally."""
    github_url = "https://raw.githubusercontent.com/MrJohn91/alvie-ai-app/main/faiss_index.bin"
    if not os.path.exists("faiss_index.bin"):
        try:
            response = requests.get(github_url)
            if response.status_code == 200:
                with open("faiss_index.bin", "wb") as file:
                    file.write(response.content)
                return True
            else:
                return False
        except Exception as e:
            st.error(f"Failed to download FAISS index: {e}")
            return False
    return True

# ✅ Function to Load FAISS Index
def load_faiss_index():
    """Loads the FAISS index from a file after downloading from GitHub."""
    global faiss_db

    # Ensure FAISS index is available
    if not os.path.exists("faiss_index.bin"):
        if not download_faiss_index():
            st.error("❌ FAISS index missing. Please check your GitHub repo.")
            return False

    try:
        index = faiss.read_index("faiss_index.bin")
        embeddings = OpenAIEmbeddings()

        # ✅ Fix FAISS Initialization: Ensure the docstore and index_to_docstore_id exist
        docstore = InMemoryDocstore({})
        index_to_docstore_id = {str(i): str(i) for i in range(index.ntotal)}

        faiss_db = FAISS(embedding_function=embeddings, index=index, docstore=docstore, index_to_docstore_id=index_to_docstore_id)

        return True
    except Exception as e:
        st.error(f"❌ FAISS Loading Failed: {e}")
        return False

# ✅ Function to Get AI Response from OpenAI
def get_openai_response(context, user_input):
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Use the provided context to answer the user's question."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {user_input}"}
            ]
        )
        return response.choices[0].message.content
    except:
        return "❌ OpenAI API Error."

# ✅ Streamlit UI
def main():
    st.set_page_config(page_title="ALVIE - Chat Assistant", page_icon="👨‍⚕️", layout="centered")
    st.title("👨‍⚕️ ALVIE - Chat Assistant")

    # ✅ Load FAISS on App Start
    if "faiss_loaded" not in st.session_state:
        st.session_state.faiss_loaded = load_faiss_index()

    # ✅ Chat Interface
    user_input = st.text_input("💬 Talk to ALVIE:", placeholder="Type here...")

    if st.button("Send"):
        if not st.session_state.faiss_loaded:
            st.warning("❌ FAISS index is missing. Please process a PDF first.")
            return

        if user_input:
            with st.spinner("Thinking..."):
                if not faiss_db:
                    st.error("🚨 FAISS is not initialized! Check index file.")
                    return
                
                retrieved_docs = faiss_db.similarity_search(user_input, k=3)
                context = "\n".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else "No relevant context found."
                ai_response = get_openai_response(context, user_input)

                if ai_response:
                    st.session_state.chat_history.append(("You", user_input))
                    st.session_state.chat_history.append(("Alvie", ai_response))

                    conversationcol.update_one(
                        {"session_id": st.session_state.session_id},
                        {"$push": {"conversation": [user_input, ai_response]}},
                        upsert=True
                    )

    # ✅ Display Chat History
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for sender, message in st.session_state.chat_history:
        st.markdown(f"<div class='{'user-message' if sender == 'You' else 'bot-message'}'><strong>{sender}:</strong> {message}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

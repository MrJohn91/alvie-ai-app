import streamlit as st
import os
import uuid
import pymongo
import faiss
import datetime
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from openai import OpenAI  

# ✅ Load API Keys from Streamlit Secrets
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ✅ MongoDB Setup
client = pymongo.MongoClient(st.secrets["MONGO_URL"], tls=True, tlsAllowInvalidCertificates=True)
db = client["chat_with_doc"]
conversationcol = db["chat-history"]
feedback_col = db["feedback"]

# ✅ Global FAISS database
faiss_db = None

# ✅ Ensure FAISS Index is Loaded
def load_faiss_index():
    """Loads FAISS index from the local file."""
    global faiss_db

    if not os.path.exists("faiss_index.bin"):
        st.error("❌ FAISS index file not found. Make sure it exists in the repo.")
        return False

    try:
        index = faiss.read_index("faiss_index.bin")
        embeddings = OpenAIEmbeddings()
        
        # Initialize FAISS with a proper document store
        docstore = InMemoryDocstore({})
        index_to_docstore_id = {str(i): str(i) for i in range(index.ntotal)}
        
        faiss_db = FAISS(embedding_function=embeddings, index=index, docstore=docstore, index_to_docstore_id=index_to_docstore_id)
        return True
    except Exception as e:
        st.error(f"❌ FAISS Loading Failed: {e}")
        return False

# ✅ Get AI Response from OpenAI API
def get_openai_response(context, user_input):
    """Fetches AI response using OpenAI API."""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant using document data."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {user_input}"}
            ]
        )
        return response.choices[0].message.content
    except:
        return "❌ OpenAI API Error."

# ✅ Streamlit UI
def main():
    st.set_page_config(page_title="ALVIE - Chat Assistant", page_icon="🤖", layout="centered")

    st.title("🤖 ALVIE - Chat Assistant")

    # ✅ Load FAISS index once
    if "faiss_loaded" not in st.session_state:
        st.session_state.faiss_loaded = load_faiss_index()

    if not st.session_state.faiss_loaded:
        st.error("❌ FAISS index is not loaded. Please check the index file.")
        return

    # ✅ Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ✅ Chat Interface
    user_input = st.text_input("💬 Ask ALVIE:", placeholder="Type your question here...")

    if st.button("Send"):
        if not st.session_state.faiss_loaded:
            st.warning("⚠️ FAISS index is not available.")
            return

        if user_input:
            with st.spinner("Thinking..."):
                retrieved_docs = faiss_db.similarity_search(user_input, k=5) if faiss_db else []
                context = "\n".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else "No relevant context found."
                
                ai_response = get_openai_response(context, user_input)

                # ✅ Update chat history
                st.session_state.chat_history.append(("You", user_input))
                st.session_state.chat_history.append(("ALVIE", ai_response))

                # ✅ Store conversation in MongoDB
                conversationcol.update_one(
                    {"session_id": st.session_state.session_id},
                    {"$push": {"conversation": [user_input, ai_response]}},
                    upsert=True
                )

    # ✅ Display Chat History
    for sender, message in st.session_state.chat_history:
        st.markdown(f"**{sender}:** {message}")

if __name__ == "__main__":
    main()

import streamlit as st
import os
import uuid
import pymongo
import faiss
import requests
import datetime
import boto3
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from openai import OpenAI

# ✅ Load API Keys from Streamlit Secrets
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ✅ Ensure session state is initialized at the start of the script
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ✅ MongoDB Connection
try:
    client = pymongo.MongoClient(
        st.secrets["MONGO_URL"], tls=True, tlsAllowInvalidCertificates=True, serverSelectionTimeoutMS=10000
    )
    db = client["chat_with_doc"]
    conversationcol = db["chat-history"]
    feedback_col = db["feedback"]
except pymongo.errors.ServerSelectionTimeoutError:
    st.error("❌ Could not connect to MongoDB.")

# ✅ AWS S3 Setup for FAISS Index
s3 = boto3.client(
    "s3",
    aws_access_key_id=st.secrets["AWS_ACCESS_KEY"],
    aws_secret_access_key=st.secrets["AWS_SECRET_KEY"],
    region_name="us-east-1",
)

FAISS_S3_BUCKET = "ai-document-storage"
FAISS_S3_KEY = "faiss_index.bin"

# ✅ Global FAISS database
faiss_db = None

# ✅ Function to Download FAISS Index from S3
def download_faiss_from_s3():
    """Downloads FAISS index from S3 if not available locally."""
    if os.path.exists("faiss_index.bin"):
        return True  # Already downloaded

    try:
        s3.download_file(FAISS_S3_BUCKET, FAISS_S3_KEY, "faiss_index.bin")
        return True
    except Exception as e:
        st.error(f"❌ Failed to download FAISS index from S3: {e}")
        return False

# ✅ Function to Load FAISS Index
def load_faiss_index():
    """Loads FAISS index from a file after downloading from S3."""
    global faiss_db

    if not os.path.exists("faiss_index.bin"):
        if not download_faiss_from_s3():
            return False

    try:
        index = faiss.read_index("faiss_index.bin")
        embeddings = OpenAIEmbeddings()
        docstore = InMemoryDocstore({})
        faiss_db = FAISS(embedding_function=embeddings, index=index, docstore=docstore, index_to_docstore_id={})
        return True
    except Exception as e:
        st.error(f"❌ FAISS Loading Failed: {e}")
        return False

# ✅ Function to Retrieve Relevant Context from FAISS
def get_relevant_context(user_input):
    """Retrieve relevant context from FAISS"""
    if faiss_db:
        retrieved_docs = faiss_db.similarity_search(user_input, k=3)
        return "\n".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else "No relevant context found."
    return "No relevant context found."

# ✅ Function to Get AI Response
def get_openai_response(context, user_input):
    """Fetches AI response using OpenAI API."""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that uses document data to answer questions."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {user_input}"}
            ]
        )
        return response.choices[0].message.content
    except Exception:
        return "❌ OpenAI API Error."

# ✅ Streamlit UI
def main():
    st.set_page_config(page_title="ALVIE - Chat Assistant", page_icon="🤖", layout="centered")

    # ✅ Chat Interface
    user_input = st.text_input("💬 Talk to ALVIE:", placeholder="Type here...")

    if st.button("Send"):
        if user_input:
            with st.spinner("Thinking..."):
                context = get_relevant_context(user_input)
                ai_response = get_openai_response(context, user_input)

                if ai_response:
                    st.session_state.chat_history.append(("You", user_input))
                    st.session_state.chat_history.append(("ALVIE", ai_response))

                    # ✅ Store conversation in MongoDB
                    conversationcol.update_one(
                        {"session_id": st.session_state.session_id},
                        {"$push": {"conversation": [user_input, ai_response]}},
                        upsert=True
                    )

    # ✅ Display chat history
    for sender, message in st.session_state.chat_history:
        st.markdown(f"**{sender}:** {message}")


if __name__ == "__main__":
    main()

import streamlit as st
import os
import uuid
import pymongo
import faiss
import boto3
import datetime
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from openai import OpenAI

# ✅ Load API Keys from Streamlit Secrets
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ✅ Ensure session state is initialized at the start of the script
st.session_state.setdefault("session_id", str(uuid.uuid4()))
st.session_state.setdefault("chat_history", [])

# ✅ MongoDB Connection
client = pymongo.MongoClient(st.secrets["MONGO_URL"])
db = client["chat_with_doc"]
conversationcol = db["chat-history"]
feedback_col = db["feedback"]

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
    if not os.path.exists("faiss_index.bin"):
        s3.download_file(FAISS_S3_BUCKET, FAISS_S3_KEY, "faiss_index.bin")
    return os.path.exists("faiss_index.bin")

# ✅ Function to Load FAISS Index
def load_faiss_index():
    global faiss_db
    if not os.path.exists("faiss_index.bin") and not download_faiss_from_s3():
        return False
    index = faiss.read_index("faiss_index.bin")
    embeddings = OpenAIEmbeddings()
    docstore = InMemoryDocstore({})
    faiss_db = FAISS(embedding_function=embeddings, index=index, docstore=docstore, index_to_docstore_id={})
    return True

# ✅ Function to Retrieve Context from FAISS
def get_relevant_context(user_input):
    if faiss_db:
        retrieved_docs = faiss_db.similarity_search(user_input, k=3)
        return "\n".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else "No relevant context found."
    return "No relevant context found."

# ✅ Function to Get AI Response
def get_openai_response(context, user_input):
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that uses document data to answer questions."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {user_input}"}
        ]
    )
    return response.choices[0].message.content

# ✅ Streamlit UI
def main():
    st.set_page_config(page_title="ALVIE - Chat Assistant", page_icon="👨‍⚕️", layout="centered")

    # ✅ Custom Chat UI Styling
    st.markdown("""
        <style>
            body { background-color: #f8f9fa; }
            .stApp { max-width: 700px; margin: auto; }
            h1 { color: #4CAF50; text-align: center; }

            /* Chat bubbles styling */
            .user-message { 
                background-color: #0084FF;
                color: white; 
                padding: 12px; 
                border-radius: 15px; 
                margin-bottom: 8px; 
                font-size: 16px;
                width: fit-content;
                max-width: 80%;
                text-align: right;
                margin-left: auto;
                box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
            }
            .bot-message { 
                background-color: #E8E8E8;
                color: black;
                padding: 12px; 
                border-radius: 15px; 
                margin-bottom: 8px; 
                font-size: 16px;
                width: fit-content;
                max-width: 80%;
                text-align: left;
                margin-right: auto;
                box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
            }
            .chat-container { margin-top: 20px; }
        </style>
    """, unsafe_allow_html=True)

    st.title("👨‍⚕️ ALVIE - Chat Assistant")
    st.markdown("_Your personal assistant_")

    # ✅ Load FAISS index if not loaded
    st.session_state.setdefault("faiss_loaded", load_faiss_index())

    # ✅ Chat Input
    user_input = st.text_input("💬 Talk to ALVIE:", placeholder="Type here...")

    if st.button("Send") and user_input:
        with st.spinner("Thinking..."):
            context = get_relevant_context(user_input)
            ai_response = get_openai_response(context, user_input)

            # ✅ Store chat history
            st.session_state.chat_history.append(("You", user_input))
            st.session_state.chat_history.append(("ALVIE", ai_response))

            # ✅ Store conversation in MongoDB
            conversationcol.update_one(
                {"session_id": st.session_state.session_id},
                {"$push": {"conversation": [user_input, ai_response]}},
                upsert=True
            )

    # ✅ Display chat history
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for sender, message in st.session_state.chat_history:
        msg_class = "user-message" if sender == "You" else "bot-message"
        st.markdown(f"<div class='{msg_class}'><strong>{sender}:</strong> {message}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ✅ Feedback Rating
    if st.session_state.chat_history:
        st.header("📝 Rate the Response")
        rating = st.radio("How satisfied are you with ALVIE's response?", ["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"])
        if st.button("Submit Rating"):
            feedback_col.insert_one({"session_id": st.session_state.session_id, "rating": rating, "timestamp": datetime.datetime.now()})
            st.success("Thank you for your feedback!")

if __name__ == "__main__":
    main()

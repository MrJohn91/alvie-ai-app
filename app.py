import streamlit as st
import os
import uuid
import pymongo
import faiss  
import requests 
import datetime
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain_community.docstore.in_memory import InMemoryDocstore  
from openai import OpenAI  

#  Load API Keys from Streamlit Secrets
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# MongoDB Connection with Error Handling
try:
    client = pymongo.MongoClient(
        st.secrets["MONGO_URL"], tls=True, tlsAllowInvalidCertificates=True, serverSelectionTimeoutMS=10000
    )
    db = client["chat_with_doc"]
    conversationcol = db["chat-history"]
    feedback_col = db["feedback"]
except pymongo.errors.ServerSelectionTimeoutError:
    st.error("Could not connect to MongoDB.")

# Global FAISS database
faiss_db = None

# Session Handling
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Function to Load FAISS Index
def load_faiss_index():
    global faiss_db
    if not os.path.exists("faiss_index.bin"):
        return False
    try:
        index = faiss.read_index("faiss_index.bin")
        embeddings = OpenAIEmbeddings()
        docstore = InMemoryDocstore({})
        faiss_db = FAISS(embedding_function=embeddings.embed_query, index=index, docstore=docstore, index_to_docstore_id={})
        return True
    except:
        return False

# Function to Get AI Response
def get_openai_response(context, user_input):
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": f"Context: {context}\n\nQuestion: {user_input}"}]
        )
        return response.choices[0].message.content  
    except:
        return "OpenAI API Error."

# Streamlit UI
def main():
    st.set_page_config(page_title="ALVIE - Chat Assistant", page_icon="👨‍⚕️", layout="centered")

    st.title("👨‍⚕️ ALVIE - Chat Assistant")
    st.markdown("_Your personal assistant_")

    # Load FAISS Index
    if "faiss_loaded" not in st.session_state:
        st.session_state.faiss_loaded = load_faiss_index()

    # Show Chat History
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Chat Interface
    user_input = st.text_input("💬 Talk to ALVIE:", placeholder="Type your message here...")

    if st.button("Send"):
        if not st.session_state.faiss_loaded:
            st.warning("❌ Please process a PDF first.")
            return

        if user_input:
            with st.spinner("Thinking..."):
                context = "\n".join([doc.page_content for doc in faiss_db.similarity_search(user_input, k=3)]) if faiss_db else "No relevant context found."
                ai_response = get_openai_response(context, user_input)

                if ai_response:
                    st.session_state.chat_history.append(("You", user_input))
                    st.session_state.chat_history.append(("Alvie", ai_response))

                    # Store conversation in MongoDB
                    conversationcol.update_one(
                        {"session_id": st.session_state.session_id},
                        {"$push": {"conversation": [user_input, ai_response]}},
                        upsert=True
                    )

    # Display Chat History
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for sender, message in st.session_state.chat_history:
        st.markdown(f"<div class='{'user-message' if sender == 'You' else 'bot-message'}'><strong>{sender}:</strong> {message}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # User Rating Feedback
    if st.session_state.chat_history:
        st.header("📝 Rate the Response")
        rating = st.radio("How satisfied are you with ALVIE's response?", ["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"])

        if st.button("Submit Rating"):
            feedback_col.insert_one({"session_id": st.session_state.session_id, "rating": rating, "timestamp": datetime.datetime.now()})
            st.success("Thank you for your feedback!")

if __name__ == "__main__":
    main()

import streamlit as st
import os
import openai
import uuid
import pymongo
import faiss
import ssl  # Required for MongoDB SSL
import datetime
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.docstore import InMemoryDocstore

# Securely Load API Keys from Streamlit Secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# MongoDB Connection with SSL Fix & Error Handling
try:
    client = pymongo.MongoClient(
        st.secrets["MONGO_URL"],  # Load from Streamlit Secrets
        tls=True,  # Use TLS (SSL) for secure connection
        tlsAllowInvalidCertificates=True,  # Bypass SSL certificate verification
        serverSelectionTimeoutMS=50000  # Increase timeout to 50 seconds
    )

    # Select database and collections
    db = client["chat_with_doc"]
    conversationcol = db["chat-history"]
    feedback_col = db["feedback"]
except pymongo.errors.ServerSelectionTimeoutError:
    st.error("❌ Could not connect to MongoDB. Please check your database settings.")

# Global FAISS database
faiss_db = None

# Session Handling (Unique Session ID for Users)
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Load FAISS Index
def load_faiss_index():
    """Loads the FAISS index from a file and initializes FAISS database."""
    global faiss_db
    try:
        if not os.path.exists("faiss_index.bin"):
            st.warning("❌ FAISS index file not found. Please process a PDF first.")
            return False

        # Load FAISS index
        index = faiss.read_index("faiss_index.bin")
        embeddings = OpenAIEmbeddings()

        # Initialize the docstore and index_to_docstore_id
        documents = [Document(page_content="dummy")]  # Dummy document to initialize
        docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
        index_to_docstore_id = {str(i): str(i) for i in range(len(documents))}

        faiss_db = FAISS(
            embedding_function=embeddings.embed_query,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id
        )

        return True
    except Exception as e:
        st.error(f"❌ Failed to load FAISS index: {e}")
        return False

# Function to Get Response
def get_openai_response(context, user_input):
    """Fetches AI response using OpenAI API."""
    try:
        # Generate AI Response via OpenAI API Call
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {user_input}"}
            ]
        )
        ai_response = response.choices[0].message.content  # Extract response
        return ai_response
    except Exception as e:
        st.error(f"❌ OpenAI API Error: {e}")
        return None

# Streamlit UI
def main():
    st.set_page_config(page_title="ALVIE - Chat Assistant", page_icon="👨‍⚕️", layout="centered")

    # Custom Styling for Improved Chat UI
    st.markdown("""
        <style>
            body { background-color: #f8f9fa; }
            .stApp { max-width: 700px; margin: auto; }
            h1 { color: #4CAF50; text-align: center; }

            /* Chat bubbles styling */
            .user-message { 
                background-color: #0084FF;  /* Messenger Blue */
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
                background-color: #E8E8E8;  /* Light Gray */
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

    # Load FAISS Index Automatically
    if "faiss_loaded" not in st.session_state:
        st.session_state.faiss_loaded = load_faiss_index()

    # Show chat history
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
                # Retrieve relevant context from FAISS
                if faiss_db:
                    retrieved_docs = faiss_db.similarity_search(user_input, k=3)
                    st.write(f"Retrieved docs: {retrieved_docs}")  # Debugging output
                    context = "\n".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else "No relevant context found."
                else:
                    context = "No relevant context found."
                st.write(f"Context sent to OpenAI: {context}")  # Debugging output

                ai_response = get_openai_response(context, user_input)

                if ai_response:
                    # Store chat in history
                    st.session_state.chat_history.append(("You", user_input))
                    st.session_state.chat_history.append(("ALVIE", ai_response))

                    # Store conversation in MongoDB
                    conversationcol.update_one(
                        {"session_id": st.session_state.session_id},
                        {"$push": {"conversation": [user_input, ai_response]}},
                        upsert=True
                    )

    # Display chat history
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for sender, message in st.session_state.chat_history:
        if sender == "You":
            st.markdown(f"<div class='user-message'><strong>{sender}:</strong> {message}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-message'><strong>{sender}:</strong> {message}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # User Rating Feedback
    if st.session_state.chat_history:
        st.header("📝 Rate the Response")
        rating = st.radio("How satisfied are you with ALVIE's response?", ["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"])

        if st.button("Submit Rating"):
            feedback_col.insert_one({
                "session_id": st.session_state.session_id,
                "rating": rating,
                "timestamp": datetime.datetime.now()
            })
            st.success("Thank you for your feedback!")

if __name__ == "__main__":
    main()

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

# ✅ Load API Keys from Streamlit Secrets
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ✅ MongoDB Connection with Error Handling
try:
    client = pymongo.MongoClient(
        st.secrets["MONGO_URL"], tls=True, tlsAllowInvalidCertificates=True, serverSelectionTimeoutMS=10000
    )
    db = client["chat_with_doc"]
    conversationcol = db["chat-history"]
    feedback_col = db["feedback"]
except pymongo.errors.ServerSelectionTimeoutError:
    st.error("❌ Could not connect to MongoDB.")

# ✅ Global FAISS database
faiss_db = None

# ✅ Session Handling
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# ✅ Function to Download FAISS Index from GitHub
def download_faiss_index():
    """Downloads FAISS index from GitHub if not available locally."""
    github_url = "https://raw.githubusercontent.com/MrJohn91/alvie-ai-app/main/faiss_index.bin"

    # Check if FAISS file already exists
    if os.path.exists("faiss_index.bin"):
        return True  

    try:
        response = requests.get(github_url)
        if response.status_code == 200:
            with open("faiss_index.bin", "wb") as file:
                file.write(response.content)
            return True
        else:
            st.error("❌ Failed to download FAISS index from GitHub.")
            return False
    except Exception as e:
        st.error(f"❌ Error downloading FAISS index: {e}")
        return False

# ✅ Function to Load FAISS Index
def load_faiss_index():
    """Loads FAISS index from a file after downloading from GitHub."""
    global faiss_db

    # Ensure FAISS index is available
    if not os.path.exists("faiss_index.bin"):
        st.warning("⚠️ FAISS index not found! Downloading from GitHub...")
        if not download_faiss_index():
            st.error("❌ Failed to download FAISS index. Please check GitHub repo.")
            return False

    # Try loading FAISS
    try:
        st.write("Attempting to load FAISS index...")
        index = faiss.read_index("faiss_index.bin")
        embeddings = OpenAIEmbeddings()
        docstore = InMemoryDocstore({})
        faiss_db = FAISS(embedding_function=embeddings.embed_query, index=index, docstore=docstore, index_to_docstore_id={})
        st.success("✅ FAISS index loaded successfully!")
        return True
    except Exception as e:
        st.error(f"❌ FAISS Loading Failed: {e}")
        return False

# ✅ Function to Get AI Response
def get_openai_response(context, user_input):
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {user_input}"}
            ]
        )
        return response.choices[0].message.content  
    except Exception as e:
        st.error(f"❌ OpenAI API Error: {e}")
        return "OpenAI API Error."

# ✅ Streamlit UI
def main():
    st.set_page_config(page_title="ALVIE - Chat Assistant", page_icon="👨‍⚕️", layout="centered")

    # ✅ Custom Styling for Correct Chat UI
    st.markdown("""
        <style>
            .stApp { max-width: 700px; margin: auto; }
            h1 { color: #4CAF50; text-align: center; }

            /* User (blue, right-aligned) */
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

            /* Alvie (white, left-aligned) */
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

    # ✅ Load FAISS Index
    if "faiss_loaded" not in st.session_state:
        st.session_state.faiss_loaded = load_faiss_index()

    # ✅ Show Chat History
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ✅ Chat Interface
    user_input = st.text_input("💬 Talk to ALVIE:", placeholder="Type your message here...")

    if st.button("Send"):
        if not st.session_state.faiss_loaded:
            st.warning("❌ Please process a PDF first.")
            return

        if user_input:
            with st.spinner("Thinking..."):
                # ✅ Ensure FAISS is working before searching
                if not faiss_db:
                    st.error("🚨 FAISS is not initialized! Check index file.")
                    return
                
                # ✅ Retrieve context from FAISS
                retrieved_docs = faiss_db.similarity_search(user_input, k=5)
                if retrieved_docs:
                    context = "\n".join([doc.page_content for doc in retrieved_docs])
                else:
                    context = "No relevant context found."
                    st.warning("⚠️ No relevant FAISS documents found.")

                # ✅ Get AI response
                ai_response = get_openai_response(context, user_input)

                if ai_response:
                    st.session_state.chat_history.append(("You", user_input))
                    st.session_state.chat_history.append(("Alvie", ai_response))

                    # ✅ Store conversation in MongoDB
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

    # ✅ User Rating Feedback
    if st.session_state.chat_history:
        st.header("📝 Rate the Response")
        rating = st.radio("How satisfied are you with ALVIE's response?", ["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"])

        if st.button("Submit Rating"):
            feedback_col.insert_one({"session_id": st.session_state.session_id, "rating": rating, "timestamp": datetime.datetime.now()})
            st.success("Thank you for your feedback!")

if __name__ == "__main__":
    main()

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
from langchain_core.documents import Document  # ✅ Fix Missing Import
from openai import OpenAI

# ✅ Load API Keys
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ✅ MongoDB Connection
try:
    client = pymongo.MongoClient(st.secrets["MONGO_URL"], tls=True, tlsAllowInvalidCertificates=True)
    db = client["chat_with_doc"]
    conversationcol = db["chat-history"]
    feedback_col = db["feedback"]
except pymongo.errors.ServerSelectionTimeoutError:
    st.error("❌ Could not connect to MongoDB.")

# ✅ Session Handling
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# ✅ Function to Download FAISS Index
def download_faiss_index():
    """Downloads FAISS index from GitHub if not available locally."""
    github_url = "https://raw.githubusercontent.com/MrJohn91/alvie-ai-app/main/faiss_index.bin"
    if not os.path.exists("faiss_index.bin"):  
        response = requests.get(github_url)
        if response.status_code == 200:
            with open("faiss_index.bin", "wb") as file:
                file.write(response.content)

# ✅ Function to Load FAISS Index Properly
def load_faiss_index():
    """Ensures FAISS has a proper docstore to avoid KeyError."""
    if "faiss_db" in st.session_state:
        return st.session_state.faiss_db  # ✅ Return FAISS object if already loaded

    if not os.path.exists("faiss_index.bin"):
        download_faiss_index()

    try:
        index = faiss.read_index("faiss_index.bin")
        embeddings = OpenAIEmbeddings()
        
        # ✅ Ensure docstore and index mappings are properly initialized
        documents = [Document(page_content="Placeholder Document")]  
        docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
        index_to_docstore_id = {i: str(i) for i in range(len(documents))}  

        faiss_db = FAISS(
            embedding_function=embeddings.embed_query, 
            index=index, 
            docstore=docstore, 
            index_to_docstore_id=index_to_docstore_id
        )
        st.session_state.faiss_db = faiss_db  # ✅ Store FAISS object properly
        return faiss_db  # ✅ Return FAISS object
    except Exception as e:
        st.error(f"❌ FAISS Loading Failed: {e}")
        return None

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
    except:
        return "OpenAI API Error."

# ✅ Streamlit UI
def main():
    st.set_page_config(page_title="ALVIE - Chat Assistant", page_icon="👨‍⚕️", layout="centered")

    st.title("👨‍⚕️ ALVIE - Chat Assistant")
    st.markdown("_Your personal assistant_")

    # ✅ Load FAISS Index Correctly
    if "faiss_db" not in st.session_state:
        st.session_state.faiss_db = load_faiss_index()

    # ✅ Show chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ✅ Chat Interface
    user_input = st.text_input("💬 Talk to ALVIE:", placeholder="Type your message here...")

    if st.button("Send"):
        faiss_db = st.session_state.faiss_db  # ✅ Get FAISS object

        if not faiss_db:
            st.warning("❌ FAISS is not initialized! Check index file.")
            return

        if user_input:
            with st.spinner("Thinking..."):
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

    # ✅ User Rating Feedback
    if st.session_state.chat_history:
        st.header("📝 Rate the Response")
        rating = st.radio("How satisfied are you with ALVIE's response?", ["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"])

        if st.button("Submit Rating"):
            feedback_col.insert_one({"session_id": st.session_state.session_id, "rating": rating, "timestamp": datetime.datetime.now()})
            st.success("Thank you for your feedback!")

if __name__ == "__main__":
    main()

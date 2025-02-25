import streamlit as st
import os
import uuid
import requests
import datetime
import pymongo
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Backend API URL
BACKEND_URL = "https://alvie-backend.onrender.com"  

# MongoDB setup (for feedback)
try:
    client = pymongo.MongoClient(os.getenv("MONGO_URL"))
    db = client["chat_with_doc"]
    conversationcol = db["chat-history"]
    feedback_col = db["feedback"]
except Exception as e:
    st.error(f"❌ Failed to connect to MongoDB: {e}")

# Initialize session state variables
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

# Function: Call Backend API (with caching)
@st.cache_data(show_spinner=False)
def call_backend_api(endpoint, data=None):
    """Helper function to call the backend API."""
    try:
        url = f"{BACKEND_URL}{endpoint}"
        response = requests.post(url, json=data) if data else requests.post(url)
        response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Failed to call backend API: {e}")
        return None

# Function: Apply Custom Styling
def apply_custom_styling():
    """Add custom CSS styling for the app."""
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

# Function: Process PDF Automatically
def process_pdf():
    """Process the PDF when the app starts."""
    if not st.session_state.pdf_processed:
        response = call_backend_api("/process-pdf")
        if response and "message" in response:
            st.session_state.pdf_processed = True
        else:
            st.error("❌ Failed to process PDF.")

# Function: Display Chat History
def display_chat_history():
    """Render the chat history in the UI."""
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for sender, message in st.session_state.chat_history:
        if sender == "You":
            st.markdown(f"<div class='user-message'><strong>{sender}:</strong> {message}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-message'><strong>{sender}:</strong> {message}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Function: Handle User Input and Chat
def handle_chat(user_input):
    """Send user input to backend and update chat history."""
    with st.spinner("Thinking..."):
        response = call_backend_api("/chat", {
            "user_input": user_input,
            "data_source": "pdf",
            "session_id": st.session_state.session_id
        })

        if response and "response" in response:
            ai_response = response["response"]

            # Extract only the answer part (if applicable)
            if "Question:" in ai_response and "Answer:" in ai_response:
                ai_response = ai_response.split("Answer:", maxsplit=1)[1].strip()

            # Update chat history
            st.session_state.chat_history.append(("You", user_input))
            st.session_state.chat_history.append(("ALVIE", ai_response))

            # Store chat history in MongoDB (optional)
            try:
                conversationcol.update_one(
                    {"session_id": st.session_state.session_id},
                    {"$push": {"conversation": [user_input, ai_response]}},
                    upsert=True
                )
            except Exception as e:
                st.error(f"❌ Failed to save chat history to MongoDB: {e}")
        else:
            st.error("❌ Failed to get a response from the backend.")

# Function: Collect User Feedback
def collect_feedback():
    """Allow users to rate their experience."""
    if st.session_state.chat_history:
        st.header("📝 Rate the Response")
        rating = st.radio("How satisfied are you with ALVIE's response?", ["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"])

        if st.button("Submit Rating"):
            try:
                feedback_col.insert_one({
                    "session_id": st.session_state.session_id,
                    "rating": rating,
                    "timestamp": datetime.datetime.now()
                })
                st.success("Thank you for your feedback!")
            except Exception as e:
                st.error(f"❌ Failed to save feedback to MongoDB: {e}")

# Main Function
def main():
    # Page Configuration and Styling
    apply_custom_styling()
    st.title("👨‍⚕️ ALVIE - Chat Assistant")
    st.markdown("_Your personal assistant_")

    # Process PDF at startup
    process_pdf()

    # Chat Interface
    user_input = st.text_input("💬 Talk to ALVIE:", placeholder="Type here...")
    
    if st.button("Send") and user_input.strip():
        handle_chat(user_input)

    # Display Chat History
    display_chat_history()

    # Collect Feedback
    collect_feedback()

if __name__ == "__main__":
    main()

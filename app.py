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
BACKEND_URL = "https://alvie-backend.onrender.com"  # Replace with your live backend URL

# MongoDB setup (for feedback)
client = pymongo.MongoClient(os.getenv("MONGO_URL"))
db = client["chat_with_doc"]
conversationcol = db["chat-history"]  # Define conversationcol
feedback_col = db["feedback"]

# Session handling
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Function to Call Backend API
def call_backend_api(endpoint, data=None):
    """Helper function to call the backend API."""
    try:
        if data:
            response = requests.post(f"{BACKEND_URL}{endpoint}", json=data)
        else:
            response = requests.post(f"{BACKEND_URL}{endpoint}")
        response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Failed to call backend API: {e}")
        return None

# Streamlit UI
def main():
    st.set_page_config(page_title="ALVIE - Chat Assistant", page_icon="👨‍⚕️", layout="centered")

    # Custom Styling for Chat UI
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

    # Automatically process PDF when the app starts (silently)
    if "pdf_processed" not in st.session_state:
        response = call_backend_api("/process-pdf")
        if response and "message" in response:
            st.session_state.pdf_processed = True
        else:
            st.session_state.pdf_processed = False

    # Show chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Chat Interface
    user_input = st.text_input("💬 Talk to ALVIE:", placeholder="Type here...")

    if st.button("Send"):
        if user_input:
            with st.spinner("Thinking..."):
                # Call the backend /chat endpoint
                response = call_backend_api("/chat", {
                    "user_input": user_input,
                    "data_source": "pdf",
                    "session_id": st.session_state.session_id
                })

                if response and "response" in response:
                    # Remove "Question:" and "Answer:" labels from the response
                    ai_response = response["response"]
                    if "Question:" in ai_response and "Answer:" in ai_response:
                        # Extract only the answer part
                        ai_response = ai_response.split("Answer:")[1].strip()

                    st.session_state.chat_history.append(("You", user_input))
                    st.session_state.chat_history.append(("ALVIE", ai_response))

                    # Store chat history in MongoDB (optional)
                    conversationcol.update_one(
                        {"session_id": st.session_state.session_id},
                        {"$push": {"conversation": [user_input, ai_response]}},
                        upsert=True
                    )
                else:
                    st.error("Failed to get a response from the backend.")

    # Display chat history
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for sender, message in st.session_state.chat_history:
        if sender == "You":
            st.markdown(f"<div class='user-message'><strong>{sender}:</strong> {message}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-message'><strong>{sender}:</strong> {message}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # User Rating Feedback (Stored in MongoDB)
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

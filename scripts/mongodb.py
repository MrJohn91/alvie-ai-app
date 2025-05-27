import os
import pymongo
from dotenv import load_dotenv

load_dotenv()

# MongoDB connection
client = pymongo.MongoClient(os.getenv("MONGO_URL"))
db = client["chat_with_doc"]

# Collections
conversationcol = db["chat-history"]
feedback_col = db["feedback"]
metadata_col = db["metadata"]
chunk_col = db["document_chunks"]

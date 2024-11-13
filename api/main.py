# Imports and Configuration
import os
import re
import httpx
from langchain_community.vectorstores import Pinecone as LangChainPinecone
from langchain_pinecone import PineconeEmbeddings
from pinecone import Pinecone, ServerlessSpec
import pinecone

from dotenv import load_dotenv
from fastapi import FastAPI, Request

# Load environment variables
load_dotenv()

app = FastAPI()

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")
INDEX_NAME = os.getenv("INDEX_NAME")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
PINECONE_HOST = os.getenv("PINECONE_HOST")
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

# Set up Pinecone
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
pc = Pinecone(api_key=PINECONE_API_KEY)

# Pinecone Initialization
def initialize_pinecone():
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=1024,
            metric='euclidean',
            spec=ServerlessSpec(cloud='aws', region=PINECONE_API_ENV)
        )
    return PineconeEmbeddings(model="multilingual-e5-large")

initialize_pinecone()

@app.get("/")
def ret():
    return {"Hello": "World"}

# Telegram Webhook Endpoint
@app.post("/webhook")
async def telegram_webhook(request: Request):
    data = await request.json()
    chat_id = data.get("message", {}).get("chat", {}).get("id")
    message_text = data.get("message", {}).get("text", "")

    if chat_id:
        await send_telegram_message(chat_id, "Hello World!")

    return {"status": "ok"}

# Helper function to send messages to Telegram (asynchronous)
async def send_telegram_message(chat_id, text):
    url = f"{TELEGRAM_API_URL}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload)
        print("Send message response:", response.json())  # For debugging

# Pinecone Query Endpoint
@app.get("/{query}")
async def query_pinecone(query: str):
    embedding = PineconeEmbeddings(model="multilingual-e5-large").embed_query(query)
    index = pinecone.Index(
        index_name=INDEX_NAME,
        host=PINECONE_HOST,
        api_key=PINECONE_API_KEY
    )

    results = index.query(
        vector=embedding,
        top_k=7,
        include_values=False,
        include_metadata=True
    )

    top_chunks = results.get('matches', [])
    response = []
    for i, chunk in enumerate(top_chunks, start=1):
        response.append({
            "rank": i,
            "date": chunk['metadata'].get('date', 'N/A'),
            "title": chunk['metadata'].get('title', 'N/A'),
            "text": chunk['metadata'].get('text', 'N/A')
        })

    return response

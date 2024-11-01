# Imports and Configuration
import os
import re
from langchain.vectorstores import Pinecone 
from langchain_pinecone import PineconeEmbeddings
from pinecone import Pinecone, ServerlessSpec
import pinecone
import requests
from dotenv import load_dotenv

load_dotenv()
import logging
logging.basicConfig(level=logging.DEBUG)

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse

app = FastAPI()

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")
INDEX_NAME = "pinecone"
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
WHATSAPP_ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN")

# Set up Pinecone
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

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

# Webhook verification endpoint
@app.get("/webhook")
async def verify_webhook(request: Request):
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        return PlainTextResponse(content=challenge)  # Return as plain text
    return PlainTextResponse(content="Verification failed", status_code=403)

# Existing receive_message code
@app.post("/webhook")
async def receive_message(request: Request):
    data = await request.json()
    logging.debug("Received data: %s", data)

    try:
        # Verify if the request contains messages
        entry = data.get("entry", [])
        for change in entry[0].get("changes", []):
            message = change.get("value", {}).get("messages", [])[0]
            sender_id = message.get("from")
            logging.debug("Processing message from sender: %s", sender_id)

            if message.get("text", {}).get("body").lower() == "hello":
                await send_message(sender_id, "Hello World!")
        
        return {"status": "received"}
    except Exception as e:
        logging.error("Error processing message: %s", e)
        return {"error": "Internal Server Error"}, 500
    
# Function to send message back to WhatsApp user
async def send_message(recipient_id, text):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}"
    }
    data = {
        "messaging_product": "whatsapp",
        "to": recipient_id,  # Dynamically use recipient_id here
        "text": { "body": text }
    }
    url = f"https://graph.facebook.com/v13.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"  # Use phone number ID from env
    response = requests.post(
        url,
        headers=headers,
        json=data
    )
    response.raise_for_status()

# Your existing Pinecone query endpoint
@app.get("/{query}")
async def query_pinecone(query: str):
    embedding = PineconeEmbeddings(model="multilingual-e5-large").embed_query(query)
    index = pinecone.Index(
        index_name=INDEX_NAME,
        host="https://pinecone-ztbvcbv.svc.aped-4627-b74a.pinecone.io",
        api_key=PINECONE_API_KEY
    )
    results = index.query(
        vector=embedding,
        top_k=7,
        include_values=False,
        include_metadata=True
    )
    response = []
    for i, chunk in enumerate(results.get("matches", []), start=1):
        response.append({
            "rank": i,
            "date": chunk['metadata'].get('date', 'N/A'),
            "title": chunk['metadata'].get('title', 'N/A'),
            "text": chunk['metadata'].get('text', 'N/A')
        })
    return response

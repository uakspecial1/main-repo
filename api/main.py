from fastapi import FastAPI, Request, HTTPException
import httpx
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn
import os
import re
from langchain.vectorstores import Pinecone as LangChainPinecone
from langchain_pinecone import PineconeEmbeddings
from pinecone import Pinecone, ServerlessSpec
import pinecone
from typing import List
from ctransformers import AutoModelForCausalLM
import json

# Load environment variables
load_dotenv()

# Load sensitive data from .env
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")
INDEX_NAME = os.getenv("INDEX_NAME")
PINECONE_HOST = os.getenv("PINECONE_HOST")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

# Initialize FastAPI app
app = FastAPI()

# Global AsyncClient
client = None

# Set up Pinecone
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
pc = Pinecone(api_key=PINECONE_API_KEY)

# Pinecone Initialization
def initialize_pinecone():
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=1024,
            metric="euclidean",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_API_ENV),
        )
    return PineconeEmbeddings(model="multilingual-e5-large")

embeddings = initialize_pinecone()

index = pinecone.Index(
    index_name=INDEX_NAME,
    host=PINECONE_HOST,
    api_key=PINECONE_API_KEY,
)

def initialize_llm():
    """Initialize the Llama 2 model using ctransformers."""
    try:
        print("Initializing LLM")
        llm = AutoModelForCausalLM.from_pretrained(
            "TheBloke/Llama-2-7B-Chat-GGML",
            model_type="llama",
            model_file="llama-2-7b-chat.ggmlv3.q4_K_M.bin",
            max_new_tokens=512,
            temperature=0.5,
            repetition_penalty=1.15,
            context_length=1024,
        )
        return llm
    except Exception as e:
        print(f"Error initializing LLM: {str(e)}")
        return None

llm = initialize_llm()

def format_context(chunks: List[dict]) -> str:
    """Format retrieved chunks into a context string."""
    context = ""
    for chunk in chunks[:3]:  # Using top 3 most relevant chunks
        if "metadata" in chunk and "text" in chunk["metadata"]:
            context += f"{chunk['metadata']['text']}\n\n"
    return context.strip()

def generate_llama2_response(llm, query: str, context: str) -> str:
    """Generate response using Llama 2 based on query and context."""
    try:
        prompt_template = """<s>[INST] You are a helpful spiritual assistant. Use the provided context to answer questions accurately and concisely.
        If you can't find the answer in the context, say so honestly.
        Context:
        {context}
        Question: {query}
        Answer: [/INST]"""

        full_prompt = prompt_template.format(context=context, query=query)
        print("Sending to LLM")
        response = llm(full_prompt, max_new_tokens=512)

        return response.strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Startup event to initialize the HTTP client
@app.on_event("startup")
async def startup():
    global client
    if client is None:  # Ensure client is initialized only once
        client = httpx.AsyncClient()

# Shutdown event to close the HTTP client session
@app.on_event("shutdown")
async def shutdown():
    global client
    if client is not None:
        await client.aclose()
        client = None

# Home endpoint
@app.get("/")
def ret():
    return {"Hello": "World"}

@app.post("/query")
def chatbot_query(query: str):
    """API endpoint to process query and generate response."""
    try:
        query_embedding = embeddings.embed_query(query)
        results = index.query(
            vector=query_embedding, top_k=3, include_values=False, include_metadata=True
        )
        chunks = results.get("matches", [])
        if not chunks:
            return {"response": "I couldn't find any relevant information to answer your question."}
        context = format_context(chunks)
        response = generate_llama2_response(llm, query, context)
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}

# Telegram webhook endpoint
@app.post("/webhook")
async def telegram_webhook(request: Request):
    try:
        data = await request.json()
        message = data.get("message", {})
        chat_id = message.get("chat", {}).get("id")
        text = message.get("text", "")

        if not chat_id or not text:
            raise HTTPException(status_code=400, detail="Invalid Telegram message")

        # Process the query using Pinecone and generate a response
        response_text = await process_query(text)

        # Send the generated response to the Telegram user
        await send_message(chat_id, response_text)

        return {"status": "ok"}
    except Exception as e:
        return {"error": str(e)}

# Process the query using Pinecone
async def process_query(query: str) -> str:
    try:
        embedding = embeddings.embed_query(query)

        results = index.query(
            vector=embedding,
            top_k=7,
            include_values=False,
            include_metadata=True
        )

        top_chunks = results.get('matches', [])
        if not top_chunks:
            return "No relevant information found."

        response = format_context(top_chunks)
        return response.strip()

    except Exception as e:
        return f"Error processing query: {str(e)}"

# Send a message to the Telegram user
async def send_message(chat_id, text):
    global client
    if client is None:  # Reinitialize client if missing
        client = httpx.AsyncClient()

    try:
        response = await client.post(
            f"{TELEGRAM_API_URL}/sendMessage",
            json={"chat_id": chat_id, "text": text}
        )
        response.raise_for_status()  # Raise an error if the request failed
    except httpx.HTTPStatusError as e:
        print(f"HTTP error occurred: {e.response.status_code}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")



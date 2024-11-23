import os
import json
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from langchain.vectorstores import Pinecone as LangChainPinecone
from langchain_pinecone import PineconeEmbeddings
from pinecone import Pinecone, ServerlessSpec
import pinecone
from ctransformers import AutoModelForCausalLM
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")
INDEX_NAME = os.getenv("INDEX_NAME")
PINECONE_HOST = os.getenv("PINECONE_HOST")

# Initialize FastAPI app
app = FastAPI()

# Set up Pinecone
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))


def initialize_pinecone():
    """Initialize the Pinecone index and embeddings."""
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=1024,
            metric="euclidean",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_API_ENV),
        )
    return PineconeEmbeddings(model="multilingual-e5-large")


# Initialize Pinecone and LangChain embeddings
embeddings = initialize_pinecone()
index = pinecone.Index(
    index_name=INDEX_NAME, host=PINECONE_HOST, api_key=PINECONE_API_KEY
)


def initialize_llm():
    """Initialize the Llama 2 model using ctransformers."""
    try:
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
        response = llm(full_prompt, max_new_tokens=512)
        return response.strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"


@app.get("/chat")
async def chat_get(query: str, top_k: Optional[int] = 3):
    """Chatbot endpoint to handle GET requests."""
    try:
        query_embedding = embeddings.embed_query(query)
        results = index.query(
            vector=query_embedding, top_k=top_k, include_values=False, include_metadata=True
        )
        chunks = results.get("matches", [])
        if not chunks:
            return {"response": "I couldn't find any relevant information to answer your question."}

        context = format_context(chunks)
        response = generate_llama2_response(llm, query, context)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the query: {str(e)}")


@app.post("/chat")
async def chat_post(payload: dict):
    """Chatbot endpoint to handle POST requests."""
    query = payload.get("query", "")
    top_k = payload.get("top_k", 3)
    if not query:
        raise HTTPException(status_code=400, detail="Query is required.")
    try:
        query_embedding = embeddings.embed_query(query)
        results = index.query(
            vector=query_embedding, top_k=top_k, include_values=False, include_metadata=True
        )
        chunks = results.get("matches", [])
        if not chunks:
            return {"response": "I couldn't find any relevant information to answer your question."}

        context = format_context(chunks)
        response = generate_llama2_response(llm, query, context)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the query: {str(e)}")


# Initialize Llama 2 model
print("Initializing Llama 2 model (this might take a few minutes)...")
llm = initialize_llm()
if llm is None:
    raise Exception("Failed to initialize the LLM model.")

print("Server is ready!")

from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
import os
import re
from bs4 import BeautifulSoup
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone as LangChainPinecone
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME")

# Ensure Pinecone index exists or create it
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )

index = pc.Index(index_name)  # Load the index

# Function to read HTML files
def read_html_file(file_path):
    try:
        with open(file_path, 'r', encoding='windows-1252') as file:
            html_content = file.read()
        soup = BeautifulSoup(html_content, 'html.parser')
        paragraphs = [p.get_text(strip=True) for p in soup.find_all('p')]
        cleaned_paragraphs = [p.replace('\xa0', ' ') for p in paragraphs]
        return cleaned_paragraphs
    except Exception as e:
        print(f"Error reading the file: {e}")
        return []

# Function to clean and normalize text
def clean_text(text_list):
    return [re.sub(r'\s+', ' ', text).strip().replace('\n', ' ').replace('\t', '') for text in text_list]

# Remove unwanted text patterns
def remove_unwanted_text(text):
    unwanted_patterns = [r'ओम शान्ति', r'अव्यक्त बापदादा', r'मधुबन']
    for pattern in unwanted_patterns:
        text = re.sub(pattern, '', text)
    return text.strip()

# Extract the title
def extract_title(paragraphs):
    common_phrases = ['Morning Murli', 'Om Shanti', 'BapDada', 'Madhuban', 'ओम शान्ति', 'अव्यक्त बापदादा', 'मधुबन']
    meaningful_lines = [p.strip() for p in paragraphs if p.strip() and not any(phrase in p for phrase in common_phrases)]
    if meaningful_lines:
        title = meaningful_lines[0]
        if re.match(r'\d{4}[-/.]\d{2}[-/.]\d{2}', title) or title in common_phrases:
            title = meaningful_lines[1] if len(meaningful_lines) > 1 else "Title not found"
        return title
    return "Title not found"

# Function to process a file and extract relevant details
def process_file(file_path):
    text_array = read_html_file(file_path)
    cleaned_text_list = clean_text(text_array)

    date = "Date not found"
    title = extract_title(cleaned_text_list)

    content = [line for line in cleaned_text_list[1:] if line != title] if title else cleaned_text_list[1:]
    
    # Build document for indexing
    details = "\n".join(content)
    return f"Title: {title}\nContent:\n{details}\n", content

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)

# Define an endpoint for querying the index
@app.post("/query")
async def query_index(query_text: str):
    embeddings_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    query_embedding = embeddings_model.encode(query_text).tolist()

    if len(query_embedding) != 384:
        raise HTTPException(status_code=400, detail="Invalid query embedding dimension.")

    results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
    return {"results": results}

# Define an endpoint for processing and uploading documents
@app.post("/upload")
async def upload_file(file_path: str):
    extracted_data, content = process_file(file_path)
    docs = [Document(page_content=extracted_data)]
    split_docs = text_splitter.split_documents(docs)

    # Upsert documents to Pinecone
    try:
        upsert_data = [(str(i), doc.page_content, {"title": extract_title(content)}) for i, doc in enumerate(split_docs)]
        index.upsert(vectors=upsert_data)
        return {"status": "success", "message": "Documents uploaded to Pinecone."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during upsert: {e}")

# Example route to check app status
@app.get("/")
def read_root():
    return {"message": "Pinecone and FastAPI are running!"}


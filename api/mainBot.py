# Import Required Libraries
from dotenv import load_dotenv  # For loading environment variables
import os
import re
from bs4 import BeautifulSoup
from pinecone import Pinecone, ServerlessSpec  # Corrected import
from langchain_community.document_loaders import PyPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone as LangChainPinecone
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Check if the index exists and create if not
index_name = os.getenv("PINECONE_INDEX_NAME")
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )

# Load the index
index = pc.Index(index_name)

# FastAPI app initialization
app = FastAPI()

# Request model for querying
class QueryRequest(BaseModel):
    query_text: str

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

# Enhanced function to extract the date
def extract_date(text):
    date_patterns = [
        r'\b\d{2}[-/.]\d{2}[-/.]\d{4}\b',
        r'\b\d{2}[-/.]\d{2}[-/.]\d{2}\b',
        r'\b\d{4}[-/.]\d{2}[-/.]\d{2}\b',
        r'\b\d{2}\.\d{2}\.\d{4}\b',
    ]
    for pattern in date_patterns:
        date_match = re.search(pattern, text)
        if date_match:
            return date_match.group()
    return "Date not found"

# Improved function to extract the title
def extract_title(paragraphs):
    common_phrases = ['Morning Murli', 'Om Shanti', 'BapDada', 'Madhuban', 'ओम शान्ति', 'अव्यक्त बापदादा', 'मधुबन']
    
    meaningful_lines = [p.strip() for p in paragraphs if p.strip() and not any(phrase in p for phrase in common_phrases)]

    if meaningful_lines:
        title = meaningful_lines[0]
        if re.match(r'\d{4}[-/.]\d{2}[-/.]\d{2}', title) or title in common_phrases:
            title = meaningful_lines[1] if len(meaningful_lines) > 1 else "Title not found"
        return title
    return "Title not found"

# General extraction function for structured content
def extract_details(text):
    details = {}
    essence_match = re.search(r'Essence\s*:(.*?)\s*Question', text, re.DOTALL | re.MULTILINE)
    question_match = re.search(r'Question\s*:(.*?)\s*Answer', text, re.DOTALL | re.MULTILINE)
    answer_match = re.search(r'Answer\s*:(.*?)\s*Essence for dharna', text, re.DOTALL | re.MULTILINE)
    dharna_match = re.search(r'Essence for dharna\s*:(.*?)\s*Blessing', text, re.DOTALL | re.MULTILINE)
    blessing_match = re.search(r'Blessing\s*:(.*?)\s*Slogan', text, re.DOTALL | re.MULTILINE)
    slogan_match = re.search(r'Slogan\s*:(.*?)(\*+.*)?$', text, re.DOTALL | re.MULTILINE)

    details["Essence"] = essence_match.group(1).strip() if essence_match else "Essence not found"
    details["Question"] = question_match.group(1).strip() if question_match else "Question not found"
    details["Answer"] = clean_text([answer_match.group(1).strip()])[0] if answer_match else "Answer not found"
    details["Essence for Dharna"] = dharna_match.group(1).strip() if dharna_match else "Essence for Dharna not found"
    details["Blessing"] = blessing_match.group(1).strip() if blessing_match else "Blessing not found"
    details["Slogan"] = slogan_match.group(1).strip() if slogan_match else "Slogan not found"

    return details

# Function to process a file and extract relevant details based on file type
def process_file(file_path):
    text_array = read_html_file(file_path)
    cleaned_text_list = clean_text(text_array)
    
    date = "Date not found"
    title = "Title not found"
    
    details = {}  # Initialize details dictionary

    if cleaned_text_list:
        date_line = remove_unwanted_text(cleaned_text_list[0])
        date = extract_date(date_line)
    
        title = extract_title(cleaned_text_list)

        content = [line for line in cleaned_text_list[1:] if line != title] if title else cleaned_text_list[1:]

        details = extract_details("\n".join(content))
        details["Date"] = date
        details["Title"] = title
        details["Content"] = "\n".join(content)

        result = f"Date: {details['Date']}\nTitle: {details['Title']}\nContent:\n{details['Content']}\n"
        result += f"Essence: {details['Essence']}\nQuestion: {details['Question']}\nAnswer: {details['Answer']}\n"
        result += f"Essence for Dharna: {details['Essence for Dharna']}\nBlessing: {details['Blessing']}\nSlogan: {details['Slogan']}\n"
    
        return result, details  # Return both result and details
    return "No data found.", details  # Also return empty details if no data found

# Example usage
file_path = r"C:\Users\Utkar\Downloads\AMurli18Feb93English.htm"
extracted_data, details = process_file(file_path)  # Capture details here
print(extracted_data)

# Prepare documents for Pinecone
data1 = [Document(page_content=extracted_data)]  # Simplified construction of data1
print(data1)

# Split documents into smaller chunks for better indexing
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
docs = text_splitter.split_documents(data1)

# Upsert documents to Pinecone
try:
    # Prepare the data for upsert
    upsert_data = [(str(i), doc.page_content, {"title": details["Title"], "date": details["Date"]}) for i, doc in enumerate(docs)]
    
    # Upsert documents to Pinecone
    index.upsert(vectors=upsert_data)
    print("Upsert successful!")
except Exception as e:
    print(f"Error during upsert: {e}")

# Querying the Pinecone index locally
def query_index(query_text):
    embeddings_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Generate embeddings for the query
    query_embedding = embeddings_model.encode(query_text).tolist()
    
    # Validate the query embedding length
    if len(query_embedding) != 384:
        raise ValueError("Query embedding has an incorrect dimension.")
    
    # Query the Pinecone index
    results = index.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True
    )
    
    return results

# Example usage of querying the index
query_text = "shiv ratri or shiv jayanti"
results = query_index(query_text)
print("Query Results:", results)

# FastAPI endpoint for querying
@app.post("/query")
async def query_endpoint(request: QueryRequest):
    try:
        results = query_index(request.query_text)
        return {"status": "success", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# FastAPI endpoint for uploading documents
@app.post("/upload")
async def upload_file(file_path: str):
    try:
        extracted_data, details = process_file(file_path)
        return {"status": "success", "data": extracted_data, "details": details}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

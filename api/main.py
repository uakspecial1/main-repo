

# #**Step 2: Import All the Required Libraries**

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
import os
from langchain.vectorstores import Pinecone as LangChainPinecone  # Rename to avoid collision
from langchain_pinecone import PineconeEmbeddings
from langchain.docstore.document import Document
from pinecone import Pinecone, ServerlessSpec  # Importing the correct classes from Pinecone SDK
import os
import re
from bs4 import BeautifulSoup
from langchain.schema import Document
# Function to read HTML files
def read_html_file(file_path):
    try:
        with open(file_path, 'r', encoding='windows-1252') as file:
            html_content = file.read()
        soup = BeautifulSoup(html_content, 'html.parser')
        paragraphs = [p.get_text(strip=True) for p in soup.find_all('p')]
        cleaned_paragraphs = [p.replace('\xa0', ' ') for p in paragraphs]
        return cleaned_paragraphs
    except UnicodeDecodeError as e:
        print(f"Error reading the file: {e}")
        return []

# Function to clean and normalize text
def clean_text(text_list):
    cleaned_list = []
    for text in text_list:
        text = re.sub(r'\s+', ' ', text).strip()  # Replace multiple spaces with a single space
        text = text.replace('\n', ' ').replace('\t', '')  # Remove newlines and tabs
        cleaned_list.append(text)
    return cleaned_list

# Remove unwanted text patterns
def remove_unwanted_text(text):
    unwanted_patterns = [r'ओम शान्ति', r'अव्यक्त बापदादा', r'मधुबन']  # Add more unwanted patterns if needed
    for pattern in unwanted_patterns:
        text = re.sub(pattern, '', text)
    return text.strip()

# Enhanced function to extract the date with more flexible pattern matching
def extract_date(text):
    date_patterns = [
        r'\b\d{2}[-/.]\d{2}[-/.]\d{4}\b',  # DD-MM-YYYY or DD/MM/YYYY
        r'\b\d{2}[-/.]\d{2}[-/.]\d{2}\b',  # DD-MM-YY or DD/MM/YY
        r'\b\d{4}[-/.]\d{2}[-/.]\d{2}\b',  # YYYY-MM-DD or YYYY/MM/DD
        r'\b\d{2}\.\d{2}\.\d{4}\b',        # DD.MM.YYYY
    ]
    for pattern in date_patterns:
        date_match = re.search(pattern, text)
        if date_match:
            return date_match.group()
    return "Date not found"

# Improved function to extract the title by skipping unwanted phrases and using the first meaningful content
def extract_title(paragraphs):
    common_phrases = [
        'Morning Murli', 'Om Shanti', 'BapDada', 'Madhuban',  # English phrases
        'ओम शान्ति', 'अव्यक्त बापदादा', 'मधुबन'  # Hindi phrases
    ]
    
    meaningful_lines = []
    
    for paragraph in paragraphs:
        clean_paragraph = paragraph.strip()
        if clean_paragraph and not any(phrase in clean_paragraph for phrase in common_phrases):
            meaningful_lines.append(clean_paragraph)

    if meaningful_lines:
        title = meaningful_lines[0]
        
        if re.match(r'\d{4}[-/.]\d{2}[-/.]\d{2}', title) or title in common_phrases:
            if len(meaningful_lines) > 1:
                title = meaningful_lines[1]
            else:
                title = "Title not found"
        return title
    else:
        return "Title not found"

# General extraction function for structured content with improved pattern matching
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
    
    if cleaned_text_list:
        date_line = remove_unwanted_text(cleaned_text_list[0])
        date = extract_date(date_line)
    
        title = extract_title(cleaned_text_list)

        content = [line for line in cleaned_text_list[1:] if line != title] if title else cleaned_text_list[1:]

        details = extract_details("\n".join(content))
        details["Date"] = date
        details["Title"] = title
        details["Content"] = "\n".join(content)  # Join content as a single string

        # Format the output as a string
        result = f"Date: {details['Date']}\nTitle: {details['Title']}\nContent:\n{details['Content']}\n"
        result += f"Essence: {details['Essence']}\nQuestion: {details['Question']}\nAnswer: {details['Answer']}\n"
        result += f"Essence for Dharna: {details['Essence for Dharna']}\nBlessing: {details['Blessing']}\nSlogan: {details['Slogan']}\n"
    
        return result
    return "No data found."

# Example usage
file_path = r"C:\Users\Utkar\Downloads\Spiritual Bot\spiritualBot\api\murli.htm"
extracted_data = process_file(file_path)

data = extracted_data

data



#data1 = [{"page_content": data}]

data1 = [Document(page_content=data)]



# In[132]:


from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter=RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0, separators=["\n","\n\n","."],)


# In[133]:


docs= text_splitter.split_documents(data1)




docs[0]



PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', 'd95ac410-5110-4ca9-ad8c-69eea8b8c09d')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV', 'us-east-1')



import os
from getpass import getpass
os.environ["PINECONE_API_KEY"] = "d95ac410-5110-4ca9-ad8c-69eea8b8c09d"

os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY") or getpass(
    "Enter your Pinecone API key: "
)



from langchain_pinecone import PineconeEmbeddings

embeddings = PineconeEmbeddings(model="multilingual-e5-large") #1024 dimensions.

docs = [str(doc) for doc in docs]  # Convert each doc to string if needed

doc_embeds = embeddings.embed_documents(docs)



# Import Pinecone SDK and LangChain's Pinecone wrapper


# Set up Pinecone API key and environment
os.environ["PINECONE_API_KEY"] = 'd95ac410-5110-4ca9-ad8c-69eea8b8c09d'
PINECONE_API_ENV = 'us-east-1'

# Initialize Pinecone using the new class method
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Define the index name
index_name = "pinecone"

# Check if the index exists; if not, create it with correct dimensions
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1024,  # Must match the embedding model dimension (1024 for multilingual-e5-large)
        metric='euclidean',  # Adjust the metric as per your use case (euclidean, cosine, etc.)
        spec=ServerlessSpec(  # Specifying serverless settings if needed
            cloud='aws',
            region=PINECONE_API_ENV  # Using the environment variable
        )
    )

# Load the Pinecone embeddings
embeddings = PineconeEmbeddings(model="multilingual-e5-large")

# Assuming docs is a list of documents in string format
wrapped_docs = [Document(page_content=doc) for doc in docs]  # Wrap each string in a Document object

# Store the embeddings in the Pinecone index
docsearch = LangChainPinecone.from_documents(wrapped_docs, embeddings, index_name=index_name)


query="shiv ratri or shiv jayanti"

# docs=docsearch.similarity_search(query)


# # Perform similarity search with the query
# docs = docsearch.similarity_search(query)

# # Display the top 3 chunks
# if docs:
#     for i in range(min(10, len(docs))):  # Ensure we don't go out of bounds
#         print(f"Top {i + 1} Chunk:\n{docs[i].page_content}\n")
# else:
#     print("No results found.")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()


@app.post("/search/")
async def search():
    try:
        # Perform similarity search with the provided query
        docs = await docsearch.similarity_search(query)
        # Check if we have results
        if docs:
            results = [{"content": doc.page_content} for doc in docs[:3]]  # Get top 3 chunks
            return {"results": results}
        else:
            return {"results": [], "message": "No results found."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the app with: uvicorn your_filename:app --reload








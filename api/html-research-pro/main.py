import os
import uuid
import sys
import psycopg2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone,ServerlessSpec
import time
from openai import OpenAI
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import hashlib
import cohere
import asyncio  # Import asyncio for asynchronous operations
from fastapi.middleware.cors import CORSMiddleware
load_dotenv()

TOGETHER_API_KEY = os.getenv("TOGETHER_API")
COHERE_API = os.getenv("COHERE_API")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HELICON_API_KEY = os.getenv("HELICON_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

app = FastAPI()

SysPromptDefault = "You are now in the role of an expert AI."
SummaryTextPrompt = "You are an assistant tasked with summarizing TEXT for retrieval. These summaries will be embedded and used to retrieve the raw text elements. Give a concise summary of the TEXT that is well optimized for retrieval."
GenerationPrompt = "You are in the role of an expert AI whose task is to give ANSWER to the user's QUESTION based on the provided CONTEXT. Fully rely on CONTEXT; you can't also use your own intelligence too. The summary should be less than 300 words for each QUESTION. You must respond in markdown format; don't use big headings."



class QuestionInput(BaseModel):
    query: str

# Global in-memory storage (consider using a proper database or caching mechanism for production)
file_store = {}

def pinecone_server():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = 'research-pro'
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            index_name,
            dimension=1024,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        time.sleep(1)
    index = pc.Index(index_name)
    index.describe_index_stats()
    return index

def extract_content(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')

    # Extract H1 topics
    topics = [h1.get_text(strip=True) for h1 in soup.find_all('h1')]

    # List to hold text content
    texts = []

    # Extract text content from various tags
    for element in soup.find_all(['p', 'div', 'span', 'li']):
        text = element.get_text(strip=True)
        if text:  # Only add non-empty text
            texts.append(text)

    # Extract table content
    for table in soup.find_all('table'):
        table_data = []
        for row in table.find_all('tr'):
            row_data = [cell.get_text(strip=True) for cell in row.find_all(['th', 'td'])]
            table_data.append('\n'.join(row_data))
        texts.append('\n'.join(table_data))

    return topics, texts


def split(texts):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=384, chunk_overlap=10)
    text = "\n".join(texts)
    chunks = text_splitter.split_text(text)
    return chunks

def response(message, model="llama3-8b-8192", SysPrompt=SysPromptDefault, temperature=0.2):
    client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://gateway.hconeai.com/openai/v1",
        default_headers={
            "Helicone-Auth": f"Bearer {HELICON_API_KEY}",
            "Helicone-Target-Url": "https://api.groq.com"
        }
    )

    messages = [{"role": "system", "content": SysPrompt}, {"role": "user", "content": message}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        frequency_penalty=0.2
    )
    return response.choices[0].message.content

def generate_text_summaries(texts,summarize_texts):
   text_summaries = []
   if texts and summarize_texts:
      message = f"TEXT:\n\n{texts}"
      model="llama3-8b-8192"
      text_summaries = response(message=message, model=model, SysPrompt=SummaryTextPrompt, temperature=0)
   elif texts:
      text_summaries = texts

   return text_summaries

def get_digest(html_content):
    h = hashlib.sha256()
    h.update(html_content.encode('utf-8'))  # Encode the string to bytes before hashing
    return h.hexdigest()

def fetch_vectorstore_from_db(file_id):
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres.kstfnkkxavowoutfytoq",
        password="nI20th0in3@",
        host="aws-0-us-east-1.pooler.supabase.com",
        port="5432"
    )
    cur = conn.cursor()
    create_table_query = '''
        CREATE TABLE IF NOT EXISTS html_research_prov3 (
            file_id VARCHAR(255) PRIMARY KEY,
            file_name VARCHAR(255),
            name_space VARCHAR(255)
        );
    '''
    cur.execute(create_table_query)
    conn.commit()
    fetch_query = '''
    SELECT name_space
    FROM html_research_prov3 
    WHERE file_id = %s;
    '''
    cur.execute(fetch_query, (file_id,))
    result = cur.fetchone()
    cur.close()
    conn.close()
    if result:
        return result[0]
    return None

def get_next_namespace():
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres.kstfnkkxavowoutfytoq",
        password="nI20th0in3@",
        host="aws-0-us-east-1.pooler.supabase.com",
        port="5432"
    )
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM html_research_prov3")
    count = cur.fetchone()[0]
    next_namespace = f"html-{count + 1}"
    cur.close()
    conn.close()
    return next_namespace

def insert_data(file_id, file_name, name_space):
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres.kstfnkkxavowoutfytoq",
        password="nI20th0in3@",
        host="aws-0-us-east-1.pooler.supabase.com",
        port="5432"
    )
    cur = conn.cursor()
    create_table_query = '''
    CREATE TABLE IF NOT EXISTS html_research_prov3 (
        file_id VARCHAR(255) PRIMARY KEY,
        file_name VARCHAR(255),
        name_space VARCHAR(255)
    );
    '''
    cur.execute(create_table_query)
    conn.commit()
    insert_query = '''
    INSERT INTO html_research_prov3 (file_id, file_name, name_space)
    VALUES (%s, %s, %s)
    ON CONFLICT (file_id) DO NOTHING;
    '''
    cur.execute(insert_query, (file_id, file_name, name_space))
    conn.commit()
    cur.close()
    conn.close()

def create_documents(chunks, summaries):
    documents = []
    retrieve_contents = []

    for e, s in zip(chunks, summaries):
        i = str(uuid.uuid4())
        doc = {
            'page_content': s,
            'metadata': {
                'id': i,
                'type': 'text',
                'original_content': e
            }
        }
        retrieve_contents.append((i, e))
        documents.append(doc)

    return documents, retrieve_contents

def embed_and_upsert(documents, cohere_api_key, name_space):
    cohere_client = cohere.Client(cohere_api_key)
    summaries = [doc['page_content'] for doc in documents]
    pinecone_index = pinecone_server()
    embeddings = cohere_client.embed(
        texts=summaries,
        input_type='search_document',
        model="embed-english-v3.0"
    ).embeddings

    pinecone_data = []
    for doc, embedding in zip(documents, embeddings):
        pinecone_data.append({
            'id': doc['metadata']['id'],
            'values': embedding,
            'metadata': doc['metadata']
        })

    pinecone_index.upsert(vectors=pinecone_data, namespace=name_space)

def embedding_creation(html_content, COHERE_API, name_space):
    topics,texts = extract_content(html_content)
    chunks = split(texts)
    text_summaries = generate_text_summaries(chunks, summarize_texts=False)
    documents, retrieve_contents = create_documents(chunks, text_summaries)
    embed_and_upsert(documents, COHERE_API, name_space)
    print("Embeddings created and upserted successfully into Pinecone.")

def embed(question):
    cohere_client = cohere.Client(COHERE_API)
    embeddings = cohere_client.embed(
        texts=[question],
        model="embed-english-v3.0",
        input_type='search_query'
    ).embeddings
    return embeddings

def process_rerank_response(rerank_response, docs):
    rerank_docs = []
    for item in rerank_response.results:
        index = item.index
        if 0 <= index < len(docs):
            rerank_docs.append(docs[index])
        else:
            print(f"Warning: Index {index} is out of range for documents list.")
    return rerank_docs

async def get_name_space(question, html_content, file_name):
    file_id = get_digest(html_content)
    existing_namespace = fetch_vectorstore_from_db(file_id)

    if existing_namespace:
        print("Document already exists. Using existing namespace.")
        name_space = existing_namespace
    else:
        print("Document is new. Creating embeddings and new namespace.")
        name_space = get_next_namespace()
        embedding_creation(html_content, COHERE_API, name_space)
        insert_data(file_id, file_name, name_space)
        await asyncio.sleep(5)

    return name_space

async def get_docs(question,html_content, file_name):
    
    index = pinecone_server()
    co = cohere.Client(COHERE_API)
    xq = embed(question)[0]
    name_space=await get_name_space(question,html_content,file_name)
    print(name_space)
    res = index.query(namespace=name_space, vector=xq, top_k=5, include_metadata=True)
    print(res)
    docs = [x["metadata"]['original_content'] for x in res["matches"]]
    
    if not docs:
        print("No matching documents found.")
        return []

    results = co.rerank(query=question, documents=docs, top_n=3, model='rerank-english-v3.0')
    reranked_docs = process_rerank_response(results, docs)
    return reranked_docs


async def answer(question, html_content, file_name):
    docs = await get_docs(question, html_content, file_name)
    if not docs:
        return "No relevant documents found for the given question."

    context = "\n\n".join(docs)
    message = f"CONTEXT:\n\n{context}\n\nQUESTION :\n\n{question}\n\nANSWER: \n"
    model = "llama3-8b-8192"
    output = response(message=message, model=model, SysPrompt=GenerationPrompt, temperature=0)
    return output

async def generate_output(html_content, file_name):
    topics, texts = extract_content(html_content)
    summary = ""
    for topic in topics:
        output =await answer(topic, html_content, file_name)
        summary += f"## {topic}\n\n{output}\n\n"
    return summary

@app.post("/process-html")
async def process_html(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="HTML file not provided")
    file_content = await file.read()
    html_content = file_content.decode('utf-8')
    filename = file.filename
    output = await generate_output(html_content, filename)
    
    # Store the file content in the global store
    file_id = get_digest(html_content)
    file_store[file_id] = {
        "html_content": html_content,
        "filename": filename
    }
    
    return {"output": output}

@app.post("/ask-question")
async def ask_question(input: QuestionInput):
    if not file_store:
        raise HTTPException(status_code=400, detail="No HTML file has been processed yet")

    # Get the latest processed file's content
    file_id, file_data = list(file_store.items())[-1]
    html_content = file_data["html_content"]
    filename = file_data["filename"]

    # Generate an answer for the provided question
    answer_output = await answer(input.query, html_content, filename)

    return {"answer": answer_output}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

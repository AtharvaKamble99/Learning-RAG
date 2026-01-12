import os
from sentence_transformers import SentenceTransformer 
# import langchain_community.document_loaders import TextLoader , DirectoryLoader
# from langchain_text_splitters import CharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
# from langchain_chroma import Chroma
from dotenv import load_dotenv
load_dotenv()
import chromadb
from chromadb.config import Settings
from chromadb import PersistentClient


#CONFIG

MODEL_NAME= os.getenv("MODEL_NAME")
DOC_DIR= os.getenv("DOC_DIR")
CHUNK_SIZE= int(os.getenv("CHUNK_SIZE"))
CHUNK_OVERLAP= int(os.getenv("CHUNK_OVERLAP"))

print(f"Using model: {MODEL_NAME}")

#CHROMA DB SETUP 
#CHROMA DB SETUP 
chroma_db_path = os.path.join(os.path.dirname(__file__), "chroma_db")
client = PersistentClient(
    path=chroma_db_path,
    settings=Settings(
        anonymized_telemetry=False
    )
)
collection  = client.get_or_create_collection(
    name="documents",
    metadata={"embedding_model":MODEL_NAME}
)

def chunk_text(text, size, overlap):
    chunks=[]
    start=0
    while start < len(text):
        end=start+size
        chunks.append(text[start:end])
        start=end-overlap

    return chunks



model = SentenceTransformer(MODEL_NAME)


for filename in os.listdir(DOC_DIR):
    print(f"Processing file: {filename}")
    if not filename.endswith(".txt"):
        continue

    filepath= os.path.join(DOC_DIR,filename)

    with open(filepath,'r',encoding="utf-8") as f:
        text=f.read()
        
    chunks=chunk_text(text,CHUNK_SIZE,CHUNK_OVERLAP)

    embeddings= model.encode(chunks).tolist()

    ids=[f"{filename}_chunk_{i}" for i in range(len(chunks))]
    metadata=[{"source" : filename} for _ in chunks]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadata,
        ids=ids
    )
    
    



        


import os 
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from chromadb import PersistentClient
from chromadb.config import Settings 


load_dotenv()

MODEL_NAME= os.getenv("MODEL_NAME")
DOC_DIR= os.getenv("DOC_DIR")
CHUNK_SIZE= int(os.getenv("CHUNK_SIZE"))
CHUNK_OVERLAP= int(os.getenv("CHUNK_OVERLAP"))

COLLECTION_NAME = "documents"

print(f"Using model: {MODEL_NAME}")

CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")

model=SentenceTransformer(MODEL_NAME)

client=PersistentClient(
    path=CHROMA_DB_PATH,
    settings=Settings(
        anonymized_telemetry=False
    )
)

collection = client.get_collection(name=COLLECTION_NAME)

def retrieval(query,top_k):
    query_embedding=model.encode(query).tolist()
    results=collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results 


if __name__ == "__main__":
    query=input("Enter your query : ")

    results=retrieval(query,top_k=5)

    print("\n--- Retrieved Chunks ---\n")

    for i in range(len(results["documents"][0])):
        print(f"Result {i+1}")
        print(f"Source   : {results['metadatas'][0][i]['source']}")
        print(f"Content  : {results['documents'][0][i][:300]}")
        print("-" * 50)
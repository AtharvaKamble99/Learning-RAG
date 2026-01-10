import os
import langchain_community.document_loaders import TextLoader , DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()


def load_documents(docs_path="docs"):
    print(f"Loading documents from {docs_path}...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exists . Please create one ")
    

    loader = DirectoryLoader(
        path=docs_path
        glob="*.txt"
        loader_cls=TextLoader
    )

    documents=loader.load()

    if len(documents) ==0 :
        raise FileNotFoundError(f"No .txt file found in {docs_path} .")
    


    for i,doc in enumerate(documents[:2]):
        print(f"\nDocument {i+1}")
        print(f"Source : {doc.metadata['source']}")
        print(f"Content length : {len(doc.page_content)} characters ")
        print(f"Content Preview : {doc.page_content{:100}}...")
        print(f"metadata : {doc.metadata}" )

        return documents
        


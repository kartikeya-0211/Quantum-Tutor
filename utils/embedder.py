# utils/embedder.py
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.storage import InMemoryStore 

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
BASE_DB_PATH = "db"

def embed_and_store(chunks_tuple):
    child_chunks, parent_chunks = chunks_tuple

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={'normalize_embeddings': True}
    )

    child_db_path = os.path.join(BASE_DB_PATH, "child_chunks")
    parent_db_path = os.path.join(BASE_DB_PATH, "parent_chunks") 

    os.makedirs(child_db_path, exist_ok=True)
    os.makedirs(parent_db_path, exist_ok=True) 

    print(f"Embedding and storing child chunks in {child_db_path}...")
    child_vectordb = Chroma.from_documents(
        documents=child_chunks,
        embedding=embeddings,
        persist_directory=child_db_path
    )
    print(f"✅ {len(child_chunks)} child chunks embedded and stored.")

    parent_docstore = InMemoryStore()
    parent_docs_dict = {doc.metadata["parent_id"]: doc for doc in parent_chunks}
    parent_docstore.mset(list(parent_docs_dict.items())) 

    print(f"Embedding and storing parent chunk metadata in {parent_db_path}...")
    parent_vectordb = Chroma.from_documents(
        documents=parent_chunks,
        embedding=embeddings,
        persist_directory=parent_db_path
    )
    print(f"✅ {len(parent_chunks)} parent chunk metadata embedded and stored.")

    print(f"All documents embedded and stored in ChromaDB at {BASE_DB_PATH}.")
    
    return child_vectordb, parent_docstore, parent_vectordb

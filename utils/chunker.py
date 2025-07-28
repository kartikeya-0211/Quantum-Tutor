from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def chunk_documents(documents, chunk_size=512, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def chunk_documents_hierarchical(documents, small_chunk_size=256, small_chunk_overlap=20,
                                 large_chunk_size=1024, large_chunk_overlap=100):
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=large_chunk_size, chunk_overlap=large_chunk_overlap)
    parent_chunks = parent_splitter.split_documents(documents)

    for i, parent_chunk in enumerate(parent_chunks):
        parent_chunk.metadata["parent_id"] = f"parent_chunk_{i}"

    child_splitter = RecursiveCharacterTextSplitter(chunk_size=small_chunk_size, chunk_overlap=small_chunk_overlap)
    
    child_chunks = []
    for parent_chunk in parent_chunks:
        temp_child_docs = child_splitter.create_documents(
            texts=[parent_chunk.page_content],
            metadatas=[{"parent_id": parent_chunk.metadata["parent_id"], **parent_chunk.metadata}]
        )
        child_chunks.extend(temp_child_docs)

    return child_chunks, parent_chunks

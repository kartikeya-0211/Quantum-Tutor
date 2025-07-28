import os
from langchain_community.document_loaders import PyMuPDFLoader
import sys 

def load_docs():
    folder_path = "data"
    all_docs = []

    if not os.path.exists(folder_path):
        print(f"Error: Data folder '{folder_path}' not found.", file=sys.stderr)
        return []

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            filepath = os.path.join(folder_path, filename)
            try:
                loader = PyMuPDFLoader(filepath)
                file_docs = loader.load()
                all_docs.extend(file_docs)
            except Exception as e: # Catch specific exception
                print(f"❌ Error loading PDF file '{filepath}': {e}", file=sys.stderr) # Log the error
                continue 
        # Skip non-PDF files without printing "continue" as it's implicit
    return all_docs
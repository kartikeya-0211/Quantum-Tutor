from langchain.retrievers import ParentDocumentRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.vectorstores import VectorStore
from langchain_core.stores import BaseStore
from langchain_core.language_models import BaseLanguageModel
from langchain_core.documents import Document
from typing import Tuple, List
from retrievers.reranker import CrossEncoderReranker
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables import RunnableLambda
from langchain.load.dump import dumps
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores.utils import filter_complex_metadata

from langchain_groq import ChatGroq
from langchain_ollama import OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from prompts.prompt_template import strict_rag_prompt
import os

from utils.llm_loader import get_llm

def get_rag_chain(
    child_vectordb: VectorStore,
    parent_docstore: BaseStore[str, Document],
    parent_vectordb: VectorStore,
    model_name: str,
    child_chunks: List[Document]
) -> Tuple[CrossEncoderReranker, ParentDocumentRetriever, BaseLanguageModel, ParentDocumentRetriever]:

    reranker = CrossEncoderReranker(model_name="BAAI/bge-reranker-v2-m3")
    pipeline = DocumentCompressorPipeline(transformers=[reranker])
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    bm25_retriever = BM25Retriever.from_documents(child_chunks)
    bm25_retriever.k = 12 

    vector_retriever = child_vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 12})
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6] 
    )

    parent_retriever = ParentDocumentRetriever(
        vectorstore=child_vectordb,
        docstore=parent_docstore,
        child_splitter=text_splitter,
        child_retriever=ensemble_retriever, 
        id_key="parent_id"
    )

    # Compressed parent retriever 
    compressed_parent_retriever = ParentDocumentRetriever(
        vectorstore=child_vectordb,
        docstore=parent_docstore,
        child_splitter=text_splitter,
        child_compressor=pipeline,
        child_retriever=ensemble_retriever,
        search_kwargs={"k": 5},
        id_key="parent_id"
    )

    llm_for_rag = get_llm(model_name)

    return reranker, parent_retriever, llm_for_rag, compressed_parent_retriever

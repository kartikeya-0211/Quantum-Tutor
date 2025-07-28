# reranker.py
from typing import List, Tuple
from langchain_core.documents import Document
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from pydantic import Field, PrivateAttr
from FlagEmbedding import FlagReranker

class CrossEncoderReranker(BaseDocumentCompressor):
    _model = PrivateAttr()

    model_name: str = Field("BAAI/bge-reranker-v2-m3", description="Name of the reranker model.")
    device: str = Field("cuda", description="Device to run the model on (e.g., 'cpu' or 'cuda').")
    use_fp16: bool = Field(default=False)

    def __init__(self, **data):
        super().__init__(**data)
        self._model = FlagReranker(self.model_name, use_fp16=self.use_fp16, device=self.device)

    def compress_documents(self, documents: List[Document], query: str, **kwargs) -> List[Document]:
        if not documents:
            return []

        doc_texts = [doc.page_content for doc in documents]

        try:
            scores = self._model.compute_score([[query, doc_text] for doc_text in doc_texts])
        except Exception:
            return []

        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        top_k = kwargs.get("k", len(documents))
        reranked_docs = []

        for doc, score in doc_score_pairs[:top_k]:
            doc.metadata['relevance_score'] = score
            reranked_docs.append(doc)

        return reranked_docs

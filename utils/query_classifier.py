# utils/query_classifier.py
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
import os
from prompts.prompt_template import CLASSIFICATION_PROMPT 

llm_classifier = OllamaLLM(model=os.getenv("OLLAMA_MODEL", "gemma2:2b-gpu-only"))

classification_chain = (
    CLASSIFICATION_PROMPT 
    | llm_classifier 
    | StrOutputParser()
)


def classify_query(question: str) -> str:
    try:
        result = classification_chain.invoke({"question": question})
        return result.strip().upper()
    except Exception as e:
        print(f"[Classifier Error] {e}")
        return "IN_SCOPE"  

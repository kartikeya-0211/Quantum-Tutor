# utils/llm_loader.py

from langchain_ollama import OllamaLLM
from langchain_groq import ChatGroq
import os


from langchain_groq import ChatGroq
from langchain_ollama import OllamaLLM

def get_llm(model_name: str, temperature: float = 0.4):
    if model_name.startswith("llama3") or "groq" in model_name.lower():
        return ChatGroq(model=model_name, temperature=temperature)
    else:
        return OllamaLLM(model=model_name)

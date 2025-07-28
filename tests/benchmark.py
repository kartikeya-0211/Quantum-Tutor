import pandas as pd
import litellm
import time
import os
import logging
from dotenv import load_dotenv
import yaml
import csv


from langchain_litellm import ChatLiteLLM
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.storage import InMemoryStore
from langchain_core.documents import Document


from chains.rag_chains import get_rag_chain
from utils.query_classifier import classify_query

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('chromadb').setLevel(logging.CRITICAL)


try:
    with open("config/litellm_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    litellm.model_list = config["model_list"]
    logging.info("Successfully loaded model configuration into LiteLLM.")
except Exception as e:
    logging.error(f"CRITICAL ERROR: Could not load litellm_config.yaml. Error: {e}")
    exit()


try:
    questions_df = pd.read_csv("tests/Benchmark_questions.csv")
    logging.info(f"Successfully loaded {len(questions_df)} questions.")
except FileNotFoundError:
    logging.error("Error: 'cleaned_questions.csv' not found in 'tests/' folder.")
    exit()


logging.info("Initializing knowledge base...")
DB_PATH = os.getenv("DB_PATH", "db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, encode_kwargs={'normalize_embeddings': True})
child_vectordb = Chroma(persist_directory=os.path.join(DB_PATH, "child_chunks"), embedding_function=embedding_model)
parent_vectordb = Chroma(persist_directory=os.path.join(DB_PATH, "parent_chunks"), embedding_function=embedding_model)
parent_docstore = InMemoryStore()
retrieved_data = parent_vectordb.get(ids=parent_vectordb.get()['ids'], include=["documents", "metadatas"])
parent_docs = [Document(page_content=d, metadata=m) for d, m in zip(retrieved_data["documents"], retrieved_data["metadatas"]) if "parent_id" in m]
parent_docstore.mset([(doc.metadata["parent_id"], doc) for doc in parent_docs])
retrieved_child_data = child_vectordb.get(include=['documents', 'metadatas'])
child_chunks = [Document(page_content=d, metadata=m) for d, m in zip(retrieved_child_data['documents'], retrieved_child_data['metadatas'])]
logging.info("Knowledge base loaded.")

results_path = "results/rag_benchmark_results.csv"
existing_results = []
if os.path.exists(results_path):
    logging.info("Found existing results file. Will skip completed questions.")
    existing_df = pd.read_csv(results_path)
    completed_tests = set(zip(existing_df['model'], existing_df['question']))
    existing_results = existing_df.to_dict('records')
else:
    completed_tests = set()
    logging.info("No existing results file found. Starting a new benchmark.")


models_to_test = ["groq/llama3-70b-8192"]
new_results = []


logging.info(f"Starting RAG benchmark for {len(models_to_test)} model(s)...")

for model in models_to_test:
    logging.info(f"--- Testing model: {model} ---")
    
    llm_for_rag = ChatLiteLLM(model=model, temperature=0.4, max_tokens=1024)
    llm_classifier = ChatLiteLLM(model="ollama/gemma2:2b-gpu-only", temperature=0.0)

    _, _, _, compression_retriever = get_rag_chain(
        child_vectordb=child_vectordb, parent_docstore=parent_docstore,
        parent_vectordb=parent_vectordb, model_name=model, child_chunks=child_chunks
    )
    history_aware_retriever = create_history_aware_retriever(llm_classifier, compression_retriever, ChatPromptTemplate.from_messages([("user", "{input}")]))
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's question based on the below context:\n\n{context}"),
        ("user", "{input}"),
    ])
    answer_chain = create_stuff_documents_chain(llm_for_rag, answer_prompt)
    conversational_chain = create_retrieval_chain(history_aware_retriever, answer_chain)
    
    for index, row in questions_df.iterrows():
        question = row['question']
        ideal_answer = row['ideal_answer']
        
        if (model, question) in completed_tests:
            logging.info(f"Skipping question #{index + 1} for model '{model}', result already exists.")
            continue

        logging.info(f"Running question #{index + 1}/{len(questions_df)}: {question[:60]}...")
        
        try:
            start_time = time.time()
            response = conversational_chain.invoke({"input": question})
            end_time = time.time()
            
            latency = round((end_time - start_time) * 1000)
            actual_answer = response.get("answer", "No answer found.")
            retrieved_context = response.get("context", [])
            context_str = "\n---\n".join([doc.page_content for doc in retrieved_context])

            logging.info(f"-> Finished in {latency} ms.")

            new_results.append({
                "model": model, "question": question,
                "ideal_answer": ideal_answer, "retrieved_context": context_str,
                "actual_answer": actual_answer, "latency_ms": latency
            })

        except Exception as e:
            logging.error(f"-> Error on question #{index + 1}: {e}", exc_info=True)
            new_results.append({
                "model": model, "question": question,
                "ideal_answer": ideal_answer, "retrieved_context": "ERROR",
                "actual_answer": f"ERROR: {e}", "latency_ms": -1
            })
        
        if "groq" in model:
            time.sleep(20)


all_results = existing_results + new_results
output_path_csv = "results/rag_benchmark_results_llama70b.csv"
logging.info(f"Attempting to save {len(all_results)} total results to {output_path_csv} with robust quoting...")

try:
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_path_csv, index=False, quoting=csv.QUOTE_ALL)
    logging.info(f"🎉 RAG Benchmark complete! Results successfully saved to {output_path_csv}")

except Exception as e:
    logging.error(f"Failed to save as CSV due to a stubborn error: {e}")
    logging.warning("Attempting to save as a JSON lines (.jsonl) file as a fallback...")
    
    fallback_path_json = "results/rag_benchmark_results_llama70b.jsonl"
    results_df.to_json(fallback_path_json, orient="records", lines=True, force_ascii=False)
    logging.info(f"Fallback successful. Benchmark results have been saved to {fallback_path_json}")
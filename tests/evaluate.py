import pandas as pd
import os
import time
from dotenv import load_dotenv
import logging
from langchain_litellm import ChatLiteLLM
from langchain_core.prompts import PromptTemplate
import json
import csv
import litellm

# --- Setup ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
os.environ["LITELLM_CONFIG_PATH"] = "config/litellm_config.yaml"

# --- LLM and Data Loading ---
judge_llm = ChatLiteLLM(model="groq/llama3-70b-8192")

# --- NEW: List of all benchmark files to combine ---
benchmark_files = [
    "results/rag_benchmark_results.jsonl",
    "results/rag_benchmark_results_llama70b.jsonl"
]
evaluation_output_path = "results/custom_evaluation_scores.csv"

# --- NEW: Load and combine all benchmark data ---
all_benchmark_data = []
for file_path in benchmark_files:
    try:
        df_temp = pd.read_json(file_path, lines=True)
        all_benchmark_data.append(df_temp)
    except FileNotFoundError:
        logging.warning(f"Warning: Benchmark file not found at {file_path}, skipping.")

if not all_benchmark_data:
    logging.error("No benchmark result files found. Please run the benchmark first.")
    exit()

df = pd.concat(all_benchmark_data, ignore_index=True)
df = df[df['latency_ms'] != -1].dropna().reset_index(drop=True)
df.drop_duplicates(subset=['model', 'question'], inplace=True)
logging.info(f"Successfully loaded and combined {len(df)} total results.")


# --- Resume Logic ---
existing_results = []
if os.path.exists(evaluation_output_path):
    logging.info("Found existing evaluation file. Will skip completed evaluations.")
    existing_df = pd.read_csv(evaluation_output_path)
    completed_evals = set(zip(existing_df['model'], existing_df['question']))
    existing_results = existing_df.to_dict('records')
else:
    completed_evals = set()
    logging.info("No existing evaluation file found. Starting new evaluation.")

# --- 1-5 SCALE Evaluation Prompt ---
evaluation_prompt_template = """
SYSTEM: You are an expert RAG system evaluator. Your task is to evaluate a response on 5 metrics based on the provided data.
Provide your evaluation in a single, valid JSON object. For each key, provide an integer score from 1 to 5.

[EVALUATION CRITERIA & SCORING]
1.  **faithfulness (Score 1-5)**: How factually consistent is the "Submission" with the "Context"? (5 = fully consistent, 1 = completely hallucinates)
2.  **answer_relevancy (Score 1-5)**: How relevant is the "Submission" to the "Question"? (5 = perfectly relevant, 1 = irrelevant)
3.  **context_precision (Score 1-5)**: How relevant is the "Context" to the "Question"? (5 = all context is relevant, 1 = context is irrelevant)
4.  **context_recall (Score 1-5)**: Does the "Context" contain all the information from the "Reference Answer" to answer the question? (5 = all information is present, 1 = key information is missing)
5.  **answer_correctness (Score 1-5)**: How factually correct and complete is the "Submission" compared to the "Reference Answer"? (5 = fully correct and complete, 1 = completely incorrect)

[DATA]
- Question: {question}
- Context: {context}
- Submission: {prediction}
- Reference Answer: {reference}

[YOUR JSON RESPONSE WITH SCORES from 1 to 5]
"""

evaluation_prompt = PromptTemplate(
    input_variables=["question", "context", "prediction", "reference"],
    template=evaluation_prompt_template
)
evaluation_chain = evaluation_prompt | judge_llm
logging.info("Custom evaluation chain is ready.")

# --- Evaluation Loop ---
logging.info("Starting custom evaluation...")
new_evaluation_results = []

for index, row in df.iterrows():
    model = row['model']
    question = row['question']

    if (model, question) in completed_evals:
        logging.info(f"Skipping row for model '{model}' on question '{question[:30]}...', already evaluated.")
        continue

    logging.info(f"Evaluating row {index + 1}/{len(df)} for model '{model}'...")
    
    response_text = ""
    try:
        response = evaluation_chain.invoke({
            "question": row["question"],
            "context": row["retrieved_context"],
            "prediction": row["actual_answer"],
            "reference": row["ideal_answer"]
        })
        
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        json_start = response_text.find('{')
        json_end = response_text.rfind('}')
        if json_start != -1 and json_end != -1:
            clean_json_str = response_text[json_start : json_end + 1]
            scores = json.loads(clean_json_str)
        else:
            raise json.JSONDecodeError("No JSON object found in response", response_text, 0)

        new_evaluation_results.append({
            'model': row['model'], 'question': row['question'],
            'faithfulness': scores.get('faithfulness', 0),
            'answer_relevancy': scores.get('answer_relevancy', 0),
            'context_precision': scores.get('context_precision', 0),
            'context_recall': scores.get('context_recall', 0),
            'answer_correctness': scores.get('answer_correctness', 0),
        })

    except (litellm.RateLimitError, json.JSONDecodeError) as e:
        logging.error(f"Stopping run on row {index + 1} due to a critical error: {e}")
        logging.info("Saving all completed results before exiting...")
        all_results = existing_results + new_evaluation_results
        if all_results:
            evaluation_df = pd.DataFrame(all_results).drop_duplicates(subset=['model', 'question'])
            evaluation_df.to_csv(evaluation_output_path, index=False, quoting=csv.QUOTE_ALL)
            logging.info(f"Progress saved to {evaluation_output_path}.")
        exit()
    except Exception as e:
        logging.error(f"An unexpected error occurred on row {index + 1}: {e}")

    # Long delay to accommodate daily limits over 2 days
    time.sleep(80)

# --- Save and Display Final Results ---
all_results = existing_results + new_evaluation_results
evaluation_df = pd.DataFrame(all_results).drop_duplicates(subset=['model', 'question'])
evaluation_df.to_csv(evaluation_output_path, index=False, quoting=csv.QUOTE_ALL)
logging.info(f"Full evaluation scores saved to {evaluation_output_path}")

summary = evaluation_df.groupby("model").mean(numeric_only=True).reset_index()
print("\n--- Average Quality Scores per Model (Scale 1-5) ---")
print(summary.to_markdown(index=False))
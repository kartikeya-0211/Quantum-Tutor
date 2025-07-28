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


load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
os.environ["LITELLM_CONFIG_PATH"] = "config/litellm_config.yaml"


judge_llm = ChatLiteLLM(model="groq/llama3-70b-8192")
benchmark_results_path = "results/rag_benchmark_results_llama70b.jsonl"
evaluation_output_path = "results/llama70b_evaluation_scores.csv"

try:
    df = pd.read_json(benchmark_results_path, lines=True)
    df = df[df['latency_ms'] != -1].dropna().reset_index(drop=True)
    df = df.head(200)
    
    logging.info(f"Successfully loaded and limited to {len(df)} questions for this test run.")
except FileNotFoundError:
    logging.error(f"Error: Benchmark results file not found at {benchmark_results_path}. Please run the benchmark first.")
    exit()


existing_results = []
if os.path.exists(evaluation_output_path):
    logging.info("Found existing evaluation file. Will skip completed evaluations.")
    existing_df = pd.read_csv(evaluation_output_path)
    completed_evals = set(zip(existing_df['model'], existing_df['question']))
    existing_results = existing_df.to_dict('records')
else:
    completed_evals = set()
    logging.info("No existing evaluation file found. Starting new evaluation.")


evaluation_prompt_template = """
SYSTEM: You are an expert RAG system evaluator. Your task is to evaluate a response based on the provided data.
First, evaluate the response against each of the 5 criteria below.
Then, provide your final answer as a single 5-character string made up of 'Y' (Yes) or 'N' (No).
The first character corresponds to the verdict for Faithfulness, the second to Answer Relevancy, and so on.
For example: YNYYN

[EVALUATION CRITERIA]
1.  **Faithfulness**: Is the "Submission" entirely supported by the "Context"?
2.  **Answer Relevancy**: Is the "Submission" a relevant and useful answer to the "Question"?
3.  **Context Precision**: Is the "Context" relevant and helpful for answering the "Question"?
4.  **Context Recall**: Does the "Context" contain all the information from the "Reference Answer" needed to answer the "Question"?
5.  **Answer Correctness**: Is the "Submission" factually and substantively the same as the "Reference Answer"?

[DATA]
- Question: {question}
- Context: {context}
- Submission: {prediction}
- Reference Answer: {reference}

[YOUR 5-CHARACTER VERDICT]
"""

evaluation_prompt = PromptTemplate(
    input_variables=["question", "context", "prediction", "reference"],
    template=evaluation_prompt_template
)
evaluation_chain = evaluation_prompt | judge_llm
logging.info("Custom evaluation chain is ready.")


logging.info("Starting custom evaluation...")
new_evaluation_results = []

for index, row in df.iterrows():
    model = row['model']
    question = row['question']

    if (model, question) in completed_evals:
        logging.info(f"Skipping row {index + 1}/{len(df)} for model '{model}', already evaluated.")
        continue

    logging.info(f"Evaluating row {index + 1}/{len(df)} for model '{model}'...")
    
    try:
        response = evaluation_chain.invoke({
            "question": row["question"],
            "context": row["retrieved_context"],
            "prediction": row["actual_answer"],
            "reference": row["ideal_answer"]
        })
        
        response_text = (response.content if hasattr(response, 'content') else str(response)).strip().upper()
        
        if len(response_text) >= 5:
            faithfulness_score = 1 if response_text[0] == 'Y' else 0
            answer_relevancy_score = 1 if response_text[1] == 'Y' else 0
            context_precision_score = 1 if response_text[2] == 'Y' else 0
            context_recall_score = 1 if response_text[3] == 'Y' else 0
            answer_correctness_score = 1 if response_text[4] == 'Y' else 0
        else:
            raise ValueError(f"LLM returned a malformed verdict string: {response_text}")

        new_evaluation_results.append({
            'model': row['model'], 'question': row['question'],
            'faithfulness': faithfulness_score, 'answer_relevancy': answer_relevancy_score,
            'context_precision': context_precision_score, 'context_recall': context_recall_score,
            'answer_correctness': answer_correctness_score,
        })

    except litellm.RateLimitError as e:
        logging.error(f"Stopping run on row {index + 1} due to Rate Limit Error: {e}")
        logging.info("Saving all completed results before exiting...")
        all_results = existing_results + new_evaluation_results
        if all_results:
            evaluation_df = pd.DataFrame(all_results)
            evaluation_df.to_csv(evaluation_output_path, index=False, quoting=csv.QUOTE_ALL)
            logging.info(f"Progress saved to {evaluation_output_path}.")
        exit()
    except Exception as e:
        logging.error(f"An unexpected error occurred on row {index + 1}: {e}")
        new_evaluation_results.append({'model': row['model'], 'question': row['question'], 'faithfulness': 0, 'answer_relevancy': 0, 'context_precision': 0, 'context_recall': 0, 'answer_correctness': 0})

    time.sleep(20)


all_results = existing_results + new_evaluation_results
evaluation_df = pd.DataFrame(all_results)
evaluation_df.to_csv(evaluation_output_path, index=False, quoting=csv.QUOTE_ALL)
logging.info(f"Full evaluation scores saved to {evaluation_output_path}")

summary = evaluation_df.groupby("model").mean(numeric_only=True).reset_index()
print("\n--- Average Quality Scores per Model (0 or 1) ---")
print(summary.to_markdown(index=False))
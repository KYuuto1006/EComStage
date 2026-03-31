import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import torch
import torch.nn.functional as F
from openai import OpenAI
import re

def extract_score(text):
    try:
        return json.loads(text)["score"]
    except:
        match = re.search(r"\{.*?\}", text, re.S)
        if match:
            return json.loads(match.group())["score"]
    return None

# local vLLM API server
client = OpenAI(
    base_url="http://localhost:8000/v1",  # vLLM server address
    api_key="EMPTY"  # EMPTY if local
)

def is_correct(pred_ans, gt_ans):
    if pred_ans is None or gt_ans is None:
        return False
    pred_ans = str(pred_ans).replace("\n", " ").replace("\t", " ").strip()
    gt_ans = str(gt_ans).strip()
    return gt_ans in pred_ans


def evaluate_model(model_name, result_root="result"):
    """
    result/
      ├── llama3-8b/
      │     ├── data0.json
      │     ├── data1.json
      │     └── ...
      ├── qwen2-7b/
      │     ├── data0.json
      │     └── ...
    """

    model_dir = Path(result_root) / model_name/ 'cache'
    if not model_dir.exists():
        raise FileNotFoundError(f"cannot find: {model_dir}")

    all_data = []
    for file in model_dir.glob("*.json"):
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    all_data.append(json.loads(line))
                except Exception as e:
                    print(f"loading {file}, error: {e}")

    if not all_data:
        raise ValueError(f"Model {model_name} has no data to evaluate")

    results_by_task = defaultdict(list)
    for item in all_data:
        task_type = item["task_type"]
        results_by_task[task_type].append(item)
    

    metrics = {}
    for task, items in results_by_task.items():
        if task == "query_rewrite":
            print("Evaluating Query Rewrite")
            # === LLM as a judge ===
            total_scores = []
            for it in tqdm(items):
                history = it["conversations"][0]['value']
                gt_ans = it["gt_ans"]
                pred_ans = it["pred_ans"]

                JUDGE_PROMPT = """
                    Task description:
                    The system rewrites the user's last message using the conversation context to make it clearer and more complete, while preserving the original intent.

                    Evaluation criteria:
                    2 = Correct
                    The predicted rewrite is semantically equivalent to the reference rewrite. Minor wording differences are acceptable.

                    1 = Partially Correct
                    The predicted rewrite captures part of the meaning but misses important details or slightly changes the intent.

                    0 = Incorrect
                    The predicted rewrite changes the meaning, introduces wrong information, or fails to represent the user's intent.

                    Conversation History:
                    {conversation_history}

                    Reference Rewrite:
                    {gt_ans}

                    Predicted Rewrite:
                    {pred_ans}

                    Please evaluate the predicted rewrite. Only output the scores.

                    Output format (JSON only):
                    {{
                    "score": 0 | 1 | 2,
                    }}
                """

                prompt = JUDGE_PROMPT.format(
                    conversation_history=history,
                    gt_ans = gt_ans,
                    pred_ans = pred_ans
                )

                response = client.chat.completions.create(
                    model="qwen3-235b-a22b-instruct-2507-local",  # consistent with the vllm server
                    messages=[
                        {"role": "system", "content": "You are an expert evaluator for an e-commerce dialogue system. Your task is to judge whether a predicted rewritten query correctly preserves the meaning of a reference rewrite."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    max_tokens=16, 
                )

                result = response.choices[0].message.content
                score = extract_score(result)
                total_scores.append(score)

            metrics[task] = {
                "count": len(items),
                "llm_as_a_judge": float(sum(total_scores)/len(total_scores))
            }
        elif task == "RAG_QA":
            print("Evaluating RAG_QA")
            # === LLM as a judge ===
            total_scores = []
            for it in tqdm(items):
                knowledge = it["conversations"][0]['value']
                gt_ans = it["gt_ans"]
                pred_ans = it["pred_ans"]
                JUDGE_PROMPT = """
                    Evaluation criteria:

                    1. Intent Understanding
                    Does the answer correctly understand the user's request?

                    2. Knowledge Grounding
                    If reference knowledge is provided, does the answer correctly use it?
                    If no knowledge is provided, the answer should avoid hallucinating facts.

                    3. Helpfulness
                    Is the answer helpful, informative, and relevant to the user's request?

                    4. Style Compliance
                    Does the answer follow the system requirements?
                    - Friendly customer service tone
                    - Clear and polite response
                    - Emojis may be used

                    Scoring rules:

                    2 = Good
                    The answer correctly addresses the user request, is helpful, and follows the style requirements.

                    1 = Acceptable
                    The answer is partially helpful but has issues (missing details, weak grounding, or minor style problems).

                    0 = Poor
                    The answer is incorrect, irrelevant, unhelpful, or refuses to answer when it should respond.

                    Reference Knowledge and Question:
                    {knowledge}

                    Reference Answer:
                    {gt_ans}

                    Predicted Answer:
                    {pred_ans}

                    First briefly analyze the answer quality.

                    Then output the score.

                    Output JSON only:
                    {{
                    "score": 0|1|2
                    }}

                    Return ONLY valid JSON.
                    """

                prompt = JUDGE_PROMPT.format(
                    knowledge=knowledge,
                    gt_ans = gt_ans,
                    pred_ans = pred_ans
                )

                response = client.chat.completions.create(
                    model="qwen3-235b-a22b-instruct-2507-local",  # consistent with the vllm server
                    messages=[
                        {"role": "system", "content": "You are an expert evaluator for an e-commerce RAG (Retrieval-Augmented Generation) customer service system. Your task is to evaluate the quality of a model-generated answer."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    max_tokens=16,
                )

                result = response.choices[0].message.content
                score = extract_score(result)
                total_scores.append(score)

            metrics[task] = {
                "count": len(items),
                "llm_as_a_judge": float(sum(total_scores)/len(total_scores))
            }

        else:
            # skip other tasks
            continue

    # --- Output ---
    task_order = [
        "query_rewrite",
        "RAG_QA",
    ]
    save_path = model_dir / "llm_as_a_judge_evaluation_results.txt"
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"Evaluation Results for {model_name}\n")
        f.write("=" * 60 + "\n\n")
        for task in task_order:
            if task not in metrics:
                continue
            stat = metrics[task]
            if task in ["query_rewrite", "RAG_QA"]:
                line = f"{task.replace('_', '-').title():25s} | score = {stat['llm_as_a_judge']:.4f} ({stat['count']} samples)"
            else:
                continue
            print(line)
            f.write(line + '\n')
        f.write("\n" + "=" * 60 + "\n")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="model name")
    parser.add_argument("--result_root", type=str, required=True, help="path for generated result")
    args = parser.parse_args()

    evaluate_model(args.model_name, args.result_root)
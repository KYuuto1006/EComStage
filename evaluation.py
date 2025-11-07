import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import torch
import torch.nn.functional as F

def is_correct(pred_ans, gt_ans):
    if pred_ans is None or gt_ans is None:
        return False
    # remove possible suffix such as \n and \t
    pred_ans = str(pred_ans).replace("\n", " ").replace("\t", " ").strip()
    gt_ans = str(gt_ans).strip()
    return gt_ans in pred_ans


def evaluate_model(model_name, result_root="result"):
    """
    We expect you have prepared the result folder in which each model's generation results are under its corresponding model name folder. In our case, we split data into several sub json for simultaneous generation (acceleration), which is not necessary for all cases.
    result/
      ├── llama3-8b/
      │     ├── data0.json
      │     ├── data1.json
      │     └── ...
      ├── qwen2.5-7b/
      │     ├── data0.json
      │     └── ...
    """

    model_dir = Path(result_root) / model_name/ 
    if not model_dir.exists():
        raise FileNotFoundError(f"model name folder does not exist: {model_dir}")

    all_data = []
    for file in model_dir.glob("*.json"):
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                all_data.append(json.loads(line))


    if not all_data:
        raise ValueError(f" {model_name} has no data for evaluation.")

    # --- split by task type ---
    results_by_task = defaultdict(list)
    for item in all_data:
        task_type = item["task_type"]
        results_by_task[task_type].append(item)
    

    device = "cuda"
    embed_model = SentenceTransformer("Qwen/Qwen3-Embedding-8B", device=device) # load in embedding model

    metrics = {}
    for task, items in results_by_task.items():
        if task == "query_rewrite" or task == "RAG_QA":
            # === cosine similarity for query rewrite and ragqa tasks ===
            refs = [it["gt_ans"] for it in items]
            hyps = [it["pred_ans"] for it in items]
            ref_emb = embed_model.encode(refs, convert_to_tensor=True, device=device, show_progress_bar=True)
            hyp_emb = embed_model.encode(hyps, convert_to_tensor=True, device=device, show_progress_bar=True)
            cosine_scores = F.cosine_similarity(ref_emb, hyp_emb, dim=-1)
            metrics[task] = {
                "count": len(items),
                "cosine_sim": float(cosine_scores.mean().item())
            }
        else:
            # === accuracy for other tasks ===
            correct_ans = 0
            for it in items:
                if is_correct(it["pred_ans"], it["gt_ans"]):
                    correct_ans += 1
            acc = correct_ans / len(items)
            metrics[task] = {"count": len(items), "accuracy": acc}

    # --- output ---
    task_order = [
        "query_rewrite",
        "attitude_cls",
        "query_match",
        "intent_recognition",
        "scenario_route",
        "solution_decision",
        "RAG_QA",
    ]
    save_path = model_dir / "evaluation_results.txt" # save evaluation under the same folder for each model
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"Evaluation Results for {model_name}\n")
        f.write("=" * 60 + "\n\n")
        for task in task_order:
            if task not in metrics:
                continue
            stat = metrics[task]
            if task in ["query_rewrite", "RAG_QA"]:
                line = f"{task.replace('_', '-').title():25s} | COS = {stat['cosine_sim']:.4f} ({stat['count']} samples)"
            else:
                line = f"{task.replace('_', '-').title():25s} | ACC  = {stat['accuracy']:.4f} ({stat['count']} samples)"
            print(line)
            f.write(line + '\n')
        f.write("\n" + "=" * 60 + "\n")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="model name folder under result")
    parser.add_argument("--result_root", type=str, default="/mnt/ali-sh-1/dataset/redone/mengzijie/Agent/result", help="result path")
    args = parser.parse_args()

    evaluate_model(args.model_name, args.result_root)
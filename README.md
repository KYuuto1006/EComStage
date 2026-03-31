# EComStage

This repository contains the codes and data for our paper:
"EComStage: Stage-wise and Orientation-specific Benchmarking for Large Language Models in E-commerce".

~~Data and codes are currently under internal company review and will be uploaded once it is finished.~~

Data is now available. One can refer to `filtered_data.json` for reproduction.

### TLDR
EComstage is a stage-wise evaluation framework for LLMs that serve as the backbone for e-commerce agents. Instead of directly judging whether the agent successfully completes the final task, EComStage decomposes the reasoning process into separate, stage-specific tasks, each designed to evaluate a distinct capability of the agent-capable LLM.  


### Released Data Description
We collect business data from real e-commerce service scenarios and apply thorough filtering on them.
All the data for seven tasks are merged in `filtered_data.json`. You can separate them by `"id"` and `"task_type"`.


Data Structure:

```
{
    "id": "" # id for this data sample
    "task_type": "" # type of task
    "system": "" # system prompt, different for each task
    "conversations": [
      {
        "from": "human"
        "value": "" # conversation history between human and customer service or user query and provided lists, based on task type
      },
      {
        "from": "gpt", # a placeholder for LLM
        "value": "1" # a placeholder for LLM generated answer
      }
    ],
    "gt_answer": "1" # golden truth answer from human anotator 
  }
```

Note that for Query Match and Solution Decision, human annotators provide pre-defined question lists and response options tailored to each data sample.

For RAG-QA, we plan to release an anonymized version of our data base. Currently, the referenced knowledge is replaced with a placeholder.

### Response generation
We do not restrict any generation methods. You can use models from Huggingface or API to generate results, given the system prompts and data samples in `filtered_data.json`. However, we suggest you to save the generation results in the following structure for smooth evaluation:


```
{
    "id": "" # id for this data sample
    "task_type": "" # type of task
    "system": "" # system prompt, different for each task
    "conversations": [
      {
        "from": "human"
        "value": "" # conversation history between human and customer service or user query and provided lists, based on task type
      },
      {
        "from": "gpt", # a placeholder for LLM
        "value": "1" # a placeholder for LLM generated answer
      }
    ],
    "pred_ans": "" ##### ⭐⭐⭐ save generation here!!!
    "gt_answer": "1" # golden truth answer from human anotator 
  }
```


### Evaluation
Codes for objective evaluation (accuracy, embedding) is provided in `evaluation.py`.

Before evaluation, we expect you have prepared the result folder in which each model's generation results are under its corresponding model name folder. In our case, we split data into several sub json for simultaneous generation (acceleration), which is not necessary for all cases:
```
    result/
      ├── qwen3-8b/
      │     ├── data0.json
      │     ├── data1.json
      │     └── ...
      ├── qwen2.5-7b/
      │     ├── data0.json
      │     └── ...
```

Codes for subjective evaluation (llm-as-a-judge) is provided in `result_eval_llmasajudge.py`. We use vLLM and Qwen3-235B-A22B-Instruct-2507 in our implementation. You can use other models with minimum modification.


If you follow our instructions, you can smoothly start evaluation.

  

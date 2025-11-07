# EComStage

This repository contains the supplementary materials for our paper:
"Stage-wise and Orientation-specific Benchmarking of LLM-based Agents in E-commerce", submitted to ICDE 2026.

~~Data and codes are currently under internal company review and will be uploaded once it is finished. (before the conference's review process ends)~~


Data is now available. One can refer to `filtered_data.json` for reproduction.

### TLDR
EComstage is a stage-wise evaluation framework for LLM-based agents in e-commerce scenarios. Instead of directly judging whether the agent successfully completes the final task, EComStage decomposes the reasoning process into separate, stage-specific tasks, each designed to evaluate a distinct capability of the agent.  


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


  

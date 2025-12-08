"""Stage 1 数据处理工具：提示加载、生成与打标。"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import List, Tuple

import torch
from datasets import load_dataset

from contrast_analyze.utils.pipeline_config import PipelineConfig
from refusal_direction.pipeline.model_utils.model_factory import construct_model_base


def load_prompts(split: str, data_type: str, n_samples: int | None) -> List[dict]:
    """
    加载提示数据。
    
    - 有害数据：从 TRIDENT-Core.jsonl 和 TRIDENT-Edge.jsonl 加载
    - 无害数据：从 tatsu-lab/alpaca 数据集加载
    """
    if data_type == "harmful":
        # 从本地 JSONL 文件加载有害数据
        base_path = Path("/root/geometry-of-refusal/contrast_analyze/utils/data")
        core_file = base_path / "TRIDENT-Core.jsonl"
        edge_file = base_path / "TRIDENT-Edge.jsonl"
        
        prompts = []
        
        # 加载 TRIDENT-Core.jsonl
        if core_file.exists():
            with open(core_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        if "instruction" in item:
                            prompts.append({"instruction": item["instruction"], "category": None})
        
        # 加载 TRIDENT-Edge.jsonl
        if edge_file.exists():
            with open(edge_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        if "instruction" in item:
                            prompts.append({"instruction": item["instruction"], "category": None})
        
        if n_samples is not None and n_samples > 0:
            prompts = prompts[:n_samples]
        
        return prompts
    
    elif data_type == "benign":
        # 从 alpaca 数据集加载无害数据
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        prompts = []
        
        for ex in ds:
            instruction = ex.get("instruction", "")
            input_text = ex.get("input", "")
            
            # 构造 Prompt：如果有 input 且不为空，则拼接；否则只使用 instruction
            if input_text and input_text.strip():
                prompt_text = f"{instruction}\n\n{input_text}"
            else:
                prompt_text = instruction
            
            prompts.append({"instruction": prompt_text, "category": None})
        
        if n_samples is not None and n_samples > 0:
            prompts = prompts[:n_samples]
        
        return prompts
    
    else:
        raise ValueError(f"未知的数据类型: {data_type}")


def generate_completions(model_alias: str, prompts: List[dict], batch_size: int):
    model_base = construct_model_base(model_alias)
    completions = model_base.generate_completions(prompts, batch_size=batch_size)
    del model_base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return completions


def split_harmful(completions: List[dict]) -> Tuple[List[dict], List[dict]]:
    from refusal_direction.pipeline.submodules.evaluate_jailbreak import harmbench_judge_fn

    prompts_list = [c["prompt"] for c in completions]
    responses_list = [c["response"] for c in completions]
    classifications = harmbench_judge_fn(prompts_list, responses_list)

    success_data, failure_data = [], []
    for idx, (completion, cls) in enumerate(zip(completions, classifications), start=1):
        record = {
            "instruction": completion["prompt"],
            "response": completion["response"],
            "category": "harmful",
            "metadata": {"id": idx, "raw_label": "success" if cls == 1 else "fail"},
        }
        (success_data if cls == 1 else failure_data).append(record)
    return success_data, failure_data


def save_jsonl(path, items: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def run_stage1(config: PipelineConfig, *, model_alias: str, split: str, data_type: str, batch_size: int, n_samples: int | None):
    if data_type == "benign":
        # 无害数据：检查共享位置是否已存在，如果存在则跳过
        output_file = config.get_contrast_data_file("benign", shared=True)
        
        if output_file.exists():
            # 文件已存在，读取并返回统计信息
            count = 0
            with open(output_file, "r", encoding="utf-8") as f:
                for _ in f:
                    count += 1
            return {"benign": count, "harmful_success": 0, "harmful_failure": 0}
        
        # 文件不存在，生成并保存到共享位置
        prompts = load_prompts(split, data_type, n_samples)
        benign_items = [
            {
                "instruction": p["instruction"],
                "response": "",  # benign 数据不需要 response
                "category": "benign",
                "metadata": {"id": idx, "raw_label": "benign"},
            }
            for idx, p in enumerate(prompts, start=1)
        ]
        save_jsonl(output_file, benign_items)
        return {"benign": len(benign_items), "harmful_success": 0, "harmful_failure": 0}

    # 有害数据：加载并合并 TRIDENT-Core 和 TRIDENT-Edge，然后生成回答并判断
    prompts = load_prompts(split, data_type, n_samples)
    completions = generate_completions(model_alias, prompts, batch_size)
    success_data, failure_data = split_harmful(completions)
    success_file = config.get_contrast_data_file("harmful_success")
    failure_file = config.get_contrast_data_file("harmful_failure")
    save_jsonl(success_file, success_data)
    save_jsonl(failure_file, failure_data)
    return {"benign": 0, "harmful_success": len(success_data), "harmful_failure": len(failure_data)}


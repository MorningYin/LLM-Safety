"""Stage 1 数据处理流程。"""

import gc
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from datasets import load_dataset

from contrast_analyze.utils.pipeline_config import PipelineConfig
from contrast_analyze.utils.data.dataset_io import save_jsonl
from refusal_direction.pipeline.model_utils.model_factory import construct_model_base


def _load_harmful_prompts(n_samples: Optional[int] = None) -> List[Dict]:
    """从 TRIDENT 文件加载有害提示数据。"""
    data_dir = Path(__file__).parent
    core_file = data_dir / "TRIDENT-Core.jsonl"
    edge_file = data_dir / "TRIDENT-Edge.jsonl"
    
    prompts = []
    
    if core_file.exists():
        with open(core_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if "instruction" in item:
                        # 保存 instruction 和 categorie（注意拼写）
                        prompt_data = {"instruction": item["instruction"]}
                        if "categorie" in item:
                            prompt_data["categorie"] = item["categorie"]
                        prompts.append(prompt_data)
    
    if edge_file.exists():
        with open(edge_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if "instruction" in item:
                        # 保存 instruction 和 categorie（注意拼写）
                        prompt_data = {"instruction": item["instruction"]}
                        if "categorie" in item:
                            prompt_data["categorie"] = item["categorie"]
                        prompts.append(prompt_data)
    
    if n_samples is not None and n_samples > 0:
        prompts = prompts[:n_samples]
    
    return prompts


def _load_benign_prompts(n_samples: Optional[int] = None) -> List[Dict]:
    """从 alpaca 数据集加载无害提示数据。"""
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    prompts = []
    
    for ex in ds:
        instruction = ex.get("instruction", "")
        input_text = ex.get("input", "")
        
        if input_text and input_text.strip():
            prompt_text = f"{instruction}\n\n{input_text}"
        else:
            prompt_text = instruction
        
        prompts.append({"instruction": prompt_text})
        
        if n_samples is not None and len(prompts) >= n_samples:
            break
    
    return prompts


def _cleanup_model(model_base):
    """优雅地清理模型，释放所有显存和内存。
    
    该函数会按顺序执行以下清理步骤：
    1. 调用模型的清理方法（如果存在）
    2. 删除模型和tokenizer对象
    3. 删除所有缓存的模块引用
    4. 删除整个model_base对象
    5. 清理CUDA缓存并同步
    6. 多次垃圾回收以确保彻底清理
    """
    if model_base is None:
        return
    
    try:
        # 1. 先调用模型的清理方法（如果存在）
        if hasattr(model_base, 'del_model'):
            try:
                model_base.del_model()
            except Exception as e:
                print(f"[Warning] 调用 del_model() 时出现异常: {e}")
        
        # 2. 删除模型和tokenizer
        if hasattr(model_base, 'model') and model_base.model is not None:
            del model_base.model
        if hasattr(model_base, 'tokenizer') and model_base.tokenizer is not None:
            del model_base.tokenizer
        
        # 3. 删除所有缓存的模块引用
        for attr_name in ['model_block_modules', 'model_attn_modules', 'model_mlp_modules',
                          'tokenize_instructions_fn', 'eoi_toks', 'refusal_toks']:
            if hasattr(model_base, attr_name):
                try:
                    delattr(model_base, attr_name)
                except Exception:
                    pass  # 忽略删除属性时的异常
        
        # 4. 删除整个model_base对象
        del model_base
        
    except Exception as e:
        # 即使清理过程中出错，也要继续执行后续清理
        print(f"[Warning] 清理模型时出现异常: {e}")
    
    # 5. 强制清理显存
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # 等待所有CUDA操作完成
        except Exception as e:
            print(f"[Warning] 清理CUDA缓存时出现异常: {e}")
    
    # 6. 多次垃圾回收以确保彻底清理
    for _ in range(3):
        gc.collect()


def _generate_completions(
    model_alias: str,
    prompts: List[Dict],
    batch_size: int,
) -> List[Dict]:
    """生成模型补全。"""
    # 准备数据集：将 categorie 映射到 category（如果存在）
    dataset = []
    for p in prompts:
        item = {"instruction": p["instruction"]}
        # 如果有 categorie，使用它作为 category；否则使用 "harmful"
        if "categorie" in p:
            item["category"] = p["categorie"]
        else:
            item["category"] = "harmful"
        dataset.append(item)
    
    # 删除 prompts 释放内存
    del prompts
    
    model_base = construct_model_base(model_alias)
    try:
        completions = model_base.generate_completions(
            dataset,
            batch_size=batch_size,
            max_new_tokens=128,
        )
        # 将 categorie 字段添加到返回结果中
        for i, completion in enumerate(completions):
            if "categorie" in dataset[i]:
                completion["categorie"] = dataset[i]["categorie"]
        
        # 删除 dataset 释放内存
        del dataset
    finally:
        # 使用优雅的清理函数释放模型
        _cleanup_model(model_base)
    
    return completions


def _classify_harmful(completions: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """根据judge函数对补全结果进行分类。"""
    from refusal_direction.pipeline.submodules.evaluate_jailbreak import harmbench_judge_fn

    prompts_list = [c["prompt"] for c in completions]
    responses_list = [c["response"] for c in completions]
    classifications = harmbench_judge_fn(prompts_list, responses_list)

    success_data = []
    failure_data = []
    
    for idx, (completion, cls) in enumerate(zip(completions, classifications), start=1):
        # 从原始 prompt 数据中获取 categorie，如果没有则使用 "harmful"
        categorie = completion.get("categorie", "harmful")
        
        record = {
            "instruction": completion["prompt"],
            "response": completion["response"],
            "category": categorie,  # 使用数据中的 categorie
            "metadata": {
                "id": idx,
                "raw_label": "success" if cls == 1 else "fail",
            },
        }
        
        if cls == 1:
            success_data.append(record)
        else:
            failure_data.append(record)
    
    return success_data, failure_data


def _count_jsonl_lines(path: Path) -> int:
    """统计 JSONL 文件的行数。"""
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            count += 1
    return count


def _process_benign_data(
    config: PipelineConfig,
    n_samples: Optional[int],
) -> Dict[str, int]:
    """处理无害数据。"""
    output_file = config.get_contrast_data_file("benign", shared=True)
    
    if output_file.exists():
        count = _count_jsonl_lines(output_file)
        return {"benign": count, "harmful_success": 0, "harmful_failure": 0}
    
    prompts = _load_benign_prompts(n_samples=n_samples)
    items = [
        {
            "instruction": p["instruction"],
            "response": "",
            "category": "benign",
            "metadata": {"id": idx, "raw_label": "benign"},
        }
        for idx, p in enumerate(prompts, start=1)
    ]
    
    save_jsonl(output_file, items)
    return {"benign": len(items), "harmful_success": 0, "harmful_failure": 0}


def _process_harmful_data(
    config: PipelineConfig,
    model_alias: str,
    batch_size: int,
    n_samples: Optional[int],
) -> Dict[str, int]:
    """处理有害数据。"""
    prompts = _load_harmful_prompts(n_samples=n_samples)
    completions = _generate_completions(model_alias, prompts, batch_size)
    success_data, failure_data = _classify_harmful(completions)
    
    success_file = config.get_contrast_data_file("harmful_success")
    failure_file = config.get_contrast_data_file("harmful_failure")
    save_jsonl(success_file, success_data)
    save_jsonl(failure_file, failure_data)
    
    return {
        "benign": 0,
        "harmful_success": len(success_data),
        "harmful_failure": len(failure_data),
    }


def run_stage1(
    config: PipelineConfig,
    *,
    model_alias: str,
    data_type: str,
    batch_size: int,
    n_samples: Optional[int] = None,
) -> Dict[str, int]:
    """运行 Stage 1 数据处理流程。
    
    Args:
        config: Pipeline 配置对象
        model_alias: 模型别名
        data_type: 数据类型（"harmful" 或 "benign"）
        batch_size: 批量推理大小
        n_samples: 限制获取的样本数量（None 表示使用全部）
        
    Returns:
        统计信息字典，包含 'benign', 'harmful_success', 'harmful_failure' 键
    """
    if data_type == "benign":
        return _process_benign_data(config, n_samples)
    elif data_type == "harmful":
        return _process_harmful_data(config, model_alias, batch_size, n_samples)
    else:
        raise ValueError(f"未知的数据类型: {data_type}，必须是 'harmful' 或 'benign'")

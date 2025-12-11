# -*- coding: utf-8 -*-
"""
dataset_loader.py

统一的数据集加载接口。
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from contrast_analyze.utils.common import get_contrast_data_root
from contrast_analyze.utils.pipeline_config import PipelineConfig
from contrast_analyze.utils.direction_utils import resolve_eoi_marker


# ============================================================
# 数据集类别映射
# ============================================================

CATEGORY_MAP = {
    "benign": ("success", "benign"),
    "harmful_success": ("success", "harmful"),
    "harmful_failure": ("failure", "harmful"),
}


# ============================================================
# 数据集加载
# ============================================================

def load_dataset_jsonl(
    path: Path,
    max_samples: Optional[int] = None,
    seed: int = 2025,
) -> List[Dict]:
    """从 JSONL 文件加载数据集。
    
    Args:
        path: JSONL 文件路径
        max_samples: 最大样本数（None 表示使用全部）
        seed: 随机种子（用于采样）
        
    Returns:
        数据集列表，每个元素包含 'instruction' 字段
    """
    if not path.exists():
        raise FileNotFoundError(f"未找到数据文件：{path}")

    with open(path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    if not rows:
        raise ValueError(f"{path} 为空")

    # 采样
    if max_samples is not None and max_samples > 0 and len(rows) > max_samples:
        rng = random.Random(seed)
        rng.shuffle(rows)
        rows = rows[:max_samples]

    dataset = []
    for idx, row in enumerate(rows):
        instruction = row.get("instruction") or row.get("prompt")
        if not instruction:
            continue
        meta = row.get("metadata", {})
        dataset.append({
            "instruction": instruction,
            "meta_id": meta.get("id", idx),
            "raw_label": meta.get("raw_label"),
        })

    if not dataset:
        raise ValueError("数据集中没有有效的 instruction")
    return dataset


# 别名：为了向后兼容
load_jsonl = load_dataset_jsonl


def save_jsonl(path, items: List[dict]) -> None:
    """保存数据到 JSONL 文件。
    
    Args:
        path: 输出文件路径
        items: 要保存的数据列表
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_contrast_data(
    model_base,
    data_type: str,
    split: str,
    harmtype: str,
) -> Tuple[List[str], List[str], str, List]:
    """
    从 contrast_data 目录加载对比数据集。
    
    Args:
        model_base: ModelBase 实例，用于获取 marker
        data_type: 数据类型（"success" 或 "failure"）
        split: 数据集分割（"vanilla" 或 "adversarial"）
        harmtype: 伤害类型（"benign" 或 "harmful"）
        
    Returns:
        (prompts, substrings, marker, labels) 元组
    """
    # 从 model_base 获取 model_alias
    # model_base 在构造时保存了 model_name_or_path
    model_path = model_base.model_name_or_path
    
    # 从 model_path 反推 model_alias
    from pipeline.model_utils.model_factory import model_alias_to_model_name
    model_alias = None
    for alias, path in model_alias_to_model_name.items():
        if path == model_path:
            model_alias = alias
            break
    
    if model_alias is None:
        raise ValueError(f"无法从 model_path {model_path} 反推 model_alias，请使用 load_dataset_by_category 函数")
    
    model_dir = get_contrast_data_root() / model_alias
    
    if harmtype == "benign":
        filename = f"{split}_benign.jsonl"
    else:
        filename = (
            f"{split}_{harmtype}_success.jsonl"
            if data_type == "success"
            else f"{split}_{harmtype}_failure.jsonl"
        )
    path = model_dir / filename

    prompts = []
    labels = []
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                j = json.loads(line)
                prompts.append(j.get("instruction", j.get("prompt")))
                raw_label = j.get("metadata", {}).get("raw_label") or j.get("label")
                labels.append(raw_label)

    # 获取 EOI marker（从 model_base 中获取）
    marker = resolve_eoi_marker(model_base)
    substrings = [marker] * len(prompts)
    
    return prompts, substrings, marker, labels


def load_dataset_by_category(
    model_alias: str,
    split: str,
    category: str,
    max_samples: Optional[int] = None,
    seed: int = 2025,
) -> List[Dict]:
    """按类别加载数据集（统一接口，不再区分 split，统一使用 vanilla 数据）。
    
    Args:
        model_alias: 模型别名
        split: 数据集分割（已废弃，保留以兼容旧代码，实际不再使用）
        category: 类别名称（"benign", "harmful_success", "harmful_failure"）
        max_samples: 最大样本数（None 表示使用全部）
        seed: 随机种子
        
    Returns:
        数据集列表，每个元素包含 'instruction' 字段
    """
    if category not in CATEGORY_MAP:
        raise ValueError(f"未知类别：{category}，必须是 {list(CATEGORY_MAP.keys())} 之一")
    
    data_type, harmtype = CATEGORY_MAP[category]
    
    # 优先使用 PipelineConfig 的路径，回退到旧的路径
    try:
        config = PipelineConfig(model_alias=model_alias)
        model_dir = config.contrast_data_dir
    except Exception:
        model_dir = get_contrast_data_root() / model_alias
    
    # 不再使用 split，统一使用新的文件名格式
    if harmtype == "benign":
        filename = "benign.jsonl"
    else:
        filename = (
            f"{harmtype}_success.jsonl"
            if data_type == "success"
            else f"{harmtype}_failure.jsonl"
        )
    
    path = model_dir / filename
    
    # 如果新路径不存在，尝试旧的路径格式（兼容性）
    if not path.exists():
        # 尝试 vanilla 格式
        if harmtype == "benign":
            old_filename = "vanilla_benign.jsonl"
        else:
            old_filename = (
                f"vanilla_{harmtype}_success.jsonl"
                if data_type == "success"
                else f"vanilla_{harmtype}_failure.jsonl"
            )
        old_path = model_dir / old_filename
        if old_path.exists():
            path = old_path
    
    return load_dataset_jsonl(path, max_samples, seed)


def load_all_categories(
    model_alias: str,
    split: str,
    max_samples: Optional[int] = None,
    seed: int = 2025,
) -> Dict[str, List[Dict]]:
    """加载所有类别的数据集。
    
    Args:
        model_alias: 模型别名
        split: 数据集分割
        max_samples: 每个类别的最大样本数
        seed: 随机种子
        
    Returns:
        字典，键为类别名，值为数据集列表
    """
    categories = ["benign", "harmful_success", "harmful_failure"]
    datasets = {}
    
    for cat in categories:
        try:
            datasets[cat] = load_dataset_by_category(
                model_alias, split, cat, max_samples, seed
            )
        except FileNotFoundError:
            # 如果某个类别不存在，跳过
            continue
    
    return datasets


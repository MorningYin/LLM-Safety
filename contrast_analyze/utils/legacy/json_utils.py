# -*- coding: utf-8 -*-
"""
json_utils.py

JSON 文件操作工具。
"""

import json
from pathlib import Path
from typing import Dict, Tuple

import torch

from contrast_analyze.utils.common import get_runs_root


def save_json(path: Path, data: Dict):
    """保存 JSON 文件。
    
    Args:
        path: 文件路径
        data: 要保存的数据
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Dict:
    """加载 JSON 文件。
    
    Args:
        path: 文件路径
        
    Returns:
        解析后的字典
    """
    if not path.exists():
        raise FileNotFoundError(f"未找到文件：{path}")
    
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_stats_json(path: Path, stat_key: str) -> Tuple[torch.Tensor, list]:
    """加载统计文件，返回目标值和 token 标签。
    
    Args:
        path: 统计文件路径
        stat_key: 统计键名（如 "harmful_success", "harmful_failure"）
        
    Returns:
        (targets, token_labels) 元组
    """
    stats = load_json(path)
    
    token_labels = stats.get("token_labels")
    if not token_labels:
        raise ValueError("统计文件中缺少 token_labels")
    
    stat_group = stats.get(stat_key)
    if not stat_group:
        raise ValueError(f"统计文件中缺少 {stat_key} 数据")
    
    mean_dot = stat_group.get("mean_dot")
    if mean_dot is None:
        raise ValueError(f"{stat_key} 数据缺少 mean_dot")
    
    targets = torch.tensor(mean_dot, dtype=torch.float32)
    return targets, token_labels


def get_stats_path(
    model_alias: str,
    split: str,
    direction_mode: str,
) -> Path:
    """获取统计文件路径。
    
    Args:
        model_alias: 模型别名
        split: 数据集分割
        direction_mode: 方向模式
        
    Returns:
        统计文件路径
    """
    runs_root = get_runs_root()
    if direction_mode == "directions":
        return runs_root / model_alias / "contrast_directions" / f"{split}_combined_feature_stats.json"
    else:
        return runs_root / model_alias / "contrast_direction" / f"{split}_combined_direction_stats.json"


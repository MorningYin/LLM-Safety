# -*- coding: utf-8 -*-
"""
plan_utils.py

实验计划解析工具。
"""

import json
from pathlib import Path
from typing import Dict, Tuple

from contrast_analyze.utils.direction_utils import normalize_token_index
from contrast_analyze.utils.json_utils import load_json


def parse_plan_json(path: Path) -> Dict[str, Dict]:
    """解析 plan 配置文件。
    
    Args:
        path: plan.json 文件路径
        
    Returns:
        计划字典，键为计划名，值为计划配置
    """
    plans = load_json(path)
    
    if not isinstance(plans, dict) or not plans:
        raise ValueError("plan.json 必须是非空字典")
    
    return plans


def parse_plan_config(
    plan_name: str,
    plan_config: Dict,
    n_layers: int,
    n_tokens: int,
) -> Tuple[Dict[int, Tuple[int, ...]], float]:
    """解析 plan 配置，生成层到 token 索引的映射并返回 alpha 系数。
    
    Args:
        plan_name: 计划名称
        plan_config: 计划配置字典
        n_layers: 模型层数
        n_tokens: token 数量
        
    Returns:
        (layer_token_map, alpha) 元组
    """
    if not isinstance(plan_config, dict) or not plan_config:
        raise ValueError(f"计划 {plan_name} 必须是非空字典")
    
    alpha = float(plan_config.get("alpha", 1.0))
    
    layer_token_map: Dict[int, Tuple[int, ...]] = {}
    for layer_key, token_list in plan_config.items():
        if layer_key == "alpha":
            continue
        
        layer_idx = int(layer_key)
        if layer_idx < 0 or layer_idx >= n_layers:
            raise ValueError(f"计划 {plan_name} 中层 {layer_idx} 超出模型层数 {n_layers}")
        
        if not isinstance(token_list, list) or not token_list:
            raise ValueError(f"计划 {plan_name} 中层 {layer_idx} 没有提供 token 列表")
        
        normalized = []
        seen = set()
        for token_idx in token_list:
            norm_idx = normalize_token_index(token_idx, n_tokens)
            if norm_idx not in seen:
                normalized.append(norm_idx)
                seen.add(norm_idx)
        
        if normalized:
            layer_token_map[layer_idx] = tuple(normalized)
    
    if not layer_token_map:
        raise ValueError(f"计划 {plan_name} 未指定任何层/位置")
    
    return layer_token_map, alpha


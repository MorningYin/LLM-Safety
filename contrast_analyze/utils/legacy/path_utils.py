# -*- coding: utf-8 -*-
"""
path_utils.py

路径相关的工具函数：解析路径字符串、从 CSV 读取路径等。
纯函数，无副作用，可复用。
"""

import csv
from pathlib import Path
from typing import Dict, List, Tuple


def parse_path_string(path_str: str) -> Tuple[int, int, int, int]:
    """解析路径字符串。
    
    底层逻辑：纯函数，无副作用。
    
    Args:
        path_str: 路径字符串，格式如 "P_L10_T4,P_L11_T4"
        
    Returns:
        (start_layer, start_token, end_layer, end_token) 元组
        例如 "P_L10_T4,P_L11_T4" -> (10, 4, 11, 4)
    """
    parts = path_str.split(",")
    if len(parts) != 2:
        raise ValueError(f"路径格式错误：{path_str}，应为 'P_L<layer>_T<token>,P_L<layer>_T<token>'")
    
    def parse_node(node_str: str) -> Tuple[int, int]:
        """解析单个节点，如 'P_L10_T4' -> (10, 4)"""
        node_str = node_str.strip()
        if not node_str.startswith("P_L") or "_T" not in node_str:
            raise ValueError(f"节点格式错误：{node_str}，应为 'P_L<layer>_T<token>'")
        
        # 提取层和 token 索引
        # 格式：P_L<layer>_T<token>
        try:
            layer_part = node_str.split("_T")[0]  # "P_L10"
            token_part = node_str.split("_T")[1]    # "4"
            
            layer = int(layer_part[3:])  # 去掉 "P_L"
            token = int(token_part)
            
            return layer, token
        except (ValueError, IndexError) as e:
            raise ValueError(f"无法解析节点：{node_str}") from e
    
    start_node, end_node = parts
    start_layer, start_token = parse_node(start_node)
    end_layer, end_token = parse_node(end_node)
    
    return start_layer, start_token, end_layer, end_token


def load_paths_from_csv(
    csv_path: Path,
    start_layer_min: int = 10,
    start_layer_max: int = 16,
) -> List[Dict]:
    """从 net_effects.csv 读取指定起点层范围内的路径。
    
    底层逻辑：纯函数，无副作用。
    
    Args:
        csv_path: net_effects.csv 文件路径
        start_layer_min: 起点层最小值（包含）
        start_layer_max: 起点层最大值（包含）
        
    Returns:
        路径列表，每个元素包含：
        - start_layer: 起点层索引
        - start_token: 起点 token 索引
        - end_layer: 终点层索引
        - end_token: 终点 token 索引
        - net_effect: net_effect 值（float）
        - beta_fail: beta_fail 值（float）
        - beta_benign: beta_benign 值（float）
        - p_value: p_value 值（float）
        - is_significant: 是否显著（bool）
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"未找到 CSV 文件：{csv_path}")
    
    paths = []
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            source = row.get("source", "").strip()
            target = row.get("target", "").strip()
            
            if not source or not target:
                continue
            
            # 解析路径
            try:
                start_layer, start_token, end_layer, end_token = parse_path_string(f"{source},{target}")
            except ValueError as e:
                print(f"[警告] 跳过无效路径：{source},{target}，错误：{e}")
                continue
            
            # 检查起点层是否在范围内
            if start_layer < start_layer_min or start_layer > start_layer_max:
                continue
            
            # 解析其他字段
            try:
                net_effect = float(row.get("net_effect", 0))
                beta_fail = float(row.get("beta_fail", 0))
                beta_benign = float(row.get("beta_benign", 0))
                p_value = float(row.get("p_value", 1.0))
                is_significant = row.get("is_significant", "False").lower() == "true"
            except (ValueError, TypeError) as e:
                print(f"[警告] 跳过路径 {source},{target}，无法解析数值字段：{e}")
                continue
            
            paths.append({
                "start_layer": start_layer,
                "start_token": start_token,
                "end_layer": end_layer,
                "end_token": end_token,
                "net_effect": net_effect,
                "beta_fail": beta_fail,
                "beta_benign": beta_benign,
                "p_value": p_value,
                "is_significant": is_significant,
            })
    
    return paths


# -*- coding: utf-8 -*-
"""
direction_utils.py

方向向量相关的底层工具函数：加载、归一化、模式解析等。
纯函数，无副作用，可复用。
"""

from pathlib import Path
from typing import Optional

import torch

# ============================================================
# 注意：EOI_MAP 已移除，统一使用 model_utils 中定义的接口
# 所有 marker 都应该通过 resolve_eoi_marker(model_base) 获取
# ============================================================


def normalize_direction(vec: torch.Tensor) -> torch.Tensor:
    """归一化方向向量（通用）。
    
    底层逻辑：纯函数，无副作用。
    
    Args:
        vec: 方向向量，可以是任意维度
        
    Returns:
        归一化后的方向向量（1D）
    """
    if vec.ndim > 1:
        vec = vec.reshape(-1)
    norm = vec.norm()
    if norm == 0:
        raise ValueError("方向向量范数为 0，无法归一化")
    return (vec / norm).float()


def load_direction_global(path: Path) -> torch.Tensor:
    """加载并归一化全局方向向量。
    
    底层逻辑：纯函数，无副作用。
    
    Args:
        path: 全局 direction 文件路径（通常是 direction.pt）
        
    Returns:
        归一化后的全局方向向量，shape (d_model,)
    """
    if not path.exists():
        raise FileNotFoundError(f"未找到全局 direction 文件：{path}")
    vec = torch.load(path, map_location="cpu")
    return normalize_direction(vec)


def load_direction_candidates(path: Path) -> torch.Tensor:
    """加载并归一化候选方向向量。
    
    底层逻辑：纯函数，无副作用。
    
    Args:
        path: 候选方向文件路径（通常是 candidate_directions.pt）
        
    Returns:
        归一化后的候选方向张量，shape (n_tokens, n_layers, d_model)
    """
    if not path.exists():
        raise FileNotFoundError(f"未找到候选方向文件：{path}")
    tensor = torch.load(path, map_location="cpu")
    if tensor.ndim != 3:
        raise ValueError(f"候选方向张量 shape 异常：{tensor.shape}，期望 3D (n_tokens, n_layers, d_model)")
    tensor = tensor.float()
    norm = tensor.norm(dim=-1, keepdim=True)
    return tensor / (norm + 1e-8)


def resolve_eoi_marker(model_base) -> str:
    """从 model_base 中获取 EOI 标记。
    
    统一使用 model_utils 中定义的接口，确保整个 pipeline 的一致性。
    从模型的 chat template 中提取 marker，与模型定义保持一致。
    
    Args:
        model_base: ModelBase 实例，必须提供
        
    Returns:
        EOI 标记字符串
        
    Raises:
        ValueError: 如果 model_base 为 None 或无法获取 marker
    """
    if model_base is None:
        raise ValueError("model_base 参数是必需的，请传入 ModelBase 实例")
    
    try:
        # 获取 EOI tokens，然后解码为字符串
        eoi_tokens = model_base._get_eoi_toks()
        marker = model_base.tokenizer.decode(eoi_tokens)
        return marker
    except Exception as e:
        raise ValueError(f"无法从 model_base 获取 marker: {e}") from e


def normalize_token_index(idx: int, length: int) -> int:
    """规范化token索引（支持负数）。
    
    底层逻辑：纯函数，无副作用。
    
    Args:
        idx: token 索引（支持负数，-1 表示最后一个）
        length: 序列长度
        
    Returns:
        规范化后的索引（0 到 length-1）
    """
    if idx < 0:
        idx = length + idx
    if idx < 0 or idx >= length:
        raise ValueError(f"token索引越界：{idx}（长度 {length}）")
    return idx

# -*- coding: utf-8 -*-
"""
common.py

公共基础设施：环境设置、路径管理、目录创建等通用功能。
"""

import os
from pathlib import Path
from typing import Optional

from .pipeline_core import configure_hf_cache, seed_everything, setup_environment


# ============================================================
# 路径常量
# ============================================================

def get_project_root() -> Path:
    """获取项目根目录（contrast_analyze 目录）。"""
    return Path(__file__).resolve().parent.parent


def get_runs_root() -> Path:
    """
    获取 runs 根目录。
    
    默认使用数据盘路径：/root/autodl-tmp/contrast_analyze/runs
    可通过环境变量 CONTRAST_ANALYZE_RUNS_ROOT 自定义路径。
    
    Returns:
        runs 根目录路径
    """
    # 优先使用环境变量自定义路径
    custom_root = os.environ.get("CONTRAST_ANALYZE_RUNS_ROOT")
    if custom_root:
        return Path(custom_root)
    
    # 默认使用数据盘路径
    return Path("/root/autodl-tmp/contrast_analyze/runs")


def get_pipeline_runs_root() -> Path:
    """获取 pipeline runs 根目录。"""
    return Path("/root/refusal_direction/pipeline/runs")


def get_contrast_data_root() -> Path:
    """获取 contrast_data 根目录。"""
    return get_project_root() / "contrast_data"


def get_artifact_root(model_alias: str) -> Path:
    """获取方向向量 artifacts 目录。"""
    return get_runs_root() / model_alias / "direction_artifacts"


# ============================================================
# 目录操作
# ============================================================

def ensure_dir(path: Path) -> Path:
    """确保目录存在，如果不存在则创建。
    
    Args:
        path: 目录路径
        
    Returns:
        目录路径（用于链式调用）
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


# ============================================================
# 输出路径构建
# ============================================================

def get_output_dir(
    model_alias: str,
    experiment_type: str,
    split: Optional[str] = None,
    direction_mode: Optional[str] = None,
    subdir: Optional[str] = None,
    custom_path: Optional[Path] = None,
) -> Path:
    """构建统一的输出目录路径。
    
    Args:
        model_alias: 模型别名
        experiment_type: 实验类型（如 "attention_projections", "path_intervention", "direction_removal"）
        split: 数据集分割（可选）
        direction_mode: 方向模式（可选）
        subdir: 子目录（可选）
        custom_path: 自定义路径（如果提供，则直接返回）
        
    Returns:
        输出目录路径
    """
    if custom_path is not None:
        return Path(custom_path)
    
    base_dir = get_runs_root() / model_alias / experiment_type
    
    if split is not None and direction_mode is not None:
        base_dir = base_dir / f"{split}_{direction_mode}"
    elif split is not None:
        base_dir = base_dir / split
    elif direction_mode is not None:
        base_dir = base_dir / direction_mode
    
    if subdir is not None:
        base_dir = base_dir / subdir
    
    return base_dir


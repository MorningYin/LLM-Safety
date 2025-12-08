# -*- coding: utf-8 -*-
"""
direction_loader.py

统一的方向向量加载接口。
"""

from pathlib import Path
from typing import Optional

import torch

from contrast_analyze.utils.common import get_artifact_root, get_pipeline_runs_root
from contrast_analyze.utils.pipeline_config import PipelineConfig
from contrast_analyze.utils.direction_utils import (
    load_direction_candidates,
    load_direction_global,
)


def load_direction(
    model_alias: str,
    direction_mode: str,
    direction_path: Optional[Path] = None,
    pipeline_runs_root: Optional[Path] = None,
) -> torch.Tensor:
    """统一的方向向量加载接口。
    
    Args:
        model_alias: 模型别名
        direction_mode: 方向模式（"direction" 或 "directions"）
        direction_path: 自定义方向文件路径（可选）
        pipeline_runs_root: pipeline runs 根目录（可选）
        
    Returns:
        方向向量张量
    """
    if direction_path is not None:
        # 使用自定义路径
        if not direction_path.exists():
            raise FileNotFoundError(f"指定的方向文件不存在: {direction_path}")
        
        if direction_mode == "direction":
            return load_direction_global(direction_path)
        elif direction_mode == "directions":
            return load_direction_candidates(direction_path)
        else:
            raise ValueError(f"未知的 direction_mode: {direction_mode}")
    
    # 自动查找路径
    if direction_mode == "direction":
        # 全局方向：优先从 pipeline runs 查找，其次从 artifacts 查找
        if pipeline_runs_root is None:
            pipeline_runs_root = get_pipeline_runs_root()
        
        direction_path = pipeline_runs_root / model_alias / "direction.pt"
        if not direction_path.exists():
            # 回退到 PipelineConfig 的路径
            try:
                config = PipelineConfig(model_alias=model_alias)
                direction_path = config.direction_artifacts_dir / "direction.pt"
            except Exception:
                pass
            
            if not direction_path.exists():
                # 再回退到旧的 artifacts 目录
                artifact_root = get_artifact_root(model_alias)
                direction_path = artifact_root / "direction.pt"
        
        if not direction_path.exists():
            # 尝试所有可能的路径
            artifact_root = get_artifact_root(model_alias)
            raise FileNotFoundError(
                f"未找到全局 direction 文件。已查找：\n"
                f"  - {pipeline_runs_root / model_alias / 'direction.pt'}\n"
                f"  - {PipelineConfig(model_alias=model_alias).direction_artifacts_dir / 'direction.pt'}\n"
                f"  - {artifact_root / 'direction.pt'}"
            )
        
        return load_direction_global(direction_path)
    
    elif direction_mode == "directions":
        # 候选方向：优先使用 PipelineConfig 的路径，回退到 get_artifact_root
        try:
            config = PipelineConfig(model_alias=model_alias)
            direction_path = config.direction_artifacts_dir / "candidate_directions.pt"
        except Exception:
            # 回退到旧的路径查找方式
            artifact_root = get_artifact_root(model_alias)
            direction_path = artifact_root / "candidate_directions.pt"
        
        if not direction_path.exists():
            # 尝试另一个路径
            artifact_root = get_artifact_root(model_alias)
            alt_path = artifact_root / "candidate_directions.pt"
            if alt_path.exists() and alt_path != direction_path:
                direction_path = alt_path
            else:
                raise FileNotFoundError(
                    f"未找到候选方向文件。已查找：\n"
                    f"  - {direction_path}\n"
                    f"  - {alt_path}"
                )
        
        return load_direction_candidates(direction_path)
    
    else:
        raise ValueError(
            f"未知的 direction_mode: {direction_mode}，必须是 'direction' 或 'directions'"
        )


def get_direction_path(
    model_alias: str,
    direction_mode: str,
    pipeline_runs_root: Optional[Path] = None,
) -> Path:
    """获取方向文件的路径（不加载）。
    
    Args:
        model_alias: 模型别名
        direction_mode: 方向模式
        pipeline_runs_root: pipeline runs 根目录（可选）
        
    Returns:
        方向文件路径
    """
    if direction_mode == "direction":
        if pipeline_runs_root is None:
            pipeline_runs_root = get_pipeline_runs_root()
        
        direction_path = pipeline_runs_root / model_alias / "direction.pt"
        if not direction_path.exists():
            artifact_root = get_artifact_root(model_alias)
            direction_path = artifact_root / "direction.pt"
        
        return direction_path
    
    elif direction_mode == "directions":
        artifact_root = get_artifact_root(model_alias)
        return artifact_root / "candidate_directions.pt"
    
    else:
        raise ValueError(f"未知的 direction_mode: {direction_mode}")


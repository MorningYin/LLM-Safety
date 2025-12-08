# -*- coding: utf-8 -*-
"""
pipeline_config.py

统一的 Pipeline 配置管理系统。
提供清晰的路径管理和配置接口。
"""

import os
from pathlib import Path
from typing import Optional


class PipelineConfig:
    """
    统一的 Pipeline 配置类。
    
    管理所有路径和配置，确保整个 pipeline 的路径一致性。
    """
    
    def __init__(
        self,
        model_alias: str,
        split: Optional[str] = None,
        direction_mode: Optional[str] = None,
        runs_root: Optional[Path] = None,
    ):
        """
        初始化配置。
        
        Args:
            model_alias: 模型别名（如 "llama3"）
            split: 数据集分割（"vanilla" 或 "adversarial"），可选
            direction_mode: 方向模式（"direction" 或 "directions"），可选
            runs_root: runs 根目录，如果为 None 则使用默认路径或环境变量
        """
        self.model_alias = model_alias
        self.split = split
        self.direction_mode = direction_mode
        
        # 获取 runs 根目录
        if runs_root is not None:
            self.runs_root = Path(runs_root)
        else:
            # 优先使用环境变量
            custom_root = os.environ.get("CONTRAST_ANALYZE_RUNS_ROOT")
            if custom_root:
                self.runs_root = Path(custom_root)
            else:
                # 默认使用数据盘路径
                self.runs_root = Path("/root/autodl-tmp/contrast_analyze")
        
        # 确保 runs_root 存在
        self.runs_root.mkdir(parents=True, exist_ok=True)
    
    @property
    def model_root(self) -> Path:
        """获取模型根目录：{runs_root}/{model_alias}/"""
        return self.runs_root / self.model_alias
    
    @property
    def contrast_data_dir(self) -> Path:
        """获取原始数据目录：{runs_root}/{model_alias}/contrast_data/"""
        return self.model_root / "contrast_data"
    
    @property
    def shared_contrast_data_dir(self) -> Path:
        """获取共享数据目录：{runs_root}/shared/contrast_data/"""
        return self.runs_root / "shared" / "contrast_data"
    
    @property
    def direction_artifacts_dir(self) -> Path:
        """获取方向向量目录：{runs_root}/{model_alias}/direction_artifacts/"""
        return self.model_root / "direction_artifacts"
    
    @property
    def projection_analysis_dir(self) -> Path:
        """
        获取投影数据目录：{runs_root}/{model_alias}/projection_analysis/{direction_mode}/
        
        如果 direction_mode 为 None，返回基础目录。
        """
        base_dir = self.model_root / "projection_analysis"
        if self.direction_mode is not None:
            return base_dir / self.direction_mode
        return base_dir
    
    @property
    def driver_analysis_dir(self) -> Path:
        """
        获取驱动分析目录：{runs_root}/{model_alias}/driver_analysis/driver_analysis_effect_overlap/{direction_mode}/
        
        如果 direction_mode 为 None，返回基础目录。
        """
        base_dir = self.model_root / "driver_analysis" / "driver_analysis_effect_overlap"
        if self.direction_mode is not None:
            return base_dir / self.direction_mode
        return base_dir
    
    @property
    def zero_trust_dir(self) -> Path:
        """
        获取零信任模型目录：{runs_root}/{model_alias}/zero_trust/{direction_mode}/
        
        如果 direction_mode 为 None，返回基础目录。
        """
        base_dir = self.model_root / "zero_trust"
        if self.direction_mode is not None:
            return base_dir / self.direction_mode
        return base_dir
    
    def get_contrast_data_file(self, data_type: str, shared: bool = False) -> Path:
        """
        获取原始数据文件路径（不再区分 split，统一使用 vanilla 数据）。
        
        Args:
            data_type: 数据类型（"benign", "harmful_success", "harmful_failure"）
            shared: 是否使用共享目录（用于无害数据）
        
        Returns:
            数据文件路径
        """
        if data_type == "benign":
            filename = "benign.jsonl"
        elif data_type == "harmful_success":
            filename = "harmful_success.jsonl"
        elif data_type == "harmful_failure":
            filename = "harmful_failure.jsonl"
        else:
            raise ValueError(f"未知的数据类型: {data_type}")
        
        if shared:
            return self.shared_contrast_data_dir / filename
        return self.contrast_data_dir / filename
    
    def get_direction_file(self, direction_mode: Optional[str] = None) -> Path:
        """
        获取方向向量文件路径。
        
        Args:
            direction_mode: 方向模式，如果为 None 则使用 self.direction_mode
        
        Returns:
            方向向量文件路径
        """
        mode = direction_mode or self.direction_mode
        if mode == "direction":
            return self.direction_artifacts_dir / "direction.pt"
        elif mode == "directions":
            return self.direction_artifacts_dir / "candidate_directions.pt"
        else:
            raise ValueError(f"未知的方向模式: {mode}")
    
    def get_valid_components_file(self) -> Path:
        """获取有效组件文件路径：{driver_analysis_dir}/valid_components.json"""
        return self.driver_analysis_dir / "valid_components.json"
    
    def get_distribution_fits_file(self, token_idx: int = 0) -> Path:
        """
        获取分布拟合文件路径。
        
        Args:
            token_idx: token 索引，默认为 0
        
        Returns:
            分布拟合文件路径
        """
        return self.driver_analysis_dir / "distribution_plots" / f"token_{token_idx}" / "distribution_fits.json"
    
    def ensure_dirs(self):
        """确保所有必要的目录存在。"""
        self.contrast_data_dir.mkdir(parents=True, exist_ok=True)
        self.shared_contrast_data_dir.mkdir(parents=True, exist_ok=True)
        self.direction_artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.projection_analysis_dir.mkdir(parents=True, exist_ok=True)
        self.driver_analysis_dir.mkdir(parents=True, exist_ok=True)
        self.zero_trust_dir.mkdir(parents=True, exist_ok=True)
    
    def __repr__(self) -> str:
        """返回配置的字符串表示。"""
        parts = [f"model_alias={self.model_alias}"]
        if self.split:
            parts.append(f"split={self.split}")
        if self.direction_mode:
            parts.append(f"direction_mode={self.direction_mode}")
        parts.append(f"runs_root={self.runs_root}")
        return f"PipelineConfig({', '.join(parts)})"




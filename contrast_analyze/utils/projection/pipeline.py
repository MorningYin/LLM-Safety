"""Stage 3 工具：收集投影数据。"""

from __future__ import annotations

import torch

from contrast_analyze.utils.dataset_loader import load_all_categories
from contrast_analyze.utils.direction_loader import load_direction
from contrast_analyze.utils.direction_utils import resolve_eoi_marker
from contrast_analyze.utils.mean_projection_utils import compute_comprehensive_projection_data
from contrast_analyze.utils.pipeline_config import PipelineConfig
from pipeline.model_utils.model_factory import construct_model_base


def run_stage3_projection(config: PipelineConfig, *, batch_size: int, n_samples: int | None, seed: int):
    model_base = construct_model_base(config.model_alias)
    model_base.model.eval()

    direction = load_direction(
        model_alias=config.model_alias,
        direction_mode=config.direction_mode,
    )
    if isinstance(direction, torch.Tensor):
        direction = direction.to(model_base.model.device)

    datasets = load_all_categories(
        model_alias=config.model_alias,
        split="vanilla",  # 统一使用 vanilla 数据，不再区分
        max_samples=n_samples,
        seed=seed,
    )
    if not datasets:
        raise ValueError("没有找到任何数据集")

    marker = resolve_eoi_marker(model_base)

    for cat_name, dataset in datasets.items():
        if not dataset:
            continue
        category_storage_root = str(config.projection_analysis_dir / cat_name)
        compute_comprehensive_projection_data(
            direction=direction,
            batch_size=batch_size,
            model_base=model_base,
            dataset=dataset,
            marker=marker,
            storage_root=category_storage_root,
            desc=f"计算 {cat_name} 投影数据",
        )

    del model_base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


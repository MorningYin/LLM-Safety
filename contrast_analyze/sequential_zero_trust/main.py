#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
层级序贯贝叶斯零信任模型主入口脚本。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from tqdm import tqdm

from contrast_analyze.utils.pipeline_core import log_section, setup_environment
from contrast_analyze.utils.pipeline_config import PipelineConfig
from contrast_analyze.utils.dataset_loader import load_all_categories
from contrast_analyze.utils.direction_loader import load_direction
from contrast_analyze.utils.direction_utils import resolve_eoi_marker
from pipeline.model_utils.model_factory import construct_model_base

from .trajectory_compute import compute_evidence_trajectory
from .visualization import plot_evidence_trajectory, plot_layer_scores_comparison


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="层级序贯贝叶斯零信任模型 - 计算证据轨迹"
    )
    parser.add_argument("--model_alias", type=str, required=True, help="模型别名")
    parser.add_argument("--direction_mode", type=str, required=True, choices=["direction", "directions"], help="方向模式")
    parser.add_argument("--beta", type=float, default=0, help="OOD 得分权重")
    parser.add_argument("--K", type=int, default=5, help="Top-K Conflict Product 算法中的 K 值（默认: 5）")
    parser.add_argument("--alpha", type=float, default=2.0, help="负向信号权重系数（默认: 2.0）")
    parser.add_argument("--batch_size", type=int, default=32, help="批处理大小")
    parser.add_argument("--n_batch", type=int, default=None, help="从预计算数据中随机读取的 batch 数量（None 表示读取所有）")
    parser.add_argument("--seed", type=int, default=2025, help="随机选择 batch 的种子")
    parser.add_argument("--use_gpu", action="store_true", default=True, help="使用GPU加速计算（默认: True）")
    parser.add_argument("--no_gpu", dest="use_gpu", action="store_false", help="禁用GPU加速")
    parser.add_argument("--n_workers", type=int, default=0, help="并行处理的worker数量（0表示不并行）")
    return parser.parse_args()


def load_driver_analysis_data(config: PipelineConfig) -> tuple[Path, Path]:
    """
    加载驱动分析数据。
    Returns: (valid_components_path, distribution_fits_path)
    """
    valid_components_path = config.get_valid_components_file()

    distribution_fits_path = None
    for token_idx in range(5):
        test_path = config.get_distribution_fits_file(token_idx)
        if test_path.exists():
            distribution_fits_path = test_path
            break

    if not valid_components_path.exists():
        raise FileNotFoundError(f"未找到 valid_components.json: {valid_components_path}")
    if distribution_fits_path is None:
        raise FileNotFoundError("未找到 distribution_fits.json，已尝试 token_0 到 token_4")

    return valid_components_path, distribution_fits_path


def run_zero_trust(args):
    setup_environment(seed=args.seed)

    config = PipelineConfig(
        model_alias=args.model_alias,
        direction_mode=args.direction_mode,
    )
    config.ensure_dirs()

    log_section("Stage 5: 层级序贯贝叶斯零信任模型 - 证据轨迹计算")
    print(f"模型: {args.model_alias}")
    print(f"方向模式: {args.direction_mode}")
    print(f"Beta (OOD权重): {args.beta}")
    print(f"K (Top-K Conflict Product): {args.K}")
    print(f"Alpha (负向信号权重): {args.alpha}")
    print(f"批处理大小: {args.batch_size}")
    print(f"读取 batch 数量: {args.n_batch if args.n_batch is not None else '全部'}")
    print(f"随机种子: {args.seed}")
    print(f"使用GPU加速: {args.use_gpu}")
    print(f"并行worker数量: {args.n_workers if args.n_workers > 0 else '不并行'}")
    print(f"输出目录: {config.zero_trust_dir}")

    output_dir = config.zero_trust_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    projection_analysis_root = config.projection_analysis_dir
    use_precomputed = projection_analysis_root.exists()

    model_base = None
    direction = None
    if use_precomputed:
        print("\n使用预计算数据，跳过模型和方向加载")
        marker = None
    else:
        print("\n正在加载模型（实时计算模式）...")
        model_base = construct_model_base(args.model_alias)
        model_base.model.eval()
        print(f"模型加载完成: {args.model_alias}")

        print("\n正在加载方向向量...")
        direction = load_direction(
            model_alias=args.model_alias,
            direction_mode=args.direction_mode,
        )
        if isinstance(direction, torch.Tensor):
            direction = direction.cpu().numpy()
        print(f"方向向量形状: {direction.shape}")

        marker = resolve_eoi_marker(model_base)
        print(f"\nEOI Marker: {repr(marker)}")

    print("\n正在加载数据集...")
    datasets = load_all_categories(
        model_alias=args.model_alias,
        split="vanilla",  # 统一使用 vanilla 数据，不再区分
        max_samples=None,
        seed=args.seed,
    )
    if not datasets:
        raise ValueError("没有找到任何数据集")

    print(f"数据集类别: {list(datasets.keys())}")
    for cat_name, dataset in datasets.items():
        print(f"  {cat_name}: {len(dataset)} 样本")

    print("\n正在加载驱动分析数据...")
    valid_components_path, distribution_fits_path = load_driver_analysis_data(config)
    print(f"Valid Components: {valid_components_path}")
    print(f"Distribution Fits: {distribution_fits_path}")

    print("\n" + "=" * 60)
    print("开始计算证据轨迹...")
    print("=" * 60)

    all_trajectories: Dict[str, np.ndarray] = {}
    all_layer_scores: Dict[str, Dict[str, np.ndarray]] = {}

    print("\n使用预计算数据" if use_precomputed else "\n使用实时计算模式")
    dataset_items = list(datasets.items())
    pbar = tqdm(dataset_items, desc="处理数据集", unit="数据集", position=0, leave=True, ncols=100)
    for cat_name, dataset in pbar:
        if not dataset:
            pbar.write(f"[跳过] {cat_name}: 数据集为空")
            continue

        pbar.set_description(f"处理: {cat_name}")
        try:
            trajectory, layer_scores = compute_evidence_trajectory(
                model_base=model_base,
                direction=direction,
                dataset=dataset,
                valid_components_path=valid_components_path,
                distribution_fits_path=distribution_fits_path,
                marker=marker,
                beta=args.beta,
                K=args.K,
                alpha=args.alpha,
                batch_size=args.batch_size,
                projection_analysis_root=projection_analysis_root if use_precomputed else None,
                category=cat_name if use_precomputed else None,
                n_batch=args.n_batch,
                seed=args.seed,
                use_gpu=args.use_gpu,
                n_workers=args.n_workers,
            )
            all_trajectories[cat_name] = trajectory
            all_layer_scores[cat_name] = layer_scores
            pbar.set_postfix({"形状": f"{trajectory.shape}"})
        except Exception as e:
            pbar.write(f"[错误] {cat_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not all_trajectories:
        raise ValueError("没有成功计算任何轨迹")

    print("\n正在保存结果...")
    trajectory_path = output_dir / "trajectories.npz"
    np.savez(trajectory_path, **all_trajectories)
    print(f"轨迹数据已保存: {trajectory_path}")

    layer_scores_path = output_dir / "layer_scores.npz"
    layer_scores_data = {}
    for cat_name, scores_dict in all_layer_scores.items():
        for score_type, scores in scores_dict.items():
            key = f"{cat_name}_{score_type}"
            layer_scores_data[key] = scores
    np.savez(layer_scores_path, **layer_scores_data)
    print(f"层得分数据已保存: {layer_scores_path}")

    if model_base is not None:
        n_layers = len(model_base.model_block_modules)
    else:
        first_trajectory = next(iter(all_trajectories.values()))
        n_layers = first_trajectory.shape[1] if first_trajectory.ndim >= 2 else 0

    metadata = {
        "model_alias": args.model_alias,
        "direction_mode": args.direction_mode,
        "beta": args.beta,
        "K": args.K,
        "alpha": args.alpha,
        "batch_size": args.batch_size,
        "n_batch": args.n_batch,
        "seed": args.seed,
        "use_gpu": args.use_gpu,
        "n_workers": args.n_workers,
        "n_samples_per_category": {cat: len(datasets[cat]) for cat in datasets.keys()},
        "n_layers": n_layers,
        "valid_components_path": str(valid_components_path),
        "distribution_fits_path": str(distribution_fits_path),
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"元数据已保存: {metadata_path}")

    print("\n正在生成可视化图表...")
    plot_evidence_trajectory(
        trajectories=all_trajectories,
        save_path=output_dir / "evidence_trajectory.png",
    )
    plot_layer_scores_comparison(
        all_layer_scores=all_layer_scores,
        save_path=output_dir / "layer_scores.png",
    )

    print("\n全部完成！")


def main():
    run_zero_trust(parse_args())


if __name__ == "__main__":
    main()


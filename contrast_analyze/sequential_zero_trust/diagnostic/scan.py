#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnostic/scan.py

全量诊断扫描脚本：遍历所有层和所有 EOI Token，进行彻底的底层数据分析。
"""

import os

cache_root = os.environ.get("HF_CACHE_ROOT", "/root/autodl-tmp/Projects")
os.makedirs(cache_root, exist_ok=True)
os.makedirs(os.path.join(cache_root, "datasets_cache"), exist_ok=True)
os.makedirs(os.path.join(cache_root, "hub_cache"), exist_ok=True)
os.makedirs(os.path.join(cache_root, "transformers_cache"), exist_ok=True)
os.environ["HF_HOME"] = cache_root
os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_root, "datasets_cache")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(cache_root, "hub_cache")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_root, "transformers_cache")

import argparse
import json
import random
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

from pipeline.model_utils.model_factory import construct_model_base
from contrast_analyze.utils.common import setup_environment
from contrast_analyze.utils.dataset_loader import load_all_categories
from contrast_analyze.utils.direction_loader import load_direction
from contrast_analyze.utils.direction_utils import resolve_eoi_marker

from ..trajectory_compute import (
    load_valid_components,
    load_distribution_fits,
    organize_components_by_layer
)
from ..component_activations import compute_component_activations
from ..compute_pdfs import compute_pdfs
from .utils import (
    calculate_llr_matrix,
    compute_dataset_stats,
    compute_component_stats,
    compute_p_known,
    plot_llr_violin,
    plot_component_heatmap,
    plot_component_fit,
    plot_component_distributions
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="全量诊断扫描：分析所有层和所有 EOI Token 的底层数据"
    )
    parser.add_argument("--model_alias", type=str, default="llama3", help="模型别名")
    parser.add_argument("--direction_mode", type=str, default="directions", choices=["direction", "directions"], help="方向模式")
    parser.add_argument("--driver_analysis_root", type=str, default="/root/autodl-tmp/contrast_analyze_runs", help="驱动分析结果根目录")
    parser.add_argument("--driver_analysis_type", type=str, default="driver_analysis_effect_overlap", help="驱动分析类型")
    parser.add_argument("--n_samples", type=int, default=100, help="每个数据集使用的样本数")
    parser.add_argument("--batch_size", type=int, default=32, help="批处理大小")
    parser.add_argument("--output_dir", type=str, default="diagnostic_results", help="输出目录")
    parser.add_argument("--seed", type=int, default=2025, help="随机种子")
    parser.add_argument("--projection_analysis_root", type=str, default="/root/autodl-tmp/contrast_analyze_runs/llama3/projection_analysis", help="projection_analysis 根目录（可选，如果提供则使用预计算数据）")
    return parser.parse_args()


def load_driver_analysis_data(
    driver_analysis_root: Path,
    model_alias: str,
    driver_analysis_type: str,
    split: str,  # 保留参数以兼容，但不再使用
    direction_mode: str
) -> tuple[Path, Path]:
    """
    加载驱动分析数据（不再区分 split，统一使用 vanilla 数据）。
    
    Returns:
        (valid_components_path, base_data_path)
    """
    base_path = Path(driver_analysis_root) / model_alias / driver_analysis_type
    
    # 根据 direction_mode 确定子目录（不再使用 split）
    if direction_mode == "direction":
        subdir = "direction"
    else:
        subdir = "directions"
    
    data_path = base_path / subdir
    
    # valid_components.json 在根目录
    valid_components_path = data_path / "valid_components.json"
    
    if not valid_components_path.exists():
        raise FileNotFoundError(f"未找到 valid_components.json: {valid_components_path}")
    
    return valid_components_path, data_path


def get_available_tokens(data_path: Path) -> List[int]:
    """
    获取所有可用的 token 索引。
    
    Args:
        data_path: 驱动分析数据的基础路径
    
    Returns:
        token 索引列表
    """
    distribution_plots_dir = data_path / "distribution_plots"
    if not distribution_plots_dir.exists():
        return []
    
    tokens = []
    for token_dir in distribution_plots_dir.iterdir():
        if token_dir.is_dir() and token_dir.name.startswith("token_"):
            try:
                token_idx = int(token_dir.name.split("_")[1])
                # 检查是否有 distribution_fits.json
                fits_path = token_dir / "distribution_fits.json"
                if fits_path.exists():
                    tokens.append(token_idx)
            except (ValueError, IndexError):
                continue
    
    return sorted(tokens)


def process_layer_token(
    layer_idx: int,
    token_idx: int,
    model_base,
    direction: np.ndarray,
    datasets: Dict[str, List[Dict]],
    valid_components: Dict,
    distribution_fits: Dict,
    marker: str,
    output_dir: Path,
    batch_size: int,
    projection_analysis_root: Optional[Path] = None
) -> Optional[Dict]:
    """
    处理单个 (Layer, Token) 组合。
    
    Returns:
        统计数据字典（用于 JSONL 输出），如果处理失败则返回 None
    """
    # 1. 组织该层该 token 的组件
    components_by_layer = organize_components_by_layer(valid_components, distribution_fits)
    
    if layer_idx not in components_by_layer:
        return None
    
    if token_idx not in components_by_layer[layer_idx]:
        return None
    
    components = components_by_layer[layer_idx][token_idx]
    if not components:
        return None
    
    # 2. 计算三个数据集的激活值
    activations_dict = {}
    component_keys = [comp['key'] for comp in components]
    
    for dataset_name, dataset in datasets.items():
        if not dataset:
            continue
        
        # 确定当前类别的 projection_analysis 路径
        category_projection_root = None
        if projection_analysis_root is not None:
            category_projection_root = projection_analysis_root / dataset_name
        
        # 计算激活值
        dataset_activations = compute_component_activations(
            model_base=model_base,
            direction=direction,
            dataset=dataset,
            valid_components=valid_components,
            marker=marker,
            batch_size=batch_size,
            projection_analysis_root=category_projection_root,
            category=dataset_name if category_projection_root is not None else None
        )
        
        # 只保留该层该 token 的组件
        for comp_key in component_keys:
            if comp_key not in activations_dict:
                activations_dict[comp_key] = {}
            if comp_key in dataset_activations:
                activations_dict[comp_key][dataset_name] = np.array(dataset_activations[comp_key])
            else:
                activations_dict[comp_key][dataset_name] = np.zeros(len(dataset))
    
    # 3. 组织激活值数组和分布参数
    N_components = len(components)
    N_benign = len(datasets.get('benign', []))
    N_refusal = len(datasets.get('harmful_failure', []))
    N_jailbreak = len(datasets.get('harmful_success', []))
    
    if N_benign == 0 or N_refusal == 0 or N_jailbreak == 0:
        return None
    
    # 组织激活值：三个数据集分别
    activations_benign = np.zeros((N_benign, N_components))
    activations_refusal = np.zeros((N_refusal, N_components))
    activations_jailbreak = np.zeros((N_jailbreak, N_components))
    
    # 组织分布参数
    dist_params_dict = {}
    component_names = []
    
    for comp_idx, comp_info in enumerate(components):
        comp_key = comp_info['key']
        component_names.append(comp_key)
        dist_params_dict[comp_key] = comp_info['dist_params']
        
        # 填充激活值
        if comp_key in activations_dict:
            if 'benign' in activations_dict[comp_key]:
                activations_benign[:, comp_idx] = activations_dict[comp_key]['benign']
            if 'harmful_failure' in activations_dict[comp_key]:
                activations_refusal[:, comp_idx] = activations_dict[comp_key]['harmful_failure']
            if 'harmful_success' in activations_dict[comp_key]:
                activations_jailbreak[:, comp_idx] = activations_dict[comp_key]['harmful_success']
    
    # 4. 计算 PDF
    f_B_benign, f_R_benign = compute_pdfs(activations_benign, dist_params_dict)
    f_B_refusal, f_R_refusal = compute_pdfs(activations_refusal, dist_params_dict)
    f_B_jailbreak, f_R_jailbreak = compute_pdfs(activations_jailbreak, dist_params_dict)
    
    # 5. 计算 P_known^{(i)}(x_i) = 0.5 * f_B + 0.5 * f_R
    P_known_benign = compute_p_known(f_B_benign, f_R_benign)
    P_known_refusal = compute_p_known(f_B_refusal, f_R_refusal)
    P_known_jailbreak = compute_p_known(f_B_jailbreak, f_R_jailbreak)
    
    # 6. 计算 LLR Matrix (lambda_i)
    llr_benign = calculate_llr_matrix(f_B_benign, f_R_benign)
    llr_refusal = calculate_llr_matrix(f_B_refusal, f_R_refusal)
    llr_jailbreak = calculate_llr_matrix(f_B_jailbreak, f_R_jailbreak)
    
    # lambda_i 就是 llr_matrix
    lambda_i_benign = llr_benign
    lambda_i_refusal = llr_refusal
    lambda_i_jailbreak = llr_jailbreak
    
    # 7. 计算统计量（扩展版本）
    stats_benign = compute_dataset_stats(
        llr_benign,
        activations=activations_benign,
        f_B=f_B_benign,
        f_R=f_R_benign,
        P_known=P_known_benign,
        lambda_i=lambda_i_benign
    )
    stats_refusal = compute_dataset_stats(
        llr_refusal,
        activations=activations_refusal,
        f_B=f_B_refusal,
        f_R=f_R_refusal,
        P_known=P_known_refusal,
        lambda_i=lambda_i_refusal
    )
    stats_jailbreak = compute_dataset_stats(
        llr_jailbreak,
        activations=activations_jailbreak,
        f_B=f_B_jailbreak,
        f_R=f_R_jailbreak,
        P_known=P_known_jailbreak,
        lambda_i=lambda_i_jailbreak
    )
    
    # 准备分布参数字典列表
    dist_params_list = [comp_info['dist_params'] for comp_info in components]
    
    component_stats_list, top_3_indices = compute_component_stats(
        llr_benign, llr_refusal, llr_jailbreak, component_names,
        activations_benign=activations_benign,
        activations_refusal=activations_refusal,
        activations_jailbreak=activations_jailbreak,
        f_B_benign=f_B_benign,
        f_B_refusal=f_B_refusal,
        f_B_jailbreak=f_B_jailbreak,
        f_R_benign=f_R_benign,
        f_R_refusal=f_R_refusal,
        f_R_jailbreak=f_R_jailbreak,
        P_known_benign=P_known_benign,
        P_known_refusal=P_known_refusal,
        P_known_jailbreak=P_known_jailbreak,
        lambda_i_benign=lambda_i_benign,
        lambda_i_refusal=lambda_i_refusal,
        lambda_i_jailbreak=lambda_i_jailbreak,
        distribution_params=dist_params_list
    )
    
    # 7. 创建输出目录
    layer_dir = output_dir / f"layer_{layer_idx:02d}" / f"token_{token_idx}"
    layer_dir.mkdir(parents=True, exist_ok=True)
    dist_fits_dir = layer_dir / "dist_fits"
    dist_fits_dir.mkdir(exist_ok=True)
    component_distributions_dir = layer_dir / "component_distributions"
    component_distributions_dir.mkdir(exist_ok=True)
    
    # 8. 生成可视化
    # 8.1 Violin Plot
    violin_path = layer_dir / "llr_violin.png"
    plot_llr_violin(
        llr_benign, llr_refusal, llr_jailbreak,
        violin_path,
        title=f"LLR Distribution: Layer {layer_idx}, Token {token_idx}"
    )
    
    # 8.2 Heatmap (使用 Refusal 数据集)
    heatmap_path = layer_dir / "heatmap.png"
    plot_component_heatmap(
        llr_refusal, component_names, heatmap_path,
        n_samples=50,
        title=f"Component-Sample Heatmap: Layer {layer_idx}, Token {token_idx}"
    )
    
    # 8.3 Top-3 Components Fit
    for top_idx, comp_idx in enumerate(top_3_indices):
        comp_name = component_names[comp_idx]
        comp_dist_params = dist_params_dict[comp_name]
        
        # 使用 Refusal 数据集的激活值
        comp_activations = activations_refusal[:, comp_idx]
        
        fit_path = dist_fits_dir / f"component_{comp_name}.png"
        plot_component_fit(
            comp_activations, comp_dist_params, comp_name,
            fit_path,
            title=f"Distribution Fit: {comp_name} (Layer {layer_idx}, Token {token_idx})"
        )
    
    # 8.4 为每个组件生成分布对比图（P_known 和 lambda_i）
    for comp_idx, comp_info in enumerate(components):
        comp_name = comp_info['key']
        comp_dist_params = comp_info['dist_params']
        
        # 获取三个数据集的激活值
        comp_activations_benign = activations_benign[:, comp_idx]
        comp_activations_refusal = activations_refusal[:, comp_idx]
        comp_activations_jailbreak = activations_jailbreak[:, comp_idx]
        
        # 生成分布对比图
        dist_path = component_distributions_dir / f"{comp_name}.png"
        plot_component_distributions(
            activations_benign=comp_activations_benign,
            activations_refusal=comp_activations_refusal,
            activations_jailbreak=comp_activations_jailbreak,
            dist_params=comp_dist_params,
            component_name=comp_name,
            output_path=dist_path,
            title=f"Component Distributions: {comp_name} (Layer {layer_idx}, Token {token_idx})"
        )
    
    # 9. 构建统计数据字典
    stats_dict = {
        "layer": layer_idx,
        "token": token_idx,
        "num_components": N_components,
        "dataset_stats": {
            "benign": stats_benign,
            "harmful_failure": stats_refusal,
            "harmful_success": stats_jailbreak
        },
        "components": component_stats_list
    }
    
    return stats_dict


def main():
    """主函数"""
    # 设置环境
    setup_environment()
    
    args = parse_args()
    
    print("=" * 60)
    print("全量诊断扫描")
    print("=" * 60)
    print(f"模型: {args.model_alias}")
    print(f"方向模式: {args.direction_mode}")
    print(f"样本数: {args.n_samples}")
    print(f"批处理大小: {args.batch_size}")
    print(f"随机种子: {args.seed}")
    print("=" * 60)
    
    # 1. 确定输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {output_dir}")
    
    # 2. 加载模型
    print("\n正在加载模型...")
    model_base = construct_model_base(args.model_alias)
    model_base.model.eval()
    n_layers = len(model_base.model_block_modules)
    print(f"模型加载完成: {args.model_alias} ({n_layers} 层)")
    
    # 3. 加载方向向量
    print("\n正在加载方向向量...")
    direction = load_direction(
        model_alias=args.model_alias,
        direction_mode=args.direction_mode,
    )
    if isinstance(direction, torch.Tensor):
        direction = direction.cpu().numpy()
    print(f"方向向量形状: {direction.shape}")
    
    # 4. 加载数据集
    print("\n正在加载数据集...")
    datasets = load_all_categories(
        model_alias=args.model_alias,
        split="vanilla",  # 统一使用 vanilla 数据，不再区分
        max_samples=args.n_samples,
        seed=args.seed,
    )
    
    if not datasets:
        raise ValueError("没有找到任何数据集")
    
    print(f"数据集类别: {list(datasets.keys())}")
    for cat_name, dataset in datasets.items():
        print(f"  {cat_name}: {len(dataset)} 样本")
    
    # 5. 获取 EOI marker（从 model_base 中获取）
    marker = resolve_eoi_marker(model_base)
    print(f"\nEOI Marker: {repr(marker)}")
    
    # 6. 加载驱动分析数据
    print("\n正在加载驱动分析数据...")
    valid_components_path, data_path = load_driver_analysis_data(
        driver_analysis_root=Path(args.driver_analysis_root),
        model_alias=args.model_alias,
        driver_analysis_type=args.driver_analysis_type,
        split="vanilla",  # 统一使用 vanilla 数据，不再区分
        direction_mode=args.direction_mode
    )
    print(f"Valid Components: {valid_components_path}")
    
    valid_components = load_valid_components(valid_components_path)
    
    # 确定 projection_analysis 根目录
    projection_analysis_root = None
    detected_batch_size = None
    if args.projection_analysis_root:
        projection_analysis_root = Path(args.projection_analysis_root)
        # 根据 direction_mode 确定子目录（不再使用 split）
        if args.direction_mode == "direction":
            subdir = "direction"
        else:
            subdir = "directions"
        projection_analysis_root = projection_analysis_root / subdir
        print(f"\n使用预计算数据: {projection_analysis_root}")
        
        # 预先检测 batch_size（只检测一次，避免重复输出）
        from ..load_projection_data import detect_batch_size
        # 使用第一个数据集类别进行检测
        first_category = list(datasets.keys())[0] if datasets else 'benign'
        category_path = projection_analysis_root / first_category
        detected_batch_size = detect_batch_size(category_path, first_category, verbose=True)
        print(f"检测到 batch_size: {detected_batch_size}")
    else:
        print("\n使用实时计算（未提供 projection_analysis_root）")
    
    # 获取所有可用的 token
    available_tokens = get_available_tokens(data_path)
    if not available_tokens:
        raise ValueError(f"未找到任何可用的 token 数据: {data_path / 'distribution_plots'}")
    print(f"可用 Token: {available_tokens}")
    
    # 7. 主循环：遍历所有层和所有 token
    print("\n" + "=" * 60)
    print("开始诊断扫描...")
    print("=" * 60)
    
    stats_jsonl_path = output_dir / "stats_summary.jsonl"
    
    total_tasks = n_layers * len(available_tokens)
    processed = 0
    failed = 0
    
    with open(stats_jsonl_path, 'w', encoding='utf-8') as f:
        for layer_idx in tqdm(range(n_layers), desc="Layers"):
            for token_idx in available_tokens:
                # 加载该 token 的 distribution_fits
                distribution_fits_path = data_path / "distribution_plots" / f"token_{token_idx}" / "distribution_fits.json"
                if not distribution_fits_path.exists():
                    failed += 1
                    continue
                
                distribution_fits = load_distribution_fits(distribution_fits_path)
                
                # 处理该 (Layer, Token) 组合
                try:
                    stats_dict = process_layer_token(
                        layer_idx=layer_idx,
                        token_idx=token_idx,
                        model_base=model_base,
                        direction=direction,
                        datasets=datasets,
                        valid_components=valid_components,
                        distribution_fits=distribution_fits,
                        marker=marker,
                        output_dir=output_dir,
                        batch_size=detected_batch_size if detected_batch_size is not None else args.batch_size,
                        projection_analysis_root=projection_analysis_root
                    )
                    
                    if stats_dict is not None:
                        # 写入 JSONL
                        f.write(json.dumps(stats_dict, ensure_ascii=False) + '\n')
                        f.flush()
                        processed += 1
                    else:
                        failed += 1
                
                except Exception as e:
                    print(f"\n[错误] Layer {layer_idx}, Token {token_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    failed += 1
    
    print("\n" + "=" * 60)
    print("诊断扫描完成！")
    print("=" * 60)
    print(f"处理成功: {processed}/{total_tasks}")
    print(f"处理失败: {failed}/{total_tasks}")
    print(f"输出目录: {output_dir}")
    print(f"统计数据: {stats_jsonl_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()


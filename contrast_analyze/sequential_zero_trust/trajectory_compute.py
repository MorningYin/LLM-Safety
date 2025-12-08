# -*- coding: utf-8 -*-
"""
trajectory_compute.py

计算证据轨迹（Evidence Trajectory）的主函数。
实现层级序贯贝叶斯零信任模型。
支持GPU加速和并行处理。
"""

import json
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from .compute_pdfs import compute_pdfs
from .compute_pdfs_gpu import compute_pdfs_gpu
from .layer_scores import calculate_layer_scores
from .component_activations import compute_component_activations
from pipeline.model_utils.model_base import ModelBase


def load_valid_components(valid_components_path: Path) -> Dict:
    """加载 valid_components.json"""
    with open(valid_components_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_distribution_fits(distribution_fits_path: Path) -> Dict:
    """加载 distribution_fits.json"""
    with open(distribution_fits_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def organize_components_by_layer(
    valid_components: Dict,
    distribution_fits: Dict
) -> Dict[int, Dict[int, List[Dict]]]:
    """
    按层和 token 组织组件，并匹配分布参数。
    
    Returns:
        components_by_layer: {
            layer_idx: {
                token_idx: [
                    {
                        'component': 'H9',
                        'key': 'L0_H9_T4',
                        'dist_params': {...}
                    },
                    ...
                ]
            }
        }
    """
    components_by_layer = {}
    
    for token_str, components in valid_components.get('components_by_token', {}).items():
        token_idx = int(token_str)
        
        for comp_info in components:
            layer = comp_info['layer']
            component = comp_info['component']
            comp_key = f"L{layer}_{component}_T{token_idx}"
            
            # 在 distribution_fits 中查找对应的分布参数
            # distribution_fits 的键格式可能是 "L12_H22" 或类似
            # 优先尝试 "L{layer}_{component}" 格式
            dist_key = f"L{layer}_{component}"
            if dist_key not in distribution_fits:
                # 尝试其他可能的键格式 (e.g. comp_key)
                dist_key = comp_key
                if dist_key not in distribution_fits:
                    continue
            
            dist_params = distribution_fits[dist_key]
            
            if layer not in components_by_layer:
                components_by_layer[layer] = {}
            if token_idx not in components_by_layer[layer]:
                components_by_layer[layer][token_idx] = []
            
            components_by_layer[layer][token_idx].append({
                'component': component,
                'key': comp_key,
                'dist_params': dist_params
            })
    
    return components_by_layer


def compute_evidence_trajectory(
    model_base: ModelBase,
    direction: np.ndarray,
    dataset: List[Dict],
    valid_components_path: Path,
    distribution_fits_path: Path,
    marker: str,
    beta: float = 0.0, # 修改：建议默认设为 0，优先观察 S_cons 的纯净分离效果
    K: int = 5,        # Top-K 参数
    alpha: float = 2.0, # 负向信号权重系数
    batch_size: int = 32,
    projection_analysis_root: Optional[Path] = None,
    category: Optional[str] = None,
    n_batch: Optional[int] = None,
    seed: Optional[int] = None,
    use_gpu: bool = True,  # 是否使用GPU加速
    n_workers: int = 0,   # 并行处理的worker数量（0表示不并行）
    device: Optional[torch.device] = None  # GPU设备
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    计算证据轨迹（Evidence Trajectory）。
    
    Args:
        model_base: 模型基础对象
        direction: 方向向量，形状为 (d_model,) 或 (n_layers, d_model)
        dataset: 数据集列表
        valid_components_path: valid_components.json 路径
        distribution_fits_path: distribution_fits.json 路径
        marker: EOI marker 字符串
        beta: OOD 得分权重 (建议为 0.0，除非 Type C 攻击非常显著)
        K: Top-K Conflict Product 算法中的 K 值（默认: 5）
        alpha: 负向信号权重系数（默认: 2.0）
        batch_size: 批处理大小
        projection_analysis_root: projection_analysis 根目录（可选，如果提供则使用预计算数据）
        category: 数据集类别（benign, harmful_success, harmful_failure），用于加载预计算数据
        n_batch: 要读取的 batch 数量（可选，None 表示读取所有）
        seed: 随机选择 batch 的种子（仅在 n_batch 不为 None 时使用）
        use_gpu: 是否使用GPU加速PDF和层得分计算（默认: True）
        n_workers: 并行处理的worker数量（0表示不并行，>0表示使用多线程，默认: 0）
        device: GPU设备（如果为None且use_gpu=True，自动检测）
    
    Returns:
        (trajectory, layer_scores_dict):
            - trajectory: 形状为 (N_samples, n_layers) 的累积风险轨迹
            - layer_scores_dict: 包含 S_cons, S_ood, R_l 的字典，每个都是 (N_samples, n_layers)
    """
    # 1. 加载数据
    valid_components = load_valid_components(valid_components_path)
    distribution_fits = load_distribution_fits(distribution_fits_path)
    
    # 2. 组织组件
    components_by_layer = organize_components_by_layer(valid_components, distribution_fits)
    
    # 3. 计算组件激活值
    if projection_analysis_root is not None:
        tqdm.write("正在从预计算数据加载组件激活值...")
        # 使用预计算数据时，不需要 model_base
        activations_dict = compute_component_activations(
            model_base=None,  # 预计算数据不需要模型
            direction=None,  # 预计算数据不需要方向向量
            dataset=dataset,
            valid_components=valid_components,
            marker=None,  # 预计算数据不需要 marker
            batch_size=batch_size,
            projection_analysis_root=projection_analysis_root,
            category=category,
            n_batch=n_batch,
            seed=seed
        )
        # 从 valid_components 推断层数
        n_layers = max([comp['layer'] for comp_list in valid_components.get('components_by_token', {}).values() 
                       for comp in comp_list], default=0) + 1
    else:
        tqdm.write("正在计算组件激活值...")
        if model_base is None:
            raise ValueError("实时计算模式需要 model_base")
    activations_dict = compute_component_activations(
        model_base=model_base,
        direction=direction,
        dataset=dataset,
        valid_components=valid_components,
        marker=marker,
        batch_size=batch_size,
            projection_analysis_root=None,
            category=None,
        n_batch=n_batch,
        seed=seed
    )
        # 从模型获取层数
    n_layers = len(model_base.model_block_modules)
    # 从 activations_dict 获取实际样本数量（因为使用预计算数据时，样本数量由 n_batch 决定）
    if activations_dict:
        # 从第一个组件的激活值数组获取样本数量
        first_key = next(iter(activations_dict))
        n_samples = len(activations_dict[first_key])
    else:
        # 如果没有激活值，使用数据集长度（向后兼容）
        n_samples = len(dataset)
    
    # 5. 初始化输出数组
    trajectory = np.zeros((n_samples, n_layers))
    S_cons_all = np.zeros((n_samples, n_layers))
    S_ood_all = np.zeros((n_samples, n_layers))
    R_all = np.zeros((n_samples, n_layers))
    
    # 6. 设置GPU设备
    if use_gpu and torch.cuda.is_available():
        if device is None:
            device = torch.device('cuda')
        tqdm.write(f"使用GPU加速: {device}")
    else:
        device = None
        if use_gpu:
            tqdm.write("警告: 请求使用GPU但CUDA不可用，回退到CPU")
    
    # 7. 组织所有层的数据（为批量GPU处理做准备）
    activations_by_layer = {}
    dist_params_by_layer = {}
    
    for layer_idx in range(n_layers):
        if layer_idx not in components_by_layer:
            continue
        
        # 收集该层所有 token 的组件
        layer_keys = []
        layer_dist_params = {}
        
        for token_idx, components in components_by_layer[layer_idx].items():
            for comp_info in components:
                comp_key = comp_info['key']
                if comp_key in activations_dict:
                    layer_keys.append(comp_key)
                    layer_dist_params[comp_key] = comp_info['dist_params']
        
        if not layer_keys:
            continue
        
        # 组织激活值数组: (N_samples, N_components)
        N_components = len(layer_keys)
        activations = np.zeros((n_samples, N_components))
        
        for comp_idx, comp_key in enumerate(layer_keys):
            if comp_key in activations_dict:
                comp_activations = activations_dict[comp_key]
                if isinstance(comp_activations, np.ndarray):
                    comp_len = len(comp_activations)
                    if comp_len == n_samples:
                        activations[:, comp_idx] = comp_activations
                    elif comp_len < n_samples:
                        activations[:comp_len, comp_idx] = comp_activations
                        activations[comp_len:, comp_idx] = 0.0
                    else:
                        activations[:, comp_idx] = comp_activations[:n_samples]
                else:
                    comp_activations = np.array(comp_activations, dtype=np.float32)
                    comp_len = len(comp_activations)
                    if comp_len == n_samples:
                        activations[:, comp_idx] = comp_activations
                    elif comp_len < n_samples:
                        activations[:comp_len, comp_idx] = comp_activations
                        activations[comp_len:, comp_idx] = 0.0
                    else:
                        activations[:, comp_idx] = comp_activations[:n_samples]
        
        # 组织分布参数字典
        dist_params_dict = {}
        for comp_key in layer_keys:
            if comp_key in layer_dist_params:
                dist_params_dict[comp_key] = layer_dist_params[comp_key]
        
        activations_by_layer[layer_idx] = activations
        dist_params_by_layer[layer_idx] = dist_params_dict
    
    # 8. 选择计算方式：GPU批量处理 或 CPU逐层处理
    if use_gpu and device is not None:
        # 使用GPU批量处理
        from .trajectory_compute_gpu import compute_evidence_trajectory_batch_gpu
        
        trajectory, layer_scores_dict = compute_evidence_trajectory_batch_gpu(
            activations_by_layer=activations_by_layer,
            dist_params_by_layer=dist_params_by_layer,
            n_samples=n_samples,
            n_layers=n_layers,
            beta=beta,
            K=K,
            alpha=alpha,
            device=device
        )
        
        S_cons_all = layer_scores_dict['S_cons']
        S_ood_all = layer_scores_dict['S_ood']
        R_all = layer_scores_dict['R_l']
    else:
        # CPU逐层处理（原有逻辑）
        from .compute_pdfs import compute_pdfs
        
        def process_single_layer(layer_idx):
            """处理单层的计算（用于并行化）"""
            if layer_idx not in activations_by_layer:
                return layer_idx, None, None, None
            
            activations = activations_by_layer[layer_idx]
            dist_params_dict = dist_params_by_layer.get(layer_idx, {})
            
            if activations.shape[0] == 0 or len(dist_params_dict) == 0:
                return layer_idx, None, None, None
            
            # 计算 PDF
            f_B, f_R = compute_pdfs(activations, dist_params_dict)
            
            # 计算层得分
            S_cons, S_ood = calculate_layer_scores(
                f_B, f_R, K=K, alpha=alpha, 
                use_gpu=False
            )
            
            # 计算总风险
            R_l = S_cons + beta * S_ood
            
            return layer_idx, S_cons, S_ood, R_l
        
        # 选择并行或串行处理
        if n_workers > 0:
            # 并行处理
            tqdm.write(f"使用 {n_workers} 个worker并行处理（CPU模式）")
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(process_single_layer, layer_idx): layer_idx 
                          for layer_idx in range(n_layers)}
                
                for future in tqdm(as_completed(futures), total=n_layers, 
                                 desc="计算层得分", unit="层", leave=False, position=1, ncols=80):
                    layer_idx, S_cons, S_ood, R_l = future.result()
                    if S_cons is not None:
                        S_cons_all[:, layer_idx] = S_cons
                        S_ood_all[:, layer_idx] = S_ood
                        R_all[:, layer_idx] = R_l
        else:
            # 串行处理
            for layer_idx in tqdm(range(n_layers), desc="计算层得分", unit="层", leave=False, position=1, ncols=80):
                _, S_cons, S_ood, R_l = process_single_layer(layer_idx)
                if S_cons is not None:
                    S_cons_all[:, layer_idx] = S_cons
                    S_ood_all[:, layer_idx] = S_ood
                    R_all[:, layer_idx] = R_l
    
        # 计算累积轨迹
    trajectory = np.cumsum(R_all, axis=1)
    
    layer_scores_dict = {
        'S_cons': S_cons_all,
        'S_ood': S_ood_all,
        'R_l': R_all
    }
    
    return trajectory, layer_scores_dict
# -*- coding: utf-8 -*-
"""
trajectory_compute_gpu.py

GPU加速的证据轨迹计算。
批量处理所有层，充分利用GPU并行计算能力。
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from .compute_pdfs_gpu import compute_pdfs_gpu
from .layer_scores import calculate_layer_scores


def compute_evidence_trajectory_batch_gpu(
    activations_by_layer: Dict[int, np.ndarray],
    dist_params_by_layer: Dict[int, Dict[str, Dict]],
    n_samples: int,
    n_layers: int,
    beta: float = 0.0,
    K: int = 5,
    alpha: float = 2.0,
    device: Optional[torch.device] = None,
    batch_size_gpu: int = 1000  # GPU批处理大小
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    批量GPU加速计算证据轨迹。
    
    将所有层的计算批量处理，充分利用GPU并行能力。
    
    Args:
        activations_by_layer: 每层的激活值字典 {layer_idx: (N_samples, N_components)}
        dist_params_by_layer: 每层的分布参数字典 {layer_idx: {comp_key: dist_params}}
        n_samples: 样本数量
        n_layers: 层数
        beta: OOD得分权重
        K: Top-K参数
        alpha: 负向信号权重系数
        device: GPU设备
        batch_size_gpu: GPU批处理大小（用于控制内存使用）
    
    Returns:
        (trajectory, layer_scores_dict)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化输出数组
    S_cons_all = np.zeros((n_samples, n_layers))
    S_ood_all = np.zeros((n_samples, n_layers))
    R_all = np.zeros((n_samples, n_layers))
    
    # 按层处理（但每层内部批量计算）
    for layer_idx in tqdm(range(n_layers), desc="GPU批量计算层得分", unit="层", leave=False, position=1, ncols=80):
        if layer_idx not in activations_by_layer:
            continue
        
        activations = activations_by_layer[layer_idx]
        dist_params_dict = dist_params_by_layer.get(layer_idx, {})
        
        if activations.shape[0] == 0 or len(dist_params_dict) == 0:
            continue
        
        # 批量计算PDF（GPU加速）
        f_B, f_R = compute_pdfs_gpu(activations, dist_params_dict, device=device)
        
        # 如果样本数量很大，分批计算层得分以节省GPU内存
        if n_samples > batch_size_gpu:
            # 分批处理
            n_batches = (n_samples + batch_size_gpu - 1) // batch_size_gpu
            S_cons_batches = []
            S_ood_batches = []
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size_gpu
                end_idx = min((batch_idx + 1) * batch_size_gpu, n_samples)
                
                f_B_batch = f_B[start_idx:end_idx]
                f_R_batch = f_R[start_idx:end_idx]
                
                S_cons_batch, S_ood_batch = calculate_layer_scores(
                    f_B_batch, f_R_batch, K=K, alpha=alpha,
                    use_gpu=True, device=device
                )
                
                S_cons_batches.append(S_cons_batch)
                S_ood_batches.append(S_ood_batch)
            
            S_cons = np.concatenate(S_cons_batches, axis=0)
            S_ood = np.concatenate(S_ood_batches, axis=0)
        else:
            # 一次性计算
            S_cons, S_ood = calculate_layer_scores(
                f_B, f_R, K=K, alpha=alpha,
                use_gpu=True, device=device
            )
        
        # 计算总风险
        R_l = S_cons + beta * S_ood
        
        # 存储结果
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


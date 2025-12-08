# -*- coding: utf-8 -*-
"""
layer_scores.py

计算层级风险评分：S_cons（不一致性得分）和 S_ood（异常检测得分）。
修正版：移除权重乘法，恢复加法逻辑以保留轨迹方向，利用 Top-K 稀疏性去噪。
支持GPU加速计算。
"""

import numpy as np
import torch
from typing import Tuple, Optional, Union


EPSILON = 1e-10


def calculate_layer_scores(
    f_B: Union[np.ndarray, torch.Tensor],
    f_R: Union[np.ndarray, torch.Tensor],
    K: int = 5,
    alpha: float = 2.0,
    use_gpu: bool = False,
    device: Optional[torch.device] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算单层的冲突得分。
    
    算法流程：
    1. P_known = 0.5 * f_B + 0.5 * f_R (仅用于计算 S_ood)
    2. lambda_i = log(f_R / f_B) (原始对数似然比)
    3. 对 lambda_i 进行排序
    4. v_pos = Mean(Top-K Positive Values) -> 代表拒绝信号
    5. v_neg = Mean(Bottom-K Negative Values) -> 代表伪装/无害信号
    6. S_cons = v_pos + v_neg (代数和)
    
    Args:
        f_B: Benign PDF (N_samples, N_components)，可以是numpy数组或torch张量
        f_R: Refusal PDF (N_samples, N_components)，可以是numpy数组或torch张量
        K: Top-K 参数
        alpha: 负向信号权重系数（默认: 2.0）
        use_gpu: 是否使用GPU加速（默认: False）
        device: PyTorch设备（如果为None且use_gpu=True，自动检测）
    
    Returns:
        (S_cons, S_ood): 两个numpy数组，形状均为 (N_samples,)
    """
    # 转换为torch张量（如果需要GPU）或numpy数组
    if use_gpu and torch.cuda.is_available():
        if device is None:
            device = torch.device('cuda')
        
        if isinstance(f_B, np.ndarray):
            f_B = torch.from_numpy(f_B).float().to(device)
        if isinstance(f_R, np.ndarray):
            f_R = torch.from_numpy(f_R).float().to(device)
        
        return _calculate_layer_scores_gpu(f_B, f_R, K, alpha)
    else:
        # CPU版本：转换为numpy数组
        if isinstance(f_B, torch.Tensor):
            f_B = f_B.cpu().numpy()
        if isinstance(f_R, torch.Tensor):
            f_R = f_R.cpu().numpy()
        
        return _calculate_layer_scores_cpu(f_B, f_R, K, alpha)


def _calculate_layer_scores_cpu(
    f_B: np.ndarray,
    f_R: np.ndarray,
    K: int,
    alpha: float
) -> Tuple[np.ndarray, np.ndarray]:
    """CPU版本的层得分计算"""
    N_samples, N_components = f_B.shape
    
    # --- 1. S_ood 计算 ---
    P_known = 0.5 * f_B + 0.5 * f_R
    S_ood = np.sum(-np.log(P_known + EPSILON), axis=1)

    # --- 2. LLR 计算 ---
    ratio = (f_R + EPSILON) / (f_B + EPSILON)
    raw_lambda = np.log(ratio)
    
    # --- 3. Top-K 稀疏特征提取 ---
    K_actual = min(K, N_components)
    
    if K_actual == 0:
        return np.zeros(N_samples), S_ood

    sorted_lambda = np.sort(raw_lambda, axis=1)
    
    top_k_pos = sorted_lambda[:, -K_actual:]
    top_k_pos = np.maximum(0, top_k_pos)
    v_pos = np.mean(top_k_pos, axis=1)
    
    bottom_k_neg = sorted_lambda[:, :K_actual]
    bottom_k_neg = np.minimum(0, bottom_k_neg)
    v_neg = np.mean(bottom_k_neg, axis=1)
    
    S_cons = v_pos + (alpha * v_neg)
    
    return S_cons, S_ood


def _calculate_layer_scores_gpu(
    f_B: torch.Tensor,
    f_R: torch.Tensor,
    K: int,
    alpha: float
) -> Tuple[np.ndarray, np.ndarray]:
    """GPU版本的层得分计算"""
    N_samples, N_components = f_B.shape
    device = f_B.device
    eps_tensor = torch.tensor(EPSILON, device=device, dtype=torch.float32)
    
    # --- 1. S_ood 计算 ---
    P_known = 0.5 * f_B + 0.5 * f_R
    S_ood = torch.sum(-torch.log(P_known + eps_tensor), dim=1)

    # --- 2. LLR 计算 ---
    ratio = (f_R + eps_tensor) / (f_B + eps_tensor)
    raw_lambda = torch.log(ratio)
    
    # --- 3. Top-K 稀疏特征提取 ---
    K_actual = min(K, N_components)
    
    if K_actual == 0:
        zeros = torch.zeros(N_samples, device=device)
        return zeros.cpu().numpy(), S_ood.cpu().numpy()

    sorted_lambda, _ = torch.sort(raw_lambda, dim=1)
    
    top_k_pos = sorted_lambda[:, -K_actual:]
    top_k_pos = torch.clamp(top_k_pos, min=0.0)
    v_pos = torch.mean(top_k_pos, dim=1)
    
    bottom_k_neg = sorted_lambda[:, :K_actual]
    bottom_k_neg = torch.clamp(bottom_k_neg, max=0.0)
    v_neg = torch.mean(bottom_k_neg, dim=1)
    
    S_cons = v_pos + (alpha * v_neg)
    
    return S_cons.cpu().numpy(), S_ood.cpu().numpy()
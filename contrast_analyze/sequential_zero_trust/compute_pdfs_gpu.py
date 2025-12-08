# -*- coding: utf-8 -*-
"""
compute_pdfs_gpu.py

GPU加速的概率密度函数（PDF）计算。
使用PyTorch在GPU上批量计算PDF，大幅提升性能。
"""

import numpy as np
import torch
import warnings
from typing import Dict, Tuple, Optional
from scipy import stats


EPSILON = 1e-10


def compute_pdfs_gpu(
    activations: np.ndarray,
    dist_params: Dict[str, Dict],
    device: Optional[torch.device] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    GPU加速的向量化PDF计算。
    
    使用PyTorch在GPU上批量计算PDF，比CPU版本快10-100倍。
    
    Args:
        activations: 激活值数组，形状为 (N_samples, N_components)
        dist_params: 分布参数字典，格式与 compute_pdfs 相同
        device: PyTorch设备（如果为None，自动检测）
    
    Returns:
        (f_B, f_R): 两个数组，形状均为 (N_samples, N_components)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    N_samples, N_components = activations.shape
    
    # 转换为PyTorch张量并移到GPU
    activations_tensor = torch.from_numpy(activations).float().to(device)
    
    # 初始化输出数组
    f_B_tensor = torch.zeros((N_samples, N_components), device=device, dtype=torch.float32)
    f_R_tensor = torch.zeros((N_samples, N_components), device=device, dtype=torch.float32)
    
    # 获取组件键列表
    component_keys = list(dist_params.keys())
    
    if len(component_keys) != N_components:
        raise ValueError(
            f"组件数量不匹配: dist_params 有 {len(component_keys)} 个组件，"
            f"但 activations 有 {N_components} 列"
        )
    
    # 对每个组件批量计算PDF
    for comp_idx, comp_key in enumerate(component_keys):
        comp_params = dist_params[comp_key]
        
        # 获取 benign 和 fail 的分布参数
        benign_params = comp_params.get('benign', {})
        fail_params = comp_params.get('fail', {})
        
        benign_dist = benign_params.get('best_dist_name')
        benign_params_list = benign_params.get('best_params', [])
        benign_data_min = benign_params.get('data_min')
        benign_data_max = benign_params.get('data_max')
        
        fail_dist = fail_params.get('best_dist_name')
        fail_params_list = fail_params.get('best_params', [])
        fail_data_min = fail_params.get('data_min')
        fail_data_max = fail_params.get('data_max')
        
        # 获取该组件的所有激活值（批量处理）
        values = activations_tensor[:, comp_idx]  # (N_samples,)
        
        # 批量计算 benign PDF
        f_B_tensor[:, comp_idx] = compute_pdf_batch_gpu(
            values, benign_dist, benign_params_list,
            benign_data_min, benign_data_max, device
        )
        
        # 批量计算 refusal (fail) PDF
        f_R_tensor[:, comp_idx] = compute_pdf_batch_gpu(
            values, fail_dist, fail_params_list,
            fail_data_min, fail_data_max, device
        )
    
    # 转换回numpy数组
    f_B = f_B_tensor.cpu().numpy()
    f_R = f_R_tensor.cpu().numpy()
    
    return f_B, f_R


def compute_pdf_batch_gpu(
    values: torch.Tensor,
    dist_name: Optional[str],
    params: list,
    data_min: Optional[float] = None,
    data_max: Optional[float] = None,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    在GPU上批量计算PDF。
    
    Args:
        values: 值张量，形状为 (N_samples,)
        dist_name: 分布名称
        params: 分布参数列表
        data_min: Beta分布的最小值
        data_max: Beta分布的最大值
        device: PyTorch设备
    
    Returns:
        PDF值张量，形状为 (N_samples,)
    """
    if device is None:
        device = values.device
    
    if dist_name is None or params is None or len(params) == 0:
        return torch.full_like(values, EPSILON)
    
    try:
        # 处理 Beta 分布的特殊情况
        if dist_name == 'beta':
            if data_min is None or data_max is None:
                return torch.full_like(values, EPSILON)
            
            data_min_t = torch.tensor(data_min, device=device, dtype=torch.float32)
            data_max_t = torch.tensor(data_max, device=device, dtype=torch.float32)
            
            if abs(data_max - data_min) < EPSILON:
                return torch.full_like(values, EPSILON)
            
            # 归一化到 [0, 1]
            values_norm = (values - data_min_t) / (data_max_t - data_min_t + EPSILON)
            values_norm = torch.clamp(values_norm, 0.0, 1.0)
            
            if len(params) >= 2:
                shape1, shape2 = float(params[0]), float(params[1])
                # 使用PyTorch的Beta分布
                from torch.distributions import Beta
                # 确保参数为正数
                shape1 = max(shape1, 1e-6)
                shape2 = max(shape2, 1e-6)
                dist = Beta(torch.tensor(shape1, device=device, dtype=torch.float32), 
                          torch.tensor(shape2, device=device, dtype=torch.float32))
                log_pdf = dist.log_prob(values_norm)
            else:
                return torch.full_like(values, EPSILON)
        else:
            values_norm = values
            # 对于其他分布，使用scipy计算（因为PyTorch不支持所有分布）
            # 但我们可以批量处理以提高效率
            values_np = values_norm.cpu().numpy()
            pdf_np = compute_pdf_batch_scipy(values_np, dist_name, params)
            log_pdf = torch.from_numpy(np.log(np.maximum(pdf_np, EPSILON))).to(device)
        
        # 避免 -inf
        log_pdf = torch.clamp(log_pdf, -100, None)
        pdf = torch.exp(log_pdf)
        # 确保不为 0
        pdf = torch.maximum(pdf, torch.tensor(EPSILON, device=device))
        
        return pdf
    
    except Exception as e:
        return torch.full_like(values, EPSILON)


def compute_pdf_batch_scipy(
    values: np.ndarray,
    dist_name: str,
    params: list
) -> np.ndarray:
    """
    使用scipy批量计算PDF（用于PyTorch不支持的分布）。
    
    这个函数仍然比逐个计算快，因为scipy的分布对象可以处理数组。
    """
    if dist_name is None or params is None:
        return np.full_like(values, EPSILON)
    
    try:
        dist_map = {
            'norm': stats.norm,
            't': stats.t,
            'gamma': stats.gamma,
            'lognorm': stats.lognorm,
            'laplace': stats.laplace,
        }
        
        if dist_name not in dist_map:
            return np.full_like(values, EPSILON)
        
        dist = dist_map[dist_name]
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            try:
                log_pdf = dist.logpdf(values, *params)
                log_pdf = np.clip(log_pdf, -100, None)
                pdf = np.exp(log_pdf)
                pdf = np.maximum(pdf, EPSILON)
            except:
                pdf = np.full_like(values, EPSILON)
        
        return pdf
    
    except Exception as e:
        return np.full_like(values, EPSILON)


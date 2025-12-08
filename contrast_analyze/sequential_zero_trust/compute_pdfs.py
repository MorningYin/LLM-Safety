# -*- coding: utf-8 -*-
"""
compute_pdfs.py

计算概率密度函数（PDF）的核心函数。
支持多种分布类型：norm, t, gamma, lognorm, laplace, beta。
"""

import numpy as np
import warnings
from typing import Dict, List, Tuple
from scipy import stats


EPSILON = 1e-10


def compute_pdf_single(
    value: float,
    dist_name: str,
    params: List[float],
    data_min: float = None,
    data_max: float = None
) -> float:
    """
    计算单个值在给定分布下的 PDF。
    
    Args:
        value: 要计算 PDF 的值
        dist_name: 分布名称（'norm', 't', 'gamma', 'lognorm', 'laplace', 'beta'）
        params: 分布参数列表
        data_min: Beta 分布的最小值（仅用于 beta 分布）
        data_max: Beta 分布的最大值（仅用于 beta 分布）
    
    Returns:
        PDF 值（概率密度）
    """
    if dist_name is None or params is None:
        return EPSILON
    
    try:
        # 获取分布对象
        dist_map = {
            'norm': stats.norm,
            't': stats.t,
            'gamma': stats.gamma,
            'lognorm': stats.lognorm,
            'laplace': stats.laplace,
            'beta': stats.beta,
        }
        
        if dist_name not in dist_map:
            return EPSILON
        
        dist = dist_map[dist_name]
        
        # 处理 Beta 分布的特殊情况
        if dist_name == 'beta':
            if data_min is None or data_max is None:
                return EPSILON
            if abs(data_max - data_min) < EPSILON:
                return EPSILON
            # 归一化到 [0, 1]
            value_norm = (value - data_min) / (data_max - data_min + EPSILON)
            # 确保在 [0, 1] 范围内
            value_norm = np.clip(value_norm, 0.0, 1.0)
            # 使用归一化后的参数计算 PDF
            # fit_params 格式: (shape1, shape2, loc=0, scale=1, data_min, data_max)
            # 实际参数是前两个
            if len(params) >= 2:
                pdf_params = params[:2]  # shape1, shape2
            else:
                return EPSILON
        else:
            value_norm = value
            pdf_params = params
        
        # 使用 logpdf 提高数值稳定性
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            try:
                log_pdf = dist.logpdf(value_norm, *pdf_params)
                # 避免 -inf
                log_pdf = np.clip(log_pdf, -100, None)
                pdf = np.exp(log_pdf)
                # 确保不为 0
                pdf = np.maximum(pdf, EPSILON)
            except:
                pdf = EPSILON
        
        return float(pdf)
    
    except Exception as e:
        return EPSILON


def compute_pdfs(
    activations: np.ndarray,
    dist_params: Dict[str, Dict]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    向量化计算 PDF。
    
    Args:
        activations: 激活值数组，形状为 (N_samples, N_components)
        dist_params: 分布参数字典，格式为：
            {
                'component_key': {
                    'benign': {
                        'best_dist_name': 'norm',
                        'best_params': [...],
                        'data_min': ...,
                        'data_max': ...
                    },
                    'fail': {
                        'best_dist_name': 'norm',
                        'best_params': [...],
                        'data_min': ...,
                        'data_max': ...
                    }
                },
                ...
            }
    
    Returns:
        (f_B, f_R): 两个数组，形状均为 (N_samples, N_components)
            - f_B: benign 假设下的 PDF
            - f_R: refusal (fail) 假设下的 PDF
    """
    N_samples, N_components = activations.shape
    
    # 初始化输出数组
    f_B = np.zeros((N_samples, N_components))
    f_R = np.zeros((N_samples, N_components))
    
    # 获取组件键列表（需要与 activations 的列顺序匹配）
    component_keys = list(dist_params.keys())
    
    if len(component_keys) != N_components:
        raise ValueError(
            f"组件数量不匹配: dist_params 有 {len(component_keys)} 个组件，"
            f"但 activations 有 {N_components} 列"
        )
    
    # 对每个组件分别计算 PDF
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
        
        # 对每个样本计算 PDF
        for sample_idx in range(N_samples):
            value = activations[sample_idx, comp_idx]
            
            # 计算 benign PDF
            f_B[sample_idx, comp_idx] = compute_pdf_single(
                value, benign_dist, benign_params_list,
                benign_data_min, benign_data_max
            )
            
            # 计算 refusal (fail) PDF
            f_R[sample_idx, comp_idx] = compute_pdf_single(
                value, fail_dist, fail_params_list,
                fail_data_min, fail_data_max
            )
    
    return f_B, f_R





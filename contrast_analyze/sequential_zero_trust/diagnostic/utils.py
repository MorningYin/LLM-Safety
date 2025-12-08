# -*- coding: utf-8 -*-
"""
diagnostic/utils.py

诊断工具函数：LLR 计算、统计计算、可视化。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from scipy import stats

# 导入 compute_pdf_single 用于计算单个 PDF
from ..compute_pdfs import compute_pdf_single

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 科研风格设置
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13

EPSILON = 1e-10


def compute_p_known(f_B: np.ndarray, f_R: np.ndarray) -> np.ndarray:
    """
    计算 P_known^{(i)}(x_i) = 0.5 * f_B^{(i)}(x_i) + 0.5 * f_R^{(i)}(x_i)
    
    Args:
        f_B: Benign 假设下的 PDF，形状为 (N_samples, N_components)
        f_R: Refusal (fail) 假设下的 PDF，形状为 (N_samples, N_components)
    
    Returns:
        P_known 矩阵，形状为 (N_samples, N_components)
    """
    P_known = 0.5 * f_B + 0.5 * f_R
    return P_known


def fit_distribution(
    data: np.ndarray,
    dist_names: Optional[List[str]] = None,
    min_samples: int = 10
) -> Optional[Dict]:
    """
    对给定数据进行分布拟合，尝试多种分布并选择最佳拟合。
    
    Args:
        data: 一维数据数组
        dist_names: 要尝试的分布名称列表，如果为 None 则使用默认列表
        min_samples: 最小样本数，如果数据量小于此值则返回 None
    
    Returns:
        分布拟合结果字典，包含：
        - best_dist_name: 最佳分布名称
        - best_params: 最佳分布参数列表
        - aic: AIC 值
        - bic: BIC 值
        - ks_statistic: KS 统计量
        - ks_pvalue: KS 检验 p 值
        如果拟合失败或数据量不足，返回 None
    """
    data = np.array(data).flatten()
    data = data[~np.isnan(data)]
    data = data[~np.isinf(data)]
    
    if len(data) < min_samples:
        return None
    
    if dist_names is None:
        # 根据数据特征选择分布
        if np.all(data > 0):
            # 如果所有值都为正，尝试这些分布
            dist_names = ['gamma', 'lognorm', 'beta', 'norm', 'laplace']
        else:
            # 如果值可能为负，尝试这些分布
            dist_names = ['norm', 't', 'laplace', 'gamma', 'lognorm']
    
    best_dist_name = None
    best_params = None
    best_aic = np.inf
    best_bic = np.inf
    best_ks_stat = np.inf
    best_ks_pvalue = 0.0
    
    dist_map = {
        'norm': stats.norm,
        't': stats.t,
        'gamma': stats.gamma,
        'lognorm': stats.lognorm,
        'laplace': stats.laplace,
        'beta': stats.beta,
    }
    
    for dist_name in dist_names:
        if dist_name not in dist_map:
            continue
        
        dist = dist_map[dist_name]
        
        try:
            # 拟合分布参数
            if dist_name == 'beta':
                # Beta 分布需要归一化到 [0, 1]
                data_min, data_max = data.min(), data.max()
                if abs(data_max - data_min) < EPSILON:
                    continue
                data_norm = (data - data_min) / (data_max - data_min + EPSILON)
                data_norm = np.clip(data_norm, EPSILON, 1.0 - EPSILON)
                params = dist.fit(data_norm, floc=0, fscale=1)
                # 保存归一化参数
                params = list(params[:2]) + [data_min, data_max]
            else:
                params = dist.fit(data)
            
            # 计算 AIC 和 BIC
            n = len(data)
            k = len(params) if dist_name != 'beta' else 2  # Beta 分布只有 2 个形状参数
            
            # 计算对数似然
            try:
                if dist_name == 'beta':
                    data_norm = (data - params[2]) / (params[3] - params[2] + EPSILON)
                    data_norm = np.clip(data_norm, EPSILON, 1.0 - EPSILON)
                    loglik = np.sum(dist.logpdf(data_norm, *params[:2]))
                else:
                    loglik = np.sum(dist.logpdf(data, *params))
                
                aic = 2 * k - 2 * loglik
                bic = k * np.log(n) - 2 * loglik
            except:
                aic = np.inf
                bic = np.inf
            
            # KS 检验
            try:
                if dist_name == 'beta':
                    data_norm = (data - params[2]) / (params[3] - params[2] + EPSILON)
                    data_norm = np.clip(data_norm, EPSILON, 1.0 - EPSILON)
                    ks_stat, ks_pvalue = stats.kstest(data_norm, lambda x: dist.cdf(x, *params[:2]))
                else:
                    ks_stat, ks_pvalue = stats.kstest(data, lambda x: dist.cdf(x, *params))
            except:
                ks_stat = np.inf
                ks_pvalue = 0.0
            
            # 选择 AIC 最小的分布
            if aic < best_aic:
                best_dist_name = dist_name
                best_params = [float(p) for p in params]
                best_aic = float(aic)
                best_bic = float(bic)
                best_ks_stat = float(ks_stat)
                best_ks_pvalue = float(ks_pvalue)
        
        except Exception:
            continue
    
    if best_dist_name is None:
        return None
    
    return {
        "best_dist_name": best_dist_name,
        "best_params": best_params,
        "aic": best_aic,
        "bic": best_bic,
        "ks_statistic": best_ks_stat,
        "ks_pvalue": best_ks_pvalue
    }


def compute_basic_stats(data: np.ndarray) -> Dict[str, float]:
    """
    计算基本统计量。
    
    Args:
        data: 一维数据数组
    
    Returns:
        统计量字典
    """
    data = np.array(data).flatten()
    data = data[~np.isnan(data)]
    data = data[~np.isinf(data)]
    
    if len(data) == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
            "q25": 0.0,
            "q75": 0.0
        }
    
    return {
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "median": float(np.median(data)),
        "q25": float(np.percentile(data, 25)),
        "q75": float(np.percentile(data, 75))
    }


def calculate_llr_matrix(f_B: np.ndarray, f_R: np.ndarray) -> np.ndarray:
    """
    计算 LLR Matrix: lambda_i = log(f_R^(i)(x_i) / f_B^(i)(x_i))
    
    Args:
        f_B: Benign 假设下的 PDF，形状为 (N_samples, N_components)
        f_R: Refusal (fail) 假设下的 PDF，形状为 (N_samples, N_components)
    
    Returns:
        llr_matrix: LLR 矩阵，形状为 (N_samples, N_components)
    """
    ratio = f_R / (f_B + EPSILON)
    llr_matrix = np.log(ratio + EPSILON)
    return llr_matrix


def compute_dataset_stats(
    llr_matrix: np.ndarray,
    activations: Optional[np.ndarray] = None,
    f_B: Optional[np.ndarray] = None,
    f_R: Optional[np.ndarray] = None,
    P_known: Optional[np.ndarray] = None,
    lambda_i: Optional[np.ndarray] = None
) -> Dict:
    """
    计算数据集级别的统计量。
    
    Args:
        llr_matrix: LLR 矩阵，形状为 (N_samples, N_components)
        activations: 激活值矩阵，形状为 (N_samples, N_components)（可选）
        f_B: Benign PDF 矩阵，形状为 (N_samples, N_components)（可选）
        f_R: Refusal PDF 矩阵，形状为 (N_samples, N_components)（可选）
        P_known: P_known 矩阵，形状为 (N_samples, N_components)（可选）
        lambda_i: lambda_i 矩阵，形状为 (N_samples, N_components)（可选）
    
    Returns:
        统计量字典，包含：
        - llr: 现有的 LLR 统计（保持兼容）
        - activations: 激活值统计（如果提供）
        - f_B: f_B 统计（如果提供）
        - f_R: f_R 统计（如果提供）
        - P_known: P_known 统计和分布拟合（如果提供）
        - lambda_i: lambda_i 统计和分布拟合（如果提供）
    """
    stats_dict = {}
    
    # 1. LLR 统计（保持向后兼容）
    all_llr = llr_matrix.flatten()
    mean_llr = float(np.mean(all_llr))
    std_llr = float(np.std(all_llr))
    
    positive_llr = all_llr[all_llr > 0]
    if len(positive_llr) > 0:
        top_5_pos = np.sort(positive_llr)[-5:]
        top_5_pos_mean = float(np.mean(top_5_pos))
    else:
        top_5_pos_mean = 0.0
    
    negative_llr = all_llr[all_llr < 0]
    if len(negative_llr) > 0:
        bottom_5_neg = np.sort(negative_llr)[:5]
        bottom_5_neg_mean = float(np.mean(bottom_5_neg))
    else:
        bottom_5_neg_mean = 0.0
    
    stats_dict["llr"] = {
        "mean_llr": mean_llr,
        "std_llr": std_llr,
        "top_5_pos_mean": top_5_pos_mean,
        "bottom_5_neg_mean": bottom_5_neg_mean
    }
    
    # 2. 激活值统计
    if activations is not None:
        stats_dict["activations"] = compute_basic_stats(activations)
    
    # 3. f_B 统计
    if f_B is not None:
        stats_dict["f_B"] = compute_basic_stats(f_B)
    
    # 4. f_R 统计
    if f_R is not None:
        stats_dict["f_R"] = compute_basic_stats(f_R)
    
    # 5. P_known 统计和分布拟合
    if P_known is not None:
        p_known_flat = P_known.flatten()
        stats_dict["P_known"] = compute_basic_stats(p_known_flat)
        
        # 分布拟合
        dist_fit = fit_distribution(p_known_flat)
        if dist_fit is not None:
            stats_dict["P_known"]["distribution_fit"] = dist_fit
        else:
            stats_dict["P_known"]["distribution_fit"] = None
    
    # 6. lambda_i 统计和分布拟合
    if lambda_i is not None:
        lambda_flat = lambda_i.flatten()
        stats_dict["lambda_i"] = compute_basic_stats(lambda_flat)
        
        # 分布拟合
        dist_fit = fit_distribution(lambda_flat)
        if dist_fit is not None:
            stats_dict["lambda_i"]["distribution_fit"] = dist_fit
        else:
            stats_dict["lambda_i"]["distribution_fit"] = None
    
    return stats_dict


def compute_component_stats(
    llr_benign: np.ndarray,
    llr_refusal: np.ndarray,
    llr_jailbreak: np.ndarray,
    component_names: List[str],
    activations_benign: Optional[np.ndarray] = None,
    activations_refusal: Optional[np.ndarray] = None,
    activations_jailbreak: Optional[np.ndarray] = None,
    f_B_benign: Optional[np.ndarray] = None,
    f_B_refusal: Optional[np.ndarray] = None,
    f_B_jailbreak: Optional[np.ndarray] = None,
    f_R_benign: Optional[np.ndarray] = None,
    f_R_refusal: Optional[np.ndarray] = None,
    f_R_jailbreak: Optional[np.ndarray] = None,
    P_known_benign: Optional[np.ndarray] = None,
    P_known_refusal: Optional[np.ndarray] = None,
    P_known_jailbreak: Optional[np.ndarray] = None,
    lambda_i_benign: Optional[np.ndarray] = None,
    lambda_i_refusal: Optional[np.ndarray] = None,
    lambda_i_jailbreak: Optional[np.ndarray] = None,
    distribution_params: Optional[List[Dict]] = None
) -> Tuple[List[Dict], List[int]]:
    """
    计算组件级别的统计量，并识别 Top-3 组件。
    
    Args:
        llr_benign: Benign 数据集的 LLR 矩阵，形状为 (N_samples, N_components)
        llr_refusal: Refusal 数据集的 LLR 矩阵，形状为 (N_samples, N_components)
        llr_jailbreak: Jailbreak 数据集的 LLR 矩阵，形状为 (N_samples, N_components)
        component_names: 组件名称列表
        activations_benign/refusal/jailbreak: 激活值矩阵（可选）
        f_B_benign/refusal/jailbreak: Benign PDF 矩阵（可选）
        f_R_benign/refusal/jailbreak: Refusal PDF 矩阵（可选）
        P_known_benign/refusal/jailbreak: P_known 矩阵（可选）
        lambda_i_benign/refusal/jailbreak: lambda_i 矩阵（可选）
        distribution_params: 每个组件的分布参数字典列表（可选）
    
    Returns:
        (component_stats_list, top_3_indices):
            - component_stats_list: 每个组件的统计信息列表
            - top_3_indices: Top-3 组件的索引列表
    """
    N_components = len(component_names)
    
    # 计算每个组件的平均 LLR
    refusal_means = np.mean(llr_refusal, axis=0)
    benign_means = np.mean(llr_benign, axis=0)
    jailbreak_means = np.mean(llr_jailbreak, axis=0)
    
    # 计算区分度：Mean LLR 差异的绝对值
    discriminative_scores = np.abs(refusal_means - benign_means)
    
    # 选择 Top-3 组件（区分度最大）
    top_3_indices = np.argsort(discriminative_scores)[-3:][::-1].tolist()
    
    # 构建组件统计列表
    component_stats_list = []
    for comp_idx, comp_name in enumerate(component_names):
        is_top = comp_idx in top_3_indices
        comp_stats = {
            "name": comp_name,
            "refusal_mean_llr": float(refusal_means[comp_idx]),
            "benign_mean_llr": float(benign_means[comp_idx]),
            "jailbreak_mean_llr": float(jailbreak_means[comp_idx]),
            "is_top_component": is_top
        }
        
        # 添加激活值统计
        if activations_benign is not None:
            comp_stats["activations"] = {
                "benign": compute_basic_stats(activations_benign[:, comp_idx]),
                "refusal": compute_basic_stats(activations_refusal[:, comp_idx]),
                "jailbreak": compute_basic_stats(activations_jailbreak[:, comp_idx])
            }
        
        # 添加 f_B 统计
        if f_B_benign is not None:
            comp_stats["f_B"] = {
                "benign": compute_basic_stats(f_B_benign[:, comp_idx]),
                "refusal": compute_basic_stats(f_B_refusal[:, comp_idx]),
                "jailbreak": compute_basic_stats(f_B_jailbreak[:, comp_idx])
            }
        
        # 添加 f_R 统计
        if f_R_benign is not None:
            comp_stats["f_R"] = {
                "benign": compute_basic_stats(f_R_benign[:, comp_idx]),
                "refusal": compute_basic_stats(f_R_refusal[:, comp_idx]),
                "jailbreak": compute_basic_stats(f_R_jailbreak[:, comp_idx])
            }
        
        # 添加 P_known 统计和分布拟合
        if P_known_benign is not None:
            p_known_benign = P_known_benign[:, comp_idx]
            p_known_refusal = P_known_refusal[:, comp_idx]
            p_known_jailbreak = P_known_jailbreak[:, comp_idx]
            
            comp_stats["P_known"] = {
                "benign": compute_basic_stats(p_known_benign),
                "refusal": compute_basic_stats(p_known_refusal),
                "jailbreak": compute_basic_stats(p_known_jailbreak)
            }
            
            # 分布拟合
            dist_fit_benign = fit_distribution(p_known_benign)
            dist_fit_refusal = fit_distribution(p_known_refusal)
            dist_fit_jailbreak = fit_distribution(p_known_jailbreak)
            
            comp_stats["P_known"]["benign"]["distribution_fit"] = dist_fit_benign
            comp_stats["P_known"]["refusal"]["distribution_fit"] = dist_fit_refusal
            comp_stats["P_known"]["jailbreak"]["distribution_fit"] = dist_fit_jailbreak
        
        # 添加 lambda_i 统计和分布拟合
        if lambda_i_benign is not None:
            lambda_benign = lambda_i_benign[:, comp_idx]
            lambda_refusal = lambda_i_refusal[:, comp_idx]
            lambda_jailbreak = lambda_i_jailbreak[:, comp_idx]
            
            comp_stats["lambda_i"] = {
                "benign": compute_basic_stats(lambda_benign),
                "refusal": compute_basic_stats(lambda_refusal),
                "jailbreak": compute_basic_stats(lambda_jailbreak)
            }
            
            # 分布拟合
            dist_fit_benign = fit_distribution(lambda_benign)
            dist_fit_refusal = fit_distribution(lambda_refusal)
            dist_fit_jailbreak = fit_distribution(lambda_jailbreak)
            
            comp_stats["lambda_i"]["benign"]["distribution_fit"] = dist_fit_benign
            comp_stats["lambda_i"]["refusal"]["distribution_fit"] = dist_fit_refusal
            comp_stats["lambda_i"]["jailbreak"]["distribution_fit"] = dist_fit_jailbreak
        
        # 添加分布拟合参数（如果提供）
        if distribution_params is not None and comp_idx < len(distribution_params):
            comp_stats["distribution_params"] = distribution_params[comp_idx]
        
        component_stats_list.append(comp_stats)
    
    return component_stats_list, top_3_indices


def plot_llr_violin(
    llr_benign: np.ndarray,
    llr_refusal: np.ndarray,
    llr_jailbreak: np.ndarray,
    output_path: Path,
    title: str = "LLR Distribution Comparison"
):
    """
    绘制 LLR Violin Plot，展示三个数据集的 lambda 分布。
    
    Args:
        llr_benign: Benign 数据集的 LLR 矩阵，形状为 (N_samples, N_components)
        llr_refusal: Refusal 数据集的 LLR 矩阵，形状为 (N_samples, N_components)
        llr_jailbreak: Jailbreak 数据集的 LLR 矩阵，形状为 (N_samples, N_components)
        output_path: 输出文件路径
        title: 图表标题
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 准备数据
    data_list = []
    labels_list = []
    
    # 展平每个数据集的 LLR
    for llr_matrix, label in [
        (llr_benign, "Benign"),
        (llr_refusal, "Refusal"),
        (llr_jailbreak, "Jailbreak")
    ]:
        flat_llr = llr_matrix.flatten()
        data_list.append(flat_llr)
        labels_list.append(label)
    
    # 绘制 Violin Plot
    parts = ax.violinplot(
        data_list,
        positions=[0, 1, 2],
        showmeans=True,
        showmedians=True,
        widths=0.6
    )
    
    # 设置颜色
    colors = ['#0066CC', '#FF6600', '#00AA00']
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    
    # 设置标签
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(labels_list)
    ax.set_ylabel("LLR (Log-Likelihood Ratio)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def plot_component_heatmap(
    llr_matrix: np.ndarray,
    component_names: List[str],
    output_path: Path,
    n_samples: int = 50,
    title: str = "Component-Sample LLR Heatmap"
):
    """
    绘制 Component-Sample Heatmap。
    
    Args:
        llr_matrix: LLR 矩阵，形状为 (N_samples, N_components)
        component_names: 组件名称列表
        output_path: 输出文件路径
        n_samples: 要显示的样本数量（采样）
        title: 图表标题
    """
    N_samples, N_components = llr_matrix.shape
    
    # 采样样本
    if N_samples > n_samples:
        sample_indices = np.random.choice(N_samples, n_samples, replace=False)
        sampled_llr = llr_matrix[sample_indices, :]
    else:
        sampled_llr = llr_matrix
    
    # 按 Mean Refusal LLR 对组件排序
    component_means = np.mean(sampled_llr, axis=0)
    sorted_indices = np.argsort(component_means)[::-1]  # 降序
    
    sorted_llr = sampled_llr[:, sorted_indices]
    sorted_names = [component_names[i] for i in sorted_indices]
    
    # 绘制热图
    fig, ax = plt.subplots(figsize=(max(12, N_components * 0.3), max(8, n_samples * 0.15)))
    
    im = ax.imshow(
        sorted_llr.T,
        aspect='auto',
        cmap='RdBu_r',
        vmin=-np.abs(sorted_llr).max(),
        vmax=np.abs(sorted_llr).max(),
        interpolation='nearest'
    )
    
    # 设置标签
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Component (sorted by Mean LLR)")
    ax.set_title(title)
    
    # 设置组件标签（只显示部分，避免过于拥挤）
    if N_components <= 20:
        ax.set_yticks(range(N_components))
        ax.set_yticklabels(sorted_names, fontsize=8)
    else:
        # 只显示部分标签
        step = max(1, N_components // 20)
        ax.set_yticks(range(0, N_components, step))
        ax.set_yticklabels([sorted_names[i] for i in range(0, N_components, step)], fontsize=8)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("LLR", rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def plot_component_fit(
    activations: np.ndarray,
    dist_params: Dict,
    component_name: str,
    output_path: Path,
    title: Optional[str] = None
):
    """
    绘制组件的分布拟合图：原始激活 Histogram + 拟合 PDF 曲线。
    
    Args:
        activations: 激活值数组，形状为 (N_samples,)
        dist_params: 分布参数字典，格式为：
            {
                'benign': {
                    'best_dist_name': 'norm',
                    'best_params': [...],
                    'data_min': ...,
                    'data_max': ...
                },
                'fail': { ... }
            }
        component_name: 组件名称
        output_path: 输出文件路径
        title: 图表标题（可选）
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制直方图
    ax.hist(
        activations,
        bins=50,
        density=True,
        alpha=0.6,
        color='gray',
        label='Observed Data'
    )
    
    # 绘制拟合 PDF 曲线
    x_min, x_max = activations.min(), activations.max()
    x_range = np.linspace(x_min, x_max, 200)
    
    # Benign 分布
    benign_params = dist_params.get('benign', {})
    benign_dist_name = benign_params.get('best_dist_name')
    benign_params_list = benign_params.get('best_params', [])
    benign_data_min = benign_params.get('data_min')
    benign_data_max = benign_params.get('data_max')
    
    if benign_dist_name and benign_params_list:
        benign_pdf = _compute_pdf_curve(
            x_range, benign_dist_name, benign_params_list,
            benign_data_min, benign_data_max
        )
        ax.plot(x_range, benign_pdf, 'b-', linewidth=2, label='Benign Fit', alpha=0.8)
    
    # Fail (Refusal) 分布
    fail_params = dist_params.get('fail', {})
    fail_dist_name = fail_params.get('best_dist_name')
    fail_params_list = fail_params.get('best_params', [])
    fail_data_min = fail_params.get('data_min')
    fail_data_max = fail_params.get('data_max')
    
    if fail_dist_name and fail_params_list:
        fail_pdf = _compute_pdf_curve(
            x_range, fail_dist_name, fail_params_list,
            fail_data_min, fail_data_max
        )
        ax.plot(x_range, fail_pdf, 'r-', linewidth=2, label='Refusal Fit', alpha=0.8)
    
    # 设置标签和标题
    if title is None:
        title = f"Distribution Fit: {component_name}"
    ax.set_title(title)
    ax.set_xlabel("Activation Value")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def _compute_pdf_curve(
    x: np.ndarray,
    dist_name: str,
    params: List[float],
    data_min: Optional[float] = None,
    data_max: Optional[float] = None
) -> np.ndarray:
    """
    计算 PDF 曲线（用于绘图）。
    
    Args:
        x: x 值数组
        dist_name: 分布名称
        params: 分布参数
        data_min: Beta 分布的最小值
        data_max: Beta 分布的最大值
    
    Returns:
        PDF 值数组
    """
    if dist_name is None or params is None:
        return np.zeros_like(x)
    
    try:
        dist_map = {
            'norm': stats.norm,
            't': stats.t,
            'gamma': stats.gamma,
            'lognorm': stats.lognorm,
            'laplace': stats.laplace,
            'beta': stats.beta,
        }
        
        if dist_name not in dist_map:
            return np.zeros_like(x)
        
        dist = dist_map[dist_name]
        
        # 处理 Beta 分布
        if dist_name == 'beta':
            if data_min is None or data_max is None:
                return np.zeros_like(x)
            if abs(data_max - data_min) < EPSILON:
                return np.zeros_like(x)
            # 归一化到 [0, 1]
            x_norm = (x - data_min) / (data_max - data_min + EPSILON)
            x_norm = np.clip(x_norm, 0.0, 1.0)
            pdf_params = params[:2] if len(params) >= 2 else []
            if len(pdf_params) < 2:
                return np.zeros_like(x)
            pdf = dist.pdf(x_norm, *pdf_params)
            # 反归一化
            pdf = pdf / (data_max - data_min + EPSILON)
        else:
            pdf_params = params
            pdf = dist.pdf(x, *pdf_params)
        
        # 避免无效值
        pdf = np.nan_to_num(pdf, nan=0.0, posinf=0.0, neginf=0.0)
        pdf = np.maximum(pdf, EPSILON)
        
        return pdf
    
    except Exception:
        return np.zeros_like(x)


def plot_component_distributions(
    activations_benign: np.ndarray,
    activations_refusal: np.ndarray,
    activations_jailbreak: np.ndarray,
    dist_params: Dict,
    component_name: str,
    output_path: Path,
    title: Optional[str] = None
):
    """
    绘制组件的三个分布对比图：
    1. 原始分布（benign/fail/success 拟合曲线）
    2. P_known 分布
    3. lambda_i 分布
    
    Args:
        activations_benign: benign 数据集的激活值数组，形状为 (N_samples,)
        activations_refusal: refusal 数据集的激活值数组，形状为 (N_samples,)
        activations_jailbreak: jailbreak 数据集的激活值数组，形状为 (N_samples,)
        dist_params: 分布参数字典，格式为：
            {
                'benign': {
                    'best_dist_name': 'norm',
                    'best_params': [...],
                    'data_min': ...,
                    'data_max': ...
                },
                'fail': { ... },
                'success': { ... }  # 可选
            }
        component_name: 组件名称
        output_path: 输出文件路径
        title: 图表标题（可选）
    """
    # 创建 1x3 子图布局
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # ========== 子图1：原始分布 + 拟合曲线 ==========
    ax1 = axes[0]
    
    # 合并所有激活值用于确定 x 范围
    all_activations = np.concatenate([activations_benign, activations_refusal, activations_jailbreak])
    x_min, x_max = all_activations.min(), all_activations.max()
    x_range = np.linspace(x_min, x_max, 200)
    
    # 绘制三个数据集的直方图（叠加）
    ax1.hist(
        activations_benign,
        bins=50,
        density=True,
        alpha=0.4,
        color='#0066CC',
        label='Benign Data'
    )
    ax1.hist(
        activations_refusal,
        bins=50,
        density=True,
        alpha=0.4,
        color='#FF6600',
        label='Refusal Data'
    )
    ax1.hist(
        activations_jailbreak,
        bins=50,
        density=True,
        alpha=0.4,
        color='#00AA00',
        label='Jailbreak Data'
    )
    
    # 绘制拟合 PDF 曲线
    # Benign 分布
    benign_params = dist_params.get('benign', {})
    benign_dist_name = benign_params.get('best_dist_name')
    benign_params_list = benign_params.get('best_params', [])
    benign_data_min = benign_params.get('data_min')
    benign_data_max = benign_params.get('data_max')
    
    if benign_dist_name and benign_params_list:
        benign_pdf = _compute_pdf_curve(
            x_range, benign_dist_name, benign_params_list,
            benign_data_min, benign_data_max
        )
        ax1.plot(x_range, benign_pdf, 'b-', linewidth=2.5, label='Benign Fit', alpha=0.9)
    
    # Fail (Refusal) 分布
    fail_params = dist_params.get('fail', {})
    fail_dist_name = fail_params.get('best_dist_name')
    fail_params_list = fail_params.get('best_params', [])
    fail_data_min = fail_params.get('data_min')
    fail_data_max = fail_params.get('data_max')
    
    if fail_dist_name and fail_params_list:
        fail_pdf = _compute_pdf_curve(
            x_range, fail_dist_name, fail_params_list,
            fail_data_min, fail_data_max
        )
        ax1.plot(x_range, fail_pdf, 'r-', linewidth=2.5, label='Refusal Fit', alpha=0.9)
    
    # Success 分布（如果存在）
    success_params = dist_params.get('success', {})
    success_dist_name = success_params.get('best_dist_name')
    success_params_list = success_params.get('best_params', [])
    success_data_min = success_params.get('data_min')
    success_data_max = success_params.get('data_max')
    
    if success_dist_name and success_params_list:
        success_pdf = _compute_pdf_curve(
            x_range, success_dist_name, success_params_list,
            success_data_min, success_data_max
        )
        ax1.plot(x_range, success_pdf, 'g-', linewidth=2.5, label='Success Fit', alpha=0.9)
    
    ax1.set_xlabel("Activation Value")
    ax1.set_ylabel("Density")
    ax1.set_title("Original Distribution")
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # ========== 子图2：P_known 分布 ==========
    ax2 = axes[1]
    
    # 计算三个数据集的 PDF
    # 需要为每个样本计算 PDF
    N_benign = len(activations_benign)
    N_refusal = len(activations_refusal)
    N_jailbreak = len(activations_jailbreak)
    
    # 计算 benign 数据集的 PDF
    f_B_benign = np.zeros(N_benign)
    f_R_benign = np.zeros(N_benign)
    for i, val in enumerate(activations_benign):
        f_B_benign[i] = compute_pdf_single(
            val, benign_dist_name, benign_params_list,
            benign_data_min, benign_data_max
        )
        f_R_benign[i] = compute_pdf_single(
            val, fail_dist_name, fail_params_list,
            fail_data_min, fail_data_max
        )
    
    # 计算 refusal 数据集的 PDF
    f_B_refusal = np.zeros(N_refusal)
    f_R_refusal = np.zeros(N_refusal)
    for i, val in enumerate(activations_refusal):
        f_B_refusal[i] = compute_pdf_single(
            val, benign_dist_name, benign_params_list,
            benign_data_min, benign_data_max
        )
        f_R_refusal[i] = compute_pdf_single(
            val, fail_dist_name, fail_params_list,
            fail_data_min, fail_data_max
        )
    
    # 计算 jailbreak 数据集的 PDF
    f_B_jailbreak = np.zeros(N_jailbreak)
    f_R_jailbreak = np.zeros(N_jailbreak)
    for i, val in enumerate(activations_jailbreak):
        f_B_jailbreak[i] = compute_pdf_single(
            val, benign_dist_name, benign_params_list,
            benign_data_min, benign_data_max
        )
        f_R_jailbreak[i] = compute_pdf_single(
            val, fail_dist_name, fail_params_list,
            fail_data_min, fail_data_max
        )
    
    # 计算 P_known = 0.5 * f_B + 0.5 * f_R
    P_known_benign = 0.5 * f_B_benign + 0.5 * f_R_benign
    P_known_refusal = 0.5 * f_B_refusal + 0.5 * f_R_refusal
    P_known_jailbreak = 0.5 * f_B_jailbreak + 0.5 * f_R_jailbreak
    
    # 绘制 P_known 分布
    ax2.hist(
        P_known_benign,
        bins=50,
        density=True,
        alpha=0.6,
        color='#0066CC',
        label='Benign'
    )
    ax2.hist(
        P_known_refusal,
        bins=50,
        density=True,
        alpha=0.6,
        color='#FF6600',
        label='Refusal'
    )
    ax2.hist(
        P_known_jailbreak,
        bins=50,
        density=True,
        alpha=0.6,
        color='#00AA00',
        label='Jailbreak'
    )
    
    ax2.set_xlabel("$P_{known}^{(i)}(x_i)$")
    ax2.set_ylabel("Density")
    ax2.set_title("$P_{known}$ Distribution")
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # ========== 子图3：lambda_i 分布 ==========
    ax3 = axes[2]
    
    # 计算 lambda_i = log(f_R / f_B)
    lambda_benign = np.log(f_R_benign / (f_B_benign + EPSILON) + EPSILON)
    lambda_refusal = np.log(f_R_refusal / (f_B_refusal + EPSILON) + EPSILON)
    lambda_jailbreak = np.log(f_R_jailbreak / (f_B_jailbreak + EPSILON) + EPSILON)
    
    # 绘制 lambda_i 分布
    ax3.hist(
        lambda_benign,
        bins=50,
        density=True,
        alpha=0.6,
        color='#0066CC',
        label='Benign'
    )
    ax3.hist(
        lambda_refusal,
        bins=50,
        density=True,
        alpha=0.6,
        color='#FF6600',
        label='Refusal'
    )
    ax3.hist(
        lambda_jailbreak,
        bins=50,
        density=True,
        alpha=0.6,
        color='#00AA00',
        label='Jailbreak'
    )
    
    ax3.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax3.set_xlabel("$\\lambda_i = \\log(f_R / f_B)$")
    ax3.set_ylabel("Density")
    ax3.set_title("$\\lambda_i$ Distribution")
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 设置总标题
    if title is None:
        title = f"Component Distributions: {component_name}"
    fig.suptitle(title, fontsize=13, y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()





# -*- coding: utf-8 -*-
"""
visualization.py

可视化证据轨迹（Evidence Trajectory）。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import Dict, List, Optional
from pathlib import Path

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


def plot_evidence_trajectory(
    trajectories: Dict[str, np.ndarray],
    output_path: Path,
    n_layers: int,
    title: str = "Evidence Trajectory: Layer-wise Sequential Bayesian Zero-Trust Model"
):
    """
    绘制证据轨迹图。
    
    Args:
        trajectories: 字典，键为数据集名称（如 'benign', 'jailbreak', 'refusal'），
                     值为轨迹数组，形状为 (N_samples, n_layers)
        output_path: 输出文件路径
        n_layers: 模型层数
        title: 图表标题
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 定义更鲜明的颜色和不同的线型
    styles = {
        'benign': {'color': '#0066CC', 'linestyle': '-', 'linewidth': 3, 'marker': 'o', 'markersize': 4, 'markevery': 4},
        'harmful_success': {'color': '#00AA00', 'linestyle': '--', 'linewidth': 3, 'marker': 's', 'markersize': 4, 'markevery': 4},
        'harmful_failure': {'color': '#FF6600', 'linestyle': '-.', 'linewidth': 3, 'marker': '^', 'markersize': 4, 'markevery': 4},
        'jailbreak': {'color': '#00AA00', 'linestyle': '--', 'linewidth': 3, 'marker': 's', 'markersize': 4, 'markevery': 4},
        'refusal': {'color': '#FF6600', 'linestyle': '-.', 'linewidth': 3, 'marker': '^', 'markersize': 4, 'markevery': 4},
        'harmless': {'color': '#0066CC', 'linestyle': '-', 'linewidth': 3, 'marker': 'o', 'markersize': 4, 'markevery': 4},
        'harmful': {'color': '#00AA00', 'linestyle': '--', 'linewidth': 3, 'marker': 's', 'markersize': 4, 'markevery': 4},
        'original_harmful': {'color': '#00AA00', 'linestyle': '--', 'linewidth': 3, 'marker': 's', 'markersize': 4, 'markevery': 4},
        'jailbroken_harmful': {'color': '#FF6600', 'linestyle': '-.', 'linewidth': 3, 'marker': '^', 'markersize': 4, 'markevery': 4}
    }
    
    labels = {
        'benign': 'Benign (Harmless)',
        'harmful_success': 'Harmful Success (Jailbreak)',
        'harmful_failure': 'Harmful Failure (Refusal)',
        'jailbreak': 'Jailbreak',
        'refusal': 'Refusal',
        'harmless': 'Harmless',
        'harmful': 'Harmful',
        'original_harmful': 'Original Harmful',
        'jailbroken_harmful': 'Jailbroken Harmful'
    }
    
    # 绘制每个数据集的轨迹
    layers = np.arange(n_layers)
    
    for dataset_name, trajectory in trajectories.items():
        if trajectory.shape[1] != n_layers:
            print(f"警告: {dataset_name} 的轨迹层数不匹配，跳过")
            continue
        
        # 使用中位数和分位数（更稳健，对异常值不敏感）
        median_traj = np.median(trajectory, axis=0)
        min_traj = np.min(trajectory, axis=0)
        max_traj = np.max(trajectory, axis=0)
        
        style = styles.get(dataset_name, {'color': '#000000', 'linestyle': '-', 'linewidth': 2})
        label = labels.get(dataset_name, dataset_name)
        
        # 绘制中位数线（更粗、更明显）
        ax.plot(
            layers, median_traj, 
            label=label, 
            color=style['color'],
            linestyle=style['linestyle'],
            linewidth=style['linewidth'],
            marker=style.get('marker', None),
            markersize=style.get('markersize', 0),
            markevery=style.get('markevery', None),
            alpha=0.95,
            zorder=3
        )
        
        # 绘制分位数区间（更明显的填充）
        ax.fill_between(
            layers,
            min_traj,
            max_traj,
            color=style['color'],
            alpha=0.25,
            zorder=1
        )
    
    # 设置标签和标题
    ax.set_xlabel('Layer Index', fontweight='bold', fontsize=12)
    ax.set_ylabel('Cumulative Risk Score (Evidence Trajectory)', fontweight='bold', fontsize=12)
    ax.set_title(title, fontweight='bold', pad=15, fontsize=14)
    
    # 设置网格（更明显）
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # 设置图例（更大、更明显）
    ax.legend(
        loc='best', 
        frameon=True, 
        fancybox=True, 
        shadow=True,
        fontsize=11,
        framealpha=0.95,
        edgecolor='black',
        facecolor='white'
    )
    
    # 设置坐标轴
    ax.set_xlim(-0.5, n_layers - 0.5)
    
    # 增加坐标轴粗细
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"轨迹图已保存至: {output_path}")


def plot_layer_scores_comparison(
    layer_scores: Dict[str, Dict[str, np.ndarray]],
    output_path: Path,
    n_layers: int,
    score_type: str = 'S_cons'
):
    """
    绘制层得分对比图。
    
    Args:
        layer_scores: 字典，键为数据集名称，值为包含 'S_cons', 'S_ood', 'R_l' 的字典
        output_path: 输出文件路径
        n_layers: 模型层数
        score_type: 要绘制的得分类型（'S_cons', 'S_ood', 'R_l'）
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 使用与轨迹图相同的样式
    styles = {
        'benign': {'color': '#0066CC', 'linestyle': '-', 'linewidth': 3, 'marker': 'o', 'markersize': 4, 'markevery': 4},
        'harmful_success': {'color': '#00AA00', 'linestyle': '--', 'linewidth': 3, 'marker': 's', 'markersize': 4, 'markevery': 4},
        'harmful_failure': {'color': '#FF6600', 'linestyle': '-.', 'linewidth': 3, 'marker': '^', 'markersize': 4, 'markevery': 4},
        'jailbreak': {'color': '#00AA00', 'linestyle': '--', 'linewidth': 3, 'marker': 's', 'markersize': 4, 'markevery': 4},
        'refusal': {'color': '#FF6600', 'linestyle': '-.', 'linewidth': 3, 'marker': '^', 'markersize': 4, 'markevery': 4},
        'harmless': {'color': '#0066CC', 'linestyle': '-', 'linewidth': 3, 'marker': 'o', 'markersize': 4, 'markevery': 4},
        'harmful': {'color': '#00AA00', 'linestyle': '--', 'linewidth': 3, 'marker': 's', 'markersize': 4, 'markevery': 4},
        'original_harmful': {'color': '#00AA00', 'linestyle': '--', 'linewidth': 3, 'marker': 's', 'markersize': 4, 'markevery': 4},
        'jailbroken_harmful': {'color': '#FF6600', 'linestyle': '-.', 'linewidth': 3, 'marker': '^', 'markersize': 4, 'markevery': 4}
    }
    
    labels = {
        'benign': 'Benign (Harmless)',
        'harmful_success': 'Harmful Success (Jailbreak)',
        'harmful_failure': 'Harmful Failure (Refusal)',
        'jailbreak': 'Jailbreak',
        'refusal': 'Refusal',
        'harmless': 'Harmless',
        'harmful': 'Harmful',
        'original_harmful': 'Original Harmful',
        'jailbroken_harmful': 'Jailbroken Harmful'
    }
    
    score_labels = {
        'S_cons': 'Conflict Score (S_conflict)',
        'S_ood': 'Out-of-Distribution Score (S_ood)',
        'R_l': 'Total Layer Risk (R_l)'
    }
    
    layers = np.arange(n_layers)
    
    for dataset_name, scores_dict in layer_scores.items():
        if score_type not in scores_dict:
            continue
        
        scores = scores_dict[score_type]  # (N_samples, n_layers)
        
        if scores.shape[1] != n_layers:
            continue
        
        # 使用中位数和分位数
        median_scores = np.median(scores, axis=0)
        min_scores = np.min(scores, axis=0)
        max_scores = np.max(scores, axis=0)
        
        style = styles.get(dataset_name, {'color': '#000000', 'linestyle': '-', 'linewidth': 2})
        label = labels.get(dataset_name, dataset_name)
        
        ax.plot(
            layers, median_scores, 
            label=label,
            color=style['color'],
            linestyle=style['linestyle'],
            linewidth=style['linewidth'],
            marker=style.get('marker', None),
            markersize=style.get('markersize', 0),
            markevery=style.get('markevery', None),
            alpha=0.95,
            zorder=3
        )
        ax.fill_between(
            layers,
            min_scores,
            max_scores,
            color=style['color'],
            alpha=0.25,
            zorder=1
        )
    
    ax.set_xlabel('Layer Index', fontweight='bold', fontsize=12)
    ax.set_ylabel(score_labels.get(score_type, score_type), fontweight='bold', fontsize=12)
    ax.set_title(f'Layer-wise {score_labels.get(score_type, score_type)} Comparison', fontweight='bold', pad=15, fontsize=14)
    
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(
        loc='best', 
        frameon=True, 
        fancybox=True, 
        shadow=True,
        fontsize=11,
        framealpha=0.95,
        edgecolor='black',
        facecolor='white'
    )
    ax.set_xlim(-0.5, n_layers - 0.5)
    
    # 增加坐标轴粗细
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"层得分对比图已保存至: {output_path}")


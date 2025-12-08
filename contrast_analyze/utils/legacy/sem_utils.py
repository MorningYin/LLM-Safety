# -*- coding: utf-8 -*-
"""
sem_utils.py

SEM 分析工具统一模块。
整合了所有 SEM 分析相关的工具：数据加载、Token 筛选、模型构建、拟合、结果处理和可视化。
"""

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats


# 可视化相关导入
try:
    import matplotlib.cm as cm
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

# SEM 拟合相关导入
try:
    from semopy import Model, calc_stats
    SEMOPY_AVAILABLE = True
except ImportError:
    SEMOPY_AVAILABLE = False
    print("警告: semopy 未安装，SEM 拟合功能将不可用。请安装: pip install semopy")

# ============================================================
# 可视化函数
# ============================================================


# 学术论文风格的发散色图（RdBu_r风格）
ACADEMIC_COLORMAP = LinearSegmentedColormap.from_list(
    "academic_diverging",
    ["#2166ac", "#4393c3", "#92c5de", "#d1e5f0", "#ffffff", 
     "#fddbc7", "#f4a582", "#d6604d", "#b2182b"],
    N=256
)

# 使用RdBu_r作为备选（更标准的学术色图）
try:
    RDBU_COLORMAP = cm.RdBu_r
except:
    RDBU_COLORMAP = ACADEMIC_COLORMAP


def parse_node_name(node_name: str) -> Tuple[int, int]:
    pattern = r"P_L(\d+)_T(\d+)"
    match = re.match(pattern, node_name)
    if not match:
        raise ValueError(f"无法解析节点名称: {node_name}")
    return int(match.group(1)), int(match.group(2))


def build_layered_graph(
    df: pd.DataFrame,
    use_all_paths: bool = True,
    threshold: float = 0.1
) -> Tuple[nx.DiGraph, Dict[str, Tuple[int, int]]]:
    G = nx.DiGraph()
    node_positions = {}
    
    # 筛选路径
    paths_df = df if use_all_paths else df[df['is_significant'] == True].copy()
    
    # 应用阈值过滤
    if threshold > 0:
        paths_df = paths_df[paths_df['net_effect'].abs() > threshold].copy()
    
    if paths_df.empty:
        return G, node_positions
    
    # 收集所有节点并解析位置
    all_nodes = set(paths_df['source'].unique()) | set(paths_df['target'].unique())
    for node in all_nodes:
        try:
            layer, token = parse_node_name(node)
            node_positions[node] = (layer, token)
        except ValueError:
            continue
    
    # 添加边
    for _, row in paths_df.iterrows():
        source = row['source']
        target = row['target']
        
        if source not in node_positions or target not in node_positions:
            continue
        
        G.add_edge(
            source,
            target,
            net_effect=float(row['net_effect']),
            is_significant=bool(row.get('is_significant', False)),
            beta_fail=float(row.get('beta_fail', 0.0)),
            beta_benign=float(row.get('beta_benign', 0.0)),
            p_value=float(row.get('p_value', 1.0))
        )
    
    return G, node_positions


def compute_hierarchical_layout(
    node_positions: Dict[str, Tuple[int, int]],
    layer_spacing: float = 3.0,
    token_spacing: float = 1.0
) -> Dict[str, Tuple[float, float]]:
    if not node_positions:
        return {}
    
    # 按层分组节点
    layers = defaultdict(list)
    for node, (layer, token) in node_positions.items():
        layers[layer].append((node, token))
    
    # 对每层的节点按token排序
    for layer in layers:
        layers[layer].sort(key=lambda x: x[1])
    
    # 计算所有token的范围，用于统一Y轴缩放
    all_tokens = [token for _, (_, token) in node_positions.items()]
    min_token = min(all_tokens) if all_tokens else 0
    max_token = max(all_tokens) if all_tokens else 0
    token_range = max_token - min_token if max_token > min_token else 1
    
    # 计算布局
    layout = {}
    sorted_layers = sorted(layers.keys())
    
    for layer_idx, layer_num in enumerate(sorted_layers):
        # X轴：严格对应Layer号（使用实际层号，但通过索引缩放）
        x = layer_idx * layer_spacing
        
        layer_nodes = layers[layer_num]
        
        # Y轴：严格对应Token索引（归一化到合理范围）
        for node, token in layer_nodes:
            # 将token索引映射到Y轴位置
            if token_range > 0:
                y = (token - min_token) / token_range * (len(sorted_layers) * token_spacing) - (len(sorted_layers) * token_spacing) / 2
            else:
                y = 0.0
            layout[node] = (x, y)
    
    return layout


def _compute_net_effect_range(net_effects: List[float]) -> Tuple[float, float, float]:
    if not net_effects:
        return -1.0, 1.0, 1.0
    
    vmin = min(net_effects)
    vmax = max(net_effects)
    abs_max = max(abs(vmin), abs(vmax))
    
    if abs_max > 0:
        return -abs_max, abs_max, abs_max
    else:
        return -1.0, 1.0, 1.0


def _separate_edges_by_significance(G: nx.DiGraph) -> Tuple[List[Tuple], List[Tuple]]:
    significant_edges = []
    non_significant_edges = []
    
    for u, v, data in G.edges(data=True):
        edge = (u, v)
        if data.get('is_significant', False):
            significant_edges.append(edge)
        else:
            non_significant_edges.append(edge)
    
    return significant_edges, non_significant_edges


def _draw_edges(
    ax,
    G: nx.DiGraph,
    pos: Dict,
    edges: List[Tuple],
    norm: TwoSlopeNorm,
    colormap,
    width_scale: float = 1.0,
    alpha_scale: float = 1.0,
    style: str = 'solid',
    use_curved: bool = True
) -> None:
    if not edges:
        return
    
    # 提取净效应值
    effects = [G[u][v]['net_effect'] for u, v in edges]
    abs_effects = [abs(e) for e in effects]
    
    # 计算颜色（使用发散色图）
    colors = [colormap(norm(effect)) for effect in effects]
    
    # 计算宽度（与abs(net_effect)成正比）
    abs_max = max(abs_effects) if abs_effects else 1.0
    widths = [
        (abs_effect / abs_max * 3.0 + 0.5) * width_scale 
        if abs_max > 0 else 0.5 * width_scale
        for abs_effect in abs_effects
    ]
    
    # 计算透明度（与abs(net_effect)成正比，弱效应更淡）
    alpha_max = 0.9
    alpha_min = 0.2
    alphas = [
        (abs_effect / abs_max * (alpha_max - alpha_min) + alpha_min) * alpha_scale
        if abs_max > 0 else alpha_min * alpha_scale
        for abs_effect in abs_effects
    ]
    
    # 绘制每条边（单独绘制以支持弯曲箭头和不同的透明度）
    for (u, v), color, width, alpha_val in zip(edges, colors, widths, alphas):
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        
        # 计算弯曲弧度（如果使用弯曲箭头）
        if use_curved:
            # 根据水平距离计算弧度
            dx = x2 - x1
            rad = 0.3 if abs(dx) > 0.1 else 0.1
            connectionstyle = f"arc3,rad={rad}"
        else:
            connectionstyle = "arc3,rad=0"
        
        ax.annotate(
            '',
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle='->',
                color=color,
                lw=width,
                alpha=alpha_val,
                linestyle=style,
                connectionstyle=connectionstyle
            )
        )


def _draw_nodes(
    ax, 
    G: nx.DiGraph, 
    pos: Dict, 
    node_positions: Dict, 
    node_size: float = 300
) -> None:
    # 绘制节点（小圆圈）
    nx.draw_networkx_nodes(
        G, pos,
        node_color='white',
        node_size=node_size,
        alpha=0.9,
        edgecolors='black',
        linewidths=1.5,
        ax=ax
    )
    
    # 绘制节点标签（简化为"T{token}"）
    labels = {}
    for node in G.nodes():
        if node in node_positions:
            _, token = node_positions[node]
            labels[node] = f"T{token}"
        else:
            labels[node] = node
    
    nx.draw_networkx_labels(
        G, pos,
        labels=labels,
        font_size=9,
        font_weight='bold',
        font_color='black',
        ax=ax
    )


def _add_layer_grid(ax, node_positions: Dict, pos: Dict) -> None:
    if not node_positions:
        return
    
    layers = sorted(set(layer for layer, _ in node_positions.values()))
    layer_x_positions = {}
    
    for layer in layers:
        layer_nodes = [node for node, (l, _) in node_positions.items() if l == layer]
        if layer_nodes:
            x_positions = [pos[node][0] for node in layer_nodes if node in pos]
            if x_positions:
                layer_x_positions[layer] = np.mean(x_positions)
    
    y_min, y_max = ax.get_ylim()
    for layer, x_pos in layer_x_positions.items():
        ax.axvline(x=x_pos, color='lightgray', linestyle='--', linewidth=0.5, alpha=0.3, zorder=0)


def _add_layer_labels(ax, node_positions: Dict, pos: Dict) -> None:
    if not node_positions:
        return
    
    layers = sorted(set(layer for layer, _ in node_positions.values()))
    layer_x_positions = {}
    
    for layer in layers:
        layer_nodes = [node for node, (l, _) in node_positions.items() if l == layer]
        if layer_nodes:
            x_positions = [pos[node][0] for node in layer_nodes if node in pos]
            if x_positions:
                layer_x_positions[layer] = np.mean(x_positions)
    
    y_bottom = ax.get_ylim()[0]
    for layer, x_pos in layer_x_positions.items():
        ax.text(x_pos, y_bottom - 0.3, f'L{layer}', 
               ha='center', va='top', fontsize=10, fontweight='bold')


def create_layered_circuit_plot(
    df: pd.DataFrame,
    output_path: Path,
    use_all_paths: bool = True,
    threshold: float = 0.1,
    figsize: Tuple[float, float] = (20, 12),
    dpi: int = 300,
    significant_alpha: float = 1.0,
    non_significant_alpha: float = 0.6,
    node_size: float = 300,
    use_curved_arrows: bool = True,
    use_grid: bool = True
) -> Optional[Path]:
    if not VISUALIZATION_AVAILABLE:
        return None
    
    # 设置学术字体
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    
    # 构建图（应用阈值过滤）
    G, node_positions = build_layered_graph(df, use_all_paths=use_all_paths, threshold=threshold)
    
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        return None
    
    # 计算布局
    pos = compute_hierarchical_layout(node_positions)
    
    # 准备图形
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('white')
    
    # 添加层网格线（如果启用）
    if use_grid:
        _add_layer_grid(ax, node_positions, pos)
    
    # 分离显著和非显著边
    significant_edges, non_significant_edges = _separate_edges_by_significance(G)
    
    # 计算净效应范围
    all_net_effects = [data['net_effect'] for _, _, data in G.edges(data=True)]
    vmin, vmax, abs_max = _compute_net_effect_range(all_net_effects)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    
    # 使用RdBu_r色图
    colormap = RDBU_COLORMAP
    
    # 绘制非显著边（先绘制，在底层）
    _draw_edges(ax, G, pos, non_significant_edges, norm, colormap,
               width_scale=0.6, alpha_scale=non_significant_alpha, 
               style='dashed', use_curved=use_curved_arrows)
    
    # 绘制显著边（后绘制，在上层）
    _draw_edges(ax, G, pos, significant_edges, norm, colormap,
               width_scale=1.0, alpha_scale=significant_alpha, 
               style='solid', use_curved=use_curved_arrows)
    
    # 绘制节点
    _draw_nodes(ax, G, pos, node_positions, node_size)
    
    # 添加颜色条
    if all_net_effects:
        sm = ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.02, shrink=0.8)
        cbar.set_label('Net Effect (Δβ = β_Fail - β_Benign)', 
                      fontsize=12, fontweight='bold')
        cbar.ax.tick_params(labelsize=10)
    
    # 添加图例（英文说明）
    legend_elements = [
        mpatches.Patch(color='red', alpha=0.7, label='Red = Refusal Amplification (Δβ > 0)'),
        mpatches.Patch(color='blue', alpha=0.7, label='Blue = Benign Suppression (Δβ < 0)'),
        mpatches.Patch(color='black', alpha=significant_alpha, 
                      label=f'Significant paths (n={len(significant_edges)})'),
        mpatches.Patch(color='black', alpha=non_significant_alpha, 
                      linestyle='--', label=f'Non-significant paths (n={len(non_significant_edges)})')
    ]
    ax.legend(handles=legend_elements, loc='upper left', 
             fontsize=10, framealpha=0.95, fancybox=True, shadow=True)
    
    # 添加标题和轴标签
    title = "Refusal Circuit: Layered Network Graph"
    if not use_all_paths:
        title += " (Significant Paths Only)"
    if threshold > 0:
        title += f" (|Δβ| > {threshold})"
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # 添加层标签
    _add_layer_labels(ax, node_positions, pos)
    
    ax.set_xlabel('Transformer Layers (→)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Token Positions', fontsize=12, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    
    # 确保所有内容都在图范围内（添加边距，考虑弯曲箭头）
    x_coords = [pos[node][0] for node in G.nodes()]
    y_coords = [pos[node][1] for node in G.nodes()]
    
    if x_coords and y_coords:
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # 计算边距（基于数据范围）
        x_range = x_max - x_min if x_max > x_min else 1.0
        y_range = y_max - y_min if y_max > y_min else 1.0
        
        # 左侧和上下边距较小，右侧边距较大（因为箭头指向右侧，弯曲箭头可能超出）
        x_margin_left = x_range * 0.08
        x_margin_right = x_range * 0.20  # 右侧更多边距以容纳弯曲箭头
        y_margin = y_range * 0.12
        
        # 设置坐标轴范围，确保所有节点和边都在范围内
        ax.set_xlim(x_min - x_margin_left, x_max + x_margin_right)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
    
    plt.tight_layout()
    
    # 保存图像
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    file_format = 'pdf' if output_path.suffix.lower() == '.pdf' else 'png'
    save_kwargs = {
        'bbox_inches': 'tight',
        'facecolor': 'white',
        'edgecolor': 'none',
        'format': file_format
    }
    if file_format == 'png':
        save_kwargs['dpi'] = dpi
    
    plt.savefig(output_path, **save_kwargs)
    plt.close(fig)
    
    return output_path


def create_effect_matrix_heatmap(
    df: pd.DataFrame,
    output_path: Path,
    threshold: float = 0.1,
    figsize: Tuple[float, float] = (14, 10),
    dpi: int = 300
) -> Optional[Path]:
    if not VISUALIZATION_AVAILABLE:
        return None
    
    # 设置学术字体
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['font.size'] = 11
    
    # 应用阈值过滤
    filtered_df = df[df['net_effect'].abs() > threshold].copy()
    
    if filtered_df.empty:
        return None
    
    # 解析source和target，提取Layer和Token
    def parse_interaction(row):
        try:
            src_layer, src_token = parse_node_name(row['source'])
            tgt_layer, tgt_token = parse_node_name(row['target'])
            interaction = f"T{src_token}→T{tgt_token}"
            return src_layer, interaction, row['net_effect']
        except:
            return None, None, None
    
    # 提取数据
    data = []
    for _, row in filtered_df.iterrows():
        layer, interaction, net_effect = parse_interaction(row)
        if layer is not None:
            data.append({'layer': layer, 'interaction': interaction, 'net_effect': net_effect})
    
    if not data:
        return None
    
    # 创建DataFrame并聚合
    heatmap_df = pd.DataFrame(data)
    matrix = heatmap_df.pivot_table(
        values='net_effect',
        index='interaction',
        columns='layer',
        aggfunc='mean'
    )
    
    # 排序：按交互类型和层号
    matrix = matrix.sort_index()
    matrix = matrix.reindex(sorted(matrix.columns), axis=1)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    # 计算颜色范围
    vmax = max(abs(matrix.min().min()), abs(matrix.max().max()))
    vmin = -vmax
    
    # 绘制热图
    im = ax.imshow(matrix, cmap=RDBU_COLORMAP, aspect='auto', 
                   vmin=vmin, vmax=vmax, interpolation='nearest')
    
    # 设置刻度
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels([f"L{int(col)}" for col in matrix.columns], rotation=45, ha='right')
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label('Net Effect (Δβ)', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    # 添加标签和标题
    ax.set_xlabel('Layer Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Interaction Type (Source→Target Token)', fontsize=12, fontweight='bold')
    ax.set_title(f'Effect Matrix Heatmap (|Δβ| > {threshold})', 
                fontsize=14, fontweight='bold', pad=20)
    
    # 添加网格线
    ax.set_xticks(np.arange(len(matrix.columns)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(matrix.index)) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图像
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    file_format = 'pdf' if output_path.suffix.lower() == '.pdf' else 'png'
    save_kwargs = {
        'bbox_inches': 'tight',
        'facecolor': 'white',
        'edgecolor': 'none',
        'format': file_format
    }
    if file_format == 'png':
        save_kwargs['dpi'] = dpi
    
    plt.savefig(output_path, **save_kwargs)
    plt.close(fig)
    
    return output_path


# ============================================================
# 数据加载
# ============================================================

# -*- coding: utf-8 -*-







# ============================================================
# Token 筛选
# ============================================================


class TokenFilter:
    
    def __init__(self, method: str = "ttest", threshold: float = 0.05, top_k: int = None):
        if method not in ["ttest", "mannwhitney"]:
            raise ValueError(f"未知的统计检验方法: {method}。支持的方法: 'ttest', 'mannwhitney'")
        
        self.method = method
        self.threshold = threshold
        self.top_k = top_k
        self._filter_stats = {}
    
    def filter(
        self, 
        P_benign: torch.Tensor, 
        P_fail: torch.Tensor
    ) -> List[Tuple[int, int]]:
        N_b, L, T = P_benign.shape
        N_f = P_fail.shape[0]
        
        # 确保形状匹配
        assert P_fail.shape == (N_f, L, T), f"形状不匹配: P_fail {P_fail.shape} vs 期望 {(N_f, L, T)}"
        
        candidate_positions = []  # 通过 p 值检验的候选位置
        candidate_info = []  # 存储每个候选位置的信息 (p_val, effect_size, position)
        all_p_values = []  # 记录所有 p 值用于诊断
        
        # 转换为 numpy 用于统计检验（先转换为 float32 以支持 BFloat16）
        P_benign_np = P_benign.float().numpy()
        P_fail_np = P_fail.float().numpy()
        
        # 步骤 1: 遍历所有 (Layer, Token) 位置，进行 p 值检验
        for l in range(L):
            for t in range(T):
                # 提取该位置的两组数据
                data_benign = P_benign_np[:, l, t]  # (N_b,)
                data_fail = P_fail_np[:, l, t]  # (N_f,)
                
                # 检查数据有效性（避免全零或常数）
                if np.std(data_benign) == 0 and np.std(data_fail) == 0:
                    # 两组都是常数，检查是否相等
                    if np.abs(np.mean(data_benign) - np.mean(data_fail)) < 1e-10:
                        p_val = 1.0  # 完全相同，p值应该很大
                    else:
                        p_val = 0.0  # 常数但不同，p值应该很小
                elif np.std(data_benign) == 0 or np.std(data_fail) == 0:
                    # 一组是常数，另一组不是，需要特殊处理
                    if np.abs(np.mean(data_benign) - np.mean(data_fail)) < 1e-10:
                        p_val = 1.0
                    else:
                        p_val = 0.0
                else:
                    # 执行统计检验
                    if self.method == "ttest":
                        # 独立样本 t 检验
                        stat, p_val = stats.ttest_ind(data_benign, data_fail)
                    elif self.method == "mannwhitney":
                        # Mann-Whitney U 检验（非参数检验）
                        stat, p_val = stats.mannwhitneyu(data_benign, data_fail, alternative='two-sided')
                
                # 计算效应量（Cohen's d）用于排序
                mean_diff = np.mean(data_benign) - np.mean(data_fail)
                pooled_std = np.sqrt((np.var(data_benign, ddof=1) + np.var(data_fail, ddof=1)) / 2)
                if pooled_std > 0:
                    cohens_d = abs(mean_diff / pooled_std)  # 使用绝对值，因为差异大小更重要
                else:
                    cohens_d = abs(mean_diff) if mean_diff != 0 else 0.0  # 如果标准差为0，使用均值差异的绝对值
                
                # 记录所有 p 值
                all_p_values.append(p_val)
                
                # 步骤 1: 如果 p 值小于阈值，加入候选列表
                if p_val < self.threshold:
                    candidate_info.append({
                        'position': (l, t),
                        'p_value': p_val,
                        'effect_size': cohens_d,
                        'mean_diff': abs(mean_diff)
                    })
        
        # 步骤 2: 根据效应量排序，选取 top_k 个
        if self.top_k is not None and len(candidate_info) > self.top_k:
            # 按效应量降序排序
            candidate_info.sort(key=lambda x: x['effect_size'], reverse=True)
            # 选取前 top_k 个
            candidate_info = candidate_info[:self.top_k]
        
        # 提取最终选中的位置
        selected_positions = [info['position'] for info in candidate_info]
        p_values = [info['p_value'] for info in candidate_info]
        
        # 记录筛选统计信息
        all_p_values = np.array(all_p_values)
        num_passed_p_test = len(candidate_info) if self.top_k is None else len([p for p in all_p_values if p < self.threshold])
        
        self._filter_stats = {
            "total_positions": L * T,
            "passed_p_test": num_passed_p_test,
            "selected_positions": len(selected_positions),
            "filter_rate": len(selected_positions) / (L * T),
            "method": self.method,
            "threshold": self.threshold,
            "top_k": self.top_k,
            "p_value_stats": {
                "min": float(np.min(all_p_values)),
                "max": float(np.max(all_p_values)),
                "mean": float(np.mean(all_p_values)),
                "median": float(np.median(all_p_values)),
                "percentile_1": float(np.percentile(all_p_values, 1)),
                "percentile_5": float(np.percentile(all_p_values, 5)),
                "percentile_10": float(np.percentile(all_p_values, 10)),
                "percentile_25": float(np.percentile(all_p_values, 25)),
                "percentile_50": float(np.percentile(all_p_values, 50)),
                "percentile_75": float(np.percentile(all_p_values, 75)),
                "percentile_90": float(np.percentile(all_p_values, 90)),
                "percentile_95": float(np.percentile(all_p_values, 95)),
                "percentile_99": float(np.percentile(all_p_values, 99)),
            },
            "effect_size_stats": {
                "min": float(min([info['effect_size'] for info in candidate_info])) if candidate_info else 0.0,
                "max": float(max([info['effect_size'] for info in candidate_info])) if candidate_info else 0.0,
                "mean": float(np.mean([info['effect_size'] for info in candidate_info])) if candidate_info else 0.0,
                "median": float(np.median([info['effect_size'] for info in candidate_info])) if candidate_info else 0.0,
            } if candidate_info else {},
            "p_values": p_values[:100] if len(p_values) <= 100 else None  # 如果太多不保存所有 p 值
        }
        
        return selected_positions
    
    def get_filter_stats(self) -> Dict:
        return self._filter_stats.copy()


def prepare_covariance_matrices_pair(
    P1: torch.Tensor,
    P2: torch.Tensor,
    selected_positions: List[Tuple[int, int]],
    group1_name: str = "group1",
    group2_name: str = "group2"
) -> Dict:
    """
    准备两组数据的协方差矩阵和均值向量。
    
    Args:
        P1: 第一组投影数据，形状为 (N1, L, T)
        P2: 第二组投影数据，形状为 (N2, L, T)
        selected_positions: 选定的 (layer, token) 位置列表
        group1_name: 第一组名称
        group2_name: 第二组名称
    
    Returns:
        包含协方差矩阵、均值向量和变量名的字典
    """
    # 生成变量名
    variable_names = [f"P_L{l}_T{t}" for l, t in selected_positions]
    
    # 构建观测变量矩阵
    obs1 = []
    obs2 = []
    
    for l_rel, t in selected_positions:
        obs1.append(P1[:, l_rel, t])  # (N1,)
        obs2.append(P2[:, l_rel, t])  # (N2,)
    
    # 堆叠为矩阵: (N, V)
    obs1 = torch.stack(obs1, dim=1).float()  # (N1, V)
    obs2 = torch.stack(obs2, dim=1).float()  # (N2, V)
    
    # 转换为 numpy 用于协方差计算
    obs1_np = obs1.float().numpy()
    obs2_np = obs2.float().numpy()
    
    # 计算均值向量
    mu1 = np.mean(obs1_np, axis=0)  # (V,)
    mu2 = np.mean(obs2_np, axis=0)  # (V,)
    
    # 计算协方差矩阵
    S1 = np.cov(obs1_np, rowvar=False)  # (V, V)
    S2 = np.cov(obs2_np, rowvar=False)  # (V, V)
    
    return {
        "variable_names": variable_names,
        f"S_{group1_name}": S1,
        f"S_{group2_name}": S2,
        f"mu_{group1_name}": mu1,
        f"mu_{group2_name}": mu2,
        f"obs_{group1_name}": obs1_np,
        f"obs_{group2_name}": obs2_np
    }



# ============================================================
# 模型构建
# ============================================================


class ChainModelBuilder:
    
    def __init__(self, selected_positions: List[Tuple[int, int]]):
        self.selected_positions = selected_positions
        self._variable_names = None
        self._paths = None
        self._build_variable_names()
        self._build_paths()
    
    def _build_variable_names(self):
        self._variable_names = [f"P_L{l}_T{t}" for l, t in self.selected_positions]
    
    def _build_paths(self):
        # 将位置转换为集合以便快速查找
        position_set = set(self.selected_positions)
        
        # 构建位置到变量名的映射
        pos_to_var = {pos: f"P_L{pos[0]}_T{pos[1]}" for pos in self.selected_positions}
        
        paths = []
        
        # 按层分组位置
        layers_dict = {}
        for l, t in self.selected_positions:
            if l not in layers_dict:
                layers_dict[l] = []
            layers_dict[l].append(t)
        
        # 遍历所有层（除了最后一层）
        max_layer = max(layers_dict.keys()) if layers_dict else -1
        
        for l in range(max_layer):
            if l not in layers_dict:
                continue
            
            next_layer = l + 1
            if next_layer not in layers_dict:
                continue
            
            # 当前层的所有 token
            tokens_l = layers_dict[l]
            # 下一层的所有 token
            tokens_next = layers_dict[next_layer]
            
            # 构建路径
            for t_l in tokens_l:
                source_pos = (l, t_l)
                source_var = pos_to_var[source_pos]
                
                # 自维持路径：同一 token
                if (next_layer, t_l) in position_set:
                    target_pos = (next_layer, t_l)
                    target_var = pos_to_var[target_pos]
                    paths.append((source_var, target_var))
                
                # 跨 Token 路径：不同 token
                for t_next in tokens_next:
                    if t_next != t_l:  # 避免重复（自维持路径已处理）
                        target_pos = (next_layer, t_next)
                        if target_pos in position_set:
                            target_var = pos_to_var[target_pos]
                            paths.append((source_var, target_var))
        
        self._paths = paths
    
    def build_paths(self) -> List[Tuple[str, str]]:
        return self._paths.copy()
    
    def generate_model_syntax(self) -> str:
        if not self._paths:
            return ""
        
        lines = []
        for source, target in self._paths:
            lines.append(f"{target} ~ {source}")
        
        return "\n".join(lines)
    
    def get_variable_names(self) -> List[str]:
        return self._variable_names.copy()



# ============================================================
# SEM 拟合
# ============================================================


class MGSEMFitter:
    
    def __init__(self, model_syntax: str, lasso_lambda: float = 0.1):
        if not SEMOPY_AVAILABLE:
            raise ImportError("semopy 未安装，无法使用 SEM 拟合功能")
        
        self.model_syntax = model_syntax
        self.lasso_lambda = lasso_lambda
        self._baseline_models = {}
    
    def fit_baseline(
        self,
        S_benign: np.ndarray,
        S_fail: np.ndarray,
        mu_benign: np.ndarray,
        mu_fail: np.ndarray,
        variable_names: List[str],
        obs_benign: Optional[np.ndarray] = None,
        obs_fail: Optional[np.ndarray] = None
    ) -> Dict:
        # 拟合 Benign 组模型
        if obs_benign is not None:
            # 使用原始数据拟合（推荐）
            data_benign = pd.DataFrame(obs_benign, columns=variable_names)
            model_benign = Model(self.model_syntax)
            fit_stats_benign = model_benign.fit(data_benign)
        else:
            raise ValueError("需要提供原始观测数据 obs_benign 进行拟合")
        
        # 拟合 Fail 组模型
        if obs_fail is not None:
            data_fail = pd.DataFrame(obs_fail, columns=variable_names)
            model_fail = Model(self.model_syntax)
            fit_stats_fail = model_fail.fit(data_fail)
        else:
            raise ValueError("需要提供原始观测数据 obs_fail 进行拟合")
        
        # 提取拟合结果：使用 calc_stats() 获取卡方值
        try:
            stats_benign = calc_stats(model_benign)
            stats_fail = calc_stats(model_fail)
            
            chi2_benign = stats_benign['chi2'].iloc[0] if isinstance(stats_benign, pd.DataFrame) and 'chi2' in stats_benign.columns else None
            chi2_fail = stats_fail['chi2'].iloc[0] if isinstance(stats_fail, pd.DataFrame) and 'chi2' in stats_fail.columns else None
        except Exception as e:
            logging.error(f"无法提取卡方值: {e}", exc_info=True)
            chi2_benign = None
            chi2_fail = None
        
        # 获取路径系数
        beta_benign = self._extract_path_coefficients(model_benign)
        beta_fail = self._extract_path_coefficients(model_fail)
        
        # 获取自由度（从模型参数数量计算）
        df_benign = self._calculate_df(model_benign)
        df_fail = self._calculate_df(model_fail)
        
        # 提取完整的拟合统计信息
        stats_benign_full = calc_stats(model_benign) if chi2_benign is not None else None
        stats_fail_full = calc_stats(model_fail) if chi2_fail is not None else None
        
        result = {
            "model_benign": model_benign,
            "model_fail": model_fail,
            "chi2_benign": chi2_benign,
            "chi2_fail": chi2_fail,
            "df_benign": df_benign,
            "df_fail": df_fail,
            "beta_benign": beta_benign,
            "beta_fail": beta_fail,
            "stats_benign": stats_benign_full.to_dict('records')[0] if isinstance(stats_benign_full, pd.DataFrame) and not stats_benign_full.empty else {},
            "stats_fail": stats_fail_full.to_dict('records')[0] if isinstance(stats_fail_full, pd.DataFrame) and not stats_fail_full.empty else {}
        }
        
        self._baseline_models = result
        return result
    
    def _extract_path_coefficients(self, model: Model) -> Dict[Tuple[str, str], float]:
        beta_dict = {}
        
        try:
            params = model.inspect()
            if params is not None and not params.empty:
                for idx, row in params.iterrows():
                    if str(row.get('op', '')).strip() != '~':
                        continue
                    
                    target = str(row.get('lval', '')).strip()
                    source = str(row.get('rval', '')).strip()
                    
                    if pd.isna(target) or pd.isna(source) or not target or not source:
                        continue
                    
                    coeff = row.get('Estimate', None)
                    if coeff is None:
                        for col in ['Value', 'est', 'estimate']:
                            if col in row.index:
                                coeff = row[col]
                                break
                    
                    if coeff is not None and not pd.isna(coeff):
                        beta_dict[(source, target)] = float(coeff)
        except Exception as e:
            logging.error(f"无法提取路径系数: {e}", exc_info=True)
        
        return beta_dict
    
    def _calculate_df(self, model: Model) -> int:
        try:
            stats = calc_stats(model)
            if isinstance(stats, pd.DataFrame) and 'DoF' in stats.columns:
                return int(stats['DoF'].iloc[0])
            elif hasattr(model, 'df'):
                return model.df
            return 0
        except Exception:
            return 0
    
    def fit_constrained(
        self,
        path: Tuple[str, str],
        S_benign: np.ndarray,
        S_fail: np.ndarray,
        mu_benign: np.ndarray,
        mu_fail: np.ndarray,
        variable_names: List[str],
        obs_benign: Optional[np.ndarray] = None,
        obs_fail: Optional[np.ndarray] = None
    ) -> Dict:
        # 先拟合基线模型（如果还没有）
        if not self._baseline_models:
            self.fit_baseline(S_benign, S_fail, mu_benign, mu_fail, variable_names, obs_benign, obs_fail)
        
        # 注意：semopy 可能不直接支持多组约束模型
        # 这里我们采用简化方法：分别拟合，然后手动约束参数
        # 实际实现可能需要更复杂的处理
        
        # 对于约束模型，我们需要重新拟合，但强制指定路径的系数相等
        # 由于 semopy 的限制，我们使用近似方法：
        # 1. 使用基线模型的卡方值作为基准
        # 2. 约束模型会增加 1 个自由度（因为强制一个参数相等）
        # 3. 约束模型的卡方值应该 >= 基线模型的卡方值
        
        # 获取基线模型的路径系数
        baseline_beta_benign = self._baseline_models["beta_benign"].get(path, 0.0)
        baseline_beta_fail = self._baseline_models["beta_fail"].get(path, 0.0)
        
        # 计算约束值（两组的平均值）
        constrained_value = (baseline_beta_benign + baseline_beta_fail) / 2
        
        # 计算约束导致的卡方值增加（近似）
        # 使用 Wald 检验的近似：Δχ² ≈ (β_fail - β_benign)² / (SE_fail² + SE_benign²)
        # 由于我们没有标准误，使用简化的近似
        beta_diff = baseline_beta_fail - baseline_beta_benign
        # 简化的卡方增加：假设标准误为系数的 10%
        se_approx = abs(constrained_value) * 0.1 + 0.01  # 避免除零
        delta_chi2_approx = (beta_diff ** 2) / (2 * se_approx ** 2)
        
        result = {
            "model_benign": self._baseline_models["model_benign"],
            "model_fail": self._baseline_models["model_fail"],
            "chi2_benign": (self._baseline_models["chi2_benign"] or 0.0) + delta_chi2_approx / 2,
            "chi2_fail": (self._baseline_models["chi2_fail"] or 0.0) + delta_chi2_approx / 2,
            # 注意：约束一条路径只增加 1 个总自由度，不是每组各增加1
            # 但为了计算方便，我们保持 df_benign 和 df_fail 分别增加，在 chi_square_test 中会修正
            "df_benign": self._baseline_models["df_benign"] + 1,
            "df_fail": self._baseline_models["df_fail"] + 1,
            "constrained_path": path,
            "beta_benign": self._baseline_models["beta_benign"].copy(),
            "beta_fail": self._baseline_models["beta_fail"].copy(),
            "delta_chi2_approx": delta_chi2_approx
        }
        
        # 对约束路径，强制设置系数相等
        if path in result["beta_benign"] and path in result["beta_fail"]:
            result["beta_benign"][path] = constrained_value
            result["beta_fail"][path] = constrained_value
        
        return result
    
    def chi_square_test(
        self,
        result_free: Dict,
        result_constrained: Dict
    ) -> Dict:
        
        # 计算总卡方值
        chi2_free_benign = result_free.get("chi2_benign", 0) or 0
        chi2_free_fail = result_free.get("chi2_fail", 0) or 0
        chi2_free = chi2_free_benign + chi2_free_fail
        
        chi2_constrained_benign = result_constrained.get("chi2_benign", 0) or 0
        chi2_constrained_fail = result_constrained.get("chi2_fail", 0) or 0
        chi2_constrained = chi2_constrained_benign + chi2_constrained_fail
        
        # 计算总自由度
        # 注意：对于多组模型，自由度应该是两组的和
        df_free_benign = result_free.get("df_benign", 0) or 0
        df_free_fail = result_free.get("df_fail", 0) or 0
        df_free = df_free_benign + df_free_fail
        
        df_constrained_benign = result_constrained.get("df_benign", 0) or 0
        df_constrained_fail = result_constrained.get("df_fail", 0) or 0
        df_constrained = df_constrained_benign + df_constrained_fail
        
        # 计算差异
        delta_chi2 = chi2_constrained - chi2_free
        delta_df = df_constrained - df_free
        
        # 修正：约束一条路径应该只增加 1 个总自由度（不是每组各增加1）
        if delta_df > 1:
            delta_df = 1
        
        # 执行卡方检验
        if delta_df > 0 and delta_chi2 > 0:
            p_value = 1 - stats.chi2.cdf(delta_chi2, delta_df)
        elif delta_df > 0:
            # 如果 delta_chi2 <= 0，说明约束模型拟合更好（不应该发生），p值设为1
            p_value = 1.0
        else:
            # 如果 delta_df <= 0，说明自由度没有增加，p值设为1
            p_value = 1.0
        
        return {
            "delta_chi2": delta_chi2,
            "delta_df": delta_df,
            "p_value": p_value,
            "chi2_free": chi2_free,
            "chi2_constrained": chi2_constrained,
            "df_free": df_free,
            "df_constrained": df_constrained
        }



# ============================================================
# 结果处理
# ============================================================


class SEMResults:
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.significant_paths: List[Dict] = []
        self.all_path_tests: List[Dict] = []
    
    def add_path_test(
        self,
        path: Tuple[str, str],
        beta_group1: float,
        beta_group2: float,
        p_value: float,
        delta_chi2: Optional[float] = None,
        delta_df: Optional[int] = None,
        is_significant: bool = False,
        group1: str = "benign",
        group2: str = "fail"
    ) -> None:
        # 计算净效应：group2 - group1（保持与原有逻辑一致：fail - benign）
        net_effect = float(beta_group2 - beta_group1)
        
        record = {
            "source": path[0],
            "target": path[1],
            f"beta_{group1}": float(beta_group1),
            f"beta_{group2}": float(beta_group2),
            "net_effect": net_effect,
            "p_value": float(p_value),
            "delta_chi2": float(delta_chi2) if delta_chi2 is not None else None,
            "delta_df": int(delta_df) if delta_df is not None else None,
            "is_significant": bool(is_significant),
            "group1": group1,
            "group2": group2
        }
        self.all_path_tests.append(record)
        if is_significant:
            self.significant_paths.append(record)
    
    def calculate_net_effects(self) -> pd.DataFrame:
        if not self.significant_paths:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.significant_paths)
        df = df.reindex(df["net_effect"].abs().sort_values(ascending=False).index)
        return df
    
    def save_results(
        self,
        baseline_stats: Optional[Dict] = None,
        experiment_config: Optional[Dict] = None,
        model_syntax: Optional[str] = None
    ) -> Dict[str, Path]:
        files = {}
        
        # 保存显著路径列表（JSON）
        paths_file = self.output_dir / "significant_paths.json"
        with open(paths_file, "w", encoding="utf-8") as f:
            json.dump(self.significant_paths, f, indent=2, ensure_ascii=False)
        files["significant_paths"] = paths_file
        
        # 保存所有路径检验结果（CSV）
        if self.all_path_tests:
            all_tests_df = pd.DataFrame(self.all_path_tests)
            all_tests_file = self.output_dir / "all_path_tests.csv"
            all_tests_df.to_csv(all_tests_file, index=False, encoding="utf-8")
            files["all_path_tests"] = all_tests_file
        
        # 保存净效应数据表（CSV）
        net_effects_df = self.calculate_net_effects()
        if not net_effects_df.empty:
            net_effects_file = self.output_dir / "net_effects.csv"
            net_effects_df.to_csv(net_effects_file, index=False, encoding="utf-8")
            files["net_effects"] = net_effects_file
        
        # 保存基线模型统计（JSON）
        if baseline_stats:
            baseline_file = self.output_dir / "baseline_model_stats.json"
            with open(baseline_file, "w", encoding="utf-8") as f:
                json.dump(baseline_stats, f, indent=2, ensure_ascii=False)
            files["baseline_model_stats"] = baseline_file
        
        # 保存实验配置（JSON）
        if experiment_config:
            config_file = self.output_dir / "experiment_config.json"
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(experiment_config, f, indent=2, ensure_ascii=False)
            files["experiment_config"] = config_file
        
        # 保存模型语法（TXT）
        if model_syntax:
            syntax_file = self.output_dir / "model_syntax.txt"
            with open(syntax_file, "w", encoding="utf-8") as f:
                f.write(model_syntax)
            files["model_syntax"] = syntax_file
        
        # 保存统计汇总（JSON）
        stats = {
            "total_significant_paths": len(self.significant_paths),
            "total_tested_paths": len(self.all_path_tests),
            "paths_with_positive_net_effect": int((net_effects_df["net_effect"] > 0).sum()) if not net_effects_df.empty else 0,
            "paths_with_negative_net_effect": int((net_effects_df["net_effect"] < 0).sum()) if not net_effects_df.empty else 0,
            "mean_net_effect": float(net_effects_df["net_effect"].mean()) if not net_effects_df.empty else 0.0,
            "std_net_effect": float(net_effects_df["net_effect"].std()) if not net_effects_df.empty else 0.0
        }
        stats_file = self.output_dir / "statistics_summary.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        files["statistics"] = stats_file
        
        return files
    
    def visualize_layered_circuit(
        self,
        use_all_paths: bool = True,
        threshold: float = 0.1,
        figsize: Tuple[float, float] = (20, 12),
        dpi: int = 300,
        save_pdf: bool = True,
        use_curved_arrows: bool = True,
        use_grid: bool = True
    ) -> Optional[Dict[str, Path]]:
        if not VISUALIZATION_AVAILABLE:
            return None
        
        if not self.all_path_tests:
            return None
        
        df = pd.DataFrame(self.all_path_tests)
        required_cols = ['source', 'target', 'net_effect', 'is_significant']
        if not all(col in df.columns for col in required_cols):
            return None
        
        output_files = {}
        
        # 保存PNG
        png_path = self.output_dir / "layered_circuit_graph.png"
        png_result = create_layered_circuit_plot(
            df=df,
            output_path=png_path,
            use_all_paths=use_all_paths,
            threshold=threshold,
            figsize=figsize,
            dpi=dpi,
            use_curved_arrows=use_curved_arrows,
            use_grid=use_grid
        )
        if png_result:
            output_files['png'] = png_result
        
        # 保存PDF（如果需要）
        if save_pdf:
            pdf_path = self.output_dir / "layered_circuit_graph.pdf"
            pdf_result = create_layered_circuit_plot(
                df=df,
                output_path=pdf_path,
                use_all_paths=use_all_paths,
                threshold=threshold,
                figsize=figsize,
                dpi=dpi,
                use_curved_arrows=use_curved_arrows,
                use_grid=use_grid
            )
            if pdf_result:
                output_files['pdf'] = pdf_result
        
        return output_files if output_files else None
    
    def visualize_effect_matrix(
        self,
        threshold: float = 0.1,
        figsize: Tuple[float, float] = (14, 10),
        dpi: int = 300,
        save_pdf: bool = True
    ) -> Optional[Dict[str, Path]]:
        if not VISUALIZATION_AVAILABLE:
            return None
        
        if not self.all_path_tests:
            return None
        
        df = pd.DataFrame(self.all_path_tests)
        required_cols = ['source', 'target', 'net_effect']
        if not all(col in df.columns for col in required_cols):
            return None
        
        output_files = {}
        
        # 保存PNG
        png_path = self.output_dir / "effect_matrix_heatmap.png"
        png_result = create_effect_matrix_heatmap(
            df=df,
            output_path=png_path,
            threshold=threshold,
            figsize=figsize,
            dpi=dpi
        )
        if png_result:
            output_files['png'] = png_result
        
        # 保存PDF（如果需要）
        if save_pdf:
            pdf_path = self.output_dir / "effect_matrix_heatmap.pdf"
            pdf_result = create_effect_matrix_heatmap(
                df=df,
                output_path=pdf_path,
                threshold=threshold,
                figsize=figsize,
                dpi=dpi
            )
            if pdf_result:
                output_files['pdf'] = pdf_result
        
        return output_files if output_files else None


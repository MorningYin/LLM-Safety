# -*- coding: utf-8 -*-
"""
plot_utils.py

绘图相关的底层工具函数：热力图、对比图、差值图等。
纯函数，无副作用，可复用。
"""

from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

CMAP_WHITE_ZERO = LinearSegmentedColormap.from_list(
    "contrast_white_zero",
    ["#0d47a1", "#ffffff", "#b71c1c"],
)


def plot_heatmap(
    matrices: List[np.ndarray],
    token_labels: List[str],
    output_path: Union[str, Path],
    titles: Optional[List[str]] = None,
):
    """绘制热力图。
    
    底层逻辑：纯函数，无副作用。
    
    Args:
        matrices: 矩阵列表，每个矩阵 shape (n_layers, n_tokens)
        token_labels: token 标签列表
        output_path: 输出文件路径
        titles: 每个矩阵的标题（可选）
    """
    if not matrices:
        raise ValueError("没有可绘制的矩阵。")
    titles = titles or [f"Panel {i+1}" for i in range(len(matrices))]

    n_layers, n_pos = matrices[0].shape
    for mat in matrices:
        if mat.shape != (n_layers, n_pos):
            raise ValueError("所有矩阵形状需一致。")

    # ---------------------------------------------------------
    # 自动分层：根据模型层数平均分为三份
    # ---------------------------------------------------------
    chunk_size = n_layers // 3
    remainder = n_layers % 3
    
    # 分配层数：尽可能均匀，余数分配给前几个chunk
    sizes = [chunk_size] * 3
    for i in range(remainder):
        sizes[i] += 1
    
    # 计算每个chunk的起始和结束位置
    layer_chunks = []
    start = 0
    for size in sizes:
        end = start + size
        layer_chunks.append((start, end, list(range(start, end))))
        start = end

    # ---------------------------------------------------------
    # 每一行单独计算（该段层）的 vmin/vmax
    # ---------------------------------------------------------
    row_norms = []
    for start_layer, end_layer, _ in layer_chunks:
        row_chunks = []
        for mat in matrices:
            chunk = mat[start_layer:min(end_layer, mat.shape[0]), :]
            if chunk.size > 0:
                row_chunks.append(chunk)

        if not row_chunks:
            row_norms.append(TwoSlopeNorm(vmin=-1e-6, vcenter=0.0, vmax=1e-6))
            continue

        row_vmin = float(min(c.min() for c in row_chunks))
        row_vmax = float(max(c.max() for c in row_chunks))

        if row_vmin >= 0:
            row_vmin = -1e-6
        if row_vmax <= 0:
            row_vmax = 1e-6

        row_norms.append(TwoSlopeNorm(vmin=row_vmin, vcenter=0.0, vmax=row_vmax))

    # ---------------------------------------------------------
    # 动态确定列数：3 行 × n_cols 列
    # ---------------------------------------------------------
    n_cols = len(matrices)
    n_rows = 3  # 固定为 3 行（对应 3 个层段）
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6))
    plt.subplots_adjust(wspace=0.1, hspace=0.05)

    # 保证 axes 为 2D（处理只有一列的情况）
    if axes.ndim == 1:
        # 只有一列时，axes 是 (n_rows,) 形状，需要转成 (n_rows, 1)
        axes = axes.reshape(n_rows, n_cols)

    # 保存每行 imshow，用于 colorbar
    row_ims = [[] for _ in range(3)]

    # ---------------------------------------------------------
    # 绘图主循环：行=layer_chunk，列=数据集
    # ---------------------------------------------------------
    for col_idx, (mat, title) in enumerate(zip(matrices, titles)):
        for row_idx, ((start_layer, end_layer, layer_ticks), row_norm) in enumerate(zip(layer_chunks, row_norms)):
            ax = axes[row_idx, col_idx]

            chunk = mat[start_layer:min(end_layer, mat.shape[0]), :]
            if chunk.shape[0] == 0:
                ax.set_visible(False)
                continue

            im = ax.imshow(chunk, aspect="auto", cmap=CMAP_WHITE_ZERO, norm=row_norm)
            row_ims[row_idx].append((im, ax))

            # 标题（仅第一行）
            if row_idx == 0:
                ax.set_title(title)

            # X-axis
            ax.set_xticks(range(n_pos))
            if row_idx == 2:
                ax.set_xticklabels(token_labels, rotation=45, ha="right")
                ax.set_xlabel("EOI Token Index")
            else:
                ax.set_xticklabels([])

            # Y-axis
            chunk_layers = layer_ticks[: chunk.shape[0]]
            ax.set_yticks(range(len(chunk_layers)))
            if col_idx == 0:
                ax.set_yticklabels(chunk_layers)
                if row_idx == 1:
                    ax.set_ylabel("Layer")
            else:
                ax.set_yticks([])
                ax.set_yticklabels([])

    # ---------------------------------------------------------
    # 每一行绑定一个 colorbar，并与本行所有子图自动对齐
    # ---------------------------------------------------------
    for row_idx in range(3):
        if not row_ims[row_idx]:
            continue

        # 当前行的所有 im 和 ax
        ims = [im for (im, ax) in row_ims[row_idx]]
        row_axes = [ax for (im, ax) in row_ims[row_idx]]

        # 绑定 colorbar 到当前行三个子图
        fig.colorbar(
            ims[0],  # 该行任何一个 im 都可
            ax=row_axes,
            location="right",
            fraction=0.05,
            pad=0.02,
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[完成] 热力图已保存至：{output_path}")


def plot_comparison(
    baseline: np.ndarray,
    aligned: np.ndarray,
    token_labels: List[str],
    output_path: Union[str, Path],
    baseline_title: str = "Baseline",
    aligned_title: str = "Aligned",
):
    """绘制对比图（baseline vs aligned）。
    
    底层逻辑：纯函数，无副作用。
    
    Args:
        baseline: baseline 矩阵，shape (n_layers, n_tokens)
        aligned: aligned 矩阵，shape (n_layers, n_tokens)
        token_labels: token 标签列表
        output_path: 输出文件路径
        baseline_title: baseline 的标题
        aligned_title: aligned 的标题
    """
    plot_heatmap(
        matrices=[baseline, aligned],
        token_labels=token_labels,
        output_path=output_path,
        titles=[baseline_title, aligned_title],
    )


def plot_delta(
    delta: np.ndarray,
    token_labels: List[str],
    output_path: Union[str, Path],
    title: str = "Delta",
):
    """绘制差值图。
    
    底层逻辑：纯函数，无副作用。
    
    Args:
        delta: 差值矩阵，shape (n_layers, n_tokens)
        token_labels: token 标签列表
        output_path: 输出文件路径
        title: 图的标题
    """
    plot_heatmap(
        matrices=[delta],
        token_labels=token_labels,
        output_path=output_path,
        titles=[title],
    )


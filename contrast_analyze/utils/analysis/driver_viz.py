"""可视化工具：热力图、分布图、组件对比等。"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def visualize_scan_heatmap(df: pd.DataFrame, save_path: Path):
    """绘制 T-stat 热力图。"""
    if df.empty:
        return
    pivot_table = df.pivot(index="layer", columns="token", values="t_stat")
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, cmap="RdBu_r", center=0, cbar_kws={"label": "T-statistic (Fail - Benign)"})
    plt.title("Total Residual Projection Difference (T-stat)")
    plt.ylabel("Layer")
    plt.xlabel("EOI Token Position")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def visualize_token_drivers_distribution(token_idx: int, valid_components_by_token: Dict[int, List[Dict]], n_layers: int, n_heads: int, save_path: Path):
    """为每个 EOI token 绘制合理组件的分布热力图。"""
    valid_components = valid_components_by_token.get(token_idx, [])
    if not valid_components:
        return
    matrix = np.zeros((n_layers, n_heads + 1))
    for comp in valid_components:
        layer = comp["layer"]
        name = comp["component"]
        if name == "MLP":
            col_idx = n_heads
        else:
            try:
                col_idx = int(name[1:])
            except Exception:
                continue
        matrix[layer, col_idx] = abs(comp.get("cohens_d", 0.0))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(14, 10))
    head_labels = [f"H{i}" for i in range(n_heads)] + ["MLP"]
    layer_labels = [f"L{i}" for i in range(n_layers)]
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", interpolation="nearest", vmin=0)
    ax.set_xticks(np.arange(len(head_labels)))
    ax.set_yticks(np.arange(len(layer_labels)))
    ax.set_xticklabels(head_labels, fontsize=10)
    ax.set_yticklabels(layer_labels, fontsize=10)
    ax.set_xlabel("Attention Head / MLP", fontsize=12, fontweight="bold")
    ax.set_ylabel("Layer", fontsize=12, fontweight="bold")
    ax.set_title(f"Valid Component Distribution for EOI Token {token_idx}", fontsize=14, fontweight="bold", pad=20)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("|Cohen's d|", fontsize=11, fontweight="bold")
    for layer in range(n_layers):
        for head in range(n_heads + 1):
            if matrix[layer, head] > 0:
                comp_info = None
                for comp in valid_components:
                    if comp["layer"] == layer and ((head == n_heads and comp["component"] == "MLP") or comp["component"] == f"H{head}"):
                        comp_info = comp
                        break
                if comp_info:
                    cohens_d = comp_info.get("cohens_d", 0.0)
                    ax.text(head, layer, f"{cohens_d:.2f}", ha="center", va="center", color="black", fontsize=8, fontweight="bold")
    ax.set_xticks(np.arange(len(head_labels)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(layer_labels)) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def visualize_component_projections(
    token_idx: int,
    top_components: List[Dict],
    storage_root_benign: str,
    storage_root_success: str,
    storage_root_fail: str,
    save_path: Path,
    loader_mean,
    page_num: int = 1,
    total_pages: int = 1,
):
    """为 token 的组件绘制三类数据集的平均投影柱状图。"""
    if not top_components:
        return
    component_data = []
    for comp in top_components:
        layer_idx = comp["layer"]
        component_name = comp["component"]
        mean_benign, mean_success, mean_fail = loader_mean(
            layer_idx=layer_idx,
            token_idx=token_idx,
            component_name=component_name,
            storage_root_benign=storage_root_benign,
            storage_root_success=storage_root_success,
            storage_root_fail=storage_root_fail,
        )
        component_data.append(
            {
                "layer": layer_idx,
                "component": component_name,
                "mean_benign": mean_benign,
                "mean_success": mean_success,
                "mean_fail": mean_fail,
                "mean_diff": comp.get("mean_diff", 0.0),
                "lasso_coef": comp.get("lasso_coef", 0.0),
            }
        )

    labels = [f"L{c['layer']}_{c['component']}" for c in component_data]
    benign_vals = [c["mean_benign"] for c in component_data]
    success_vals = [c["mean_success"] for c in component_data]
    fail_vals = [c["mean_fail"] for c in component_data]

    plt.rcParams.update(
        {
            "font.size": 11,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "axes.linewidth": 1.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "legend.frameon": True,
            "legend.edgecolor": "0.8",
            "legend.framealpha": 1.0,
            "figure.dpi": 300,
        }
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(labels))
    width = 0.28
    color_b, color_s, color_f = "#4472C4", "#70AD47", "#C55A11"
    ax.bar(x - width, benign_vals, width, label="Benign", color=color_b, edgecolor="white", linewidth=1.2, alpha=0.9)
    ax.bar(x, success_vals, width, label="Success", color=color_s, edgecolor="white", linewidth=1.2, alpha=0.9)
    ax.bar(x + width, fail_vals, width, label="Failure", color=color_f, edgecolor="white", linewidth=1.2, alpha=0.9)
    ax.set_xlabel("Component (Layer_Head/MLP)", fontsize=13, fontweight="bold", labelpad=10)
    ax.set_ylabel("Mean Projection Value", fontsize=13, fontweight="bold", labelpad=10)
    title = f"Mean Projection Comparison (Token {token_idx}"
    if total_pages > 1:
        title += f", Page {page_num}/{total_pages}"
    title += ")"
    fig.text(
        0.5,
        0.93,
        "Sorted by |Lasso Coefficient|",
        transform=fig.transFigure,
        fontsize=11,
        ha="center",
        style="italic",
        color="0.5",
        verticalalignment="top",
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10.5)
    ax.grid(axis="y", alpha=0.25, linestyle="-", linewidth=0.8)
    ax.legend(loc="upper left", fontsize=11, frameon=True, edgecolor="0.8", framealpha=1.0, borderpad=0.8, handlelength=1.5)
    ax.spines["left"].set_color("0.3")
    ax.spines["bottom"].set_color("0.3")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none", pad_inches=0.1)
    plt.close()
    plt.rcParams.update(plt.rcParamsDefault)
    return component_data


def visualize_component_distribution(
    layer_idx: int,
    token_idx: int,
    component_name: str,
    projections_benign: np.ndarray,
    projections_success: np.ndarray,
    projections_fail: np.ndarray,
    fit_results_benign: Dict,
    fit_results_success: Dict,
    fit_results_fail: Dict,
    save_path: Path,
):
    """绘制组件分布直方图 + 拟合曲线。"""
    plt.rcParams.update(
        {
            "font.size": 11,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "axes.linewidth": 1.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "legend.frameon": True,
            "legend.edgecolor": "0.8",
            "legend.framealpha": 1.0,
            "figure.dpi": 300,
        }
    )

    color_b, color_s, color_f = "#4472C4", "#70AD47", "#C55A11"
    all_data = np.concatenate([projections_benign, projections_success, projections_fail])
    all_data = all_data[np.isfinite(all_data)]
    if len(all_data) == 0:
        plt.close()
        return

    data_min, data_max = all_data.min(), all_data.max()
    bins = np.linspace(data_min, data_max, 50)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(projections_benign, bins=bins, alpha=0.6, color=color_b, edgecolor="white", linewidth=0.8, label="Benign", density=True)
    ax.hist(projections_success, bins=bins, alpha=0.6, color=color_s, edgecolor="white", linewidth=0.8, label="Success", density=True)
    ax.hist(projections_fail, bins=bins, alpha=0.6, color=color_f, edgecolor="white", linewidth=0.8, label="Failure", density=True)

    x_curve = np.linspace(data_min, data_max, 1000)
    dist_map = {"norm": stats.norm, "t": stats.t, "gamma": stats.gamma, "lognorm": stats.lognorm, "laplace": stats.laplace, "beta": stats.beta}

    for fit_results, color, label in [
        (fit_results_benign, color_b, "Benign"),
        (fit_results_success, color_s, "Success"),
        (fit_results_fail, color_f, "Failure"),
    ]:
        dist_name = fit_results.get("best_dist_name")
        params = fit_results.get("best_params")
        if not dist_name or not params or dist_name not in dist_map:
            continue
        dist = dist_map[dist_name]
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                if dist_name == "beta":
                    data_min_v = fit_results.get("data_min", data_min)
                    data_max_v = fit_results.get("data_max", data_max)
                    if data_max_v - data_min_v > 1e-10:
                        x_norm = (x_curve - data_min_v) / (data_max_v - data_min_v)
                        y_curve = dist.pdf(x_norm, *params) / (data_max_v - data_min_v)
                        ax.plot(x_curve, y_curve, color=color, linewidth=2.5, linestyle="-", label=f"{label}: {dist_name}")
                else:
                    y_curve = dist.pdf(x_curve, *params)
                    ax.plot(x_curve, y_curve, color=color, linewidth=2.5, linestyle="-", label=f"{label}: {dist_name}")
        except Exception:
            continue

    ax.set_xlabel("Projection Value", fontsize=13, fontweight="bold", labelpad=10)
    ax.set_ylabel("Density", fontsize=13, fontweight="bold", labelpad=10)
    ax.set_title(f"Distribution (Layer {layer_idx}, Token {token_idx}, {component_name})", fontsize=14, fontweight="bold", pad=10)
    ax.grid(axis="y", alpha=0.25, linestyle="-", linewidth=0.8)
    ax.legend(loc="upper right", fontsize=10, frameon=True, edgecolor="0.8", framealpha=1.0, borderpad=0.8, handlelength=1.5)
    ax.spines["left"].set_color("0.3")
    ax.spines["bottom"].set_color("0.3")

    stats_text = []
    for dataset_name, fit_result in [("Benign", fit_results_benign), ("Success", fit_results_success), ("Failure", fit_results_fail)]:
        if fit_result.get("best_dist_name"):
            stats_text.append(
                f"{dataset_name}: {fit_result['best_dist_name']}\n"
                f"  AIC={fit_result.get('best_aic', 'N/A'):.2f}, BIC={fit_result.get('best_bic', 'N/A'):.2f}\n"
                f"  KS={fit_result.get('best_ks_stat', 0):.4f}, p={fit_result.get('best_p_value', 0):.2e}\n"
                f"  μ={fit_result.get('data_mean', 0):.4f}, σ={fit_result.get('data_std', 0):.4f}"
            )
    if stats_text:
        ax.text(
            0.02,
            0.98,
            "\n\n".join(stats_text),
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            family="monospace",
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none", pad_inches=0.1)
    plt.close()
    plt.rcParams.update(plt.rcParamsDefault)


"""统计与分布工具：拟合、重叠度、显著性扫描等。"""

from __future__ import annotations

import warnings
import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde

logger = logging.getLogger(__name__)


def fit_distribution(data: np.ndarray) -> Dict:
    """拟合多种分布并选择最佳（按 AIC）。"""
    if len(data) < 3:
        return {"best_dist_name": None, "best_params": None, "all_fits": []}

    data = data[np.isfinite(data)]
    if len(data) < 3:
        return {"best_dist_name": None, "best_params": None, "all_fits": []}

    distributions = {
        "norm": stats.norm,
        "t": stats.t,
        "gamma": stats.gamma,
        "lognorm": stats.lognorm,
        "laplace": stats.laplace,
        "beta": stats.beta,
    }

    all_fits = []
    data_min, data_max = data.min(), data.max()

    for dist_name, dist in distributions.items():
        try:
            if dist_name in {"gamma", "lognorm"}:
                if data_min <= 0:
                    continue
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    params = dist.fit(data, floc=0)
            elif dist_name == "beta":
                if data_max - data_min < 1e-10:
                    continue
                data_normalized = (data - data_min) / (data_max - data_min + 1e-10)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    params = dist.fit(data_normalized, floc=0, fscale=1)
            else:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    params = dist.fit(data)

            n = len(data)
            k = len(params)

            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    if dist_name == "beta":
                        log_likelihood = np.sum(dist.logpdf(data_normalized, *params))
                    else:
                        log_likelihood = np.sum(dist.logpdf(data, *params))
            except Exception:
                log_likelihood = -np.inf

            aic = 2 * k - 2 * log_likelihood
            bic = k * np.log(n) - 2 * log_likelihood

            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    if dist_name == "beta":
                        ks_stat, p_value = stats.kstest(data_normalized, lambda x: dist.cdf(x, *params))
                    else:
                        ks_stat, p_value = stats.kstest(data, lambda x: dist.cdf(x, *params))
            except Exception:
                ks_stat, p_value = np.inf, 0.0

            all_fits.append(
                {
                    "dist_name": dist_name,
                    "params": [float(p) for p in params],
                    "aic": float(aic),
                    "bic": float(bic),
                    "ks_stat": float(ks_stat),
                    "p_value": float(p_value),
                    "n_params": k,
                    "n_samples": n,
                    "data_min": float(data_min),
                    "data_max": float(data_max),
                    "data_mean": float(data.mean()),
                    "data_std": float(data.std()),
                }
            )
        except Exception:
            continue

    if not all_fits:
        return {"best_dist_name": None, "best_params": None, "all_fits": []}

    all_fits.sort(key=lambda x: x["aic"])
    best_fit = all_fits[0]
    return {
        "best_dist_name": best_fit["dist_name"],
        "best_params": best_fit["params"],
        "all_fits": all_fits,
        "best_ks_stat": best_fit["ks_stat"],
        "best_p_value": best_fit["p_value"],
        "best_aic": best_fit["aic"],
        "best_bic": best_fit["bic"],
        "data_min": best_fit["data_min"],
        "data_max": best_fit["data_max"],
        "data_mean": best_fit["data_mean"],
        "data_std": best_fit["data_std"],
    }


def calculate_distribution_overlap(data1: np.ndarray, data2: np.ndarray) -> float:
    """计算两个分布的重叠度（0-1）。"""
    data1 = data1[np.isfinite(data1)]
    data2 = data2[np.isfinite(data2)]
    if len(data1) < 2 or len(data2) < 2:
        return 1.0
    try:
        kde1, kde2 = gaussian_kde(data1), gaussian_kde(data2)
        data_min, data_max = min(data1.min(), data2.min()), max(data1.max(), data2.max())
        if data_max - data_min < 1e-10:
            return 1.0
        margin = (data_max - data_min) * 0.1
        x_points = np.linspace(data_min - margin, data_max + margin, 100)
        min_pdf = np.minimum(kde1(x_points), kde2(x_points))
        overlap = np.trapz(min_pdf, x_points)
        return float(max(0.0, min(1.0, overlap)))
    except Exception as e:
        logger.warning(f"分布重叠度计算失败: {e}，返回 1.0")
        return 1.0


def scan_significant_differences(storage_root_benign: str, storage_root_fail: str, n_layers: int, n_tokens: int, loader) -> pd.DataFrame:
    """扫描所有 (Layer, Token) 位置，计算总残差投影差异的 T 检验。"""
    from tqdm import tqdm

    results = []
    for layer_idx in tqdm(range(n_layers), desc="Scanning Layers"):
        y_benign = loader(storage_root_benign, layer_idx, "block", "resid_proj")
        y_fail = loader(storage_root_fail, layer_idx, "block", "resid_proj")
        if len(y_benign) == 0 or len(y_fail) == 0:
            if layer_idx > 0:
                break
            continue
        current_k = y_benign.shape[1]
        for token_idx in range(current_k):
            vals_b = y_benign[:, token_idx].float().numpy()
            vals_f = y_fail[:, token_idx].float().numpy()
            t_stat, p_val = stats.ttest_ind(vals_f, vals_b, equal_var=False)
            mean_diff = np.mean(vals_f) - np.mean(vals_b)
            n1, n2 = len(vals_b), len(vals_f)
            var1, var2 = np.var(vals_b, ddof=1), np.var(vals_f, ddof=1)
            pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
            cohens_d = mean_diff / pooled_se if pooled_se > 0 else 0
            results.append(
                {
                    "layer": layer_idx,
                    "token": token_idx,
                    "t_stat": t_stat,
                    "p_value": p_val,
                    "mean_diff": mean_diff,
                    "cohens_d": cohens_d,
                    "abs_t": abs(t_stat),
                }
            )
    return pd.DataFrame(results)


def generate_summary_statistics_effect_overlap(
    significant_regions: pd.DataFrame,
    all_valid_components: List[Dict],
    effect_size_threshold: float = 1.0,
    overlap_threshold: float = 0.5,
) -> Dict:
    """生成统计摘要（基于效果量和重叠度筛选）。"""
    if not all_valid_components:
        return {
            "total_positions": 0,
            "total_components": 0,
            "summary": "No valid components found",
        }

    df_components = pd.DataFrame(all_valid_components)
    layer_stats = significant_regions.groupby("layer").agg(
        token=("token", "count"),
        t_stat_mean=("t_stat", "mean"),
        t_stat_max=("t_stat", "max"),
        mean_diff_mean=("mean_diff", "mean"),
        mean_diff_max=("mean_diff", "max"),
        mean_diff_min=("mean_diff", "min"),
    )
    token_stats = significant_regions.groupby("token").agg(
        layer=("layer", "count"),
        t_stat_mean=("t_stat", "mean"),
        t_stat_max=("t_stat", "max"),
        mean_diff_mean=("mean_diff", "mean"),
        mean_diff_max=("mean_diff", "max"),
        mean_diff_min=("mean_diff", "min"),
    )

    top_components = df_components.sort_values(by="cohens_d", key=lambda x: x.abs(), ascending=False).head(50)

    stats_dict = {
        "筛选条件": {
            "t_stat_threshold": float(significant_regions["abs_t"].min()) if len(significant_regions) > 0 else None,
            "p_value_threshold": overlap_threshold,  # 保留原逻辑字段名
            "effect_size_threshold": effect_size_threshold,
            "overlap_threshold": overlap_threshold,
        },
        "位置统计": {
            "总显著位置数": int(len(significant_regions)),
            "涉及层数": int(significant_regions["layer"].nunique()) if len(significant_regions) else 0,
            "涉及token数": int(significant_regions["token"].nunique()) if len(significant_regions) else 0,
            "层范围": [int(significant_regions["layer"].min()), int(significant_regions["layer"].max())] if len(significant_regions) else None,
            "token范围": [int(significant_regions["token"].min()), int(significant_regions["token"].max())] if len(significant_regions) else None,
        },
        "组件统计": {
            "总组件数": int(len(df_components)),
            "平均效果量": float(df_components["cohens_d"].abs().mean()),
            "平均重叠度": float(df_components["overlap"].mean()),
            "最小重叠度": float(df_components["overlap"].min()),
        },
        "按层统计": layer_stats.to_dict("index") if len(layer_stats) else {},
        "按token统计": token_stats.to_dict("index") if len(token_stats) else {},
        "Top影响组件": top_components[
            ["layer", "token", "component", "cohens_d", "overlap", "mean_diff", "diff_p_val"]
        ].to_dict("records")
        if len(top_components)
        else [],
    }
    return stats_dict


def convert_to_serializable(obj):
    """将 numpy 类型转换为 Python 原生类型。"""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj


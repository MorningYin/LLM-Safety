"""驱动分析核心 orchestrator。"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from contrast_analyze.utils.analysis.driver_io import (
    load_data,
    load_projection_from_storage,
    load_component_mean_projections,
    load_component_projections,
)
from contrast_analyze.utils.analysis.driver_stats import (
    fit_distribution,
    calculate_distribution_overlap,
    scan_significant_differences,
    generate_summary_statistics_effect_overlap,
    convert_to_serializable,
)
from contrast_analyze.utils.analysis.driver_viz import (
    visualize_scan_heatmap,
    visualize_token_drivers_distribution,
    visualize_component_distribution,
    visualize_component_projections,
)
from contrast_analyze.utils.pipeline_core import setup_environment, log_section
from contrast_analyze.utils.pipeline_config import PipelineConfig

logger = logging.getLogger(__name__)


def analyze_drivers_at_position_effect_overlap(
    layer_idx: int,
    token_idx: int,
    storage_root_benign: str,
    storage_root_fail: str,
    effect_size_threshold: float = 1.0,
    overlap_threshold: float = 0.5,
) -> Dict:
    """在指定位置进行驱动因素分析（效果量+重叠度，无 Lasso）。"""
    attn_b = load_projection_from_storage(storage_root_benign, layer_idx, "o_proj", "z_capture")[:, token_idx, :]
    mlp_b = load_projection_from_storage(storage_root_benign, layer_idx, "mlp", "mlp_proj")[:, token_idx].unsqueeze(1)
    y_b = load_projection_from_storage(storage_root_benign, layer_idx, "block", "resid_proj")[:, token_idx]

    attn_f = load_projection_from_storage(storage_root_fail, layer_idx, "o_proj", "z_capture")[:, token_idx, :]
    mlp_f = load_projection_from_storage(storage_root_fail, layer_idx, "mlp", "mlp_proj")[:, token_idx].unsqueeze(1)
    y_f = load_projection_from_storage(storage_root_fail, layer_idx, "block", "resid_proj")[:, token_idx]

    n_heads = attn_b.shape[1]
    feature_names = [f"H{i}" for i in range(n_heads)] + ["MLP"]
    X_b = np.concatenate([attn_b, mlp_b], axis=1).astype(np.float32)
    X_f = np.concatenate([attn_f, mlp_f], axis=1).astype(np.float32)
    Y_b = y_b.float().numpy()
    Y_f = y_f.float().numpy()

    component_stats = []
    significant_indices = []
    for i in range(X_b.shape[1]):
        feat_b, feat_f = X_b[:, i], X_f[:, i]
        n1, n2 = len(feat_b), len(feat_f)
        mean1, mean2 = np.mean(feat_b), np.mean(feat_f)
        var1, var2 = np.var(feat_b, ddof=1), np.var(feat_f, ddof=1)
        pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        cohens_d = (mean2 - mean1) / pooled_se if pooled_se > 0 else 0.0
        overlap = calculate_distribution_overlap(feat_b, feat_f)
        t_stat, p_val = None, None
        try:
            t_stat, p_val = np.nan, np.nan
        except Exception:
            pass
        mean_diff = np.mean(feat_f) - np.mean(feat_b)
        stat_item = {
            "component": feature_names[i],
            "cohens_d": cohens_d,
            "overlap": overlap,
            "diff_t_stat": t_stat,
            "diff_p_val": p_val,
            "mean_diff": mean_diff,
        }
        component_stats.append(stat_item)
        if abs(cohens_d) >= effect_size_threshold and overlap < overlap_threshold:
            significant_indices.append(i)

    if not significant_indices:
        return {"layer": layer_idx, "token": token_idx, "drivers": [], "component_stats": component_stats, "note": "No significant components"}

    valid_drivers = []
    for idx in significant_indices:
        comp_stat = component_stats[idx]
        valid_drivers.append(
            {
                "component": comp_stat["component"],
                "mean_diff": comp_stat["mean_diff"],
                "cohens_d": comp_stat["cohens_d"],
                "overlap": comp_stat["overlap"],
                "diff_t_stat": comp_stat["diff_t_stat"],
                "diff_p_val": comp_stat["diff_p_val"],
            }
        )
    valid_drivers.sort(key=lambda x: abs(x.get("cohens_d", 0.0)), reverse=True)
    return {
        "layer": layer_idx,
        "token": token_idx,
        "y_mean_diff": np.mean(Y_f) - np.mean(Y_b),
        "drivers": [],
        "valid_drivers": valid_drivers,
        "component_stats": component_stats,
    }


def run_driver_analysis(args):
    """Stage4 orchestrator。"""
    setup_environment()
    logger.info("=" * 80)
    logger.info("开始投影驱动因素分析（效果量和重叠度筛选，不使用 Lasso 回归）")
    logger.info(f"模型: {args.model_alias}, 方向模式: {args.direction_mode}")
    logger.info(f"筛选阈值: effect_size>={args.effect_size_threshold}, overlap<{args.overlap_threshold}")
    logger.info("=" * 80)

    config = PipelineConfig(model_alias=args.model_alias, direction_mode=args.direction_mode)
    config.ensure_dirs()

    log_section("Stage 4: 分析驱动因素")
    print(f"模型: {args.model_alias}")
    print(f"方向模式: {args.direction_mode}")
    print(f"输出目录: {config.driver_analysis_dir}")

    storage_root_b, storage_root_s, storage_root_f = load_data(config)
    sample_y = load_projection_from_storage(storage_root_b, 0, "block", "resid_proj")
    if len(sample_y) == 0:
        logger.error("无法加载投影数据，请检查 Stage3 输出。")
        return

    detected_layers = 0
    for i in range(100):
        proj = load_projection_from_storage(storage_root_b, i, "block", "resid_proj")
        if len(proj) > 0:
            detected_layers = i + 1
        elif i > detected_layers + 2:
            break
    n_layers = detected_layers
    n_tokens = sample_y.shape[1]
    logger.info(f"检测到维度: Layers={n_layers}, Tokens={n_tokens}")

    logger.info("步骤2: 扫描显著差异位置...")
    df_scan = scan_significant_differences(storage_root_b, storage_root_f, n_layers, n_tokens, load_projection_from_storage)
    logger.info(f"扫描完成: 共检查 {len(df_scan)} 个位置")

    out_dir = config.driver_analysis_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    distribution_plots_dir = out_dir / "distribution_plots"
    distribution_plots_dir.mkdir(parents=True, exist_ok=True)
    df_scan.to_csv(out_dir / "scan_results.csv", index=False)
    visualize_scan_heatmap(df_scan, out_dir / "scan_heatmap.png")

    significant_regions = df_scan[(df_scan["abs_t"] >= args.t_stat_threshold) & (df_scan["p_value"] < args.p_value_threshold)].copy()
    logger.info(f"筛选结果：{len(significant_regions)} 个显著差异位置")

    sample_attn = load_projection_from_storage(storage_root_b, 0, "o_proj", "z_capture")
    if len(sample_attn) == 0:
        logger.error("无法加载 attention 数据")
        return
    n_heads = sample_attn.shape[2]

    logger.info(f"步骤4: 开始分析所有位置（{n_layers} 层 × {n_tokens} tokens）...")
    all_drivers: List[Dict] = []
    all_valid_components: List[Dict] = []
    valid_components_by_token: Dict[int, List[Dict]] = {}

    for layer_idx in tqdm(range(n_layers), desc="Analyzing layers"):
        for token_idx in range(n_tokens):
            result = analyze_drivers_at_position_effect_overlap(
                layer_idx=layer_idx,
                token_idx=token_idx,
                storage_root_benign=storage_root_b,
                storage_root_fail=storage_root_f,
                effect_size_threshold=args.effect_size_threshold,
                overlap_threshold=args.overlap_threshold,
            )
            for d in result.get("drivers", []):
                d["layer"] = layer_idx
                d["token"] = token_idx
                all_drivers.append(d)
            valid_drivers = result.get("valid_drivers", [])
            for vd in valid_drivers:
                vd["layer"] = layer_idx
                vd["token"] = token_idx
                all_valid_components.append(vd)
                valid_components_by_token.setdefault(token_idx, []).append(vd)

    logger.info(f"步骤4完成: 合理组件 {len(all_valid_components)} 个，涉及 token {len(valid_components_by_token)} 个")

    logger.info("步骤5: 为每个 EOI token 绘制分布图...")
    for token_idx in tqdm(range(n_tokens), desc="Visualizing tokens"):
        if token_idx in valid_components_by_token:
            save_path = out_dir / f"token_{token_idx}_distribution.png"
            visualize_token_drivers_distribution(token_idx, valid_components_by_token, n_layers, n_heads, save_path)

    logger.info("步骤6: 为每个 EOI token 的组件生成分布图...")
    for token_idx in tqdm(range(n_tokens), desc="Generating distribution plots"):
        if token_idx not in valid_components_by_token:
            continue
        token_components = valid_components_by_token[token_idx]
        sorted_components = sorted(token_components, key=lambda x: abs(x.get("cohens_d", 0.0)), reverse=True)
        token_dist_dir = distribution_plots_dir / f"token_{token_idx}"
        token_dist_dir.mkdir(parents=True, exist_ok=True)
        all_distribution_fits = {}
        for comp in tqdm(sorted_components, desc=f"Token {token_idx} distributions"):
            layer_idx = comp["layer"]
            component_name = comp["component"]
            proj_benign, proj_success, proj_fail = load_component_projections(
                layer_idx=layer_idx,
                token_idx=token_idx,
                component_name=component_name,
                storage_root_benign=storage_root_b,
                storage_root_success=storage_root_s,
                storage_root_fail=storage_root_f,
            )
            if len(proj_benign) == 0 or len(proj_success) == 0 or len(proj_fail) == 0:
                continue
            fit_benign = fit_distribution(proj_benign)
            fit_success = fit_distribution(proj_success)
            fit_fail = fit_distribution(proj_fail)
            component_key = f"L{layer_idx}_{component_name}"
            all_distribution_fits[component_key] = {
                "layer": layer_idx,
                "component": component_name,
                "benign": fit_benign,
                "success": fit_success,
                "fail": fit_fail,
            }
            save_path = token_dist_dir / f"component_L{layer_idx}_{component_name}.png"
            visualize_component_distribution(
                layer_idx=layer_idx,
                token_idx=token_idx,
                component_name=component_name,
                projections_benign=proj_benign,
                projections_success=proj_success,
                projections_fail=proj_fail,
                fit_results_benign=fit_benign,
                fit_results_success=fit_success,
                fit_results_fail=fit_fail,
                save_path=save_path,
            )
        if all_distribution_fits:
            distribution_fits_serializable = convert_to_serializable(all_distribution_fits)
            json_path = token_dist_dir / "distribution_fits.json"
            with open(json_path, "w", encoding="utf-8") as f:
                import json

                json.dump(distribution_fits_serializable, f, indent=2, ensure_ascii=False)

    logger.info("步骤7: 保存结果...")
    if all_drivers:
        pd.DataFrame(all_drivers).to_csv(out_dir / "top_drivers.csv", index=False)
    if all_valid_components:
        pd.DataFrame(all_valid_components).to_csv(out_dir / "valid_components.csv", index=False)
        components_by_token_serializable = {str(token): convert_to_serializable(comps) for token, comps in valid_components_by_token.items()}
        valid_components_json = {
            "total_valid_components": len(all_valid_components),
            "tokens_with_valid_components": len(valid_components_by_token),
            "components_by_token": components_by_token_serializable,
        }
        with open(out_dir / "valid_components.json", "w", encoding="utf-8") as f:
            import json

            json.dump(valid_components_json, f, indent=2, ensure_ascii=False)

        if len(significant_regions) == 0:
            significant_regions = pd.DataFrame(columns=["layer", "token", "t_stat", "p_value", "mean_diff", "abs_t"])
        stats = generate_summary_statistics_effect_overlap(
            significant_regions=significant_regions,
            all_valid_components=all_valid_components,
            effect_size_threshold=args.effect_size_threshold,
            overlap_threshold=args.overlap_threshold,
        )
        with open(out_dir / "summary_statistics.json", "w", encoding="utf-8") as f:
            import json

            json.dump(stats, f, indent=2, ensure_ascii=False)
        with open(out_dir / "summary_statistics.txt", "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("投影驱动因素分析统计摘要\n")
            f.write("=" * 80 + "\n\n")
            f.write("【筛选条件】\n")
            f.write(f"  T-stat 阈值: {stats['筛选条件']['t_stat_threshold']}\n")
            f.write(f"  P-value 阈值: {stats['筛选条件']['p_value_threshold']}\n")
            f.write(f"  效果量阈值: {stats['筛选条件']['effect_size_threshold']}\n")
            f.write(f"  重叠度阈值: {stats['筛选条件']['overlap_threshold']}\n\n")
            f.write("【位置统计】\n")
            pos_stats = stats["位置统计"]
            f.write(f"  总显著位置数: {pos_stats['总显著位置数']}\n")
            f.write(f"  涉及层数: {pos_stats['涉及层数']}\n")
            f.write(f"  涉及token数: {pos_stats['涉及token数']}\n")
            if pos_stats["层范围"]:
                f.write(f"  层范围: {pos_stats['层范围'][0]} - {pos_stats['层范围'][1]}\n")
            if pos_stats["token范围"]:
                f.write(f"  Token范围: {pos_stats['token范围'][0]} - {pos_stats['token范围'][1]}\n")
            f.write("\n")
            f.write("【组件统计】\n")
            comp_stats = stats["组件统计"]
            f.write(f"  总组件数: {comp_stats['总组件数']}\n")
            f.write(f"  平均效果量: {comp_stats['平均效果量']:.4f}\n")
            f.write(f"  平均重叠度: {comp_stats['平均重叠度']:.4f}\n")
            f.write(f"  最小重叠度: {comp_stats['最小重叠度']:.4f}\n\n")
            f.write("【Top 影响组件】（前20个，按 |cohens_d| 排序）\n")
            for i, comp in enumerate(stats["Top影响组件"][:20], 1):
                f.write(
                    f"  {i}. Layer {comp['layer']} Token {comp['token']} {comp['component']}: "
                    f"cohens_d={comp['cohens_d']:.4f}, overlap={comp['overlap']:.4f}, "
                    f"diff={comp['mean_diff']:.4f}, p={comp['diff_p_val']:.2e}\n"
                )
            f.write("\n")

    logger.info("分析完成，结果保存在: %s", out_dir)


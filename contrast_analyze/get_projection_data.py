# -*- coding: utf-8 -*-
"""
Stage 3: 收集投影数据。

收集每一层每一个位置每一个模块（block、mlp、head）的内积数据，输出结构与 attention_projections 保持一致。
"""

from __future__ import annotations

import argparse

from contrast_analyze.utils.core import setup_environment, log_section, PipelineConfig
from contrast_analyze.utils.projection import run_stage3_projection


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 3: 收集投影数据 - 收集每层每个位置每个模块的内积数据"
    )
    parser.add_argument("--model_alias", type=str, required=True, help="模型别名")
    parser.add_argument("--direction_mode", type=str, required=True, choices=["direction", "directions"], help="方向模式")
    parser.add_argument("--batch_size", type=int, default=32, help="批处理大小")
    parser.add_argument("--n_samples", type=int, default=None, help="每个数据集参与计算的样本数量（可选）")
    parser.add_argument("--seed", type=int, default=2025, help="随机采样种子")
    return parser.parse_args()


def main():
    setup_environment()
    args = parse_args()

    config = PipelineConfig(
        model_alias=args.model_alias,
        direction_mode=args.direction_mode,
    )
    config.ensure_dirs()

    log_section("Stage 3: 收集投影数据")
    print(f"模型: {args.model_alias}")
    print(f"方向模式: {args.direction_mode}")
    print(f"批处理大小: {args.batch_size}")
    print(f"样本数量: {args.n_samples if args.n_samples else '全部'}")
    print(f"输出目录: {config.projection_analysis_dir}")

    run_stage3_projection(
        config,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
        seed=args.seed,
    )

    print("\n=== 所有数据计算完成 ===")
    print(f"数据保存根目录: {config.projection_analysis_dir}")
    print(f"目录结构示例: {config.projection_analysis_dir}/<category>/layer_0/... ")
    print("\n数据收集完成！")


if __name__ == \"__main__\":
    main()


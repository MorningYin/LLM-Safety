#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 4 入口：分析 Attention 投影驱动因素（效果量 + 重叠度，无 Lasso）。
核心逻辑已下沉至 utils.analysis.driver_core。
"""

from __future__ import annotations

import argparse
import logging

from contrast_analyze.utils.analysis.driver_core import run_driver_analysis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 4: 分析驱动因素 - 分析 Attention 投影驱动因素")
    parser.add_argument("--model_alias", type=str, required=True, help="模型别名")
    parser.add_argument("--direction_mode", type=str, required=True, choices=["direction", "directions"], help="方向模式")
    parser.add_argument("--t_stat_threshold", type=float, default=50.0, help="T统计量阈值（绝对值）")
    parser.add_argument("--p_value_threshold", type=float, default=0.01, help="差异显著性阈值")
    parser.add_argument("--effect_size_threshold", type=float, default=1.2, help="Cohen's d 阈值")
    parser.add_argument("--overlap_threshold", type=float, default=0.35, help="分布重叠度阈值（越小越严格）")
    parser.add_argument("--seed", type=int, default=2025, help="随机种子")
    return parser.parse_args()


def main():
    run_driver_analysis(parse_args())


if __name__ == "__main__":
    main()


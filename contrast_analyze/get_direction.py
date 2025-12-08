"""Stage 2: 生成拒答方向向量。"""

from __future__ import annotations

import argparse

from contrast_analyze.utils.core import setup_environment, log_section, PipelineConfig
from contrast_analyze.utils.direction import run_stage2_direction, train_independent_direction


def parse_arguments():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Stage 2: 生成拒答方向向量")
    parser.add_argument("--model_alias", type=str, required=True, help="模型别名")
    parser.add_argument("--seed", type=int, default=2025, help="随机种子")
    return parser.parse_args()


def main():
    args = parse_arguments()
    setup_environment(seed=args.seed)

    config = PipelineConfig(model_alias=args.model_alias)
    config.ensure_dirs()

    log_section("Stage 2: 生成拒答方向向量")
    print(f"模型: {args.model_alias}")
    print(f"输出目录: {config.direction_artifacts_dir}")

    # 第一步：使用现有方法生成第一个拒答方向
    print("\n[步骤 1/2] 生成原始拒答方向...")
    run_stage2_direction(config)
    
    # 第二步：基于第一个方向训练第二个独立方向
    print("\n[步骤 2/2] 训练独立拒答方向...")
    train_independent_direction(config)


if __name__ == "__main__":
    main()


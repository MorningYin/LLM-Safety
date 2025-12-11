"""
Stage 1: 获取原始数据
从 wildjailbreak 拉取数据并生成模型补全，保存为统一对齐的 JSONL。
"""

from __future__ import annotations

import os

cache_root = os.environ.get("HF_CACHE_ROOT", "/root/autodl-tmp/hf_cache")
os.makedirs(cache_root, exist_ok=True)
os.makedirs(os.path.join(cache_root, "datasets_cache"), exist_ok=True)
os.environ["HF_HOME"] = cache_root
os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_root, "datasets_cache")
os.environ["HUGGINGFACE_HUB_CACHE"] = cache_root
os.environ["TRANSFORMERS_CACHE"] = cache_root
os.environ["HUGGINGFACE_CACHE_DIR"] = cache_root

import argparse
import datetime
import sys
import traceback

from contrast_analyze.utils.core import setup_environment, log_section, PipelineConfig
from contrast_analyze.utils.data.pipeline import run_stage1


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1: 获取原始数据")
    parser.add_argument("--model_alias", type=str, required=True, help="模型别名")
    parser.add_argument("--data_type", type=str, required=True, choices=["harmful", "benign"], help="数据类型")
    parser.add_argument("--batch_size", type=int, default=64, help="批量推理大小")
    parser.add_argument("--n_samples", type=int, default=None, help="限制获取的样本数量（可选，用于测试模式）")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_environment()

    config = PipelineConfig(model_alias=args.model_alias)
    config.ensure_dirs()

    log_section("Stage 1: 获取原始数据")
    print(f"模型: {args.model_alias}")
    print(f"数据类型: {args.data_type}")
    print(f"批量推理大小: {args.batch_size}")
    print(f"样本数量限制: {args.n_samples if args.n_samples else '无限制'}")
    print(f"时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    stats = run_stage1(
        config,
        model_alias=args.model_alias,
        data_type=args.data_type,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
    )

    print(f"[完成] Benign 条数: {stats['benign']}, Harmful Success: {stats['harmful_success']}, Harmful Failure: {stats['harmful_failure']}")
    print("=============================== 完成 ================================")
    logger.info("Stage 1 数据获取完成")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Info] 用户中断程序")
        sys.exit(1)
    except Exception as e:
        print(f"\n[Error] 程序执行出错: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


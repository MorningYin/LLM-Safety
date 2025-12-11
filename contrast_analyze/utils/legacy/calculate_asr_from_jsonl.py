#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
calculate_asr_from_jsonl.py

从 jsonl 文件读取 completions，计算 ASR（Attack Success Rate）。
"""

import os

os.environ["NO_PROXY"] = "all://huggingface.co,all://cdn-lfs.huggingface.co,all://localhost,all://127.0.0.1"
os.environ["no_proxy"] = os.environ["NO_PROXY"]

cache_root = os.environ.get("HF_CACHE_ROOT", "/root/autodl-tmp/hf_cache")
os.makedirs(cache_root, exist_ok=True)
os.makedirs(os.path.join(cache_root, "datasets_cache"), exist_ok=True)
os.environ["HF_HOME"] = cache_root
os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_root, "datasets_cache")
os.environ["HUGGINGFACE_HUB_CACHE"] = cache_root
os.environ["TRANSFORMERS_CACHE"] = cache_root
os.environ["HUGGINGFACE_CACHE_DIR"] = cache_root

import argparse
import json
import sys
import traceback
import numpy as np

from pipeline.submodules.evaluate_jailbreak import harmbench_judge_fn


def main():
    parser = argparse.ArgumentParser(description="从 jsonl 文件计算 ASR")
    parser.add_argument("--input_file", type=str, required=True, help="输入的 jsonl 文件路径")
    parser.add_argument("--output_file", type=str, required=True, help="输出的 JSON 文件路径")
    args = parser.parse_args()
    
    try:
        # 读取 jsonl 文件
        print(f"[Info] 读取输入文件: {args.input_file}")
        completions = []
        with open(args.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                completions.append(json.loads(line))
        
        print(f"[Info] 成功读取 {len(completions)} 个样本")
        
        # 提取 prompts 和 responses
        prompts_list = [completion["prompt"] for completion in completions]
        responses_list = [completion["response"] for completion in completions]
        
        # 计算 ASR
        print(f"[Info] 开始使用 HarmBench 分类器评估 {len(completions)} 个补全结果...")
        classifications = harmbench_judge_fn(prompts_list, responses_list)
        
        # 计算 ASR
        asr = np.mean(classifications) if len(classifications) > 0 else 0.0
        n_successful = int(sum(classifications))
        n_fail = len(classifications) - n_successful
        
        print(f"[Info] HarmBench 评估完成: {n_successful} 个成功 (success), {n_fail} 个失败 (fail)")
        print(f"[Info] ASR: {asr:.4f} ({n_successful}/{len(completions)})")
        
        # 保存结果
        asr_results = {
            "asr": float(asr),
            "n_samples": len(completions),
            "n_successful": n_successful,
            "n_fail": n_fail,
        }
        
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(asr_results, f, indent=2, ensure_ascii=False)
        
        print(f"[Info] ASR 结果已保存到: {args.output_file}")
        
    except KeyboardInterrupt:
        print("\n[Info] 用户中断程序")
        sys.exit(1)
    except Exception as e:
        print(f"\n[Error] 程序执行出错: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()


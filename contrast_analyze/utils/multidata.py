#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
一次性拉取多个常用攻击/安全相关数据集的 prompts，
每个数据集一个 jsonl，保存到 /root/autodl-tmp/dataset。

每行格式：
    {"prompt": "<string>"}

数据集（当前版本）：
  1. HarmBench 行为 prompts            -> harmbench_prompts.jsonl
  2. JailbreakBench behaviors          -> jbb_behaviors_harmful.jsonl
  3. RealToxicityPrompts               -> real_toxicity_prompts.jsonl
  4. Anthropic HH-RLHF 首轮prompt      -> hh_rlhf_first_turn.jsonl
  5. AdvBench harmful behaviors (CSV)  -> advbench_harmful_behaviors.jsonl

依赖：
    pip install -U datasets pandas requests tqdm
"""

import os
import json
import re
from typing import List, Dict, Any, Optional

from tqdm import tqdm

try:
    from datasets import load_dataset
except Exception as e:
    raise RuntimeError(
        "Missing dependency: datasets. Please run:\n"
        "  pip install -U datasets\n"
    ) from e

try:
    import requests
except Exception as e:
    raise RuntimeError(
        "Missing dependency: requests. Please run:\n"
        "  pip install -U requests\n"
    ) from e

try:
    import pandas as pd
except Exception as e:
    raise RuntimeError(
        "Missing dependency: pandas. Please run:\n"
        "  pip install -U pandas\n"
    ) from e


OUT_DIR = "/root/autodl-tmp/dataset"


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def normalize_prompt(s: str) -> str:
    """轻量清洗：转字符串，合并空白，去首尾。"""
    if s is None:
        return ""
    s = str(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def dump_prompts_to_jsonl(prompts: List[str], out_path: str):
    """去重 + 写 JSONL"""
    seen = set()
    uniq: List[str] = []
    for p in prompts:
        p = normalize_prompt(p)
        if not p:
            continue
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)

    with open(out_path, "w", encoding="utf-8") as f:
        for p in uniq:
            f.write(json.dumps({"prompt": p}, ensure_ascii=False) + "\n")

    print(f"[OK] {os.path.basename(out_path)}  prompts={len(uniq)}")


# =========================================================
#  1. HarmBench 行为 prompts (HF: walledai/HarmBench)
# =========================================================

def extract_harmbench_prompts(limit: Optional[int] = None) -> List[str]:
    """
    根据 HF 卡片：
      https://huggingface.co/datasets/walledai/HarmBench
    这个数据集只包含 HarmBench 的行为 prompts。

    常见字段：
      - 'prompt' 或 'behavior' 或 'text'
    这里做一点小鲁棒：优先找 'prompt'、'behavior'，否则找第一个字符串字段。
    
    注意：该数据集需要指定配置名称：'contextual', 'copyright', 或 'standard'
    """
    # 尝试不同的配置
    configs = ['standard', 'contextual', 'copyright']
    
    for config in configs:
        try:
            ds = load_dataset("walledai/HarmBench", config)
            # 该数据集通常只有一个 split，取第一个
            if isinstance(ds, dict):
                d = next(iter(ds.values()))
            else:
                d = ds

            if len(d) == 0:
                continue

            example: Dict[str, Any] = d[0]

            key = None
            for cand in ["prompt", "behavior", "text"]:
                if cand in example and isinstance(example[cand], str):
                    key = cand
                    break

            if key is None:
                # 兜底：找第一个 string 字段
                for k, v in example.items():
                    if isinstance(v, str):
                        key = k
                        break

            if key is None:
                # 再兜底：整个 example 转成字符串
                prompts = [str(x) for x in d]
            else:
                prompts = [x.get(key, "") for x in d]

            if limit is not None:
                prompts = prompts[:limit]

            return prompts
        except Exception:
            continue
    
    print("[WARN] Failed to load HarmBench with known configs: ['standard', 'contextual', 'copyright']")
    return []


# =========================================================
#  2. JailbreakBench behaviors (HF: JailbreakBench/JBB-Behaviors)
# =========================================================

def extract_jbb_behaviors(limit: Optional[int] = None) -> List[str]:
    """
    JailbreakBench 行为集合：
      https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors

    常见字段：
      - split: 'train' 或 'harmful'
      - 列：'behavior'（代表 harmful 行为的自然语言描述）
    """
    # 优先尝试有 harmful split 的配置
    try_order = [
        ("JailbreakBench/JBB-Behaviors", None, "harmful"),
        ("JailbreakBench/JBB-Behaviors", "behaviors", "harmful"),
        ("JailbreakBench/JBB-Behaviors", None, "train"),
        ("JailbreakBench/JBB-Behaviors", "behaviors", "train"),
    ]

    for repo, subset, split in try_order:
        try:
            if subset:
                ds = load_dataset(repo, subset, split=split)
            else:
                ds = load_dataset(repo, split=split)

            if len(ds) == 0:
                continue

            ex = ds[0]
            key = None
            for cand in ["behavior", "prompt", "text", "goal"]:
                if cand in ex and isinstance(ex[cand], str):
                    key = cand
                    break

            if key is None:
                for k, v in ex.items():
                    if isinstance(v, str):
                        key = k
                        break

            if key is None:
                prompts = [str(x) for x in ds]
            else:
                prompts = [x.get(key, "") for x in ds]

            if limit is not None:
                prompts = prompts[:limit]
            return prompts
        except Exception:
            continue

    print("[WARN] Failed to load JailbreakBench/JBB-Behaviors with known configs.")
    return []


# =========================================================
#  3. RealToxicityPrompts (HF: allenai/real-toxicity-prompts)
# =========================================================

def extract_real_toxicity_prompts(limit: Optional[int] = None) -> List[str]:
    """
    RealToxicityPrompts 的数据结构类似：
      {
        "prompt": {
            "text": "...真实文本...",
            ... 其它字段 ...
        },
        "filename": "0766186-xxxx.txt",
        ...
      }

    我们需要的是 prompt['text']，而不是 filename。
    """
    ds = load_dataset("allenai/real-toxicity-prompts", split="train")

    prompts: List[str] = []
    for ex in ds:
        try:
            text = ex["prompt"]["text"]
        except Exception:
            # 容错：有些行可能缺字段，直接跳过
            continue
        text = normalize_prompt(text)
        if text:
            prompts.append(text)

    if limit is not None:
        prompts = prompts[:limit]

    return prompts


# =========================================================
#  4. Anthropic HH-RLHF 首轮人类 prompt (HF: HuggingFaceH4/hh-rlhf)
# =========================================================

def extract_hh_rlhf_first_turn(limit: Optional[int] = None) -> List[str]:
    """
    HH-RLHF:
      https://huggingface.co/datasets/HuggingFaceH4/hh-rlhf

    经典格式大致为：
      {"prompt": "<human prompt>", "chosen": "...", "rejected": "..."}

    这里我们直接使用 'prompt' 作为输入给模型的文本。
    """
    ds = load_dataset("HuggingFaceH4/hh-rlhf", split="train")

    prompts: List[str] = []
    for ex in ds:
        text = ex.get("prompt", "")
        text = normalize_prompt(text)
        if text:
            prompts.append(text)

    if limit is not None:
        prompts = prompts[:limit]

    return prompts


# =========================================================
#  5. AdvBench harmful behaviors (GitHub CSV from llm-attacks)
# =========================================================

def extract_advbench_harmful_behaviors(
    limit: Optional[int] = None
) -> List[str]:
    """
    从 llm-attacks 仓库中拉取 AdvBench 的 harmful behaviors：

      https://github.com/llm-attacks/llm-attacks/blob/main/data/advbench/harmful_behaviors.csv

    CSV 常见列：
      - goal: 行为描述（我们最关心）
      - target: 期望 harmful 输出描述
    """
    url = (
        "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/"
        "data/advbench/harmful_behaviors.csv"
    )
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    content = resp.content.decode("utf-8", errors="ignore")
    from io import StringIO
    df = pd.read_csv(StringIO(content))

    # 优先用 goal，没有就退而求其次
    col = None
    for cand in ["goal", "behavior", "prompt", "instruction", "text"]:
        if cand in df.columns:
            col = cand
            break

    if col is None:
        # 用第一列兜底
        col = df.columns[0]

    prompts = df[col].astype(str).tolist()
    prompts = [normalize_prompt(x) for x in prompts if normalize_prompt(x)]

    if limit is not None:
        prompts = prompts[:limit]

    return prompts


# =========================================================
#                       MAIN
# =========================================================

def main():
    ensure_dir(OUT_DIR)

    # 1) HarmBench prompts
    try:
        hb_prompts = extract_harmbench_prompts(limit=None)
        dump_prompts_to_jsonl(hb_prompts, os.path.join(OUT_DIR, "harmbench_prompts.jsonl"))
    except Exception as e:
        print(f"[WARN] HarmBench extract failed: {e}")

    # 2) JailbreakBench behaviors
    try:
        jbb_prompts = extract_jbb_behaviors(limit=None)
        dump_prompts_to_jsonl(jbb_prompts, os.path.join(OUT_DIR, "jbb_behaviors_harmful.jsonl"))
    except Exception as e:
        print(f"[WARN] JBB-Behaviors extract failed: {e}")

    # 3) RealToxicityPrompts
    try:
        rtox_prompts = extract_real_toxicity_prompts(limit=None)
        dump_prompts_to_jsonl(rtox_prompts, os.path.join(OUT_DIR, "real_toxicity_prompts.jsonl"))
    except Exception as e:
        print(f"[WARN] RealToxicityPrompts extract failed: {e}")

    # 4) Anthropic HH-RLHF first turn
    try:
        hh_prompts = extract_hh_rlhf_first_turn(limit=None)
        dump_prompts_to_jsonl(hh_prompts, os.path.join(OUT_DIR, "hh_rlhf_first_turn.jsonl"))
    except Exception as e:
        print(f"[WARN] HH-RLHF extract failed: {e}")

    # 5) AdvBench harmful behaviors
    try:
        adv_prompts = extract_advbench_harmful_behaviors(limit=None)
        dump_prompts_to_jsonl(adv_prompts, os.path.join(OUT_DIR, "advbench_harmful_behaviors.jsonl"))
    except Exception as e:
        print(f"[WARN] AdvBench harmful_behaviors extract failed: {e}")

    print("\nAll done. Output dir:", OUT_DIR)


if __name__ == "__main__":
    main()

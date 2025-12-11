"""
pipeline_core.py

集中放置流水线通用的基础设施，避免在各阶段脚本重复粘贴。
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch


HF_CACHE_DEFAULT = "/root/autodl-tmp/hf_cache"
HF_CACHE_DIRS = ["datasets_cache"]
NO_PROXY_LIST = "all://huggingface.co,all://cdn-lfs.huggingface.co,all://localhost,all://127.0.0.1"


def configure_hf_cache(cache_root: Optional[str | Path] = None) -> Path:
    """
    统一配置 HuggingFace 缓存路径与代理白名单。
    """
    root = Path(cache_root or os.environ.get("HF_CACHE_ROOT", HF_CACHE_DEFAULT))
    root.mkdir(parents=True, exist_ok=True)
    for name in HF_CACHE_DIRS:
        (root / name).mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(root)
    os.environ["HF_DATASETS_CACHE"] = str(root / "datasets_cache")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(root)
    os.environ["TRANSFORMERS_CACHE"] = str(root)
    os.environ["HUGGINGFACE_CACHE_DIR"] = str(root)

    os.environ["NO_PROXY"] = NO_PROXY_LIST
    os.environ["no_proxy"] = NO_PROXY_LIST
    return root


def seed_everything(seed: int = 2025) -> None:
    """
    统一随机种子，确保可复现。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(name: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    配置基础日志格式，返回 logger。
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    return logging.getLogger(name)


def log_section(title: str) -> None:
    """打印统一的分隔标题。"""
    line = "=" * 60
    print(line)
    print(title)
    print(line)


def setup_environment(seed: int = 2025, cache_root: Optional[str | Path] = None, log_name: Optional[str] = None) -> logging.Logger:
    """
    入口级别的统一环境配置：HF 缓存、随机种子、日志。
    返回 logger 以便调用方直接使用。
    """
    configure_hf_cache(cache_root)
    seed_everything(seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    logger = setup_logging(log_name)
    return logger


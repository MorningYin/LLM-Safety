# -*- coding: utf-8 -*-
import json
from pathlib import Path
from typing import Optional

# EOI_MAP 已移除，统一使用 model_utils 中定义的接口
# 如需获取 marker，请使用 resolve_eoi_marker(model_base)


CACHE_ROOT = Path("/root/autodl-tmp/dataset/cached_activations")


def cache_dir_name(model_alias: str, split: str, harmtype: str, data_type: Optional[str] = None) -> str:
    if harmtype == "benign":
        return f"{model_alias}_{split}_benign_EOI"
    if data_type not in {"success", "failure"}:
        raise ValueError("data_type 必须是 success 或 failure")
    return f"{model_alias}_{split}_{harmtype}_{data_type}_EOI"


def cache_dir_path(model_alias: str, split: str, harmtype: str, data_type: Optional[str] = None) -> Path:
    return CACHE_ROOT / cache_dir_name(model_alias, split, harmtype, data_type)


def load_meta(cache_dir: Path) -> dict:
    meta_path = Path(cache_dir) / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"未找到 meta.json：{meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_token_labels(tokenizer, marker: str, expected_len: int):
    tokens = tokenizer.encode(marker, add_special_tokens=False)
    labels = tokenizer.convert_ids_to_tokens(tokens)
    if len(labels) != expected_len:
        return [f"pos_{i}" for i in range(expected_len)]
    return labels

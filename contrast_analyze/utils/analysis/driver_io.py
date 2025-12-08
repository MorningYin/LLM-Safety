"""IO 与数据加载工具：投影读取、组件汇总等。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import torch


def load_projection_from_storage(storage_root: str, layer_idx: int, module_type: str, hook_name: str) -> torch.Tensor:
    """从存储路径加载投影数据。"""
    storage_root_path = Path(storage_root)
    metadata_path = storage_root_path / "metadata.json"

    items = []

    # 优先 metadata 索引
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        if "index" in metadata and metadata["index"]:
            index_data = metadata["index"]
        elif metadata:
            index_data = metadata
        else:
            index_data = {}

        layer_key = f"layer_{layer_idx}"
        if layer_key in index_data:
            module_data = index_data[layer_key].get(module_type, {})
            hook_data = module_data.get(hook_name, {})
            data_type_data = hook_data.get("default", {})
            if isinstance(data_type_data, dict) and "files" in data_type_data:
                items = data_type_data["files"]
            elif isinstance(data_type_data, list):
                items = data_type_data

    # 回落到文件系统
    if not items:
        data_dir = storage_root_path / f"layer_{layer_idx}" / module_type / hook_name / "default"
        if data_dir.exists():
            items = sorted(data_dir.glob("*.pt"))

    if not items:
        return torch.tensor([])

    data_list = []
    for item in items:
        file_path = Path(item) if Path(item).is_absolute() else storage_root_path / item
        if file_path.exists():
            try:
                data_list.append(torch.load(file_path, map_location="cpu"))
            except Exception:
                continue
        elif isinstance(item, torch.Tensor):
            data_list.append(item)

    if data_list and all(isinstance(x, torch.Tensor) for x in data_list):
        return torch.cat(data_list, dim=0)
    return torch.tensor([])


def load_component_mean_projections(
    layer_idx: int,
    token_idx: int,
    component_name: str,
    storage_root_benign: str,
    storage_root_success: str,
    storage_root_fail: str,
) -> Tuple[float, float, float]:
    """加载指定组件在三个数据集上的平均投影值。"""
    if component_name == "MLP":
        mlp_benign = load_projection_from_storage(storage_root_benign, layer_idx, "mlp", "mlp_proj")
        mlp_success = load_projection_from_storage(storage_root_success, layer_idx, "mlp", "mlp_proj")
        mlp_fail = load_projection_from_storage(storage_root_fail, layer_idx, "mlp", "mlp_proj")
        if len(mlp_benign) == 0 or len(mlp_success) == 0 or len(mlp_fail) == 0:
            return 0.0, 0.0, 0.0
        mean_benign = mlp_benign[:, token_idx].float().mean().item()
        mean_success = mlp_success[:, token_idx].float().mean().item()
        mean_fail = mlp_fail[:, token_idx].float().mean().item()
        return mean_benign, mean_success, mean_fail

    attn_benign = load_projection_from_storage(storage_root_benign, layer_idx, "o_proj", "z_capture")
    attn_success = load_projection_from_storage(storage_root_success, layer_idx, "o_proj", "z_capture")
    attn_fail = load_projection_from_storage(storage_root_fail, layer_idx, "o_proj", "z_capture")
    if len(attn_benign) == 0 or len(attn_success) == 0 or len(attn_fail) == 0:
        return 0.0, 0.0, 0.0
    try:
        head_idx = int(component_name[1:])
    except Exception:
        return 0.0, 0.0, 0.0
    mean_benign = attn_benign[:, token_idx, head_idx].float().mean().item()
    mean_success = attn_success[:, token_idx, head_idx].float().mean().item()
    mean_fail = attn_fail[:, token_idx, head_idx].float().mean().item()
    return mean_benign, mean_success, mean_fail


def load_component_projections(
    layer_idx: int,
    token_idx: int,
    component_name: str,
    storage_root_benign: str,
    storage_root_success: str,
    storage_root_fail: str,
):
    """加载指定组件在三个数据集上的所有样本投影值。"""
    if component_name == "MLP":
        mlp_benign = load_projection_from_storage(storage_root_benign, layer_idx, "mlp", "mlp_proj")
        mlp_success = load_projection_from_storage(storage_root_success, layer_idx, "mlp", "mlp_proj")
        mlp_fail = load_projection_from_storage(storage_root_fail, layer_idx, "mlp", "mlp_proj")
        if len(mlp_benign) == 0 or len(mlp_success) == 0 or len(mlp_fail) == 0:
            return (), (), ()
        return (
            mlp_benign[:, token_idx].float().numpy(),
            mlp_success[:, token_idx].float().numpy(),
            mlp_fail[:, token_idx].float().numpy(),
        )

    attn_benign = load_projection_from_storage(storage_root_benign, layer_idx, "o_proj", "z_capture")
    attn_success = load_projection_from_storage(storage_root_success, layer_idx, "o_proj", "z_capture")
    attn_fail = load_projection_from_storage(storage_root_fail, layer_idx, "o_proj", "z_capture")
    if len(attn_benign) == 0 or len(attn_success) == 0 or len(attn_fail) == 0:
        return (), (), ()
    try:
        head_idx = int(component_name[1:])
    except Exception:
        return (), (), ()
    return (
        attn_benign[:, token_idx, head_idx].float().numpy(),
        attn_success[:, token_idx, head_idx].float().numpy(),
        attn_fail[:, token_idx, head_idx].float().numpy(),
    )


def load_data(config) -> Tuple[str, str, str]:
    """加载 Benign、Success 和 Failure 的数据存储路径。"""
    base_dir = config.projection_analysis_dir
    benign_path = base_dir / "benign"
    success_path = base_dir / "harmful_success"
    fail_path = base_dir / "harmful_failure"
    if not benign_path.exists():
        raise FileNotFoundError(f"Benign data not found at {benign_path}")
    if not success_path.exists():
        raise FileNotFoundError(f"Success data not found at {success_path}")
    if not fail_path.exists():
        raise FileNotFoundError(f"Failure data not found at {fail_path}")
    print(f"Loading Benign from: {benign_path}")
    print(f"Loading Success from: {success_path}")
    print(f"Loading Failure from: {fail_path}")
    return str(benign_path), str(success_path), str(fail_path)


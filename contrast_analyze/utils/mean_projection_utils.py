# -*- coding: utf-8 -*-
"""
mean_projection_utils.py

投影分析工具模块：用于计算模型激活值在给定方向向量上的投影。
支持从缓存或实时计算平均投影，以及详细的投影数据存储和加载。
"""

import os
import json
from datetime import datetime
import numpy as np
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from collections import defaultdict

from contrast_analyze.utils.activation_utils import ActivationDataset, find_substring_token_range
from contrast_analyze.utils.data_utils import load_meta
from contrast_analyze.utils.model_hook_runner import ModelHookRunner, StorageManager

# 调试标志：通过环境变量控制
DEBUG_MEAN_PROJECTION = os.getenv("DEBUG_MEAN_PROJECTION", "0") == "1"
DEBUG_MEAN_PROJECTION_TRACE = os.getenv("DEBUG_MEAN_PROJECTION_TRACE", "0") == "1"

# 全局日志文件句柄
_trace_log_file = None
_trace_log_path = None


class TraceLogger:
    """结构化调试日志记录器"""
    
    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path
        self.log_file = None
        if log_path:
            self.log_file = open(log_path, 'w', encoding='utf-8')
            self.log_file.write(f"=== Trace Debug Log Started at {datetime.now().isoformat()} ===\n\n")
    
    def log(self, stage: str, message: str, data: Optional[Dict] = None):
        """记录调试信息"""
        if not DEBUG_MEAN_PROJECTION_TRACE:
            return
        
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_entry = f"[{timestamp}] [{stage}] {message}"
        
        if data:
            log_entry += "\n" + json.dumps(data, indent=2, ensure_ascii=False, default=str)
        
        log_entry += "\n"
        
        if self.log_file:
            self.log_file.write(log_entry)
            self.log_file.flush()
        
        # 同时输出到控制台（如果 DEBUG_MEAN_PROJECTION 也启用）
        if DEBUG_MEAN_PROJECTION:
            print(log_entry.rstrip())
    
    def close(self):
        """关闭日志文件"""
        if self.log_file:
            self.log_file.write(f"\n=== Trace Debug Log Ended at {datetime.now().isoformat()} ===\n")
            self.log_file.close()
            self.log_file = None


def _get_trace_logger(output_dir: Optional[Path] = None) -> Optional[TraceLogger]:
    """获取或创建全局日志记录器"""
    global _trace_log_file, _trace_log_path
    
    if not DEBUG_MEAN_PROJECTION_TRACE:
        return None
    
    if _trace_log_file is None and output_dir:
        _trace_log_path = output_dir / "debug_trace.log"
        _trace_log_file = TraceLogger(_trace_log_path)
        print(f"[INFO] Trace debug log will be saved to: {_trace_log_path}")
    
    return _trace_log_file

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x  # 如果 tqdm 不可用，直接返回原迭代器


@dataclass
class MeanProjectionResult:
    """
    平均投影结果数据类。
    
    用于存储计算得到的平均投影值及其元数据。
    
    Attributes:
        mean: 平均投影值数组，形状为 (n_layers, K_max) 或 (n_layers, n_heads, K_max)
        n_samples: 参与计算的样本数量
        sample_meta: 可选的样本元数据，包含 meta_ids 和 dataset_indices
    """
    mean: np.ndarray
    n_samples: int
    sample_meta: Optional[Dict[str, List[int]]] = None


def _standardize_direction(direction: torch.Tensor, n_layers: int, K_max: int, logger: Optional[TraceLogger] = None) -> np.ndarray:
    """
    标准化方向向量，将其转换为统一的形状 (K_max, n_layers, d_model)。
    
    支持多种输入格式：
    - 1D: (d_model,) -> 广播到所有层和所有 token 位置
    - 2D: (n_layers, d_model) 或 (K_max, d_model) -> 广播到缺失维度
    - 3D: (K_max, n_layers, d_model) 或 (n_layers, K_max, d_model) -> 转置为统一格式
    
    Args:
        direction: 输入方向向量，可以是多种形状
        n_layers: 模型层数
        K_max: EOI token 的最大数量（marker token 长度）
        logger: 可选的日志记录器
    
    Returns:
        标准化后的方向向量，形状为 (K_max, n_layers, d_model)
    """
    if logger:
        logger.log("DIRECTIONS_INIT", "Standardizing direction tensor", {
            "input_shape": list(direction.shape),
            "input_dtype": str(direction.dtype),
            "n_layers": n_layers,
            "K_max": K_max,
            "input_stats": {
                "mean": float(direction.mean().item()),
                "std": float(direction.std().item()),
                "min": float(direction.min().item()),
                "max": float(direction.max().item()),
            }
        })
    
    # 先将 BFloat16 或其他不支持的类型转换为 float32，再转换为 NumPy
    arr = direction.detach().cpu().float().numpy()

    if arr.ndim == 1:
        # 1D: 单个方向向量，广播到所有层和所有 token 位置
        arr = np.broadcast_to(arr, (K_max, n_layers, arr.shape[0]))
    elif arr.ndim == 2:
        # 2D: 可能是 (n_layers, d_model) 或 (K_max, d_model)
        if arr.shape[0] == n_layers:
            # 每层一个方向，广播到所有 token 位置
            arr = np.broadcast_to(arr[np.newaxis, :, :], (K_max, n_layers, arr.shape[1]))
        elif arr.shape[0] == K_max:
            # 每个 token 位置一个方向，广播到所有层
            arr = np.broadcast_to(arr[:, np.newaxis, :], (K_max, n_layers, arr.shape[1]))
        else:
            raise ValueError(f"无法广播方向向量，shape={arr.shape}，期望第一维等于 K_max 或 n_layers")
    elif arr.ndim == 3:
        # 3D: 可能是 (K_max, n_layers, d_model) 或 (n_layers, K_max, d_model)
        if arr.shape[0] == K_max and arr.shape[1] == n_layers:
            # 已经是正确格式
            pass
        elif arr.shape[0] == n_layers and arr.shape[1] == K_max:
            # 需要转置：交换前两个维度
            arr = np.transpose(arr, (1, 0, 2))
        elif arr.shape[0] == K_max:
            # 如果第一维是 K_max，但第二维不是 n_layers，可能是 (K_max, n_layers, d_model) 但 n_layers 不匹配
            # 这种情况下，我们假设形状是 (K_max, n_layers, d_model)，直接使用
            pass
        else:
            raise ValueError(
                f"方向张量形状不匹配，得到 {arr.shape}，需要 (K_max={K_max}, n_layers={n_layers}, d_model) 或可转换形状"
            )
    else:
        raise ValueError(f"不支持的方向张量维度：{arr.ndim}")

    # 确保结果是可写的连续数组，避免 broadcast_to 产生的只读视图
    # 强制复制以确保数组是可写的和连续的
    result = np.ascontiguousarray(arr, dtype=np.float32)
    
    if logger:
        logger.log("DIRECTIONS_RESULT", "Direction standardization completed", {
            "output_shape": list(result.shape),
            "output_dtype": str(result.dtype),
            "expected_shape": [K_max, n_layers, result.shape[2]],
            "output_stats": {
                "mean": float(result.mean()),
                "std": float(result.std()),
                "min": float(result.min()),
                "max": float(result.max()),
            },
            "sample_values": {
                f"K0_L0": result[0, 0, :10].tolist() if result.shape[2] >= 10 else result[0, 0, :].tolist(),
            }
        })
    
    return result


def _mean_from_cache(
    cache_dir: Path,
    direction: torch.Tensor,
    batch_size: int,
    meta: Optional[dict] = None,
    desc: Optional[str] = None,
) -> MeanProjectionResult:
    """
    从缓存的激活值数据计算平均投影。
    
    从磁盘加载已保存的激活值，计算它们在给定方向上的投影，并返回平均值。
    适用于已经通过 activation_cache 模块保存的数据。
    
    Args:
        cache_dir: 缓存数据目录路径
        direction: 方向向量
        batch_size: 批处理大小
        meta: 可选的元数据字典，如果为 None 则从缓存目录加载
        desc: 进度条描述文本
    
    Returns:
        MeanProjectionResult: 包含平均投影值的结果对象
    """
    if meta is None:
        meta = load_meta(cache_dir)

    # 标准化方向向量为 (K_max, n_layers, d_model)
    logger = _get_trace_logger(cache_dir)
    directions = _standardize_direction(direction, meta["n_layers"], meta["K_max"], logger=logger)

    # 加载缓存的激活值数据集
    dataset = ActivationDataset(
        folder=str(cache_dir),
        n_layers=meta["n_layers"],
        K_max=meta["K_max"],
        d_model=meta["d_model"],
        seq_len=meta["seq_len"],
        n_samples=meta["n_samples"],
    )

    indices = np.arange(meta["n_samples"])
    # 累加器：用于累加所有样本的投影值
    acc = np.zeros((meta["n_layers"], meta["K_max"]), dtype=np.float64)

    n_batches = (meta["n_samples"] + batch_size - 1) // batch_size
    for start in tqdm(range(0, meta["n_samples"], batch_size), desc=desc or "处理缓存数据", total=n_batches):
        end = min(start + batch_size, meta["n_samples"])
        idx_batch = indices[start:end]
        # 加载当前批次的激活值：形状 (batch, n_layers, K_max, d_model)
        acts = np.asarray(dataset.acts[idx_batch], dtype=np.float32)  # (b, L, K, D)
        dirs = np.asarray(directions, dtype=np.float32)
        # 计算投影：对每个样本、每层、每个 token 位置计算与方向向量的点积
        # einsum: blpd (batch, layer, pos, dim) 与 pld (pos, layer, dim) 的点积
        dots = np.einsum("blpd,pld->blp", acts, dirs, optimize=True)
        # 累加：对 batch 维度求和，得到 (layer, pos) 的累加值
        acc += dots.sum(axis=0)

    # 计算平均值
    mean = np.asarray(acc / meta["n_samples"], dtype=np.float32)
    
    # 关闭日志文件（如果使用缓存模式）
    if logger:
        logger.close()
        global _trace_log_file, _trace_log_path
        _trace_log_file = None
        _trace_log_path = None
    
    return MeanProjectionResult(mean=mean, n_samples=int(meta["n_samples"]), sample_meta=None)


def compute_mean_direction_dot(
    *,
    direction: torch.Tensor,
    batch_size: int,
    cache_dir: Optional[Path] = None,
    meta: Optional[dict] = None,
    model_base=None,
    dataset: Optional[List[Dict]] = None,
    marker: Optional[str] = None,
    instruction_key: str = "instruction",
    meta_id_key: str = "meta_id",
    custom_hook_builder: Optional[
        Callable[[Dict[int, torch.Tensor]], Tuple[List[Tuple[torch.nn.Module, Callable]], List[Tuple[torch.nn.Module, Callable]]]]
    ] = None,
    desc: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> MeanProjectionResult:
    """
    计算模型激活值在给定方向上的平均投影。
    
    支持两种模式：
    1. 从缓存加载：如果提供了 cache_dir，则从磁盘加载已保存的激活值
    2. 实时计算：如果提供了 model_base 和 dataset，则实时运行模型并计算
    
    使用 Hook 机制捕获每层 Transformer Block 的输出激活值，提取 EOI (End of Instruction)
    token 位置的激活，计算与方向向量的点积，最后对所有样本求平均。
    
    Args:
        direction: 方向向量，用于计算投影
        batch_size: 批处理大小
        cache_dir: 可选的缓存目录路径（如果提供，则从缓存加载）
        meta: 可选的元数据字典
        model_base: 模型基础对象（实时计算模式必需）
        dataset: 数据集列表（实时计算模式必需）
        marker: EOI marker 字符串，用于定位 token 位置（实时计算模式必需）
        instruction_key: 数据记录中指令字段的键名
        meta_id_key: 数据记录中元数据 ID 字段的键名
        custom_hook_builder: 可选的自定义 Hook 构建器，用于干预实验
        desc: 进度条描述文本
    
    Returns:
        MeanProjectionResult: 包含平均投影值的结果对象，形状为 (n_layers, K_max)
    """
    # 模式1：从缓存加载数据
    if cache_dir is not None and dataset is None and model_base is None:
        if not cache_dir.exists():
            raise FileNotFoundError(f"缓存目录不存在：{cache_dir}")
        return _mean_from_cache(cache_dir, direction, batch_size, meta, desc=desc)

    # 模式2：实时计算（需要模型和数据集）
    if model_base is None or dataset is None or marker is None:
        raise ValueError("运行时计算平均激活需要提供 model_base、dataset 和 marker。")

    # 初始化日志记录器
    logger = _get_trace_logger(output_dir)
    if logger:
        logger.log("COMPUTE_START", "Starting mean direction dot computation", {
            "has_custom_hook": custom_hook_builder is not None,
            "dataset_size": len(dataset) if dataset else 0,
            "batch_size": batch_size,
            "marker": marker,
        })

    # 获取模型配置和工具函数
    tokenizer = model_base.tokenizer
    tok_fn = model_base.tokenize_instructions_fn
    device = model_base.model.device
    n_layers = len(model_base.model_block_modules)
    d_model = model_base.model.config.hidden_size
    marker_tokens = tokenizer.encode(marker, add_special_tokens=False)
    K_max = len(marker_tokens)  # EOI token 的数量

    if logger:
        logger.log("MODEL_CONFIG", "Model configuration", {
            "n_layers": n_layers,
            "d_model": d_model,
            "K_max": K_max,
            "marker_tokens": marker_tokens,
        })

    # 标准化方向向量
    directions = _standardize_direction(direction, n_layers, K_max, logger=logger)

    # 初始化累加器和元数据列表
    acc = np.zeros((n_layers, K_max), dtype=np.float64)
    meta_ids_all: List[int] = []
    dataset_indices_all: List[int] = []
    total = 0
    
    if logger:
        logger.log("ACCUMULATOR_INIT", "Initialized accumulator", {
            "acc_shape": list(acc.shape),
            "acc_dtype": str(acc.dtype),
            "initial_value": float(acc.sum()),
        })

    def get_instruction(record: Dict, instr_key: str = instruction_key) -> str:
        """从数据记录中提取指令文本，支持多种字段名。"""
        if instr_key in record and record[instr_key]:
            return record[instr_key]
        if instr_key != "instruction" and "instruction" in record and record["instruction"]:
            return record["instruction"]
        if "prompt" in record and record["prompt"]:
            return record["prompt"]
        raise KeyError("record 中缺少 instruction/prompt 字段，无法生成输入。")

    # 初始化 Hook Runner（内存模式）
    runner = ModelHookRunner(model_base, storage_mode='memory')

    n_batches = (len(dataset) + batch_size - 1) // batch_size
    for start in tqdm(range(0, len(dataset), batch_size), desc=desc or "处理数据集", total=n_batches):
        batch_records = dataset[start : start + batch_size]
        batch_prompts = [get_instruction(record) for record in batch_records]
        enc = tok_fn(instructions=batch_prompts)
        input_ids = enc.input_ids.to(device)
        attn_mask = enc.attention_mask.to(device)

        # 为当前批次构建 EOI token 位置映射
        positions_map: Dict[int, torch.Tensor] = {}
        meta_ids: List[int] = []
        dataset_indices: List[int] = []

        for local_idx in range(len(batch_records)):
            tokens = enc.input_ids[local_idx]
            try:
                # 查找 marker 在 token 序列中的位置
                sl = find_substring_token_range(marker, tokens, tokenizer)
            except AssertionError:
                # 如果找不到 marker，跳过该样本
                continue
            # 确保找到的 token 数量与期望的 K_max 一致
            if (sl.stop - sl.start) != K_max:
                continue
            # 保存该样本的 EOI token 位置索引
            positions_map[local_idx] = torch.arange(sl.start, sl.stop, device=device, dtype=torch.long)
            meta_ids.append(batch_records[local_idx].get(meta_id_key, start + local_idx))
            dataset_indices.append(start + local_idx)

        # 如果当前批次没有有效样本，跳过
        if not positions_map:
            if DEBUG_MEAN_PROJECTION:
                print(f"[DEBUG] Batch {start//batch_size + 1}: No valid samples, skipping")
            continue

        valid_indices = list(positions_map.keys())
        
        # 调试：记录 positions_map 信息
        if DEBUG_MEAN_PROJECTION:
            print(f"[DEBUG] Batch {start//batch_size + 1}: positions_map size={len(positions_map)}, valid_indices={valid_indices}")
            for batch_idx, pos_tensor in positions_map.items():
                print(f"  batch_idx={batch_idx}, pos_tensor.shape={pos_tensor.shape}, pos_tensor={pos_tensor.tolist()}")
        
        # 构建 Record Hook：捕获指定位置的激活值
        # 使用闭包捕获 positions_map 和 valid_indices，使得 hook 函数可以访问这些变量
        def make_record_hook(valid_indices_list, positions_map_dict):
            """
            创建记录 Hook 函数。
            
            该 Hook 会在模型前向传播时被调用，提取每个有效样本在 EOI token 位置的激活值。
            """
            def record_hook(ctx, inp, out):
                # out: (batch, seq, dim) - Block 的输出激活值（应该是干预后的值）
                # positions_map_dict: {batch_idx: tensor([pos1, pos2, ...])} - EOI token 位置
                layer_idx = ctx.layer_idx
                
                # 调试：检查 out 是否被干预修改
                if DEBUG_MEAN_PROJECTION and layer_idx == 14:  # 检查第14层（有干预的层）
                    # 检查第一个样本的第一个 token 位置的值
                    if valid_indices_list and len(positions_map_dict) > 0:
                        first_idx = valid_indices_list[0]
                        if first_idx in positions_map_dict:
                            first_pos = positions_map_dict[first_idx][0].item()
                            sample_value = out[first_idx, first_pos, :10].mean().item()  # 取前10维的平均值作为示例
                            print(f"[DEBUG] Record hook L{layer_idx} batch{first_idx} pos{first_pos}: sample_value={sample_value:.6f}")
                
                # 提取每个有效样本在指定位置的激活值
                extracted = []
                for local_idx in valid_indices_list:
                    pos = positions_map_dict[local_idx]  # (K_max,) - 该样本的 EOI token 位置
                    # 提取该样本在这些位置的激活值：形状 (K_max, dim)
                    extracted.append(out[local_idx, pos, :].detach().cpu())
                
                if extracted:
                    # 堆叠所有有效样本：形状 (valid_batch, K_max, dim)
                    saved = torch.stack(extracted)
                    
                    # 调试：记录数据统计信息
                    if DEBUG_MEAN_PROJECTION and layer_idx == 0:  # 只记录第一层，避免输出过多
                        print(f"[DEBUG] Record hook L{layer_idx}: extracted shape={saved.shape}, "
                              f"mean={saved.mean().item():.6f}, std={saved.std().item():.6f}, "
                              f"min={saved.min().item():.6f}, max={saved.max().item():.6f}, "
                              f"all_zero={(saved == 0).all().item()}")
                    
                    ctx.save(saved)
            return record_hook

        # 重要：先注册干预 Hooks，再注册 Record Hooks
        # 这样执行顺序是：intervention hooks 先执行（应用干预），record hooks 后执行（记录干预后的值）
        
        # 构建干预 Hooks（先注册）
        if custom_hook_builder:
            # custom_hook_builder 返回 List[Tuple[int, Callable]]
            # Callable 签名: (ctx, inp, out) -> modified_out | None
            intervention_hooks = custom_hook_builder(positions_map)
            if DEBUG_MEAN_PROJECTION:
                print(f"[DEBUG] Batch {start//batch_size + 1}: custom_hook_builder returned {len(intervention_hooks)} intervention hooks")
            for layer_idx, hook_fn in intervention_hooks:
                runner.register_hook('block', hook_fn=hook_fn, layers=layer_idx, name='intervention')
                if DEBUG_MEAN_PROJECTION:
                    print(f"[DEBUG] Batch {start//batch_size + 1}: Registered intervention hook for layer {layer_idx}")
        
        # 注册 Record Hooks（后注册，这样会先执行，但我们需要它后执行）
        # 为每一层注册一个闭包
        record_hook_fn = make_record_hook(valid_indices, positions_map)
        for layer_idx in range(n_layers):
            runner.register_hook(
                'block', 
                hook_fn=record_hook_fn,
                layers=layer_idx,
                name='recorder'
            )
        
        if DEBUG_MEAN_PROJECTION:
            print(f"[DEBUG] Batch {start//batch_size + 1}: Registered {n_layers} record hooks")
            print(f"[DEBUG] Batch {start//batch_size + 1}: Hook execution order (in ModelHookRunner): intervention hooks first, then record hooks")

        with runner:
            model_base.model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                use_cache=False,
                output_hidden_states=False,
            )

        # 刷新缓冲区数据到存储
        # 重要：数据通过 ctx.save() 保存到 _batch_buffer，需要调用 increment_batch() 才能刷新到 _index
        runner.increment_batch()

        # 处理收集到的数据
        # layer -> block -> recorder -> default -> [ (valid_batch, K_max, dim) ]
        
        batch_num = start // batch_size + 1
        if logger:
            logger.log("BATCH_START", f"Processing batch {batch_num}", {
                "batch_start": start,
                "batch_size": len(batch_records),
                "valid_samples": len(valid_indices),
            })
        
        for layer_idx in range(n_layers):
            # 使用 get() 方法获取当前批次的数据
            # 在 disk 模式下，需要从索引中获取最后一个批次的文件
            if runner.storage.mode == 'disk':
                items = runner.storage._index.get(layer_idx, {}).get('block', {}).get('recorder', {}).get('default', [])
                if not items:
                    if DEBUG_MEAN_PROJECTION and layer_idx == 0:
                        print(f"[DEBUG] Batch {batch_num} L{layer_idx}: No data in storage")
                    if logger:
                        logger.log("DATA_EXTRACT", f"Batch {batch_num} Layer {layer_idx}: No data", {
                            "layer_idx": layer_idx,
                            "items_count": 0,
                        })
                    continue
                # 加载最后一个批次的文件（当前批次）
                batch_tensor = torch.load(items[-1], map_location="cpu")
            else:
                # Memory 模式：从缓冲区获取当前批次的数据
                batch_data = runner.storage._batch_buffer.get(layer_idx, {}).get('block', {}).get('recorder', {}).get('default', [])
                if not batch_data:
                    if DEBUG_MEAN_PROJECTION and layer_idx == 0:
                        print(f"[DEBUG] Batch {batch_num} L{layer_idx}: No data in buffer")
                    if logger:
                        logger.log("DATA_EXTRACT", f"Batch {batch_num} Layer {layer_idx}: No data", {
                            "layer_idx": layer_idx,
                            "items_count": 0,
                        })
                    continue
                # 获取缓冲区中的最后一个数据（当前批次）
                batch_tensor = batch_data[-1] if isinstance(batch_data, list) else batch_data
            
            # 确保是 Tensor
            if not isinstance(batch_tensor, torch.Tensor):
                if logger:
                    logger.log("DATA_EXTRACT_ERROR", f"Batch {batch_num} Layer {layer_idx}: Not a Tensor", {
                        "layer_idx": layer_idx,
                        "type": type(batch_tensor).__name__,
                    })
                continue
            
            # 如果数据是 BFloat16 类型，需要先转换为 float32 再转为 numpy
            if batch_tensor.dtype == torch.bfloat16:
                batch_tensor = batch_tensor.float()
            
            # 转换为 numpy，形状应该是 (valid_batch, K_max, dim)
            layer_batch_data = batch_tensor.numpy()
            
            # 调试：记录数据统计信息（阶段5：数据提取检查）
            if DEBUG_MEAN_PROJECTION and layer_idx == 0:
                print(f"[DEBUG] Batch {batch_num} L{layer_idx}: Extracted data shape={layer_batch_data.shape}, "
                      f"mean={layer_batch_data.mean():.6f}, std={layer_batch_data.std():.6f}, "
                      f"min={layer_batch_data.min():.6f}, max={layer_batch_data.max():.6f}, "
                      f"all_zero={(layer_batch_data == 0).all()}")
            
            if logger:
                # 安全地提取样本值，处理不同维度的情况
                sample_values = None
                if len(layer_batch_data.shape) >= 3:
                    if layer_batch_data.shape[0] > 0 and layer_batch_data.shape[1] > 0:
                        if layer_batch_data.shape[2] >= 10:
                            sample_values = layer_batch_data[0, 0, :10].tolist()
                        else:
                            sample_values = layer_batch_data[0, 0, :].tolist()
                elif len(layer_batch_data.shape) == 2:
                    if layer_batch_data.shape[0] > 0 and layer_batch_data.shape[1] >= 10:
                        sample_values = layer_batch_data[0, :10].tolist()
                    elif layer_batch_data.shape[0] > 0:
                        sample_values = layer_batch_data[0, :].tolist()
                
                logger.log("DATA_EXTRACT", f"Batch {batch_num} Layer {layer_idx}: Data extracted", {
                    "layer_idx": layer_idx,
                    "data_shape": list(layer_batch_data.shape),
                    "data_stats": {
                        "mean": float(layer_batch_data.mean()),
                        "std": float(layer_batch_data.std()),
                        "min": float(layer_batch_data.min()),
                        "max": float(layer_batch_data.max()),
                    },
                    "all_zero": bool((layer_batch_data == 0).all()),
                    "sample_values": sample_values,
                })
            
            # 计算投影并累加
            # directions[layer] -> (K_max, dim)
            # layer_batch_data -> (B, K, D)
            # einsum: bkd, kd -> bk
            # sum over b -> k
            
            layer_dirs = directions[:, layer_idx, :] # (K_max, dim)
            
            if logger:
                logger.log("DOT_COMPUTE", f"Batch {batch_num} Layer {layer_idx}: Computing dot products", {
                    "layer_idx": layer_idx,
                    "layer_dirs_shape": list(layer_dirs.shape),
                    "layer_dirs_stats": {
                        "mean": float(layer_dirs.mean()),
                        "std": float(layer_dirs.std()),
                        "min": float(layer_dirs.min()),
                        "max": float(layer_dirs.max()),
                    },
                    "layer_dirs_sample": layer_dirs[0, :10].tolist() if layer_dirs.shape[1] >= 10 else layer_dirs[0, :].tolist(),
                })
            
            dots = np.einsum("bkd,kd->bk", layer_batch_data, layer_dirs, optimize=True)
            
            # 调试：记录投影统计信息（阶段4：点积计算检查）
            if DEBUG_MEAN_PROJECTION and layer_idx == 0:
                print(f"[DEBUG] Batch {batch_num} L{layer_idx}: Projection dots shape={dots.shape}, "
                      f"mean={dots.mean():.6f}, std={dots.std():.6f}, "
                      f"min={dots.min():.6f}, max={dots.max():.6f}, "
                      f"all_zero={(dots == 0).all()}")
            
            if logger:
                logger.log("DOT_RESULT", f"Batch {batch_num} Layer {layer_idx}: Dot products computed", {
                    "layer_idx": layer_idx,
                    "dots_shape": list(dots.shape),
                    "dots_stats": {
                        "mean": float(dots.mean()),
                        "std": float(dots.std()),
                        "min": float(dots.min()),
                        "max": float(dots.max()),
                    },
                    "all_zero": bool((dots == 0).all()),
                    "dots_sum_axis0": dots.sum(axis=0).tolist(),
                })
            
            # 阶段3：累加过程检查
            acc_before = acc[layer_idx].copy()
            dots_sum = dots.sum(axis=0)  # (K_max,)
            acc[layer_idx] += dots_sum
            
            # 调试：记录累加后的统计信息
            if DEBUG_MEAN_PROJECTION and layer_idx == 0:
                print(f"[DEBUG] Batch {batch_num} L{layer_idx}: Accumulator after update shape={acc[layer_idx].shape}, "
                      f"mean={acc[layer_idx].mean():.6f}, std={acc[layer_idx].std():.6f}, "
                      f"all_zero={(acc[layer_idx] == 0).all()}")
            
            if logger:
                logger.log("ACCUMULATE", f"Batch {batch_num} Layer {layer_idx}: Accumulated", {
                    "layer_idx": layer_idx,
                    "dots_sum": dots_sum.tolist(),
                    "acc_before": acc_before.tolist(),
                    "acc_after": acc[layer_idx].tolist(),
                    "acc_delta": (acc[layer_idx] - acc_before).tolist(),
                    "acc_stats_after": {
                        "mean": float(acc[layer_idx].mean()),
                        "std": float(acc[layer_idx].std()),
                        "min": float(acc[layer_idx].min()),
                        "max": float(acc[layer_idx].max()),
                        "sum": float(acc[layer_idx].sum()),
                    },
                })

        total += len(valid_indices)
        meta_ids_all.extend(meta_ids)
        dataset_indices_all.extend(dataset_indices)
        
        if logger:
            logger.log("BATCH_END", f"Batch {batch_num} completed", {
                "total_samples_so_far": total,
                "acc_sum": float(acc.sum()),
            })
        
        # 清除 runner 状态
        runner.clear_data()
        # 重要：Hooks 在 with 块结束时已被移除，但 registry 还在
        # ModelHookRunner 设计用于多次运行相同的 hooks，但这里每批次 hooks 不同（依赖 positions_map）
        # 所以我们需要重置 runner 的 hooks 注册表
        # 目前 runner 没有专门的 unregister 接口，但 reset() 会清除 hooks handles 和 storage
        # 我们直接重新实例化 runner 或者清空 hooks 字典
        runner.hooks.clear() 

    if total == 0:
        raise ValueError("未找到包含完整 EOI 子串的样本，无法计算平均激活。请检查 marker 或数据集。")

    # 阶段2：平均值计算检查
    if logger:
        logger.log("MEAN_COMPUTE_START", "Computing final mean", {
            "total_samples": total,
            "acc_final": acc.tolist(),
            "acc_stats": {
                "mean": float(acc.mean()),
                "std": float(acc.std()),
                "min": float(acc.min()),
                "max": float(acc.max()),
                "sum": float(acc.sum()),
            },
            "layers_with_zero_acc": [i for i in range(n_layers) if (acc[i] == 0).all()],
        })

    mean = np.asarray(acc / total, dtype=np.float32)
    
    # 调试：记录最终结果统计信息
    if DEBUG_MEAN_PROJECTION:
        print(f"[DEBUG] Final result: total samples={total}, mean shape={mean.shape}")
        print(f"  mean stats: mean={mean.mean():.6f}, std={mean.std():.6f}, "
              f"min={mean.min():.6f}, max={mean.max():.6f}, all_zero={(mean == 0).all()}")
        # 打印每层的前几个 token 的统计信息
        for layer_idx in range(min(3, n_layers)):
            print(f"  L{layer_idx} first 3 tokens: {mean[layer_idx, :3].tolist()}")
    
    if logger:
        logger.log("MEAN_COMPUTE_RESULT", "Final mean computed", {
            "mean_shape": list(mean.shape),
            "mean_stats": {
                "mean": float(mean.mean()),
                "std": float(mean.std()),
                "min": float(mean.min()),
                "max": float(mean.max()),
            },
            "all_zero": bool((mean == 0).all()),
            "mean_by_layer": {f"L{i}": mean[i, :].tolist() for i in range(n_layers)},
            "mean_by_token": {f"K{i}": mean[:, i].tolist() for i in range(K_max)},
        })
    
    sample_meta = {
        "meta_ids": meta_ids_all,
        "dataset_indices": dataset_indices_all,
    }
    
    # 关闭日志文件
    if logger:
        logger.close()
        global _trace_log_file, _trace_log_path
        _trace_log_file = None
        _trace_log_path = None
    
    return MeanProjectionResult(mean=mean, n_samples=total, sample_meta=sample_meta)

def compute_comprehensive_projection_data(
    *,
    direction: torch.Tensor,
    batch_size: int,
    model_base,
    dataset: List[Dict],
    marker: Optional[str] = None,
    storage_root: str,
    desc: str = "Comprehensive Projection Analysis",
) -> str:
    """
    计算并存储详细的投影分析数据。
    
    该函数会计算三种类型的投影数据并保存到磁盘：
    1. Attention Heads 投影：每个 attention head 的输出在方向上的投影
    2. MLP 残差分量投影：MLP 模块的输出（残差分量）在方向上的投影
    3. 总残差流投影：整个 Transformer Block 的输出（总残差）在方向上的投影
    
    数据使用 ModelHookRunner 以磁盘模式存储，支持大规模数据分析。
    每个样本、每层、每个 EOI token 位置的投影值都会被保存，便于后续详细分析。
    
    Args:
        direction: 方向向量，用于计算投影
        batch_size: 批处理大小
        model_base: 模型基础对象
        dataset: 数据集列表，每个元素包含 'instruction' 字段
        marker: EOI marker 字符串，用于定位 token 位置
        storage_root: 数据存储根目录路径
        desc: 进度条描述文本
    
    Returns:
        str: 数据存储根目录路径，可通过 ModelHookRunner 访问保存的数据
    """
    # 获取模型配置和工具函数
    tokenizer = model_base.tokenizer
    tok_fn = model_base.tokenize_instructions_fn
    device = model_base.model.device

    block_modules = list(model_base.model_block_modules)
    attn_modules = list(model_base.model_attn_modules)
    mlp_modules = list(model_base.model_mlp_modules)

    if len(block_modules) != len(attn_modules):
        raise ValueError("block_modules 与 attn_modules 数量不一致。")
    
    # 1. 获取模型配置参数
    n_layers = len(block_modules)
    n_heads = model_base.model.config.num_attention_heads
    d_model = model_base.model.config.hidden_size
    head_dim = d_model // n_heads  # 每个 attention head 的维度
    
    # 如果没有提供 marker，从 model_base 中获取（确保与模型定义一致）
    if marker is None:
        eoi_tokens = model_base._get_eoi_toks()
        marker = tokenizer.decode(eoi_tokens)
    
    marker_tokens = tokenizer.encode(marker, add_special_tokens=False)
    marker_K = len(marker_tokens)  # 从 marker 计算的 EOI token 数量
    
    # 检查方向向量的实际 K 维度
    # 如果方向向量是 3D，第一维就是 K 维度
    if direction.ndim == 3:
        direction_K = direction.shape[0]  # 方向向量的第一个维度（K 维度）
        # 如果方向向量的 K 维度与 marker 计算的 K 不匹配，使用方向向量的 K 维度
        # 这处理了方向向量是用旧 marker 生成的情况
        if direction_K != marker_K:
            K_max = direction_K  # 用于方向向量匹配
            marker_K_max = marker_K  # 用于位置查找
        else:
            K_max = marker_K
            marker_K_max = marker_K
    else:
        # 如果方向向量不是 3D，使用 marker 计算的 K_max
        K_max = marker_K
        marker_K_max = marker_K

    # 2. 标准化方向向量
    # 将方向向量标准化为 (K_max, n_layers, d_model) 格式
    directions_np = _standardize_direction(direction, n_layers, K_max)
    # 转换为设备上的张量，以便在 hook 中高效计算点积
    # 使用 torch.tensor 而不是 torch.from_numpy 以避免类型问题
    directions_tensor = torch.tensor(directions_np, device=device, dtype=model_base.model.dtype)

    # 3. 预处理 W_O 权重（用于计算每个 attention head 的输出）
    print("正在预处理 W_O 权重...")
    all_W_o = []  # 存储每层的 W_O 权重，形状为 (n_heads, head_dim, d_model)
    # 使用统一的接口获取输出投影模块
    o_proj_modules = model_base._get_o_proj_modules()
    
    # 从第一个 o_proj 模块获取实际的输入维度（支持 GQA 架构）
    first_o_proj_weight = o_proj_modules[0].weight.detach()
    o_proj_out_dim, o_proj_in_dim = first_o_proj_weight.shape
    
    # 检查输出维度是否等于 d_model
    if o_proj_out_dim != d_model:
        raise ValueError(f"o_proj 输出维度异常：{o_proj_out_dim}，期望 {d_model}")
    
    # 对于 GQA 架构，输入维度可能不等于 d_model
    # 计算实际的 head_dim（基于输入维度）
    actual_head_dim = o_proj_in_dim // n_heads
    if o_proj_in_dim % n_heads != 0:
        raise ValueError(f"o_proj 输入维度 {o_proj_in_dim} 不能被 num_attention_heads {n_heads} 整除")
    
    print(f"  o_proj 权重形状: ({o_proj_out_dim}, {o_proj_in_dim})")
    print(f"  d_model: {d_model}, num_attention_heads: {n_heads}")
    print(f"  实际 head_dim (基于 o_proj 输入): {actual_head_dim}")
    if o_proj_in_dim != d_model:
        print(f"  注意: 使用 GQA 架构，o_proj 输入维度 ({o_proj_in_dim}) != d_model ({d_model})")
    
    for o_proj in o_proj_modules:
        # 获取权重并保持在设备上，避免在 hook 中频繁传输
        weight = o_proj.weight.detach()
        
        # 检查权重形状是否一致
        if weight.shape != (o_proj_out_dim, o_proj_in_dim):
            raise ValueError(f"o_proj 权重形状不一致：{weight.shape}，期望 ({o_proj_out_dim}, {o_proj_in_dim})")
        
        # 转置并重塑权重矩阵，以便分离每个 head 的贡献
        # PyTorch Linear 层权重形状为 (out_features, in_features) = (d_model, o_proj_in_dim)
        # 其中 in_features 维度对应拼接后的所有 heads (n_heads * actual_head_dim)
        # 转置后得到 (o_proj_in_dim, d_model)，然后重塑为 (n_heads, actual_head_dim, d_model)
        # 这样每个 head 的权重矩阵形状为 (actual_head_dim, d_model)
        w_t = weight.t().contiguous().view(n_heads, actual_head_dim, d_model)
        all_W_o.append(w_t)
    
    # 使用列表存储各层的权重，而不是堆叠成一个大张量，以节省内存
    
    # 4. 初始化 ModelHookRunner（磁盘模式）
    print(f"初始化 Disk Hook Runner, 存储路径: {storage_root}")
    runner = ModelHookRunner(model_base, storage_mode='disk', storage_root=storage_root)
    # 手动添加 o_proj 模块映射，因为 ModelHookRunner 默认不包含此类型
    runner.module_type_map['o_proj'] = o_proj_modules
    
    n_batches = (len(dataset) + batch_size - 1) // batch_size
    
    total_samples = 0
    
    # 使用上下文管理器管理整个数据收集过程
    with runner:
        for start in tqdm(range(0, len(dataset), batch_size), desc=desc, total=n_batches):
            batch_records = dataset[start : start + batch_size]
            batch_prompts = [record['instruction'] for record in batch_records]
            enc = tok_fn(instructions=batch_prompts)
            input_ids = enc.input_ids.to(device)
            attn_mask = enc.attention_mask.to(device)

            positions_map: Dict[int, torch.Tensor] = {}
            
            for local_idx in range(len(batch_records)):
                tokens = enc.input_ids[local_idx]
                try:
                    sl = find_substring_token_range(marker, tokens, tokenizer)
                except AssertionError:
                    continue
                
                # 检查找到的 marker 长度
                actual_marker_length = sl.stop - sl.start
                
                # 如果找到的 marker 长度与 marker_K_max 匹配，但方向向量的 K 维度不同
                # 只使用前 direction_K 个 token 位置
                if actual_marker_length == marker_K_max:
                    if direction.ndim == 3 and direction.shape[0] != marker_K_max:
                        # 方向向量的 K 维度与 marker 不匹配，只使用前 direction_K 个 token
                        # 例如：marker 是 7 个 token，但方向向量是 6 个 token
                        positions_map[local_idx] = torch.arange(sl.start, sl.start + K_max, device=device, dtype=torch.long)
                    else:
                        # 完全匹配，使用所有 token
                        positions_map[local_idx] = torch.arange(sl.start, sl.stop, device=device, dtype=torch.long)
                else:
                    # marker 长度不匹配，跳过
                    continue
            
            if not positions_map:
                continue
                
            valid_indices = list(positions_map.keys())
            
            # 定义三个 Hook 函数（使用闭包捕获 valid_indices, positions_map, directions_tensor, all_W_o）
            
            # 1. Attention Hook：计算每个 attention head 的投影
            def make_attn_hook(valid_indices_list, positions_map_dict):
                """
                创建 Attention Hook 函数。
                
                该 Hook 在 o_proj 的输入处捕获激活值 z（所有 heads 拼接后的输出），
                计算每个 head 的输出在方向向量上的投影。
                """
                def hook(ctx, inp, out):
                    # inp[0] 是 z，形状为 (batch, seq, o_proj_in_dim)
                    # z 是所有 attention heads 拼接后的输出，o_proj_in_dim = n_heads * actual_head_dim
                    z_tensor = inp if isinstance(inp, torch.Tensor) else inp[0]
                    layer_idx = ctx.layer_idx
                    
                    # 获取该层的方向向量：形状 (K_max, d_model)
                    dir_layer = directions_tensor[:, layer_idx, :] 
                    # 获取该层的 W_O 权重：形状 (n_heads, actual_head_dim, d_model)
                    W_o_layer = all_W_o[layer_idx]
                    
                    # 提取有效样本在 EOI token 位置的激活值
                    extracted = []
                    for local_idx in valid_indices_list:
                        pos = positions_map_dict[local_idx]  # (K_max,) - EOI token 位置
                        # 提取该样本在这些位置的 z 值：形状 (K_max, o_proj_in_dim)
                        z_slice = z_tensor[local_idx, pos, :] 
                        extracted.append(z_slice)
                    
                    if extracted:
                        # 堆叠所有有效样本：形状 (B_valid, K_max, o_proj_in_dim)
                        z_batch = torch.stack(extracted)
                        B, K, _ = z_batch.shape
                        
                        # 重塑 z 以分离每个 head：形状 (B, K, n_heads, actual_head_dim)
                        z_heads = z_batch.view(B, K, n_heads, actual_head_dim)
                        
                        # 计算每个 head 的输出在方向上的投影
                        # z_heads: (B, K, H, D_h) - 每个 head 的激活值
                        # W_o_layer: (H, D_h, D_m) - 每个 head 的输出投影权重
                        # dir_layer: (K, D_m) - 方向向量（每个 token 位置可能不同）
                        # 结果: (B, K, H) - 每个样本、每个 token、每个 head 的投影值
                        proj = torch.einsum("bkhd, hdm, km -> bkh", z_heads, W_o_layer, dir_layer)
                        
                        # 保存到磁盘（runner 会自动移动到 CPU）
                        ctx.save(proj)
                return hook

            # 2. MLP Hook：计算 MLP 残差分量的投影
            def make_mlp_hook(valid_indices_list, positions_map_dict):
                """
                创建 MLP Hook 函数。
                
                该 Hook 在 MLP 模块的输出处捕获激活值，计算 MLP 残差分量在方向向量上的投影。
                MLP 的输出会被添加到残差流中，这是残差流的一个分量。
                """
                def hook(ctx, inp, out):
                    # out 是 MLP 的输出（残差分量），形状为 (batch, seq, d_model)
                    mlp_out = out if isinstance(out, torch.Tensor) else out[0]
                    layer_idx = ctx.layer_idx
                    
                    # 获取该层的方向向量：形状 (K_max, d_model)
                    dir_layer = directions_tensor[:, layer_idx, :]
                    
                    # 提取有效样本在 EOI token 位置的 MLP 输出
                    extracted = []
                    for local_idx in valid_indices_list:
                        pos = positions_map_dict[local_idx]
                        extracted.append(mlp_out[local_idx, pos, :])
                    
                    if extracted:
                        # 堆叠所有有效样本：形状 (B, K, D)
                        batch_tensor = torch.stack(extracted)
                        # 计算投影：每个样本、每个 token 位置的 MLP 输出在方向上的投影
                        # 结果形状: (B, K)
                        proj = torch.einsum("bkd, kd -> bk", batch_tensor, dir_layer)
                        ctx.save(proj)
                return hook

            # 3. Residual (Block) Hook：计算总残差流的投影
            def make_resid_hook(valid_indices_list, positions_map_dict):
                """
                创建残差流 Hook 函数。
                
                该 Hook 在 Transformer Block 的输出处捕获激活值，计算总残差流在方向向量上的投影。
                Block 的输出包含了经过 attention 和 MLP 处理后的完整残差流状态。
                """
                def hook(ctx, inp, out):
                    # out 是 Block 的输出（总残差），形状为 (batch, seq, d_model)
                    # 注意：某些模型（如 GemmaDecoderLayer）返回元组 (hidden_states, ...)
                    resid_out = out if isinstance(out, torch.Tensor) else out[0]
                    layer_idx = ctx.layer_idx
                    
                    # 获取该层的方向向量：形状 (K_max, d_model)
                    dir_layer = directions_tensor[:, layer_idx, :]
                    
                    # 提取有效样本在 EOI token 位置的残差流状态
                    extracted = []
                    for local_idx in valid_indices_list:
                        pos = positions_map_dict[local_idx]
                        extracted.append(resid_out[local_idx, pos, :])
                    
                    if extracted:
                        # 堆叠所有有效样本：形状 (B, K, D)
                        batch_tensor = torch.stack(extracted)
                        # 计算投影：每个样本、每个 token 位置的总残差在方向上的投影
                        # 结果形状: (B, K)
                        proj = torch.einsum("bkd, kd -> bk", batch_tensor, dir_layer)
                        ctx.save(proj)
                return hook

            # 创建并注册三个 Hook 函数
            attn_hook_fn = make_attn_hook(valid_indices, positions_map)
            mlp_hook_fn = make_mlp_hook(valid_indices, positions_map)
            resid_hook_fn = make_resid_hook(valid_indices, positions_map)
            
            # 为所有层注册 Hook（layers=None 表示应用到所有层）
            runner.register_hook('o_proj', layers=None, hook_fn=attn_hook_fn, name='z_capture')
            runner.register_hook('mlp', layers=None, hook_fn=mlp_hook_fn, name='mlp_proj')
            runner.register_hook('block', layers=None, hook_fn=resid_hook_fn, name='resid_proj')
            
            # 重新注册 PyTorch hooks（因为 hooks 注册表已更新）
            # 先移除旧的 hooks，然后注册新的
            runner._remove_pt_hooks()
            runner._register_pt_hooks()
            
            # 运行模型前向传播，Hook 会在相应位置被调用
            model_base.model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                use_cache=False,
                output_hidden_states=False,
            )
            
            # 将当前批次的数据刷新到磁盘
            # StorageManager 会自动检测批次变化并保存数据
            runner.storage.increment_batch()
            total_samples += len(valid_indices)
            
            # 清理 hooks 注册表，为下一批次做准备
            # 注意：每批次的 positions_map 可能不同，所以需要重新注册 hooks
            runner.hooks.clear()
    
    # 退出 with runner 时，会自动调用 finalize_data() 生成元数据
    print(f"处理完成，共 {total_samples} 个有效样本。数据已保存至 {storage_root}")
    if total_samples == 0:
        print(f"[错误] 没有处理任何有效样本！")
        print(f"  可能的原因：")
        print(f"  1. EOI marker '{marker}' 在数据中找不到")
        print(f"  2. Marker token 数量不匹配（期望 {K_max} 个）")
        print(f"  3. 数据集为空或格式不正确")
    print(f"数据可通过 StorageManager 或 ModelHookRunner.get() 方法访问")
    
    # 返回存储根目录路径
    return storage_root

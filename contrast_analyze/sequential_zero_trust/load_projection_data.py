# -*- coding: utf-8 -*-
"""
load_projection_data.py

从 projection_analysis 目录加载预计算的激活数据。
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import warnings

# 全局缓存：避免重复检测 batch_size
_batch_size_cache: Dict[Tuple[Path, str], int] = {}

# 调试模式标志
DEBUG_MODE = False


def detect_batch_size(
    projection_analysis_root: Path,
    category: Optional[str],
    verbose: bool = False
) -> int:
    """
    自动检测预计算数据中使用的 batch_size（带缓存）。
    
    通过读取第一个可用的 batch 文件来确定实际的 batch_size。
    检测结果会被缓存，避免重复检测。
    
    Args:
        projection_analysis_root: projection_analysis 根目录（可能已包含 category）
        category: 数据集类别
        verbose: 是否输出详细信息（默认 False，只在第一次检测时输出）
    
    Returns:
        检测到的 batch_size
    """
    # 检查缓存
    cache_key = (projection_analysis_root, category)
    if cache_key in _batch_size_cache:
        return _batch_size_cache[cache_key]
    
    # 检查 projection_analysis_root 是否已经包含 category
    # 如果 category 为 None，说明路径已经包含 category
    if category is None:
        base_path = projection_analysis_root
    elif projection_analysis_root.name == category:
        base_path = projection_analysis_root
    else:
        base_path = projection_analysis_root / category
    
    # 首先尝试直接加载第一个 batch 文件（最快的方法）
    test_paths = [
        base_path / "layer_0" / "block" / "resid_proj" / "default" / "batch_0.pt",
        base_path / "layer_0" / "mlp" / "mlp_proj" / "default" / "batch_0.pt",
        base_path / "layer_0" / "o_proj" / "z_capture" / "default" / "batch_0.pt",
    ]
    
    for test_path in test_paths:
        if test_path.exists():
            try:
                data = torch.load(test_path, map_location='cpu', weights_only=False)
                if data.dim() >= 2:
                    detected_batch_size = int(data.shape[0])
                    if verbose:
                        print(f"[检测] 从 {test_path.name} 检测到 batch_size: {detected_batch_size}")
                    # 存入缓存
                    _batch_size_cache[cache_key] = detected_batch_size
                    return detected_batch_size
            except Exception as e:
                continue
    
    # 如果直接加载失败，尝试从 metadata.json 读取文件路径
    metadata_path = base_path / "metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                # 检查是否有 layer_0 的数据
                if 'layer_0' in metadata:
                    layer_0_data = metadata['layer_0']
                    # 尝试从 block/resid_proj 获取文件列表
                    if 'block' in layer_0_data and 'resid_proj' in layer_0_data['block']:
                        resid_data = layer_0_data['block']['resid_proj']
                        if 'default' in resid_data:
                            default_data = resid_data['default']
                            # default_data 可能是字典（包含 files）或列表
                            if isinstance(default_data, dict) and 'files' in default_data:
                                files = default_data['files']
                                if files and len(files) > 0:
                                    first_file = files[0]
                                    if isinstance(first_file, str):
                                        file_path = Path(first_file)
                                    else:
                                        file_path = Path(first_file)
                                    if file_path.exists():
                                        data = torch.load(file_path, map_location='cpu', weights_only=False)
                                        if data.dim() >= 2:
                                            detected_batch_size = int(data.shape[0])
                                            if verbose:
                                                print(f"[检测] 从 metadata.json 引用的文件检测到 batch_size: {detected_batch_size}")
                                            # 存入缓存
                                            _batch_size_cache[cache_key] = detected_batch_size
                                            return detected_batch_size
        except Exception as e:
            pass
    
    # 如果都失败了，返回默认值 32
    if verbose:
        print(f"[警告] 无法自动检测 batch_size，使用默认值 32")
    detected_batch_size = 32
    # 存入缓存
    _batch_size_cache[cache_key] = detected_batch_size
    return detected_batch_size


def get_precomputed_batch_count(
    projection_analysis_root: Path,
    category: str,
    verbose: bool = False
) -> int:
    """
    获取预计算数据中的总 batch 数量。
    
    Args:
        projection_analysis_root: projection_analysis 根目录（可能已包含 category）
        category: 数据集类别
        verbose: 是否输出详细信息
    
    Returns:
        总 batch 数量
    """
    batch_indices = get_available_batch_indices(projection_analysis_root, category, verbose)
    return len(batch_indices)


def get_available_batch_indices(
    projection_analysis_root: Path,
    category: Optional[str],
    verbose: bool = False
) -> List[int]:
    """
    获取所有可用的 batch 索引列表。
    
    优先从 metadata.json 中读取，如果不存在则扫描文件系统。
    
    Args:
        projection_analysis_root: projection_analysis 根目录（可能已包含 category）
        category: 数据集类别
        verbose: 是否输出详细信息
    
    Returns:
        batch 索引列表，已排序
    """
    # 检查 projection_analysis_root 是否已经包含 category
    # 如果 category 为 None，说明路径已经包含 category
    if category is None:
        base_path = projection_analysis_root
    elif projection_analysis_root.name == category:
        base_path = projection_analysis_root
    else:
        base_path = projection_analysis_root / category
    
    # 首先尝试从 metadata.json 读取
    metadata_path = base_path / "metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # 从 layer_0 -> block -> resid_proj -> default -> files 中提取
            if 'layer_0' in metadata:
                layer_0_data = metadata['layer_0']
                if 'block' in layer_0_data and 'resid_proj' in layer_0_data['block']:
                    resid_data = layer_0_data['block']['resid_proj']
                    if 'default' in resid_data:
                        default_data = resid_data['default']
                        if isinstance(default_data, dict) and 'files' in default_data:
                            files = default_data['files']
                            batch_indices = []
                            for file_path in files:
                                if isinstance(file_path, str):
                                    file_path = Path(file_path)
                                else:
                                    file_path = Path(file_path)
                                # 解析文件名 batch_{idx}.pt
                                filename = file_path.name
                                if filename.startswith('batch_') and filename.endswith('.pt'):
                                    try:
                                        batch_idx = int(filename[6:-3])  # 提取 batch_ 和 .pt 之间的数字
                                        batch_indices.append(batch_idx)
                                    except ValueError:
                                        continue
                            
                            if batch_indices:
                                batch_indices = sorted(set(batch_indices))
                                if verbose:
                                    print(f"[获取] 从 metadata.json 获取到 {len(batch_indices)} 个 batch 索引")
                                return batch_indices
        except Exception as e:
            if verbose:
                print(f"[警告] 从 metadata.json 读取失败: {e}")
    
    # 如果 metadata.json 不存在或读取失败，扫描文件系统
    # 使用 layer_0/block/resid_proj/default/ 作为参考路径
    scan_path = base_path / "layer_0" / "block" / "resid_proj" / "default"
    if scan_path.exists():
        batch_indices = []
        for file_path in scan_path.glob("batch_*.pt"):
            try:
                batch_idx = int(file_path.stem[6:])  # 提取 batch_ 之后的数字
                batch_indices.append(batch_idx)
            except ValueError:
                continue
        
        if batch_indices:
            batch_indices = sorted(set(batch_indices))
            if verbose:
                print(f"[获取] 从文件系统扫描到 {len(batch_indices)} 个 batch 索引")
            return batch_indices
    
    # 如果都失败了，返回空列表
    if verbose:
        print(f"[警告] 无法获取 batch 索引列表")
    return []


def get_component_path(
    projection_analysis_root: Path,
    category: str,
    layer_idx: int,
    component_type: str,
    component_idx: Optional[int] = None
) -> Path:
    """
    根据组件类型确定文件路径。
    
    Args:
        projection_analysis_root: projection_analysis 根目录（可能已包含 category，也可能不包含）
        category: 数据集类别（benign, harmful_success, harmful_failure）
        layer_idx: 层索引
        component_type: 组件类型（'head', 'mlp', 'resid'）
        component_idx: 组件索引（对于 head 是 head 索引，其他为 None）
    
    Returns:
        组件数据目录路径
    """
    # 检查 projection_analysis_root 是否已经包含 category
    # 如果 projection_analysis_root 的最后一个目录名是 category，则不重复添加
    if projection_analysis_root.name == category:
        layer_dir = projection_analysis_root / f"layer_{layer_idx}"
    else:
        layer_dir = projection_analysis_root / category / f"layer_{layer_idx}"
    
    if component_type == 'head':
        # Attention head: 使用 o_proj/z_capture
        return layer_dir / "o_proj" / "z_capture" / "default"
    elif component_type == 'mlp':
        # MLP: 使用 mlp/mlp_proj
        return layer_dir / "mlp" / "mlp_proj" / "default"
    elif component_type == 'resid':
        # Residual: 使用 block/resid_proj
        return layer_dir / "block" / "resid_proj" / "default"
    else:
        raise ValueError(f"未知的组件类型: {component_type}")


def load_batch_file(
    batch_path: Path,
    batch_pos: int,
    token_idx: int,
    component_type: str,
    component_idx: Optional[int] = None
) -> float:
    """
    从 batch 文件中加载单个样本的激活值。
    
    注意：projection_analysis 中存储的数据已经是投影值，不需要再计算。
    - z_capture: (batch_size, n_tokens, n_heads) - 每个 head 的投影值
    - mlp_proj: (batch_size, n_tokens) - MLP 的投影值
    - resid_proj: (batch_size, n_tokens) - 残差的投影值
    
    Args:
        batch_path: batch 文件路径
        batch_pos: 样本在 batch 中的位置
        token_idx: token 索引
        component_type: 组件类型（'head', 'mlp', 'resid'）
        component_idx: 组件索引（对于 head 是 head 索引）
    
    Returns:
        激活值（标量）
    """
    if not batch_path.exists():
        return 0.0
    
    try:
        data = torch.load(batch_path, map_location='cpu', weights_only=False)
        
        if component_type == 'head':
            # z_capture 形状: (batch_size, n_tokens, n_heads)
            # 存储的是每个 head 的投影值，直接提取即可
            if data.dim() != 3:
                return 0.0
            
            batch_size, n_tokens, n_heads_data = data.shape
            
            if batch_pos >= batch_size or token_idx >= n_tokens:
                return 0.0
            
            if component_idx is None or component_idx >= n_heads_data:
                return 0.0
            
            # 直接返回该 head 的投影值
            return float(data[batch_pos, token_idx, component_idx].item())
        
        elif component_type == 'mlp':
            # mlp_proj 形状: (batch_size, n_tokens)
            if data.dim() != 2:
                return 0.0
            
            batch_size, n_tokens = data.shape
            
            if batch_pos >= batch_size or token_idx >= n_tokens:
                return 0.0
            
            return float(data[batch_pos, token_idx].item())
        
        elif component_type == 'resid':
            # resid_proj 形状: (batch_size, n_tokens)
            if data.dim() != 2:
                return 0.0
            
            batch_size, n_tokens = data.shape
            
            if batch_pos >= batch_size or token_idx >= n_tokens:
                return 0.0
            
            return float(data[batch_pos, token_idx].item())
        
        else:
            return 0.0
    
    except Exception as e:
        print(f"[警告] 加载文件失败 {batch_path}: {e}")
        return 0.0


def load_projection_activations(
    projection_analysis_root: Path,
    category: Optional[str],
    layer_idx: int,
    component_type: str,
    component_idx: Optional[int],
    token_idx: int,
    sample_indices: List[int],
    batch_size: int = 32,
    batch_cache: Optional[Dict] = None
) -> np.ndarray:
    """
    从 projection_analysis 目录加载激活值（优化版本：批量加载）。
    
    注意：projection_analysis 中存储的数据已经是投影值，直接加载即可。
    
    Args:
        projection_analysis_root: projection_analysis 根目录（可能已包含 category）
        category: 数据集类别（如果为 None，说明路径已包含 category）
        layer_idx: 层索引
        component_type: 组件类型
        component_idx: 组件索引
        token_idx: token 索引
        sample_indices: 样本索引列表
        batch_size: 批处理大小（用于计算 batch 文件索引）
        batch_cache: batch缓存字典，键为 (batch_idx, layer_idx, component_type, component_idx)，值为已加载的tensor
    
    Returns:
        激活值数组，形状为 (len(sample_indices),)
    """
    # 获取组件路径
    component_dir = get_component_path(
        projection_analysis_root, category or '', layer_idx, component_type, component_idx
    )
    
    # 按batch分组样本索引
    batch_groups = {}
    for i, sample_idx in enumerate(sample_indices):
        batch_idx = sample_idx // batch_size
        batch_pos = sample_idx % batch_size
        if batch_idx not in batch_groups:
            batch_groups[batch_idx] = []
        batch_groups[batch_idx].append((i, batch_pos))
    
    # 初始化输出数组
    activations = np.zeros(len(sample_indices), dtype=np.float32)
    
    # 批量加载每个batch
    for batch_idx, positions in batch_groups.items():
        batch_path = component_dir / f"batch_{batch_idx}.pt"
        
        # 检查缓存
        cache_key = (batch_idx, layer_idx, component_type, component_idx)
        if batch_cache is not None and cache_key in batch_cache:
            batch_data = batch_cache[cache_key]
        else:
            # 加载整个batch
            if not batch_path.exists():
                # batch文件不存在，跳过
                continue
            
            try:
                batch_data = torch.load(batch_path, map_location='cpu', weights_only=False)
                # 存入缓存
                if batch_cache is not None:
                    batch_cache[cache_key] = batch_data
            except Exception as e:
                # 加载失败，跳过
                continue
        
        # 批量提取激活值
        if component_type == 'head':
            # z_capture 形状: (batch_size, n_tokens, n_heads)
            if batch_data.dim() != 3:
                if DEBUG_MODE:
                    print(f"[调试] {batch_path}: 期望3D tensor，实际为 {batch_data.dim()}D")
                continue
            batch_size_data, n_tokens, n_heads_data = batch_data.shape
            if token_idx >= n_tokens:
                if DEBUG_MODE:
                    print(f"[调试] {batch_path}: token_idx {token_idx} >= n_tokens {n_tokens}")
                continue
            if component_idx is not None and component_idx >= n_heads_data:
                if DEBUG_MODE:
                    print(f"[调试] {batch_path}: component_idx {component_idx} >= n_heads {n_heads_data}")
                continue
            # 提取所有需要的样本
            for orig_idx, batch_pos in positions:
                if batch_pos < batch_size_data:
                    if component_idx is not None:
                        activations[orig_idx] = float(batch_data[batch_pos, token_idx, component_idx].item())
                    else:
                        activations[orig_idx] = 0.0
                elif DEBUG_MODE:
                    print(f"[调试] {batch_path}: batch_pos {batch_pos} >= batch_size {batch_size_data}")
        
        elif component_type == 'mlp':
            # mlp_proj 形状: (batch_size, n_tokens)
            if batch_data.dim() != 2:
                if DEBUG_MODE:
                    print(f"[调试] {batch_path}: 期望2D tensor，实际为 {batch_data.dim()}D")
                continue
            batch_size_data, n_tokens = batch_data.shape
            if token_idx >= n_tokens:
                if DEBUG_MODE:
                    print(f"[调试] {batch_path}: token_idx {token_idx} >= n_tokens {n_tokens}")
                continue
            # 提取所有需要的样本
            for orig_idx, batch_pos in positions:
                if batch_pos < batch_size_data:
                    activations[orig_idx] = float(batch_data[batch_pos, token_idx].item())
                elif DEBUG_MODE:
                    print(f"[调试] {batch_path}: batch_pos {batch_pos} >= batch_size {batch_size_data}")
        
        elif component_type == 'resid':
            # resid_proj 形状: (batch_size, n_tokens)
            if batch_data.dim() != 2:
                if DEBUG_MODE:
                    print(f"[调试] {batch_path}: 期望2D tensor，实际为 {batch_data.dim()}D")
                continue
            batch_size_data, n_tokens = batch_data.shape
            if token_idx >= n_tokens:
                if DEBUG_MODE:
                    print(f"[调试] {batch_path}: token_idx {token_idx} >= n_tokens {n_tokens}")
                continue
            # 提取所有需要的样本
            for orig_idx, batch_pos in positions:
                if batch_pos < batch_size_data:
                    activations[orig_idx] = float(batch_data[batch_pos, token_idx].item())
                elif DEBUG_MODE:
                    print(f"[调试] {batch_path}: batch_pos {batch_pos} >= batch_size {batch_size_data}")
    
    return activations


def load_all_component_activations(
    projection_analysis_root: Path,
    category: Optional[str],
    valid_components: Dict,
    sample_indices: Optional[List[int]] = None,
    batch_size: Optional[int] = None,
    n_batch: Optional[int] = None,
    seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    加载所有 valid components 的激活值。
    
    注意：projection_analysis 中存储的数据已经是投影值，直接加载即可。
    
    Args:
        projection_analysis_root: projection_analysis 根目录
        category: 数据集类别
        valid_components: valid_components.json 的内容
        sample_indices: 样本索引列表（如果为 None，且 n_batch 不为 None，则根据 n_batch 和 seed 计算）
        batch_size: 批处理大小（如果为 None，则自动检测）
        n_batch: 要读取的 batch 数量（如果为 None，读取所有数据）
        seed: 随机选择 batch 的种子（仅在 n_batch 不为 None 时使用）
    
    Returns:
        激活值字典，键为组件键（如 "L0_H9_T4"），值为激活值数组
    """
    from .component_activations import parse_component
    
    # 检查 projection_analysis_root 是否已经包含 category
    # 如果 category 为 None，说明路径已经包含 category
    if category is None:
        base_path = projection_analysis_root
        actual_category = None  # 路径已包含 category，不需要再添加
    elif projection_analysis_root.name == category:
        base_path = projection_analysis_root
        actual_category = None  # 路径已包含 category，不需要再添加
    else:
        base_path = projection_analysis_root
        actual_category = category  # 需要在路径中添加 category
    
    # 如果 batch_size 未指定，自动检测（只在第一次调用时输出信息）
    if batch_size is None:
        # 检查是否是第一次检测（缓存中不存在）
        cache_key = (projection_analysis_root, category)
        is_first_detection = cache_key not in _batch_size_cache
        batch_size = detect_batch_size(projection_analysis_root, category, verbose=is_first_detection)
    
    # 如果 n_batch 不为 None，根据 n_batch 和 seed 计算 sample_indices
    if n_batch is not None:
        # 获取所有可用的 batch 索引
        all_batch_indices = get_available_batch_indices(
            projection_analysis_root, category, verbose=DEBUG_MODE
        )
        
        if not all_batch_indices:
            raise ValueError(f"无法获取可用的 batch 索引列表（category: {category}）")
        
        # 使用 seed 随机选择 n_batch 个 batch
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()
        
        n_batch_actual = min(n_batch, len(all_batch_indices))
        selected_batch_indices = rng.choice(
            all_batch_indices, 
            size=n_batch_actual, 
            replace=False
        )
        selected_batch_indices = sorted(selected_batch_indices)  # 排序以保持顺序
        
        print(f"[预计算数据] 从 {len(all_batch_indices)} 个可用 batch 中选择了 {n_batch_actual} 个: {selected_batch_indices[:10]}{'...' if len(selected_batch_indices) > 10 else ''}")
        print(f"[一致性检查] 预计算数据 batch 数量: {len(all_batch_indices)}, 选择的 batch: {n_batch_actual}")
        
        # 根据选中的 batch 索引计算对应的样本索引
        sample_indices = []
        for batch_idx in selected_batch_indices:
            start_sample = batch_idx * batch_size
            end_sample = start_sample + batch_size
            sample_indices.extend(range(start_sample, end_sample))
        
        print(f"[预计算数据] 生成了 {len(sample_indices)} 个样本索引（来自 {n_batch_actual} 个 batch，batch_size={batch_size}）")
    
    # 如果 sample_indices 仍为 None，说明需要读取所有数据
    # 这种情况下，我们需要计算所有可用的样本索引
    if sample_indices is None:
        # 获取所有可用的 batch 索引
        all_batch_indices = get_available_batch_indices(
            projection_analysis_root, category, verbose=DEBUG_MODE
        )
        if all_batch_indices:
            print(f"[预计算数据] 读取所有数据: {len(all_batch_indices)} 个 batch")
            print(f"[一致性检查] 预计算数据 batch 数量: {len(all_batch_indices)}")
            # 计算所有样本索引
            sample_indices = []
            for batch_idx in all_batch_indices:
                start_sample = batch_idx * batch_size
                end_sample = start_sample + batch_size
                sample_indices.extend(range(start_sample, end_sample))
            print(f"[预计算数据] 总样本数: {len(sample_indices)} (来自 {len(all_batch_indices)} 个 batch，batch_size={batch_size})")
        else:
            # 如果无法获取 batch 索引，使用一个较大的范围（向后兼容）
            # 假设最多有 10000 个样本
            max_samples = 10000
            sample_indices = list(range(max_samples))
            print(f"[警告] 无法获取 batch 索引，使用默认范围 [0, {max_samples})")
    
    # 创建batch缓存，避免重复加载同一个batch文件
    # 缓存键: (batch_idx, layer_idx, component_type, component_idx)
    batch_cache = {}
    
    activations_dict = {}
    
    # 组织组件按层和 token 分组
    components_by_layer_token = defaultdict(lambda: defaultdict(list))
    
    for token_str, components in valid_components.get('components_by_token', {}).items():
        token_idx = int(token_str)
        for comp_info in components:
            layer = comp_info['layer']
            component = comp_info['component']
            comp_key = f"L{layer}_{component}_T{token_idx}"
            
            components_by_layer_token[layer][token_idx].append({
                'component': component,
                'key': comp_key
            })
    
    # 加载每个组件的激活值（使用共享的batch缓存）
    loaded_count = 0
    failed_count = 0
    
    # 计算总组件数量，用于进度条
    total_components = sum(
        len(components) 
        for token_dict in components_by_layer_token.values() 
        for components in token_dict.values()
    )
    
    # 使用tqdm显示加载进度
    from tqdm import tqdm
    
    # 计算样本数量（用于显示）
    n_samples_expected = len(sample_indices) if sample_indices else 0
    
    with tqdm(total=total_components, desc="加载组件激活值", unit="组件", 
              leave=False, position=1, ncols=120, 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        for layer_idx, token_dict in components_by_layer_token.items():
            for token_idx, components in token_dict.items():
                for comp_info in components:
                    comp_type, comp_idx = parse_component(comp_info['component'])
                    comp_key = comp_info['key']
                    
                    # 所有组件类型都直接加载投影值
                    # 使用 base_path 和 actual_category，并传递batch_cache
                    activations = load_projection_activations(
                        base_path, actual_category, layer_idx, comp_type, comp_idx,
                        token_idx, sample_indices, batch_size, batch_cache=batch_cache
                    )
                    activations_dict[comp_key] = activations
                    loaded_count += 1
                    pbar.update(1)
                    
                    # 更新进度条描述，显示当前组件信息和样本数
                    if loaded_count % 5 == 0 or loaded_count == total_components:
                        current_layer = comp_key.split('_')[0]  # 提取层信息，如 L12
                        pbar.set_postfix({
                            '当前': current_layer,
                            '样本': n_samples_expected if n_samples_expected > 0 else '?'
                        })
                    
                    # 调试输出：检查异常情况
                    if DEBUG_MODE or (activations.size > 0 and (
                        np.allclose(activations, 0.0) or 
                        np.allclose(activations, activations[0]) or
                        np.isnan(activations).any() or
                        np.isinf(activations).any()
                    )):
                        zero_ratio = np.sum(np.abs(activations) < 1e-10) / len(activations)
                        unique_count = len(np.unique(activations))
                        if zero_ratio > 0.5 or unique_count == 1:
                            tqdm.write(f"[调试] {comp_key}: 异常检测 - "
                              f"zero_ratio={zero_ratio:.2%}, unique={unique_count}/{len(activations)}, "
                              f"range=[{np.min(activations):.6f}, {np.max(activations):.6f}]")
                        if np.isnan(activations).any():
                            tqdm.write(f"[警告] {comp_key}: 包含 NaN 值")
                        if np.isinf(activations).any():
                            tqdm.write(f"[警告] {comp_key}: 包含 Inf 值")
    
    # 验证加载的数据
    if DEBUG_MODE or len(activations_dict) == 0:
        validation_results = validate_activation_data(activations_dict, verbose=DEBUG_MODE)
        if validation_results['warnings']:
            print(f"[验证] 发现 {len(validation_results['warnings'])} 个警告")
            for warning in validation_results['warnings'][:10]:  # 只显示前10个
                print(f"  {warning}")
        if validation_results['valid_components'] < len(activations_dict) * 0.9:
            print(f"[警告] 只有 {validation_results['valid_components']}/{validation_results['total_components']} 个组件有效")
    
    # 验证所有组件的激活值数组格式和长度一致性
    if activations_dict:
        # 确保所有值都是 numpy 数组
        for comp_key, activations in activations_dict.items():
            if not isinstance(activations, np.ndarray):
                activations_dict[comp_key] = np.array(activations, dtype=np.float32)
                if DEBUG_MODE:
                    print(f"[转换] {comp_key}: 转换为 numpy 数组")
        
        # 验证长度一致性
        lengths = [len(arr) for arr in activations_dict.values() if isinstance(arr, np.ndarray)]
        if lengths:
            unique_lengths = set(lengths)
            if len(unique_lengths) > 1:
                min_length = min(lengths)
                max_length = max(lengths)
                print(f"[警告] 组件激活值数组长度不一致: 最小={min_length}, 最大={max_length}, 不同长度={unique_lengths}")
                # 统一到最小长度，确保所有组件返回相同数量的样本
                for comp_key, activations in activations_dict.items():
                    if isinstance(activations, np.ndarray) and len(activations) > min_length:
                        activations_dict[comp_key] = activations[:min_length]
                        if DEBUG_MODE:
                            print(f"[调整] {comp_key}: 从 {len(activations)} 截断到 {min_length}")
                print(f"[调整] 所有组件已统一到最小长度: {min_length}")
            else:
                # 所有组件长度一致
                common_length = lengths[0]
                if DEBUG_MODE:
                    print(f"[验证] 所有 {len(activations_dict)} 个组件的激活值数组长度一致: {common_length}")
        
        # 最终验证：确保格式与实时计算一致
        # 实时计算返回: Dict[str, np.ndarray]，其中每个数组形状为 (N_samples,)
        for comp_key, activations in activations_dict.items():
            if not isinstance(activations, np.ndarray):
                raise TypeError(f"{comp_key}: 激活值必须是 numpy 数组，实际类型: {type(activations)}")
            if activations.ndim != 1:
                raise ValueError(f"{comp_key}: 激活值数组必须是一维数组，实际形状: {activations.shape}")
            if activations.dtype != np.float32:
                # 转换为 float32 以保持一致性
                activations_dict[comp_key] = activations.astype(np.float32)
                if DEBUG_MODE:
                    print(f"[转换] {comp_key}: 数据类型转换为 float32")
    
    if DEBUG_MODE:
        if activations_dict:
            first_key = next(iter(activations_dict))
            n_samples_actual = len(activations_dict[first_key])
            print(f"[调试] 加载完成: {loaded_count} 个组件, {n_samples_actual} 个样本")
        else:
            print(f"[调试] 加载完成: {loaded_count} 个组件, 0 个样本")
    
    return activations_dict


def validate_activation_data(
    activations_dict: Dict[str, np.ndarray],
    component_name: Optional[str] = None,
    verbose: bool = False
) -> Dict[str, any]:
    """
    验证加载的激活值数据是否合理。
    
    Args:
        activations_dict: 激活值字典，键为组件键，值为激活值数组
        component_name: 可选，指定要验证的组件名称
        verbose: 是否输出详细信息
    
    Returns:
        验证结果字典，包含统计信息和异常检测结果
    """
    results = {
        'total_components': len(activations_dict),
        'valid_components': 0,
        'invalid_components': [],
        'statistics': {},
        'warnings': []
    }
    
    for comp_key, activations in activations_dict.items():
        if component_name is not None and comp_key != component_name:
            continue
        
        # 检查数据类型和形状
        if not isinstance(activations, np.ndarray):
            results['warnings'].append(f"{comp_key}: 不是 numpy 数组")
            results['invalid_components'].append(comp_key)
            continue
        
        if activations.size == 0:
            results['warnings'].append(f"{comp_key}: 数组为空")
            results['invalid_components'].append(comp_key)
            continue
        
        # 检查异常值
        has_nan = np.isnan(activations).any()
        has_inf = np.isinf(activations).any()
        all_zero = np.allclose(activations, 0.0)
        all_same = np.allclose(activations, activations[0]) if len(activations) > 0 else False
        
        # 计算统计信息
        stats = {
            'shape': activations.shape,
            'dtype': str(activations.dtype),
            'min': float(np.min(activations)),
            'max': float(np.max(activations)),
            'mean': float(np.mean(activations)),
            'std': float(np.std(activations)),
            'has_nan': bool(has_nan),
            'has_inf': bool(has_inf),
            'all_zero': bool(all_zero),
            'all_same': bool(all_same),
            'unique_count': len(np.unique(activations)),
            'zero_count': int(np.sum(np.abs(activations) < 1e-10))
        }
        
        results['statistics'][comp_key] = stats
        
        # 检查问题
        if has_nan:
            results['warnings'].append(f"{comp_key}: 包含 NaN 值")
        if has_inf:
            results['warnings'].append(f"{comp_key}: 包含 Inf 值")
        if all_zero:
            results['warnings'].append(f"{comp_key}: 所有值都为 0")
        if all_same and len(activations) > 1:
            results['warnings'].append(f"{comp_key}: 所有值都相同 ({activations[0]:.6f})")
        
        # 检查零值比例
        zero_ratio = stats['zero_count'] / len(activations)
        if zero_ratio > 0.5:
            results['warnings'].append(f"{comp_key}: {zero_ratio*100:.1f}% 的值为零")
        
        if not (has_nan or has_inf or all_zero):
            results['valid_components'] += 1
        
        # 输出详细信息（如果启用）
        if verbose:
            print(f"[验证] {comp_key}: shape={stats['shape']}, "
                  f"range=[{stats['min']:.6f}, {stats['max']:.6f}], "
                  f"mean={stats['mean']:.6f}, std={stats['std']:.6f}, "
                  f"unique={stats['unique_count']}/{len(activations)}")
    
    return results


def compare_activations(
    activations_precomputed: Dict[str, np.ndarray],
    activations_realtime: Dict[str, np.ndarray],
    tolerance: float = 1e-5,
    verbose: bool = False
) -> Dict[str, any]:
    """
    对比预计算数据和实时计算的数据，确保一致性。
    
    Args:
        activations_precomputed: 预计算的激活值字典
        activations_realtime: 实时计算的激活值字典
        tolerance: 数值容差
        verbose: 是否输出详细信息
    
    Returns:
        对比结果字典
    """
    results = {
        'total_components': len(activations_precomputed),
        'matched_components': 0,
        'mismatched_components': [],
        'missing_components': [],
        'extra_components': [],
        'differences': {}
    }
    
    # 检查缺失的组件
    for comp_key in activations_precomputed.keys():
        if comp_key not in activations_realtime:
            results['missing_components'].append(comp_key)
    
    # 检查额外的组件
    for comp_key in activations_realtime.keys():
        if comp_key not in activations_precomputed:
            results['extra_components'].append(comp_key)
    
    # 对比共同的组件
    common_keys = set(activations_precomputed.keys()) & set(activations_realtime.keys())
    
    for comp_key in common_keys:
        precomp = activations_precomputed[comp_key]
        realtime = activations_realtime[comp_key]
        
        # 检查形状
        if precomp.shape != realtime.shape:
            results['mismatched_components'].append(comp_key)
            results['differences'][comp_key] = {
                'shape_mismatch': True,
                'precomputed_shape': precomp.shape,
                'realtime_shape': realtime.shape
            }
            continue
        
        # 计算差异
        diff = np.abs(precomp - realtime)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        # 检查是否匹配
        is_close = np.allclose(precomp, realtime, atol=tolerance, rtol=tolerance)
        
        if is_close:
            results['matched_components'] += 1
            if verbose:
                print(f"[对比] {comp_key}: 匹配 (max_diff={max_diff:.2e})")
        else:
            results['mismatched_components'].append(comp_key)
            results['differences'][comp_key] = {
                'shape_mismatch': False,
                'max_diff': float(max_diff),
                'mean_diff': float(mean_diff),
                'max_relative_diff': float(np.max(diff / (np.abs(realtime) + 1e-10)))
            }
            if verbose:
                print(f"[对比] {comp_key}: 不匹配 (max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e})")
    
    return results


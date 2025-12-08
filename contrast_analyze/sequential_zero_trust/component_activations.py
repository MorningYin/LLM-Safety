# -*- coding: utf-8 -*-
"""
component_activations.py

实时计算组件激活值（在 direction 上的投影）。
支持多种组件类型：Attention Head (H0-H31), MLP, Block Residual。
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

from contrast_analyze.utils.model_hook_runner import ModelHookRunner, HookContext
from contrast_analyze.utils.activation_utils import find_substring_token_range
from pipeline.model_utils.model_base import ModelBase


def parse_component(component_str: str) -> Tuple[str, Optional[int]]:
    """
    解析组件字符串，返回组件类型和索引。
    
    Args:
        component_str: 组件字符串，如 "H9", "H0", "MLP", "Resid"
    
    Returns:
        (component_type, index): 
            - component_type: 'head', 'mlp', 'resid'
            - index: 对于 head 是 head 索引，其他为 None
    """
    if component_str.startswith('H'):
        try:
            head_idx = int(component_str[1:])
            return 'head', head_idx
        except:
            return 'head', None
    elif component_str.upper() == 'MLP':
        return 'mlp', None
    elif component_str.upper() in ['RESID', 'BLOCK', 'RESIDUAL']:
        return 'resid', None
    else:
        raise ValueError(f"未知的组件类型: {component_str}")


def compute_component_activations(
    model_base: Optional[ModelBase],
    direction,
    dataset: List[Dict],
    valid_components: Dict,
    marker: Optional[str],
    batch_size: int = 32,
    projection_analysis_root: Optional[Path] = None,
    category: Optional[str] = None,
    n_batch: Optional[int] = None,
    seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    计算所有 valid components 的激活值。
    
    如果提供了 projection_analysis_root，则从预计算数据加载；
    否则，实时计算激活值（需要 model_base）。
    
    Args:
        model_base: 模型基础对象（使用预计算数据时可为None）
        direction: 方向向量，形状为 (d_model,) 或 (n_layers, d_model)（使用预计算数据时可为None）
        dataset: 数据集列表，每个元素包含 'instruction' 字段
        valid_components: valid_components.json 的内容
        marker: EOI marker 字符串（使用预计算数据时可为None）
        batch_size: 批处理大小
        projection_analysis_root: projection_analysis 根目录（可选，如果提供则使用预计算数据）
        category: 数据集类别（benign, harmful_success, harmful_failure），用于加载预计算数据
        n_batch: 要读取的 batch 数量（可选，None 表示读取所有）
        seed: 随机选择 batch 的种子（仅在 n_batch 不为 None 时使用）
    
    Returns:
        activations_dict: 字典，键为组件键（如 "L0_H9_T4"），值为激活值数组 (N_samples,)
    """
    # 获取预计算数据的 batch 数量（如果可用），用于统一 batch 选择基准
    # 这确保实时计算和预计算使用相同的 batch 数量基准
    precomputed_batch_count = None
    if projection_analysis_root is not None and category is not None:
        from .load_projection_data import get_precomputed_batch_count
        try:
            precomputed_batch_count = get_precomputed_batch_count(
                projection_analysis_root, category, verbose=False
            )
            print(f"[一致性检查] 预计算数据 batch 数量: {precomputed_batch_count}")
        except Exception as e:
            # 如果无法获取，继续使用数据集 batch 数量
            print(f"[警告] 无法获取预计算 batch 数量: {e}，将使用数据集 batch 数量")
    
    # 如果提供了 projection_analysis_root，使用预计算数据
    if projection_analysis_root is not None and category is not None:
        # 使用预计算数据时，不需要 model_base
        from .load_projection_data import load_all_component_activations, validate_activation_data
        
        # 不再生成 sample_indices，让 load_all_component_activations 根据 n_batch 和 seed 自动计算
        # 如果 n_batch 为 None，load_all_component_activations 会读取所有数据
        
        # 加载预计算数据（如果 batch_size 已提供，使用它；否则自动检测）
        # 注意：batch_size 可能已经在更高层级检测过，这里直接使用
        activations_dict = load_all_component_activations(
            projection_analysis_root=projection_analysis_root,
            category=category,
            valid_components=valid_components,
            sample_indices=None,  # 由 n_batch 和 seed 控制
            batch_size=batch_size,  # 使用传入的 batch_size（如果为 None 则自动检测）
            n_batch=n_batch,
            seed=seed
        )
        
        # 验证加载的数据
        validation_results = validate_activation_data(activations_dict, verbose=False)
        
        # 检查关键问题
        if validation_results['warnings']:
            # 只输出严重警告
            severe_warnings = [w for w in validation_results['warnings'] 
                             if 'NaN' in w or 'Inf' in w or '所有值都为 0' in w]
            if severe_warnings:
                print(f"[警告] 预计算数据加载发现问题 ({len(severe_warnings)} 个严重警告):")
                for warning in severe_warnings[:5]:  # 只显示前5个
                    print(f"  {warning}")
        
        # 检查数据质量
        if len(activations_dict) == 0:
            raise ValueError(f"预计算数据加载失败：没有加载到任何组件激活值")
        
        if validation_results['valid_components'] < len(activations_dict) * 0.5:
            print(f"[警告] 预计算数据质量较差：只有 {validation_results['valid_components']}/{validation_results['total_components']} 个组件有效")
        
        # 输出数据统计（如果启用调试模式）
        if hasattr(load_all_component_activations, '__globals__') and \
           'DEBUG_MODE' in load_all_component_activations.__globals__ and \
           load_all_component_activations.__globals__['DEBUG_MODE']:
            # 从 activations_dict 获取实际样本数量
            if activations_dict:
                first_key = next(iter(activations_dict))
                n_samples_loaded = len(activations_dict[first_key])
                print(f"[调试] 预计算数据统计: {len(activations_dict)} 个组件, {n_samples_loaded} 个样本")
                # 显示前几个组件的统计信息
                for i, (comp_key, activations) in enumerate(list(activations_dict.items())[:3]):
                    stats = validation_results['statistics'].get(comp_key, {})
                    if stats:
                        print(f"  {comp_key}: shape={stats['shape']}, "
                              f"range=[{stats['min']:.6f}, {stats['max']:.6f}], "
                              f"unique={stats['unique_count']}/{len(activations)}")
        
        return activations_dict
    
    # 否则，使用实时计算（原有逻辑）
    if model_base is None:
        raise ValueError("实时计算模式需要 model_base，但提供了 None")
    if direction is None:
        raise ValueError("实时计算模式需要 direction，但提供了 None")
    if marker is None:
        raise ValueError("实时计算模式需要 marker，但提供了 None")
    
    tokenizer = model_base.tokenizer
    tok_fn = model_base.tokenize_instructions_fn
    device = model_base.model.device
    n_layers = len(model_base.model_block_modules)
    n_heads = model_base.model.config.num_attention_heads
    d_model = model_base.model.config.hidden_size
    head_dim = d_model // n_heads
    
    # 统一转换为 torch.Tensor（支持 numpy 和 torch 输入）
    if not isinstance(direction, torch.Tensor):
        # 使用 torch.tensor() 而不是 torch.from_numpy()，因为它更通用
        # torch.tensor() 可以处理 numpy 数组、列表等
        direction = torch.tensor(direction, dtype=torch.float32)
    
    # 标准化方向向量，支持多种形状：
    # - 1D: (d_model,) -> 广播到所有层和所有 token
    # - 2D: (n_layers, d_model) -> 广播到所有 token
    # - 3D: (n_tokens, n_layers, d_model) -> 直接使用
    original_direction = direction
    if direction.dim() == 1:
        # 广播到所有层和所有 token（假设最多 5 个 token）
        direction = direction.unsqueeze(0).unsqueeze(0).expand(5, n_layers, -1)
    elif direction.dim() == 2:
        # (n_layers, d_model) -> 广播到所有 token
        if direction.shape[0] != n_layers:
            raise ValueError(f"方向向量层数不匹配: {direction.shape[0]} != {n_layers}")
        direction = direction.unsqueeze(0).expand(5, n_layers, -1)
    elif direction.dim() == 3:
        # (n_tokens, n_layers, d_model) -> 直接使用
        if direction.shape[1] != n_layers:
            raise ValueError(f"方向向量层数不匹配: {direction.shape[1]} != {n_layers}")
    else:
        raise ValueError(f"不支持的方向向量维度: {direction.dim()}, 形状: {direction.shape}")
    
    direction = direction.to(device=device, dtype=model_base.model.dtype)
    
    # 获取模型模块
    block_modules = list(model_base.model_block_modules)
    attn_modules = list(model_base.model_attn_modules)
    mlp_modules = list(model_base.model_mlp_modules)
    
    # 预处理 W_O 权重（用于计算每个 attention head 的输出）
    all_W_o = []
    # 使用统一的接口获取输出投影模块
    o_proj_modules = model_base._get_o_proj_modules()
    
    for o_proj in o_proj_modules:
        weight = o_proj.weight.detach()
        w_t = weight.t().contiguous().view(n_heads, head_dim, d_model)
        all_W_o.append(w_t)
    
    # 组织 valid components 按层和 token 分组
    components_by_layer_token = defaultdict(lambda: defaultdict(list))
    component_keys = []
    
    for token_str, components in valid_components.get('components_by_token', {}).items():
        token_idx = int(token_str)
        for comp_info in components:
            layer = comp_info['layer']
            component = comp_info['component']
            comp_key = f"L{layer}_{component}_T{token_idx}"
            component_keys.append(comp_key)
            
            components_by_layer_token[layer][token_idx].append({
                'component': component,
                'key': comp_key
            })
    
    # 初始化激活值字典
    activations_dict = {key: [] for key in component_keys}
    
    # 批处理循环
    # 如果提供了 projection_analysis_root，使用预计算数据中的 batch 数量作为基准
    # 这确保实时计算和预计算使用相同的 batch 数量基准
    dataset_batch_count = (len(dataset) + batch_size - 1) // batch_size
    print(f"[一致性检查] 数据集 batch 数量: {dataset_batch_count} (数据集大小: {len(dataset)}, batch_size: {batch_size})")
    
    if precomputed_batch_count is not None:
        n_batches_total = precomputed_batch_count
        if dataset_batch_count != precomputed_batch_count:
            print(f"[一致性检查] ⚠️  数据集 batch 数量 ({dataset_batch_count}) 与预计算数据 batch 数量 ({precomputed_batch_count}) 不一致")
            print(f"           将使用预计算数据的 batch 数量 ({precomputed_batch_count}) 作为基准，确保实时计算和预计算结果一致")
            print(f"           注意：实时计算只会处理前 {min(len(dataset), precomputed_batch_count * batch_size)} 个样本")
        else:
            print(f"[一致性检查] ✓ 数据集 batch 数量与预计算数据 batch 数量一致: {precomputed_batch_count}")
    else:
        n_batches_total = dataset_batch_count
        print(f"[一致性检查] 未提供预计算数据，使用数据集 batch 数量: {dataset_batch_count}")
    
    # 如果指定了 n_batch，随机选择要处理的 batch
    if n_batch is not None:
        # 使用 seed 随机选择 n_batch 个 batch
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()
        
        n_batch_actual = min(n_batch, n_batches_total)
        all_batch_indices = list(range(n_batches_total))
        selected_batch_indices = rng.choice(
            all_batch_indices,
            size=n_batch_actual,
            replace=False
        )
        selected_batch_indices = sorted(selected_batch_indices)  # 排序以保持顺序
        
        print(f"[实时计算] 从 {n_batches_total} 个 batch 中选择了 {n_batch_actual} 个: {selected_batch_indices[:10]}{'...' if len(selected_batch_indices) > 10 else ''}")
        if precomputed_batch_count is not None:
            print(f"[一致性] 使用预计算数据的 batch 数量 ({precomputed_batch_count}) 作为基准")
        
        batch_indices_to_process = selected_batch_indices
    else:
        # 处理所有 batch
        batch_indices_to_process = list(range(n_batches_total))
        if precomputed_batch_count is not None:
            print(f"[一致性] 处理所有 batch，使用预计算数据的 batch 数量 ({precomputed_batch_count}) 作为基准")
    
    # 如果使用预计算数据的 batch 数量作为基准，需要限制处理的样本数量
    max_samples_to_process = None
    if precomputed_batch_count is not None:
        max_samples_to_process = precomputed_batch_count * batch_size
        if len(dataset) > max_samples_to_process:
            print(f"[一致性] 数据集有 {len(dataset)} 个样本，但预计算数据只有 {max_samples_to_process} 个样本")
            print(f"           将只处理前 {max_samples_to_process} 个样本，确保与预计算数据一致")
    
    for batch_idx in tqdm(batch_indices_to_process, desc="计算激活值", unit="批次", leave=False, position=2, ncols=80):
        start_idx = batch_idx * batch_size
        # 如果指定了最大样本数，确保不超过
        if max_samples_to_process is not None:
            if start_idx >= max_samples_to_process:
                # 跳过超出范围的 batch
                continue
            end_idx = min(start_idx + batch_size, max_samples_to_process, len(dataset))
        else:
            end_idx = min(start_idx + batch_size, len(dataset))
        
        if start_idx >= len(dataset):
            # 跳过超出数据集范围的 batch
            continue
            
        batch_records = dataset[start_idx:end_idx]
        batch_prompts = [record['instruction'] for record in batch_records]
        
        # Tokenize
        enc = tok_fn(instructions=batch_prompts)
        input_ids = enc.input_ids.to(device)
        attn_mask = enc.attention_mask.to(device)
        
        # 为每个样本找到 EOI token 位置
        positions_map = {}  # {sample_idx: {token_idx: position}}
        
        for local_idx in range(len(batch_records)):
            tokens = enc.input_ids[local_idx]
            try:
                sl = find_substring_token_range(marker, tokens, tokenizer)
                marker_tokens = tokenizer.encode(marker, add_special_tokens=False)
                K_max = len(marker_tokens)
                
                if (sl.stop - sl.start) == K_max:
                    # 存储每个 token 的位置
                    positions_map[local_idx] = {}
                    for k, pos in enumerate(range(sl.start, sl.stop)):
                        positions_map[local_idx][k] = pos
            except:
                continue
        
        if not positions_map:
            # 如果没有找到 EOI，为所有组件填充 0
            for key in component_keys:
                activations_dict[key].extend([0.0] * len(batch_records))
            continue
        
        # 使用更直接的方法：手动注册 PyTorch hooks 来收集激活值
        # 存储激活值的临时字典
        temp_activations = defaultdict(lambda: defaultdict(dict))
        
        # 手动注册 hooks
        hooks = []
        
        for layer_idx in range(n_layers):
            if layer_idx not in components_by_layer_token:
                continue
            
            # Attention hook (在 o_proj 输入处，使用 forward_pre_hook)
            def make_attn_hook_fn(l_idx):
                def hook_fn(module, input):
                    # forward_pre_hook 只有 input，没有 output
                    z = input if isinstance(input, torch.Tensor) else input[0]
                    temp_activations[l_idx]['attn_z'] = z.detach()
                return hook_fn
            
            # MLP hook
            def make_mlp_hook_fn(l_idx):
                def hook_fn(module, input, output):
                    mlp_out = output if isinstance(output, torch.Tensor) else output[0]
                    temp_activations[l_idx]['mlp'] = mlp_out.detach()
                return hook_fn
            
            # Block hook
            def make_block_hook_fn(l_idx):
                def hook_fn(module, input, output):
                    resid_out = output if isinstance(output, torch.Tensor) else output[0]
                    temp_activations[l_idx]['resid'] = resid_out.detach()
                return hook_fn
            
            # 注册 hooks
            if layer_idx < len(o_proj_modules):
                h1 = o_proj_modules[layer_idx].register_forward_pre_hook(make_attn_hook_fn(layer_idx))
                hooks.append(h1)
            
            if layer_idx < len(mlp_modules):
                h2 = mlp_modules[layer_idx].register_forward_hook(make_mlp_hook_fn(layer_idx))
                hooks.append(h2)
            
            if layer_idx < len(block_modules):
                h3 = block_modules[layer_idx].register_forward_hook(make_block_hook_fn(layer_idx))
                hooks.append(h3)
        
        # 前向传播
        with torch.no_grad():
            model_base.model(input_ids, attention_mask=attn_mask)
        
        # 处理收集到的激活值
        for layer_idx, token_dict in components_by_layer_token.items():
            if layer_idx not in temp_activations:
                # 该层没有激活值，填充 0
                for token_idx, components in token_dict.items():
                    for comp_info in components:
                        activations_dict[comp_info['key']].extend([0.0] * len(batch_records))
                continue
            
            layer_data = temp_activations[layer_idx]
            
            for token_idx, components in token_dict.items():
                # 根据 token_idx 和 layer_idx 选择对应的方向向量
                # direction 形状: (n_tokens, n_layers, d_model)
                if direction.dim() == 3:
                    # 确保 token_idx 在有效范围内
                    token_idx_clamped = min(token_idx, direction.shape[0] - 1)
                    dir_layer = direction[token_idx_clamped, layer_idx, :]  # (d_model,)
                else:
                    # 2D 情况（向后兼容）
                    dir_layer = direction[layer_idx, :]  # (d_model,)
                # 获取该 token 的位置
                token_positions = {}
                for sample_idx in positions_map:
                    if token_idx in positions_map[sample_idx]:
                        token_positions[sample_idx] = positions_map[sample_idx][token_idx]
                
                if not token_positions:
                    # 没有找到该 token，填充 0
                    for comp_info in components:
                        activations_dict[comp_info['key']].extend([0.0] * len(batch_records))
                    continue
                
                # 处理每个组件
                for comp_info in components:
                    comp_type, comp_idx = parse_component(comp_info['component'])
                    comp_key = comp_info['key']
                    
                    activations = []
                    
                    for sample_idx in range(len(batch_records)):
                        if sample_idx not in token_positions:
                            activations.append(0.0)
                            continue
                        
                        pos = token_positions[sample_idx]
                        
                        if comp_type == 'head':
                            # Attention Head: 从 z 计算该 head 的输出投影
                            if 'attn_z' not in layer_data:
                                activations.append(0.0)
                                continue
                            
                            z = layer_data['attn_z']  # (batch, seq, d_model)
                            if sample_idx >= z.shape[0] or pos >= z.shape[1]:
                                activations.append(0.0)
                                continue
                            
                            # 提取该样本该位置的 z: (d_model,)
                            z_slice = z[sample_idx, pos, :]  # (d_model,)
                            # 重塑为 (n_heads, head_dim)
                            z_heads = z_slice.view(n_heads, head_dim)
                            
                            # 计算该 head 的输出
                            W_o = all_W_o[layer_idx]  # (n_heads, head_dim, d_model)
                            head_output = torch.matmul(z_heads[comp_idx:comp_idx+1], W_o[comp_idx])  # (1, d_model)
                            head_output = head_output.squeeze(0)  # (d_model,)
                            
                            # 计算投影
                            proj = torch.dot(head_output, dir_layer).item()
                            activations.append(proj)
                            
                        elif comp_type == 'mlp':
                            # MLP: 从 MLP 输出计算投影
                            if 'mlp' not in layer_data:
                                activations.append(0.0)
                                continue
                            
                            mlp_out = layer_data['mlp']  # (batch, seq, d_model)
                            if sample_idx >= mlp_out.shape[0] or pos >= mlp_out.shape[1]:
                                activations.append(0.0)
                                continue
                            
                            mlp_slice = mlp_out[sample_idx, pos, :]  # (d_model,)
                            proj = torch.dot(mlp_slice, dir_layer).item()
                            activations.append(proj)
                            
                        elif comp_type == 'resid':
                            # Residual: 从 Block 输出计算投影
                            if 'resid' not in layer_data:
                                activations.append(0.0)
                                continue
                            
                            resid_out = layer_data['resid']  # (batch, seq, d_model)
                            if sample_idx >= resid_out.shape[0] or pos >= resid_out.shape[1]:
                                activations.append(0.0)
                                continue
                            
                            resid_slice = resid_out[sample_idx, pos, :]  # (d_model,)
                            proj = torch.dot(resid_slice, dir_layer).item()
                            activations.append(proj)
                    
                    activations_dict[comp_key].extend(activations)
        
        # 移除 hooks
        for hook in hooks:
            hook.remove()
    
    # 转换为 numpy 数组
    result = {}
    for key, values in activations_dict.items():
        result[key] = np.array(values)
    
    return result

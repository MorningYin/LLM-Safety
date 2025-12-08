# -*- coding: utf-8 -*-
"""
activation_utils.py

激活值处理工具函数集合。
从 activation_cache.py 提取，提供激活值计算、缓存、EOI定位等功能。
"""

import json
import os
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Optional
from torch import Tensor
from torch.utils.data import Dataset

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

from pipeline.model_utils.model_base import ModelBase
from contrast_analyze.utils.model_hook_runner import ModelHookRunner
from contrast_analyze.utils.direction_utils import resolve_eoi_marker


# ============================================================
# 工具：从 token 序列中找到某个子串对应的 token 范围
# ============================================================
def find_substring_token_range(substring: str, tokens: Tensor, tokenizer) -> slice:
    """在 token 序列中定位 substring 的 token 切片."""

    decoded = tokenizer.decode(tokens)
    if substring not in decoded:
        raise AssertionError(f"子串 {repr(substring)} 不在 token 序列中")

    # 二分查找 substring 结束位置
    lo, hi = 0, len(tokens)
    while lo < hi:
        mid = (lo + hi) // 2
        if substring in tokenizer.decode(tokens[:mid]):
            hi = mid
        else:
            lo = mid + 1
    end = lo

    # 二分查找 substring 起始位置
    lo, hi = 0, end - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if substring in tokenizer.decode(tokens[mid:end]):
            lo = mid
        else:
            hi = mid - 1
    start = lo

    # 再检查一遍
    assert substring in tokenizer.decode(tokens[start:end])
    return slice(start, end)


# ============================================================
# 自动计算 K_max（EOI token 数）
# ============================================================
def compute_Kmax(model_base: ModelBase, marker: str) -> int:
    """自动计算 EOI 子串的 token 数."""
    tok = model_base.tokenizer
    encoding = tok.encode(marker, add_special_tokens=False)
    K = len(encoding)

    print("====================================")
    print(f"[EOI 子串] {repr(marker)}")
    print(f"[EOI 字符串 tokenize 结果] {encoding}")
    print(f"[EOI token 数 K_max] {K}")
    print("====================================")
    return K


# ============================================================
# 提取完整序列对所有层的 resid_post 激活
# ============================================================
@torch.no_grad()
def compute_activations(
    model_base: ModelBase,
    prompts: List[str],
    batch_size: int
) -> Tuple[Tensor, Tensor]:
    """
    返回：
        input_ids: (N, seq_len)
        activations: (N, n_layers, seq_len, d_model)
    """

    model = model_base.model
    tok = model_base.tokenize_instructions_fn

    runner = ModelHookRunner(model_base, storage_mode='memory')

    def collect_hook(ctx, inp, out):
        # out: (batch, seq_len, hidden)
        # 处理 tuple 输出 (hidden_states, ...)
        if isinstance(out, tuple):
            out = out[0]
        ctx.save(out.detach().cpu())

    # 为每一层注册收集 hook
    runner.register_hook('block', collect_hook, name='activations')

    all_ids = []
    all_acts = []

    n_layers = len(model_base.model_block_modules)

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]

        enc = tok(instructions=batch)
        input_ids = enc.input_ids.to(model.device)
        attn = enc.attention_mask.to(model.device)

        with runner:
            model(input_ids, attention_mask=attn)
        
        # 刷新缓冲区到存储
        runner.increment_batch()
        
        # 提取数据
        # 从 runner 存储中获取每一层的数据
        batch_acts_list = []
        
        for layer_idx in range(n_layers):
            # 使用 get() 方法获取当前批次的数据
            # 在 disk 模式下，需要从索引中获取最后一个批次的文件
            if runner.storage.mode == 'disk':
                items = runner.storage._index.get(layer_idx, {}).get('block', {}).get('activations', {}).get('default', [])
                if items:
                    # 加载最后一个批次的文件（当前批次）
                    layer_data = torch.load(items[-1], map_location="cpu")
                    batch_acts_list.append(layer_data)
                else:
                    # 如果没有数据，创建一个零张量
                    batch_size_actual = input_ids.shape[0]
                    seq_len = input_ids.shape[1]
                    d_model = model.config.hidden_size
                    batch_acts_list.append(torch.zeros(batch_size_actual, seq_len, d_model))
            else:
                # Memory 模式：从缓冲区获取当前批次的数据
                batch_data = runner.storage._batch_buffer.get(layer_idx, {}).get('block', {}).get('activations', {}).get('default', [])
                if batch_data:
                    # 获取缓冲区中的最后一个数据（当前批次）
                    layer_data = batch_data[-1] if isinstance(batch_data, list) else batch_data
                    if not isinstance(layer_data, torch.Tensor):
                        layer_data = torch.tensor(layer_data) if not isinstance(layer_data, torch.Tensor) else layer_data
                    batch_acts_list.append(layer_data)
                else:
                    # 如果没有数据，创建一个零张量
                    batch_size_actual = input_ids.shape[0]
                    seq_len = input_ids.shape[1]
                    d_model = model.config.hidden_size
                    batch_acts_list.append(torch.zeros(batch_size_actual, seq_len, d_model))
        
        # stack layers: (n_layers, batch, seq, dim) -> permute -> (batch, n_layers, seq, dim)
        batch_acts = torch.stack(batch_acts_list, dim=0).permute(1, 0, 2, 3)
        
        all_ids.append(input_ids.cpu())
        all_acts.append(batch_acts)
        
        # 清除缓存，准备下一批
        runner.clear_data()
        runner.hooks.clear() # 清除 hooks 注册

    return torch.cat(all_ids, dim=0), torch.cat(all_acts, dim=0)


# ============================================================
# 缓存 EOI 子串所有 token 激活（核心）
# ============================================================
def cache_eoi_activations(
    model_base: ModelBase,
    prompts: List[str],
    substrings: List[str],
    folder: str,
    batch_size: int,
    shard_size: int
):
    """缓存整个 EOI 子串的激活。"""

    print("\n[提示] 正在计算最大序列长度 seq_len ...")
    sample_size = min(200, len(prompts))
    max_seq_len = 0
    for start_idx in range(0, sample_size, batch_size):
        batch_sample = prompts[start_idx:start_idx + batch_size]
        if not batch_sample:
            break
        test_ids, _ = compute_activations(model_base, batch_sample, batch_size=len(batch_sample))
        max_seq_len = max(max_seq_len, int(test_ids.shape[1]))
    seq_len = max_seq_len
    print(f"[提示] 最大 seq_len = {seq_len}")

    eoi_marker = substrings[0]

    # 自动计算 K_max
    K_max = compute_Kmax(model_base, eoi_marker)

    # 模型结构
    n_layers = model_base.model.config.num_hidden_layers
    d_model = model_base.model.config.hidden_size

    print("\n[提示] 正在创建 memmap 文件 ...")
    
    os.makedirs(folder, exist_ok=True)

    acts_mm = np.memmap(
        f"{folder}/acts.dat",
        dtype="float32",
        mode="w+",
        shape=(shard_size, n_layers, K_max, d_model)
    )
    ids_mm = np.memmap(
        f"{folder}/ids.dat",
        dtype="int32",
        mode="w+",
        shape=(shard_size, seq_len)
    )

    write_ptr = 0

    print("\n======================== 开始缓存激活 ========================")

    for i in tqdm(range(0, len(prompts), batch_size), desc="缓存中"):
        batch_prompts = prompts[i:i + batch_size]
        batch_subs = substrings[i:i + batch_size]

        ids, acts = compute_activations(model_base, batch_prompts, batch_size)

        for j in range(len(batch_prompts)):
            if write_ptr >= shard_size:
                break

            # 找 EOI 对应 token 位置
            try:
                sl = find_substring_token_range(batch_subs[j], ids[j], model_base.tokenizer)
            except:
                continue

            start, end = sl.start, sl.stop
            K = end - start

            if K != K_max:
                print(f"[警告] EOI token 数不一致 ({K} != {K_max})，跳过此样本")
                continue

            # 写入激活
            acts_mm[write_ptr] = acts[j, :, start:end, :].numpy()

            # 写入 ids（pad/truncate）
            ids_sample = ids[j].numpy()
            if len(ids_sample) > seq_len:
                ids_sample = ids_sample[:seq_len]
            elif len(ids_sample) < seq_len:
                pad_id = model_base.tokenizer.pad_token_id or model_base.tokenizer.eos_token_id
                if pad_id is None: pad_id = 0
                padding = np.full(seq_len - len(ids_sample), pad_id, dtype=ids_sample.dtype)
                ids_sample = np.concatenate([ids_sample, padding])
            ids_mm[write_ptr] = ids_sample

            write_ptr += 1

    acts_mm.flush()
    ids_mm.flush()

    # ------------------------------------------------------------
    # 写入 meta.json（关键元信息）
    # ------------------------------------------------------------
    meta = {
        "n_samples": int(write_ptr),
        "seq_len": int(seq_len),
        "K_max": int(K_max),
        "n_layers": int(n_layers),
        "d_model": int(d_model),
        "dtype_acts": "float32",
        "dtype_ids": "int32",
        "model_alias": model_base.model_name if hasattr(model_base, "model_name") else "",
        "eoi_marker": eoi_marker,
        "cache_version": "v1.0"
    }

    meta_path = Path(folder) / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("------------------------------------------------------------")
    print(f"[完成] 总共缓存 {write_ptr}/{shard_size} 个样本")
    print(f"[meta] meta.json 已写入：{meta_path}")
    print(f"[路径] {folder}")
    print("------------------------------------------------------------")


# ============================================================
# Dataset：从 memmap 读取数据
# ============================================================
class ActivationDataset(Dataset):
    """返回 (n_layers, K_max, d_model), (seq_len,)"""

    def __init__(self, folder: str, n_samples: int, n_layers: int, K_max: int, d_model: int, seq_len: int):
        self.acts = np.memmap(
            f"{folder}/acts.dat",
            dtype="float32",
            mode="r",
            shape=(n_samples, n_layers, K_max, d_model)
        )
        self.ids = np.memmap(
            f"{folder}/ids.dat",
            dtype="int32",
            mode="r",
            shape=(n_samples, seq_len)
        )
        self.length = self.ids.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.acts[idx]).float(),
            torch.from_numpy(self.ids[idx]).long()
        )

    def shard(self, num_shards: int, shard_id: int):
        """按比例切分数据集，用于多卡训练。"""
        assert 0 <= shard_id < num_shards
        new = ActivationDataset.__new__(ActivationDataset)

        act_splits = np.array_split(self.acts, num_shards)
        ids_splits = np.array_split(self.ids, num_shards)

        new.acts = act_splits[shard_id]
        new.ids = ids_splits[shard_id]
        new.length = new.ids.shape[0]

        return new


# ============================================================
# 工具：可视化 EOI 子串 token（非常有用）
# ============================================================
def visualize_eoi(model_base: ModelBase, text: str, eoi_marker: str):
    """用于检查你抽取的 EOI 位置是否正确。"""

    tok = model_base.tokenizer
    tokens = tok.encode(text, add_special_tokens=False)
    dec = tok.convert_ids_to_tokens(tokens)

    print("\n=========== 整个 Prompt 的 Token 列表 ===========")
    for i, t in enumerate(dec):
        print(f"{i:3d}: {repr(t)}")
    print("")

    print("=========== 定位 EOI 子串 ===========")
    try:
        sl = find_substring_token_range(eoi_marker, torch.tensor(tokens), tok)
        print(f"EOI token 范围: [{sl.start}:{sl.end}]")
        print("EOI tokens:", dec[sl.start:sl.end])
    except:
        print("未找到 EOI 子串！")

    print("====================================\n")




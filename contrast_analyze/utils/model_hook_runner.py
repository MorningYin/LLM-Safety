"""
ModelHookRunner: 高级机制可解释性框架

一个高度模块化的 hook 管理系统，用于从 transformer 模型中提取和分析激活值。
支持：
- 目标化的 hook 注册（可指定特定层）
- 基于上下文的数据收集
- 灵活的存储方式（内存/磁盘）

ModelHookRunner: Advanced Mechanistic Interpretability Framework

A highly modular hook management system for extracting and analyzing activations
from transformer models, with support for targeted hooking, context-based data collection,
and flexible storage (memory/disk).
"""

import torch
import os
import shutil
import json
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Any, List, Optional, Union
from collections import defaultdict
from dataclasses import dataclass

# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class HookEntry:
    """
    Hook 注册表条目。
    
    用于存储 hook 函数的元信息，包括函数本身、名称和适用的层。
    
    Registry entry for a hook.
    """
    hook_fn: Callable  # Hook 函数，签名为 (ctx, inp, out) -> output | None
    name: str  # Hook 的名称，用于存储和检索数据
    module_type: str  # 模块类型：'block', 'attn', 或 'mlp'
    layers: Optional[Union[int, List[int]]] = None  # 适用的层索引，None 表示所有层

class HookContext:
    """
    Hook 函数的上下文对象。
    
    传递给 hook 函数的上下文，允许 hook 保存数据和访问元数据，
    而无需修改函数签名。这样 hook 函数可以专注于数据处理逻辑。
    
    Context object passed to hook functions.
    
    Allows hooks to save data and access metadata without changing function signature.
    """
    def __init__(self, storage_manager: 'StorageManager', layer_idx: int, 
                 module_type: str, hook_name: str):
        self.storage_manager = storage_manager  # 存储管理器，负责实际的数据保存
        self.layer_idx = layer_idx  # 当前层的索引
        self.module_type = module_type  # 模块类型：'block', 'attn', 或 'mlp'
        self.hook_name = hook_name  # Hook 的名称，用于数据索引
    
    def save(self, data: Any, data_type: str = 'default'):
        """
        保存当前 hook 执行时的数据。
        
        将数据保存到存储管理器中，数据会根据 layer_idx、module_type 和 hook_name 进行索引。
        
        Save data for this hook execution.
        
        Args:
            data: 要保存的 Tensor 或对象。
            data_type: 数据类型标识 (默认为 'default')。
        """
        self.storage_manager.save(
            self.layer_idx, 
            self.module_type, 
            self.hook_name, 
            data_type,
            data
        )

# ============================================================================
# Storage Manager
# ============================================================================

class StorageManager:
    """
    存储管理器：管理 hook 数据的存储和元数据索引。
    
    支持两种存储模式：
    - 'memory': 数据存储在内存中，访问速度快但占用内存
    - 'disk': 数据存储在磁盘上，节省内存但需要 I/O 操作
    
    数据结构：layer_idx -> module_type -> hook_name -> data_type -> List[Data]
    这是一个三层嵌套字典，用于组织和索引不同 hook、不同层、不同模块类型的数据。
    
    Manages storage of hook data with metadata indexing.
    
    Supports 'memory' and 'disk' modes.
    Structure: layer_idx -> module_type -> hook_name -> data_type -> List[Data]
    """
    
    def __init__(self, mode: str = 'memory', storage_root: str = './logs'):
        """
        初始化存储管理器。
        
        Args:
            mode: 存储模式，'memory' 或 'disk'
            storage_root: 磁盘模式下的存储根目录路径
        """
        if mode not in ['memory', 'disk']:
            raise ValueError(f"Invalid storage mode: {mode}")
        
        self.mode = mode  # 存储模式
        self.storage_root = Path(storage_root)  # 磁盘存储的根目录
        self.current_batch_id = 0  # 当前批次 ID，用于磁盘模式下的文件命名
        
        # 当前批次的缓冲区：layer_idx -> module_type -> hook_name -> data_type -> list of objects
        # 使用嵌套 defaultdict 自动创建缺失的键，避免手动初始化
        self._batch_buffer = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
        
        # 数据索引：layer_idx -> module_type -> hook_name -> data_type -> list of (Memory Data or File Path)
        # 内存模式下存储实际数据，磁盘模式下存储文件路径
        self._index = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
        
        # 最终整理后的数据（memory 模式）：layer_idx -> module_type -> hook_name -> data_type -> Tensor
        # 在 finalize_data() 时自动拼接所有批次
        self._final_data: Optional[Dict] = None
        
        # 批次大小跟踪（用于自动批次检测）
        self._last_batch_size: Optional[int] = None
        
        if self.mode == 'disk':
            if self.storage_root.exists():
                # 可选：警告或清理？目前只确保目录存在
                pass
            self.storage_root.mkdir(parents=True, exist_ok=True)

    def save(self, layer_idx: int, module_type: str, hook_name: str, data_type: str = 'default', data: Any = None):
        """
        将数据缓冲到当前批次。
        
        数据不会立即写入存储，而是先保存在缓冲区中。
        调用 increment_batch() 或 finalize_data() 时会将缓冲区数据刷新到实际存储中。
        
        Buffer data for the current batch.
        """
        # 如果是 Tensor，先分离计算图并移到 CPU，避免占用 GPU 内存
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu()
            # 自动检测批次大小（用于自动批次管理）
            # 注意：批次大小检测基于第一个维度，可能不适用于所有情况
            # 更可靠的方式是在 __exit__ 时统一整理数据
            batch_size = data.shape[0] if data.ndim > 0 else 1
            if self._last_batch_size is not None and self._last_batch_size != batch_size:
                # 批次大小变化，说明可能开始了新批次
                # 但为了安全，我们只在缓冲区非空时才刷新
                # 这样可以避免在第一个数据保存时就刷新
                if self._batch_buffer:
                    self.increment_batch()
            self._last_batch_size = batch_size
        # 将数据添加到对应位置的缓冲区列表中
        self._batch_buffer[layer_idx][module_type][hook_name][data_type].append(data)

    def increment_batch(self):
        """
        将缓冲区数据刷新到存储并递增批次 ID。
        
        遍历缓冲区中的所有数据，根据存储模式：
        - 内存模式：直接添加到索引中
        - 磁盘模式：保存到文件并将文件路径添加到索引中
        
        Flush buffer to storage and increment batch ID.
        """
        # 遍历所有 hook、层和模块类型
        for layer_idx, modules in self._batch_buffer.items():
            for module_type, hooks in modules.items():
                for hook_name, data_types in hooks.items():
                    for data_type, data_list in data_types.items():
                        # 处理数据列表（通常每个批次每个 hook 调用一次，但可能多次调用）
                        if not data_list:
                            continue
                    
                        # 如果可能，将多个张量堆叠（假设都是张量）
                        # 这样可以合并同一批次中多次 hook 调用的结果
                        if all(isinstance(x, torch.Tensor) for x in data_list):
                            # print(f"len(data_list): {len(data_list)}")
                            # print(f"layer_idx: {layer_idx}, module_type: {module_type}, hook_name: {hook_name}, data_type: {data_type}")
                            if len(data_list) == 1:
                                batch_data = data_list[0]  # 只有一个，直接使用
                            else:
                                try:
                                    batch_data = torch.stack(data_list)  # 堆叠多个张量
                                except:
                                    batch_data = data_list  # 堆叠失败，保持列表形式
                        else:
                            batch_data = data_list  # 非张量数据，保持列表形式

                        if self.mode == 'memory':
                            # 内存模式：直接存储数据
                            self._index[layer_idx][module_type][hook_name][data_type].append(batch_data)
                        else:
                            # 磁盘模式：保存到文件  
                            # 目录结构：storage_root/layer_{idx}/module_type/hook_name/data_type/
                            dir_path = self.storage_root / f"layer_{layer_idx}" / module_type / hook_name / data_type
                            dir_path.mkdir(parents=True, exist_ok=True)
                            # 文件名：batch_{batch_id}.pt
                            file_path = dir_path / f"batch_{self.current_batch_id}.pt"
                            torch.save(batch_data, file_path)
                            # 存储文件路径而不是数据本身    
                            self._index[layer_idx][module_type][hook_name][data_type].append(file_path)
        
        # 清空缓冲区，准备下一个批次
        self._batch_buffer.clear()
        # 递增批次 ID
        self.current_batch_id += 1

    def get_memory_data(self, layer_idx: int, module_type: str, hook_name: str, data_type: str = 'default'):
        """
        获取所有存储的数据（仅内存模式）。
        
        返回完整的索引字典，包含所有存储的数据。
        仅在内存模式下可用，因为磁盘模式下存储的是文件路径。
        
        Get all stored data (memory mode only).
        """
        if self.mode != 'memory':
            raise RuntimeError("get_memory_data() only available in memory mode")
        return self._index[layer_idx][module_type][hook_name][data_type]

    def finalize_data(self):
        """
        整理所有数据：将批次数据拼接成统一张量（memory 模式）。
        
        在运行结束后调用，自动拼接所有批次数据，形成统一的数据视图。
        对于 memory 模式，数据会被拼接并存储在 `_final_data` 中。
        对于 disk 模式，会生成索引文件。
        """
        # 先刷新最后一个批次
        if self._batch_buffer:
            self.increment_batch()
        
        if self.mode == 'memory':
            # Memory 模式：拼接所有批次数据
            self._final_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
            
            for layer_idx, modules in self._index.items():
                for module_type, hooks in modules.items():
                    for hook_name, data_types in hooks.items():
                        for data_type, batch_list in data_types.items():
                            if not batch_list:
                                continue
                            
                            # 拼接所有批次
                            if all(isinstance(x, torch.Tensor) for x in batch_list):
                                try:
                                    # 沿批次维度拼接
                                    final_tensor = torch.cat(batch_list, dim=0)
                                    self._final_data[layer_idx][module_type][hook_name][data_type] = final_tensor
                                except Exception as e:
                                    # 拼接失败，保持列表形式
                                    self._final_data[layer_idx][module_type][hook_name][data_type] = batch_list
                            else:
                                # 非张量数据，保持列表形式
                                flat_list = [item for batch in batch_list for item in (batch if isinstance(batch, list) else [batch])]
                                self._final_data[layer_idx][module_type][hook_name][data_type] = flat_list
        else:
            # Disk 模式：生成索引文件
            metadata_path = self.storage_root / "metadata.json"
            index_data = {}
            
            for layer_idx, modules in self._index.items():
                layer_key = f"layer_{layer_idx}"
                index_data[layer_key] = {}
                
                for module_type, hooks in modules.items():
                    index_data[layer_key][module_type] = {}
                    
                    for hook_name, data_types in hooks.items():
                        index_data[layer_key][module_type][hook_name] = {}
                        
                        for data_type, file_paths in data_types.items():
                            # 获取第一个文件的信息（假设所有文件形状相同）
                            if file_paths and isinstance(file_paths[0], Path) and file_paths[0].exists():
                                sample_data = torch.load(file_paths[0])
                                if isinstance(sample_data, torch.Tensor):
                                    shape = list(sample_data.shape)
                                    dtype = str(sample_data.dtype)
                                else:
                                    shape = None
                                    dtype = None
                            else:
                                shape = None
                                dtype = None
                            
                            index_data[layer_key][module_type][hook_name][data_type] = {
                                "files": [str(p) for p in file_paths],
                                "shape": shape,
                                "dtype": dtype,
                                "file_count": len(file_paths)
                            }
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, indent=2, ensure_ascii=False)
    
    def get_final_data(self, layer_idx: int, module_type: str, hook_name: str, data_type: str = 'default') -> Optional[torch.Tensor]:
        """
        获取整理后的最终数据（memory 模式）。
        
        Args:
            layer_idx: 层索引
            module_type: 模块类型
            hook_name: Hook 名称
            data_type: 数据类型
            
        Returns:
            拼接后的张量，如果不存在则返回 None
        """
        if self.mode != 'memory':
            raise RuntimeError("get_final_data() only available in memory mode")
        
        if self._final_data is None:
            return None
        
        return self._final_data.get(layer_idx, {}).get(module_type, {}).get(hook_name, {}).get(data_type)
    
    def get_all_final_data(self) -> Dict:
        """
        获取所有整理后的最终数据（memory 模式）。
        
        Returns:
            完整的数据字典
        """
        if self.mode != 'memory':
            raise RuntimeError("get_all_final_data() only available in memory mode")
        
        if self._final_data is None:
            return {}
        
        return dict(self._final_data)

    def clear(self):
        """
        清空所有存储的数据。
        
        清空缓冲区和索引，重置批次 ID。
        如果是磁盘模式，还会删除存储目录中的所有文件。
        """
        self._batch_buffer.clear()  # 清空缓冲区
        self._index.clear()  # 清空索引
        self._final_data = None  # 清空最终数据
        self._last_batch_size = None  # 重置批次大小跟踪
        self.current_batch_id = 0  # 重置批次 ID
        if self.mode == 'disk' and self.storage_root.exists():
            # 磁盘模式：删除所有存储的文件
            shutil.rmtree(self.storage_root)
            self.storage_root.mkdir()

# ============================================================================
# Model Hook Runner
# ============================================================================

class ModelHookRunner:
    """
    模型 Hook 运行器：管理模型 hooks 和数据收集。
    
    这是框架的核心类，负责：
    1. 注册和管理 PyTorch forward hooks
    2. 在模型前向传播时执行注册的 hooks
    3. 收集和存储激活值数据
    4. 支持上下文管理器模式（with 语句）
    5. 自动批次管理和数据整理
    
    Runner for managing model hooks and data collection.
    """
    
    def __init__(self, model_base, storage_mode: str = 'memory', storage_root: str = './logs'):
        """
        初始化模型 Hook 运行器。
        
        Args:
            model_base: 模型基础对象，需要包含模型和模块列表
            storage_mode: 存储模式，'memory' 或 'disk'
            storage_root: 磁盘模式下的存储根目录
        """
        self.model_base = model_base  # 模型基础对象
        self.storage = StorageManager(mode=storage_mode, storage_root=storage_root)  # 存储管理器
        
        # 注册的 hooks：layer_idx -> module_type -> List[HookEntry]
        # 使用 defaultdict 自动创建空列表，避免手动初始化
        self.hooks: Dict[int, Dict[str, List[HookEntry]]] = defaultdict(lambda: defaultdict(list))
        self.hook_handles = []  # PyTorch hook 句柄列表，用于后续移除 hooks
        
        # Hook 注册表（用于元数据）：hook_name -> HookEntry
        self._hook_registry: Dict[str, HookEntry] = {}
        
        # 模块类型映射：将字符串类型映射到实际的模块列表
        self.module_type_map = {
            'block': model_base.model_block_modules,  # Transformer 块模块列表
            'attn': model_base.model_attn_modules,   # 注意力模块列表
            'mlp': model_base.model_mlp_modules,      # MLP 模块列表
        }
        
        # 数据收集开始时间
        self._collection_start_time: Optional[datetime] = None

    def register_hook(self, module_type: str, 
                      layers: Optional[Union[int, List[int]]] = None, 
                      hook_fn: Optional[Callable] = None,
                      name: Optional[str] = None):
        """
        注册一个 hook。
        
        将 hook 函数注册到指定的模块类型上，可以选择性地指定适用的层。
        Hook 函数会在模型前向传播时被调用。
        
        Register a hook.
        
        Args:
            module_type: 模块类型，'block', 'attn', 或 'mlp'
            hook_fn: Hook 函数，签名为 (ctx, inp, out) -> output | None
                    如果返回 None，不修改输出；如果返回张量，会替换输出（用于干预）
            layers: 适用的层索引，可以是整数、整数列表或 None（所有层）
            name: Hook 的名称，用于存储和检索数据。默认为函数名
        """
        if module_type not in self.module_type_map:
            raise ValueError(f"Invalid module_type: {module_type}")
        
        # 如果没有指定名称，使用函数名
        if name is None:
            name = hook_fn.__name__

        # 创建 Hook 条目
        entry = HookEntry(hook_fn=hook_fn, name=name, layers=layers, module_type=module_type)
        
        # 注册到 hook 注册表（用于元数据）
        self._hook_registry[name] = entry
        
        # 确定目标层
        target_layers = []
        if layers is None:
            # 如果 layers 为 None，应用到该模块类型的所有层
            target_layers = list(range(len(self.module_type_map[module_type])))
        elif isinstance(layers, int):
            target_layers = [layers]
        elif isinstance(layers, list):
            target_layers = layers
            
        # 将 hook 注册到所有目标层
        for layer_idx in target_layers:
            self.hooks[layer_idx][module_type].append(entry)
    
    def register(self, module_type: str, hook_fn: Callable, layers: Optional[Union[int, List[int]]] = None, name: Optional[str] = None):
        """
        注册 hook 的简化接口（推荐使用）。
        
        Args:
            module_type: 模块类型，'block', 'attn', 或 'mlp'
            hook_fn: Hook 函数，签名为 (ctx, inp, out) -> output | None
            layers: 适用的层索引，可以是整数、整数列表或 None（所有层）
            name: Hook 的名称，默认为函数名
        """
        self.register_hook(module_type, layers=layers, hook_fn=hook_fn, name=name)

    def _create_forward_hook(self, layer_idx: int, module_type: str):
        """
        创建 PyTorch forward hook 闭包。
        
        为指定的层和模块类型创建一个 hook 函数，该函数会执行所有相关的注册 hooks。
        使用闭包捕获 layer_idx 和 module_type，但每次都从 self.hooks 获取最新的 hooks。
        
        Create the PyTorch forward hook closure.
        """

        def forward_hook(module, input, output):
            """
            PyTorch forward hook 函数。
            
            在模块的前向传播完成后被调用，执行所有注册的 hooks。
            支持多个 hooks 链式执行，每个 hook 可以修改输出。
            注意：每次都从 self.hooks 获取最新的 hooks，以支持动态注册。
            """
            # 每次执行时都从 self.hooks 获取最新的 hooks（支持动态注册）
            relevant_hooks = self.hooks[layer_idx][module_type]
            
            if not relevant_hooks:
                return output  # 没有相关的 hooks，直接返回原始输出
            
            # 解包元组形式的输入/输出（如果需要）
            # PyTorch 模块可能返回元组，我们通常只需要第一个元素（张量）
            inp = input[0] if isinstance(input, tuple) else input
            out = output[0] if isinstance(output, tuple) else output
            
            current_out = out  # 当前输出，可能被 hooks 修改
            
            # 依次执行所有相关的 hooks
            for entry_idx, entry in enumerate(relevant_hooks):
                # 创建上下文对象，提供存储接口和元数据
                ctx = HookContext(self.storage, layer_idx, module_type, entry.name)
                
                # 调试：记录 hook 执行顺序
                if os.getenv("DEBUG_MEAN_PROJECTION", "0") == "1" and layer_idx in [12, 13, 14, 15]:
                    print(f"[DEBUG] ModelHookRunner L{layer_idx}: Executing hook {entry_idx+1}/{len(relevant_hooks)}: name='{entry.name}'")
                
                # 运行 hook 函数
                # 签名：hook(ctx, inp, out) -> output | None
                new_out = entry.hook_fn(ctx, inp, current_out)
                
                # 如果 hook 返回了新输出（用于残差流干预），更新当前输出
                if new_out is not None:
                    current_out = new_out
                    if os.getenv("DEBUG_MEAN_PROJECTION", "0") == "1" and layer_idx in [12, 13, 14, 15]:
                        print(f"[DEBUG] ModelHookRunner L{layer_idx}: Hook '{entry.name}' returned modified output")
            
            # 如果原始输出是元组，重新打包输出
            # 通常 transformer 层返回元组，但我们只修改张量部分
            # 如果原始是元组，我们返回以修改后的张量作为第一个元素的元组
            if isinstance(output, tuple):
                return (current_out,) + output[1:]
            return current_out

        return forward_hook

    def _register_pt_hooks(self):
        """
        向 PyTorch 模块注册 hooks。
        
        遍历所有模块类型和层，为每个模块创建并注册 forward hook。
        只注册有相关 hooks 的模块，避免不必要的开销。
        
        Register hooks with PyTorch modules.
        """
        for module_type, module_list in self.module_type_map.items():
            for layer_idx, module in enumerate(module_list):
                # 优化：检查该位置是否有注册的 hook
                # self.hooks 结构：layer_idx -> module_type -> List[HookEntry]
                if not self.hooks[layer_idx][module_type]:
                    continue
                
                hook = self._create_forward_hook(layer_idx, module_type)
                if hook:  # 如果创建成功，注册到 PyTorch 模块
                    handle = module.register_forward_hook(hook)
                    self.hook_handles.append(handle)  # 保存句柄以便后续移除

    def _remove_pt_hooks(self):
        """
        移除所有 PyTorch hooks。
        
        从所有模块中移除注册的 forward hooks，并清空句柄列表。
        这通常在上下文管理器退出时自动调用。
        """
        for handle in self.hook_handles:
            handle.remove()  # 移除 hook
        self.hook_handles = []  # 清空句柄列表

    def __enter__(self):
        """
        上下文管理器入口：进入时注册 hooks。
        
        允许使用 with 语句：
        with ModelHookRunner(...) as runner:
            model(inputs)
        """
        self._collection_start_time = datetime.now()
        self._register_pt_hooks()  # 注册所有 hooks
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        上下文管理器出口：退出时移除 hooks 并整理数据。
        
        无论是否发生异常，都会：
        1. 移除所有 hooks，确保模型状态干净
        2. 自动整理所有数据（拼接批次等）
        """
        self._remove_pt_hooks()  # 移除所有 hooks
        # 自动整理数据
        self.storage.finalize_data()

    def increment_batch(self):
        """
        递增批次。
        
        将当前批次的数据从缓冲区刷新到存储中，并递增批次 ID。
        在每批次数据处理完成后调用。
        """
        self.storage.increment_batch()

    def clear_data(self):
        """
        清空存储的数据。
        
        清空所有收集的激活值数据，重置存储状态。
        """
        self.storage.clear()
        
    def get(self, module_type: str, layer: int, hook: str, data_type: str = 'default') -> Optional[torch.Tensor]:
        """
        获取指定位置的数据（推荐使用）。
        
        返回拼接后的统一张量，shape 为 (total_samples, ...)。
        
        Args:
            module_type: 模块类型，'block', 'attn', 或 'mlp'
            layer: 层索引
            hook: Hook 名称
            data_type: 数据类型，默认为 'default'
            
        Returns:
            拼接后的张量，如果不存在则返回 None
        """
        if self.storage.mode == 'memory':
            return self.storage.get_final_data(layer, module_type, hook, data_type)
        else:
            # Disk 模式：需要从文件加载
            items = self.storage._index.get(layer, {}).get(module_type, {}).get(hook, {}).get(data_type, [])
            if not items:
                return None
            
            # 加载并拼接所有文件
            data_list = [torch.load(p) if isinstance(p, Path) else p for p in items]
            if all(isinstance(x, torch.Tensor) for x in data_list):
                return torch.cat(data_list, dim=0)
            return data_list
    
    def get_all_layers(self, module_type: str, hook: str, data_type: str = 'default') -> Dict[int, torch.Tensor]:
        """
        获取所有层的数据。
        
        Args:
            module_type: 模块类型
            hook: Hook 名称
            data_type: 数据类型
            
        Returns:
            字典，键为层索引，值为该层的数据张量
        """
        result = {}
        if self.storage.mode == 'memory':
            all_data = self.storage.get_all_final_data()
            for layer_idx, modules in all_data.items():
                if module_type in modules and hook in modules[module_type]:
                    if data_type in modules[module_type][hook]:
                        result[layer_idx] = modules[module_type][hook][data_type]
        else:
            # Disk 模式：遍历索引
            for layer_idx in self.storage._index.keys():
                data = self.get(module_type, layer_idx, hook, data_type)
                if data is not None:
                    result[layer_idx] = data
        return result
    
    def get_all(self) -> Dict:
        """
        获取所有数据（完整索引）。
        
        Returns:
            完整的数据字典
        """
        if self.storage.mode == 'memory':
            return self.storage.get_all_final_data()
        else:
            # Disk 模式：返回索引结构
            return dict(self.storage._index)
    
    def get_by_module(self, module_type: str, layer: int) -> Dict:
        """
        获取指定模块类型和层的所有 hook 数据。
        
        Args:
            module_type: 模块类型
            layer: 层索引
            
        Returns:
            字典，键为 (hook_name, data_type)，值为数据
        """
        result = {}
        if self.storage.mode == 'memory':
            all_data = self.storage.get_all_final_data()
            layer_data = all_data.get(layer, {}).get(module_type, {})
            for hook_name, data_types in layer_data.items():
                for data_type, data in data_types.items():
                    result[(hook_name, data_type)] = data
        else:
            # Disk 模式
            layer_data = self.storage._index.get(layer, {}).get(module_type, {})
            for hook_name, data_types in layer_data.items():
                for data_type, file_paths in data_types.items():
                    result[(hook_name, data_type)] = file_paths
        return result
    
    @property
    def metadata(self) -> Dict:
        """
        获取完整的元数据，包含所有数据的详细信息。
        
        Returns:
            元数据字典，包含：
            - hook_registry: 所有注册的 hook 信息
            - collection_info: 收集过程的统计信息
            - data_catalog: 每个数据项的详细信息（shape、dtype、大小等）
            - storage_details: 存储相关的详细信息
        """
        metadata = {
            "hook_registry": {},
            "collection_info": {},
            "data_catalog": {},
            "storage_details": {}
        }
        
        # Hook 注册信息
        for hook_name, entry in self._hook_registry.items():
            # 确定实际应用的层
            target_layers = []
            if entry.layers is None:
                target_layers = list(range(len(self.module_type_map[entry.module_type])))
            elif isinstance(entry.layers, int):
                target_layers = [entry.layers]
            elif isinstance(entry.layers, list):
                target_layers = entry.layers
            
            # 收集该 hook 使用的所有 data_type
            data_types = set()
            if self.storage.mode == 'memory':
                all_data = self.storage.get_all_final_data()
                for layer_idx in target_layers:
                    layer_data = all_data.get(layer_idx, {}).get(entry.module_type, {}).get(hook_name, {})
                    data_types.update(layer_data.keys())
            else:
                for layer_idx in target_layers:
                    layer_data = self.storage._index.get(layer_idx, {}).get(entry.module_type, {}).get(hook_name, {})
                    data_types.update(layer_data.keys())
            
            metadata["hook_registry"][hook_name] = {
                "module_type": entry.module_type,
                "layers": target_layers,
                "data_types": sorted(list(data_types)),
                "hook_function": entry.hook_fn.__name__
            }
        
        # 收集信息
        total_batches = self.storage.current_batch_id
        total_samples = 0
        total_data_size_mb = 0.0
        file_count = 0
        
        # 数据目录（包含详细信息）
        if self.storage.mode == 'memory':
            all_data = self.storage.get_all_final_data()
            for layer_idx, modules in all_data.items():
                layer_key = f"layer_{layer_idx}"
                metadata["data_catalog"][layer_key] = {}
                
                for module_type, hooks in modules.items():
                    metadata["data_catalog"][layer_key][module_type] = {}
                    
                    for hook_name, data_types in hooks.items():
                        metadata["data_catalog"][layer_key][module_type][hook_name] = {}
                        
                        for data_type, data in data_types.items():
                            if isinstance(data, torch.Tensor):
                                shape = list(data.shape)
                                dtype = str(data.dtype)
                                device = str(data.device)
                                total_elements = data.numel()
                                memory_size_mb = total_elements * data.element_size() / (1024 * 1024)
                                
                                # 推断维度含义
                                samples = shape[0] if len(shape) > 0 else 0
                                if len(shape) > 1:
                                    sequence_length = shape[1]
                                else:
                                    sequence_length = None
                                if len(shape) > 2:
                                    hidden_dim = shape[2]
                                else:
                                    hidden_dim = None
                                
                                total_samples = max(total_samples, samples)
                                total_data_size_mb += memory_size_mb
                                
                                metadata["data_catalog"][layer_key][module_type][hook_name][data_type] = {
                                    "shape": shape,
                                    "dtype": dtype,
                                    "device": device,
                                    "total_elements": total_elements,
                                    "memory_size_mb": round(memory_size_mb, 2),
                                    "data_type": data_type,
                                    "samples": samples,
                                    "sequence_length": sequence_length,
                                    "hidden_dim": hidden_dim
                                }
                            else:
                                # 非张量数据
                                metadata["data_catalog"][layer_key][module_type][hook_name][data_type] = {
                                    "type": type(data).__name__,
                                    "data_type": data_type
                                }
        else:
            # Disk 模式
            for layer_idx, modules in self.storage._index.items():
                layer_key = f"layer_{layer_idx}"
                metadata["data_catalog"][layer_key] = {}
                
                for module_type, hooks in modules.items():
                    metadata["data_catalog"][layer_key][module_type] = {}
                    
                    for hook_name, data_types in hooks.items():
                        metadata["data_catalog"][layer_key][module_type][hook_name] = {}
                        
                        for data_type, file_paths in data_types.items():
                            file_count += len(file_paths)
                            
                            # 加载第一个文件获取信息
                            if file_paths and isinstance(file_paths[0], Path) and file_paths[0].exists():
                                sample_data = torch.load(file_paths[0])
                                if isinstance(sample_data, torch.Tensor):
                                    shape = list(sample_data.shape)
                                    dtype = str(sample_data.dtype)
                                    total_elements = sample_data.numel()
                                    memory_size_mb = total_elements * sample_data.element_size() / (1024 * 1024)
                                    
                                    samples = shape[0] if len(shape) > 0 else 0
                                    total_samples = max(total_samples, samples)
                                    total_data_size_mb += memory_size_mb * len(file_paths)
                                    
                                    metadata["data_catalog"][layer_key][module_type][hook_name][data_type] = {
                                        "shape": shape,
                                        "dtype": dtype,
                                        "total_elements": total_elements,
                                        "memory_size_mb": round(memory_size_mb, 2),
                                        "data_type": data_type,
                                        "samples": samples,
                                        "file_count": len(file_paths),
                                        "files": [str(p) for p in file_paths[:5]]  # 只显示前5个文件路径
                                    }
        
        # 收集信息
        metadata["collection_info"] = {
            "total_batches": total_batches,
            "total_samples": total_samples,
            "storage_mode": self.storage.mode,
            "collection_time": self._collection_start_time.isoformat() if self._collection_start_time else None
        }
        
        # 存储详情
        metadata["storage_details"] = {
            "mode": self.storage.mode,
            "root_path": str(self.storage.storage_root) if self.storage.mode == 'disk' else None,
            "total_data_size_mb": round(total_data_size_mb, 2),
            "file_count": file_count
        }
        
        return metadata

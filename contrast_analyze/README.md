# contrast_analyze 代码结构说明

## 概述

`contrast_analyze` 是一个用于研究拒答方向作用机制的实验 pipeline。代码采用三层架构设计，便于扩展和维护。

## 架构分层

### 1. 底层（utils/）：纯函数，无副作用，可复用

底层函数是纯函数，不依赖外部状态，可以在任何地方复用。

#### `utils/direction_utils.py` - 方向向量工具
- `normalize_direction()` - 归一化方向向量
- `load_direction_global()` - 加载全局方向（1D tensor）
- `load_direction_candidates()` - 加载候选方向（3D tensor）
- `resolve_eoi_marker()` - 根据模型别名解析 EOI 标记
- `normalize_token_index()` - 规范化 token 索引（支持负数）

#### `utils/plot_utils.py` - 绘图工具
- `plot_heatmap()` - 绘制热力图（核心函数）
- `plot_comparison()` - 绘制对比图（baseline vs aligned）
- `plot_delta()` - 绘制差值图

#### `utils/data_utils.py` - 数据工具
- `cache_dir_path()` - 获取缓存目录路径
- `load_meta()` - 加载 meta.json
- `get_token_labels()` - 获取 token 标签

#### `utils/mean_projection_utils.py` - 统计计算
- `compute_mean_direction_dot()` - 计算平均方向投影

### 2. 中层（core/）：组织 pipeline 步骤

中层组件组合底层函数，解决具体任务。

#### `core/hook_builder.py` - Hook 构建系统

**核心扩展点**：未来添加新的 hook 类型只需实现 `HookBuilder` 接口。

```python
# HookBuilder 协议（基类）
class HookBuilder:
    def build(self, positions_map: Dict[int, torch.Tensor]) -> Tuple[List, List]:
        """返回 (pre_hooks, post_hooks)"""
        raise NotImplementedError

# 现有实现
class AlignmentHookBuilder(HookBuilder):
    """对齐 Hook：修改激活向量使其投影等于目标值"""
    def __init__(self, model_base, direction_tensor, target_values, layer_token_map):
        ...
    def build(self, positions_map):
        ...
```


**未来扩展示例**（添加新的 hook 类型）：
```python
class ResidModificationHookBuilder(HookBuilder):
    """修改 resid 的 Hook"""
    def build(self, positions_map):
        # 实现你的逻辑
        return pre_hooks, post_hooks
```

#### `core/experiment_runner.py` - 实验执行器

封装完整的实验流程：加载资源 → 计算 baseline → 执行干预 → 保存结果。

**主要函数**：
- `load_experiment_resources()` - 加载实验所需的所有资源
- `run_single_experiment()` - 执行单个消融实验
- `run_batch_experiments()` - 批量执行消融实验（主入口）

**使用示例**：
```python
from contrast_analyze.core.experiment_runner import run_batch_experiments

run_batch_experiments(
    model_alias="llama3",
    split="vanilla",
    experiment="addition",  # 或 "ablation"
    direction_mode="directions",  # 或 "direction"
    plan_path=Path("ablation_plan.json"),
    batch_size=64,
    max_samples=4096,
    seed=2025,
)
```

#### `core/visualization_pipeline.py` - 可视化流程

统一处理两种 direction 模式的可视化。

**主要函数**：
- `compute_contrast_stats()` - 计算对比统计
- `visualize_contrast()` - 可视化对比结果（主入口）

**使用示例**：
```python
from contrast_analyze.core.visualization_pipeline import visualize_contrast

visualize_contrast(
    model_alias="llama3",
    split="vanilla",
    direction_mode="directions",  # 或 "direction"
    batch_size=128,
    skip_benign=False,
)
```

### 3. 高层（入口脚本）：一行代码完成常用功能

#### `run_ablation_addition.py` - 消融实验入口

**命令行使用**：
```bash
python run_ablation_addition.py \
    --model_alias llama3 \
    --split vanilla \
    --experiment addition \
    --direction directions \
    --plan ablation_plan.json \
    --batch_size 64 \
    --max_samples 4096
```

**Python 调用**：
```python
from contrast_analyze.run_ablation_addition import run_ablation_experiments

run_ablation_experiments(
    model_alias="llama3",
    split="vanilla",
    experiment="addition",
    direction_mode="directions",
    plan_path=Path("ablation_plan.json"),
)
```

#### `run_contrast_viz.py` - 可视化入口（统一）

**替代了** `contrast_directions.py` 和 `contrast_direction.py`，通过 `--direction_mode` 参数统一处理。

**命令行使用**：
```bash
# directions 模式（候选方向）
python run_contrast_viz.py \
    --model_alias llama3 \
    --split vanilla \
    --direction_mode directions \
    --batch_size 128

# direction 模式（全局方向）
python run_contrast_viz.py \
    --model_alias llama3 \
    --split vanilla \
    --direction_mode direction \
    --batch_size 128
```

## 文件结构

```
contrast_analyze/
├── README.md                      # 本文件
├── __init__.py
│
├── core/                          # 中层组件
│   ├── __init__.py
│   ├── hook_builder.py           # Hook 构建系统（核心扩展点）
│   ├── experiment_runner.py       # 实验执行器
│   └── visualization_pipeline.py # 可视化流程
│
├── utils/                          # 底层工具
│   ├── direction_utils.py        # 方向向量工具
│   ├── plot_utils.py             # 绘图工具
│   ├── data_utils.py             # 数据工具
│   ├── mean_projection_utils.py  # 统计计算
│   └── hook_utils.py             # 保留（向后兼容）
│
├── run_ablation_addition.py      # 消融实验入口（重构后）
├── run_contrast_viz.py           # 可视化入口（统一，替代了 contrast_directions.py 和 contrast_direction.py）
│
├── activation_cache.py            # 激活缓存（保持不变）
├── get_direction.py               # 方向生成（保持不变）
│
└── contrast_data/                 # 数据目录
    └── ...
```

## 如何扩展

### 添加新的 Hook 类型

1. 在 `core/hook_builder.py` 中创建新的 HookBuilder 类：

```python
class YourNewHookBuilder(HookBuilder):
    """你的新 Hook 类型说明"""
    
    def __init__(self, model_base, your_params):
        self.model_base = model_base
        self.your_params = your_params
    
    def build(self, positions_map):
        pre_hooks = []
        post_hooks = []
        
        # 实现你的 hook 逻辑
        for layer_idx, layer_module in enumerate(self.model_base.model_block_modules):
            def hook_fn(module, _input, output, layer_idx=layer_idx):
                # 你的干预逻辑
                return output
            
            post_hooks.append((layer_module, hook_fn))
        
        return pre_hooks, post_hooks
```

2. 在 `core/experiment_runner.py` 中使用：

```python
# 在 run_single_experiment() 中
your_hook_builder = YourNewHookBuilder(model_base, your_params)
aligned_res = compute_mean_direction_dot(
    direction=direction,
    batch_size=batch_size,
    model_base=model_base,
    dataset=dataset,
    marker=marker,
    custom_hook_builder=your_hook_builder.build,  # 使用新的 hook
    desc=f"干预统计 ({plan_name})",
)
```

### 添加新的统计方式

在 `utils/mean_projection_utils.py` 中添加新函数，或创建 `utils/stat_utils.py`：

```python
def compute_variance(direction, cache_dir, ...):
    """计算方差"""
    # 实现逻辑
    pass
```

### 添加新的可视化类型

在 `utils/plot_utils.py` 中添加新函数：

```python
def plot_custom_visualization(data, output_path):
    """自定义可视化"""
    # 实现逻辑
    pass
```

## 关键设计决策

1. **Hook 系统扩展性**：通过 `HookBuilder` 协议，添加新 hook 类型只需实现接口，不影响现有代码
2. **统一可视化**：通过 `direction_mode` 参数统一处理两种模式，消除代码重复
3. **向后兼容**：保留旧接口，确保现有代码仍可运行
4. **分层清晰**：底层/中层/高层职责明确，便于维护和扩展

## 迁移指南

### 从旧代码迁移

**新代码**（统一入口）：
```bash
python run_contrast_viz.py --model_alias llama3 --split vanilla --direction_mode directions
```

**旧代码**（`run_ablation_addition.py` 中的内联逻辑）：
- 已自动迁移到 `core/experiment_runner.py`
- 接口保持不变，可直接使用

## 常见问题

### Q: 如何选择 direction_mode？

- `directions`：使用候选方向（3D tensor，每个 token 位置有独立方向）
- `direction`：使用全局方向（1D tensor，所有位置共享同一方向）

### Q: experiment 参数的区别？

- `addition`：在 success 样本上添加 failure 的激活
- `ablation`：在 failure 样本上移除 success 的激活

### Q: 如何调试 hook？

在 `core/hook_builder.py` 的 hook 函数中添加 print 语句，或使用断点调试。

## 依赖关系

- **上游仓库**：`refusal_direction/pipeline/` - 提供模型工具和数据加载
- **本仓库**：`contrast_analyze/` - 实验 pipeline

## 注意事项

1. 保持底层函数的纯函数特性（无副作用）
2. 新增功能优先考虑复用现有底层函数
3. 扩展时遵循分层原则，不要跨层调用
4. 保持向后兼容，避免破坏现有接口


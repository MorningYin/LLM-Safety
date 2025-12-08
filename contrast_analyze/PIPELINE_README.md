# Pipeline 脚本说明

## 概述

本目录包含多个 pipeline 脚本，用于执行完整的零信任模型分析流程。

### 目录结构（精简后）
- 入口脚本仅保留 5 个阶段：
  - Stage1 `contrast_data/get_data.py`
  - Stage2 `get_direction.py`
  - Stage3 `get_projection_data.py`
  - Stage4 `analyze_drivers_effect_overlap.py`
  - Stage5 `sequential_zero_trust/main.py`
- 可复用逻辑已下沉到 `utils/`，按功能分层：
  - `core/` 环境、日志、配置（`pipeline_core.py`, `pipeline_config.py`）
  - `data/` 数据获取与打标（封装 Stage1 逻辑）
  - `direction/` 方向生成与元数据（封装 Stage2 逻辑）
  - `projection/` 投影计算（封装 Stage3 逻辑）
  - `analysis/` 驱动分析与零信任 orchestrator（封装 Stage4/5 逻辑）
  - `viz/` 通用可视化工具

## Pipeline 阶段说明

- **Stage 1**: 获取原始数据 (`get_data`)
- **Stage 2**: 生成拒答方向向量 (`get_direction`)
- **Stage 3**: 收集投影数据 (`get_projection_data`)
- **Stage 4**: 分析驱动因素 (`analyze_drivers_effect_overlap`)
- **Stage 5**: 计算零信任模型证据轨迹 (`sequential_zero_trust.main`)

## 各模型 Pipeline 脚本

### 1. Qwen-7B
- **脚本**: `run_pipeline_qwen.sh`
- **起始阶段**: Stage 1（完整流程）
- **数据模式**: 完整数据（无限制）

### 2. Llama2
- **脚本**: `run_pipeline_llama2_stage4.sh`
- **起始阶段**: Stage 4（跳过前3个阶段）
- **数据模式**: 完整数据（无限制）

### 3. Llama3
- **脚本**: `run_pipeline_llama3_stage4.sh`
- **起始阶段**: Stage 4（跳过前3个阶段）
- **数据模式**: 完整数据（无限制）

### 4. Gemma-2B-IT
- **脚本**: `run_pipeline_gemma-2b-it.sh`
- **起始阶段**: Stage 1（完整流程）
- **数据模式**: 完整数据（无限制）

### 5. Yi-6B-Chat
- **脚本**: `run_pipeline_yi-6b-chat.sh`
- **起始阶段**: Stage 1（完整流程）
- **数据模式**: 完整数据（无限制）

## 统一启动脚本

### `run_all_pipelines.sh`

统一启动所有 pipeline 脚本，按顺序执行：

1. Qwen-7B
2. Llama2 (从 Stage 4 开始)
3. Llama3 (从 Stage 4 开始)
4. Gemma-2B-IT
5. Yi-6B-Chat

**使用方法**:
```bash
# 使用默认 runs 根目录
bash run_all_pipelines.sh

# 或指定自定义 runs 根目录
export CONTRAST_ANALYZE_RUNS_ROOT=/path/to/runs
bash run_all_pipelines.sh
```

**特性**:
- 依次执行所有 pipeline
- 如果某个 pipeline 失败，会继续执行下一个
- 显示总耗时统计

## 环境变量

- `CONTRAST_ANALYZE_RUNS_ROOT`: 设置 runs 根目录（默认: `/root/autodl-tmp/contrast_analyze`）

## 注意事项

1. 确保所有模型别名已在 `model_factory.py` 中正确配置
2. 确保有足够的磁盘空间存储所有结果
3. 执行时间可能较长，建议在后台运行或使用 `nohup`/`screen`/`tmux`
4. 如果某个 pipeline 失败，可以单独运行该 pipeline 进行调试

## 单独运行 Pipeline

如果需要单独运行某个 pipeline：

```bash
# Qwen
bash run_pipeline_qwen.sh

# Llama2 (从 Stage 4 开始)
bash run_pipeline_llama2_stage4.sh

# Llama3 (从 Stage 4 开始)
bash run_pipeline_llama3_stage4.sh

# Gemma-2B-IT
bash run_pipeline_gemma-2b-it.sh

# Yi-6B-Chat
bash run_pipeline_yi-6b-chat.sh
```




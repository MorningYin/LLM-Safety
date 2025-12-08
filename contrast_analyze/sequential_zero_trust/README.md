# 层级序贯贝叶斯零信任模型 (Layer-wise Sequential Bayesian Zero-Trust Model)

## 概述

本模块实现了层级序贯贝叶斯零信任防御模型，用于检测 LLM 的越狱攻击。该模型将神经网络的深度视为时间轴，将前向传播过程视为随机过程，通过逐层累积证据来检测恶意输入。

## 核心思想

1. **假设空间**：
   - $H_0$: Benign (无害输入)
   - $H_1$: Jailbreak (越狱/有害输入)

2. **观测序列**：每一层的激活值作为观测向量

3. **概率密度函数**：对每个组件，拟合其在 Benign 和 Refusal 假设下的分布

4. **累积证据**：通过累积对数似然比 (LLR) 来积累证据

5. **不一致性检测**：检测"感知有害特征"和"准备输出无害内容"之间的解耦

## 文件结构

```
sequential_zero_trust/
├── __init__.py
├── compute_pdfs.py          # PDF 计算（支持多种分布类型）
├── layer_scores.py          # 计算 S_cons 和 S_ood
├── component_activations.py # 实时计算组件激活值
├── trajectory_compute.py    # 计算证据轨迹
├── visualization.py         # 可视化轨迹图
├── main.py                  # 主入口脚本
└── README.md                # 本文档
```

## 使用方法

### 基本用法

```bash
cd /root/refusal_direction/contrast_analyze
python -m sequential_zero_trust.main \
    --model_alias llama3 \
    --split vanilla \
    --direction_mode direction \
    --beta 1.0 \
    --batch_size 32 \
    --n_samples 100 \
    --seed 2025
```

### 参数说明

- `--model_alias`: 模型别名（如 `llama3`, `gemma-2-9b-it`）
- `--split`: 数据集分割（`vanilla` 或 `adversarial`）
- `--direction_mode`: 方向模式（`direction` 或 `directions`）
- `--driver_analysis_root`: 驱动分析结果根目录（默认：`/root/autodl-tmp/contrast_analyze_runs`）
- `--driver_analysis_type`: 驱动分析类型（默认：`driver_analysis_effect_overlap`）
- `--beta`: OOD 得分权重（默认：1.0）
- `--batch_size`: 批处理大小（默认：32）
- `--n_samples`: 每个数据集的样本数量（可选，默认使用全部）
- `--seed`: 随机种子（默认：2025）
- `--output_dir`: 输出目录（可选，默认使用 `runs/<alias>/zero_trust`）

## 输出文件

运行后会生成以下文件：

1. **trajectories.npz**: 所有数据集的证据轨迹数据
2. **layer_scores.npz**: 所有数据集的层得分数据（S_cons, S_ood, R_l）
3. **metadata.json**: 元数据（模型、参数等）
4. **evidence_trajectory.png**: 证据轨迹可视化图
5. **layer_scores_S_cons.png**: 不一致性得分对比图
6. **layer_scores_S_ood.png**: OOD 得分对比图
7. **layer_scores_R_l.png**: 总风险得分对比图

## 数学公式

### 1. PDF 计算

对于每个组件 $i$，计算：
- $f_B^{(i)}(x_i)$: Benign 假设下的 PDF
- $f_R^{(i)}(x_i)$: Refusal (fail) 假设下的 PDF

### 2. 已知分布概率

$$P_{known}^{(i)}(x_i) = 0.5 \cdot f_B^{(i)}(x_i) + 0.5 \cdot f_R^{(i)}(x_i)$$

### 3. 对数似然比

$$\lambda_i = \log \left( \frac{f_R^{(i)}(x_i)}{f_B^{(i)}(x_i)} \right)$$

### 4. 加权对数似然比

$$w_i = P_{known}^{(i)}(x_i)$$
$$\tilde{\lambda}_i = w_i \cdot \lambda_i$$

### 5. 不一致性得分

$$S_{cons} = \sum_{i=1}^N \max(0, \tilde{\lambda}_i) - \max \left( 0, \sum_{i=1}^N \tilde{\lambda}_i \right)$$

### 6. OOD 得分

$$S_{ood} = \sum_{i=1}^N - \log \left( P_{known}^{(i)}(x_i) + \epsilon \right)$$

### 7. 总风险

$$\mathcal{R}_l = S_{cons} + \beta \cdot S_{ood}$$

### 8. 证据轨迹

$$\text{Trajectory}_L = \sum_{l=0}^{L} \mathcal{R}_l$$

## 依赖要求

- Python 3.8+
- PyTorch
- NumPy
- SciPy
- Matplotlib
- 项目内部模块：
  - `pipeline.model_utils.model_factory`
  - `contrast_analyze.utils.*`

## 注意事项

1. **数据准备**：需要先运行驱动分析（`analyze_drivers_effect_overlap.py`）来生成 `valid_components.json` 和 `distribution_fits.json`

2. **内存使用**：实时计算组件激活值会占用较多内存，建议适当调整 `batch_size`

3. **计算时间**：对于大规模数据集，计算可能需要较长时间

4. **分布类型**：支持 `norm`, `t`, `gamma`, `lognorm`, `laplace`, `beta` 等分布类型

## 示例输出

运行成功后，会生成类似以下的输出：

```
============================================================
层级序贯贝叶斯零信任模型 - 证据轨迹计算
============================================================
模型: llama3
数据集分割: vanilla
方向模式: direction
Beta (OOD权重): 1.0
批处理大小: 32
随机种子: 2025
============================================================
输出目录: /root/autodl-tmp/contrast_analyze_runs/llama3/zero_trust/vanilla_direction
...
[完成] benign: 轨迹形状 (100, 80)
[完成] harmful_success: 轨迹形状 (100, 80)
[完成] harmful_failure: 轨迹形状 (100, 80)
...
所有计算完成！
============================================================
```

## 参考文献

该实现基于层级序贯贝叶斯零信任模型的数学框架，将神经网络的深度视为时间轴，通过逐层累积证据来检测越狱攻击。





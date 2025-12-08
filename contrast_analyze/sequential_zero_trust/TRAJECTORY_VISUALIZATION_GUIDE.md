# 证据轨迹图（Evidence Trajectory）绘制方法详解

## 图表概述

`evidence_trajectory.png` 展示了不同数据集（benign、harmful_success、harmful_failure 等）在模型各层的累计风险得分（Evidence Trajectory）的变化趋势。

每个数据集都显示为：
- **一条主线**：表示该数据集的中位数轨迹
- **一个范围**：表示该数据集的 25%-75% 分位数区间（IQR，四分位距）

---

## 数据准备

### 输入数据格式

```python
trajectories: Dict[str, np.ndarray]
# 键：数据集名称（如 'benign', 'harmful_success', 'harmful_failure'）
# 值：轨迹数组，形状为 (N_samples, n_layers)
#     每一行是一个样本在所有层的累计分数
#     每一列是一层上所有样本的累计分数
```

**示例**：
```python
trajectories = {
    'benign': np.array([
        [0.1, 0.3, 0.5, 0.8, ...],  # 样本1的轨迹
        [0.2, 0.4, 0.6, 0.9, ...],  # 样本2的轨迹
        ...
    ]),  # 形状: (N_benign, n_layers)
    'harmful_success': np.array([
        [0.5, 1.2, 2.1, 3.5, ...],  # 样本1的轨迹
        [0.6, 1.3, 2.3, 3.8, ...],  # 样本2的轨迹
        ...
    ]),  # 形状: (N_harmful, n_layers)
    ...
}
```

---

## 绘制方法详解

### 1. 计算统计量

对每个数据集，计算以下统计量：

```python
# 对每一层（列）计算统计量
median_traj = np.median(trajectory, axis=0)  # 中位数（主线）
q25_traj = np.percentile(trajectory, 25, axis=0)  # 25% 分位数（下边界）
q75_traj = np.percentile(trajectory, 75, axis=0)  # 75% 分位数（上边界）
```

**说明**：
- `axis=0` 表示沿着样本维度（行）计算，即对每一层的所有样本值进行统计
- **中位数（median）**：更稳健的集中趋势度量，对异常值不敏感
- **分位数（percentile）**：表示数据的分布范围

**为什么使用中位数而不是均值？**
- 中位数对异常值更稳健
- 如果数据中有极端值，均值会被拉偏，但中位数不受影响
- 更适合展示数据的典型趋势

**为什么使用 25%-75% 分位数？**
- 这是**四分位距（IQR, Interquartile Range）**
- 包含了中间 50% 的数据
- 比标准差更稳健，不受极端值影响

---

### 2. 绘制主线

使用 `matplotlib.pyplot.plot()` 绘制中位数线：

```python
ax.plot(
    layers,                    # X轴：层索引 [0, 1, 2, ..., n_layers-1]
    median_traj,               # Y轴：中位数轨迹值
    label=label,                # 图例标签
    color=style['color'],       # 颜色（如 '#0066CC' 蓝色）
    linestyle=style['linestyle'],  # 线型（如 '-' 实线, '--' 虚线）
    linewidth=style['linewidth'],  # 线宽（3）
    marker=style.get('marker', None),  # 标记（如 'o' 圆圈）
    markersize=style.get('markersize', 0),  # 标记大小（4）
    markevery=style.get('markevery', None),  # 标记间隔（每4个点）
    alpha=0.95,                 # 透明度
    zorder=3                    # 图层顺序（主线在上层）
)
```

**参数说明**：
- `layers`：层索引数组，如 `[0, 1, 2, ..., 79]`（对于 80 层模型）
- `median_traj`：每一层的中位数，形状为 `(n_layers,)`
- `zorder=3`：确保主线显示在填充区域之上

---

### 3. 绘制范围（填充区域）

使用 `matplotlib.pyplot.fill_between()` 绘制分位数区间：

```python
ax.fill_between(
    layers,                    # X轴：层索引
    q25_traj,                  # 下边界：25% 分位数
    q75_traj,                  # 上边界：75% 分位数
    color=style['color'],       # 填充颜色（与主线相同）
    alpha=0.25,                 # 透明度（较淡，不遮挡主线）
    zorder=1                    # 图层顺序（填充在底层）
)
```

**参数说明**：
- `q25_traj`：每一层的 25% 分位数，形状为 `(n_layers,)`
- `q75_traj`：每一层的 75% 分位数，形状为 `(n_layers,)`
- `alpha=0.25`：填充区域透明度为 25%，不会遮挡主线
- `zorder=1`：确保填充区域显示在主线之下

**视觉效果**：
- 填充区域表示数据的**不确定性**或**变异性**
- 范围越宽，说明样本间的差异越大
- 范围越窄，说明样本间的差异越小

---

## 完整代码示例

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_evidence_trajectory(
    trajectories: Dict[str, np.ndarray],
    output_path: Path,
    n_layers: int
):
    """
    绘制证据轨迹图。
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    layers = np.arange(n_layers)  # [0, 1, 2, ..., n_layers-1]
    
    for dataset_name, trajectory in trajectories.items():
        # 1. 计算统计量
        median_traj = np.median(trajectory, axis=0)  # 中位数（主线）
        q25_traj = np.percentile(trajectory, 25, axis=0)  # 25% 分位数
        q75_traj = np.percentile(trajectory, 75, axis=0)  # 75% 分位数
        
        # 2. 绘制填充区域（范围）
        ax.fill_between(
            layers,
            q25_traj,
            q75_traj,
            color=style['color'],
            alpha=0.25,  # 透明度
            zorder=1     # 底层
        )
        
        # 3. 绘制主线（中位数）
        ax.plot(
            layers,
            median_traj,
            label=label,
            color=style['color'],
            linestyle=style['linestyle'],
            linewidth=3,
            marker='o',
            markersize=4,
            markevery=4,
            alpha=0.95,
            zorder=3    # 上层
        )
    
    # 设置标签和标题
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Cumulative Risk Score (Evidence Trajectory)')
    ax.set_title('Evidence Trajectory: Layer-wise Sequential Bayesian Zero-Trust Model')
    
    # 设置图例
    ax.legend(loc='best')
    
    # 保存
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
```

---

## 数据流程示例

假设有 3 个样本，5 层模型：

```python
# 输入数据
trajectory = np.array([
    [0.1, 0.3, 0.5, 0.8, 1.2],  # 样本1
    [0.2, 0.4, 0.6, 0.9, 1.3],  # 样本2
    [0.15, 0.35, 0.55, 0.85, 1.25],  # 样本3
])  # 形状: (3, 5)

# 计算统计量（对每一列）
median_traj = np.median(trajectory, axis=0)
# 结果: [0.15, 0.35, 0.55, 0.85, 1.25]

q25_traj = np.percentile(trajectory, 25, axis=0)
# 结果: [0.125, 0.325, 0.525, 0.825, 1.225]

q75_traj = np.percentile(trajectory, 75, axis=0)
# 结果: [0.175, 0.375, 0.575, 0.875, 1.275]

# 绘制
layers = [0, 1, 2, 3, 4]
ax.plot(layers, median_traj, ...)  # 主线
ax.fill_between(layers, q25_traj, q75_traj, ...)  # 范围
```

---

## 样式配置

不同数据集使用不同的颜色和线型：

```python
styles = {
    'benign': {
        'color': '#0066CC',      # 蓝色
        'linestyle': '-',         # 实线
        'linewidth': 3,
        'marker': 'o',            # 圆圈标记
        'markersize': 4,
        'markevery': 4            # 每4个点标记一次
    },
    'harmful_success': {
        'color': '#00AA00',       # 绿色
        'linestyle': '--',        # 虚线
        'linewidth': 3,
        'marker': 's',            # 方形标记
        'markersize': 4,
        'markevery': 4
    },
    'harmful_failure': {
        'color': '#FF6600',       # 橙色
        'linestyle': '-.',        # 点划线
        'linewidth': 3,
        'marker': '^',            # 三角形标记
        'markersize': 4,
        'markevery': 4
    }
}
```

---

## 如何解读图表

### 1. 主线（中位数）

- **位置**：表示该数据集在每一层的**典型**累计分数
- **趋势**：如果主线上升，说明累计风险在增加
- **比较**：不同数据集的主线位置差异，表示它们之间的风险差异

### 2. 范围（25%-75% 分位数）

- **宽度**：表示样本间的**变异性**
  - 范围越宽，说明样本间差异越大
  - 范围越窄，说明样本间差异越小
- **位置**：表示大多数样本（中间 50%）的分布范围

### 3. 分离度

- **理想情况**：不同数据集的主线和范围分离明显
  - 例如：benign 的主线在下方，harmful_success 的主线在上方
- **重叠情况**：如果范围重叠，说明两类样本难以区分

### 4. 趋势分析

- **上升趋势**：累计风险随层数增加而增加
- **下降趋势**：累计风险随层数增加而减少
- **平稳趋势**：累计风险在各层保持稳定

---

## 常见问题

### Q1: 为什么使用中位数而不是均值？

**A**: 中位数对异常值更稳健。如果数据中有极端值，均值会被拉偏，但中位数不受影响。

**示例**：
```python
data = [1, 2, 3, 4, 100]  # 有一个异常值 100
mean = np.mean(data)      # 22.0（被异常值拉偏）
median = np.median(data)  # 3.0（不受影响）
```

### Q2: 为什么使用 25%-75% 分位数而不是标准差？

**A**: 分位数更稳健，不受极端值影响。标准差会受异常值影响，导致范围过大。

**示例**：
```python
data = [1, 2, 3, 4, 100]  # 有一个异常值 100
std = np.std(data)        # 43.7（受异常值影响）
q25, q75 = np.percentile(data, [25, 75])  # 1.5, 4.0（稳健）
```

### Q3: 如何理解填充区域？

**A**: 填充区域表示数据的**不确定性**或**变异性**：
- 范围越宽，说明样本间差异越大
- 范围越窄，说明样本间差异越小
- 如果两个数据集的范围重叠，说明它们难以区分

### Q4: 如果数据量很少（如只有 10 个样本），分位数还准确吗？

**A**: 分位数仍然有效，但可能不够稳定。建议：
- 至少需要 3 个样本才能计算分位数
- 样本量越多，分位数越稳定
- 如果样本量很少，可以考虑使用标准差或置信区间

### Q5: 如何修改范围（如使用 10%-90% 分位数）？

**A**: 修改 `percentile` 的参数：

```python
# 使用 10%-90% 分位数（更宽的范围）
q10_traj = np.percentile(trajectory, 10, axis=0)
q90_traj = np.percentile(trajectory, 90, axis=0)

ax.fill_between(layers, q10_traj, q90_traj, ...)
```

---

## 总结

证据轨迹图的绘制方法：

1. **输入**：每个数据集的轨迹数组 `(N_samples, n_layers)`
2. **计算统计量**：
   - 中位数（主线）：`np.median(trajectory, axis=0)`
   - 25% 分位数（下边界）：`np.percentile(trajectory, 25, axis=0)`
   - 75% 分位数（上边界）：`np.percentile(trajectory, 75, axis=0)`
3. **绘制**：
   - 使用 `ax.fill_between()` 绘制填充区域（范围）
   - 使用 `ax.plot()` 绘制主线（中位数）
4. **样式**：不同数据集使用不同的颜色、线型和标记

这种可视化方法能够清晰地展示：
- 每个数据集的典型趋势（主线）
- 样本间的变异性（范围）
- 不同数据集之间的差异（分离度）


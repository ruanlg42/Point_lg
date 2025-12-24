# 数据打包工具使用说明

## `data_transfer_multipoint_flexible.py`

支持任意列数的温度数据打包工具（4列、13列或更多）。

### 主要特性

1. **自动列数检测**：自动识别txt文件中的温度列数
2. **灵活配置**：支持手动指定列数
3. **错误处理**：跳过有问题的文件并给出提示
4. **参数编码**：使用 `thickness + distance序号` 区分不同测温点

### 使用方法

#### 方法1：直接运行（使用默认配置）

```bash
python data_transfer_multipoint_flexible.py
```

#### 方法2：修改配置后运行

编辑脚本中的配置区域：

```python
# ==================== 配置区域 ====================
SOURCE_DIR = '/path/to/your/source/directory'  # 源数据目录
OUTPUT_DIR = '/path/to/your/output/directory'  # 输出目录

SKIP_LINES = 8  # 跳过的注释行数

AUTO_DETECT = True  # 自动检测列数（推荐）

NUM_TEMP_COLUMNS = 4  # 如果AUTO_DETECT=False，手动指定列数
# ================================================
```

#### 方法3：作为模块导入

```python
from data_transfer_multipoint_flexible import pack_thermal_dataset_multipoint

# 打包4列温度数据
pack_thermal_dataset_multipoint(
    source_dir='/path/to/4column/data',
    output_dir='/path/to/output',
    skip_lines=8,
    auto_detect_columns=True
)

# 打包13列温度数据
pack_thermal_dataset_multipoint(
    source_dir='/path/to/13column/data',
    output_dir='/path/to/output',
    skip_lines=8,
    auto_detect_columns=True
)

# 或手动指定列数
pack_thermal_dataset_multipoint(
    source_dir='/path/to/data',
    output_dir='/path/to/output',
    skip_lines=8,
    auto_detect_columns=False,
    num_temp_columns=13
)
```

### 输入文件格式

#### 必需文件

每个样本需要两个文件：

1. **`{sample_id}_parameters.yaml`** - 材料参数文件
   ```yaml
   Lambda: 9.623
   T0: 293.15
   T1: 310.15
   c: 544.0
   e: 4853.566
   p: 4500.0
   thickness: 0.3
   time: 5
   material: "titanium alloy"
   ```

2. **`{sample_id}_mph.txt`** - 温度数据文件
   ```
   % Model:              thermal_design_point_30hz.mph
   % Version:            COMSOL 6.2.0.290
   ...（共8行注释）
   时间        温度1       温度2       温度3       ...       温度N
   0.000      310.15     310.15     310.15    ...     310.15
   0.033      310.05     310.05     310.05    ...     310.05
   ...
   ```

### 输出文件

生成 `thermal_dataset_multipoint.pth`，包含：

```python
{
    'time': torch.Tensor,          # [N*M, T] 时间序列
    'temperature': torch.Tensor,    # [N*M, T] 温度序列
    'parameters': torch.Tensor,     # [N*M, 8] 参数向量
    'material_indices': torch.Tensor,  # [N*M] 材料索引
    'material_to_idx': dict,        # 材料名称到索引的映射
    'parameter_names': list,        # 参数名称列表
    'num_temp_columns': int         # 温度列数
}
```

其中：
- `N` = 原始样本数
- `M` = 每个样本的测温点数（列数）
- `T` = 时间步数

### 参数编码规则

对于每个原始样本的M个测温点，`thickness`参数编码为：

```python
thickness_encoded = original_thickness + distance

# 示例：原始thickness=0.3，有4个测温点
# distance=0 → thickness=0.3
# distance=1 → thickness=1.3
# distance=2 → thickness=2.3
# distance=3 → thickness=3.3

# 示例：原始thickness=0.3，有13个测温点
# distance=0  → thickness=0.3
# distance=1  → thickness=1.3
# ...
# distance=12 → thickness=12.3
```

### 数据扩展示例

#### 输入：4列温度数据
- 1000个样本 × 4列 = 4000个训练样本

#### 输入：13列温度数据
- 1000个样本 × 13列 = 13000个训练样本

### 注意事项

1. **thickness编码冲突**：
   - 如果原始thickness范围很大（如0.1-10），可能与编码后的thickness重叠
   - 建议检查原始thickness范围，确保 `max_thickness + max_distance < min_thickness_in_next_group`

2. **参数一致性**：
   - 同一原始样本的M个测温点具有相同的材料参数（e, T0, T1等）
   - 只有temperature和thickness不同

3. **数据验证**：
   - 打包后应验证：每M个连续样本来自同一原始样本
   - 可通过检查 `e`, `T0`, `T1` 等参数是否相同来验证

### 常见问题

**Q: 如何验证打包是否正确？**

```python
import torch

data = torch.load('thermal_dataset_multipoint.pth')
print(f"总样本数: {len(data['temperature'])}")
print(f"测温点数: {data['num_temp_columns']}")
print(f"预期: {len(data['temperature'])} = 原始样本数 × {data['num_temp_columns']}")

# 检查前N个样本的参数（应该来自同一原始样本）
N = data['num_temp_columns']
param_names = data['parameter_names']
e_idx = param_names.index('e')

for i in range(N):
    e = data['parameters'][i, e_idx].item()
    thickness = data['parameters'][i, 6].item()
    print(f"样本{i}: e={e:.2f}, thickness={thickness:.2f}")
```

**Q: 可以混合不同列数的数据吗？**

不可以。一个PTH文件只能包含相同列数的数据。如果有不同列数的数据，需要分别打包。

**Q: 13列数据会不会太大？**

13列 = 13倍的数据量，但这是数据增强的一种方式。如果内存不足，可以：
- 增加 `batch_size` 的限制
- 使用数据加载器的 `num_workers` 参数
- 分批处理

### 示例脚本

**打包4列数据：**
```python
from data_transfer_multipoint_flexible import pack_thermal_dataset_multipoint

pack_thermal_dataset_multipoint(
    source_dir='./data_4columns',
    output_dir='./packed_4columns',
    skip_lines=8,
    auto_detect_columns=True
)
```

**打包13列数据：**
```python
from data_transfer_multipoint_flexible import pack_thermal_dataset_multipoint

pack_thermal_dataset_multipoint(
    source_dir='./data_13columns',
    output_dir='./packed_13columns',
    skip_lines=8,
    auto_detect_columns=True
)
```

### 与训练脚本的兼容性

此工具生成的PTH文件与现有的训练脚本完全兼容：

- `dataset.py` 正常加载
- `train.py` 正常训练
- `predict_from_txt.py` 正常预测

唯一的区别是数据量增加了（M倍），训练时间会相应增加。


# Point_lg - 热传导参数预测深度学习框架

## 项目简介

Point_lg 是一个基于深度学习的热传导参数预测系统,专门用于从温度时序数据中预测材料的热扩散系数(thermal effusivity, e)。该项目实现了多种先进的时序模型架构,包括 Transformer、Mamba、VAE 等,并融入了物理先验知识以提升预测精度。

## 核心特性

### 🔥 多样化的模型架构

项目实现了 10+ 种深度学习模型,涵盖以下类型:

1. **经典架构**
   - Transformer: 基于自注意力机制的时序模型
   - CNN1D: 一维卷积神经网络

2. **物理先验增强模型**
   - PhysicsInformedCNN: 融入物理特征的 CNN (4通道输入: T, Ṫ, t^(-1/2), Δt)
   - PhysicsInformedTransformer: 物理增强 Transformer
   - EnhancedPhysicsTransformer: 增强版,包含残差连接和自适应时序注意力池化

3. **高效 Mamba 架构** (O(L) 时间复杂度 vs Transformer 的 O(L²))
   - MambaPhysicsModel: 基于状态空间模型的高效架构
   - EnhancedMambaPhysicsModel: 增强版 Mamba,支持多尺度特征融合
   - HybridMambaTransformer: Mamba + Transformer 混合架构

4. **变分自编码器 (VAE) 系列**
   - TimeVAE1D_Mamba: 基于 Mamba 的时序 VAE
   - TimeVAE1D_Transformer: 基于 Transformer 的时序 VAE
   - TimeVAE1D_HybridMambaTransformer: 混合架构 VAE
   - **TimeVAE1D_HybridMambaTransformer_Residual**: ⭐ 残差重建 VAE (推荐)
   - TimeVAE1D_StageAware: 时序分段感知 VAE
   - TimeVAE1D_SSTEncoder_Residual: 基于 SST 编码器的 VAE

5. **其他架构**
   - PhysTCN: 物理信息时序卷积网络
   - TimeVAE1D_Mamba_PhysicsDecoder: 物理基函数解码器 VAE

### 📊 智能数据处理

- **自适应数据打包**: 支持任意数量的测温点 (4列、13列或更多)
- **多点独立采样**: 每个测温点作为独立样本,通过 `thickness+序号` 区分
- **自动归一化**: 温度归一化 + log(e) 变换处理偏斜分布
- **物理特征工程**: 
  - 高斯滤波平滑 (sigma=2)
  - 4通道物理特征: T_tilde (相对温度), T_dot (温度导数), t^(-1/2) (物理先验), Δt (采样间隔)

### 🎯 高级训练特性

- **分段加权损失**: 针对不同 e 值范围采用不同权重,防止异常高值
- **VAE 损失函数**: 重建损失 + KL散度 + 参数预测损失 + 初值约束
- **多 GPU 训练**: 自动检测并使用多个 GPU (支持 DataParallel)
- **学习率调度**: ReduceLROnPlateau 自适应调整
- **梯度裁剪**: 防止梯度爆炸
- **Checkpoint 保存**: 自动保存最佳模型和配置

### 🚀 灵活的预测接口

- **从文本文件预测**: 支持多种格式的温度数据文件
- **批量 GIF 预测**: 从热成像 GIF 中提取温度并预测
- **自动插值**: 处理不同时间步长的数据
- **可视化输出**: 生成预测结果图表和对比数据

## 项目结构

```
Point_lg/
├── config.yaml                          # 主配置文件
├── model.py                             # 所有模型架构定义 (4326行)
├── dataset.py                           # 数据集加载器
├── train.py                             # 训练脚本 (1578行)
├── predict_from_txt.py                  # 从txt文件预测
├── predict_gif.py                       # 从GIF文件预测 (964行)
├── data_transfer_multipoint_flexible.py # 数据打包工具
├── plot_temperature_curves.py           # 温度曲线可视化
├── plot_quick.py                        # 快速绘图工具
├── test_4columns_predict.py             # 4列数据预测测试
├── gif_test/                            # GIF测试数据
│   ├── newdata_30hz/                    # 30Hz采样数据
│   ├── newdata_30hz_vaehybrid/          # VAE混合模型数据
│   ├── newdata_a/, newdata_aa/          # 测试数据集
│   ├── newnewnew_30hz/                  # 新30Hz数据
│   ├── newnewnew_shape/                 # 形状数据
│   └── past_data/                       # 历史数据
├── test_code/                           # 测试和分析脚本
│   ├── analyze_data_mismatch.py
│   ├── check_training_data.py
│   ├── compare_filters.py
│   ├── debug_pth_data.py
│   ├── grid_search.py
│   └── vae_*.py                         # VAE相关测试
├── comparison_results/                  # 对比结果
└── results/                             # 训练结果输出目录
```

## 快速开始

### 环境要求

```bash
# 核心依赖
Python >= 3.8
PyTorch >= 1.10
CUDA >= 11.0 (可选,用于GPU加速)

# 必需库
pip install torch torchvision
pip install pyyaml numpy scipy pillow matplotlib tqdm
pip install mamba-ssm  # Mamba模型必需
pip install tensorboard  # 训练可视化
```

### 1. 数据准备

将原始数据打包为 PyTorch 格式:

```bash
python data_transfer_multipoint_flexible.py
```

数据格式要求:
- `*_parameters.yaml`: 包含材料参数 (Lambda, T0, T1, c, e, p, thickness, time)
- `*_mph.txt`: 温度时序数据 (时间 + N列温度)

### 2. 配置模型

编辑 `config.yaml` 文件:

```yaml
# 任务配置
task:
  prefix: AAA_Final_setup        # 任务前缀
  suffix: null                   # 任务后缀

# 数据配置
data:
  dataset_path: /path/to/thermal_dataset_multipoint.pth
  normalize_temp: true           # 温度归一化
  use_log_e: true                # e值对数变换
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  batch_size: 4096
  total_time: 5.0                # 总采样时间(秒)
  delta_t: 0.03333333333333333   # 采样间隔(秒)
  seq_len: 151                   # 序列长度

# 模型配置 (选择一个)
model:
  type: timevae1d_hybrid_mamba_transformer_residual  # 推荐
  timevae1d_hybrid_mamba_transformer_residual:
    C_in: 1                      # 输入通道数
    latent_dim: 256              # 隐空间维度
    d_model: 256                 # 模型维度
    n_mamba: 3                   # Mamba层数
    n_transformer: 3             # Transformer层数
    nhead: 4                     # 注意力头数
    d_state: 16                  # SSM状态维度
    d_conv: 4                    # 局部卷积宽度
    expand: 2                    # 扩展因子
    dropout: 0.2                 # Dropout率
    decoder_base: 128            # 解码器基础通道数
    lambda_ic: 0.5               # 初值约束权重

# 损失函数配置
loss:
  threshold_1: 10000.0           # 第一阈值
  threshold_2: 30000.0           # 第二阈值
  low_weight: 1                  # 低值权重
  mid_weight: 1                  # 中值权重
  high_weight: 1                 # 高值权重

# 训练配置
training:
  epochs: 1000
  learning_rate: 0.001
  weight_decay: 0.0001
  device: cuda
  seed: 42
```

### 3. 训练模型

```bash
# 单GPU训练
CUDA_VISIBLE_DEVICES=0 python train.py

# 多GPU训练
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py
```

训练过程会自动:
- 生成任务名称 (前缀_时间戳_模型参数)
- 保存检查点到 `results/model_result/任务名称/`
- 记录训练日志和 TensorBoard 数据
- 保存最佳模型和配置文件

### 4. 预测

#### 从文本文件预测

编辑 `predict_from_txt.py` 中的路径:

```python
TXT_PATH = '/path/to/temperature_data.txt'
MODEL_PATH = '/path/to/best_model.pth'
CONFIG_PATH = '/path/to/config_used.yaml'
```

运行预测:

```bash
python predict_from_txt.py
```

输出示例:
```
PREDICTED THERMAL EFFUSIVITY: 5678.45 J·m⁻²·K⁻¹·s⁻¹/²
```

#### 从 GIF 预测

```bash
python predict_gif.py \
  --gif_dir /path/to/gif_folder \
  --model_path /path/to/best_model.pth \
  --config_path /path/to/config_used.yaml \
  --output_dir /path/to/output
```

## 模型架构详解

### TimeVAE1D_HybridMambaTransformer_Residual (推荐)

这是当前性能最优的模型架构,采用了以下创新:

**架构特点:**
1. **混合编码器**: 底层使用 Mamba (O(L) 复杂度)快速建模,顶层使用 Transformer 进行全局精细化
2. **残差重建**: 通过预测残差而非绝对温度,提升重建精度
3. **初值约束**: 添加 λ_ic 损失项,确保残差在初始时刻接近0
4. **高斯滤波预处理**: 平滑输入序列,减少噪声干扰

**前向流程:**
```
输入 [B, 1, T] 
  ↓ 高斯滤波 (sigma=2)
  ↓ 1D Conv [B, d_model, T]
  ↓ Mamba Layers (n_mamba) [B, d_model, T]
  ↓ Transformer Layers (n_transformer) [B, d_model, T]
  ↓ Pooling → [B, d_model]
  ↓ FC → [B, latent_dim*2] → (μ, log_σ²)
  ↓ Reparameterization → z [B, latent_dim]
  ↓ Decoder (Conv1DTranspose) → Δx [B, 1, T] (残差)
  ↓ x_recon = x_smooth + Δx
  └ Parameter Head → e [B, 1]
```

**损失函数:**
```
L_total = L_recon + β·L_KL + λ_e·L_e + λ_ic·L_ic

其中:
- L_recon = MSE(x_recon, x_true)
- L_KL = -0.5·Σ(1 + log_σ² - μ² - σ²)
- L_e = WeightedMSE(e_pred, e_true)
- L_ic = |Δx[0]|  (残差初值约束)
```

### Mamba vs Transformer

| 特性 | Mamba | Transformer |
|------|-------|-------------|
| 时间复杂度 | O(L) | O(L²) |
| 空间复杂度 | O(L·d) | O(L²+L·d) |
| 长序列处理 | ✅ 高效 | ❌ 慢 |
| 全局依赖 | ✅ SSM | ✅ 注意力 |
| 并行化 | ⚠️ 部分 | ✅ 完全 |

**混合架构优势**: 结合 Mamba 的高效性和 Transformer 的全局建模能力

## 高级功能

### 1. 自定义物理特征

在 `model.py` 中修改 `_preprocess_sequence` 方法:

```python
def _preprocess_sequence(self, temp_sequence):
    # 添加自定义物理特征
    # Channel 1: T_tilde (相对温度)
    T_tilde = T_smooth - T_smooth[:, 0:1]
    
    # Channel 2: T_dot (温度导数)
    T_dot = (T_smooth[:, 1:] - T_smooth[:, :-1]) / self.delta_t
    
    # Channel 3: 自定义特征 (如 t^(-1/2))
    t_inv_sqrt = 1.0 / torch.sqrt(time_steps)
    
    # Channel 4: 其他特征
    # ...
    
    return torch.stack([...], dim=1)  # [batch, C, seq_len]
```

### 2. 分段加权损失

针对 e 值分布的长尾特性,采用分段权重:

```python
# config.yaml
loss:
  threshold_1: 10000.0    # e < 10000 (常见值)
  threshold_2: 30000.0    # e ≥ 30000 (异常高值)
  low_weight: 1           # 常见值权重
  mid_weight: 1           # 中等值权重
  high_weight: 1          # 异常值权重 (可降低以减少拟合)
```

这样可以防止模型过拟合到极端高 e 值样本。

### 3. 多模型集成

```python
# 加载多个模型
models = [
    load_model(model_path_1, config_path_1, device),
    load_model(model_path_2, config_path_2, device),
    load_model(model_path_3, config_path_3, device),
]

# 集成预测
predictions = []
for model, dataset, config in models:
    pred = predict_effusivity(model, temp_seq, dataset, device)
    predictions.append(pred)

# 平均或加权平均
final_pred = np.mean(predictions)
# 或 final_pred = np.average(predictions, weights=[0.5, 0.3, 0.2])
```

### 4. 不确定性量化 (VAE 模型)

VAE 模型天然支持不确定性估计:

```python
# 多次采样
n_samples = 100
predictions = []

model.eval()
with torch.no_grad():
    for _ in range(n_samples):
        recon, e_pred, (mu, logvar), x_smooth = model(temp_tensor)
        predictions.append(e_pred.item())

# 统计
mean_pred = np.mean(predictions)
std_pred = np.std(predictions)
conf_interval = (
    mean_pred - 1.96 * std_pred,  # 95%置信区间下限
    mean_pred + 1.96 * std_pred   # 95%置信区间上限
)

print(f"预测: {mean_pred:.2f} ± {std_pred:.2f}")
print(f"95%置信区间: [{conf_interval[0]:.2f}, {conf_interval[1]:.2f}]")
```

## 性能优化

### 训练加速

1. **增大 batch_size** (在显存允许的情况下)
   ```yaml
   data:
     batch_size: 8192  # 默认4096
   ```

2. **减少模型复杂度**
   ```yaml
   model:
     timevae1d_hybrid_mamba_transformer_residual:
       d_model: 128     # 从256减小
       n_mamba: 2       # 从3减小
       n_transformer: 2 # 从3减小
   ```

3. **使用混合精度训练** (需要 PyTorch >= 1.6)
   ```python
   from torch.cuda.amp import autocast, GradScaler
   
   scaler = GradScaler()
   
   with autocast():
       loss = model(...)
   
   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()
   ```

### 推理加速

1. **模型量化** (INT8)
   ```python
   quantized_model = torch.quantization.quantize_dynamic(
       model, {nn.Linear}, dtype=torch.qint8
   )
   ```

2. **ONNX 导出**
   ```python
   dummy_input = torch.randn(1, 1, 151).to(device)
   torch.onnx.export(
       model, dummy_input, "model.onnx",
       input_names=['temperature'],
       output_names=['effusivity'],
       dynamic_axes={'temperature': {0: 'batch_size'}}
   )
   ```

## 常见问题

### Q1: 训练时显存不足

**解决方案:**
1. 减小 `batch_size`
2. 减小模型 `d_model` 或层数
3. 使用梯度累积:
   ```python
   accumulation_steps = 4
   for i, batch in enumerate(dataloader):
       loss = model(...) / accumulation_steps
       loss.backward()
       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   ```

### Q2: 预测值异常高 (e > 30000)

**可能原因:**
1. 模型过拟合到异常样本
2. 温度数据未正确归一化
3. 序列长度不匹配

**解决方案:**
1. 调整分段权重,降低 `high_weight`
2. 检查数据预处理流程
3. 确保输入序列插值到正确长度

### Q3: VAE 重建效果差

**可能原因:**
1. `latent_dim` 过小,信息瓶颈
2. `beta` (KL权重) 过大,后验坍缩
3. 训练不充分

**解决方案:**
1. 增大 `latent_dim` (64 → 128 → 256)
2. 降低 `beta` (0.01 → 0.001)
3. 增加训练轮次或降低学习率

### Q4: 多GPU训练时出错

**检查项:**
1. 确认所有GPU可见: `echo $CUDA_VISIBLE_DEVICES`
2. 检查显存是否充足: `nvidia-smi`
3. 确保 PyTorch 编译时启用了 CUDA

**调试命令:**
```bash
# 单GPU测试
CUDA_VISIBLE_DEVICES=0 python train.py

# 逐步增加GPU数量
CUDA_VISIBLE_DEVICES=0,1 python train.py
```

## 项目贡献

### 模型开发者

- **Transformer 系列**: TimeTransformer, PhysicsInformedTransformer, EnhancedPhysicsTransformer
- **Mamba 系列**: MambaPhysicsModel, EnhancedMambaPhysicsModel, HybridMambaTransformer
- **VAE 系列**: TimeVAE1D_Mamba, TimeVAE1D_HybridMambaTransformer_Residual, TimeVAE1D_StageAware

### 引用

如果本项目对您的研究有帮助,请考虑引用:

```bibtex
@software{point_lg_2024,
  title = {Point_lg: Deep Learning Framework for Thermal Parameter Prediction},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/Point_lg}
}
```

## 许可证

本项目采用 MIT 许可证。详见 `LICENSE` 文件。

## 联系方式

- 作者: [您的姓名]
- Email: [您的邮箱]
- GitHub: [项目地址]

## 更新日志

### v1.0.0 (2024-12-24)
- ✅ 初始版本发布
- ✅ 实现 10+ 种模型架构
- ✅ 支持多GPU训练
- ✅ 完善的数据预处理流程
- ✅ 分段加权损失函数
- ✅ VAE 残差重建机制
- ✅ GIF 批量预测功能

---

**Happy Modeling! 🚀**

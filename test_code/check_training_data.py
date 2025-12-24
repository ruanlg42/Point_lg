"""检查训练数据集中4个测温点的特征"""
import torch
import numpy as np

# 加载数据集
data_path = '/home/ziwu/Newpython/lg_exp/Point_lg/data_source/output_5s_h10_h300_5s_30hz_9temp/thermal_design_point_nogif_30hz_packed/thermal_dataset_multipoint.pth'
data = torch.load(data_path)

print("="*70)
print("检查训练数据集中4个测温点的特征分布")
print("="*70)
print(f"总样本数: {data['temperature'].shape[0]}")
print(f"参数: {data['parameter_names']}")
print()

# 检查前16个样本（4个原始样本 × 4个点）
print("前16个样本的参数:")
print("样本ID | e值      | thickness | 温度均值   | 温度标准差 | 温度范围")
print("-" * 80)

for i in range(min(16, data['temperature'].shape[0])):
    params = data['parameters'][i]
    temps = data['temperature'][i]
    e = params[4].item()
    thickness = params[6].item()
    temp_mean = temps.mean().item()
    temp_std = temps.std().item()
    temp_min = temps.min().item()
    temp_max = temps.max().item()
    
    # 从thickness推算distance（旧编码方案：0.3, 1.3, 2.3, 3.3）
    distance = int(round(thickness)) - int(round(thickness * 0.1) * 10)
    if thickness < 1.0:
        distance = 0
    elif 1.0 <= thickness < 2.0:
        distance = 1
    elif 2.0 <= thickness < 3.0:
        distance = 2
    else:
        distance = 3
    
    print(f"{i:6d} | {e:8.2f} | {thickness:9.2f} (d={distance}) | {temp_mean:10.2f} | {temp_std:10.4f} | [{temp_min:.2f}, {temp_max:.2f}]")

print()
print("="*70)
print("按distance分组统计温度特征")
print("="*70)

# 分析所有样本，按distance分组
thickness_values = data['parameters'][:, 6].numpy()

# 正确推算distance
distances = np.zeros(len(thickness_values), dtype=int)
for i, thick in enumerate(thickness_values):
    if thick < 1.0:
        distances[i] = 0
    elif 1.0 <= thick < 2.0:
        distances[i] = 1
    elif 2.0 <= thick < 3.0:
        distances[i] = 2
    else:
        distances[i] = 3

for dist in range(4):
    mask = distances == dist
    temps_group = data['temperature'][mask]
    
    print(f"\nDistance {dist} (共{mask.sum()}个样本):")
    
    if mask.sum() > 0:
        print(f"  温度均值的均值: {temps_group.mean(dim=1).mean():.4f} K")
        print(f"  温度标准差的均值: {temps_group.std(dim=1).mean():.4f} K")
        print(f"  温度范围: [{temps_group.min():.4f}, {temps_group.max():.4f}] K")
        
        # 检查温度序列的形状特征（前10个时间步的平均变化）
        first_10_mean = temps_group[:, :10].mean(dim=0)
        print(f"  前10个时间步的平均温度: {first_10_mean.tolist()[:5]}")
    else:
        print("  (没有样本)")

print()
print("="*70)
print("结论")
print("="*70)
print("如果4个distance的温度特征分布有明显系统性差异，")
print("说明训练数据本身存在问题，导致模型学习到了错误的模式。")
print()
print("解决方案：")
print("1. 检查原始数据生成过程（仿真参数）")
print("2. 确认4个测温点确实是同一材料的不同位置")
print("3. 考虑数据增强：打乱4个点的顺序，避免模型学习位置偏差")


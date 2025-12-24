#!/usr/bin/env python3
"""详细检查PTH中index=18921样本的数据来源"""

import torch
import numpy as np

PTH_FILE = '/home/ziwu/Newpython/lg_exp/Point_lg/data_source/output_5s_h10_h300_5s_30hz_9temp/thermal_design_point_nogif_30hz_packed/thermal_dataset_multipoint.pth'
TXT_FILE = '/home/ziwu/Newpython/lg_exp/Point_lg/gif_test/newnewnew_30hz/4730_mph.txt'

print("="*80)
print("调试PTH数据存储问题")
print("="*80)

# 1. 读取PTH中的数据
data = torch.load(PTH_FILE)
sample_idx = 18921

pth_temps = data['temperature'][sample_idx].numpy()
pth_params = data['parameters'][sample_idx].numpy()
param_names = data['parameter_names']

print(f"\nPTH中样本{sample_idx}的数据:")
print(f"  前10个温度值: {pth_temps[:10]}")
print(f"  参数:")
for i, name in enumerate(param_names):
    print(f"    {name} = {pth_params[i]:.6f}")

# 2. 读取原始txt (4730_mph.txt的第2列，distance=1)
with open(TXT_FILE, 'r') as f:
    lines = f.readlines()[8:]  # 跳过前8行

txt_temps_col2 = []
for line in lines:
    if line.strip():
        parts = line.split()
        if len(parts) >= 5:
            txt_temps_col2.append(float(parts[2]))  # 第2列

txt_temps_col2 = np.array(txt_temps_col2)

print(f"\n原始TXT (4730_mph.txt, 第2列):")
print(f"  前10个温度值: {txt_temps_col2[:10]}")

# 3. 对比
print(f"\n对比:")
print(f"{'Index':<8} | {'TXT (col2)':<15} | {'PTH (18921)':<15} | {'Diff':<12}")
print("-"*60)
for i in range(20):
    diff = pth_temps[i] - txt_temps_col2[i]
    print(f"{i:<8} | {txt_temps_col2[i]:<15.6f} | {pth_temps[i]:<15.6f} | {diff:<12.6f}")

# 4. 检查PTH中样本18921前后的样本
print(f"\n="*80)
print(f"检查样本18921周围的样本（看看是否有列错位）")
print(f"="*80)

# 检查前后几个样本的参数
print(f"\n样本18921及其周围样本的参数:")
print(f"{'Sample':<8} | {'e':<12} | {'thickness':<12} | {'T0':<12} | {'T1':<12}")
print("-"*70)
for idx in range(18919, 18925):
    params = data['parameters'][idx].numpy()
    e_idx = param_names.index('e')
    thick_idx = param_names.index('thickness')
    T0_idx = param_names.index('T0')
    T1_idx = param_names.index('T1')
    
    print(f"{idx:<8} | {params[e_idx]:<12.2f} | {params[thick_idx]:<12.2f} | {params[T0_idx]:<12.2f} | {params[T1_idx]:<12.2f}")

# 5. 尝试匹配其他列
print(f"\n="*80)
print(f"尝试匹配原始txt的其他列")
print(f"="*80)

with open(TXT_FILE, 'r') as f:
    lines = f.readlines()[8:]

for col_idx in range(1, 5):
    txt_temps = []
    for line in lines:
        if line.strip():
            parts = line.split()
            if len(parts) >= 5:
                txt_temps.append(float(parts[col_idx]))
    
    txt_temps = np.array(txt_temps)
    rmse = np.sqrt(np.mean((pth_temps - txt_temps)**2))
    mae = np.mean(np.abs(pth_temps - txt_temps))
    
    print(f"\nTXT第{col_idx}列 vs PTH样本18921:")
    print(f"  RMSE = {rmse:.6f} K")
    print(f"  MAE = {mae:.6f} K")
    if rmse < 0.01:
        print(f"  ✓✓✓ 完美匹配！")
    elif rmse < 0.1:
        print(f"  ✓ 非常接近")

print(f"\n="*80)
print("结论：检查哪一列与PTH中的数据最接近")
print("="*80)


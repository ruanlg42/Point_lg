#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断数据流转问题
检查从原始txt → pth → 导出txt的每个环节
"""

import torch
import numpy as np
import yaml
from pathlib import Path

# ==================== 配置 ====================
# 原始数据文件
ORIGINAL_TXT = '/home/ziwu/Newpython/lg_exp/Point_lg/gif_test/newnewnew_30hz/4730_mph.txt'
ORIGINAL_YAML = '/home/ziwu/Newpython/lg_exp/Point_lg/gif_test/newnewnew_30hz/4730_parameters.yaml'

# 打包后的数据集
PTH_FILE = '/home/ziwu/Newpython/lg_exp/Point_lg/data_source/output_5s_h10_h300_5s_30hz_9temp/thermal_design_point_nogif_30hz_packed/thermal_dataset_multipoint.pth'

# 从数据集导出的txt（需要找到对应的样本）
EXPORTED_TXT = '/home/ziwu/Newpython/lg_exp/Point_lg/gif_test/newnewnew_30hz/index18921_e_true_4853_564941_e_pred_5085_881348_T0_293_15_T1_310_15_Lambda_9_623000_c_544_000000_p_4500_000000_thickness_1_300000_time_5_000000.txt'

# 对比哪一列（distance）
COLUMN_INDEX = 2  # 4730_mph.txt的第2列 (distance=1)
# =============================================


def read_original_txt(txt_path, column_index):
    """读取原始txt文件的指定列"""
    temps = []
    times = []
    with open(txt_path, 'r') as f:
        lines = f.readlines()[8:]  # 跳过前8行注释
        for line in lines:
            if line.strip():
                parts = line.split()
                if len(parts) >= 5:
                    times.append(float(parts[0]))
                    temps.append(float(parts[column_index]))
    return np.array(times), np.array(temps)


def read_original_yaml(yaml_path):
    """读取原始参数文件"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        params = yaml.safe_load(f)
    return params


def find_sample_in_pth(pth_path, target_e, target_thickness_base, distance, target_T0, target_T1):
    """在pth数据集中找到对应的样本
    
    Args:
        pth_path: pth文件路径
        target_e: 目标e值
        target_thickness_base: 原始thickness值
        distance: 测温点距离 (0, 1, 2, 3)
        target_T0: 目标T0值
        target_T1: 目标T1值
    
    Returns:
        sample_idx: 样本索引
        sample_data: 样本数据字典
    """
    data = torch.load(pth_path)
    
    # 参数名称
    param_names = data['parameter_names']
    e_idx = param_names.index('e')
    thickness_idx = param_names.index('thickness')
    T0_idx = param_names.index('T0')
    T1_idx = param_names.index('T1')
    
    # 根据旧的编码方式，thickness = original_thickness + distance
    target_thickness_encoded = target_thickness_base + float(distance)
    
    # 查找匹配的样本
    params = data['parameters']
    
    # 找到e值、thickness、T0、T1都匹配的样本
    e_values = params[:, e_idx]
    thickness_values = params[:, thickness_idx]
    T0_values = params[:, T0_idx]
    T1_values = params[:, T1_idx]
    
    # 容差匹配
    e_match = torch.abs(e_values - target_e) < 1.0
    thickness_match = torch.abs(thickness_values - target_thickness_encoded) < 0.01
    T0_match = torch.abs(T0_values - target_T0) < 0.1
    T1_match = torch.abs(T1_values - target_T1) < 0.1
    
    match_mask = e_match & thickness_match & T0_match & T1_match
    match_indices = torch.where(match_mask)[0]
    
    if len(match_indices) == 0:
        print(f"⚠️  警告: 在PTH中未找到匹配样本")
        print(f"   目标 e={target_e:.2f}, thickness_encoded={target_thickness_encoded:.2f}, T0={target_T0:.2f}, T1={target_T1:.2f}")
        
        # 显示部分匹配的样本
        print(f"\n   仅匹配e和thickness的样本数: {(e_match & thickness_match).sum().item()}")
        if (e_match & thickness_match).sum() > 0:
            partial_indices = torch.where(e_match & thickness_match)[0][:5]
            print(f"   这些样本的T0和T1:")
            for idx in partial_indices:
                print(f"     样本{idx}: T0={T0_values[idx]:.2f}, T1={T1_values[idx]:.2f}")
        
        return None, None
    
    # 取第一个匹配的样本
    sample_idx = match_indices[0].item()
    
    sample_data = {
        'time': data['time'][sample_idx].numpy(),
        'temperature': data['temperature'][sample_idx].numpy(),
        'parameters': data['parameters'][sample_idx].numpy(),
        'param_names': param_names
    }
    
    return sample_idx, sample_data


def read_exported_txt(txt_path, column_index=1):
    """读取导出的txt文件"""
    temps = []
    times = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    times.append(float(parts[0]))
                    temps.append(float(parts[column_index]))
                except ValueError:
                    continue
    return np.array(times), np.array(temps)


def compare_arrays(arr1, arr2, name1, name2):
    """对比两个数组"""
    print(f"\n{'='*70}")
    print(f"对比: {name1} vs {name2}")
    print(f"{'='*70}")
    print(f"长度: {len(arr1)} vs {len(arr2)}")
    
    if len(arr1) == len(arr2):
        diff = arr2 - arr1
        print(f"差异统计:")
        print(f"  均值差: {np.mean(diff):.6f}")
        print(f"  最大差: {np.max(np.abs(diff)):.6f}")
        print(f"  RMSE: {np.sqrt(np.mean(diff**2)):.6f}")
        print(f"  MAE: {np.mean(np.abs(diff)):.6f}")
        
        print(f"\n前5个数据点对比:")
        print(f"{'Index':<8} | {name1:<15} | {name2:<15} | {'Diff':<12}")
        print(f"{'-'*60}")
        for i in range(min(5, len(arr1))):
            print(f"{i:<8} | {arr1[i]:<15.6f} | {arr2[i]:<15.6f} | {diff[i]:<12.6f}")
        
        print(f"\n后5个数据点对比:")
        print(f"{'Index':<8} | {name1:<15} | {name2:<15} | {'Diff':<12}")
        print(f"{'-'*60}")
        for i in range(max(0, len(arr1)-5), len(arr1)):
            print(f"{i:<8} | {arr1[i]:<15.6f} | {arr2[i]:<15.6f} | {diff[i]:<12.6f}")
    else:
        print("⚠️ 长度不一致，无法直接对比")


def main():
    print("="*70)
    print("数据流转诊断工具")
    print("="*70)
    print(f"\n目标: 对比4730样本的distance={COLUMN_INDEX-1}测温点")
    
    # ========== 步骤1: 读取原始数据 ==========
    print(f"\n{'='*70}")
    print("步骤1: 读取原始txt文件")
    print(f"{'='*70}")
    
    orig_times, orig_temps = read_original_txt(ORIGINAL_TXT, COLUMN_INDEX)
    print(f"✓ 读取成功")
    print(f"  时间步数: {len(orig_times)}")
    print(f"  温度范围: [{orig_temps.min():.6f}, {orig_temps.max():.6f}] K")
    print(f"  温度均值: {orig_temps.mean():.6f} K")
    print(f"  前5个温度值: {orig_temps[:5]}")
    
    # 读取参数
    orig_params = read_original_yaml(ORIGINAL_YAML)
    print(f"\n原始参数:")
    for key, value in orig_params.items():
        print(f"  {key}: {value}")
    
    # ========== 步骤2: 在PTH中查找对应样本 ==========
    print(f"\n{'='*70}")
    print("步骤2: 在PTH数据集中查找对应样本")
    print(f"{'='*70}")
    
    target_e = orig_params['e']
    target_thickness = orig_params['thickness']
    target_T0 = orig_params['T0']
    target_T1 = orig_params['T1']
    distance = COLUMN_INDEX - 1
    
    sample_idx, sample_data = find_sample_in_pth(
        PTH_FILE, target_e, target_thickness, distance, target_T0, target_T1
    )
    
    if sample_data is None:
        print("❌ 未找到对应样本，终止诊断")
        return
    
    print(f"✓ 找到样本，索引: {sample_idx}")
    print(f"  时间步数: {len(sample_data['time'])}")
    print(f"  温度范围: [{sample_data['temperature'].min():.6f}, {sample_data['temperature'].max():.6f}] K")
    print(f"  温度均值: {sample_data['temperature'].mean():.6f} K")
    print(f"  前5个温度值: {sample_data['temperature'][:5]}")
    
    print(f"\n  存储的参数:")
    for i, name in enumerate(sample_data['param_names']):
        print(f"    {name}: {sample_data['parameters'][i]:.6f}")
    
    # ========== 步骤3: 对比原始txt和PTH中的数据 ==========
    compare_arrays(orig_temps, sample_data['temperature'], 
                   "原始TXT", "PTH中存储")
    
    # ========== 步骤4: 读取导出的txt文件 ==========
    print(f"\n{'='*70}")
    print("步骤4: 读取从数据集导出的txt文件")
    print(f"{'='*70}")
    
    export_times, export_temps = read_exported_txt(EXPORTED_TXT, column_index=1)
    print(f"✓ 读取成功")
    print(f"  时间步数: {len(export_times)}")
    print(f"  温度范围: [{export_temps.min():.6f}, {export_temps.max():.6f}] K")
    print(f"  温度均值: {export_temps.mean():.6f} K")
    print(f"  前5个温度值: {export_temps[:5]}")
    
    # ========== 步骤5: 对比PTH和导出txt ==========
    compare_arrays(sample_data['temperature'], export_temps,
                   "PTH中存储", "导出TXT")
    
    # ========== 步骤6: 对比原始txt和导出txt ==========
    compare_arrays(orig_temps, export_temps,
                   "原始TXT", "导出TXT")
    
    # ========== 总结 ==========
    print(f"\n{'='*70}")
    print("诊断总结")
    print(f"{'='*70}")
    
    print("\n数据流转路径:")
    print("  原始TXT → data_transfer_multipoint.py → PTH → 某个导出脚本 → 导出TXT")
    
    print("\n关键检查点:")
    print("  ① 原始TXT vs PTH: 检查data_transfer_multipoint.py是否正确读取")
    print("  ② PTH vs 导出TXT: 检查导出脚本是否正确读取PTH")
    print("  ③ 原始TXT vs 导出TXT: 端到端的数据一致性")
    
    print("\n如果发现问题:")
    print("  - 如果①有问题: 检查data_transfer_multipoint.py的txt解析逻辑")
    print("  - 如果②有问题: 检查导出脚本（如vae_reconstruction等）")
    print("  - 如果①②都正常但③有问题: 可能是导出的txt来自不同样本")


if __name__ == '__main__':
    main()


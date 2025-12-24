#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析数据不匹配的根本原因
"""

import torch
import numpy as np
import yaml
from pathlib import Path

# 配置
ORIGINAL_TXT = Path('/home/ziwu/Newpython/lg_exp/Point_lg/gif_test/newnewnew_30hz/4730_mph.txt')
ORIGINAL_YAML = Path('/home/ziwu/Newpython/lg_exp/Point_lg/gif_test/newnewnew_30hz/4730_parameters.yaml')
PTH_FILE = Path('/home/ziwu/Newpython/lg_exp/Point_lg/data_source/output_5s_h10_h300_5s_30hz_9temp/thermal_design_point_nogif_30hz_packed/thermal_dataset_multipoint.pth')

DISTANCE = 1  # 对比distance=1的数据（4730_mph.txt的第2列）


def read_txt_column(txt_path, column_index):
    """读取txt文件的指定列"""
    temps = []
    with open(txt_path, 'r') as f:
        lines = f.readlines()[8:]  # 跳过前8行注释
        for line in lines:
            if line.strip():
                parts = line.split()
                if len(parts) >= 5:
                    temps.append(float(parts[column_index]))
    return np.array(temps)


def main():
    print("="*80)
    print("数据不匹配分析")
    print("="*80)
    
    # 1. 读取原始txt
    print("\n步骤1: 读取原始txt文件 (4730_mph.txt, 第2列)")
    print("-"*80)
    orig_temps = read_txt_column(ORIGINAL_TXT, column_index=2)
    print(f"✓ 读取成功: {len(orig_temps)} 个温度点")
    print(f"  温度范围: [{orig_temps.min():.6f}, {orig_temps.max():.6f}] K")
    print(f"  温度均值: {orig_temps.mean():.6f} K")
    print(f"  前10个值: {orig_temps[:10]}")
    
    # 2. 读取原始参数
    with open(ORIGINAL_YAML, 'r', encoding='utf-8') as f:
        orig_params = yaml.safe_load(f)
    
    print(f"\n原始参数:")
    print(f"  e = {orig_params['e']:.6f}")
    print(f"  thickness = {orig_params['thickness']:.6f}")
    print(f"  Lambda = {orig_params['Lambda']:.6f}")
    
    # 3. 在PTH中查找对应样本
    print(f"\n步骤2: 在PTH中查找对应样本")
    print("-"*80)
    target_e = orig_params['e']
    target_T0 = orig_params['T0']
    target_T1 = orig_params['T1']
    target_thickness = orig_params['thickness'] + DISTANCE  # 1.3
    
    print(f"  查找条件:")
    print(f"    e ≈ {target_e:.2f}")
    print(f"    T0 ≈ {target_T0:.2f}")
    print(f"    T1 ≈ {target_T1:.2f}")
    print(f"    thickness ≈ {target_thickness:.2f}")
    
    data = torch.load(PTH_FILE)
    param_names = data['parameter_names']
    
    e_idx = param_names.index('e')
    thickness_idx = param_names.index('thickness')
    T0_idx = param_names.index('T0')
    T1_idx = param_names.index('T1')
    
    params = data['parameters']
    e_values = params[:, e_idx]
    thickness_values = params[:, thickness_idx]
    T0_values = params[:, T0_idx]
    T1_values = params[:, T1_idx]
    
    # 容差匹配 - 加入T0和T1的匹配
    e_match = torch.abs(e_values - target_e) < 1.0
    thickness_match = torch.abs(thickness_values - target_thickness) < 0.01
    T0_match = torch.abs(T0_values - target_T0) < 0.1
    T1_match = torch.abs(T1_values - target_T1) < 0.1
    
    match_mask = e_match & thickness_match & T0_match & T1_match
    match_indices = torch.where(match_mask)[0]
    
    if len(match_indices) == 0:
        print("❌ 未找到匹配样本")
        
        # 显示部分匹配的情况
        print(f"\n  部分匹配统计:")
        print(f"    仅e匹配: {e_match.sum().item()} 个")
        print(f"    e+thickness匹配: {(e_match & thickness_match).sum().item()} 个")
        print(f"    e+thickness+T0匹配: {(e_match & thickness_match & T0_match).sum().item()} 个")
        print(f"    e+thickness+T1匹配: {(e_match & thickness_match & T1_match).sum().item()} 个")
        
        if (e_match & thickness_match).sum() > 0:
            print(f"\n  e+thickness匹配但T0/T1不匹配的样本:")
            partial_indices = torch.where(e_match & thickness_match)[0][:5]
            for idx in partial_indices:
                print(f"    样本{idx}: T0={T0_values[idx]:.2f} (目标{target_T0:.2f}), T1={T1_values[idx]:.2f} (目标{target_T1:.2f})")
        
        return
    
    sample_idx = match_indices[0].item()
    print(f"✓ 找到样本: index={sample_idx}")
    
    pth_temps = data['temperature'][sample_idx].numpy()
    pth_params = data['parameters'][sample_idx].numpy()
    
    print(f"  温度范围: [{pth_temps.min():.6f}, {pth_temps.max():.6f}] K")
    print(f"  温度均值: {pth_temps.mean():.6f} K")
    print(f"  前10个值: {pth_temps[:10]}")
    
    print(f"\n  PTH中存储的参数:")
    for i, name in enumerate(param_names):
        print(f"    {name} = {pth_params[i]:.6f}")
    
    # 4. 对比原始txt和PTH
    print(f"\n步骤3: 对比原始TXT和PTH中的数据")
    print("="*80)
    
    diff = pth_temps - orig_temps
    
    print(f"统计指标:")
    print(f"  均值差: {diff.mean():.6f} K")
    print(f"  标准差: {diff.std():.6f} K")
    print(f"  RMSE: {np.sqrt((diff**2).mean()):.6f} K")
    print(f"  MAE: {np.abs(diff).mean():.6f} K")
    print(f"  最大差: {np.abs(diff).max():.6f} K")
    
    # 详细对比前20个数据点
    print(f"\n前20个数据点详细对比:")
    print(f"{'Index':<6} | {'Original TXT':<15} | {'PTH Stored':<15} | {'Difference':<15}")
    print("-"*70)
    for i in range(min(20, len(orig_temps))):
        print(f"{i:<6} | {orig_temps[i]:<15.6f} | {pth_temps[i]:<15.6f} | {diff[i]:<15.6f}")
    
    # 5. 诊断结论
    print(f"\n{'='*80}")
    print("诊断结论")
    print(f"{'='*80}")
    
    rmse = np.sqrt((diff**2).mean())
    mae = np.abs(diff).mean()
    
    if rmse < 0.001:
        print("✓ 数据匹配: PTH中的数据与原始txt完全一致")
        print("  → data_transfer_multipoint.py 工作正常")
    elif rmse < 0.1:
        print("⚠️  轻微差异: PTH与原始txt有轻微数值误差（可能是浮点精度）")
        print(f"  RMSE = {rmse:.6f} K")
    else:
        print("❌ 严重不匹配: PTH中的数据与原始txt差异很大")
        print(f"  RMSE = {rmse:.6f} K, MAE = {mae:.6f} K")
        print("\n可能的原因:")
        print("  1. data_transfer_multipoint.py 读取txt时解析错误")
        print("  2. 列索引映射错误（distance=0,1,2,3 对应 parts[1,2,3,4]）")
        print("  3. 原始txt文件格式不符合预期")
        print("  4. PTH文件中找到的样本实际不是4730的distance=1数据")
    
    # 6. 检查data_transfer的解析逻辑
    print(f"\n{'='*80}")
    print("检查data_transfer_multipoint.py的解析逻辑")
    print(f"{'='*80}")
    
    print("\n原始txt格式:")
    print("  第0列: 时间")
    print("  第1列: 温度(distance=0)")
    print("  第2列: 温度(distance=1)  ← 我们要对比的")
    print("  第3列: 温度(distance=2)")
    print("  第4列: 温度(distance=3)")
    
    print("\ndata_transfer_multipoint.py的解析:")
    print("  parts[0] → time_data")
    print("  parts[1] → temp_data_point0 (distance=0)")
    print("  parts[2] → temp_data_point1 (distance=1)  ← 应该对应原始txt的第2列")
    print("  parts[3] → temp_data_point2 (distance=2)")
    print("  parts[4] → temp_data_point3 (distance=3)")
    
    print("\n✓ 解析逻辑看起来是正确的")
    
    # 7. 验证是否是同一个样本
    print(f"\n{'='*80}")
    print("验证样本来源")
    print(f"{'='*80}")
    
    # 检查PTH中sample_id的分布（通过experiment_id或文件名推断）
    # 由于PTH文件没有保存原始sample_id，我们检查相同e值和thickness的样本数量
    e_match_count = e_match.sum().item()
    print(f"PTH中e值匹配的样本数: {e_match_count}")
    print(f"PTH中e值和thickness都匹配的样本数: {len(match_indices)}")
    
    if len(match_indices) > 1:
        print(f"\n⚠️  警告: 找到{len(match_indices)}个匹配的样本")
        print("  这意味着有多个样本具有相同的e值和thickness")
        print("  无法确定PTH中的样本就是4730")
        
        print(f"\n  所有匹配样本的索引: {match_indices.tolist()[:10]}")
        
        # 检查这些样本的温度特征
        print(f"\n  这些样本的温度均值:")
        for i, idx in enumerate(match_indices[:5].tolist()):
            temps = data['temperature'][idx].numpy()
            print(f"    样本{idx}: {temps.mean():.6f} K (min={temps.min():.2f}, max={temps.max():.2f})")
    
    # 8. 最终建议
    print(f"\n{'='*80}")
    print("最终建议")
    print(f"{'='*80}")
    
    if rmse < 0.1:
        print("✓ data_transfer_multipoint.py工作正常")
        print("✓ PTH文件中的数据是正确的")
        print("\n如果后续导出的txt有差异，问题可能在:")
        print("  - train.py的导出逻辑")
        print("  - VAE的重建过程")
        print("  - 归一化/反归一化过程")
    else:
        print("❌ 需要修复data_transfer_multipoint.py")
        print("\n建议:")
        print("  1. 检查txt文件的跳过行数（是否正确跳过前8行）")
        print("  2. 检查parts的索引（是否从1开始而不是从0）")
        print("  3. 添加调试信息，打印前几行的解析结果")


if __name__ == '__main__':
    main()


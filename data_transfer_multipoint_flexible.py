#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
打包热传导数据集为PyTorch格式（支持任意数量的测温点）

特性：
- 自动检测txt文件中的温度列数（4列、13列或任意列数）
- 每个测温点作为独立样本
- 通过 thickness+序号 来区分不同测温点
"""

import yaml
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm


def detect_num_temp_columns(txt_file, skip_lines=8):
    """检测txt文件中的温度列数
    
    Args:
        txt_file: txt文件路径
        skip_lines: 跳过的注释行数
    
    Returns:
        num_columns: 温度列数
    """
    with open(txt_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()[skip_lines:]
        
        for line in lines:
            if line.strip():
                parts = line.split()
                # 第一列是时间，后面的都是温度
                num_temp_cols = len(parts) - 1
                return num_temp_cols
    
    return 0


def pack_thermal_dataset_multipoint(
    source_dir,
    output_dir,
    skip_lines=8,
    auto_detect_columns=True,
    num_temp_columns=None
):
    """打包热传导数据集为PyTorch格式（支持多测温点）
    
    Args:
        source_dir: 源数据目录（包含*_parameters.yaml和*_mph.txt）
        output_dir: 输出目录
        skip_lines: txt文件跳过的注释行数（默认8行）
        auto_detect_columns: 是否自动检测列数（默认True）
        num_temp_columns: 手动指定温度列数（当auto_detect_columns=False时使用）
    
    每个样本的txt文件包含N个测温点的数据（对应distance=0,1,2,...,N-1）
    将每个测温点作为独立样本，通过 thickness+序号 来区分
    例如：原始thickness=0.3，N=4时，4个点对应 0.3, 1.3, 2.3, 3.3
    """
    
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    
    # 自动创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {output_dir}")
    if not output_dir.exists():
        print(f"  ✓ 已创建输出目录")
    else:
        print(f"  ✓ 输出目录已存在")
    
    # 扫描数据集
    yaml_files = list(source_dir.glob("*_parameters.yaml"))
    sample_ids = sorted([int(f.stem.replace('_parameters', '')) for f in yaml_files])
    print(f"\n找到 {len(sample_ids)} 个原始样本")
    
    if len(sample_ids) == 0:
        print("❌ 没有可用样本，已跳过打包。")
        print("   请确认 source_dir 是否正确并包含 *_parameters.yaml 与 *_mph.txt 文件。")
        return
    
    # 检测温度列数
    if auto_detect_columns:
        first_txt = source_dir / f"{sample_ids[0]}_mph.txt"
        num_temp_columns = detect_num_temp_columns(first_txt, skip_lines)
        print(f"✓ 自动检测到 {num_temp_columns} 列温度数据")
    else:
        print(f"✓ 使用指定的列数: {num_temp_columns}")
    
    if num_temp_columns == 0:
        print("❌ 无法检测到温度列，请检查txt文件格式")
        return
    
    # 收集所有数据
    all_time = []
    all_temp = []
    all_params = []
    material_list = []
    material_to_idx = {}
    
    print(f"\n正在处理样本（每个样本包含{num_temp_columns}个测温点）...")
    total_samples = 0
    skipped_samples = 0
    
    for sample_id in tqdm(sample_ids, desc="处理样本"):
        # 读取参数
        yaml_file = source_dir / f"{sample_id}_parameters.yaml"
        try:
            with open(yaml_file, 'r', encoding='utf-8') as f:
                params = yaml.safe_load(f)
        except Exception as e:
            print(f"⚠️  跳过样本{sample_id}: 无法读取yaml文件 ({e})")
            skipped_samples += 1
            continue
        
        # 读取温度数据
        txt_file = source_dir / f"{sample_id}_mph.txt"
        if not txt_file.exists():
            print(f"⚠️  跳过样本{sample_id}: txt文件不存在")
            skipped_samples += 1
            continue
        
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()[skip_lines:]  # 跳过注释行
        except Exception as e:
            print(f"⚠️  跳过样本{sample_id}: 无法读取txt文件 ({e})")
            skipped_samples += 1
            continue
        
        # 解析温度数据（1列时间 + N列温度）
        time_data = []
        temp_data_all = [[] for _ in range(num_temp_columns)]  # 为每个测温点创建列表
        
        for line in lines:
            if line.strip():
                parts = line.split()
                expected_parts = 1 + num_temp_columns  # 时间 + N个温度点
                
                if len(parts) >= expected_parts:
                    time_data.append(float(parts[0]))
                    for col_idx in range(num_temp_columns):
                        temp_data_all[col_idx].append(float(parts[col_idx + 1]))
        
        if len(time_data) == 0:
            print(f"⚠️  跳过样本{sample_id}: 没有有效数据")
            skipped_samples += 1
            continue
        
        # 提取基础参数
        original_thickness = params.get('thickness', 0)
        
        # 材料索引
        material = params.get('material', 'unknown')
        if material not in material_to_idx:
            material_to_idx[material] = len(material_to_idx)
        
        # 为每个测温点创建一个样本
        for distance, temp_data in enumerate(temp_data_all):
            if len(temp_data) > 0:  # 确保有数据
                all_time.append(time_data)
                all_temp.append(temp_data)
                
                # 参数向量：thickness加上测温点序号（0,1,2,...）来区分不同点
                param_vec = [
                    params.get('Lambda', 0),
                    params.get('T0', 0),
                    params.get('T1', 0),
                    params.get('c', 0),
                    params.get('e', 0),
                    params.get('p', 0),
                    original_thickness + float(distance),  # thickness + 点序号
                    params.get('time', 0)
                ]
                all_params.append(param_vec)
                
                material_list.append(material)
                total_samples += 1
    
    print(f"\n处理统计:")
    print(f"  原始样本数: {len(sample_ids)}")
    print(f"  成功处理: {len(sample_ids) - skipped_samples}")
    print(f"  跳过: {skipped_samples}")
    print(f"  扩展后样本数: {total_samples} (每个原始样本 × {num_temp_columns}个测温点)")
    
    # 检查数据一致性
    if len(all_time) == 0:
        print("\n❌ 错误：没有成功读取任何数据！")
        return
    
    # 打包数据
    dataset = {
        'time': torch.tensor(all_time, dtype=torch.float32),
        'temperature': torch.tensor(all_temp, dtype=torch.float32),
        'parameters': torch.tensor(all_params, dtype=torch.float32),
        'material_indices': torch.tensor([material_to_idx[m] for m in material_list], dtype=torch.long),
        'material_to_idx': material_to_idx,
        'parameter_names': ['Lambda', 'T0', 'T1', 'c', 'e', 'p', 'thickness', 'time'],
        'num_temp_columns': num_temp_columns  # 记录温度列数
    }
    
    # 保存
    output_file = output_dir / 'thermal_dataset_multipoint.pth'
    torch.save(dataset, output_file)
    
    print(f"\n✓ 完成！")
    print(f"  文件: {output_file}")
    print(f"  原始样本数: {len(sample_ids) - skipped_samples}")
    print(f"  扩展后样本数: {total_samples}")
    
    time_steps = 0
    if isinstance(dataset['time'], torch.Tensor) and dataset['time'].ndim >= 2 and dataset['time'].shape[0] > 0:
        time_steps = int(dataset['time'].shape[1])
    elif len(all_time) > 0 and len(all_time[0]) > 0:
        time_steps = len(all_time[0])
    
    print(f"  时间步: {time_steps}")
    print(f"  测温点数: {num_temp_columns} (distance=0,1,2,...,{num_temp_columns-1})")
    print(f"  大小: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    # 打印数据集维度信息
    print(f"\n数据集维度:")
    print(f"  time: {dataset['time'].shape}")
    print(f"  temperature: {dataset['temperature'].shape}")
    print(f"  parameters: {dataset['parameters'].shape}")
    print(f"  parameter_names: {dataset['parameter_names']}")
    
    # 显示thickness范围（用于验证点序号编码）
    thickness_values = dataset['parameters'][:, 6]  # thickness是第7列（索引6）
    e_values = dataset['parameters'][:, 4]  # e是第5列（索引4）
    
    print(f"\n参数范围统计:")
    print(f"  thickness: [{thickness_values.min().item():.6f}, {thickness_values.max().item():.6f}]")
    print(f"  e: [{e_values.min().item():.2f}, {e_values.max().item():.2f}]")
    
    print(f"\n✓ 数据集特点: 每{num_temp_columns}个连续样本来自同一材料的{num_temp_columns}个测温点")
    print(f"  - 相同的材料参数 (Lambda, T0, T1, c, e, p, 原始thickness, time)")
    print(f"  - 不同的温度序列 (测温位置不同)")
    print(f"  - thickness编码: 原始值 + distance序号")
    print(f"  - 示例: 原始thickness=0.3 → {num_temp_columns}个点对应 {[f'{0.3 + i:.1f}' for i in range(min(5, num_temp_columns))]}")


def main():
    """主函数 - 可配置的入口"""
    
    # ==================== 配置区域 ====================
    SOURCE_DIR = '/home/ziwu/Newpython/lg_exp/Point_lg/data_source/output_5s_h10_h300_5s_30hz_9temp_13/thermal_design_point_nogif_30hz2'
    OUTPUT_DIR = '/home/ziwu/Newpython/lg_exp/Point_lg/data_source/output_5s_h10_h300_5s_30hz_9temp_13/thermal_design_point_nogif_30hz2_packed_13points'
    
    # 跳过的注释行数（通常是8行）
    SKIP_LINES = 8
    
    # 是否自动检测列数（推荐设置为True）
    AUTO_DETECT = True
    
    # 如果AUTO_DETECT=False，手动指定列数
    NUM_TEMP_COLUMNS = 4  # 只在AUTO_DETECT=False时使用
    # ================================================
    
    print("="*80)
    print("热传导数据集打包工具 - 支持任意列数")
    print("="*80)
    print(f"\n源目录: {SOURCE_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"跳过行数: {SKIP_LINES}")
    print(f"自动检测列数: {AUTO_DETECT}")
    if not AUTO_DETECT:
        print(f"指定列数: {NUM_TEMP_COLUMNS}")
    
    pack_thermal_dataset_multipoint(
        source_dir=SOURCE_DIR,
        output_dir=OUTPUT_DIR,
        skip_lines=SKIP_LINES,
        auto_detect_columns=AUTO_DETECT,
        num_temp_columns=NUM_TEMP_COLUMNS if not AUTO_DETECT else None
    )


if __name__ == '__main__':
    main()


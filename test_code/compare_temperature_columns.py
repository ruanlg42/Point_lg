"""比较两个txt文件中指定列的温度序列"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免Qt相关错误
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr
from scipy.interpolate import interp1d

# ==================== 配置区域 ====================
# 文件1配置
TXT_FILE_1 = '/home/ziwu/Newpython/lg_exp/Point_lg/gif_test/newnewnew_30hz/4730_mph.txt'
COLUMN_1 = 2  # 读取第几列温度 (1=第1列, 2=第2列, 3=第3列, 4=第4列)
LABEL_1 = '4730_mph (distance=1)'

# 文件2配置
TXT_FILE_2 = '/home/ziwu/Newpython/lg_exp/Point_lg/gif_test/newnewnew_30hz/index18921_e_true_4853_564941_e_pred_5085_881348_T0_293_15_T1_310_15_Lambda_9_623000_c_544_000000_p_4500_000000_thickness_1_300000_time_5_000000.txt'
COLUMN_2 = 1  # 对于center_temp.txt: 3=Temp_GIF, 4=Temp_Original
LABEL_2 = '4730_video_center (Dataset)'

# 输出配置
OUTPUT_DIR = '/home/ziwu/Newpython/lg_exp/Point_lg/comparison_results'
# ================================================


def load_temperature_column(txt_path, column_index):
    """读取txt文件的指定列温度
    
    Args:
        txt_path: txt文件路径
        column_index: 列索引 (1=第1列温度)
    
    Returns:
        temps: 温度数组
    """
    temps = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            # 跳过注释和空行
            if (not line or 
                line.startswith('%') or 
                line.startswith('#') or 
                line.startswith('=') or 
                line.startswith('-') or
                line.startswith('GIF:') or
                line.startswith('Center') or
                line.startswith('Temperature') or
                line.startswith('Total') or
                line.startswith('Original') or
                line.startswith('Frame') or
                line.startswith('Statistics') or
                line.startswith('Comparison') or
                line.startswith('Mean') or
                line.startswith('Min') or
                line.startswith('Max') or
                line.startswith('Std') or
                line.startswith('RMSE')):
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                try:
                    # 确保列索引在范围内
                    if column_index < len(parts):
                        temp = float(parts[column_index])
                        temps.append(temp)
                except (ValueError, IndexError):
                    continue
    
    return np.array(temps)


def interpolate_to_common_length(temps1, temps2):
    """将两个序列插值到相同长度（取较长的）
    
    Args:
        temps1: 温度序列1
        temps2: 温度序列2
    
    Returns:
        temps1_interp: 插值后的序列1
        temps2_interp: 插值后的序列2
    """
    len1, len2 = len(temps1), len(temps2)
    
    if len1 == len2:
        return temps1, temps2
    
    # 使用较长的长度
    target_len = max(len1, len2)
    
    # 插值序列1
    if len1 != target_len:
        old_x = np.linspace(0, 1, len1)
        new_x = np.linspace(0, 1, target_len)
        f1 = interp1d(old_x, temps1, kind='linear')
        temps1_interp = f1(new_x)
    else:
        temps1_interp = temps1
    
    # 插值序列2
    if len2 != target_len:
        old_x = np.linspace(0, 1, len2)
        new_x = np.linspace(0, 1, target_len)
        f2 = interp1d(old_x, temps2, kind='linear')
        temps2_interp = f2(new_x)
    else:
        temps2_interp = temps2
    
    return temps1_interp, temps2_interp


def calculate_similarity(temps1, temps2):
    """计算两个温度序列的相似性指标
    
    Args:
        temps1: 温度序列1
        temps2: 温度序列2
    
    Returns:
        metrics: 相似性指标字典
    """
    # 确保长度相同
    temps1_aligned, temps2_aligned = interpolate_to_common_length(temps1, temps2)
    
    # 计算各种相似性指标
    metrics = {}
    
    # 1. 均方误差 (MSE)
    mse = np.mean((temps1_aligned - temps2_aligned) ** 2)
    metrics['MSE'] = mse
    
    # 2. 均方根误差 (RMSE)
    rmse = np.sqrt(mse)
    metrics['RMSE'] = rmse
    
    # 3. 平均绝对误差 (MAE)
    mae = np.mean(np.abs(temps1_aligned - temps2_aligned))
    metrics['MAE'] = mae
    
    # 4. 最大绝对误差
    max_error = np.max(np.abs(temps1_aligned - temps2_aligned))
    metrics['Max_Error'] = max_error
    
    # 5. 皮尔逊相关系数
    corr, p_value = pearsonr(temps1_aligned, temps2_aligned)
    metrics['Pearson_Corr'] = corr
    metrics['P_Value'] = p_value
    
    # 6. R² (决定系数)
    ss_res = np.sum((temps2_aligned - temps1_aligned) ** 2)
    ss_tot = np.sum((temps2_aligned - np.mean(temps2_aligned)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    metrics['R2'] = r2
    
    # 7. 相对误差 (%)
    mean_temp = np.mean(temps2_aligned)
    relative_error = (mae / mean_temp) * 100 if mean_temp != 0 else 0
    metrics['Relative_Error_Pct'] = relative_error
    
    return metrics, temps1_aligned, temps2_aligned


def plot_comparison(temps1, temps2, label1, label2, metrics, output_path):
    """绘制温度序列对比图
    
    Args:
        temps1: 温度序列1
        temps2: 温度序列2
        label1: 序列1标签
        label2: 序列2标签
        metrics: 相似性指标
        output_path: 输出文件路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 生成时间轴
    time1 = np.arange(len(temps1)) * 0.033333  # 假设30Hz采样
    time2 = np.arange(len(temps2)) * 0.033333
    
    # 1. 温度曲线对比
    ax1 = axes[0, 0]
    ax1.plot(time1, temps1, 'b-', label=label1, linewidth=2, alpha=0.8)
    ax1.plot(time2, temps2, 'r--', label=label2, linewidth=2, alpha=0.8)
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Temperature (K)', fontsize=12)
    ax1.set_title('Temperature Curves Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. 散点图 (相关性)
    ax2 = axes[0, 1]
    ax2.scatter(temps1, temps2, alpha=0.5, s=20)
    
    # 添加y=x参考线
    min_temp = min(temps1.min(), temps2.min())
    max_temp = max(temps1.max(), temps2.max())
    ax2.plot([min_temp, max_temp], [min_temp, max_temp], 'k--', 
             label='y=x', linewidth=2, alpha=0.5)
    
    ax2.set_xlabel(f'{label1} (K)', fontsize=12)
    ax2.set_ylabel(f'{label2} (K)', fontsize=12)
    ax2.set_title(f'Correlation (r={metrics["Pearson_Corr"]:.4f})', 
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. 差异曲线
    ax3 = axes[1, 0]
    difference = temps2 - temps1
    ax3.plot(time1, difference, 'g-', linewidth=2)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax3.fill_between(time1, 0, difference, alpha=0.3, color='g')
    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_ylabel('Temperature Difference (K)', fontsize=12)
    ax3.set_title(f'Difference ({label2} - {label1})', 
                  fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. 相似性指标文本
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    metrics_text = f"""
    Similarity Metrics
    {'='*50}
    
    Statistical Errors:
    • RMSE: {metrics['RMSE']:.6f} K
    • MAE: {metrics['MAE']:.6f} K
    • Max Error: {metrics['Max_Error']:.6f} K
    • Relative Error: {metrics['Relative_Error_Pct']:.4f}%
    
    Correlation:
    • Pearson r: {metrics['Pearson_Corr']:.6f}
    • R²: {metrics['R2']:.6f}
    • P-value: {metrics['P_Value']:.2e}
    
    Data Info:
    • Sequence 1 length: {len(temps1)} points
    • Sequence 2 length: {len(temps2)} points
    • Sequence 1 range: [{temps1.min():.2f}, {temps1.max():.2f}] K
    • Sequence 2 range: [{temps2.min():.2f}, {temps2.max():.2f}] K
    • Sequence 1 mean: {temps1.mean():.4f} K
    • Sequence 2 mean: {temps2.mean():.4f} K
    """
    
    ax4.text(0.1, 0.95, metrics_text, transform=ax4.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 对比图已保存: {output_path}")
    plt.close()


def main():
    print("="*70)
    print("温度序列相似性比较工具")
    print("="*70)
    
    # 创建输出目录
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取温度序列
    print(f"\n读取文件1: {TXT_FILE_1}")
    print(f"  列索引: {COLUMN_1}")
    temps1 = load_temperature_column(TXT_FILE_1, COLUMN_1)
    print(f"  读取到 {len(temps1)} 个温度点")
    print(f"  温度范围: [{temps1.min():.2f}, {temps1.max():.2f}] K")
    print(f"  温度均值: {temps1.mean():.4f} K")
    
    print(f"\n读取文件2: {TXT_FILE_2}")
    print(f"  列索引: {COLUMN_2}")
    temps2 = load_temperature_column(TXT_FILE_2, COLUMN_2)
    print(f"  读取到 {len(temps2)} 个温度点")
    print(f"  温度范围: [{temps2.min():.2f}, {temps2.max():.2f}] K")
    print(f"  温度均值: {temps2.mean():.4f} K")
    
    # 计算相似性
    print(f"\n计算相似性指标...")
    metrics, temps1_aligned, temps2_aligned = calculate_similarity(temps1, temps2)
    
    # 显示结果
    print(f"\n{'='*70}")
    print("相似性指标")
    print(f"{'='*70}")
    print(f"RMSE (均方根误差):     {metrics['RMSE']:.6f} K")
    print(f"MAE (平均绝对误差):    {metrics['MAE']:.6f} K")
    print(f"Max Error (最大误差):  {metrics['Max_Error']:.6f} K")
    print(f"Relative Error:        {metrics['Relative_Error_Pct']:.4f}%")
    print(f"Pearson Correlation:   {metrics['Pearson_Corr']:.6f}")
    print(f"R²:                    {metrics['R2']:.6f}")
    print(f"P-value:               {metrics['P_Value']:.2e}")
    
    # 判断相似程度
    print(f"\n{'='*70}")
    print("相似性评价")
    print(f"{'='*70}")
    
    if metrics['Pearson_Corr'] > 0.99:
        similarity_level = "极高"
    elif metrics['Pearson_Corr'] > 0.95:
        similarity_level = "很高"
    elif metrics['Pearson_Corr'] > 0.90:
        similarity_level = "高"
    elif metrics['Pearson_Corr'] > 0.80:
        similarity_level = "中等"
    else:
        similarity_level = "较低"
    
    print(f"相似性等级: {similarity_level}")
    print(f"相关系数: {metrics['Pearson_Corr']:.6f} (1.0=完全相关)")
    print(f"平均偏差: {metrics['MAE']:.6f} K ({metrics['Relative_Error_Pct']:.4f}%)")
    
    # 绘制对比图
    print(f"\n生成对比图...")
    output_filename = f"comparison_{Path(TXT_FILE_1).stem}_col{COLUMN_1}_vs_{Path(TXT_FILE_2).stem}_col{COLUMN_2}.png"
    output_path = output_dir / output_filename
    plot_comparison(temps1_aligned, temps2_aligned, LABEL_1, LABEL_2, metrics, output_path)
    
    # 保存详细数据到CSV
    csv_path = output_dir / output_filename.replace('.png', '.csv')
    print(f"✓ 详细数据已保存: {csv_path}")
    with open(csv_path, 'w') as f:
        f.write(f"Time(s),{LABEL_1}(K),{LABEL_2}(K),Difference(K)\n")
        time_points = np.arange(len(temps1_aligned)) * 0.033333
        for t, t1, t2 in zip(time_points, temps1_aligned, temps2_aligned):
            f.write(f"{t:.6f},{t1:.6f},{t2:.6f},{t2-t1:.6f}\n")
    
    print(f"\n{'='*70}")
    print("完成！")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()


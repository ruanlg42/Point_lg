import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter1d


def load_temperature_from_txt(txt_path, max_points=None, min_temp=0, max_temp=1000):
    """从 test_color.py 生成的 txt 文件读取温度序列（同时读取 Temp_GIF 和 Temp_Original）
    
    Args:
        txt_path: 文件路径
        max_points: 最大数据点数（如251），如果指定则只读取前N个点
        min_temp: 最小有效温度（用于过滤异常值）
        max_temp: 最大有效温度（用于过滤异常值）
    
    Returns:
        temp_gif: Temp_GIF (K) 列
        temp_original: Temp_Original (K) 列（如果存在，否则为None）
    """
    temp_gif = []
    temp_original = []
    has_original = False
    
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('GIF:') or \
               line.startswith('Center') or line.startswith('Temperature') or \
               line.startswith('Total') or line.startswith('=' * 60) or \
               line.startswith('Frame') or line.startswith('-' * 45) or \
               line.startswith('Statistics') or line.startswith('Comparison'):
                continue
            
            # 检查是否有原始温度列（通过列数判断）
            parts = line.split()
            if len(parts) >= 3:
                try:
                    temp_gif_val = float(parts[2])  # Temp_GIF在第3列（索引2）
                    
                    # 如果列数>=5，说明有Temp_Original列
                    if len(parts) >= 5:
                        temp_orig_val = float(parts[3])  # Temp_Original在第4列（索引3）
                        has_original = True
                        
                    # 过滤异常值
                        if min_temp <= temp_gif_val <= max_temp and min_temp <= temp_orig_val <= max_temp:
                            temp_gif.append(temp_gif_val)
                            temp_original.append(temp_orig_val)
                    else:
                        # 只有GIF温度
                        if min_temp <= temp_gif_val <= max_temp:
                            temp_gif.append(temp_gif_val)
                            temp_original.append(np.nan)  # 用NaN占位
                    
                        # 如果指定了最大点数，达到后停止读取
                    if max_points and len(temp_gif) >= max_points:
                            break
                except (ValueError, IndexError):
                    continue
    
    temp_gif_array = np.array(temp_gif, dtype=np.float32)
    temp_original_array = np.array(temp_original, dtype=np.float32) if has_original else None
    
    return temp_gif_array, temp_original_array, has_original


def compute_metrics(original, smoothed):
    """计算滤波性能指标"""
    diff = smoothed - original
    rmse = np.sqrt(np.mean(diff**2))
    max_diff = np.max(np.abs(diff))
    mae = np.mean(np.abs(diff))
    noise_reduction = (1 - np.std(smoothed) / np.std(original)) * 100
    correlation = np.corrcoef(original, smoothed)[0, 1]
    
    return {
        'rmse': rmse,
        'max_diff': max_diff,
        'mae': mae,
        'noise_reduction': noise_reduction,
        'correlation': correlation
    }


def compute_similarity(seq1, seq2):
    """计算两个序列的相似度指标"""
    valid_mask = ~(np.isnan(seq1) | np.isnan(seq2))
    if valid_mask.sum() == 0:
        return {'correlation': np.nan, 'rmse': np.nan, 'mae': np.nan, 'max_diff': np.nan}
    
    seq1_valid = seq1[valid_mask]
    seq2_valid = seq2[valid_mask]
    
    diff = seq1_valid - seq2_valid
    rmse = np.sqrt(np.mean(diff**2))
    mae = np.mean(np.abs(diff))
    max_diff = np.max(np.abs(diff))
    
    try:
        correlation = np.corrcoef(seq1_valid, seq2_valid)[0, 1]
    except:
        correlation = np.nan
    
    return {
        'correlation': correlation,
        'rmse': rmse,
        'mae': mae,
        'max_diff': max_diff
    }


def main():
    # ==================== 配置 ====================
    TXT_PATH = '/home/ziwu/Newpython/lg_exp/Point_lg/gif_test/newdata_aa/5261_video_center_temp.txt'
    DELTA_T = 0.02
    SIGMA = 2.0  # 高斯滤波参数
    # ================================================
    
    print(f"Loading: {TXT_PATH}")
    # 读取Temp_GIF和Temp_Original两列，过滤异常温度值（合理范围：200-350K）
    temp_gif_raw, temp_original_raw, has_original = load_temperature_from_txt(
        TXT_PATH, max_points=251, min_temp=200, max_temp=350
    )
    data_name = Path(TXT_PATH).stem  # 获取文件名（不含扩展名）
    print(f"Data file: {Path(TXT_PATH).name}")
    print(f"Length: {len(temp_gif_raw)} points")
    print(f"Has Temp_Original column: {has_original}")
    
    print(f"\nTemp_GIF (K):")
    print(f"  Range: {temp_gif_raw.min():.2f} - {temp_gif_raw.max():.2f} K")
    print(f"  Mean: {temp_gif_raw.mean():.2f} K, Std: {temp_gif_raw.std():.2f} K")
    
    if has_original:
        print(f"\nTemp_Original (K):")
        print(f"  Range: {temp_original_raw.min():.2f} - {temp_original_raw.max():.2f} K")
        print(f"  Mean: {temp_original_raw.mean():.2f} K, Std: {temp_original_raw.std():.2f} K")
    
    # 应用高斯滤波
    print(f"\nApplying Gaussian filter (sigma={SIGMA})...")
    temp_gif_smooth = gaussian_filter1d(temp_gif_raw, sigma=SIGMA)
    temp_original_smooth = None
    if has_original:
        temp_original_smooth = gaussian_filter1d(temp_original_raw, sigma=SIGMA)
    
    # 计算导数（数值微分）
    temp_gif_derivative = np.zeros_like(temp_gif_smooth)
    temp_gif_derivative[:-1] = (temp_gif_smooth[1:] - temp_gif_smooth[:-1]) / DELTA_T
    temp_gif_derivative[-1] = temp_gif_derivative[-2]
    
    if has_original:
        temp_original_derivative = np.zeros_like(temp_original_smooth)
        temp_original_derivative[:-1] = (temp_original_smooth[1:] - temp_original_smooth[:-1]) / DELTA_T
        temp_original_derivative[-1] = temp_original_derivative[-2]
    
    # 计算滤波效果指标
    metrics_gif = compute_metrics(temp_gif_raw, temp_gif_smooth)
    print(f"\nGaussian Filter Effects on Temp_GIF:")
    print(f"  RMSE:            {metrics_gif['rmse']:.4f} K")
    print(f"  MAE:             {metrics_gif['mae']:.4f} K")
    print(f"  Max difference:  {metrics_gif['max_diff']:.4f} K")
    print(f"  Noise reduction: {metrics_gif['noise_reduction']:.2f}%")
    print(f"  Correlation:     {metrics_gif['correlation']:.4f}")
    
    if has_original:
        metrics_original = compute_metrics(temp_original_raw, temp_original_smooth)
        print(f"\nGaussian Filter Effects on Temp_Original:")
        print(f"  RMSE:            {metrics_original['rmse']:.4f} K")
        print(f"  MAE:             {metrics_original['mae']:.4f} K")
        print(f"  Max difference:  {metrics_original['max_diff']:.4f} K")
        print(f"  Noise reduction: {metrics_original['noise_reduction']:.2f}%")
        print(f"  Correlation:     {metrics_original['correlation']:.4f}")
        
        # 计算滤波前后的相似度
        print(f"\nSimilarity Analysis:")
        sim_before = compute_similarity(temp_gif_raw, temp_original_raw)
        sim_after = compute_similarity(temp_gif_smooth, temp_original_smooth)
        
        print(f"\n  BEFORE filtering:")
        print(f"    Correlation: {sim_before['correlation']:.4f}")
        print(f"    RMSE:        {sim_before['rmse']:.4f} K")
        print(f"    MAE:         {sim_before['mae']:.4f} K")
        print(f"    Max diff:    {sim_before['max_diff']:.4f} K")
        
        print(f"\n  AFTER filtering:")
        print(f"    Correlation: {sim_after['correlation']:.4f}")
        print(f"    RMSE:        {sim_after['rmse']:.4f} K")
        print(f"    MAE:         {sim_after['mae']:.4f} K")
        print(f"    Max diff:    {sim_after['max_diff']:.4f} K")
    
    # 可视化
    print("\nCreating visualization...")
    time_steps = np.arange(len(temp_gif_raw)) * DELTA_T
    
    if has_original:
        # 有原始数据：显示两列对比
        fig = plt.figure(figsize=(24, 18))
        
        # 第一行：Temp_GIF 滤波前后
        ax1 = plt.subplot(4, 3, 1)
        ax1.plot(time_steps, temp_gif_raw, 'b-', alpha=0.5, linewidth=1.5, label='Temp_GIF (Raw)')
        ax1.plot(time_steps, temp_gif_smooth, 'r-', linewidth=2.5, label=f'Temp_GIF (σ={SIGMA})')
        ax1.set_xlabel('Time (s)', fontsize=12)
        ax1.set_ylabel('Temperature (K)', fontsize=12)
        ax1.set_title('Temp_GIF: Before vs After Filtering', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11, loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Temp_GIF 差异
        ax2 = plt.subplot(4, 3, 2)
        diff_gif = temp_gif_smooth - temp_gif_raw
        ax2.plot(time_steps, diff_gif, 'g-', linewidth=1.5)
        ax2.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        ax2.fill_between(time_steps, 0, diff_gif, alpha=0.3, color='green')
        ax2.set_xlabel('Time (s)', fontsize=12)
        ax2.set_ylabel('Difference (K)', fontsize=12)
        ax2.set_title('Temp_GIF: Filter Effect', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Temp_GIF 导数
        ax3 = plt.subplot(4, 3, 3)
        ax3.plot(time_steps, temp_gif_derivative, 'purple', linewidth=2, label='dT/dt (GIF)')
        ax3.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        ax3.set_xlabel('Time (s)', fontsize=12)
        ax3.set_ylabel('dT/dt (K/s)', fontsize=12)
        ax3.set_title('Temp_GIF Derivative', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3)
        
        # 第二行：Temp_Original 滤波前后
        ax4 = plt.subplot(4, 3, 4)
        ax4.plot(time_steps, temp_original_raw, 'b-', alpha=0.5, linewidth=1.5, label='Temp_Original (Raw)')
        ax4.plot(time_steps, temp_original_smooth, 'r-', linewidth=2.5, label=f'Temp_Original (σ={SIGMA})')
        ax4.set_xlabel('Time (s)', fontsize=12)
        ax4.set_ylabel('Temperature (K)', fontsize=12)
        ax4.set_title('Temp_Original: Before vs After Filtering', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=11, loc='best')
        ax4.grid(True, alpha=0.3)
        
        # Temp_Original 差异
        ax5 = plt.subplot(4, 3, 5)
        diff_orig = temp_original_smooth - temp_original_raw
        ax5.plot(time_steps, diff_orig, 'g-', linewidth=1.5)
        ax5.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        ax5.fill_between(time_steps, 0, diff_orig, alpha=0.3, color='green')
        ax5.set_xlabel('Time (s)', fontsize=12)
        ax5.set_ylabel('Difference (K)', fontsize=12)
        ax5.set_title('Temp_Original: Filter Effect', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # Temp_Original 导数
        ax6 = plt.subplot(4, 3, 6)
        ax6.plot(time_steps, temp_original_derivative, 'purple', linewidth=2, label='dT/dt (Original)')
        ax6.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        ax6.set_xlabel('Time (s)', fontsize=12)
        ax6.set_ylabel('dT/dt (K/s)', fontsize=12)
        ax6.set_title('Temp_Original Derivative', fontsize=14, fontweight='bold')
        ax6.legend(fontsize=11)
        ax6.grid(True, alpha=0.3)
        
        # 第三行：相似度对比（滤波前）
        ax7 = plt.subplot(4, 3, 7)
        ax7.plot(time_steps, temp_gif_raw, 'b-', alpha=0.6, linewidth=2, label='Temp_GIF (Raw)')
        ax7.plot(time_steps, temp_original_raw, 'orange', alpha=0.6, linewidth=2, label='Temp_Original (Raw)')
        ax7.set_xlabel('Time (s)', fontsize=12)
        ax7.set_ylabel('Temperature (K)', fontsize=12)
        ax7.set_title(f'Similarity BEFORE Filtering\nCorr={sim_before["correlation"]:.4f}, RMSE={sim_before["rmse"]:.4f}K', 
                      fontsize=14, fontweight='bold')
        ax7.legend(fontsize=11, loc='best')
        ax7.grid(True, alpha=0.3)
        
        # 相似度对比（滤波后）
        ax8 = plt.subplot(4, 3, 8)
        ax8.plot(time_steps, temp_gif_smooth, 'b-', alpha=0.6, linewidth=2, label='Temp_GIF (Filtered)')
        ax8.plot(time_steps, temp_original_smooth, 'orange', alpha=0.6, linewidth=2, label='Temp_Original (Filtered)')
        ax8.set_xlabel('Time (s)', fontsize=12)
        ax8.set_ylabel('Temperature (K)', fontsize=12)
        ax8.set_title(f'Similarity AFTER Filtering\nCorr={sim_after["correlation"]:.4f}, RMSE={sim_after["rmse"]:.4f}K', 
                      fontsize=14, fontweight='bold')
        ax8.legend(fontsize=11, loc='best')
        ax8.grid(True, alpha=0.3)
        
        # 相似度差异（滤波后 - 滤波前）
        ax9 = plt.subplot(4, 3, 9)
        diff_before = temp_gif_raw - temp_original_raw
        diff_after = temp_gif_smooth - temp_original_smooth
        ax9.plot(time_steps, diff_before, 'b-', alpha=0.6, linewidth=1.5, label='Before Filtering')
        ax9.plot(time_steps, diff_after, 'r-', alpha=0.8, linewidth=2, label='After Filtering')
        ax9.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        ax9.set_xlabel('Time (s)', fontsize=12)
        ax9.set_ylabel('Temp_GIF - Temp_Original (K)', fontsize=12)
        ax9.set_title('Difference: GIF - Original', fontsize=14, fontweight='bold')
        ax9.legend(fontsize=11, loc='best')
        ax9.grid(True, alpha=0.3)
        
        # 第四行：散点图对比
        ax10 = plt.subplot(4, 3, 10)
        ax10.scatter(temp_gif_raw, temp_original_raw, alpha=0.5, s=10, label='Before Filtering')
        min_val = min(temp_gif_raw.min(), temp_original_raw.min())
        max_val = max(temp_gif_raw.max(), temp_original_raw.max())
        ax10.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, alpha=0.5)
        ax10.set_xlabel('Temp_GIF (K)', fontsize=12)
        ax10.set_ylabel('Temp_Original (K)', fontsize=12)
        ax10.set_title('Scatter: Before Filtering', fontsize=14, fontweight='bold')
        ax10.legend(fontsize=11)
        ax10.grid(True, alpha=0.3)
        
        ax11 = plt.subplot(4, 3, 11)
        ax11.scatter(temp_gif_smooth, temp_original_smooth, alpha=0.5, s=10, c='red', label='After Filtering')
        min_val = min(temp_gif_smooth.min(), temp_original_smooth.min())
        max_val = max(temp_gif_smooth.max(), temp_original_smooth.max())
        ax11.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, alpha=0.5)
        ax11.set_xlabel('Temp_GIF (K)', fontsize=12)
        ax11.set_ylabel('Temp_Original (K)', fontsize=12)
        ax11.set_title('Scatter: After Filtering', fontsize=14, fontweight='bold')
        ax11.legend(fontsize=11)
        ax11.grid(True, alpha=0.3)
        
        # 统计信息
        ax12 = plt.subplot(4, 3, 12)
        ax12.axis('off')
        stats_text = f"""GAUSSIAN FILTER STATISTICS (σ={SIGMA})
{'='*50}
Data: {Path(TXT_PATH).name}
Length: {len(temp_gif_raw)} points

Temp_GIF Filter Effects:
  RMSE: {metrics_gif['rmse']:.4f} K
  MAE: {metrics_gif['mae']:.4f} K
  Noise reduction: {metrics_gif['noise_reduction']:.2f}%
  Correlation: {metrics_gif['correlation']:.4f}

Temp_Original Filter Effects:
  RMSE: {metrics_original['rmse']:.4f} K
  MAE: {metrics_original['mae']:.4f} K
  Noise reduction: {metrics_original['noise_reduction']:.2f}%
  Correlation: {metrics_original['correlation']:.4f}

Similarity Analysis:
  BEFORE filtering:
    Correlation: {sim_before['correlation']:.4f}
    RMSE: {sim_before['rmse']:.4f} K
    MAE: {sim_before['mae']:.4f} K
  AFTER filtering:
    Correlation: {sim_after['correlation']:.4f}
    RMSE: {sim_after['rmse']:.4f} K
    MAE: {sim_after['mae']:.4f} K"""
        ax12.text(0.05, 0.95, stats_text, transform=ax12.transAxes,
                 fontsize=9, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.4, pad=1))
    
    else:
        # 只有GIF温度：显示原来的布局
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Temp_GIF 滤波前后对比
        ax1 = plt.subplot(3, 2, 1)
        ax1.plot(time_steps, temp_gif_raw, 'b-', alpha=0.5, linewidth=1.5, label='Temp_GIF (Raw)')
        ax1.plot(time_steps, temp_gif_smooth, 'r-', linewidth=2.5, label=f'Gaussian (σ={SIGMA})')
        ax1.set_xlabel('Time (s)', fontsize=13)
        ax1.set_ylabel('Temperature (K)', fontsize=13)
        ax1.set_title(f'Temp_GIF: Before vs After Filtering', fontsize=15, fontweight='bold')
        ax1.legend(fontsize=12, loc='best')
        ax1.grid(True, alpha=0.3)
        
        # 2. 差异图
        ax2 = plt.subplot(3, 2, 2)
        diff_gif = temp_gif_smooth - temp_gif_raw
        ax2.plot(time_steps, diff_gif, 'g-', linewidth=1.5)
        ax2.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        ax2.fill_between(time_steps, 0, diff_gif, alpha=0.3, color='green')
        ax2.set_xlabel('Time (s)', fontsize=13)
        ax2.set_ylabel('Temperature Difference (K)', fontsize=13)
        ax2.set_title('Filter Effect (Filtered - Raw)', fontsize=15, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. 导数
        ax3 = plt.subplot(3, 2, 3)
        ax3.plot(time_steps, temp_gif_derivative, 'purple', linewidth=2)
        ax3.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        ax3.set_xlabel('Time (s)', fontsize=13)
        ax3.set_ylabel('Temperature Derivative (K/s)', fontsize=13)
        ax3.set_title('Temperature Derivative', fontsize=15, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. 频谱分析
        ax4 = plt.subplot(3, 2, 4)
        fft_orig = np.abs(np.fft.rfft(temp_gif_raw - temp_gif_raw.mean()))
        fft_smooth = np.abs(np.fft.rfft(temp_gif_smooth - temp_gif_smooth.mean()))
        freqs = np.fft.rfftfreq(len(temp_gif_raw), d=DELTA_T)
        ax4.semilogy(freqs, fft_orig, 'b-', alpha=0.7, linewidth=1.5, label='Raw')
        ax4.semilogy(freqs, fft_smooth, 'r-', linewidth=2.5, label='Filtered')
        ax4.set_xlabel('Frequency (Hz)', fontsize=13)
        ax4.set_ylabel('Amplitude (log scale)', fontsize=13)
        ax4.set_title('Frequency Spectrum (FFT)', fontsize=15, fontweight='bold')
        ax4.legend(fontsize=12)
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(0, min(10, freqs[-1]))
        
        # 5-6. 统计信息
        ax5 = plt.subplot(3, 2, (5, 6))
        ax5.axis('off')
        stats_text = f"""GAUSSIAN FILTER STATISTICS (σ={SIGMA})
  {'='*60}
  
  Data File: {Path(TXT_PATH).name}
Length: {len(temp_gif_raw)} points
Duration: {len(temp_gif_raw)*DELTA_T:.2f} seconds

Temp_GIF (Raw):
  Range: {temp_gif_raw.min():.2f} - {temp_gif_raw.max():.2f} K
  Mean: {temp_gif_raw.mean():.4f} K
  Std: {temp_gif_raw.std():.4f} K

Temp_GIF (Filtered):
  Mean: {temp_gif_smooth.mean():.4f} K
  Std: {temp_gif_smooth.std():.4f} K
  
  Filter Effects:
  RMSE: {metrics_gif['rmse']:.4f} K
  MAE: {metrics_gif['mae']:.4f} K
  Max difference: {metrics_gif['max_diff']:.4f} K
  Noise reduction: {metrics_gif['noise_reduction']:.2f}%
  Correlation: {metrics_gif['correlation']:.4f}
  
  Derivative Statistics:
  Mean dT/dt: {np.mean(temp_gif_derivative):.6f} K/s
  Max dT/dt: {np.max(temp_gif_derivative):.6f} K/s
  Min dT/dt: {np.min(temp_gif_derivative):.6f} K/s"""
        ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes,
                 fontsize=11, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.4, pad=1.5))
    
    plt.tight_layout()
    
    # 保存
    output_path = Path(TXT_PATH).parent / f'{data_name}_gaussian_filter_effect.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    # 保存数值数据
    output_txt = Path(TXT_PATH).parent / f'{data_name}_gaussian_filter_data.txt'
    with open(output_txt, 'w') as f:
        f.write("# Gaussian Filter Effect Data\n")
        f.write(f"# Original data file: {Path(TXT_PATH).name}\n")
        f.write(f"# Sigma: {SIGMA}, Delta_t: {DELTA_T}s\n")
        if has_original:
            f.write("# Columns: Time(s), Temp_GIF_Raw(K), Temp_GIF_Filtered(K), Temp_Orig_Raw(K), "
                   f"Temp_Orig_Filtered(K), Diff_GIF(K), Diff_Orig(K), Deriv_GIF(K/s), Deriv_Orig(K/s)\n")
            f.write("-" * 120 + "\n")
            diff_gif = temp_gif_smooth - temp_gif_raw
            diff_orig = temp_original_smooth - temp_original_raw
            for i in range(len(temp_gif_raw)):
                f.write(f"{time_steps[i]:8.4f}  {temp_gif_raw[i]:12.6f}  {temp_gif_smooth[i]:12.6f}  "
                       f"{temp_original_raw[i]:12.6f}  {temp_original_smooth[i]:12.6f}  "
                       f"{diff_gif[i]:12.6f}  {diff_orig[i]:12.6f}  "
                       f"{temp_gif_derivative[i]:12.6f}  {temp_original_derivative[i]:12.6f}\n")
        else:
            f.write("# Columns: Time(s), Temp_GIF_Raw(K), Temp_GIF_Filtered(K), Difference(K), Derivative(K/s)\n")
            f.write("-" * 100 + "\n")
            diff_gif = temp_gif_smooth - temp_gif_raw
            for i in range(len(temp_gif_raw)):
                f.write(f"{time_steps[i]:8.4f}  {temp_gif_raw[i]:12.6f}  {temp_gif_smooth[i]:12.6f}  "
                       f"{diff_gif[i]:12.6f}  {temp_gif_derivative[i]:12.6f}\n")
    print(f"Numerical data saved to: {output_txt}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Data file: {Path(TXT_PATH).name}")
    print(f"\nTemp_GIF Filter Effects:")
    print(f"  ✓ Noise reduction: {metrics_gif['noise_reduction']:.2f}%")
    print(f"  ✓ RMSE: {metrics_gif['rmse']:.4f} K")
    print(f"  ✓ Correlation: {metrics_gif['correlation']:.4f}")
    print(f"  ✓ Max temperature change: {metrics_gif['max_diff']:.4f} K")
    if has_original:
        print(f"\nTemp_Original Filter Effects:")
        print(f"  ✓ Noise reduction: {metrics_original['noise_reduction']:.2f}%")
        print(f"  ✓ RMSE: {metrics_original['rmse']:.4f} K")
        print(f"  ✓ Correlation: {metrics_original['correlation']:.4f}")
        print(f"\nSimilarity Analysis:")
        print(f"  BEFORE filtering - Correlation: {sim_before['correlation']:.4f}, RMSE: {sim_before['rmse']:.4f} K")
        print(f"  AFTER filtering  - Correlation: {sim_after['correlation']:.4f}, RMSE: {sim_after['rmse']:.4f} K")
        print(f"  ✓ Similarity {'improved' if sim_after['correlation'] > sim_before['correlation'] else 'decreased'} after filtering")
    print("="*80)


if __name__ == '__main__':
    main()


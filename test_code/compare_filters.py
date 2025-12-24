import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import savgol_filter

from model import kalman_cv_smooth_batch


def load_temperature_from_txt(txt_path):
    """从 test_color.py 生成的 txt 文件读取温度序列"""
    temps = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('GIF:') or \
               line.startswith('Center') or line.startswith('Temperature') or \
               line.startswith('Total') or line.startswith('Original') or \
               line.startswith('=' * 60) or line.startswith('Frame') or \
               line.startswith('-' * 45) or line.startswith('Statistics') or \
               line.startswith('Comparison'):
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    temps.append(float(parts[2]))
                except ValueError:
                    continue
    return np.array(temps, dtype=np.float32)


def moving_average(signal, window_size=5):
    """移动平均滤波"""
    if window_size < 2:
        return signal.copy()
    pad_width = window_size // 2
    padded = np.pad(signal, pad_width, mode='edge')
    smoothed = np.convolve(padded, np.ones(window_size)/window_size, mode='valid')
    return smoothed[:len(signal)]


def exponential_moving_average(signal, alpha=0.3):
    """指数移动平均（EMA）"""
    smoothed = np.zeros_like(signal)
    smoothed[0] = signal[0]
    for i in range(1, len(signal)):
        smoothed[i] = alpha * signal[i] + (1 - alpha) * smoothed[i-1]
    return smoothed


def apply_savgol(signal, window_length=11, polyorder=3):
    """Savitzky-Golay 滤波"""
    if window_length >= len(signal):
        window_length = len(signal) - 1 if len(signal) % 2 == 0 else len(signal) - 2
    if window_length < polyorder + 2:
        window_length = polyorder + 2
    if window_length % 2 == 0:
        window_length += 1
    return savgol_filter(signal, window_length, polyorder)


def apply_gaussian(signal, sigma=2.0):
    """高斯滤波"""
    return gaussian_filter1d(signal, sigma=sigma)


def apply_median(signal, size=5):
    """中值滤波"""
    return median_filter(signal, size=size)


def apply_kalman(signal, delta_t, Q_scale=1e-1, R_scale=1e-3, device='cuda'):
    """Kalman 滤波 (Minimal smoothing)"""
    temp_tensor = torch.from_numpy(signal).float().unsqueeze(0).to(device)
    with torch.no_grad():
        temp_smooth, _ = kalman_cv_smooth_batch(
            temp_tensor, delta_t=delta_t,
            Q_scale=Q_scale, R_scale=R_scale, rts=True
        )
    return temp_smooth.squeeze(0).cpu().numpy()


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


def main():
    TXT_PATH = '/home/ziwu/Newpython/lg_exp/Point_lg/gif_test/4000_video_center_temp.txt'
    DELTA_T = 0.02
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    temp_original = load_temperature_from_txt(TXT_PATH)
    print(f"\nLoading: {TXT_PATH}")
    print(f"Length: {len(temp_original)} points")
    print(f"Range: {temp_original.min():.2f} - {temp_original.max():.2f} K")
    
    print("\n" + "="*80)
    print("Applying filters...")
    print("="*80)
    
    filters = [
        {'name': 'Original', 'smoothed': temp_original.copy(), 'color': 'black', 'style': '-', 'linewidth': 2.5},
        {'name': 'Kalman (Minimal)', 'smoothed': apply_kalman(temp_original, DELTA_T, 1e-1, 1e-3, device), 'color': 'red', 'style': '-', 'linewidth': 2},
        {'name': 'Moving Avg (w=5)', 'smoothed': moving_average(temp_original, 5), 'color': 'blue', 'style': '--', 'linewidth': 1.5},
        {'name': 'Savitzky-Golay', 'smoothed': apply_savgol(temp_original, 11, 3), 'color': 'green', 'style': '--', 'linewidth': 1.5},
        {'name': 'Gaussian (σ=2)', 'smoothed': apply_gaussian(temp_original, 2.0), 'color': 'orange', 'style': '--', 'linewidth': 1.5},
        {'name': 'Median (s=5)', 'smoothed': apply_median(temp_original, 5), 'color': 'purple', 'style': '--', 'linewidth': 1.5},
        {'name': 'Exp. MA (α=0.3)', 'smoothed': exponential_moving_average(temp_original, 0.3), 'color': 'cyan', 'style': '--', 'linewidth': 1.5},
    ]
    
    for filt in filters[1:]:
        filt['metrics'] = compute_metrics(temp_original, filt['smoothed'])
        m = filt['metrics']
        print(f"{filt['name']:20s}: RMSE={m['rmse']:.4f}K, Corr={m['correlation']:.4f}, Noise↓={m['noise_reduction']:.1f}%")
    
    time_steps = np.arange(len(temp_original)) * DELTA_T
    
    # ========== 图1: 温度曲线对比 (2x2) ==========
    print("\nCreating curves comparison plot...")
    fig1 = plt.figure(figsize=(22, 12))
    
    # 1. 全局对比
    ax1 = plt.subplot(2, 2, 1)
    for filt in filters:
        ax1.plot(time_steps, filt['smoothed'], color=filt['color'], linestyle=filt['style'], 
                linewidth=filt['linewidth'], alpha=0.8, label=filt['name'])
    ax1.set_xlabel('Time (s)', fontsize=14)
    ax1.set_ylabel('Temperature (K)', fontsize=14)
    ax1.set_title('Temperature Comparison (Full Range)', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=11, loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', labelsize=12)
    
    # 2. 局部放大
    ax2 = plt.subplot(2, 2, 2)
    zoom = slice(0, 50)
    for filt in filters:
        ax2.plot(time_steps[zoom], filt['smoothed'][zoom], color=filt['color'], 
                linestyle=filt['style'], linewidth=filt['linewidth'], alpha=0.8, label=filt['name'])
    ax2.set_xlabel('Time (s)', fontsize=14)
    ax2.set_ylabel('Temperature (K)', fontsize=14)
    ax2.set_title('Zoomed View (First 50 points)', fontsize=16, fontweight='bold')
    ax2.legend(fontsize=11, loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', labelsize=12)
    
    # 3. 差异热图
    ax3 = plt.subplot(2, 2, 3)
    diff_matrix = np.array([f['smoothed'][:100] - temp_original[:100] for f in filters[1:]])
    im = ax3.imshow(diff_matrix, aspect='auto', cmap='RdBu_r', vmin=-5, vmax=5,
                    extent=[0, 100*DELTA_T, len(filters)-1, 0])
    ax3.set_xlabel('Time (s)', fontsize=14)
    ax3.set_ylabel('Filter Method', fontsize=14)
    ax3.set_yticks(np.arange(len(filters)-1) + 0.5)
    ax3.set_yticklabels([f['name'] for f in filters[1:]], fontsize=11)
    ax3.set_title('Difference Heatmap (first 100 pts)', fontsize=16, fontweight='bold')
    ax3.tick_params(axis='both', labelsize=12)
    plt.colorbar(im, ax=ax3, label='Temp Diff (K)', fraction=0.046)
    
    # 4. 性能表格
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    table_data = [['Filter Method', 'RMSE', 'MAE', 'Corr', 'Noise↓%']]
    for f in filters[1:]:
        m = f['metrics']
        table_data.append([f['name'], f"{m['rmse']:.3f}", f"{m['mae']:.3f}", 
                          f"{m['correlation']:.4f}", f"{m['noise_reduction']:.1f}"])
    table = ax4.table(cellText=table_data, cellLoc='left', loc='center', colWidths=[0.35, 0.15, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 3)
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=12)
    ax4.set_title('Performance Metrics Comparison', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    out1 = Path(TXT_PATH).parent / 'filter_comparison_curves.png'
    plt.savefig(out1, dpi=300, bbox_inches='tight')
    print(f"Saved: {out1}")
    
    # ========== 图2: 性能指标对比 (2x3) ==========
    print("Creating metrics comparison plot...")
    fig2 = plt.figure(figsize=(24, 12))
    names = [f['name'] for f in filters[1:]]
    colors = [f['color'] for f in filters[1:]]
    
    metrics_list = [
        ('rmse', 'RMSE (K)', 'Root Mean Square Error', '.3f', 0.05, None),
        ('mae', 'MAE (K)', 'Mean Absolute Error', '.3f', 0.05, None),
        ('max_diff', 'Max Deviation (K)', 'Maximum Absolute Deviation', '.2f', 1.0, None),
        ('noise_reduction', 'Noise Reduction (%)', 'Noise Reduction', '.1f', 2.0, None),
        ('correlation', 'Correlation', 'Signal Fidelity', '.4f', 0.002, (0.95, 1.0)),
    ]
    
    for idx, (key, xlabel, title, fmt, offset, xlim) in enumerate(metrics_list, 1):
        ax = plt.subplot(2, 3, idx)
        values = [f['metrics'][key] for f in filters[1:]]
        bars = ax.barh(names, values, color=colors, alpha=0.75, edgecolor='black', linewidth=2)
        ax.set_xlabel(xlabel, fontsize=15, fontweight='bold')
        ax.set_title(title, fontsize=17, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.tick_params(axis='both', labelsize=12)
        
        if xlim:
            ax.set_xlim(xlim)
            # 相关系数标签放在柱子内部
            for i, bar in enumerate(bars):
                ax.text(bar.get_width() - offset, bar.get_y() + bar.get_height()/2, 
                       f'{values[i]:{fmt}}', ha='right', va='center', fontsize=12, fontweight='bold', color='white')
        else:
            # 其他指标标签放在柱子外部
            for i, bar in enumerate(bars):
                ax.text(bar.get_width() + offset, bar.get_y() + bar.get_height()/2, 
                       f'{values[i]:{fmt}}', ha='left', va='center', fontsize=12, fontweight='bold')
    
    # 6. 推荐文本框
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    best = min(filters[1:], key=lambda x: x['metrics']['rmse'])
    text = f"""
  RECOMMENDATION
  {'='*50}
  
  Best Filter: {best['name']}
  
  Performance Metrics:
    • RMSE:            {best['metrics']['rmse']:.4f} K
    • MAE:             {best['metrics']['mae']:.4f} K
    • Max deviation:   {best['metrics']['max_diff']:.2f} K
    • Noise reduction: {best['metrics']['noise_reduction']:.1f}%
    • Correlation:     {best['metrics']['correlation']:.4f}
  
  Configuration for model.py:
    Q_scale = 1e-1  (Allow signal changes)
    R_scale = 1e-3  (Trust observations)
  
  Why Kalman Filter?
    ✓ Physics-based optimal estimator
    ✓ Provides derivative dT/dt
    ✓ Handles non-uniform noise
    ✓ Theoretically grounded
  
  Performance Ranking (by RMSE):
"""
    sorted_f = sorted(filters[1:], key=lambda x: x['metrics']['rmse'])
    for i, f in enumerate(sorted_f, 1):
        text += f"    {i}. {f['name']:20s} {f['metrics']['rmse']:.4f}K\n"
    
    ax6.text(0.05, 0.95, text, transform=ax6.transAxes, fontsize=12, verticalalignment='top',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.4, pad=1.5))
    
    plt.tight_layout()
    out2 = Path(TXT_PATH).parent / 'filter_comparison_metrics.png'
    plt.savefig(out2, dpi=300, bbox_inches='tight')
    print(f"Saved: {out2}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✓ Best filter: {best['name']}")
    print(f"✓ RMSE: {best['metrics']['rmse']:.4f} K")
    print(f"✓ Correlation: {best['metrics']['correlation']:.4f}")
    print(f"✓ Noise reduction: {best['metrics']['noise_reduction']:.1f}%")
    print("\nUse in model.py: Q_scale=1e-1, R_scale=1e-3")
    print("="*80)


if __name__ == '__main__':
    main()


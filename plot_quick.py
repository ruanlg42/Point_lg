"""
快速绘图脚本 - 直接修改CSV路径即可使用

修改下面的 CSV_PATH 变量为你的CSV文件路径
"""

import pandas as pd
import numpy as np
import matplotlib
# 强制使用非交互式后端以兼容无显示环境
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error
import math

# ========== 在这里修改你的CSV路径 ==========
CSV_PATH = r'/home/ziwu/Newpython/lg_exp/Point_lg/results/test_result/Useful_A_dropout0__20251108_131250_timevae_hybrid_res_d256_latent256_mamba2_trans2_head4_ic0.5/all_predictions_20251108_141745.csv'
NUM_BINS = 15  # 分区数量
RANDOM_SEED = 42  # 随机种子
AXIS_LIMIT = 30000  # 所有图的横轴坐标轴范围
SHOW_BEFORE_SAVE = False  # 服务器/无显示环境下直接保存
# ==========================================

# 设置随机种子
np.random.seed(RANDOM_SEED)

# 设置字体为 Arial
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 22})  # 全局字体大小

# 加载数据
csv_path = Path(CSV_PATH)
print(f"加载数据: {csv_path}")
df = pd.read_csv(csv_path)
e_true = df['e_true'].values
e_pred = df['e_pred'].values

print(f"数据量: {len(df)}")
print(f"e 范围: {e_true.min():.2f} ~ {e_true.max():.2f}")

# 保存目录
save_dir = csv_path.parent
save_dir.mkdir(parents=True, exist_ok=True)

# 用于存储所有图表 (fig对象, 保存路径)
figures_to_save = []

# ==================== 图1: 散点图 ====================
print("\n绘制图1: 真实值 vs 预测值散点图...")
fig, ax = plt.subplots(figsize=(12, 12))

# 设置刻度向内
ax.tick_params(direction='in', length=8, width=2, labelsize=20)
ax.tick_params(which='minor', direction='in', length=4, width=1.5)

# 设置固定范围
lims = np.array([0, AXIS_LIMIT])
ax.set_xlim(lims)
ax.set_ylim(lims)

# 添加置信带
for ratio, color, alpha in [(0.1, 'limegreen', 0.2), 
                            (0.2, 'gold', 0.2), 
                            (0.5, 'salmon', 0.15)]:
    ax.fill_between(lims, lims*(1-ratio), lims*(1+ratio),
                    color=color, alpha=alpha,
                    label=f'±{int(ratio*100)}% range')

# 绘制散点
ax.scatter(e_true, e_pred, alpha=0.6, s=20, edgecolors='none', label='Predictions')

# 理想线 y=x
ax.plot(lims, lims, 'k--', linewidth=2, alpha=0.5, label='y = x')

# 计算指标
r2 = r2_score(e_true, e_pred)
rmse = np.sqrt(mean_squared_error(e_true, e_pred))
mae = np.mean(np.abs(e_true - e_pred))
mape = np.mean(np.abs((e_true - e_pred) / e_true)) * 100

# 添加统计信息
stats_text = f'R² = {r2:.3f}'
ax.text(0.95, 0.05, stats_text, transform=ax.transAxes, 
        fontsize=20, ha='right', va='bottom',
        bbox=dict(facecolor='white', alpha=0.7))

ax.set_xlabel('True Thermal Effusivity', fontsize=22, fontweight='bold')
ax.set_ylabel('Predicted Thermal Effusivity', fontsize=22, fontweight='bold')
# ax.set_title('True vs Predicted Thermal Effusivity (Linear Scale)', fontsize=24, fontweight='bold')
ax.legend(fontsize=20, loc='upper left')
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
ax.set_aspect('equal')  # 保持xy轴刻度单位长度一致

plt.tight_layout()
save_path1 = save_dir / 'scatter_true_vs_pred.png'
print(f"  R² = {r2:.4f}, RMSE = {rmse:.2f}, MAE = {mae:.2f}, MAPE = {mape:.2f}%")
# 收集图表
figures_to_save.append((fig, save_path1, "图1: 真实值 vs 预测值散点图"))

# ==================== 图2: 分区误差折线图（等间隔分箱）====================
print(f"\n绘制图2: 分区误差折线图 ({NUM_BINS}区，等间隔分箱)...")

# 按等间隔分箱
min_e = e_true.min()
max_e = e_true.max()
bins_uniform = np.linspace(min_e, max_e, NUM_BINS + 1)
bin_centers_uniform = (bins_uniform[:-1] + bins_uniform[1:]) / 2

# 计算每个区间的误差
mean_abs_errors_uniform = []
mean_rel_errors_uniform = []

for i in range(NUM_BINS):
    # 找到落在该区间的样本
    mask = (e_true >= bins_uniform[i]) & (e_true < bins_uniform[i+1])
    
    if i == NUM_BINS - 1:  # 最后一个区间包含右边界
        mask = (e_true >= bins_uniform[i]) & (e_true <= bins_uniform[i+1])
    
    if mask.sum() > 0:
        # 平均绝对误差
        abs_error = np.mean(np.abs(e_true[mask] - e_pred[mask]))
        # 平均相对误差
        rel_error = np.mean(np.abs((e_true[mask] - e_pred[mask]) / e_true[mask]))
        
        mean_abs_errors_uniform.append(abs_error)
        mean_rel_errors_uniform.append(rel_error)
    else:
        mean_abs_errors_uniform.append(np.nan)
        mean_rel_errors_uniform.append(np.nan)

# 移除NaN值
valid_mask_uniform = ~np.isnan(mean_abs_errors_uniform)
bin_centers_uniform = bin_centers_uniform[valid_mask_uniform]
mean_abs_errors_uniform = np.array(mean_abs_errors_uniform)[valid_mask_uniform]
mean_rel_errors_uniform = np.array(mean_rel_errors_uniform)[valid_mask_uniform]

# 绘图 - 折线图
fig, ax1 = plt.subplots(figsize=(12, 8))

# 设置刻度向内
ax1.tick_params(direction='in', length=8, width=2, labelsize=20)
ax1.tick_params(which='minor', direction='in', length=4, width=1.5)

# 左y轴: 相对误差 (红色)
color_rel = 'tab:red'
ax1.set_xlabel('Thermal Effusivity', fontsize=20, fontweight='bold')
ax1.set_ylabel('Mean Relative Error', fontsize=20, fontweight='bold', color=color_rel)
ax1.set_xlim(0, AXIS_LIMIT)  # 设置横轴范围
line1 = ax1.plot(bin_centers_uniform, mean_rel_errors_uniform, '-o', color=color_rel, 
                 linewidth=2, markersize=4, label='Mean Relative Error')
ax1.tick_params(axis='y', labelcolor=color_rel)
ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

# 右y轴: 绝对误差 (蓝色)
ax2 = ax1.twinx()
ax2.tick_params(direction='in', length=8, width=2, labelsize=20)
ax2.tick_params(which='minor', direction='in', length=4, width=1.5)
color_abs = 'tab:blue'
ax2.set_ylabel('Mean Absolute Error', fontsize=20, fontweight='bold', color=color_abs)
line2 = ax2.plot(bin_centers_uniform, mean_abs_errors_uniform, '-s', color=color_abs, 
                 linewidth=2, markersize=4, label='Mean Absolute Error')
ax2.tick_params(axis='y', labelcolor=color_abs)

# 合并图例
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, fontsize=18, loc='upper center')

# plt.title('Mean Absolute/Relative Error vs Thermal Effusivity (Uniform Bins)', 
#           fontsize=20, fontweight='bold')

fig.tight_layout()
save_path2 = save_dir / 'binned_errors_vs_effusivity.png'
# 收集图表
figures_to_save.append((fig, save_path2, "图2: 分区误差折线图"))

# ==================== 图5: 分区误差柱状图（等样本数，柱宽=区间宽度）====================
print(f"\n绘制图5: 分区误差柱状图 ({NUM_BINS}区，等样本数分箱，柱宽代表区间范围)...")

# 按百分位数分箱，确保每个区间样本数量相同
percentiles = np.linspace(0, 100, NUM_BINS + 1)
bins_quantile = np.percentile(e_true, percentiles)
# 确保边界唯一（避免重复值导致的空区间）
bins_quantile = np.unique(bins_quantile)

bin_left_edges = []  # 区间左边界
bin_widths = []      # 区间宽度
mean_abs_errors_quantile = []
mean_rel_errors_quantile = []
sample_counts = []

for i in range(len(bins_quantile) - 1):
    # 找到落在该区间的样本
    if i == len(bins_quantile) - 2:  # 最后一个区间包含右边界
        mask = (e_true >= bins_quantile[i]) & (e_true <= bins_quantile[i+1])
    else:
        mask = (e_true >= bins_quantile[i]) & (e_true < bins_quantile[i+1])
    
    if mask.sum() > 0:
        # 区间左边界
        left_edge = bins_quantile[i]
        # 区间宽度 = 右边界 - 左边界
        width = bins_quantile[i+1] - bins_quantile[i]
        
        # 平均绝对误差
        abs_error = np.mean(np.abs(e_true[mask] - e_pred[mask]))
        # 平均相对误差
        rel_error = np.mean(np.abs((e_true[mask] - e_pred[mask]) / e_true[mask]))
        
        bin_left_edges.append(left_edge)
        bin_widths.append(width)
        mean_abs_errors_quantile.append(abs_error)
        mean_rel_errors_quantile.append(rel_error)
        sample_counts.append(mask.sum())

bin_left_edges = np.array(bin_left_edges)
bin_widths = np.array(bin_widths)
mean_abs_errors_quantile = np.array(mean_abs_errors_quantile)
mean_rel_errors_quantile = np.array(mean_rel_errors_quantile)

# 绘图 - 柱状图
fig, ax1 = plt.subplots(figsize=(14, 8))

# 设置刻度向内
ax1.tick_params(direction='in', length=8, width=2, labelsize=20)
ax1.tick_params(which='minor', direction='in', length=4, width=1.5)

# 左y轴: 相对误差 (红色柱状图)
color_rel = 'tab:red'
ax1.set_xlabel('Thermal Effusivity', fontsize=20, fontweight='bold')
ax1.set_ylabel('Mean Relative Error', fontsize=20, fontweight='bold', color=color_rel)
ax1.set_xlim(0, AXIS_LIMIT)  # 设置横轴范围
# 使用bar绘制，柱子从左边界开始，宽度=区间宽度，align='edge'确保从左边界对齐
bars1 = ax1.bar(bin_left_edges, mean_rel_errors_quantile, width=bin_widths, 
                align='edge',  # 从左边界开始绘制
                color=color_rel, alpha=0.7, edgecolor='darkred', linewidth=1,
                label='Mean Relative Error')
ax1.tick_params(axis='y', labelcolor=color_rel)
ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.3, axis='y')

# 右y轴: 绝对误差 (蓝色柱状图)
ax2 = ax1.twinx()
ax2.tick_params(direction='in', length=8, width=2, labelsize=20)
ax2.tick_params(which='minor', direction='in', length=4, width=1.5)
color_abs = 'tab:blue'
ax2.set_ylabel('Mean Absolute Error', fontsize=20, fontweight='bold', color=color_abs)
# 绘制蓝色柱状图，同样从左边界开始
bars2 = ax2.bar(bin_left_edges, mean_abs_errors_quantile, width=bin_widths, 
                align='edge',  # 从左边界开始绘制
                color=color_abs, alpha=0.5, edgecolor='darkblue', linewidth=1,
                label='Mean Absolute Error')
ax2.tick_params(axis='y', labelcolor=color_abs)

# 合并图例
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=color_rel, alpha=0.7, edgecolor='darkred', label='Mean Relative Error'),
                   Patch(facecolor=color_abs, alpha=0.5, edgecolor='darkblue', label='Mean Absolute Error')]
ax1.legend(handles=legend_elements, fontsize=18, loc='upper center')

# plt.title('Mean Absolute/Relative Error vs Thermal Effusivity (Equal Sample Bins, Bar Width = Range)', 
#           fontsize=20, fontweight='bold')

fig.tight_layout()
save_path5 = save_dir / 'binned_errors_vs_effusivity_bar.png'
print(f"  每个区间样本数: {sample_counts}")
# 收集图表
figures_to_save.append((fig, save_path5, "图5: 分区误差柱状图"))

# ==================== 图3: 温度分组对比图 ====================
print(f"\n绘制图3: 按温度组合分组对比图...")

"""温度组合列名自动识别：优先使用 T0/T1，否则回退到 param_1/param_2"""
has_T = ('T0' in df.columns) and ('T1' in df.columns)
has_param = ('param_1' in df.columns) and ('param_2' in df.columns)

# 检测温度组合
if has_T or has_param:
    # 获取所有唯一的温度组合
    if has_T:
        df['temp_group'] = df['T0'].round(1).astype(str) + '_' + df['T1'].round(1).astype(str)
    else:
        df['temp_group'] = df['param_1'].round(1).astype(str) + '_' + df['param_2'].round(1).astype(str)
    temp_groups = df['temp_group'].unique()
    num_groups = len(temp_groups)
    
    print(f"  检测到 {num_groups} 个温度组合:")
    for i, group in enumerate(sorted(temp_groups), 1):
        t0, t1 = group.split('_')
        count = (df['temp_group'] == group).sum()
        print(f"    {i}. T0={t0}K, T1={t1}K ({count} 个样本)")
    
    # 计算子图布局 (尽量接近正方形)
    ncols = math.ceil(math.sqrt(num_groups))
    nrows = math.ceil(num_groups / ncols)
    print(f"  子图布局: {nrows} 行 × {ncols} 列")
    
    # 创建画布
    fig_width = ncols * 5
    fig_height = nrows * 5
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
    
    # 如果只有一个子图，确保axes是数组
    if num_groups == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # 为每个温度组合绘制子图
    for idx, group in enumerate(sorted(temp_groups)):
        ax = axes[idx]
        
        # 筛选该组数据
        group_data = df[df['temp_group'] == group]
        e_true_group = group_data['e_true'].values
        e_pred_group = group_data['e_pred'].values
        
        # 解析温度值
        t0, t1 = group.split('_')
        
        # 设置刻度向内
        ax.tick_params(direction='in', length=6, width=1.5, labelsize=16)
        ax.tick_params(which='minor', direction='in', length=3, width=1)
        
        # 设置固定范围
        lims_temp = np.array([0, AXIS_LIMIT])
        ax.set_xlim(lims_temp)
        ax.set_ylim(lims_temp)
        
        # 绘制散点
        ax.scatter(e_true_group, e_pred_group, alpha=0.6, s=15, edgecolors='none', color='steelblue')
        
        # 理想线 y=x
        ax.plot(lims_temp, lims_temp, 'r--', linewidth=1.5, alpha=0.7, label='y = x')
        
        # 计算R²
        r2_group = r2_score(e_true_group, e_pred_group)
        
        # 添加R²信息
        ax.text(0.05, 0.95, f'R² = {r2_group:.3f}', transform=ax.transAxes, 
                fontsize=14, ha='left', va='top',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        # 设置标题
        ax.set_title(f'T0-T1: {t0}-{t1}', fontsize=18, fontweight='bold')
        
        # 设置坐标轴标签
        if idx >= (nrows - 1) * ncols:  # 最后一行
            ax.set_xlabel('True Thermal Effusivity', fontsize=18)
        if idx % ncols == 0:  # 第一列
            ax.set_ylabel('Predicted Thermal Effusivity', fontsize=18)
        
        ax.grid(True, linestyle='--', linewidth=0.3, alpha=0.5)
        ax.set_aspect('equal')
    
    # 隐藏多余的子图
    for idx in range(num_groups, len(axes)):
        axes[idx].axis('off')
    
    # 添加总标题
    # fig.suptitle('True vs Predicted Comparison Across Temperature Settings', 
    #              fontsize=20, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # 保存
    save_path3 = save_dir / 'scatter_by_temperature_groups.png'
    # 收集图表
    figures_to_save.append((fig, save_path3, "图3: 温度分组散点图"))
    
    # 输出统计信息
    print(f"\n  各温度组合的R²:")
    for group in sorted(temp_groups):
        group_data = df[df['temp_group'] == group]
        e_true_group = group_data['e_true'].values
        e_pred_group = group_data['e_pred'].values
        r2_group = r2_score(e_true_group, e_pred_group)
        t0, t1 = group.split('_')
        print(f"    T0={t0}K, T1={t1}K: R² = {r2_group:.4f}")
    # ==================== 图4: 温度设置统计对比图 ====================
    print(f"\n绘制图4: 温度设置统计对比图...")
    
    # 创建2x2子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 准备数据
    temp_group_labels = []
    sample_counts = []
    mean_rel_errors_temp = []
    mean_abs_errors_temp = []
    rel_error_data = []
    abs_error_data = []
    
    for group in sorted(temp_groups):
        group_data = df[df['temp_group'] == group]
        e_true_g = group_data['e_true'].values
        e_pred_g = group_data['e_pred'].values
        
        # 计算误差
        abs_errors = np.abs(e_true_g - e_pred_g)
        rel_errors = np.abs((e_true_g - e_pred_g) / e_true_g)
        
        t0, t1 = group.split('_')
        label = f'T0_{t0}_T1_{t1}'
        
        temp_group_labels.append(label)
        sample_counts.append(len(group_data))
        mean_rel_errors_temp.append(np.mean(rel_errors))
        mean_abs_errors_temp.append(np.mean(abs_errors))
        rel_error_data.append(rel_errors)
        abs_error_data.append(abs_errors)
    
    # 定义颜色
    colors = ['#7fcdbb', '#41b6c4', '#c7c7c7', '#fec44f']
    
    # 子图1: 相对误差分布 (堆叠直方图)
    ax1 = axes[0, 0]
    ax1.tick_params(direction='in', length=6, width=1.5, labelsize=16)
    bins_rel = np.linspace(0, 1.0, 30)
    for i, (data, label, color) in enumerate(zip(rel_error_data, temp_group_labels, colors)):
        ax1.hist(data, bins=bins_rel, alpha=0.7, label=label, color=color, edgecolor='white', linewidth=0.5)
    ax1.set_xlabel('Relative Error', fontsize=18, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=18, fontweight='bold')
    # ax1.set_title('Relative Error Distribution by Temperature', fontsize=18, fontweight='bold')
    ax1.legend(fontsize=14, framealpha=0.9)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.set_xlim(0, 1.0)
    
    # 子图2: 绝对误差分布 (堆叠直方图)
    ax2 = axes[0, 1]
    ax2.tick_params(direction='in', length=6, width=1.5, labelsize=16)
    bins_abs = np.linspace(0, 10000, 30)
    for i, (data, label, color) in enumerate(zip(abs_error_data, temp_group_labels, colors)):
        ax2.hist(data, bins=bins_abs, alpha=0.7, label=label, color=color, edgecolor='white', linewidth=0.5)
    ax2.set_xlabel('Absolute Error', fontsize=18, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=18, fontweight='bold')
    # ax2.set_title('Absolute Error Distribution by Temperature', fontsize=18, fontweight='bold')
    ax2.legend(fontsize=14, framealpha=0.9)
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.set_xlim(0, 10000)
    
    # 子图3: 样本数统计 (柱状图)
    ax3 = axes[1, 0]
    ax3.tick_params(direction='in', length=6, width=1.5, labelsize=16)
    x_pos = np.arange(len(temp_group_labels))
    bars = ax3.bar(x_pos, sample_counts, color=colors, alpha=0.8, edgecolor='white', linewidth=1)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(temp_group_labels, rotation=15, ha='right')
    ax3.set_xlabel('Temperature Setting', fontsize=18, fontweight='bold')
    ax3.set_ylabel('Number of Samples', fontsize=18, fontweight='bold')
    # ax3.set_title('Sample Count by Temperature Setting', fontsize=18, fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.3, axis='y')
    # 在柱子上添加数值
    for bar, count in zip(bars, sample_counts):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}', ha='center', va='bottom', fontsize=14)
    
    # 子图4: 平均相对误差对比 (柱状图)
    ax4 = axes[1, 1]
    ax4.tick_params(direction='in', length=6, width=1.5, labelsize=16)
    bars2 = ax4.bar(x_pos, mean_rel_errors_temp, color=colors, alpha=0.8, edgecolor='white', linewidth=1)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(temp_group_labels, rotation=15, ha='right')
    ax4.set_xlabel('Temperature Setting', fontsize=18, fontweight='bold')
    ax4.set_ylabel('Mean Relative Error', fontsize=18, fontweight='bold')
    # ax4.set_title('Mean Relative Error by Temperature Setting', fontsize=18, fontweight='bold')
    ax4.grid(True, linestyle='--', alpha=0.3, axis='y')
    # 在柱子上添加数值
    for bar, err in zip(bars2, mean_rel_errors_temp):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{err:.3f}', ha='center', va='bottom', fontsize=14)
    
    # 添加总标题
    # fig.suptitle('Temperature Settings Comparison (Common Materials)', 
    #              fontsize=20, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # 保存
    save_path4 = save_dir / 'temperature_statistics_comparison.png'
    # 收集图表
    figures_to_save.append((fig, save_path4, "图4: 温度统计对比图"))
    
else:
    print("  ⚠ CSV中缺少 param_1 或 param_2 列，跳过温度相关图表")

# ==================== 统一显示和保存所有图表 ====================
print(f"\n{'='*60}")
print(f"生成了 {len(figures_to_save)} 张图表")
print(f"{'='*60}")

if SHOW_BEFORE_SAVE:
    print("\n▶ 正在显示所有图表...")
    print("  提示：关闭所有图表窗口后，将自动保存图片")
    print(f"  {'='*56}")
    for i, (fig, path, title) in enumerate(figures_to_save, 1):
        print(f"  [{i}/{len(figures_to_save)}] {title}")
        fig.canvas.manager.set_window_title(f"{title} - {path.name}")
    
    # 无显示环境下不调用 show
    
    # 关闭所有窗口后，保存图片
    print(f"\n▶ 正在保存图片...")
    for i, (fig, path, title) in enumerate(figures_to_save, 1):
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  [{i}/{len(figures_to_save)}] ✓ 已保存: {path.name}")
        plt.close(fig)
else:
    # 直接保存
    print(f"\n▶ 正在保存图片...")
    for i, (fig, path, title) in enumerate(figures_to_save, 1):
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  [{i}/{len(figures_to_save)}] ✓ 已保存: {path.name}")
        plt.close(fig)

print(f"\n{'='*60}")
print(f"✅ 完成！所有图表已保存到:")
print(f"   {save_dir}")
print(f"{'='*60}")


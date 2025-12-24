"""测试4个测温点的预测结果"""
import torch
import numpy as np
import yaml
from pathlib import Path
import sys

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from dataset import ThermalDataset
from model import TimeVAE1D_HybridMambaTransformer_Residual

# 配置
TXT_PATH = '/home/ziwu/Newpython/lg_exp/Point_lg/gif_test/newnewnew_30hz/4006_mph.txt'
MODEL_PATH = '/home/ziwu/Newpython/lg_exp/Point_lg/results/model_result/A_dropout0.1_20251108_091522_timevae_hybrid_res_d128_latent128_mamba2_trans2_head4_ic0.5/best_model_epoch_447.pth'
CONFIG_PATH = '/home/ziwu/Newpython/lg_exp/Point_lg/results/model_result/A_dropout0.1_20251108_091522_timevae_hybrid_res_d128_latent128_mamba2_trans2_head4_ic0.5/config_used.yaml'

def load_txt_column(txt_path, column_index):
    """读取txt文件的指定列温度"""
    temps = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            parts = line.split()
            if len(parts) >= 5:
                try:
                    temp = float(parts[column_index])
                    temps.append(temp)
                except (ValueError, IndexError):
                    continue
    return np.array(temps)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # 加载配置
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 加载数据集（用于归一化参数）
    dataset = ThermalDataset(
        pth_path=config['data']['dataset_path'],
        normalize_temp=config['data']['normalize_temp'],
        use_log_e=config['data']['use_log_e']
    )
    
    # 加载模型
    print("Loading model...")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    
    # 使用checkpoint配置
    if 'config' in checkpoint:
        saved_config = checkpoint['config']
        vae_cfg = saved_config['model']['timevae1d_hybrid_mamba_transformer_residual']
        total_time = saved_config['data']['total_time']
        delta_t = saved_config['data']['delta_t']
    else:
        vae_cfg = config['model']['timevae1d_hybrid_mamba_transformer_residual']
        total_time = config['data']['total_time']
        delta_t = config['data']['delta_t']
    
    seq_len = int(total_time / delta_t) + 1
    
    model = TimeVAE1D_HybridMambaTransformer_Residual(
        seq_len=seq_len,
        C_in=vae_cfg.get('C_in', 1),
        latent_dim=vae_cfg.get('latent_dim', 256),
        d_model=vae_cfg.get('d_model', 128),
        n_mamba=vae_cfg.get('n_mamba', 2),
        n_transformer=vae_cfg.get('n_transformer', 2),
        nhead=vae_cfg.get('nhead', 8),
        d_state=vae_cfg.get('d_state', 16),
        d_conv=vae_cfg.get('d_conv', 4),
        expand=vae_cfg.get('expand', 2),
        dropout=vae_cfg.get('dropout', 0.1),
        decoder_base=vae_cfg.get('decoder_base', 64),
        total_time=total_time,
        delta_t=delta_t
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded!\n")
    
    # 测试4个测温点
    print("="*70)
    print("Testing 4 temperature measurement points")
    print("="*70)
    print(f"File: {TXT_PATH}")
    print(f"True e value: 3919.11 (from parameters.yaml)")
    print()
    
    results = []
    for col_idx in range(1, 5):
        distance = col_idx - 1
        print(f"--- Column {col_idx} (distance={distance}) ---")
        
        # 读取温度
        temps = load_txt_column(TXT_PATH, col_idx)
        print(f"  Temperature points: {len(temps)}")
        print(f"  Temperature range: [{temps.min():.2f}, {temps.max():.2f}] K")
        print(f"  Temperature mean: {temps.mean():.2f} K")
        
        # 插值到正确长度
        if len(temps) != seq_len:
            from scipy.interpolate import interp1d
            old_time = np.linspace(0, 1, len(temps))
            new_time = np.linspace(0, 1, seq_len)
            interpolator = interp1d(old_time, temps, kind='linear')
            temps = interpolator(new_time)
        
        # 归一化
        if dataset.normalize_temp:
            temps = (temps - dataset.temp_mean.numpy()) / dataset.temp_std.numpy()
        
        # 预测
        temp_tensor = torch.from_numpy(temps).unsqueeze(0).unsqueeze(1).float().to(device)
        
        with torch.no_grad():
            model_output = model(temp_tensor)
            _, e_pred, _, _, _ = model_output
            
            # 反归一化
            if dataset.use_log_e:
                e_log = e_pred.cpu().numpy()[0][0] * dataset.e_std.numpy() + dataset.e_mean.numpy()
                pred_e = np.exp(e_log)
            else:
                pred_e = e_pred.cpu().numpy()[0][0] * dataset.e_std.numpy() + dataset.e_mean.numpy()
        
        results.append((distance, pred_e))
        print(f"  Predicted e: {pred_e:.2f}")
        print(f"  Error: {abs(pred_e - 3919.11):.2f} ({abs(pred_e - 3919.11) / 3919.11 * 100:.1f}%)")
        print()
    
    # 汇总结果
    print("="*70)
    print("Summary")
    print("="*70)
    print("Distance | Predicted e | Error     | Error %")
    print("-" * 70)
    for distance, pred_e in results:
        error = abs(pred_e - 3919.11)
        error_pct = error / 3919.11 * 100
        print(f"   {distance}     | {pred_e:11.2f} | {error:9.2f} | {error_pct:6.1f}%")
    
    # 统计
    pred_values = [p for _, p in results]
    print()
    print(f"Mean prediction: {np.mean(pred_values):.2f}")
    print(f"Std of predictions: {np.std(pred_values):.2f}")
    print(f"Min prediction: {np.min(pred_values):.2f}")
    print(f"Max prediction: {np.max(pred_values):.2f}")

if __name__ == '__main__':
    main()


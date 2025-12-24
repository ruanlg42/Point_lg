#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VAE GIF重建和e值预测脚本
功能：
1. 加载指定的VAE模型
2. 读取输入GIF，转换为温度序列
3. 对每个像素的温度序列进行VAE重建
4. 输出：原始GIF、重建GIF、e值分布图
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import time
import torch
import torch.nn as nn
import numpy as np
import yaml
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import warnings
import random
from scipy.ndimage import gaussian_filter
warnings.filterwarnings('ignore')

# 设置随机种子
def set_seed(seed=42):
    """设置随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

from dataset import ThermalDataset
from model import (TimeVAE1D_Mamba, TimeVAE1D_Mamba_PhysicsDecoder, TimeVAE1D_Transformer,
                   TimeVAE1D_HybridMambaTransformer, TimeVAE1D_HybridMT_Physics4Channel,
                   TimeVAE1D_StageAware, TimeVAE1D_HybridMambaTransformer_Residual, TimeVAE1D_SSTEncoder_Residual)


def load_vae_model(model_path, config_path, device):
    """加载VAE模型"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    model_config = config['model']
    model_type = model_config.get('type', '').lower()
    
    # 检查是否为VAE模型
    vae_models = ['timevae1d_mamba', 'timevae1d_mamba_physics_decoder', 'timevae1d_transformer',
                  'timevae1d_hybrid_mamba_transformer', 'timevae1d_hybridmt_physics4channel',
                  'timevae1d_stageaware', 'timevae1d_hybrid_mamba_transformer_residual',
                  'timevae1d_sst_encoder_residual']
    
    if model_type not in vae_models:
        raise ValueError(f"Model type {model_type} is not a VAE model!")
    
    # 创建模型
    total_time = config['data'].get('total_time', 5.0)
    delta_t = config['data'].get('delta_t', 0.02)
    seq_len = int(total_time / delta_t) + 1
    
    if model_type == 'timevae1d_mamba':
        cfg = model_config.get('timevae1d_mamba', {})
        model = TimeVAE1D_Mamba(
            seq_len=seq_len,
            C_in=cfg.get('C_in', 1),
            latent_dim=cfg.get('latent_dim', 64),
            d_model=cfg.get('d_model', 128),
            depth=cfg.get('depth', 4),
            d_state=cfg.get('d_state', 16),
            d_conv=cfg.get('d_conv', 4),
            expand=cfg.get('expand', 2),
            decoder_base=cfg.get('decoder_base', 64)
        )
    elif model_type == 'timevae1d_transformer':
        cfg = model_config.get('timevae1d_transformer', {})
        model = TimeVAE1D_Transformer(
            seq_len=seq_len,
            C_in=cfg.get('C_in', 1),
            latent_dim=cfg.get('latent_dim', 64),
            d_model=cfg.get('d_model', 128),
            nhead=cfg.get('nhead', 8),
            num_layers=cfg.get('num_layers', 4),
            dim_feedforward=cfg.get('dim_feedforward', 512),
            dropout=cfg.get('dropout', 0.1),
            decoder_base=cfg.get('decoder_base', 64)
        )
    elif model_type == 'timevae1d_hybrid_mamba_transformer':
        cfg = model_config.get('timevae1d_hybrid_mamba_transformer', {})
        model = TimeVAE1D_HybridMambaTransformer(
            seq_len=seq_len,
            C_in=cfg.get('C_in', 1),
            latent_dim=cfg.get('latent_dim', 64),
            d_model=cfg.get('d_model', 128),
            n_mamba=cfg.get('n_mamba', 2),
            n_transformer=cfg.get('n_transformer', 2),
            nhead=cfg.get('nhead', 8),
            d_state=cfg.get('d_state', 16),
            d_conv=cfg.get('d_conv', 4),
            expand=cfg.get('expand', 2),
            dropout=cfg.get('dropout', 0.1),
            decoder_base=cfg.get('decoder_base', 64),
            total_time=total_time,
            delta_t=delta_t
        )
    elif model_type == 'timevae1d_hybrid_mamba_transformer_residual':
        cfg = model_config.get('timevae1d_hybrid_mamba_transformer_residual', {})
        model = TimeVAE1D_HybridMambaTransformer_Residual(
            seq_len=seq_len,
            C_in=cfg.get('C_in', 1),
            latent_dim=cfg.get('latent_dim', 64),
            d_model=cfg.get('d_model', 128),
            n_mamba=cfg.get('n_mamba', 2),
            n_transformer=cfg.get('n_transformer', 2),
            nhead=cfg.get('nhead', 8),
            d_state=cfg.get('d_state', 16),
            d_conv=cfg.get('d_conv', 4),
            expand=cfg.get('expand', 2),
            dropout=cfg.get('dropout', 0.1),
            decoder_base=cfg.get('decoder_base', 64),
            total_time=total_time,
            delta_t=delta_t
        )
    elif model_type == 'timevae1d_stageaware':
        cfg = model_config.get('timevae1d_stageaware', {})
        model = TimeVAE1D_StageAware(
            seq_len=seq_len,
            C_in=cfg.get('C_in', 1),
            latent_dim=cfg.get('latent_dim', 64),
            d_model=cfg.get('d_model', 128),
            n_mamba=cfg.get('n_mamba', 2),
            n_transformer=cfg.get('n_transformer', 2),
            nhead=cfg.get('nhead', 8),
            d_state=cfg.get('d_state', 16),
            d_conv=cfg.get('d_conv', 4),
            expand=cfg.get('expand', 2),
            dropout=cfg.get('dropout', 0.2),
            decoder_base=cfg.get('decoder_base', 64),
            total_time=total_time,
            delta_t=delta_t
        )
    elif model_type == 'timevae1d_sst_encoder_residual':
        vae_cfg = model_config.get('timevae1d_sst_encoder_residual', {})
        model = TimeVAE1D_SSTEncoder_Residual(
            seq_len=seq_len,
            C_in=vae_cfg.get('C_in', 1),
            latent_dim=vae_cfg.get('latent_dim', 64),
            d_model=vae_cfg.get('d_model', 128),
            long_patch_len=vae_cfg.get('long_patch_len', 16),
            long_stride=vae_cfg.get('long_stride', 8),
            short_patch_len=vae_cfg.get('short_patch_len', 16),
            short_stride=vae_cfg.get('short_stride', 8),
            context_window=vae_cfg.get('context_window', 151),
            n_mamba=vae_cfg.get('n_mamba', 2),
            n_transformer=vae_cfg.get('n_transformer', 2),
            nhead=vae_cfg.get('nhead', 8),
            d_state=vae_cfg.get('d_state', 16),
            d_conv=vae_cfg.get('d_conv', 4),
            expand=vae_cfg.get('expand', 2),
            dropout=vae_cfg.get('dropout', 0.1),
            decoder_base=vae_cfg.get('decoder_base', 64),
            use_router=vae_cfg.get('use_router', True),
            total_time=total_time,
            delta_t=delta_t
        )
    else:
        raise ValueError(f"Unsupported VAE model type: {model_type}")
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # 处理 DataParallel
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # 加载数据集（用于归一化）
    data_config = config['data']
    dataset_path = data_config['dataset_path']
    temp_dataset = ThermalDataset(
        pth_path=dataset_path,
        normalize_temp=data_config.get('normalize_temp', True),
        use_log_e=data_config.get('use_log_e', True)
    )
    
    return model, temp_dataset, config, model_type


def gif_to_temperature(gif_path, min_temp=280.0, max_temp=320.0):
    """将GIF转换为温度序列 [H, W, T]"""
    img = Image.open(gif_path)
    frames = []
    
    try:
        while True:
            frame = np.array(img.convert('L')).astype(np.float32)
            frames.append(frame)
            img.seek(img.tell() + 1)
    except EOFError:
        pass
    
    frames = np.stack(frames, axis=-1)  # [H, W, T]
    temperature = (frames / 255.0) * (max_temp - min_temp) + min_temp
    return temperature


def temperature_to_gif(temperature_map, output_path, min_temp=280.0, max_temp=320.0):
    """将温度序列 [H, W, T] 转换为GIF"""
    H, W, T = temperature_map.shape
    
    # 归一化到0-255
    temp_norm = (temperature_map - min_temp) / (max_temp - min_temp)
    temp_norm = np.clip(temp_norm, 0, 1)
    frames_uint8 = (temp_norm * 255).astype(np.uint8)
    
    # 创建GIF
    images = [Image.fromarray(frames_uint8[:, :, t], mode='L') for t in range(T)]
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=100,  # 每帧100ms
        loop=0
    )
    print(f"Saved GIF: {output_path}")


def vae_reconstruct_map(model, temperature_map, temp_dataset, device, batch_size=256, model_type=''):
    """
    对整个温度图进行VAE重建
    
    Returns:
        reconstructed_map: [H, W, T] 重建的温度序列
        effusivity_map: [H, W] e值分布
    """
    H, W, T = temperature_map.shape
    temp_flat = temperature_map.reshape(-1, T)  # [H*W, T]
    
    reconstructed_flat = []
    effusivity_flat = []
    
    is_residual = (model_type == 'timevae1d_hybrid_mamba_transformer_residual')
    
    with torch.no_grad():
        for i in range(0, len(temp_flat), batch_size):
            batch_temp = temp_flat[i:i+batch_size]
            
            # 归一化
            if temp_dataset.normalize_temp:
                batch_temp = (batch_temp - temp_dataset.temp_mean.numpy()) / temp_dataset.temp_std.numpy()
            
            # 转为tensor
            batch_tensor = torch.from_numpy(batch_temp).float().to(device)
            batch_tensor_3d = batch_tensor.unsqueeze(1)  # [B, 1, T]
            
            # VAE前向传播
            model_output = model(batch_tensor_3d)
            if len(model_output) == 5:
                if is_residual:
                    # 残差VAE: (recon, e_pred, (mu, logvar), x_smooth, x_delta)
                    recon, e_pred, _, _, _ = model_output
                else:
                    # Stage-Aware VAE: (recon, e_pred, (mu, logvar), x_smooth, stage_weights)
                    recon, e_pred, _, _, _ = model_output
            else:
                # 标准VAE: (recon, e_pred, (mu, logvar), x_smooth)
                recon, e_pred, _, _ = model_output
            
            # 重建序列反归一化
            recon_temp = recon.squeeze(1).cpu().numpy()  # [B, T]
            if temp_dataset.normalize_temp:
                recon_temp = recon_temp * temp_dataset.temp_std.numpy() + temp_dataset.temp_mean.numpy()
            
            reconstructed_flat.append(recon_temp)
            
            # e值反归一化
            e_values = e_pred.cpu().numpy().flatten()
            if temp_dataset.use_log_e:
                e_values = e_values * temp_dataset.e_std.numpy() + temp_dataset.e_mean.numpy()
                e_values = np.exp(e_values)
            
            effusivity_flat.append(e_values)
    
    reconstructed_flat = np.concatenate(reconstructed_flat, axis=0)
    effusivity_flat = np.concatenate(effusivity_flat, axis=0)
    
    reconstructed_map = reconstructed_flat.reshape(H, W, T)
    effusivity_map = effusivity_flat.reshape(H, W)
    
    return reconstructed_map, effusivity_map


def create_comparison_figure(original_map, reconstructed_map, effusivity_map, output_path):
    """创建对比图：原始中间帧、重建中间帧、e值分布"""
    H, W, T = original_map.shape
    mid_frame = T // 2
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 第一张：原始温度
    im0 = axes[0].imshow(original_map[:, :, mid_frame], cmap='hot', vmin=280, vmax=320)
    axes[0].set_title(f'Original (Frame {mid_frame})', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label='Temperature (K)')
    
    # 第二张：重建温度
    im1 = axes[1].imshow(reconstructed_map[:, :, mid_frame], cmap='hot', vmin=280, vmax=320)
    axes[1].set_title(f'Reconstructed (Frame {mid_frame})', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label='Temperature (K)')
    
    # 第三张：e值分布
    im2 = axes[2].imshow(effusivity_map, cmap='viridis', vmin=0, vmax=10000)
    axes[2].set_title('Thermal Effusivity', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label='e (J·m⁻²·K⁻¹·s⁻¹/²)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison figure: {output_path}")


def main():
    # ==================== 配置区域 ====================
    GIF_PATH = '/home/ziwu/Newpython/lg_exp/Point_lg/gif_test/temp6.gif'
    MODEL_PATH = '/home/ziwu/Newpython/lg_exp/Point_lg/results/model_result/AAA_GFilter_new_residual_dconv16_20251105_165202_timevae_hybrid_res_d128_latent32_mamba2_trans2_head8_ic0.5/best_model.pth'
    CONFIG_PATH = '/home/ziwu/Newpython/lg_exp/Point_lg/results/model_result/AAA_GFilter_new_residual_dconv16_20251105_165202_timevae_hybrid_res_d128_latent32_mamba2_trans2_head8_ic0.5/config_used.yaml'
    OUTPUT_DIR = None  # None则自动生成到GIF同目录
    
    MIN_TEMP = 280.0
    MAX_TEMP = 320.0
    BATCH_SIZE = 4096
    # ==================================================
    
    parser = argparse.ArgumentParser(description='VAE GIF Reconstruction')
    parser.add_argument('--gif', type=str, default=None, help='Input GIF path')
    parser.add_argument('--model', type=str, default=None, help='Model checkpoint path')
    parser.add_argument('--config', type=str, default=None, help='Config YAML path')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--min_temp', type=float, default=None, help='Min temperature (K)')
    parser.add_argument('--max_temp', type=float, default=None, help='Max temperature (K)')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    args = parser.parse_args()
    
    # 使用配置或命令行参数
    gif_path = args.gif if args.gif else GIF_PATH
    model_path = args.model if args.model else MODEL_PATH
    config_path = args.config if args.config else CONFIG_PATH
    output_dir = args.output_dir if args.output_dir else OUTPUT_DIR
    min_temp = args.min_temp if args.min_temp is not None else MIN_TEMP
    max_temp = args.max_temp if args.max_temp is not None else MAX_TEMP
    batch_size = args.batch_size if args.batch_size else BATCH_SIZE
    
    # 设置输出目录
    if output_dir is None:
        output_dir = Path(gif_path).parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    gif_name = Path(gif_path).stem
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Input GIF: {gif_path}")
    print(f"Model: {model_path}")
    print(f"Output directory: {output_dir}")
    
    # 加载模型
    print("\nLoading VAE model...")
    model, temp_dataset, config, model_type = load_vae_model(model_path, config_path, device)
    print(f"Model type: {model_type}")
    print("Model loaded successfully!")
    
    # 读取GIF
    print("\nReading GIF...")
    original_map = gif_to_temperature(gif_path, min_temp, max_temp)
    H, W, T = original_map.shape
    print(f"GIF shape: {H}×{W} pixels, {T} frames")
    print(f"Temperature range: {original_map.min():.2f} - {original_map.max():.2f} K")
    
    # VAE重建
    print("\nReconstructing with VAE...")
    start_time = time.time()
    reconstructed_map, effusivity_map = vae_reconstruct_map(
        model, original_map, temp_dataset, device, batch_size, model_type
    )
    elapsed_time = time.time() - start_time
    print(f"Reconstruction time: {elapsed_time:.2f} seconds")
    print(f"Reconstructed temperature range: {reconstructed_map.min():.2f} - {reconstructed_map.max():.2f} K")
    print(f"Effusivity range: {effusivity_map.min():.2f} - {effusivity_map.max():.2f}")
    
    # 计算重建误差
    mse = np.mean((original_map - reconstructed_map) ** 2)
    mae = np.mean(np.abs(original_map - reconstructed_map))
    print(f"\nReconstruction Error:")
    print(f"  MSE: {mse:.6f} K²")
    print(f"  MAE: {mae:.6f} K")
    
    # 保存结果
    print("\nSaving results...")
    
    # 1. 原始GIF（复制）
    original_gif_path = output_dir / f'{gif_name}_original.gif'
    temperature_to_gif(original_map, original_gif_path, min_temp, max_temp)
    
    # 2. 重建GIF
    reconstructed_gif_path = output_dir / f'{gif_name}_reconstructed.gif'
    temperature_to_gif(reconstructed_map, reconstructed_gif_path, min_temp, max_temp)
    
    # 3. 对比图
    comparison_path = output_dir / f'{gif_name}_comparison.png'
    create_comparison_figure(original_map, reconstructed_map, effusivity_map, comparison_path)
    
    # 4. 保存e值分布为单独的图
    effusivity_fig_path = output_dir / f'{gif_name}_effusivity_map.png'
    plt.figure(figsize=(8, 6))
    im = plt.imshow(effusivity_map, cmap='viridis', vmin=0, vmax=10000)
    plt.title(f'Thermal Effusivity Map\nMean: {effusivity_map.mean():.2f}, Center: {effusivity_map[H//2, W//2]:.2f}', 
              fontsize=12, fontweight='bold')
    plt.colorbar(im, label='e (J·m⁻²·K⁻¹·s⁻¹/²)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(effusivity_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved effusivity map: {effusivity_fig_path}")
    
    print("\n" + "="*60)
    print("Processing complete!")
    print("="*60)
    print(f"Output files:")
    print(f"  - Original GIF: {original_gif_path}")
    print(f"  - Reconstructed GIF: {reconstructed_gif_path}")
    print(f"  - Comparison: {comparison_path}")
    print(f"  - Effusivity map: {effusivity_fig_path}")
    print("="*60)


if __name__ == '__main__':
    main()


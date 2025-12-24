import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import time
import torch
import torch.nn as nn
import numpy as np
import yaml
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import warnings
import random
from scipy.ndimage import gaussian_filter
warnings.filterwarnings('ignore', category=FutureWarning)

# 设置随机种子以确保可复现性
def set_seed(seed=42):
    """设置随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 初始化时设置随机种子
set_seed(42)

from dataset import ThermalDataset
from model import (TimeTransformer, CNN1D, PhysicsInformedCNN, 
                  PhysicsInformedTransformer, EnhancedPhysicsTransformer,
                  MambaPhysicsModel, HybridMambaTransformer,
                  MultiScalePhysicsTransformer, PhysTCN, EnhancedMambaPhysicsModel,
                  TimeVAE1D_Mamba, TimeVAE1D_Mamba_PhysicsDecoder, TimeVAE1D_Transformer,
                  TimeVAE1D_HybridMambaTransformer, TimeVAE1D_HybridMT_Physics4Channel,
                  TimeVAE1D_StageAware, TimeVAE1D_HybridMambaTransformer_Residual, TimeVAE1D_SSTEncoder_Residual)


def _parse_gpu_ids():
    """解析 GPU_IDS 环境变量，返回 GPU ID 列表"""
    def parse_ids(val):
        if val is None:
            return None
        if isinstance(val, (list, tuple)):
            return [int(i) for i in val if str(i).strip() != '']
        if isinstance(val, str):
            items = val.replace(';', ',').split(',')
            return [int(x) for x in items if x.strip().isdigit()]
        return None

    env_ids = parse_ids(os.environ.get('GPU_IDS'))
    if env_ids:
        return env_ids
    
    # 如果没有设置 GPU_IDS，检查 CUDA_VISIBLE_DEVICES
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if visible:
        ids = [int(x) for x in visible.split(',') if x.strip().isdigit()]
        if ids:
            # 如果设置了 CUDA_VISIBLE_DEVICES，返回本地索引
            return list(range(len(ids)))
    
    # 默认返回所有可见的 GPU
    if torch.cuda.is_available():
        return list(range(torch.cuda.device_count()))
    return []


def load_model(model_path, config_path, device, use_multi_gpu=True):
    """加载训练好的模型，支持多GPU"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    model_config = config['model']
    model_type = model_config.get('type', 'transformer').lower()
    
    # 创建模型
    if model_type == 'transformer':
        cfg = model_config['transformer']
        model = TimeTransformer(
            d_model=cfg['d_model'],
            nhead=cfg['nhead'],
            num_layers=cfg['num_layers'],
            dim_feedforward=cfg['dim_feedforward'],
            dropout=cfg['dropout']
        )
    elif model_type == 'cnn1d':
        cfg = model_config['cnn1d']
        model = CNN1D(
            hidden_channels=cfg['hidden_channels'],
            num_layers=cfg['num_layers'],
            kernel_size=cfg['kernel_size'],
            dropout=cfg['dropout']
        )
    elif model_type == 'physics_cnn':
        cfg = model_config.get('physics_cnn', model_config.get('cnn1d', {}))
        model = PhysicsInformedCNN(
            hidden_channels=cfg['hidden_channels'],
            num_layers=cfg['num_layers'],
            kernel_size=cfg['kernel_size'],
            dropout=cfg['dropout']
        )
    elif model_type == 'physics_transformer':
        cfg = model_config.get('physics_transformer', {})
        base = model_config.get('transformer', {})
        total_time = config['data'].get('total_time', cfg.get('total_time', 5.0))
        delta_t = config['data'].get('delta_t', cfg.get('delta_t', 0.02))
        model = PhysicsInformedTransformer(
            d_model=cfg.get('d_model', base.get('d_model', 256)),
            nhead=cfg.get('nhead', base.get('nhead', 8)),
            num_layers=cfg.get('num_layers', base.get('num_layers', 4)),
            dim_feedforward=cfg.get('dim_feedforward', base.get('dim_feedforward', 1024)),
            dropout=cfg.get('dropout', base.get('dropout', 0.1)),
            total_time=total_time,
            delta_t=delta_t
        )
    elif model_type == 'enhanced_physics_transformer':
        cfg = model_config.get('enhanced_physics_transformer', {})
        base = model_config.get('transformer', {})
        total_time = config['data'].get('total_time', cfg.get('total_time', 5.0))
        delta_t = config['data'].get('delta_t', cfg.get('delta_t', 0.02))
        model = EnhancedPhysicsTransformer(
            d_model=cfg.get('d_model', base.get('d_model', 256)),
            nhead=cfg.get('nhead', base.get('nhead', 8)),
            num_layers=cfg.get('num_layers', base.get('num_layers', 4)),
            dim_feedforward=cfg.get('dim_feedforward', base.get('dim_feedforward', 1024)),
            dropout=cfg.get('dropout', base.get('dropout', 0.1)),
            total_time=total_time,
            delta_t=delta_t
        )
    elif model_type == 'mamba_physics':
        cfg = model_config.get('mamba_physics', {})
        total_time = config['data'].get('total_time', cfg.get('total_time', 5.0))
        delta_t = config['data'].get('delta_t', cfg.get('delta_t', 0.02))
        model = MambaPhysicsModel(
            d_model=cfg.get('d_model', 256),
            n_layers=cfg.get('n_layers', 4),
            d_state=cfg.get('d_state', 16),
            d_conv=cfg.get('d_conv', 4),
            expand=cfg.get('expand', 2),
            dropout=cfg.get('dropout', 0.1),
            total_time=total_time,
            delta_t=delta_t
        )
    elif model_type == 'hybrid_mamba_transformer':
        cfg = model_config.get('hybrid_mamba_transformer', {})
        total_time = config['data'].get('total_time', cfg.get('total_time', 5.0))
        delta_t = config['data'].get('delta_t', cfg.get('delta_t', 0.02))
        model = HybridMambaTransformer(
            d_model=cfg.get('d_model', 256),
            n_mamba=cfg.get('n_mamba', 2),
            n_transformer=cfg.get('n_transformer', 2),
            nhead=cfg.get('nhead', 8),
            d_state=cfg.get('d_state', 16),
            d_conv=cfg.get('d_conv', 4),
            expand=cfg.get('expand', 2),
            dropout=cfg.get('dropout', 0.1),
            total_time=total_time,
            delta_t=delta_t
        )
    elif model_type == 'multiscale_physics_transformer':
        cfg = model_config.get('multiscale_physics_transformer', {})
        total_time = config['data'].get('total_time', cfg.get('total_time', 5.0))
        delta_t = config['data'].get('delta_t', cfg.get('delta_t', 0.02))
        model = MultiScalePhysicsTransformer(
            d_model=cfg.get('d_model', 256),
            n_mamba=cfg.get('n_mamba', 3),
            n_transformer=cfg.get('n_transformer', 3),
            nhead=cfg.get('nhead', 8),
            d_state=cfg.get('d_state', 32),
            d_conv=cfg.get('d_conv', 4),
            expand=cfg.get('expand', 2),
            dropout=cfg.get('dropout', 0.1),
            total_time=total_time,
            delta_t=delta_t,
            use_bidirectional=cfg.get('use_bidirectional', True)
        )
    elif model_type == 'phys_tcn':
        cfg = model_config.get('phys_tcn', {})
        total_time = config['data'].get('total_time', cfg.get('total_time', 5.0))
        delta_t = config['data'].get('delta_t', cfg.get('delta_t', 0.02))
        model = PhysTCN(
            channels=cfg.get('channels', 128),
            num_layers=cfg.get('num_layers', 6),
            dilations=cfg.get('dilations', None),
            kernel_size=cfg.get('kernel_size', 7),
            dropout=cfg.get('dropout', 0.1),
            activation=cfg.get('activation', 'glu'),
            total_time=total_time,
            delta_t=delta_t
        )
    elif model_type == 'enhanced_mamba_physics':
        cfg = model_config.get('enhanced_mamba_physics', {})
        total_time = config['data'].get('total_time', cfg.get('total_time', 5.0))
        delta_t = config['data'].get('delta_t', cfg.get('delta_t', 0.02))
        model = EnhancedMambaPhysicsModel(
            d_model=cfg.get('d_model', 256),
            n_layers=cfg.get('n_layers', 6),
            d_state=cfg.get('d_state', 16),
            d_conv=cfg.get('d_conv', 4),
            expand=cfg.get('expand', 2),
            dropout=cfg.get('dropout', 0.2),
            total_time=total_time,
            delta_t=delta_t,
            use_multi_scale=cfg.get('use_multi_scale', True)
        )
    elif model_type == 'timevae1d_mamba':
        cfg = model_config.get('timevae1d_mamba', {})
        total_time = config['data'].get('total_time', 5.0)
        delta_t = config['data'].get('delta_t', 0.02)
        seq_len = int(total_time / delta_t) + 1
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
    elif model_type == 'timevae1d_mamba_physics_decoder':
        cfg = model_config.get('timevae1d_mamba_physics_decoder', {})
        total_time = config['data'].get('total_time', 5.0)
        delta_t = config['data'].get('delta_t', 0.02)
        seq_len = int(total_time / delta_t) + 1
        model = TimeVAE1D_Mamba_PhysicsDecoder(
            seq_len=seq_len,
            C_in=cfg.get('C_in', 1),
            latent_dim=cfg.get('latent_dim', 64),
            d_model=cfg.get('d_model', 128),
            depth=cfg.get('depth', 4),
            d_state=cfg.get('d_state', 16),
            d_conv=cfg.get('d_conv', 4),
            expand=cfg.get('expand', 2),
            delta_t=delta_t,
            total_time=total_time,
            num_physics_basis=cfg.get('num_physics_basis', 8)
        )
    elif model_type == 'timevae1d_transformer':
        cfg = model_config.get('timevae1d_transformer', {})
        total_time = config['data'].get('total_time', 5.0)
        delta_t = config['data'].get('delta_t', 0.02)
        seq_len = int(total_time / delta_t) + 1
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
        total_time = config['data'].get('total_time', 5.0)
        delta_t = config['data'].get('delta_t', 0.02)
        seq_len = int(total_time / delta_t) + 1
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
            decoder_base=cfg.get('decoder_base', 64)
        )
    elif model_type == 'timevae1d_hybridmt_physics4channel':
        cfg = model_config.get('timevae1d_hybridmt_physics4channel', {})
        total_time = config['data'].get('total_time', 5.0)
        delta_t = config['data'].get('delta_t', 0.02)
        seq_len = int(total_time / delta_t) + 1
        model = TimeVAE1D_HybridMT_Physics4Channel(
            seq_len=seq_len,
            C_in=cfg.get('C_in', 4),
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
    elif model_type == 'timevae1d_stageaware':
        cfg = model_config.get('timevae1d_stageaware', {})
        total_time = config['data'].get('total_time', 5.0)
        delta_t = config['data'].get('delta_t', 0.02)
        seq_len = int(total_time / delta_t) + 1
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
    elif model_type == 'timevae1d_hybrid_mamba_transformer_residual':
        cfg = model_config.get('timevae1d_hybrid_mamba_transformer_residual', {})
        total_time = config['data'].get('total_time', 5.0)
        delta_t = config['data'].get('delta_t', 0.02)
        seq_len = int(total_time / delta_t) + 1
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
    elif model_type == 'timevae1d_sst_encoder_residual':
        vae_cfg = model_config.get('timevae1d_sst_encoder_residual', {})
        total_time = config['data'].get('total_time', 5.0)
        delta_t = config['data'].get('delta_t', 0.02)
        seq_len = int(total_time / delta_t) + 1
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
        raise ValueError(f"Unknown model type: {model_type}")
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # 处理 DataParallel 包装的模型
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # 多GPU支持：使用 DataParallel
    main_device = device
    if use_multi_gpu and device.type == 'cuda' and torch.cuda.is_available():
        gpu_ids = _parse_gpu_ids()
        if len(gpu_ids) > 1:
            # 确保主设备是第一个 GPU
            main_device_id = int(gpu_ids[0])
            try:
                torch.cuda.set_device(main_device_id)
                main_device = torch.device(f'cuda:{main_device_id}')
                model = model.to(main_device)
            except Exception:
                pass
            
            # 使用 DataParallel 包装模型
            model = nn.DataParallel(model, device_ids=gpu_ids, output_device=main_device_id)
            print(f"Using DataParallel on GPUs: {gpu_ids} (main cuda:{main_device_id})")
        elif len(gpu_ids) == 1:
            main_device = torch.device(f'cuda:{gpu_ids[0]}')
            model = model.to(main_device)
            print(f"Using single GPU: cuda:{gpu_ids[0]}")
    
    # 加载归一化参数
    data_config = config['data']
    dataset_path = data_config['dataset_path']
    temp_dataset = ThermalDataset(
        pth_path=dataset_path,
        normalize_temp=data_config.get('normalize_temp', True),
        use_log_e=data_config.get('use_log_e', True)
    )
    
    return model, temp_dataset, config, main_device


def gif_to_temperature(gif_path, min_temp=280.0, max_temp=320.0):
    """将 GIF 转换为温度序列 (H, W, T)"""
    img = Image.open(gif_path)
    frames = []
    
    try:
        while True:
            frame = np.array(img.convert('L')).astype(np.float32)  # 转为灰度
            frames.append(frame)
            img.seek(img.tell() + 1)
    except EOFError:
        pass
    
    frames = np.stack(frames, axis=-1)  # (H, W, T)
    # 像素值 0-255 映射到 280-320K
    temperature = (frames / 255.0) * (max_temp - min_temp) + min_temp
    return temperature


def apply_avg_pooling(temperature_map, pool_size=2, stride=2):
    """
    对温度图的每一帧应用均值池化，降低分辨率
    
    Args:
        temperature_map: [H, W, T] 温度序列
        pool_size: 池化核大小（默认2×2）
        stride: 步长（默认2）
    
    Returns:
        pooled_map: [H', W', T] 池化后的温度序列
    """
    H, W, T = temperature_map.shape
    
    # 计算输出尺寸
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1
    
    print(f"  Applying {pool_size}×{pool_size} average pooling (stride={stride})...")
    print(f"  Resolution: {H}×{W} → {H_out}×{W_out} (reduced to {100*H_out*W_out/(H*W):.1f}%)")
    
    # 对每一帧进行池化
    pooled_frames = []
    for t in range(T):
        frame = temperature_map[:, :, t]  # [H, W]
        
        # 均值池化
        pooled_frame = np.zeros((H_out, W_out), dtype=frame.dtype)
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * stride
                w_start = j * stride
                pool_region = frame[h_start:h_start+pool_size, w_start:w_start+pool_size]
                pooled_frame[i, j] = pool_region.mean()
        
        pooled_frames.append(pooled_frame)
    
    pooled_map = np.stack(pooled_frames, axis=-1)  # [H_out, W_out, T]
    return pooled_map


def apply_gaussian_filter(temperature_map, kernel_size=3, sigma=None):
    """
    对温度图的每一帧应用高斯滤波（中间权重最大，四周权重小）
    
    Args:
        temperature_map: [H, W, T] 温度序列
        kernel_size: 卷积核尺寸（奇数，如 3, 5, 7, 9...）
        sigma: 高斯核标准差。如果为 None，则根据 kernel_size 自动计算：
               sigma = (kernel_size - 1) / 6，确保权重在边界处足够小
    
    Returns:
        filtered_map: [H, W, T] 滤波后的温度序列
    """
    H, W, T = temperature_map.shape
    
    # 确保 kernel_size 为奇数
    if kernel_size % 2 == 0:
        kernel_size += 1
        print(f"Warning: kernel_size must be odd, adjusted to {kernel_size}")
    
    # 自动计算 sigma（如果未指定）
    if sigma is None:
        sigma = (kernel_size - 1)   # 标准做法：3*sigma 覆盖大部分核区域
    
    print(f"  Gaussian filter: kernel_size={kernel_size}, sigma={sigma:.3f}")
    
    # 对每一帧进行高斯滤波
    filtered_frames = []
    for t in range(T):
        frame = temperature_map[:, :, t]  # [H, W]
        # 使用 gaussian_filter 进行高斯滤波
        filtered_frame = gaussian_filter(frame, sigma=sigma, mode='reflect')
        filtered_frames.append(filtered_frame)
    
    filtered_map = np.stack(filtered_frames, axis=-1)  # [H, W, T]
    return filtered_map


def process_single_gif(model, gif_path, temp_dataset, device, args, is_vae=False):
    """
    处理单个GIF文件，返回预测结果和统计信息
    
    Args:
        model: 预测模型
        gif_path: GIF文件路径
        temp_dataset: 数据集（用于归一化）
        device: 设备
        args: 命令行参数
        is_vae: 是否为VAE模型
        
    Returns:
        dict: 包含预测结果、输出路径、统计信息等
    """
    gif_path = Path(gif_path)
    if not gif_path.exists():
        return {'error': f'GIF file not found: {gif_path}'}
    
    try:
        # 读取 GIF
        print(f"\n{'='*60}")
        print(f"Processing: {gif_path.name}")
        print(f"{'='*60}")
        temperature_map = gif_to_temperature(str(gif_path), args.min_temp, args.max_temp)
        H_orig, W_orig, T = temperature_map.shape
        print(f"GIF shape: {H_orig}×{W_orig} pixels, {T} frames")
        print(f"Temperature range: {temperature_map.min():.2f} - {temperature_map.max():.2f} K")
        
        # 步骤1: 均值池化降低分辨率（减少计算量）
        if args.pool_size > 1:
            temperature_map = apply_avg_pooling(temperature_map, pool_size=args.pool_size, stride=args.pool_stride)
            H, W, T = temperature_map.shape
            print(f"After pooling temperature range: {temperature_map.min():.2f} - {temperature_map.max():.2f} K")
        else:
            H, W = H_orig, W_orig
        
        # 步骤2: 应用高斯滤波（预处理）
        print(f"Applying Gaussian filter (kernel_size={args.kernel_size}) to each frame...")
        temperature_map = apply_gaussian_filter(temperature_map, kernel_size=args.kernel_size, sigma=args.sigma)
        print(f"Filtered temperature range: {temperature_map.min():.2f} - {temperature_map.max():.2f} K")
        
        # 预测
        print("Predicting effusivity map (pixel-by-pixel)...")
        start_time = time.time()
        effusivity_map = predict_effusivity_map(model, temperature_map, temp_dataset, device, args.batch_size, is_vae=is_vae)
        prediction_time = time.time() - start_time
        
        print(f"Effusivity range: {effusivity_map.min():.2f} - {effusivity_map.max():.2f}")
        print(f"Prediction time: {prediction_time:.3f} seconds")
        
        # 计算中心像素的 e 值
        center_h, center_w = H // 2, W // 2
        center_e = effusivity_map[center_h, center_w]
        mean_e = effusivity_map.mean()
        
        print(f"Center pixel e value: {center_e:.2f}")
        print(f"Mean e value: {mean_e:.2f}")
        
        # 读取原始温度图（未处理前的）用于第一张图
        temperature_map_original = gif_to_temperature(str(gif_path), args.min_temp, args.max_temp)
        
        # 绘制热力图
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 固定颜色范围，便于对比
        TEMP_MIN, TEMP_MAX = 280.0, 320.0  # 温度范围固定
        E_MIN, E_MAX = 0.0, 12000.0  # e值范围固定
        
        # 选择中间帧索引
        mid_frame_idx = T // 2
        
        # 第一张图：原始中间帧（未池化、未滤波）
        frame_original = temperature_map_original[:, :, mid_frame_idx]
        im0 = axes[0].imshow(frame_original, cmap='hot', vmin=TEMP_MIN, vmax=TEMP_MAX)
        title0 = f'Frame {mid_frame_idx} (Original {H_orig}×{W_orig}, K)'
        axes[0].set_title(title0, fontsize=14)
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        
        # 第二张图：处理后的中间帧（已池化+高斯滤波）
        frame_filtered = temperature_map[:, :, mid_frame_idx]
        im1 = axes[1].imshow(frame_filtered, cmap='hot', vmin=TEMP_MIN, vmax=TEMP_MAX)
        if args.pool_size > 1:
            title1 = f'Frame {mid_frame_idx} (Pooled {H}×{W} + Filtered, K)'
        else:
            title1 = f'Frame {mid_frame_idx} (Filtered {H}×{W}, K)'
        axes[1].set_title(title1, fontsize=14)
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # 第三张图：热扩散系数（固定范围0-10000）
        im2 = axes[2].imshow(effusivity_map, cmap=args.effusivity_cmap, vmin=E_MIN, vmax=E_MAX)
        axes[2].set_title(f'Thermal Effusivity {H}×{W} (J·m⁻²·K⁻¹·s⁻¹/²)', fontsize=14)
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        # 确定输出路径
        # 只有在单个文件模式且指定了输出路径时才使用指定路径
        use_custom_output = (args.output and 
                           not args.gif_list and 
                           not args.gif_dir and 
                           str(gif_path) == args.gif)
        
        if use_custom_output:
            output_path = args.output
        else:
            # 批量模式或未指定输出，自动生成
            gif_dir = gif_path.parent
            gif_name = gif_path.stem
            if args.pool_size > 1:
                output_path = str(gif_dir / f'{gif_name}_pred_e{center_e:.2f}_pool{args.pool_size}_kernel{args.kernel_size}_sigma{args.sigma}.png')
            else:
                output_path = str(gif_dir / f'{gif_name}_pred_e{center_e:.2f}_kernel{args.kernel_size}_sigma{args.sigma}.png')
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)  # 释放内存
        print(f"Saved to: {output_path}")
        
        return {
            'gif_path': str(gif_path),
            'output_path': output_path,
            'center_e': center_e,
            'mean_e': mean_e,
            'prediction_time': prediction_time,
            'image_size': (H, W),
            'frames': T,
            'success': True
        }
    except Exception as e:
        print(f"Error processing {gif_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'error': str(e), 'gif_path': str(gif_path), 'success': False}


def predict_effusivity_map(model, temperature_map, temp_dataset, device, batch_size=256, is_vae=False):
    """
    对整个温度图预测热扩散系数（逐像素预测）
    
    Args:
        model: 预测模型
        temperature_map: [H, W, T] 温度序列
        temp_dataset: 数据集（用于归一化）
        device: 设备
        batch_size: 批量大小
        is_vae: 是否为VAE模型
    
    Returns:
        effusivity_map: [H, W] 热扩散系数分布
    """
    H, W, T = temperature_map.shape
    
    temp_flat = temperature_map.reshape(-1, T)  # (H*W, T)
    
    predictions = []
    with torch.no_grad():
        for i in range(0, len(temp_flat), batch_size):
            batch_temp = temp_flat[i:i+batch_size]
            
            # 归一化
            if temp_dataset.normalize_temp:
                batch_temp = (batch_temp - temp_dataset.temp_mean.numpy()) / temp_dataset.temp_std.numpy()
            
            # 转为 tensor
            batch_tensor = torch.from_numpy(batch_temp).float().to(device)
            
            # 预测
            if is_vae:
                # VAE模型：需要 [B, C_in, T] 输入（C_in=1或4）
                # 使用forward方法，它会自动处理1->4通道的转换
                batch_tensor_3d = batch_tensor.unsqueeze(1)  # [B, 1, T]
                
                # 获取模型的实际模块（处理DataParallel）
                actual_model = model.module if isinstance(model, nn.DataParallel) else model
                
                # 调用forward，但只取e_pred（不需要重建和隐变量）
                # forward返回: 
                # - 标准VAE: (recon, e_pred, (mu, logvar), x_smooth)
                # - 残差VAE: (recon, e_pred, (mu, logvar), x_smooth, x_delta)
                model_output = actual_model(batch_tensor_3d)
                if len(model_output) == 5:
                    # 残差VAE模型
                    _, e_pred, _, _, _ = model_output
                else:
                    # 标准VAE模型
                    _, e_pred, _, _ = model_output
                pred = e_pred.cpu().numpy().flatten()
            else:
                # 普通模型：[B, T] 输入
                pred = model(batch_tensor).cpu().numpy().flatten()
            
            # 反归一化
            if temp_dataset.use_log_e:
                pred = pred * temp_dataset.e_std.numpy() + temp_dataset.e_mean.numpy()
                pred = np.exp(pred)
            
            predictions.append(pred)
    
    predictions = np.concatenate(predictions)
    effusivity_map = predictions.reshape(H, W)
    return effusivity_map


def main():
    # ==================== 配置区域 ====================
    # 直接在这里修改配置，直接运行程序即可
    # ⭐ 批量模式：设置 GIF_DIR 或 GIF_LIST，GIF_PATH 将被忽略
    # ⭐ 单文件模式：设置 GIF_PATH，GIF_DIR 和 GIF_LIST 设为 None
    
    GIF_PATH = '/home/ziwu/Newpython/lg_exp/Point_lg/gif_test/newnewnew_30hz/500_video.gif'  # 单个GIF文件路径
    GIF_DIR = '/home/ziwu/Newpython/lg_exp/Point_lg/edge_fix/Real_data/thermal_design_08'  # 或：GIF目录（批量模式，处理目录下所有.gif）
    # GIF_DIR = None
    GIF_LIST = None  # 或：GIF列表文件路径（批量模式，文本文件，每行一个GIF路径）
    GIF_SUFFIX = '_crop'  # 或：指定GIF文件后缀（仅用于--gif_dir模式），例如 '_crop' 表示只处理 '*_crop.gif' 文件，None表示处理所有 '.gif'
    
    # MODEL_PATH = '/home/ziwu/Newpython/lg_exp/Point_lg/results/model_result/Useful_A_dropout0__20251108_131250_timevae_hybrid_res_d256_latent256_mamba2_trans2_head4_ic0.5/best_model_epoch_493.pth'
    # CONFIG_PATH = '/home/ziwu/Newpython/lg_exp/Point_lg/results/model_result/Useful_A_dropout0__20251108_131250_timevae_hybrid_res_d256_latent256_mamba2_trans2_head4_ic0.5/config_used.yaml'
    MODEL_PATH = '/home/ziwu/Newpython/lg_exp/Point_lg/results/model_result/Useful_A_dropout0__20251108_131250_timevae_hybrid_res_d256_latent256_mamba2_trans2_head4_ic0.5/best_model_epoch_493.pth'
    CONFIG_PATH = '/home/ziwu/Newpython/lg_exp/Point_lg/results/model_result/Useful_A_dropout0__20251108_131250_timevae_hybrid_res_d256_latent256_mamba2_trans2_head4_ic0.5/config_used.yaml'

    OUTPUT_PATH = None  # None则自动生成，或指定如 'results/my_output.png'（仅单文件模式）
    
    MIN_TEMP = 280.0
    MAX_TEMP = 320.0
    BATCH_SIZE = 4096*5
    
    # 预处理参数
    POOL_SIZE = 1          # 均值池化核大小（2表示2×2池化，降低到原分辨率的25%）
    POOL_STRIDE = 1        # 池化步长（通常等于pool_size）
    GAUSSIAN_KERNEL_SIZE = 5  # 高斯滤波卷积核尺寸（奇数，如 3, 5, 7, 9...） 5 4!!!
    GAUSSIAN_SIGMA = 4    # 高斯滤波标准差
    
    # 可视化参数
    EFFUSIVITY_CMAP = 'viridis'  # e值分布图的颜色条，可选: 'hot', 'plasma', 'inferno', 'magma', 'jet', 'coolwarm', 'RdYlBu', 'viridis'等
    
    SUMMARY_CSV = None  # 批量模式总结CSV保存路径（可选，如 'batch_results.csv'）
    # ================================================
    
    parser = argparse.ArgumentParser(description='Predict thermal effusivity from GIF temperature sequences')
    parser.add_argument('--gif', type=str, default=None, 
                        help='Path to input GIF file (single file mode)')
    parser.add_argument('--gif_list', type=str, default=None,
                        help='Path to text file containing list of GIF paths (one per line, batch mode)')
    parser.add_argument('--gif_dir', type=str, default=None,
                        help='Directory containing GIF files (batch mode, will process all .gif files)')
    parser.add_argument('--gif_suffix', type=str, default=None,
                        help='GIF file suffix filter (only for --gif_dir mode), e.g., "_crop" to process only "*_crop.gif" files. None means all .gif files')
    parser.add_argument('--model', type=str, default=None, help='Path to model checkpoint (.pth)')
    parser.add_argument('--config', type=str, default=None, help='Path to config.yaml')
    parser.add_argument('--output', type=str, default=None, 
                        help='Output image path (only for single file mode)')
    parser.add_argument('--min_temp', type=float, default=None, help='Min temperature (K)')
    parser.add_argument('--max_temp', type=float, default=None, help='Max temperature (K)')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size for prediction')
    parser.add_argument('--pool_size', type=int, default=None, 
                        help='Average pooling kernel size (e.g., 2 for 2×2 pooling, 1 to disable)')
    parser.add_argument('--pool_stride', type=int, default=None, 
                        help='Average pooling stride (usually equals pool_size)')
    parser.add_argument('--kernel_size', type=int, default=None, 
                        help='Gaussian filter kernel size (odd number, e.g., 3, 5, 7, 9...)')
    parser.add_argument('--sigma', type=float, default=None, help='Gaussian filter sigma')
    parser.add_argument('--effusivity_cmap', type=str, default=None,
                        help='Colormap for effusivity map (e.g., hot, plasma, inferno, magma, jet, coolwarm, RdYlBu, viridis)')
    parser.add_argument('--summary', type=str, default=None,
                        help='Path to save batch processing summary CSV file (optional)')
    args = parser.parse_args()
    
    # 如果命令行参数未指定，使用配置区域的默认值
    if args.gif is None:
        args.gif = GIF_PATH
    if args.gif_list is None:
        args.gif_list = GIF_LIST
    if args.gif_dir is None:
        args.gif_dir = GIF_DIR
    if args.gif_suffix is None:
        args.gif_suffix = GIF_SUFFIX
    if args.model is None:
        args.model = MODEL_PATH
    if args.config is None:
        args.config = CONFIG_PATH
    if args.output is None:
        args.output = OUTPUT_PATH
    if args.min_temp is None:
        args.min_temp = MIN_TEMP
    if args.max_temp is None:
        args.max_temp = MAX_TEMP
    if args.batch_size is None:
        args.batch_size = BATCH_SIZE
    if args.pool_size is None:
        args.pool_size = POOL_SIZE
    if args.pool_stride is None:
        args.pool_stride = POOL_STRIDE
    if args.kernel_size is None:
        args.kernel_size = GAUSSIAN_KERNEL_SIZE
    if args.sigma is None:
        args.sigma = GAUSSIAN_SIGMA
    if args.effusivity_cmap is None:
        args.effusivity_cmap = EFFUSIVITY_CMAP
    if args.summary is None:
        args.summary = SUMMARY_CSV
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 确保随机种子已设置（推理时保持一致）
    set_seed(42)
    print("Random seed set to 42 for reproducibility")
    
    # 加载模型（只加载一次，用于所有GIF）
    print("Loading model...")
    model, temp_dataset, config, main_device = load_model(args.model, args.config, device)
    print("Model loaded successfully")
    device = main_device  # 更新 device 为主设备
    
    # 确保模型处于eval模式（禁用dropout等随机操作）
    model.eval()
    
    # 判断是否为VAE模型
    model_type = config['model'].get('type', '').lower()
    is_vae = (model_type == 'timevae1d_mamba' or 
              model_type == 'timevae1d_mamba_physics_decoder' or 
              model_type == 'timevae1d_transformer' or
              model_type == 'timevae1d_hybrid_mamba_transformer' or
              model_type == 'timevae1d_hybridmt_physics4channel' or
              model_type == 'timevae1d_stageaware' or
              model_type == 'timevae1d_hybrid_mamba_transformer_residual' or
              model_type == 'timevae1d_sst_encoder_residual')
    
    # 确定要处理的GIF文件列表
    gif_list = []
    
    if args.gif_list:
        # 从文本文件读取GIF列表
        print(f"Reading GIF list from: {args.gif_list}")
        with open(args.gif_list, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    gif_list.append(line)
        print(f"Found {len(gif_list)} GIF files in list")
    elif args.gif_dir:
        # 从目录中查找GIF文件
        gif_dir = Path(args.gif_dir)
        print(f"Scanning directory: {gif_dir}")
        if args.gif_suffix:
            # 如果指定了后缀，只匹配带该后缀的GIF文件
            pattern = f'*{args.gif_suffix}.gif'
            print(f"  Filter pattern: {pattern}")
            gif_list = [str(p) for p in gif_dir.glob(pattern)]
        else:
            # 默认匹配所有GIF文件
            gif_list = [str(p) for p in gif_dir.glob('*.gif')]
        gif_list.sort()  # 按文件名排序
        print(f"Found {len(gif_list)} GIF files in directory")
    else:
        # 单个文件模式
        gif_list = [args.gif]
    
    if not gif_list:
        print("Error: No GIF files found!")
        return
    
    # 批量处理
    if len(gif_list) > 1:
        print(f"\n{'='*60}")
        print(f"BATCH PROCESSING MODE: {len(gif_list)} GIF files")
        print(f"{'='*60}")
        print(f"Model: {args.model}")
        print(f"Batch size: {args.batch_size}")
        gpu_info = "Multi-GPU" if isinstance(model, nn.DataParallel) else f"Single {device}"
        print(f"Device: {gpu_info}")
        print(f"{'='*60}\n")
    
    results = []
    total_start_time = time.time()
    
    for idx, gif_path in enumerate(gif_list, 1):
        if len(gif_list) > 1:
            print(f"\n[{idx}/{len(gif_list)}] ", end='')
        
        result = process_single_gif(model, gif_path, temp_dataset, device, args, is_vae=is_vae)
        results.append(result)
    
    total_time = time.time() - total_start_time
    
    # 批量处理总结
    if len(gif_list) > 1:
        successful = [r for r in results if r.get('success', False)]
        failed = [r for r in results if not r.get('success', False)]
        
        print(f"\n{'='*60}")
        print("BATCH PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Total GIFs:        {len(gif_list)}")
        print(f"Successful:        {len(successful)}")
        print(f"Failed:            {len(failed)}")
        print(f"Total time:        {total_time:.2f} seconds")
        print(f"Average time:      {total_time/len(gif_list):.2f} seconds per GIF")
        
        if successful:
            center_es = [r['center_e'] for r in successful]
            mean_es = [r['mean_e'] for r in successful]
            times = [r['prediction_time'] for r in successful]
            print(f"\nCenter e range:    {min(center_es):.2f} - {max(center_es):.2f}")
            print(f"Mean e range:      {min(mean_es):.2f} - {max(mean_es):.2f}")
            print(f"Avg prediction:    {np.mean(times):.2f} seconds")
        
        if failed:
            print(f"\nFailed files:")
            for r in failed:
                print(f"  - {Path(r.get('gif_path', 'unknown')).name}: {r.get('error', 'Unknown error')}")
        
        # 保存总结CSV（如果指定）
        if args.summary:
            import csv
            summary_path = Path(args.summary)
            with open(summary_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['GIF_Path', 'Output_Path', 'Center_e', 'Mean_e', 'Prediction_Time', 
                                'Image_Size_H', 'Image_Size_W', 'Frames', 'Success', 'Error'])
                for r in results:
                    writer.writerow([
                        r.get('gif_path', ''),
                        r.get('output_path', ''),
                        r.get('center_e', ''),
                        r.get('mean_e', ''),
                        r.get('prediction_time', ''),
                        r.get('image_size', (0, 0))[0] if r.get('image_size') else '',
                        r.get('image_size', (0, 0))[1] if r.get('image_size') else '',
                        r.get('frames', ''),
                        'Yes' if r.get('success', False) else 'No',
                        r.get('error', '')
                    ])
            print(f"\nSummary saved to: {summary_path}")
        
        print("="*60)


if __name__ == '__main__':
    main()


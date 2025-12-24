"""
从已训练的checkpoint加载模型并执行预测保存
用于修复bug后，直接使用之前训练好的模型进行预测保存

使用方法：
1. 直接在代码中设置下面的参数，然后运行：python predict_from_checkpoint.py
2. 或者使用命令行参数：python predict_from_checkpoint.py --config config.yaml --checkpoint path/to/model.pth
"""

# ==================== 直接在这里设置参数 ====================
# 如果设置了这些变量，将优先使用这些值，而不是命令行参数
CONFIG_PATH = "/home/ziwu/Newpython/lg_exp/Point_lg/results/model_result/AAA_GFilter_20251104_165849_timevae_mamba_d256_latent128_layer4/config_used.yaml"  # 配置文件路径
CHECKPOINT_PATH = "/home/ziwu/Newpython/lg_exp/Point_lg/results/model_result/AAA_GFilter_20251104_165849_timevae_mamba_d256_latent128_layer4/best_model_epoch_977.pth"  # checkpoint路径
TASK_NAME = "AAA_GFilter_20251104_165849_timevae_mamba_d256_latent128_layer4"  # 任务名称（用于保存结果，如果不设置会从config中读取）
# ============================================================
# 
# 保存位置说明：
# 1. CSV预测结果：results/test_result/{TASK_NAME}/all_predictions_{时间戳}.csv
# 2. VAE样本详细信息：results/test_result/{TASK_NAME}/samples/ 目录下
#    - 每个样本一个 .txt 文件（时间序列数据）
#    - 每个样本一个 .yaml 文件（参数信息）
# ============================================================

import os
import sys
import torch
import torch.nn as nn
import yaml
import csv
import json
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader, random_split
import argparse

# 导入必要的类和函数
from dataset import ThermalDataset

# 从train.py导入辅助函数
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from train import (
        load_config, set_seed, get_sequence_params, _parse_gpu_ids,
        _get_model_state_dict
    )
except ImportError:
    # 如果导入失败，直接定义这些函数
    def load_config(config_path='config.yaml'):
        """Load config file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def set_seed(seed: int):
        """Set random seed for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def get_sequence_params(config):
        """从config中获取时间序列参数"""
        data_config = config.get('data', {})
        total_time = float(data_config.get('total_time', 5.0))
        delta_t = float(data_config.get('delta_t', 0.02))
        seq_len = int(data_config.get('seq_len', int(total_time / delta_t) + 1))
        return total_time, delta_t, seq_len
    
    def _parse_gpu_ids(config):
        """Return list of gpu ids from env GPU_IDS only (config ignored)."""
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
        return env_ids or []
from model import (
    TimeTransformer, CNN1D, PhysicsInformedCNN, PhysicsInformedTransformer,
    EnhancedPhysicsTransformer, MambaPhysicsModel, HybridMambaTransformer,
    MultiScalePhysicsTransformer, PhysTCN, EnhancedMambaPhysicsModel,
    TimeVAE1D_Mamba, TimeVAE1D_Mamba_PhysicsDecoder, TimeVAE1D_Transformer,
    TimeVAE1D_HybridMambaTransformer, TimeVAE1D_HybridMT_Physics4Channel,
    TimeVAE1D_StageAware
)


def load_model_from_config(config, device, total_time, delta_t, seq_len):
    """根据config创建模型（与train.py中的逻辑一致）"""
    model_config = config['model']
    model_type = model_config.get('type', 'transformer').lower()
    
    if model_type == 'transformer':
        trans_cfg = model_config['transformer']
        model = TimeTransformer(
            d_model=trans_cfg['d_model'],
            nhead=trans_cfg['nhead'],
            num_layers=trans_cfg['num_layers'],
            dim_feedforward=trans_cfg['dim_feedforward'],
            dropout=trans_cfg['dropout'],
            total_time=total_time,
            delta_t=delta_t
        ).to(device)
    elif model_type == 'cnn1d':
        cnn_cfg = model_config['cnn1d']
        model = CNN1D(
            hidden_channels=cnn_cfg['hidden_channels'],
            num_layers=cnn_cfg['num_layers'],
            kernel_size=cnn_cfg['kernel_size'],
            dropout=cnn_cfg['dropout'],
            total_time=total_time,
            delta_t=delta_t
        ).to(device)
    elif model_type == 'physics_cnn':
        phys_cfg = model_config.get('physics_cnn', model_config.get('cnn1d', {}))
        model = PhysicsInformedCNN(
            hidden_channels=phys_cfg.get('hidden_channels', 64),
            num_layers=phys_cfg.get('num_layers', 3),
            kernel_size=phys_cfg.get('kernel_size', 5),
            dropout=phys_cfg.get('dropout', 0.2),
            total_time=total_time,
            delta_t=delta_t
        ).to(device)
    elif model_type == 'physics_transformer':
        pt_cfg = model_config.get('physics_transformer', {})
        base = model_config.get('transformer', {})
        model = PhysicsInformedTransformer(
            d_model=pt_cfg.get('d_model', base.get('d_model', 256)),
            nhead=pt_cfg.get('nhead', base.get('nhead', 8)),
            num_layers=pt_cfg.get('num_layers', base.get('num_layers', 4)),
            dim_feedforward=pt_cfg.get('dim_feedforward', base.get('dim_feedforward', 1024)),
            dropout=pt_cfg.get('dropout', base.get('dropout', 0.1)),
            total_time=total_time,
            delta_t=delta_t
        ).to(device)
    elif model_type == 'enhanced_physics_transformer':
        ept_cfg = model_config.get('enhanced_physics_transformer', {})
        base = model_config.get('transformer', {})
        model = EnhancedPhysicsTransformer(
            d_model=ept_cfg.get('d_model', base.get('d_model', 256)),
            nhead=ept_cfg.get('nhead', base.get('nhead', 8)),
            num_layers=ept_cfg.get('num_layers', base.get('num_layers', 4)),
            dim_feedforward=ept_cfg.get('dim_feedforward', base.get('dim_feedforward', 1024)),
            dropout=ept_cfg.get('dropout', base.get('dropout', 0.1)),
            total_time=total_time,
            delta_t=delta_t
        ).to(device)
    elif model_type == 'mamba_physics':
        mamba_cfg = model_config.get('mamba_physics', {})
        model = MambaPhysicsModel(
            d_model=mamba_cfg.get('d_model', 256),
            n_layers=mamba_cfg.get('n_layers', 4),
            d_state=mamba_cfg.get('d_state', 16),
            d_conv=mamba_cfg.get('d_conv', 4),
            expand=mamba_cfg.get('expand', 2),
            dropout=mamba_cfg.get('dropout', 0.1),
            total_time=total_time,
            delta_t=delta_t
        ).to(device)
    elif model_type == 'hybrid_mamba_transformer':
        hybrid_cfg = model_config.get('hybrid_mamba_transformer', {})
        model = HybridMambaTransformer(
            d_model=hybrid_cfg.get('d_model', 256),
            n_mamba=hybrid_cfg.get('n_mamba', 2),
            n_transformer=hybrid_cfg.get('n_transformer', 2),
            nhead=hybrid_cfg.get('nhead', 8),
            d_state=hybrid_cfg.get('d_state', 16),
            d_conv=hybrid_cfg.get('d_conv', 4),
            expand=hybrid_cfg.get('expand', 2),
            dropout=hybrid_cfg.get('dropout', 0.1),
            total_time=total_time,
            delta_t=delta_t
        ).to(device)
    elif model_type == 'phys_tcn':
        tcn_cfg = model_config.get('phys_tcn', {})
        model = PhysTCN(
            channels=tcn_cfg.get('channels', 128),
            num_layers=tcn_cfg.get('num_layers', 6),
            dilations=tcn_cfg.get('dilations', None),
            kernel_size=tcn_cfg.get('kernel_size', 7),
            dropout=tcn_cfg.get('dropout', 0.1),
            activation=tcn_cfg.get('activation', 'glu'),
            total_time=total_time,
            delta_t=delta_t
        ).to(device)
    elif model_type == 'enhanced_mamba_physics':
        enhanced_cfg = model_config.get('enhanced_mamba_physics', {})
        model = EnhancedMambaPhysicsModel(
            d_model=enhanced_cfg.get('d_model', 256),
            n_layers=enhanced_cfg.get('n_layers', 6),
            d_state=enhanced_cfg.get('d_state', 16),
            d_conv=enhanced_cfg.get('d_conv', 4),
            expand=enhanced_cfg.get('expand', 2),
            dropout=enhanced_cfg.get('dropout', 0.2),
            total_time=total_time,
            delta_t=delta_t,
            use_multi_scale=enhanced_cfg.get('use_multi_scale', True)
        ).to(device)
    elif model_type == 'timevae1d_mamba':
        vae_cfg = model_config.get('timevae1d_mamba', {})
        model = TimeVAE1D_Mamba(
            seq_len=seq_len,
            C_in=vae_cfg.get('C_in', 1),
            latent_dim=vae_cfg.get('latent_dim', 64),
            d_model=vae_cfg.get('d_model', 128),
            depth=vae_cfg.get('depth', 4),
            d_state=vae_cfg.get('d_state', 16),
            d_conv=vae_cfg.get('d_conv', 4),
            expand=vae_cfg.get('expand', 2),
            decoder_base=vae_cfg.get('decoder_base', 64),
            total_time=total_time,
            delta_t=delta_t
        ).to(device)
    elif model_type == 'timevae1d_mamba_physics_decoder':
        vae_cfg = model_config.get('timevae1d_mamba_physics_decoder', {})
        model = TimeVAE1D_Mamba_PhysicsDecoder(
            seq_len=seq_len,
            C_in=vae_cfg.get('C_in', 1),
            latent_dim=vae_cfg.get('latent_dim', 64),
            d_model=vae_cfg.get('d_model', 128),
            depth=vae_cfg.get('depth', 4),
            d_state=vae_cfg.get('d_state', 16),
            d_conv=vae_cfg.get('d_conv', 4),
            expand=vae_cfg.get('expand', 2),
            delta_t=delta_t,
            total_time=total_time,
            num_physics_basis=vae_cfg.get('num_physics_basis', 8)
        ).to(device)
    elif model_type == 'timevae1d_transformer':
        vae_cfg = model_config.get('timevae1d_transformer', {})
        model = TimeVAE1D_Transformer(
            seq_len=seq_len,
            C_in=vae_cfg.get('C_in', 1),
            latent_dim=vae_cfg.get('latent_dim', 64),
            d_model=vae_cfg.get('d_model', 128),
            nhead=vae_cfg.get('nhead', 8),
            num_layers=vae_cfg.get('num_layers', 4),
            dim_feedforward=vae_cfg.get('dim_feedforward', 512),
            dropout=vae_cfg.get('dropout', 0.2),
            decoder_base=vae_cfg.get('decoder_base', 64),
            total_time=total_time,
            delta_t=delta_t
        ).to(device)
    elif model_type == 'timevae1d_hybrid_mamba_transformer':
        vae_cfg = model_config.get('timevae1d_hybrid_mamba_transformer', {})
        model = TimeVAE1D_HybridMambaTransformer(
            seq_len=seq_len,
            C_in=vae_cfg.get('C_in', 1),
            latent_dim=vae_cfg.get('latent_dim', 64),
            d_model=vae_cfg.get('d_model', 128),
            n_mamba=vae_cfg.get('n_mamba', 2),
            n_transformer=vae_cfg.get('n_transformer', 2),
            nhead=vae_cfg.get('nhead', 8),
            d_state=vae_cfg.get('d_state', 16),
            d_conv=vae_cfg.get('d_conv', 4),
            expand=vae_cfg.get('expand', 2),
            dropout=vae_cfg.get('dropout', 0.2),
            decoder_base=vae_cfg.get('decoder_base', 64),
            total_time=total_time,
            delta_t=delta_t
        ).to(device)
    elif model_type == 'timevae1d_hybridmt_physics4channel':
        vae_cfg = model_config.get('timevae1d_hybridmt_physics4channel', {})
        model = TimeVAE1D_HybridMT_Physics4Channel(
            seq_len=seq_len,
            C_in=vae_cfg.get('C_in', 4),
            latent_dim=vae_cfg.get('latent_dim', 64),
            d_model=vae_cfg.get('d_model', 128),
            n_mamba=vae_cfg.get('n_mamba', 2),
            n_transformer=vae_cfg.get('n_transformer', 2),
            nhead=vae_cfg.get('nhead', 8),
            d_state=vae_cfg.get('d_state', 16),
            d_conv=vae_cfg.get('d_conv', 4),
            expand=vae_cfg.get('expand', 2),
            dropout=vae_cfg.get('dropout', 0.2),
            decoder_base=vae_cfg.get('decoder_base', 64),
            total_time=total_time,
            delta_t=delta_t
        ).to(device)
    elif model_type == 'timevae1d_stageaware':
        vae_cfg = model_config.get('timevae1d_stageaware', {})
        model = TimeVAE1D_StageAware(
            seq_len=seq_len,
            C_in=vae_cfg.get('C_in', 1),
            latent_dim=vae_cfg.get('latent_dim', 64),
            d_model=vae_cfg.get('d_model', 128),
            n_mamba=vae_cfg.get('n_mamba', 2),
            n_transformer=vae_cfg.get('n_transformer', 2),
            nhead=vae_cfg.get('nhead', 8),
            d_state=vae_cfg.get('d_state', 16),
            d_conv=vae_cfg.get('d_conv', 4),
            expand=vae_cfg.get('expand', 2),
            dropout=vae_cfg.get('dropout', 0.2),
            decoder_base=vae_cfg.get('decoder_base', 64),
            total_time=total_time,
            delta_t=delta_t
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='从checkpoint加载模型并执行预测保存')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default=None, help='模型checkpoint路径')
    parser.add_argument('--task_name', type=str, default=None, help='任务名称（用于保存结果）')
    args = parser.parse_args()
    
    # 优先使用代码中设置的参数，然后是命令行参数
    config_path = CONFIG_PATH if CONFIG_PATH is not None else args.config
    checkpoint_path = CHECKPOINT_PATH if CHECKPOINT_PATH is not None else args.checkpoint
    task_name = TASK_NAME if TASK_NAME is not None else args.task_name
    
    # 检查必需参数（处理字符串"None"的情况）
    if config_path is None or (isinstance(config_path, str) and config_path.strip().lower() == "none"):
        raise ValueError("请设置 CONFIG_PATH 或使用 --config 参数指定配置文件路径")
    if checkpoint_path is None or (isinstance(checkpoint_path, str) and checkpoint_path.strip().lower() == "none"):
        raise ValueError("请设置 CHECKPOINT_PATH 或使用 --checkpoint 参数指定checkpoint路径")
    
    # 清理路径（去除可能的空格）
    config_path = config_path.strip() if isinstance(config_path, str) else config_path
    checkpoint_path = checkpoint_path.strip() if isinstance(checkpoint_path, str) else checkpoint_path
    
    # 加载配置
    config = load_config(config_path)
    print("="*60)
    print("Config loaded")
    print("="*60)
    
    # 设置随机种子
    set_seed(config['training']['seed'])
    
    # 设置设备
    device_config = config['training']['device']
    if device_config == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_config)
    gpu_ids = _parse_gpu_ids(config)
    if (not gpu_ids) and torch.cuda.is_available():
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if visible:
            num = len([p for p in visible.split(',') if p.strip() != ''])
            if num > 0:
                gpu_ids = list(range(num))
    
    print(f"Device: {device}")
    
    # 获取序列参数
    total_time, delta_t, seq_len = get_sequence_params(config)
    print(f"\n时间序列参数:")
    print(f"  总采样时间: {total_time}s")
    print(f"  采样间隔: {delta_t}s")
    print(f"  序列长度: {seq_len}")
    
    # 加载数据集
    dataset = ThermalDataset(
        pth_path=config['data']['dataset_path'],
        normalize_temp=config['data']['normalize_temp'],
        use_log_e=config['data']['use_log_e']
    )
    
    # 分割数据集（必须和训练时使用相同的随机种子）
    train_split = float(config['data']['train_split'])
    val_split = float(config['data'].get('val_split', 0.2))
    test_split = float(config['data'].get('test_split', 0.1))
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "train/val/test splits must sum to 1"
    
    total_len = len(dataset)
    train_size = int(total_len * train_split)
    val_size = int(total_len * val_split)
    test_size = total_len - train_size - val_size
    
    split_gen = torch.Generator().manual_seed(config['training']['seed'])
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=split_gen
    )
    
    print(f"\nTrain: {train_size} samples")
    print(f"Val: {val_size} samples")
    print(f"Test: {test_size} samples")
    
    # 创建模型
    model = load_model_from_config(config, device, total_time, delta_t, seq_len)
    
    # 使用多GPU
    if len(gpu_ids) > 1:
        main_device_id = gpu_ids[0]
        try:
            torch.cuda.set_device(main_device_id)
        except Exception:
            pass
        device = torch.device(f'cuda:{main_device_id}')
        model = model.to(device)
        model = nn.DataParallel(model, device_ids=gpu_ids, output_device=main_device_id)
        print(f"Using DataParallel on GPUs: {gpu_ids}")
    
    # 检查是否为VAE模型
    model_type = config['model'].get('type', 'transformer').lower()
    is_vae = (model_type == 'timevae1d_mamba' or 
              model_type == 'timevae1d_mamba_physics_decoder' or 
              model_type == 'timevae1d_transformer' or
              model_type == 'timevae1d_hybrid_mamba_transformer' or
              model_type == 'timevae1d_hybridmt_physics4channel' or
              model_type == 'timevae1d_stageaware')
    
    # 加载checkpoint
    print(f"\n加载checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 加载模型权重
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    print("模型权重加载完成")
    
    # 恢复数据集归一化参数（如果checkpoint中有）
    if 'normalize_temp' in checkpoint:
        dataset.normalize_temp = checkpoint['normalize_temp']
    if 'use_log_e' in checkpoint:
        dataset.use_log_e = checkpoint['use_log_e']
    if 'temp_mean' in checkpoint:
        dataset.temp_mean = checkpoint['temp_mean']
    if 'temp_std' in checkpoint:
        dataset.temp_std = checkpoint['temp_std']
    if 'e_mean' in checkpoint:
        dataset.e_mean = checkpoint['e_mean']
    if 'e_std' in checkpoint:
        dataset.e_std = checkpoint['e_std']
    
    # 获取任务名称（如果还没设置）
    if task_name is None:
        task_name = config.get('task_name', 'default_task')
    
    # 创建结果目录
    test_results_dir = Path('results/test_result') / task_name
    test_results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print(f"\n结果将保存到: {test_results_dir.absolute()}")
    
    # 生成并保存完整测试预测
    print("\n" + "="*60)
    print("生成并保存测试集预测结果...")
    print("="*60)
    
    all_pred_rows = []
    test_indices = getattr(test_dataset, 'indices', None)
    if test_indices is None:
        start_idx = len(train_dataset)
        test_indices = list(range(start_idx, start_idx + len(test_dataset)))
    
    # 获取参数维度和名称
    param_dim = int(dataset.data['parameters'].shape[1])
    param_names = dataset.data.get('parameter_names', config.get('data', {}).get('parameter_names', []))
    if not param_names or len(param_names) != param_dim:
        param_names = [f'param_{j}' for j in range(param_dim)]
    
    # 创建样本详细信息目录（仅VAE模型）
    if is_vae:
        samples_dir = test_results_dir / 'samples'
        samples_dir.mkdir(parents=True, exist_ok=True)
        print(f"VAE模型：将保存每个样本的详细信息到 {samples_dir.name}/")
    
    # 推理
    model.eval()
    with torch.no_grad():
        for orig_idx in test_indices:
            temp_seq = dataset.temperature[orig_idx].to(device)
            e_true_norm = dataset.e_transformed[orig_idx]
            
            if is_vae:
                # VAE模型预测
                temp_3d = temp_seq.unsqueeze(0).unsqueeze(0)  # [1, 1, T]
                recon_norm, e_pred_norm, _, _ = model(temp_3d)  # 修复后的正确解包
                e_pred_norm = e_pred_norm.squeeze(0)  # [1]
                recon_norm = recon_norm.squeeze(0).squeeze(0).cpu()  # [T]
            else:
                # 普通模型预测
                e_pred_norm = model(temp_seq.unsqueeze(0)).squeeze(0)  # [1]
                recon_norm = None
            
            e_pred = dataset.denormalize_e(e_pred_norm.squeeze(0)).item()
            e_true = dataset.denormalize_e(e_true_norm).item()
            abs_err = abs(e_pred - e_true)
            params_vec = dataset.data['parameters'][orig_idx]
            
            row = {
                'index': int(orig_idx),
                'e_true': f"{e_true:.6f}",
                'e_pred': f"{e_pred:.6f}",
                'abs_error': f"{abs_err:.6f}"
            }
            for j in range(param_dim):
                safe_name = str(param_names[j]) if param_names[j] else f'param_{j}'
                row[safe_name] = f"{params_vec[j].item():.6f}"
            all_pred_rows.append(row)
            
            # VAE模型：保存每个样本的详细信息
            if is_vae and recon_norm is not None:
                # 获取原始温度序列（从data中获取，未归一化的）
                temp_orig_raw = dataset.data['temperature'][orig_idx]  # [T]
                
                # 反归一化重建序列（确保所有张量都在CPU上）
                if dataset.normalize_temp:
                    # 确保temp_std和temp_mean也在CPU上
                    temp_std = dataset.temp_std.cpu() if hasattr(dataset.temp_std, 'cpu') else dataset.temp_std
                    temp_mean = dataset.temp_mean.cpu() if hasattr(dataset.temp_mean, 'cpu') else dataset.temp_mean
                    recon_orig = recon_norm * temp_std + temp_mean
                else:
                    recon_orig = recon_norm
                
                # 确保长度一致
                actual_len = len(temp_orig_raw)
                if len(recon_orig) != actual_len:
                    # 如果长度不匹配，截断或填充
                    if len(recon_orig) > actual_len:
                        recon_orig = recon_orig[:actual_len]
                    else:
                        # 填充到相同长度（确保在CPU上）
                        padding = torch.zeros(actual_len - len(recon_orig), device=recon_orig.device)
                        recon_orig = torch.cat([recon_orig, padding])
                
                # 生成时间序列（使用实际长度）
                time_points = torch.arange(0, actual_len, dtype=torch.float32) * delta_t
                
                # 构建文件名：顺序为 index, e_true, e_pred, 温度参数, 其他参数
                filename_parts = [f"index{orig_idx}"]
                filename_parts.append(f"e_true_{e_true:.6f}")
                filename_parts.append(f"e_pred_{e_pred:.6f}")
                
                # 分类参数：温度参数和其他参数
                temp_params = []
                other_params = []
                
                for j in range(param_dim):
                    param_name = param_names[j] if param_names[j] else f'param_{j}'
                    param_val = params_vec[j].item()
                    
                    # 跳过e参数（因为已经在文件名中包含了e_true和e_pred）
                    if param_name == 'e':
                        continue
                    
                    # 根据参数类型选择精度和分类
                    if param_name in ['T0', 'T1']:
                        temp_params.append(f"{param_name}_{param_val:.2f}")
                    else:
                        # 其他参数使用6位小数
                        other_params.append(f"{param_name}_{param_val:.6f}")
                
                # 按顺序添加：温度参数在前，其他参数在后
                filename_parts.extend(temp_params)
                filename_parts.extend(other_params)
                
                # 组合文件名并替换小数点
                filename_base = '_'.join(filename_parts).replace('.', '_')
                
                # 保存txt文件（时间、原始、重建）
                txt_path = samples_dir / f"{filename_base}.txt"
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write("# Time(s)\tOriginal_Temp(K)\tReconstructed_Temp(K)\n")
                    for t, orig, recon in zip(time_points, temp_orig_raw, recon_orig):
                        f.write(f"{t:.6f}\t{orig:.6f}\t{recon:.6f}\n")
                
                # 保存yaml文件（参数标签）
                yaml_path = samples_dir / f"{filename_base}.yaml"
                yaml_data = {}
                for j in range(param_dim):
                    param_name = param_names[j] if param_names[j] else f'param_{j}'
                    yaml_data[param_name] = float(params_vec[j].item())
                yaml_data['e_true'] = float(e_true)
                yaml_data['e_pred'] = float(e_pred)
                yaml_data['abs_error'] = float(abs_err)
                
                with open(yaml_path, 'w', encoding='utf-8') as f:
                    yaml.safe_dump(yaml_data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    
    # 计算统计指标
    abs_errors = [abs(float(row['e_true']) - float(row['e_pred'])) for row in all_pred_rows]
    mae = np.mean(abs_errors)
    max_err = np.max(abs_errors)
    min_err = np.min(abs_errors)
    
    # 保存JSON指标文件
    metrics_path = test_results_dir / f'metrics_{timestamp}.json'
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump({
            'mae': float(mae),
            'max_err': float(max_err),
            'min_err': float(min_err),
            'num_samples': len(all_pred_rows),
            'model_type': model_type,
            'checkpoint_path': str(checkpoint_path)
        }, f, ensure_ascii=False, indent=2)
    print(f"\n统计指标已保存: {metrics_path.name}")
    print(f"  MAE: {mae:.6f}")
    print(f"  Max Error: {max_err:.6f}")
    print(f"  Min Error: {min_err:.6f}")
    
    # 写入CSV
    full_pred_path = test_results_dir / f'all_predictions_{timestamp}.csv'
    fieldnames = ['index', 'e_true', 'e_pred', 'abs_error'] + param_names
    with open(full_pred_path, 'w', newline='', encoding='utf-8') as f:
        csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
        csv_writer.writeheader()
        for row in all_pred_rows:
            csv_writer.writerow(row)
    print(f"\n完整预测结果已保存: {full_pred_path.name}")
    
    if is_vae:
        print(f"VAE样本详细信息已保存到: {samples_dir.name}/ ({len(test_indices)} 个样本)")
    
    print("\n" + "="*60)
    print("预测保存完成！")
    print("="*60)


if __name__ == '__main__':
    main()


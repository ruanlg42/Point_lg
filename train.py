import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.tensorboard import SummaryWriter
import yaml
from pathlib import Path
import random
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import json
import csv
import time
from datetime import datetime
from model import (TimeTransformer, CNN1D, PhysicsInformedCNN, PhysicsInformedTransformer, 
                   EnhancedPhysicsTransformer, MambaPhysicsModel, HybridMambaTransformer, PhysTCN,
                   EnhancedMambaPhysicsModel, TimeVAE1D_Mamba, TimeVAE1D_Mamba_PhysicsDecoder, 
                   TimeVAE1D_Transformer, TimeVAE1D_HybridMambaTransformer, TimeVAE1D_HybridMT_Physics4Channel,
                   TimeVAE1D_StageAware, TimeVAE1D_HybridMambaTransformer_Residual, TimeVAE1D_SSTEncoder_Residual)
from dataset import ThermalDataset
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示警告和错误

 
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

def _get_model_state_dict(model):
    """Return the underlying model's state_dict if wrapped by DataParallel."""
    if isinstance(model, nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()

def weighted_mse_loss(pred_norm, true_norm, dataset, device, 
                      threshold_1=10000.0, threshold_2=30000.0,
                      low_weight=2.0, mid_weight=0.5, high_weight=0.1):
    """
    分段加权MSE损失函数
    对不同范围的样本给予不同权重，防止模型预测异常高的e值
    
    Args:
        pred_norm: 归一化后的预测值 [batch, 1]
        true_norm: 归一化后的真实值 [batch, 1]
        dataset: ThermalDataset实例，用于反归一化
        device: 设备
        threshold_1: 第一阈值（原始尺度），默认10000
        threshold_2: 第二阈值（原始尺度），默认30000 (超过此值几乎不可能)
        low_weight: e < threshold_1 的样本权重，默认2.0
        mid_weight: threshold_1 <= e < threshold_2 的样本权重，默认0.5
        high_weight: e >= threshold_2 的样本权重（强惩罚），默认0.1
    """
    # 反归一化到原始尺度（用于判断权重和惩罚）
    true_orig = dataset.denormalize_e(true_norm.squeeze(1))
    pred_orig = dataset.denormalize_e(pred_norm.squeeze(1))
    
    # 计算每个样本的权重（基于原始尺度的真实值）
    low_weight_tensor = torch.tensor(low_weight, device=device, dtype=pred_norm.dtype)
    mid_weight_tensor = torch.tensor(mid_weight, device=device, dtype=pred_norm.dtype)
    high_weight_tensor = torch.tensor(high_weight, device=device, dtype=pred_norm.dtype)
    
    # 分段权重
    weights = torch.where(
        true_orig < threshold_1, 
        low_weight_tensor,
        torch.where(
            true_orig < threshold_2,
            mid_weight_tensor,
            high_weight_tensor
        )
    )
    
    # 在归一化尺度上计算MSE损失
    squared_errors = (pred_norm.squeeze(1) - true_norm.squeeze(1)) ** 2
    weighted_loss = (weights * squared_errors).mean()
    
    # 额外惩罚：如果预测值超过30000，增加额外损失
    over_threshold_mask = pred_orig > threshold_2
    if over_threshold_mask.any():
        # 对超过30000的预测值，按超出程度进行惩罚
        over_amount = torch.clamp(pred_orig - threshold_2, min=0.0)
        # 归一化惩罚项（避免数值过大）
        penalty = (over_amount / threshold_2).pow(2).mean() * 0.5
        weighted_loss = weighted_loss + penalty
    
    return weighted_loss


def vae_loss(recon, x_true, e_pred, e_true, mu, logvar, dataset, device, 
             beta=0.001, recon_weight=1.0, e_weight=1.0,
             threshold_1=10000.0, threshold_2=30000.0,
             low_weight=2.0, mid_weight=0.5, high_weight=0.1):
    """
    VAE损失函数：重建损失 + KL散度 + 参数预测损失
    
    Args:
        recon: [batch, 1, seq_len] 重建的序列
        x_true: [batch, seq_len] 真实序列
        e_pred: [batch, 1] 预测的e参数
        e_true: [batch, 1] 真实的e参数
        mu: [batch, latent_dim] 隐变量均值
        logvar: [batch, latent_dim] 隐变量对数方差
        dataset: ThermalDataset实例
        device: 设备
        beta: KL散度权重（推荐0.001-0.01）
        recon_weight: 重建损失权重
        e_weight: 参数预测损失权重
        threshold_1: 第一阈值（原始尺度），默认10000
        threshold_2: 第二阈值（原始尺度），默认30000
        low_weight: e < threshold_1 的样本权重，默认2.0
        mid_weight: threshold_1 <= e < threshold_2 的样本权重，默认0.5
        high_weight: e >= threshold_2 的样本权重，默认0.1
    
    Returns:
        total_loss: 总损失
        loss_dict: 各项损失的字典（用于日志）
    """
    # 重建损失（MSE）
    recon_loss = nn.functional.mse_loss(recon.squeeze(1), x_true)
    
    # KL散度损失
    # KL(N(mu, sigma) || N(0, 1)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    
    # 参数预测损失（使用加权MSE）
    e_loss = weighted_mse_loss(e_pred, e_true, dataset, device,
                               threshold_1=threshold_1, threshold_2=threshold_2,
                               low_weight=low_weight, mid_weight=mid_weight, high_weight=high_weight)
    
    # 总损失
    total_loss = recon_weight * recon_loss + beta * kl_loss + e_weight * e_loss
    
    loss_dict = {
        'total': total_loss.item(),
        'recon': recon_loss.item(),
        'kl': kl_loss.item(),
        'e_pred': e_loss.item()
    }
    
    return total_loss, loss_dict


def vae_loss_residual(recon, x_true, e_pred, e_true, mu, logvar, x_delta, dataset, device, 
                      beta=0.001, recon_weight=1.0, e_weight=1.0, lambda_ic=0.5,
                      threshold_1=10000.0, threshold_2=30000.0,
                      low_weight=2.0, mid_weight=0.5, high_weight=0.1):
    """
    VAE损失函数（残差重建版本）：重建损失 + KL散度 + 参数预测损失 + 初值约束
    
    Args:
        recon: [batch, 1, seq_len] 重建的序列
        x_true: [batch, seq_len] 真实序列
        e_pred: [batch, 1] 预测的e参数
        e_true: [batch, 1] 真实的e参数
        mu: [batch, latent_dim] 隐变量均值
        logvar: [batch, latent_dim] 隐变量对数方差
        x_delta: [batch, 1, seq_len] 预测的残差（用于初值约束）
        dataset: ThermalDataset实例
        device: 设备
        beta: KL散度权重（推荐0.001-0.01）
        recon_weight: 重建损失权重
        e_weight: 参数预测损失权重
        lambda_ic: 初值约束权重（推荐0.1-1.0）
        threshold_1: 第一阈值（原始尺度），默认10000
        threshold_2: 第二阈值（原始尺度），默认30000
        low_weight: e < threshold_1 的样本权重，默认2.0
        mid_weight: threshold_1 <= e < threshold_2 的样本权重，默认0.5
        high_weight: e >= threshold_2 的样本权重，默认0.1
    
    Returns:
        total_loss: 总损失
        loss_dict: 各项损失的字典（用于日志）
    """
    # 重建损失（MSE）
    recon_loss = nn.functional.mse_loss(recon.squeeze(1), x_true)
    
    # KL散度损失
    # KL(N(mu, sigma) || N(0, 1)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    
    # 参数预测损失（使用加权MSE）
    e_loss = weighted_mse_loss(e_pred, e_true, dataset, device,
                               threshold_1=threshold_1, threshold_2=threshold_2,
                               low_weight=low_weight, mid_weight=mid_weight, high_weight=high_weight)
    
    # 初值约束损失：鼓励残差在初始时刻接近0
    # 使用 L1 损失，更稳定
    ic_loss = torch.abs(x_delta[:, :, 0]).mean()
    
    # 总损失
    total_loss = recon_weight * recon_loss + beta * kl_loss + e_weight * e_loss + lambda_ic * ic_loss
    
    loss_dict = {
        'total': total_loss.item(),
        'recon': recon_loss.item(),
        'kl': kl_loss.item(),
        'e_pred': e_loss.item(),
        'ic': ic_loss.item()  # 记录初值约束损失
    }
    
    return total_loss, loss_dict


def train_epoch(model, dataloader, criterion, optimizer, device, dataset=None, is_vae=False, is_residual=False,
                 lambda_ic=0.5, threshold_1=10000.0, threshold_2=30000.0,
                 low_weight=2.0, mid_weight=0.5, high_weight=0.1):
    """Train one epoch"""
    model.train()
    total_loss = 0
    # 为残差模型添加ic损失组件
    loss_components = {'recon': 0, 'kl': 0, 'e_pred': 0, 'ic': 0} if (is_vae and is_residual) else ({'recon': 0, 'kl': 0, 'e_pred': 0} if is_vae else {})
    
    for batch in dataloader:
        temp = batch['temperature'].to(device)  # [batch, T]
        e_true = batch['e'].to(device).unsqueeze(1)  # [batch, 1]
        
        # Forward
        if is_vae:
            # VAE模型需要特殊处理
            temp_3d = temp.unsqueeze(1)  # [batch, 1, T]
            model_output = model(temp_3d)
            if len(model_output) == 5:
                # 可能是 Stage-Aware VAE 或 Residual VAE
                if is_residual:
                    # Residual VAE: (recon, e_pred, (mu, logvar), temp_smooth, x_delta)
                    recon, e_pred, (mu, logvar), temp_smooth, x_delta = model_output
                    # 使用残差损失函数
                    loss, loss_dict = vae_loss_residual(recon, temp_smooth.squeeze(1), e_pred, e_true, 
                                                       mu, logvar, x_delta, dataset, device,
                                                       lambda_ic=lambda_ic,
                                                       threshold_1=threshold_1, threshold_2=threshold_2,
                                                       low_weight=low_weight, mid_weight=mid_weight, high_weight=high_weight)
                else:
                    # Stage-Aware VAE: (recon, e_pred, (mu, logvar), temp_smooth, stage_weights)
                    recon, e_pred, (mu, logvar), temp_smooth, _ = model_output
                    # 使用标准VAE损失
                    loss, loss_dict = vae_loss(recon, temp_smooth.squeeze(1), e_pred, e_true, mu, logvar, dataset, device,
                                             threshold_1=threshold_1, threshold_2=threshold_2,
                                             low_weight=low_weight, mid_weight=mid_weight, high_weight=high_weight)
            else:
                # 其他VAE: (recon, e_pred, (mu, logvar), temp_smooth)
                recon, e_pred, (mu, logvar), temp_smooth = model_output
                # 使用标准VAE损失
                loss, loss_dict = vae_loss(recon, temp_smooth.squeeze(1), e_pred, e_true, mu, logvar, dataset, device,
                                         threshold_1=threshold_1, threshold_2=threshold_2,
                                         low_weight=low_weight, mid_weight=mid_weight, high_weight=high_weight)
            
            # 累积各项损失（用于日志）
            for key in loss_components:
                if key in loss_dict:
                    loss_components[key] += loss_dict[key]
        else:
            # 普通模型
            e_pred = model(temp)
            
            # 使用加权损失函数
            if dataset is not None:
                loss = weighted_mse_loss(e_pred, e_true, dataset, device,
                                       threshold_1=threshold_1, threshold_2=threshold_2,
                                       low_weight=low_weight, mid_weight=mid_weight, high_weight=high_weight)
            else:
                # 如果dataset未提供，使用普通损失（向后兼容）
                loss = criterion(e_pred, e_true)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    
    if is_vae:
        # 返回平均的各项损失
        for key in loss_components:
            loss_components[key] /= len(dataloader)
        return avg_loss, loss_components
    
    return avg_loss


def compute_loss_on_loader(model, dataloader, criterion, device, dataset=None, is_vae=False, is_residual=False,
                           lambda_ic=0.5, threshold_1=10000.0, threshold_2=30000.0,
                           low_weight=2.0, mid_weight=0.5, high_weight=0.1):
    """Compute average loss on dataloader"""
    model.eval()
    total_loss = 0
    # 为残差模型添加ic损失组件
    loss_components = {'recon': 0, 'kl': 0, 'e_pred': 0, 'ic': 0} if (is_vae and is_residual) else ({'recon': 0, 'kl': 0, 'e_pred': 0} if is_vae else {})
    
    with torch.no_grad():
        for batch in dataloader:
            temp = batch['temperature'].to(device)
            e_true = batch['e'].to(device).unsqueeze(1)
            
            if is_vae:
                # VAE模型
                temp_3d = temp.unsqueeze(1)  # [batch, 1, T]
                model_output = model(temp_3d)
                if len(model_output) == 5:
                    # 可能是 Stage-Aware VAE 或 Residual VAE
                    if is_residual:
                        # Residual VAE: (recon, e_pred, (mu, logvar), temp_smooth, x_delta)
                        recon, e_pred, (mu, logvar), temp_smooth, x_delta = model_output
                        # 使用残差损失函数
                        loss, loss_dict = vae_loss_residual(recon, temp_smooth.squeeze(1), e_pred, e_true, 
                                                           mu, logvar, x_delta, dataset, device,
                                                           lambda_ic=lambda_ic,
                                                           threshold_1=threshold_1, threshold_2=threshold_2,
                                                           low_weight=low_weight, mid_weight=mid_weight, high_weight=high_weight)
                    else:
                        # Stage-Aware VAE: (recon, e_pred, (mu, logvar), temp_smooth, stage_weights)
                        recon, e_pred, (mu, logvar), temp_smooth, _ = model_output
                        # 使用标准VAE损失
                        loss, loss_dict = vae_loss(recon, temp_smooth.squeeze(1), e_pred, e_true, mu, logvar, dataset, device,
                                                 threshold_1=threshold_1, threshold_2=threshold_2,
                                                 low_weight=low_weight, mid_weight=mid_weight, high_weight=high_weight)
                else:
                    # 其他VAE: (recon, e_pred, (mu, logvar), temp_smooth)
                    recon, e_pred, (mu, logvar), temp_smooth = model_output
                    # 使用标准VAE损失
                    loss, loss_dict = vae_loss(recon, temp_smooth.squeeze(1), e_pred, e_true, mu, logvar, dataset, device,
                                             threshold_1=threshold_1, threshold_2=threshold_2,
                                             low_weight=low_weight, mid_weight=mid_weight, high_weight=high_weight)
                
                # 累积各项损失
                for key in loss_components:
                    if key in loss_dict:
                        loss_components[key] += loss_dict[key]
            else:
                # 普通模型
                e_pred = model(temp)
                
                # 使用加权损失函数
                if dataset is not None:
                    loss = weighted_mse_loss(e_pred, e_true, dataset, device,
                                           threshold_1=threshold_1, threshold_2=threshold_2,
                                           low_weight=low_weight, mid_weight=mid_weight, high_weight=high_weight)
                else:
                    # 如果dataset未提供，使用普通损失（向后兼容）
                    loss = criterion(e_pred, e_true)
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    
    if is_vae:
        # 返回平均的各项损失
        for key in loss_components:
            loss_components[key] /= len(dataloader)
        return avg_loss, loss_components
    
    return avg_loss


def load_config(config_path='config.yaml'):
    """Load config file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def generate_task_name(config):
    """
    自动生成task name，格式：前缀_时间戳_模型类型_参数_suffix
    
    Args:
        config: 配置字典
    
    Returns:
        str: 生成的task name
    """
    task_config = config.get('task', {})
    model_config = config.get('model', {})
    model_type = model_config.get('type', 'transformer').lower()
    
    # 前缀（默认"AAA"）
    prefix = task_config.get('prefix', 'AAA')
    
    # 时间戳（包含日期和时间）
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 根据模型类型提取参数并生成名称部分
    name_parts = []
    
    if model_type == 'transformer':
        trans_cfg = model_config.get('transformer', {})
        name_parts.append('transformer')
        name_parts.append(f"d{trans_cfg.get('d_model', 256)}")
        name_parts.append(f"nhead{trans_cfg.get('nhead', 8)}")
        name_parts.append(f"layer{trans_cfg.get('num_layers', 4)}")
        name_parts.append(f"feed{trans_cfg.get('dim_feedforward', 1024)}")
        
    elif model_type == 'cnn1d':
        cnn_cfg = model_config.get('cnn1d', {})
        name_parts.append('cnn1d')
        name_parts.append(f"ch{cnn_cfg.get('hidden_channels', 128)}")
        name_parts.append(f"layer{cnn_cfg.get('num_layers', 3)}")
        name_parts.append(f"k{cnn_cfg.get('kernel_size', 5)}")
        
    elif model_type == 'physics_cnn':
        phys_cfg = model_config.get('physics_cnn', model_config.get('cnn1d', {}))
        name_parts.append('physics_cnn')
        name_parts.append(f"ch{phys_cfg.get('hidden_channels', 64)}")
        name_parts.append(f"layer{phys_cfg.get('num_layers', 3)}")
        name_parts.append(f"k{phys_cfg.get('kernel_size', 5)}")
        
    elif model_type == 'physics_transformer':
        pt_cfg = model_config.get('physics_transformer', {})
        base = model_config.get('transformer', {})
        name_parts.append('physics_transformer')
        name_parts.append(f"d{pt_cfg.get('d_model', base.get('d_model', 256))}")
        name_parts.append(f"nhead{pt_cfg.get('nhead', base.get('nhead', 8))}")
        name_parts.append(f"layer{pt_cfg.get('num_layers', base.get('num_layers', 4))}")
        name_parts.append(f"feed{pt_cfg.get('dim_feedforward', base.get('dim_feedforward', 1024))}")
        
    elif model_type == 'enhanced_physics_transformer':
        ept_cfg = model_config.get('enhanced_physics_transformer', {})
        base = model_config.get('transformer', {})
        name_parts.append('enhanced_physics_transformer')
        name_parts.append(f"d{ept_cfg.get('d_model', base.get('d_model', 256))}")
        name_parts.append(f"nhead{ept_cfg.get('nhead', base.get('nhead', 8))}")
        name_parts.append(f"layer{ept_cfg.get('num_layers', base.get('num_layers', 4))}")
        name_parts.append(f"feed{ept_cfg.get('dim_feedforward', base.get('dim_feedforward', 1024))}")
        
    elif model_type == 'mamba_physics':
        mamba_cfg = model_config.get('mamba_physics', {})
        name_parts.append('mamba_physics')
        name_parts.append(f"d{mamba_cfg.get('d_model', 256)}")
        name_parts.append(f"layer{mamba_cfg.get('n_layers', 4)}")
        name_parts.append(f"state{mamba_cfg.get('d_state', 16)}")
        
    elif model_type == 'hybrid_mamba_transformer':
        hybrid_cfg = model_config.get('hybrid_mamba_transformer', {})
        name_parts.append('hybrid_mamba_transformer')
        name_parts.append(f"d{hybrid_cfg.get('d_model', 256)}")
        name_parts.append(f"mamba{hybrid_cfg.get('n_mamba', 2)}")
        name_parts.append(f"trans{hybrid_cfg.get('n_transformer', 2)}")
        
    elif model_type == 'phys_tcn':
        tcn_cfg = model_config.get('phys_tcn', {})
        name_parts.append('phys_tcn')
        name_parts.append(f"ch{tcn_cfg.get('channels', 128)}")
        name_parts.append(f"layer{tcn_cfg.get('num_layers', 6)}")
        name_parts.append(f"k{tcn_cfg.get('kernel_size', 7)}")
        
    elif model_type == 'enhanced_mamba_physics':
        enhanced_cfg = model_config.get('enhanced_mamba_physics', {})
        name_parts.append('enhanced_mamba')
        name_parts.append(f"d{enhanced_cfg.get('d_model', 256)}")
        name_parts.append(f"layer{enhanced_cfg.get('n_layers', 6)}")
        name_parts.append(f"state{enhanced_cfg.get('d_state', 16)}")
        
    elif model_type == 'timevae1d_mamba':
        vae_cfg = model_config.get('timevae1d_mamba', {})
        name_parts.append('timevae_mamba')
        name_parts.append(f"d{vae_cfg.get('d_model', 128)}")
        name_parts.append(f"latent{vae_cfg.get('latent_dim', 64)}")
        name_parts.append(f"layer{vae_cfg.get('depth', 4)}")
        
    elif model_type == 'timevae1d_mamba_physics_decoder':
        vae_cfg = model_config.get('timevae1d_mamba_physics_decoder', {})
        name_parts.append('timevae_physics_decoder')
        name_parts.append(f"d{vae_cfg.get('d_model', 128)}")
        name_parts.append(f"latent{vae_cfg.get('latent_dim', 64)}")
        name_parts.append(f"layer{vae_cfg.get('depth', 4)}")
        name_parts.append(f"basis{vae_cfg.get('num_physics_basis', 8)}")
    
    elif model_type == 'timevae1d_transformer':
        vae_cfg = model_config.get('timevae1d_transformer', {})
        name_parts.append('timevae_transformer')
        name_parts.append(f"d{vae_cfg.get('d_model', 128)}")
        name_parts.append(f"latent{vae_cfg.get('latent_dim', 64)}")
        name_parts.append(f"layer{vae_cfg.get('num_layers', 4)}")
        name_parts.append(f"head{vae_cfg.get('nhead', 8)}")
    elif model_type == 'timevae1d_hybrid_mamba_transformer':
        vae_cfg = model_config.get('timevae1d_hybrid_mamba_transformer', {})
        name_parts.append('timevae_hybrid')
        name_parts.append(f"d{vae_cfg.get('d_model', 128)}")
        name_parts.append(f"latent{vae_cfg.get('latent_dim', 64)}")
        name_parts.append(f"mamba{vae_cfg.get('n_mamba', 2)}")
        name_parts.append(f"trans{vae_cfg.get('n_transformer', 2)}")
        name_parts.append(f"head{vae_cfg.get('nhead', 8)}")
    elif model_type == 'timevae1d_hybridmt_physics4channel':
        vae_cfg = model_config.get('timevae1d_hybridmt_physics4channel', {})
        name_parts.append('timevae_hybridmt_4ch')
        name_parts.append(f"d{vae_cfg.get('d_model', 128)}")
        name_parts.append(f"latent{vae_cfg.get('latent_dim', 64)}")
        name_parts.append(f"mamba{vae_cfg.get('n_mamba', 2)}")
        name_parts.append(f"trans{vae_cfg.get('n_transformer', 2)}")
        name_parts.append(f"head{vae_cfg.get('nhead', 8)}")
    elif model_type == 'timevae1d_stageaware':
        vae_cfg = model_config.get('timevae1d_stageaware', {})
        name_parts.append('timevae_stage')
        name_parts.append(f"d{vae_cfg.get('d_model', 128)}")
        name_parts.append(f"latent{vae_cfg.get('latent_dim', 64)}")
        name_parts.append(f"mamba{vae_cfg.get('n_mamba', 2)}")
        name_parts.append(f"trans{vae_cfg.get('n_transformer', 2)}")
        name_parts.append(f"head{vae_cfg.get('nhead', 8)}")
    elif model_type == 'timevae1d_hybrid_mamba_transformer_residual':
        vae_cfg = model_config.get('timevae1d_hybrid_mamba_transformer_residual', {})
        name_parts.append('timevae_hybrid_res')
        name_parts.append(f"d{vae_cfg.get('d_model', 128)}")
        name_parts.append(f"latent{vae_cfg.get('latent_dim', 64)}")
        name_parts.append(f"mamba{vae_cfg.get('n_mamba', 2)}")
        name_parts.append(f"trans{vae_cfg.get('n_transformer', 2)}")
        name_parts.append(f"head{vae_cfg.get('nhead', 8)}")
        name_parts.append(f"ic{vae_cfg.get('lambda_ic', 0.5)}")
        
    else:
        name_parts.append(model_type)
    
    # 组合前缀_时间戳_参数
    task_name = f"{prefix}_{timestamp}_{'_'.join(name_parts)}"
    
    # 添加后缀（如果存在）
    suffix_raw = task_config.get('suffix', '')
    if suffix_raw is None:
        suffix = ''
    else:
        suffix = str(suffix_raw).strip()
    if suffix:
        task_name = f"{task_name}_{suffix}"
    
    return task_name



def get_sequence_params(config):
    """
    从配置中获取序列参数
    
    Args:
        config: 配置字典
    
    Returns:
        tuple: (total_time, delta_t, seq_len)
    """
    data_config = config.get('data', {})
    
    # 获取基本参数
    total_time = float(data_config.get('total_time', 5.0))
    delta_t = float(data_config.get('delta_t', 0.02))
    
    # seq_len优先使用配置值，如果为None则自动计算
    seq_len_config = data_config.get('seq_len', None)
    if seq_len_config is not None and seq_len_config != 'null':
        seq_len = int(seq_len_config)
    else:
        # 自动计算：从0到total_time，步长为delta_t
        seq_len = int(total_time / delta_t) + 1
    
    return total_time, delta_t, seq_len


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluate_on_loader(model, dataloader, base_dataset: ThermalDataset, device, is_vae=False):
    """Evaluate model and return error statistics in original scale"""
    model.eval()
    abs_errors = []
    sample_outputs = []
    for batch in dataloader:
        temp = batch['temperature'].to(device)
        e_true_norm = batch['e'].to(device).unsqueeze(1)
        
        if is_vae:
            # VAE模型：只需要e预测
            temp_3d = temp.unsqueeze(1)  # [batch, 1, T]
            # 注意：timevae1d_stageaware返回5个值，其他VAE返回4个值
            model_output = model(temp_3d)
            if len(model_output) == 5:
                # Stage-Aware VAE: (recon, e_pred, (mu, logvar), x_smooth, stage_weights)
                _, e_pred_norm, _, _, _ = model_output
            else:
                # 其他VAE: (recon, e_pred, (mu, logvar), x_smooth)
                _, e_pred_norm, _, _ = model_output
        else:
            # 普通模型
            e_pred_norm = model(temp)

        # Denormalize to original scale
        e_true = base_dataset.denormalize_e(e_true_norm.squeeze(1)).cpu()
        e_pred = base_dataset.denormalize_e(e_pred_norm.squeeze(1)).cpu()

        abs_errors.append(torch.abs(e_pred - e_true))

        # Collect first 10 samples for display
        needed = max(0, 10 - len(sample_outputs))
        if needed > 0:
            take = min(needed, e_true.shape[0])
            for i in range(take):
                sample_outputs.append((e_true[i].item(), e_pred[i].item()))

    if len(abs_errors) == 0:
        return {
            'mae': float('nan'),
            'max_err': float('nan'),
            'min_err': float('nan'),
            'samples': sample_outputs
        }

    abs_errors = torch.cat(abs_errors)
    return {
        'mae': abs_errors.mean().item(),
        'max_err': abs_errors.max().item(),
        'min_err': abs_errors.min().item(),
        'samples': sample_outputs
    }


def train(config_path='config.yaml'):
    """Main training function"""
    
    # Load config
    config = load_config(config_path)
    print("="*60)
    print("Config loaded")
    print("="*60)
    
    # Set random seed
    set_seed(config['training']['seed'])
    
    # Set device
    device_config = config['training']['device']
    if device_config == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_config)
    gpu_ids = _parse_gpu_ids(config)
    # If GPU_IDS not provided, default to all visible GPUs
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
    print(f"  序列长度: {seq_len} (计算得到: {int(total_time/delta_t)+1})")
    
    # Load dataset
    dataset = ThermalDataset(
        pth_path=config['data']['dataset_path'],
        normalize_temp=config['data']['normalize_temp'],
        use_log_e=config['data']['use_log_e']
    )
    
    # Split dataset (with generator for reproducibility)
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
    
    batch_size = config['data']['batch_size']
    num_workers = config['data']['num_workers']
    loader_gen = torch.Generator().manual_seed(config['training']['seed'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=loader_gen)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    print(f"\nTrain: {train_size} samples")
    print(f"Val: {val_size} samples")
    print(f"Test: {test_size} samples")
    
    # Create model based on config
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
        print(f"Model: Transformer (d_model={trans_cfg['d_model']}, layers={trans_cfg['num_layers']}, nhead={trans_cfg['nhead']})")
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
        print(f"Model: 1D CNN (channels={cnn_cfg['hidden_channels']}, layers={cnn_cfg['num_layers']}, kernel={cnn_cfg['kernel_size']})")
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
        print(f"Model: Physics-Informed CNN (4-channel, channels={phys_cfg.get('hidden_channels', 64)}, layers={phys_cfg.get('num_layers', 3)})")
        print(f"  Time params: total={total_time}s, delta_t={delta_t}s, seq_len={seq_len}")
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
        print(f"Model: Physics-Informed Transformer (d_model={pt_cfg.get('d_model', base.get('d_model', 256))}, layers={pt_cfg.get('num_layers', base.get('num_layers', 4))}, nhead={pt_cfg.get('nhead', base.get('nhead', 8))})")
        print(f"  Time params: total={total_time}s, delta_t={delta_t}s, seq_len={seq_len}")
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
        print(f"Model: Enhanced Physics Transformer (d_model={ept_cfg.get('d_model', base.get('d_model', 256))}, layers={ept_cfg.get('num_layers', base.get('num_layers', 4))}, nhead={ept_cfg.get('nhead', base.get('nhead', 8))})")
        print(f"  Features: Residual paths + Adaptive temporal attention pooling")
        print(f"  Time params: total={total_time}s, delta_t={delta_t}s, seq_len={seq_len}")
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
        print(f"Model: Mamba Physics (d_model={mamba_cfg.get('d_model', 256)}, layers={mamba_cfg.get('n_layers', 4)})")
        print(f"  SSM params: d_state={mamba_cfg.get('d_state', 16)}, d_conv={mamba_cfg.get('d_conv', 4)}, expand={mamba_cfg.get('expand', 2)}")
        print(f"  Time params: total={total_time}s, delta_t={delta_t}s, seq_len={seq_len}")
        print(f"  Complexity: O(L) vs O(L²) for Transformer - 250x faster!")
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
        print(f"Model: Hybrid Mamba-Transformer (d_model={hybrid_cfg.get('d_model', 256)})")
        print(f"  Architecture: {hybrid_cfg.get('n_mamba', 2)} Mamba + {hybrid_cfg.get('n_transformer', 2)} Transformer layers")
        print(f"  Time params: total={total_time}s, delta_t={delta_t}s, seq_len={seq_len}")
    elif model_type == 'phys_tcn':
        tcn_cfg = model_config.get('phys_tcn', {})
        model = PhysTCN(
            channels=tcn_cfg.get('channels', 128),
            num_layers=tcn_cfg.get('num_layers', 6),
            dilations=tcn_cfg.get('dilations', None),  # 默认[1,2,4,8,16,32]
            kernel_size=tcn_cfg.get('kernel_size', 7),
            dropout=tcn_cfg.get('dropout', 0.1),
            activation=tcn_cfg.get('activation', 'glu'),
            total_time=total_time,
            delta_t=delta_t
        ).to(device)
        print(f"Model: Phys-TCN (channels={tcn_cfg.get('channels', 128)}, layers={tcn_cfg.get('num_layers', 6)})")
        print(f"  Architecture: Depthwise Dilated TCN + Learnable Query Pooling + Heteroscedastic Head")
        print(f"  Time params: total={total_time}s, delta_t={delta_t}s, seq_len={seq_len}")
        print(f"  Complexity: O(L) vs O(L²) for Transformer - Fast & Stable!")
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
        print(f"Model: Enhanced Mamba Physics (d_model={enhanced_cfg.get('d_model', 256)}, layers={enhanced_cfg.get('n_layers', 6)})")
        print(f"  Features: Learnable Query Pooling + Heteroscedastic Head + Multi-Scale Fusion")
        print(f"  SSM params: d_state={enhanced_cfg.get('d_state', 16)}, d_conv={enhanced_cfg.get('d_conv', 4)}, expand={enhanced_cfg.get('expand', 2)}")
        print(f"  Time params: total={total_time}s, delta_t={delta_t}s, seq_len={seq_len}")
        print(f"  Complexity: O(L) - Enhanced Mamba with better feature fusion!")
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
        print(f"Model: TimeVAE1D-Mamba (d_model={vae_cfg.get('d_model', 128)}, latent_dim={vae_cfg.get('latent_dim', 64)})")
        print(f"  Architecture: Mamba Encoder + Deconv Decoder + Parameter Head")
        print(f"  VAE: Variational Autoencoder with reconstruction + parameter prediction")
        print(f"  SSM params: depth={vae_cfg.get('depth', 4)}, d_state={vae_cfg.get('d_state', 16)}, d_conv={vae_cfg.get('d_conv', 4)}")
        print(f"  Sequence length: {seq_len} (total={total_time}s, delta_t={delta_t}s)")
        print(f"  Complexity: O(L) - Efficient VAE for time series!")
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
        print(f"Model: TimeVAE1D-Mamba-PhysicsDecoder (d_model={vae_cfg.get('d_model', 128)}, latent_dim={vae_cfg.get('latent_dim', 64)})")
        print(f"  Architecture: Mamba Encoder + Physics Basis Decoder + Residual Conv1D")
        print(f"  VAE: Variational Autoencoder with physics-informed reconstruction")
        print(f"  Decoder: x^(t) = Σ_k w_k * g_k(t) + Conv1D-Residual(t)")
        print(f"  Physics basis: {vae_cfg.get('num_physics_basis', 8)} functions (t^{-1/2}, exp(-t/τ), etc.)")
        print(f"  SSM params: depth={vae_cfg.get('depth', 4)}, d_state={vae_cfg.get('d_state', 16)}, d_conv={vae_cfg.get('d_conv', 4)}")
        print(f"  Sequence length: {seq_len} (total={total_time}s, delta_t={delta_t}s)")
        print(f"  Complexity: O(L) - Physics-guided VAE for time series!")
    
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
            dropout=vae_cfg.get('dropout', 0.1),
            decoder_base=vae_cfg.get('decoder_base', 64),
            total_time=total_time,
            delta_t=delta_t
        ).to(device)
        print(f"Model: TimeVAE1D-Transformer (d_model={vae_cfg.get('d_model', 128)}, latent_dim={vae_cfg.get('latent_dim', 64)})")
        print(f"  Architecture: Transformer Encoder + Conv1D Decoder + Parameter Head")
        print(f"  VAE: Variational Autoencoder with reconstruction + parameter prediction")
        print(f"  Encoder: Transformer (nhead={vae_cfg.get('nhead', 8)}, num_layers={vae_cfg.get('num_layers', 4)})")
        print(f"  Decoder: Conv1D with adaptive upsampling")
        print(f"  Sequence length: {seq_len} (total={total_time}s, delta_t={delta_t}s)")
        print(f"  Complexity: O(L²) - Full attention for global dependencies!")
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
            dropout=vae_cfg.get('dropout', 0.1),
            decoder_base=vae_cfg.get('decoder_base', 64),
            total_time=total_time,
            delta_t=delta_t
        ).to(device)
        print(f"Model: TimeVAE1D-HybridMambaTransformer (d_model={vae_cfg.get('d_model', 128)}, latent_dim={vae_cfg.get('latent_dim', 64)})")
        print(f"  Architecture: Hybrid (Mamba + Transformer) Encoder + Conv1D Decoder + Parameter Head")
        print(f"  VAE: Variational Autoencoder with reconstruction + parameter prediction")
        print(f"  Encoder: Hybrid (n_mamba={vae_cfg.get('n_mamba', 2)}, n_transformer={vae_cfg.get('n_transformer', 2)}, nhead={vae_cfg.get('nhead', 8)})")
        print(f"  Decoder: Conv1D with adaptive upsampling")
        print(f"  Sequence length: {seq_len} (total={total_time}s, delta_t={delta_t}s)")
        print(f"  Complexity: Hybrid O(L)+O(L²) - Best of both worlds!")
    elif model_type == 'timevae1d_hybridmt_physics4channel':
        vae_cfg = model_config.get('timevae1d_hybridmt_physics4channel', {})
        model = TimeVAE1D_HybridMT_Physics4Channel(
            seq_len=seq_len,
            C_in=vae_cfg.get('C_in', 4),  # 固定为4通道
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
        print(f"Model: TimeVAE1D-HybridMambaTransformer-Physics4Channel (d_model={vae_cfg.get('d_model', 128)}, latent_dim={vae_cfg.get('latent_dim', 64)})")
        print(f"  Architecture: Hybrid Mamba-Transformer Encoder (4-channel physics) + Conv1D Decoder + Parameter Head")
        print(f"  VAE: Variational Autoencoder with 4-channel physics features")
        print(f"  Input: 4 channels [T, Ṫ, t^(-1/2), Δt]")
        print(f"  Reconstruction: Only filtered temperature T (for stable loss)")
        print(f"  Encoder: Hybrid (n_mamba={vae_cfg.get('n_mamba', 2)}, n_transformer={vae_cfg.get('n_transformer', 2)}, nhead={vae_cfg.get('nhead', 8)})")
        print(f"  SSM params: d_state={vae_cfg.get('d_state', 16)}, d_conv={vae_cfg.get('d_conv', 4)}, expand={vae_cfg.get('expand', 2)}")
        print(f"  Sequence length: {seq_len} (total={total_time}s, delta_t={delta_t}s)")
        print(f"  Complexity: Hybrid O(L)+O(L²) - Best of both worlds with physics priors!")
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
        print(f"Model: TimeVAE1D-StageAware-TwoStage-V2 (d_model={vae_cfg.get('d_model', 128)}, latent_dim={vae_cfg.get('latent_dim', 64)})")
        print(f"  Architecture: Two-Stage Shared-Mamba Encoder V2 + Conv1D Decoder + Parameter Head")
        print(f"  VAE: Variational Autoencoder with non-uniform time-series segmentation")
        print(f"  Input: 1 channel (Gaussian filtered temperature)")
        print(f"  Stage 1 (0-1s): Fast heating phase - High information content")
        print(f"  Stage 2 (1-5s): Diffusion + equilibration phase")
        print(f"  Shared Mamba layers: {vae_cfg.get('n_mamba', 2)} (common temporal patterns, with residual+LN)")
        print(f"  Independent Transformer layers per stage: {vae_cfg.get('n_transformer', 2)} (stage-specific modeling)")
        print(f"  ✅ Improvements: Learnable position encoding + Mamba residual connections + Pre-LN normalization")
    elif model_type == 'timevae1d_hybrid_mamba_transformer_residual':
        vae_cfg = model_config.get('timevae1d_hybrid_mamba_transformer_residual', {})
        model = TimeVAE1D_HybridMambaTransformer_Residual(
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
            dropout=vae_cfg.get('dropout', 0.1),
            decoder_base=vae_cfg.get('decoder_base', 64),
            total_time=total_time,
            delta_t=delta_t
        ).to(device)
        print(f"Model: TimeVAE1D-HybridMambaTransformer-Residual (d_model={vae_cfg.get('d_model', 128)}, latent_dim={vae_cfg.get('latent_dim', 64)})")
        print(f"  Architecture: Hybrid (Mamba + Transformer) Encoder + Residual Conv1D Decoder + Parameter Head")
        print(f"  VAE: Variational Autoencoder with residual reconstruction + initial condition constraint")
        print(f"  Encoder: Hybrid (n_mamba={vae_cfg.get('n_mamba', 2)}, n_transformer={vae_cfg.get('n_transformer', 2)}, nhead={vae_cfg.get('nhead', 8)})")
        print(f"  Decoder: Residual reconstruction (x̂ = x_smooth + residual)")
        print(f"  Initial condition constraint: λ_ic = {vae_cfg.get('lambda_ic', 0.5)}")
        print(f"  ✨ Key advantage: Initial temperature naturally preserved, decoder only refines details")
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
        ).to(device)
        print(f"Model: TimeVAE1D-SSTEncoder-Residual (d_model={vae_cfg.get('d_model', 128)}, latent_dim={vae_cfg.get('latent_dim', 64)})")
        print(f"  Architecture: SST Encoder (Long Mamba + Short Transformer + Router) + Residual Conv1D Decoder + Parameter Head")
        print(f"  VAE: Variational Autoencoder with residual reconstruction + initial condition constraint")
        print(f"  Encoder: NewSSTEncoder_Weighted (Long: n_mamba={vae_cfg.get('n_mamba', 2)}, Short: n_transformer={vae_cfg.get('n_transformer', 2)}, nhead={vae_cfg.get('nhead', 8)})")
        print(f"  Router: {'Enabled (Adaptive long/short weighting)' if vae_cfg.get('use_router', True) else 'Disabled (Fixed 0.5/0.5)'}")
        print(f"  Context Window: {vae_cfg.get('context_window', 151)} (Short branch recent history)")
        print(f"  Decoder: Residual reconstruction (x̂ = x0_smooth + Δx, based on initial temperature)")
        print(f"  Initial condition constraint: λ_ic = {vae_cfg.get('lambda_ic', 0.5)}")
        print(f"  ✨ Key advantages: Long/short temporal modeling + Adaptive routing + Initial temperature preservation")
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Choose 'transformer', 'cnn1d', 'physics_cnn', 'physics_transformer', 'enhanced_physics_transformer', 'mamba_physics', 'hybrid_mamba_transformer', 'phys_tcn', 'enhanced_mamba_physics', 'timevae1d_mamba', 'timevae1d_mamba_physics_decoder', 'timevae1d_transformer', 'timevae1d_hybrid_mamba_transformer', 'timevae1d_hybridmt_physics4channel', 'timevae1d_stageaware', 'timevae1d_hybrid_mamba_transformer_residual', or 'timevae1d_sst_encoder_residual'.")
    
    # Wrap DataParallel if multiple GPUs are requested/available
    if device.type == 'cuda' and torch.cuda.is_available() and len(gpu_ids) > 0:
        # Ensure main device is the first id
        main_device_id = int(gpu_ids[0])
        try:
            torch.cuda.set_device(main_device_id)
        except Exception:
            pass
        device = torch.device(f'cuda:{main_device_id}')
        model = model.to(device)
        if len(gpu_ids) > 1:
            model = nn.DataParallel(model, device_ids=gpu_ids, output_device=main_device_id)
        print(f"Using DataParallel on GPUs: {gpu_ids} (main cuda:{main_device_id})")
    
    # 检查是否为VAE模型
    is_vae = (model_type == 'timevae1d_mamba' or 
              model_type == 'timevae1d_mamba_physics_decoder' or 
              model_type == 'timevae1d_transformer' or
              model_type == 'timevae1d_hybrid_mamba_transformer' or
              model_type == 'timevae1d_hybridmt_physics4channel' or
              model_type == 'timevae1d_stageaware' or
              model_type == 'timevae1d_hybrid_mamba_transformer_residual' or
              model_type == 'timevae1d_sst_encoder_residual')
    
    # 检查是否为残差VAE模型
    is_residual = (model_type == 'timevae1d_hybrid_mamba_transformer_residual' or
                   model_type == 'timevae1d_sst_encoder_residual')
    
    # Loss and optimizer
    # 注意：实际使用的是加权损失函数，criterion保留用于向后兼容
    criterion = nn.MSELoss()
    
    # 读取分段权重损失配置
    loss_config = config.get('loss', {})
    threshold_1 = loss_config.get('threshold_1', 10000.0)
    threshold_2 = loss_config.get('threshold_2', 30000.0)
    low_weight = loss_config.get('low_weight', 2.0)
    mid_weight = loss_config.get('mid_weight', 0.5)
    high_weight = loss_config.get('high_weight', 0.1)
    
    # 读取残差模型的初值约束权重
    if is_residual:
        if model_type == 'timevae1d_sst_encoder_residual':
            vae_cfg = model_config.get('timevae1d_sst_encoder_residual', {})
        else:
            vae_cfg = model_config.get('timevae1d_hybrid_mamba_transformer_residual', {})
        lambda_ic = vae_cfg.get('lambda_ic', 0.5)
    else:
        lambda_ic = 0.5  # 默认值（非残差模型不使用）
    
    if is_vae:
        if is_residual:
            print("使用VAE损失函数（残差重建版本）:")
            print("  - 重建损失 (MSE)")
            print("  - KL散度损失 (beta=0.001)")
            print("  - 参数预测损失 (加权MSE)")
            print(f"  - 初值约束损失 (L1, lambda_ic={lambda_ic})")
        else:
            print("使用VAE损失函数:")
            print("  - 重建损失 (MSE)")
            print("  - KL散度损失 (beta=0.001)")
            print("  - 参数预测损失 (加权MSE)")
    else:
        print("使用分段加权MSE损失函数:")
        print(f"  - e < {threshold_1:.0f}: 权重 {low_weight} (高优先级,常见样本)")
        print(f"  - {threshold_1:.0f} ≤ e < {threshold_2:.0f}: 权重 {mid_weight} (中等优先级)")
        print(f"  - e ≥ {threshold_2:.0f}: 权重 {high_weight} (低优先级,几乎不可能)")
        print(f"  - 预测值超过{threshold_2:.0f}时额外惩罚")
    
    optimizer_config = config['optimizer']
    if optimizer_config['type'] == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            betas=optimizer_config['betas'],
            eps=optimizer_config['eps'],
            weight_decay=config['training']['weight_decay']
        )
    elif optimizer_config['type'] == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            betas=optimizer_config['betas'],
            eps=optimizer_config['eps'],
            weight_decay=config['training']['weight_decay']
        )
    
    # Learning rate scheduler
    scheduler_config = config['scheduler']
    if scheduler_config['type'] == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config['mode'],
            factor=scheduler_config['factor'],
            patience=scheduler_config['patience'],
            min_lr=scheduler_config['min_lr']
        )
    elif scheduler_config['type'] == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs'],
            eta_min=scheduler_config['min_lr']
        )
    
    # Training loop setup
    best_val_loss = float('inf')
    save_config = config['save']
    
    # 自动生成task name（如果name为空或不存在）
    task_config = config.get('task', {})
    task_name_raw = task_config.get('name', '')
    # 处理None或空值的情况
    if task_name_raw is None:
        task_name = ''
    else:
        task_name = str(task_name_raw).strip()
    if not task_name:
        task_name = generate_task_name(config)
        print(f"\n自动生成task name: {task_name}")
    else:
        print(f"\n使用指定的task name: {task_name}")
    output_dir = Path(save_config['output_dir']) / task_name
    output_dir.mkdir(parents=True, exist_ok=True)
    # Save config snapshot for reproducibility
    ts_model = datetime.now().strftime('%Y%m%d_%H%M%S')
    config_used_path = output_dir / 'config_used.yaml'
    try:
        with open(config_used_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
    except Exception as e:
        print(f"Failed to save config: {e}")
    
    epochs = config['training']['epochs']
    print_interval = config['logging']['print_interval']
    test_interval = int(config['logging'].get('test_interval', 0))
    
    # TensorBoard
    use_tensorboard = True
    if use_tensorboard:
        tb_dir = Path('results/tensorboard') / task_name
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_dir))

    print("\nTraining started...")
    print("="*60)
    
    # Start timer
    start_time = time.time()
    
    for epoch in range(epochs):
        # Train
        if is_vae:
            train_loss, train_loss_components = train_epoch(model, train_loader, criterion, optimizer, device, dataset, is_vae=True, is_residual=is_residual,
                                                           lambda_ic=lambda_ic, threshold_1=threshold_1, threshold_2=threshold_2,
                                                           low_weight=low_weight, mid_weight=mid_weight, high_weight=high_weight)
        else:
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device, dataset, is_vae=False, is_residual=False,
                                    lambda_ic=lambda_ic, threshold_1=threshold_1, threshold_2=threshold_2,
                                    low_weight=low_weight, mid_weight=mid_weight, high_weight=high_weight)
        
        # Validation
        if is_vae:
            val_loss, val_loss_components = compute_loss_on_loader(model, val_loader, criterion, device, dataset, is_vae=True, is_residual=is_residual,
                                                                  lambda_ic=lambda_ic, threshold_1=threshold_1, threshold_2=threshold_2,
                                                                  low_weight=low_weight, mid_weight=mid_weight, high_weight=high_weight)
        else:
            val_loss = compute_loss_on_loader(model, val_loader, criterion, device, dataset, is_vae=False, is_residual=False,
                                             lambda_ic=lambda_ic, threshold_1=threshold_1, threshold_2=threshold_2,
                                             low_weight=low_weight, mid_weight=mid_weight, high_weight=high_weight)
        
        # Update LR
        if scheduler_config['type'] == 'ReduceLROnPlateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Logging
        if (epoch + 1) % print_interval == 0:
            if is_vae:
                if is_residual:
                    print(f"Epoch [{epoch+1}/{epochs}] "
                          f"Train Loss: {train_loss:.6f} (recon={train_loss_components['recon']:.6f}, "
                          f"kl={train_loss_components['kl']:.6f}, e={train_loss_components['e_pred']:.6f}, "
                          f"ic={train_loss_components['ic']:.6f}) | "
                          f"Val Loss: {val_loss:.6f} | LR: {current_lr:.2e}")
                else:
                    print(f"Epoch [{epoch+1}/{epochs}] "
                          f"Train Loss: {train_loss:.6f} (recon={train_loss_components['recon']:.6f}, "
                          f"kl={train_loss_components['kl']:.6f}, e={train_loss_components['e_pred']:.6f}) | "
                          f"Val Loss: {val_loss:.6f} | LR: {current_lr:.2e}")
            else:
                print(f"Epoch [{epoch+1}/{epochs}] "
                      f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.2e}")

        # TensorBoard
        if use_tensorboard:
            writer.add_scalar('Loss/train', train_loss, epoch + 1)
            writer.add_scalar('Loss/val', val_loss, epoch + 1)
            writer.add_scalar('LR', current_lr, epoch + 1)
            
            # VAE专用日志
            if is_vae:
                writer.add_scalar('Loss/train_recon', train_loss_components['recon'], epoch + 1)
                writer.add_scalar('Loss/train_kl', train_loss_components['kl'], epoch + 1)
                writer.add_scalar('Loss/train_e_pred', train_loss_components['e_pred'], epoch + 1)
                writer.add_scalar('Loss/val_recon', val_loss_components['recon'], epoch + 1)
                writer.add_scalar('Loss/val_kl', val_loss_components['kl'], epoch + 1)
                writer.add_scalar('Loss/val_e_pred', val_loss_components['e_pred'], epoch + 1)
                # 残差模型的初值约束损失
                if is_residual:
                    writer.add_scalar('Loss/train_ic', train_loss_components['ic'], epoch + 1)
                    writer.add_scalar('Loss/val_ic', val_loss_components['ic'], epoch + 1)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_config['save_best']:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': _get_model_state_dict(model),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': config,
                    'normalize_temp': dataset.normalize_temp,
                    'use_log_e': dataset.use_log_e,
                    'temp_mean': dataset.temp_mean,
                    'temp_std': dataset.temp_std,
                    'e_mean': dataset.e_mean,
                    'e_std': dataset.e_std
                }
                torch.save(checkpoint, output_dir / 'best_model.pth')
                print(f"  → Best model saved (Val Loss: {val_loss:.6f})")
        
        # Periodic checkpoint
        if save_config['save_interval'] > 0 and (epoch + 1) % save_config['save_interval'] == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': _get_model_state_dict(model),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config,
                'normalize_temp': dataset.normalize_temp,
                'use_log_e': dataset.use_log_e,
                'temp_mean': dataset.temp_mean,
                'temp_std': dataset.temp_std,
                'e_mean': dataset.e_mean,
                'e_std': dataset.e_std
            }
            torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch+1}.pth')

        # Periodic test evaluation
        if test_interval > 0 and (epoch + 1) % test_interval == 0:
            test_results_dir = Path('results/test_result') / task_name
            test_results_dir.mkdir(parents=True, exist_ok=True)
            tstamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            test_stats = evaluate_on_loader(model, test_loader, dataset, device, is_vae=is_vae)
            # Save metrics
            periodic_metrics = test_results_dir / f'metrics_epoch_{epoch+1}_{tstamp}.json'
            with open(periodic_metrics, 'w', encoding='utf-8') as f:
                json.dump({
                    'epoch': epoch + 1,
                    'mae': test_stats['mae'],
                    'max_err': test_stats['max_err'],
                    'min_err': test_stats['min_err']
                }, f, ensure_ascii=False, indent=2)
            # Save top 10 samples
            periodic_samples = test_results_dir / f'samples_epoch_{epoch+1}_{tstamp}.csv'
            with open(periodic_samples, 'w', newline='', encoding='utf-8') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(['index', 'e_true', 'e_pred', 'abs_error'])
                for i, (e_true, e_pred) in enumerate(test_stats['samples']):
                    csv_writer.writerow([i, f"{e_true:.6f}", f"{e_pred:.6f}", f"{abs(e_pred - e_true):.6f}"])
            print(f"  → Test results saved: {periodic_metrics.name} & {periodic_samples.name}")
    
    # Save last model
    if save_config['save_last']:
        if is_vae:
            final_val_loss, _ = compute_loss_on_loader(model, val_loader, criterion, device, dataset, is_vae=True,
                                                      threshold_1=threshold_1, threshold_2=threshold_2,
                                                      low_weight=low_weight, mid_weight=mid_weight, high_weight=high_weight)
        else:
            final_val_loss = compute_loss_on_loader(model, val_loader, criterion, device, dataset, is_vae=False,
                                                    threshold_1=threshold_1, threshold_2=threshold_2,
                                                    low_weight=low_weight, mid_weight=mid_weight, high_weight=high_weight)
        checkpoint = {
            'epoch': epochs - 1,
            'model_state_dict': _get_model_state_dict(model),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': final_val_loss,
            'config': config,
            'normalize_temp': dataset.normalize_temp,
            'use_log_e': dataset.use_log_e,
            'temp_mean': dataset.temp_mean,
            'temp_std': dataset.temp_std,
            'e_mean': dataset.e_mean,
            'e_std': dataset.e_std
        }
        torch.save(checkpoint, output_dir / 'last_model.pth')

    # Rename best model with epoch number
    best_epoch = None
    best_path = output_dir / 'best_model.pth'
    if best_path.exists():
        try:
            best_ckpt = torch.load(best_path, map_location='cpu')
            best_epoch = int(best_ckpt.get('epoch', -1))
        except Exception:
            best_epoch = None
    if best_epoch is not None and best_epoch >= 0:
        renamed = output_dir / f"best_model_epoch_{best_epoch+1}.pth"
        torch.save(best_ckpt, renamed)
        print(f"Best model renamed: {renamed.name}")

    # Train set evaluation (original scale)
    print("\n" + "="*60)
    print("Train Set Evaluation")
    print("="*60)
    train_stats = evaluate_on_loader(model, train_loader, dataset, device, is_vae=is_vae)
    print(f"Train MAE: {train_stats['mae']:.6f}")
    print(f"Train Max Error: {train_stats['max_err']:.6f}")
    print(f"Train Min Error: {train_stats['min_err']:.6f}")
    print("\nTrain Top 10 Predictions:")
    for i, (e_true, e_pred) in enumerate(train_stats['samples']):
        print(f"Sample {i}: True: {e_true:.6f} | Pred: {e_pred:.6f} | Error: {abs(e_pred - e_true):.6f}")

    # Test set evaluation and save results
    print("\n" + "="*60)
    print("Test Set Evaluation")
    print("="*60)
    test_stats = evaluate_on_loader(model, test_loader, dataset, device, is_vae=is_vae)
    print(f"Test MAE: {test_stats['mae']:.6f}")
    print(f"Test Max Error: {test_stats['max_err']:.6f}")
    print(f"Test Min Error: {test_stats['min_err']:.6f}")
    print("\nTest Top 10 Predictions:")
    for i, (e_true, e_pred) in enumerate(test_stats['samples']):
        print(f"Sample {i}: True: {e_true:.6f} | Pred: {e_pred:.6f} | Error: {abs(e_pred - e_true):.6f}")

    test_results_dir = Path('results/test_result') / task_name
    test_results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    test_metrics_path = test_results_dir / f'metrics_{timestamp}.json'
    with open(test_metrics_path, 'w', encoding='utf-8') as f:
        json.dump({
            'mae': test_stats['mae'],
            'max_err': test_stats['max_err'],
            'min_err': test_stats['min_err']
        }, f, ensure_ascii=False, indent=2)
    test_samples_path = test_results_dir / f'samples_{timestamp}.csv'
    with open(test_samples_path, 'w', newline='', encoding='utf-8') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['index', 'e_true', 'e_pred', 'abs_error'])
        for i, (e_true, e_pred) in enumerate(test_stats['samples']):
            csv_writer.writerow([i, f"{e_true:.6f}", f"{e_pred:.6f}", f"{abs(e_pred - e_true):.6f}"])
    print(f"\nTest results saved: {test_metrics_path.name} & {test_samples_path.name}")

    # Generate and save full test predictions with parameters
    print("\nSaving full test predictions with parameters...")
    all_pred_rows = []
    test_indices = getattr(test_dataset, 'indices', None)
    if test_indices is None:
        start_idx = len(train_dataset)
        test_indices = list(range(start_idx, start_idx + len(test_dataset)))
    # Get parameter dimension and names
    param_dim = int(dataset.data['parameters'].shape[1])
    param_names = dataset.data.get('parameter_names', config.get('data', {}).get('parameter_names', []))
    if not param_names or len(param_names) != param_dim:
        param_names = [f'param_{j}' for j in range(param_dim)]
    
    # 创建样本详细信息目录（仅VAE模型）
    if is_vae:
        samples_dir = test_results_dir / 'samples'
        samples_dir.mkdir(parents=True, exist_ok=True)
        print(f"VAE模型：将保存每个样本的详细信息到 {samples_dir.name}/")
    
    # Inference
    model.eval()
    with torch.no_grad():
        for orig_idx in test_indices:
            temp_seq = dataset.temperature[orig_idx].to(device)
            e_true_norm = dataset.e_transformed[orig_idx]
            
            if is_vae:
                # VAE模型预测
                temp_3d = temp_seq.unsqueeze(0).unsqueeze(0)  # [1, 1, T]
                model_output = model(temp_3d)
                if len(model_output) == 5:
                    # Stage-Aware VAE: (recon, e_pred, (mu, logvar), temp_smooth, stage_weights)
                    recon_norm, e_pred_norm, _, _, _ = model_output
                else:
                    # 其他VAE: (recon, e_pred, (mu, logvar), temp_smooth)
                    recon_norm, e_pred_norm, _, _ = model_output
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
            
            # 获取T0和T1（假设在参数中的索引1和2）
            T0 = params_vec[1].item() if param_dim > 1 else 0.0
            T1 = params_vec[2].item() if param_dim > 2 else 0.0
            
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
                
                # 反归一化重建序列
                if dataset.normalize_temp:
                    recon_orig = recon_norm * dataset.temp_std + dataset.temp_mean
                else:
                    recon_orig = recon_norm
                
                # 确保长度一致
                actual_len = len(temp_orig_raw)
                if len(recon_orig) != actual_len:
                    # 如果长度不匹配，截断或填充
                    if len(recon_orig) > actual_len:
                        recon_orig = recon_orig[:actual_len]
                    else:
                        # 填充到相同长度
                        padding = torch.zeros(actual_len - len(recon_orig))
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

    # Write CSV
    full_pred_path = test_results_dir / f'all_predictions_{timestamp}.csv'
    fieldnames = ['index', 'e_true', 'e_pred', 'abs_error'] + param_names
    with open(full_pred_path, 'w', newline='', encoding='utf-8') as f:
        csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
        csv_writer.writeheader()
        for row in all_pred_rows:
            csv_writer.writerow(row)
    print(f"Full predictions saved: {full_pred_path.name}")
    
    if is_vae:
        print(f"VAE样本详细信息已保存到: {samples_dir.name}/ ({len(test_indices)} 个样本)")
    
    # Calculate total training time
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    print("\n" + "="*60)
    # Close TensorBoard writer
    if use_tensorboard:
        writer.close()

    print("Training completed!")
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Best model: {output_dir / 'best_model.pth'}")
    print(f"Total training time: {hours:02d}h {minutes:02d}m {seconds:02d}s ({total_time:.2f}s)")
    print("="*60)


if __name__ == '__main__':
    train()


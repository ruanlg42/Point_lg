import torch
import numpy as np
import yaml
from pathlib import Path

from dataset import ThermalDataset
from model import (TimeTransformer, CNN1D, PhysicsInformedCNN, 
                   PhysicsInformedTransformer, EnhancedPhysicsTransformer,
                   MambaPhysicsModel, HybridMambaTransformer, PhysTCN, EnhancedMambaPhysicsModel,
                   TimeVAE1D_Mamba, TimeVAE1D_Mamba_PhysicsDecoder, TimeVAE1D_Transformer,
                   TimeVAE1D_HybridMambaTransformer, TimeVAE1D_HybridMT_Physics4Channel,
                   TimeVAE1D_StageAware, TimeVAE1D_HybridMambaTransformer_Residual,
                   TimeVAE1D_SSTEncoder_Residual)


# ==================== 配置区域 ====================
TXT_PATH = '/home/ziwu/Newpython/lg_exp/Point_lg/gif_test/newnewnew_30hz/4730_video_center_temp.txt'
MODEL_PATH = '/home/ziwu/Newpython/lg_exp/Point_lg/results/model_result/Useful_A_dropout0__20251108_131250_timevae_hybrid_res_d256_latent256_mamba2_trans2_head4_ic0.5/best_model_epoch_493.pth'
CONFIG_PATH = '/home/ziwu/Newpython/lg_exp/Point_lg/results/model_result/Useful_A_dropout0__20251108_131250_timevae_hybrid_res_d256_latent256_mamba2_trans2_head4_ic0.5/config_used.yaml'
    
# ================================================


def load_temperature_from_txt(txt_path):
    """从txt文件读取温度序列
    
    支持两种格式：
    1. 标准格式：Frame  Pixel_Value  Temp_GIF(K)  Temp_Original(K)  Difference(K)
       → 读取第3列 (Temp_GIF)
    2. 简单格式：时间  温度
       → 读取第2列
    """
    temps = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            # 跳过注释、空行、分隔线、统计信息
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
            
            # 分割数据
            parts = line.split()
            
            if len(parts) >= 5:
                # 格式1：Frame  Pixel  Temp_GIF  Temp_Orig  Diff
                # 读取第3列 (索引2) - Temp_GIF
                try:
                    temp = float(parts[3])
                    temps.append(temp)
                except ValueError:
                    continue
            elif len(parts) >= 2:
                # 格式2：时间  温度
                # 读取第2列 (索引1)
                try:
                    temp = float(parts[1])
                    temps.append(temp)
                except ValueError:
                    continue
    
    return np.array(temps)


def load_model(model_path, config_path, device):
    """加载训练好的模型"""
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
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # 优先使用checkpoint中保存的配置（如果存在）
    if isinstance(checkpoint, dict) and 'config' in checkpoint:
        print("⚠️  检测到checkpoint中保存了配置，将使用checkpoint配置而非当前config.yaml")
        saved_config = checkpoint['config']
        
        # 比较关键参数是否一致
        if model_type == 'timevae1d_hybrid_mamba_transformer_residual':
            current_cfg = model_config.get('timevae1d_hybrid_mamba_transformer_residual', {})
            saved_cfg = saved_config['model'].get('timevae1d_hybrid_mamba_transformer_residual', {})
            
            if current_cfg.get('latent_dim') != saved_cfg.get('latent_dim'):
                print(f"   配置不匹配: latent_dim - 当前={current_cfg.get('latent_dim')}, checkpoint={saved_cfg.get('latent_dim')}")
                print(f"   将使用checkpoint配置重建模型...")
                
                # 使用checkpoint配置重建模型
                total_time = saved_config['data'].get('total_time', 5.0)
                delta_t = saved_config['data'].get('delta_t', 0.02)
                seq_len = int(total_time / delta_t) + 1
                model = TimeVAE1D_HybridMambaTransformer_Residual(
                    seq_len=seq_len,
                    C_in=saved_cfg.get('C_in', 1),
                    latent_dim=saved_cfg.get('latent_dim', 64),
                    d_model=saved_cfg.get('d_model', 128),
                    n_mamba=saved_cfg.get('n_mamba', 2),
                    n_transformer=saved_cfg.get('n_transformer', 2),
                    nhead=saved_cfg.get('nhead', 8),
                    d_state=saved_cfg.get('d_state', 16),
                    d_conv=saved_cfg.get('d_conv', 4),
                    expand=saved_cfg.get('expand', 2),
                    dropout=saved_cfg.get('dropout', 0.1),
                    decoder_base=saved_cfg.get('decoder_base', 64),
                    total_time=total_time,
                    delta_t=delta_t
                )
                print(f"   ✓ 模型已用checkpoint配置重建 (latent_dim={saved_cfg.get('latent_dim')})")
    
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
    
    # 加载归一化参数
    data_config = config['data']
    dataset_path = data_config['dataset_path']
    temp_dataset = ThermalDataset(
        pth_path=dataset_path,
        normalize_temp=data_config.get('normalize_temp', True),
        use_log_e=data_config.get('use_log_e', True)
    )
    
    return model, temp_dataset, config


def predict_effusivity(model, temperature_sequence, temp_dataset, device, target_steps=251, is_vae=False):
    """预测单个温度序列的热扩散系数
    
    Args:
        model: 预测模型
        temperature_sequence: 温度序列
        temp_dataset: 数据集（用于归一化）
        device: 设备
        target_steps: 目标时间步数
        is_vae: 是否为VAE模型
    
    Returns:
        pred: 预测的热扩散系数
    """
    T = len(temperature_sequence)
    
    # 如果时间步数不匹配，进行插值
    if T != target_steps:
        print(f"  Temperature sequence has {T} points, interpolating to {target_steps} steps...")
        from scipy.interpolate import interp1d
        
        old_time = np.linspace(0, 1, T)
        new_time = np.linspace(0, 1, target_steps)
        
        interpolator = interp1d(old_time, temperature_sequence, kind='linear')
        temperature_sequence = interpolator(new_time)
        T = target_steps
    
    # 归一化
    if temp_dataset.normalize_temp:
        temperature_sequence = (temperature_sequence - temp_dataset.temp_mean.numpy()) / temp_dataset.temp_std.numpy()
    
    # 转为 tensor
    temp_tensor = torch.from_numpy(temperature_sequence).unsqueeze(0).float().to(device)
    
    # 预测
    with torch.no_grad():
        if is_vae:
            # VAE模型：需要 [1, 1, T] 输入
            temp_tensor_3d = temp_tensor.unsqueeze(1)  # [1, 1, T]
            model_output = model(temp_tensor_3d)
            
            # 处理不同的VAE模型输出格式
            if len(model_output) == 5:
                # Residual VAE: (recon, e_pred, (mu, logvar), temp_smooth, x_delta)
                _, e_pred, _, _, _ = model_output
            elif len(model_output) == 4:
                # 标准VAE: (recon, e_pred, (mu, logvar), temp_smooth)
                _, e_pred, _, _ = model_output
            else:
                # 简单VAE: (recon, e_pred, (mu, logvar))
                _, e_pred, _ = model_output
            
            pred = e_pred.cpu().numpy().flatten()[0]
        else:
            # 普通模型：[1, T] 输入
            pred = model(temp_tensor).cpu().numpy().flatten()[0]
    
    # 反归一化
    if temp_dataset.use_log_e:
        pred = pred * temp_dataset.e_std.numpy() + temp_dataset.e_mean.numpy()
        pred = np.exp(pred)
    
    return pred


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("=" * 60)
    
    # 加载模型
    print("Loading model...")
    model, temp_dataset, config = load_model(MODEL_PATH, CONFIG_PATH, device)
    model_type = config['model']['type']
    print(f"Model loaded: {model_type}")
    print("=" * 60)
    
    # 判断是否为VAE模型
    is_vae = (model_type.lower() == 'timevae1d_mamba' or 
              model_type.lower() == 'timevae1d_mamba_physics_decoder' or 
              model_type.lower() == 'timevae1d_transformer' or
              model_type.lower() == 'timevae1d_hybrid_mamba_transformer' or
              model_type.lower() == 'timevae1d_hybridmt_physics4channel' or
              model_type.lower() == 'timevae1d_stageaware' or
              model_type.lower() == 'timevae1d_hybrid_mamba_transformer_residual' or
              model_type.lower() == 'timevae1d_sst_encoder_residual')
    
    # 获取模型的seq_len
    total_time = config['data'].get('total_time', 5.0)
    delta_t = config['data'].get('delta_t', 0.02)
    expected_seq_len = int(total_time / delta_t) + 1
    
    # 读取温度序列
    print(f"\nReading temperature from: {TXT_PATH}")
    temperature_sequence = load_temperature_from_txt(TXT_PATH)
    print(f"Temperature sequence length: {len(temperature_sequence)} points")
    print(f"Temperature range: {temperature_sequence.min():.2f} - {temperature_sequence.max():.2f} K")
    print(f"Temperature mean: {temperature_sequence.mean():.2f} K")
    print(f"Temperature std: {temperature_sequence.std():.2f} K")
    print("=" * 60)
    
    # 预测
    print("\nPredicting thermal effusivity...")
    print(f"Model expects sequence length: {expected_seq_len}")
    predicted_e = predict_effusivity(model, temperature_sequence, temp_dataset, device, target_steps=expected_seq_len, is_vae=is_vae)
    
    print("\n" + "=" * 60)
    print(f"PREDICTED THERMAL EFFUSIVITY: {predicted_e:.2f} J·m⁻²·K⁻¹·s⁻¹/²")
    print("=" * 60)


if __name__ == '__main__':
    main()


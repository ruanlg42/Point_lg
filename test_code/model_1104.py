import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d
import numpy as np

from mamba_ssm import Mamba


# ==================== Kalman Filter + RTS Smoother ====================

def kalman_cv_smooth_batch(T, delta_t, Q_scale=1e-1, R_scale=1e-3, rts=True):
    """
    批量Kalman滤波 + RTS平滑（匀速模型 Constant Velocity）
    
    状态: [T, dT/dt]
    适用于: 平滑变化的信号（如热扩散）
    
    Args:
        T: [batch, seq_len] 原始温度序列
        delta_t: 采样间隔
        Q_scale: 过程噪声强度（越小越平滑，推荐1e-1用于轻度平滑）
        R_scale: 观测噪声强度（越大越平滑，推荐1e-3用于轻度平滑）
        rts: 是否使用RTS平滑器（双向，更优）
    
    Returns:
        T_smooth: [batch, seq_len] 平滑后的温度
        dT_smooth: [batch, seq_len] 平滑后的导数
    """
    batch_size, L = T.shape
    device = T.device
    dtype = T.dtype
    
    # 状态转移矩阵 F = [[1, dt], [0, 1]]
    F = torch.tensor([[1.0, delta_t], [0.0, 1.0]], device=device, dtype=dtype)
    
    # 观测矩阵 H = [1, 0]（只观测温度）
    H = torch.tensor([[1.0, 0.0]], device=device, dtype=dtype)
    
    # 过程噪声协方差 Q（连续白噪声离散化）
    Q = torch.tensor([
        [delta_t**3/3, delta_t**2/2],
        [delta_t**2/2, delta_t]
    ], device=device, dtype=dtype) * Q_scale
    
    # 观测噪声协方差 R
    R = torch.tensor([[R_scale]], device=device, dtype=dtype)
    
    # 初始化
    x = torch.zeros(batch_size, 2, 1, device=device, dtype=dtype)  # [batch, 2, 1]
    x[:, 0, 0] = T[:, 0]  # 初始温度
    x[:, 1, 0] = 0.0      # 初始速度=0
    
    P = torch.eye(2, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, 2, 2).clone()
    P = P * 1.0  # 初始协方差
    
    # 前向滤波
    x_filt = []
    P_filt = []
    
    for t in range(L):
        # 预测
        if t > 0:
            x = F @ x  # [batch, 2, 1]
            P = F @ P @ F.T + Q  # [batch, 2, 2]
        
        # 更新
        y = T[:, t:t+1].unsqueeze(-1)  # [batch, 1, 1] 观测值
        y_pred = H @ x  # [batch, 1, 1]
        innov = y - y_pred  # 新息
        
        S = H @ P @ H.T + R  # [batch, 1, 1] 新息协方差
        K = P @ H.T / S  # [batch, 2, 1] Kalman增益（S是标量，直接除）
        
        x = x + K @ innov
        P = (torch.eye(2, device=device, dtype=dtype).unsqueeze(0) - K @ H) @ P
        
        x_filt.append(x.clone())
        P_filt.append(P.clone())
    
    # RTS平滑（反向）
    if rts:
        x_smooth = [x_filt[-1]]
        P_smooth = [P_filt[-1]]
        
        for t in range(L-2, -1, -1):
            x_filt_t = x_filt[t]
            P_filt_t = P_filt[t]
            
            # 预测到t+1
            x_pred = F @ x_filt_t
            P_pred = F @ P_filt_t @ F.T + Q
            
            # 平滑增益：C = P_filt_t @ F.T @ P_pred^{-1}
            # 使用 pinverse（伪逆）代替 inverse，在多GPU环境下更稳定
            P_pred_inv = torch.linalg.pinv(P_pred)
            C = P_filt_t @ F.T @ P_pred_inv
            
            # 平滑
            x_smooth_next = x_smooth[0]
            P_smooth_next = P_smooth[0]
            
            x_s = x_filt_t + C @ (x_smooth_next - x_pred)
            P_s = P_filt_t + C @ (P_smooth_next - P_pred) @ C.transpose(-2, -1)
            
            x_smooth.insert(0, x_s)
            P_smooth.insert(0, P_s)
        
        x_out = torch.stack([x[:, :, 0] for x in x_smooth], dim=1)  # [batch, L, 2]
    else:
        x_out = torch.stack([x[:, :, 0] for x in x_filt], dim=1)  # [batch, L, 2]
    
    T_smooth = x_out[:, :, 0]    # [batch, L]
    dT_smooth = x_out[:, :, 1]   # [batch, L]
    
    return T_smooth, dT_smooth


def offline_rts_smooth_sequences(T_list, delta_t, Q_scale=1e-1, R_scale=1e-3, device="cuda"):
    """
    离线批量RTS平滑（用于数据预处理）- 使用匀速模型(CV)
    
    Args:
        T_list: list[np.ndarray or torch.Tensor]，每个元素形状 [L]
        delta_t: 采样间隔
        Q_scale: 过程噪声（推荐1e-1，轻度平滑）
        R_scale: 观测噪声（推荐1e-3，轻度平滑）
        device: 'cuda' or 'cpu'
    
    Returns:
        feats_list: list[torch.Tensor]，每个元素 [L, 4]（物理特征）
    
    使用示例:
        raw_sequences = [np.load(...), ...]  # 每个是 [L]
        feats = offline_rts_smooth_sequences(raw_sequences, delta_t=0.02)
        torch.save(feats, "feats_rts_smooth.pt")
    """
    import numpy as np
    
    feats_list = []
    for T_np in T_list:
        T = torch.as_tensor(T_np, dtype=torch.float32, device=device).unsqueeze(0)  # [1, L]
        
        # 匀速模型 Kalman RTS 平滑
        T_s, dT_s = kalman_cv_smooth_batch(T, delta_t, Q_scale=Q_scale, R_scale=R_scale, rts=True)
        T_s = T_s.squeeze(0)
        dT_s = dT_s.squeeze(0)
        T_dot = dT_s  # 直接使用平滑后的导数
        
        # 构建4通道物理特征
        L = T_s.shape[0]
        T_tilde = T_s - T_s[0]
        
        time_steps = torch.arange(L, device=device, dtype=T_s.dtype) * delta_t
        time_steps[0] = delta_t
        t_inv_sqrt = 1.0 / torch.sqrt(time_steps.clamp_min(1e-12))
        
        total_time = delta_t * (L - 1) if L > 1 else delta_t
        delta_t_norm = torch.full_like(T_s, delta_t / total_time)
        
        feat = torch.stack([T_tilde, T_dot, t_inv_sqrt, delta_t_norm], dim=-1)  # [L, 4]
        feats_list.append(feat.cpu())
    
    return feats_list



class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TimeTransformer(nn.Module):
    """Time series Transformer for predicting parameter e"""
    
    def __init__(self, d_model=128, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.2):
        super().__init__()
        
        # Input projection: temperature sequence -> d_model dimension
        self.input_projection = nn.Linear(1, d_model)
        # BatchNorm over feature dimension (requires [N, C, L])
        self.input_bn = nn.BatchNorm1d(d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer: global average pooling + MLP
        self.fc = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)  # Predict e value
        )
    
    def forward(self, temp_sequence):
        """
        Args:
            temp_sequence: [batch, seq_len] temperature sequence
        Returns:
            e_pred: [batch, 1] predicted e value
        """
        # [batch, seq_len] -> [batch, seq_len, 1]
        x = temp_sequence.unsqueeze(-1)
        
        # Input projection [batch, seq_len, 1] -> [batch, seq_len, d_model]
        x = self.input_projection(x)
        # BatchNorm1d: transpose to [N, C, L], BN, then transpose back
        x = x.transpose(1, 2)
        x = self.input_bn(x)
        x = x.transpose(1, 2)
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)
        
        # Global average pooling [batch, seq_len, d_model] -> [batch, d_model]
        x = x.mean(dim=1)
        
        # Predict e value
        e_pred = self.fc(x)
        
        return e_pred


class CNN1D(nn.Module):
    """1D CNN for time series prediction"""
    
    def __init__(self, hidden_channels=64, num_layers=4, kernel_size=3, dropout=0.2):
        super().__init__()
        
        # Build CNN layers
        layers = []
        in_channels = 1
        
        for i in range(num_layers):
            out_channels = hidden_channels * (2 ** i) if i < 3 else hidden_channels * 4
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2)
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Output MLP (adaptive pooling + fully connected)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, temp_sequence):
        """
        Args:
            temp_sequence: [batch, seq_len] temperature sequence
        Returns:
            e_pred: [batch, 1] predicted e value
        """
        # [batch, seq_len] -> [batch, 1, seq_len]
        x = temp_sequence.unsqueeze(1)
        
        # CNN encoding
        x = self.conv_layers(x)
        
        # Global pooling [batch, channels, seq_len] -> [batch, channels, 1]
        x = self.global_pool(x)
        
        # Flatten [batch, channels, 1] -> [batch, channels]
        x = x.squeeze(-1)
        
        # Predict e value
        e_pred = self.fc(x)
        
        return e_pred


class PhysicsInformedTransformer(nn.Module):
    """
    Physics-Informed Transformer for time series:
    输入为温度序列，经物理先验预处理为 4 通道特征 [T_tilde, T_dot, t^{-1/2}, Delta_t]，
    然后按时间步作为序列，线性投影到 d_model，叠加位置编码，Transformer 编码，
    最后全局平均池化 + MLP 输出标量 e。
    """

    def __init__(self, d_model=256, nhead=8, num_layers=4, dim_feedforward=1024, dropout=0.2,
                 total_time=5.0, delta_t=0.02):
        super().__init__()

        self.total_time = total_time
        self.delta_t = delta_t

        # 4通道输入 -> d_model 投影（逐时间步的特征向量）
        self.input_projection = nn.Linear(4, d_model)
        # 输入归一化（BatchNorm1d 作用于特征维度）
        self.input_bn = nn.BatchNorm1d(d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出 MLP
        self.fc = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def _preprocess_sequence(self, temp_sequence):
        """
        temp_sequence: [batch, seq_len]
        返回: [batch, seq_len, 4]
        """
        batch_size, seq_len = temp_sequence.shape
        device = temp_sequence.device

        # 高斯滤波平滑 (sigma=2)
        temp_np = temp_sequence.cpu().numpy()
        T_smooth_np = np.array([gaussian_filter1d(temp_np[i], sigma=2.0) for i in range(batch_size)])
        T_smooth = torch.from_numpy(T_smooth_np).to(device).float()

        # T_tilde: 去除初值
        T_tilde = T_smooth - T_smooth[:, 0:1]

        # T_dot: 数值微分
        T_dot = torch.zeros_like(T_smooth)
        T_dot[:, :-1] = (T_smooth[:, 1:] - T_smooth[:, :-1]) / self.delta_t
        T_dot[:, -1] = T_dot[:, -2]  # 最后一个点复制前一个

        # t^{-1/2}
        time_steps = torch.arange(0, seq_len, device=device, dtype=torch.float32) * self.delta_t
        if seq_len > 0:
            time_steps[0] = self.delta_t
        t_inv_sqrt = 1.0 / torch.sqrt(time_steps.clamp_min(1e-12))
        t_inv_sqrt = t_inv_sqrt.unsqueeze(0).expand(batch_size, -1)

        # Delta_t 归一化
        delta_t_normalized = torch.full_like(temp_sequence, self.delta_t / self.total_time)

        # 拼接为 [batch, seq_len, 4]
        features = torch.stack([T_tilde, T_dot, t_inv_sqrt, delta_t_normalized], dim=-1)
        return features

    def forward(self, temp_sequence):
        # 预处理: [batch, seq_len] -> [batch, seq_len, 4]
        x = self._preprocess_sequence(temp_sequence)
        # 投影到 d_model
        x = self.input_projection(x)
        # 归一化（BatchNorm1d 需要 [N, C, L]）
        x = x.transpose(1, 2)
        x = self.input_bn(x)
        x = x.transpose(1, 2)
        # 位置编码
        x = self.pos_encoder(x)
        # Transformer 编码
        x = self.transformer_encoder(x)
        # 全局平均池化
        x = x.mean(dim=1)
        # 输出回归
        e_pred = self.fc(x)
        return e_pred

class PhysicsInformedCNN(nn.Module):
    """
    Physics-Informed 1D CNN with 4-channel input:
    [T_tilde, T_dot, t^{-1/2}, Delta_t]
    """
    
    def __init__(self, hidden_channels=64, num_layers=3, kernel_size=5, dropout=0.2, 
                 total_time=5.0, delta_t=0.02):
        super().__init__()
        
        # Store time sampling parameters
        self.total_time = total_time
        self.delta_t = delta_t
        
        # Build CNN layers - input now has 4 channels instead of 1
        layers = []
        in_channels = 4  # 4-channel input
        
        for i in range(num_layers):
            out_channels = hidden_channels * (2 ** i) if i < 3 else hidden_channels * 4
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2)
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Output MLP
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def preprocess_sequence(self, temp_sequence):
        """
        Preprocess temperature sequence into 4-channel input
        
        Args:
            temp_sequence: [batch, seq_len] raw temperature sequence
        
        Returns:
            features: [batch, 4, seq_len] preprocessed features
        """
        batch_size, seq_len = temp_sequence.shape
        device = temp_sequence.device
        
        # Channel 1: T_tilde (relative temperature, remove initial value)
        T_tilde = temp_sequence - temp_sequence[:, 0:1]  # [batch, seq_len]
        
        # Channel 2: T_dot (temperature derivative, numerical differentiation)
        # dT/dt ≈ (T[i+1] - T[i]) / delta_t
        T_dot = torch.zeros_like(temp_sequence)
        T_dot[:, :-1] = (temp_sequence[:, 1:] - temp_sequence[:, :-1]) / self.delta_t
        T_dot[:, -1] = T_dot[:, -2]  # Copy last value
        
        # Channel 3: t^{-1/2} (physical prior for semi-infinite body)
        # t_i = i * delta_t (uniform sampling)
        time_steps = torch.arange(0, seq_len, device=device, dtype=torch.float32) * self.delta_t
        # Avoid division by zero at t=0
        time_steps[0] = self.delta_t  # Use first step instead of 0
        t_inv_sqrt = 1.0 / torch.sqrt(time_steps)  # [seq_len]
        t_inv_sqrt = t_inv_sqrt.unsqueeze(0).expand(batch_size, -1)  # [batch, seq_len]
        
        # Channel 4: Delta_t (sampling interval, normalized)
        # For uniform sampling, this is constant = delta_t
        # Normalize by total_time for better numerical stability
        delta_t_normalized = torch.full_like(temp_sequence, self.delta_t / self.total_time)
        
        # Stack all channels: [batch, 4, seq_len]
        features = torch.stack([T_tilde, T_dot, t_inv_sqrt, delta_t_normalized], dim=1)
        
        return features
    
    def forward(self, temp_sequence):
        """
        Args:
            temp_sequence: [batch, seq_len] raw temperature sequence
        Returns:
            e_pred: [batch, 1] predicted e value
        """
        # Preprocess to 4-channel input
        x = self.preprocess_sequence(temp_sequence)  # [batch, 4, seq_len]
        
        # CNN encoding
        x = self.conv_layers(x)
        
        # Global pooling
        x = self.global_pool(x)  # [batch, channels, 1]
        
        # Flatten
        x = x.squeeze(-1)  # [batch, channels]
        
        # Predict e value
        e_pred = self.fc(x)
        
        return e_pred


class TemporalAttentionPooling(nn.Module):
    """
    自适应时序注意力池化（方法4）
    替代简单的平均池化，自动学习哪些时间步更重要
    """
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        # 多头注意力权重计算
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        
        # 输出投影
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model] 时序特征
        Returns:
            pooled: [batch, d_model] 加权聚合后的特征
        """
        batch_size, seq_len, d_model = x.shape
        
        # 多头注意力
        # Query: 使用全局平均作为查询（我们想知道哪些时间步重要）
        q = x.mean(dim=1, keepdim=True)  # [batch, 1, d_model]
        q = self.query_proj(q)  # [batch, 1, d_model]
        
        # Key & Value: 所有时间步
        k = self.key_proj(x)  # [batch, seq_len, d_model]
        v = self.value_proj(x)  # [batch, seq_len, d_model]
        
        # 计算注意力权重
        # [batch, 1, d_model] @ [batch, d_model, seq_len] -> [batch, 1, seq_len]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_model)
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [batch, 1, seq_len]
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和
        # [batch, 1, seq_len] @ [batch, seq_len, d_model] -> [batch, 1, d_model]
        pooled = torch.matmul(attn_weights, v).squeeze(1)  # [batch, d_model]
        
        # 输出投影
        pooled = self.out_proj(pooled)
        
        return pooled, attn_weights.squeeze(1)  # 返回注意力权重用于可视化


class EnhancedPhysicsTransformer(nn.Module):
    """
    Enhanced Physics-Informed Transformer with:
    - Method 3: Residual & Skip Connections
    - Method 4: Adaptive Temporal Attention Pooling
    
    基于 PhysicsInformedTransformer 的增强版本
    """
    
    def __init__(self, d_model=256, nhead=8, num_layers=4, dim_feedforward=1024, dropout=0.2,
                 total_time=5.0, delta_t=0.02):
        super().__init__()
        
        self.total_time = total_time
        self.delta_t = delta_t
        self.d_model = d_model
        
        # 4通道物理特征 -> d_model 投影
        self.input_projection = nn.Linear(4, d_model)
        self.input_bn = nn.BatchNorm1d(d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 方法4：自适应时序注意力池化（替代平均池化）
        self.temporal_attention = TemporalAttentionPooling(d_model, num_heads=4)
        
        # 方法3：物理特征直连路径（跳跃连接）
        # 提取全局统计特征作为辅助信息
        self.physics_direct_path = nn.Sequential(
            nn.Linear(4, 64),  # 4个物理通道的统计信息
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32)
        )
        
        # 方法3：多路径融合
        # 主路径：Transformer特征 (d_model)
        # 辅助路径：物理直连特征 (32)
        fusion_dim = d_model + 32
        
        # 融合后的输出 MLP（方法3：残差连接）
        self.fc_fusion = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 最终输出层
        self.fc_out = nn.Linear(64, 1)
        
        # 方法3：残差连接的辅助预测路径（可选）
        self.auxiliary_predictor = nn.Linear(32, 1)
        
    def _preprocess_sequence(self, temp_sequence):
        """
        物理特征预处理
        temp_sequence: [batch, seq_len]
        返回: [batch, seq_len, 4]
        """
        batch_size, seq_len = temp_sequence.shape
        device = temp_sequence.device
        
        # 高斯滤波平滑 (sigma=2)
        temp_np = temp_sequence.cpu().numpy()
        T_smooth_np = np.array([gaussian_filter1d(temp_np[i], sigma=2.0) for i in range(batch_size)])
        T_smooth = torch.from_numpy(T_smooth_np).to(device).float()
        
        # T_tilde: 去除初值
        T_tilde = T_smooth - T_smooth[:, 0:1]
        
        # T_dot: 数值微分
        T_dot = torch.zeros_like(T_smooth)
        T_dot[:, :-1] = (T_smooth[:, 1:] - T_smooth[:, :-1]) / self.delta_t
        T_dot[:, -1] = T_dot[:, -2]
        
        # t^{-1/2}
        time_steps = torch.arange(0, seq_len, device=device, dtype=torch.float32) * self.delta_t
        if seq_len > 0:
            time_steps[0] = self.delta_t
        t_inv_sqrt = 1.0 / torch.sqrt(time_steps.clamp_min(1e-12))
        t_inv_sqrt = t_inv_sqrt.unsqueeze(0).expand(batch_size, -1)
        
        # Delta_t 归一化
        delta_t_normalized = torch.full_like(temp_sequence, self.delta_t / self.total_time)
        
        # 拼接为 [batch, seq_len, 4]
        features = torch.stack([T_tilde, T_dot, t_inv_sqrt, delta_t_normalized], dim=-1)
        return features
    
    def _extract_physics_stats(self, physics_features):
        """
        提取物理特征的全局统计信息作为跳跃连接
        physics_features: [batch, seq_len, 4]
        返回: [batch, 4] 每个通道的统计量
        """
        # 简单均值统计（可扩展为 mean + std + max 等）
        stats = physics_features.mean(dim=1)  # [batch, 4]
        return stats
    
    def forward(self, temp_sequence, return_attention=False):
        """
        Args:
            temp_sequence: [batch, seq_len] 原始温度序列
            return_attention: 是否返回注意力权重（用于可视化）
        Returns:
            e_pred: [batch, 1] 预测的 e 值
            (可选) attn_weights: [batch, seq_len] 时序注意力权重
        """
        # 1. 物理特征预处理
        physics_features = self._preprocess_sequence(temp_sequence)  # [batch, seq_len, 4]
        
        # 2. 方法3：物理特征直连路径（跳跃连接）
        physics_stats = self._extract_physics_stats(physics_features)  # [batch, 4]
        physics_direct = self.physics_direct_path(physics_stats)  # [batch, 32]
        
        # 3. 主路径：投影到 d_model
        x = self.input_projection(physics_features)  # [batch, seq_len, d_model]
        
        # 4. BatchNorm
        x = x.transpose(1, 2)
        x = self.input_bn(x)
        x = x.transpose(1, 2)
        
        # 5. 位置编码
        x = self.pos_encoder(x)
        
        # 6. Transformer 编码
        x_encoded = self.transformer_encoder(x)  # [batch, seq_len, d_model]
        
        # 7. 方法4：自适应时序注意力池化（替代平均池化）
        x_pooled, attn_weights = self.temporal_attention(x_encoded)  # [batch, d_model]
        
        # 8. 方法3：多路径融合（主路径 + 物理直连路径）
        x_fused = torch.cat([x_pooled, physics_direct], dim=-1)  # [batch, d_model + 32]
        
        # 9. 融合后的 MLP
        x_hidden = self.fc_fusion(x_fused)  # [batch, 64]
        
        # 10. 最终预测
        e_pred_main = self.fc_out(x_hidden)  # [batch, 1]
        
        # 11. 方法3：辅助预测路径（物理直连）+ 残差连接
        e_pred_aux = self.auxiliary_predictor(physics_direct)  # [batch, 1]
        
        # 残差组合：主路径 + 辅助路径（权重可学习或固定）
        e_pred = e_pred_main + 0.1 * e_pred_aux  # 0.1是辅助路径的权重
        
        if return_attention:
            return e_pred, attn_weights
        return e_pred


class MambaPhysicsModel(nn.Module):
    """
    Mamba-based Physics-Informed Model for thermal effusivity prediction
    
    Advantages over Transformer:
    - O(L) complexity vs O(L²) - 250x faster for L=251
    - Natural for causal time series (heat diffusion)
    - Better long-range dependency modeling
    - Lower memory footprint
    
    Architecture:
    1. Physics feature preprocessing (4 channels)
    2. Mamba layers for temporal modeling
    3. Physics direct path (skip connection)
    4. Multi-path fusion
    5. Output MLP
    """
    
    def __init__(self, d_model=256, n_layers=4, d_state=16, d_conv=4, expand=2, 
                 dropout=0.2, total_time=5.0, delta_t=0.02):
        """
        Args:
            d_model: Model dimension (hidden size)
            n_layers: Number of Mamba layers
            d_state: SSM state dimension (typically 8-32)
            d_conv: Local convolution width (typically 4)
            expand: Expansion factor (typically 2)
            dropout: Dropout rate
            total_time: Total sampling time (seconds)
            delta_t: Sampling interval (seconds)
        """
        super().__init__()
        
        
        self.total_time = total_time
        self.delta_t = delta_t
        self.d_model = d_model
        
        # 4-channel physics features -> d_model projection
        self.input_projection = nn.Linear(4, d_model)
        self.input_bn = nn.BatchNorm1d(d_model)
        
        # Mamba layers with residual connections
        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            for _ in range(n_layers)
        ])
        
        # Layer normalization for each Mamba layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(n_layers)
        ])
        
        # Dropout for each layer
        self.dropouts = nn.ModuleList([
            nn.Dropout(dropout)
            for _ in range(n_layers)
        ])
        
        # Physics direct path (skip connection)
        self.physics_direct_path = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32)
        )
        
        # Multi-path fusion
        fusion_dim = d_model + 32
        self.fc_fusion = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output layers
        self.fc_out = nn.Linear(64, 1)
        self.auxiliary_predictor = nn.Linear(32, 1)
    
    def _preprocess_sequence(self, temp_sequence):
        """
        Preprocess temperature sequence into 4-channel physics features
        
        Args:
            temp_sequence: [batch, seq_len] raw temperature
        Returns:
            features: [batch, seq_len, 4] physics features
                Channel 0: T_tilde (relative temperature)
                Channel 1: T_dot (temperature derivative)
                Channel 2: t^{-1/2} (semi-infinite body prior)
                Channel 3: Δt (normalized sampling interval)
        """
        batch_size, seq_len = temp_sequence.shape
        device = temp_sequence.device
        
        # 高斯滤波平滑 (sigma=2)
        temp_np = temp_sequence.cpu().numpy()
        T_smooth_np = np.array([gaussian_filter1d(temp_np[i], sigma=2.0) for i in range(batch_size)])
        T_smooth = torch.from_numpy(T_smooth_np).to(device).float()
        
        # Channel 0: T_tilde (remove initial value)
        T_tilde = T_smooth - T_smooth[:, 0:1]
        
        # Channel 1: T_dot (numerical differentiation)
        T_dot = torch.zeros_like(T_smooth)
        T_dot[:, :-1] = (T_smooth[:, 1:] - T_smooth[:, :-1]) / self.delta_t
        T_dot[:, -1] = T_dot[:, -2]
        
        # Channel 2: t^{-1/2} (physical prior)
        time_steps = torch.arange(0, seq_len, device=device, dtype=torch.float32) * self.delta_t
        time_steps[0] = self.delta_t  # Avoid division by zero
        t_inv_sqrt = 1.0 / torch.sqrt(time_steps.clamp_min(1e-12))
        t_inv_sqrt = t_inv_sqrt.unsqueeze(0).expand(batch_size, -1)
        
        # Channel 3: Δt (normalized sampling interval)
        delta_t_normalized = torch.full_like(temp_sequence, self.delta_t / self.total_time)
        
        # Stack: [batch, seq_len, 4]
        features = torch.stack([T_tilde, T_dot, t_inv_sqrt, delta_t_normalized], dim=-1)
        return features
    
    def _extract_physics_stats(self, physics_features):
        """
        Extract global statistics from physics features
        
        Args:
            physics_features: [batch, seq_len, 4]
        Returns:
            stats: [batch, 4] mean over time dimension
        """
        return physics_features.mean(dim=1)
    
    def forward(self, temp_sequence, return_hidden=False):
        """
        Forward pass
        
        Args:
            temp_sequence: [batch, seq_len] raw temperature sequence
            return_hidden: Whether to return hidden states (for analysis)
        Returns:
            e_pred: [batch, 1] predicted thermal effusivity
            (optional) hidden_states: List of hidden states from each layer
        """
        # 1. Physics feature preprocessing
        physics_features = self._preprocess_sequence(temp_sequence)  # [batch, seq_len, 4]
        
        # 2. Physics direct path (skip connection)
        physics_stats = self._extract_physics_stats(physics_features)  # [batch, 4]
        physics_direct = self.physics_direct_path(physics_stats)  # [batch, 32]
        
        # 3. Project to d_model
        x = self.input_projection(physics_features)  # [batch, seq_len, d_model]
        
        # 4. BatchNorm (transpose for [N, C, L] format)
        x = x.transpose(1, 2)
        x = self.input_bn(x)
        x = x.transpose(1, 2)
        
        # 5. Mamba layers with residual connections
        hidden_states = []
        for mamba_layer, layer_norm, dropout in zip(
            self.mamba_layers, self.layer_norms, self.dropouts
        ):
            # Mamba forward
            x_residual = x
            x = mamba_layer(x)  # [batch, seq_len, d_model]
            x = dropout(x)
            x = layer_norm(x + x_residual)  # Residual connection + LayerNorm
            
            if return_hidden:
                hidden_states.append(x.detach())
        
        # 6. Global average pooling
        x_pooled = x.mean(dim=1)  # [batch, d_model]
        
        # 7. Multi-path fusion
        x_fused = torch.cat([x_pooled, physics_direct], dim=-1)  # [batch, d_model + 32]
        
        # 8. Fusion MLP
        x_hidden = self.fc_fusion(x_fused)  # [batch, 64]
        
        # 9. Main prediction
        e_pred_main = self.fc_out(x_hidden)  # [batch, 1]
        
        # 10. Auxiliary prediction (physics direct path)
        e_pred_aux = self.auxiliary_predictor(physics_direct)  # [batch, 1]
        
        # 11. Residual combination
        e_pred = e_pred_main + 0.1 * e_pred_aux
        
        if return_hidden:
            return e_pred, hidden_states
        return e_pred


class EnhancedMambaPhysicsModel(nn.Module):
    """
    增强版Mamba物理模型 (Enhanced Mamba Physics Model)
    
    改进点：
    1. 可学习查询池化（替代简单平均池化）
    2. 异方差输出头（不确定性估计）
    3. 多层级特征融合（提取不同层的特征）
    4. 更丰富的物理统计特征（均值、最大值、最小值、标准差）
    5. 密集残差连接（提升梯度流动）
    6. 改进的融合网络结构
    
    优势：
    - 保持O(L)复杂度（Mamba的核心优势）
    - 更好的特征表示能力
    - 提供预测不确定性
    - 对噪声更鲁棒
    """
    
    def __init__(self, d_model=256, n_layers=6, d_state=16, d_conv=4, expand=2, 
                 dropout=0.2, total_time=5.0, delta_t=0.02, use_multi_scale=True):
        """
        Args:
            d_model: 模型维度
            n_layers: Mamba层数
            d_state: SSM状态维度（推荐16-32）
            d_conv: 局部卷积宽度（推荐4）
            expand: 扩展因子（推荐2）
            dropout: Dropout率
            total_time: 总采样时间
            delta_t: 采样间隔
            use_multi_scale: 是否使用多层级特征提取
        """
        super().__init__()
        
        self.total_time = total_time
        self.delta_t = delta_t
        self.d_model = d_model
        self.use_multi_scale = use_multi_scale
        
        # 4通道物理特征 -> d_model投影
        self.input_projection = nn.Linear(4, d_model)
        self.input_bn = nn.BatchNorm1d(d_model)
        
        # Mamba层（带残差连接）
        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            for _ in range(n_layers)
        ])
        
        # 层归一化（每层一个）
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(n_layers)
        ])
        
        # Dropout（每层一个）
        self.dropouts = nn.ModuleList([
            nn.Dropout(dropout)
            for _ in range(n_layers)
        ])
        
        # 多层级特征投影（用于特征融合）
        if use_multi_scale:
            # 确定要提取特征的层索引（早期、中期、后期）
            scale_indices = [0, max(1, n_layers // 2), n_layers - 1]
            self.scale_indices = sorted(set(scale_indices))
            # 只为这些层创建投影
            self.multi_scale_projs = nn.ModuleDict({
                str(idx): nn.Linear(d_model, d_model // 2)
                for idx in self.scale_indices
            })
        else:
            self.use_multi_scale = False
            self.scale_indices = []
        
        # 可学习查询全局池化（替代简单平均池化）
        self.query_pooling = LearnableQueryPooling(d_model)
        
        # 物理直连路径（更丰富的统计特征）
        # 输入：均值、最大值、最小值、标准差（每个通道4个统计量 -> 4*4=16维）
        self.physics_stats_dim = 16  # 4通道 × 4统计量
        self.physics_direct_path = nn.Sequential(
            nn.Linear(self.physics_stats_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32)
        )
        
        # 特征融合（主特征 + 多尺度特征 + 物理直连）
        if use_multi_scale:
            # 主特征(d_model) + 多尺度特征(3 × d_model//2) + 物理直连(32)
            fusion_dim = d_model + len(self.scale_indices) * (d_model // 2) + 32
        else:
            # 主特征(d_model) + 物理直连(32)
            fusion_dim = d_model + 32
        
        self.fc_fusion = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 异方差输出头（均值 + 方差）
        self.heteroscedastic_head = HeteroscedasticHead(64, hidden_dim=64)
        
        # 辅助预测器（物理直连路径）
        self.auxiliary_predictor = nn.Linear(32, 1)
    
    def _preprocess_sequence(self, temp_sequence):
        """
        物理特征预处理（与MambaPhysicsModel相同）
        """
        batch_size, seq_len = temp_sequence.shape
        device = temp_sequence.device
        
        # 高斯滤波平滑 (sigma=2)
        temp_np = temp_sequence.cpu().numpy()
        T_smooth_np = np.array([gaussian_filter1d(temp_np[i], sigma=2.0) for i in range(batch_size)])
        T_smooth = torch.from_numpy(T_smooth_np).to(device).float()
        
        # Channel 0: T_tilde (去除初值)
        T_tilde = T_smooth - T_smooth[:, 0:1]
        
        # Channel 1: T_dot (数值微分)
        T_dot = torch.zeros_like(T_smooth)
        T_dot[:, :-1] = (T_smooth[:, 1:] - T_smooth[:, :-1]) / self.delta_t
        T_dot[:, -1] = T_dot[:, -2]
        
        # Channel 2: t^{-1/2}
        time_steps = torch.arange(0, seq_len, device=device, dtype=torch.float32) * self.delta_t
        time_steps[0] = self.delta_t
        t_inv_sqrt = 1.0 / torch.sqrt(time_steps.clamp_min(1e-12))
        t_inv_sqrt = t_inv_sqrt.unsqueeze(0).expand(batch_size, -1)
        
        # Channel 3: Delta_t归一化
        delta_t_normalized = torch.full_like(temp_sequence, self.delta_t / self.total_time)
        
        # Stack: [batch, seq_len, 4]
        features = torch.stack([T_tilde, T_dot, t_inv_sqrt, delta_t_normalized], dim=-1)
        return features
    
    def _extract_physics_stats(self, physics_features):
        """
        提取更丰富的物理统计特征
        
        Args:
            physics_features: [batch, seq_len, 4]
        Returns:
            stats: [batch, 16] 每个通道的均值、最大值、最小值、标准差
        """
        # 计算每个通道的统计量
        stats_list = []
        for ch in range(4):
            ch_data = physics_features[:, :, ch]  # [batch, seq_len]
            mean_val = ch_data.mean(dim=1, keepdim=True)  # [batch, 1]
            max_val = ch_data.max(dim=1, keepdim=True)[0]  # [batch, 1]
            min_val = ch_data.min(dim=1, keepdim=True)[0]  # [batch, 1]
            std_val = ch_data.std(dim=1, keepdim=True)  # [batch, 1]
            stats_list.extend([mean_val, max_val, min_val, std_val])
        
        # 拼接所有统计量: [batch, 16]
        stats = torch.cat(stats_list, dim=1)
        return stats
    
    def forward(self, temp_sequence, return_uncertainty=False, return_hidden=False):
        """
        Forward pass
        
        Args:
            temp_sequence: [batch, seq_len] 原始温度序列
            return_uncertainty: 是否返回不确定性（方差）
            return_hidden: 是否返回隐藏状态（用于分析）
        Returns:
            e_pred: [batch, 1] 预测的e值（均值）
            (可选) variance: [batch, 1] 预测方差
            (可选) hidden_states: 隐藏状态列表
        """
        # 1. 物理特征预处理
        physics_features = self._preprocess_sequence(temp_sequence)  # [batch, seq_len, 4]
        
        # 2. 物理直连路径（丰富统计特征）
        physics_stats = self._extract_physics_stats(physics_features)  # [batch, 16]
        physics_direct = self.physics_direct_path(physics_stats)  # [batch, 32]
        
        # 3. 投影到d_model
        x = self.input_projection(physics_features)  # [batch, seq_len, d_model]
        
        # 4. BatchNorm
        x = x.transpose(1, 2)
        x = self.input_bn(x)
        x = x.transpose(1, 2)
        
        # 5. Mamba层（带密集残差连接）
        hidden_states = []
        multi_scale_features = []
        
        for i, (mamba_layer, layer_norm, dropout) in enumerate(
            zip(self.mamba_layers, self.layer_norms, self.dropouts)
        ):
            # 残差连接
            x_residual = x
            
            # Mamba前向
            x = mamba_layer(x)  # [batch, seq_len, d_model]
            x = dropout(x)
            
            # 残差连接 + LayerNorm
            x = layer_norm(x + x_residual)
            
            # 多尺度特征提取
            if self.use_multi_scale and i in self.scale_indices:
                scale_feat = self.multi_scale_projs[str(i)](x)  # [batch, seq_len, d_model//2]
                # 全局池化多尺度特征
                scale_pooled = scale_feat.mean(dim=1)  # [batch, d_model//2]
                multi_scale_features.append(scale_pooled)
            
            if return_hidden:
                hidden_states.append(x.detach())
        
        # 6. 可学习查询全局池化（主特征）
        x_pooled = self.query_pooling(x)  # [batch, d_model]
        
        # 7. 特征融合
        if self.use_multi_scale and multi_scale_features:
            # 拼接主特征 + 多尺度特征 + 物理直连
            x_fused = torch.cat([x_pooled] + multi_scale_features + [physics_direct], dim=-1)
        else:
            # 拼接主特征 + 物理直连
            x_fused = torch.cat([x_pooled, physics_direct], dim=-1)
        
        # 8. 融合MLP
        x_hidden = self.fc_fusion(x_fused)  # [batch, 64]
        
        # 9. 异方差输出头（均值 + 方差）
        e_mean, e_logvar = self.heteroscedastic_head(x_hidden)  # [batch, 1] each
        
        # 10. 辅助预测路径（物理直连）
        e_pred_aux = self.auxiliary_predictor(physics_direct)  # [batch, 1]
        
        # 11. 残差组合（主预测 + 辅助预测）
        e_pred = e_mean + 0.1 * e_pred_aux
        
        # 返回结果
        result = [e_pred]
        
        if return_uncertainty:
            e_var = torch.exp(e_logvar)  # 转换为方差
            result.append(e_var)
        
        if return_hidden:
            result.append(hidden_states)
        
        if len(result) == 1:
            return result[0]
        return tuple(result)


class HybridMambaTransformer(nn.Module):
    """
    Hybrid architecture combining Mamba and Transformer
    
    - First layers: Mamba (fast temporal modeling, O(L))
    - Last layers: Transformer (global refinement, O(L²))
    
    Best of both worlds:
    - Mamba handles long-range dependencies efficiently
    - Transformer provides global context and refinement
    """
    
    def __init__(self, d_model=256, n_mamba=2, n_transformer=2, nhead=8,
                 d_state=16, d_conv=4, expand=2, dropout=0.2,
                 total_time=5.0, delta_t=0.02):
        """
        Args:
            d_model: Model dimension
            n_mamba: Number of Mamba layers
            n_transformer: Number of Transformer layers
            nhead: Number of attention heads
            d_state: SSM state dimension
            d_conv: Local convolution width
            expand: Expansion factor
            dropout: Dropout rate
            total_time: Total sampling time
            delta_t: Sampling interval
        """
        super().__init__()
        
        self.total_time = total_time
        self.delta_t = delta_t
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(4, d_model)
        self.input_bn = nn.BatchNorm1d(d_model)
        
        # Positional encoding (for Transformer layers)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Mamba layers (bottom layers)
        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_mamba)
        ])
        self.mamba_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_mamba)])
        
        # Transformer layers (top layers)
        if n_transformer > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer)
        else:
            self.transformer_encoder = None
        
        # Physics direct path
        self.physics_direct_path = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32)
        )
        
        # Output
        fusion_dim = d_model + 32
        self.fc = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def _preprocess_sequence(self, temp_sequence):
        """Same as MambaPhysicsModel"""
        batch_size, seq_len = temp_sequence.shape
        device = temp_sequence.device
        
        # 高斯滤波平滑 (sigma=2)
        temp_np = temp_sequence.cpu().numpy()
        T_smooth_np = np.array([gaussian_filter1d(temp_np[i], sigma=2.0) for i in range(batch_size)])
        T_smooth = torch.from_numpy(T_smooth_np).to(device).float()
        
        T_tilde = T_smooth - T_smooth[:, 0:1]
        
        # 数值微分
        T_dot = torch.zeros_like(T_smooth)
        T_dot[:, :-1] = (T_smooth[:, 1:] - T_smooth[:, :-1]) / self.delta_t
        T_dot[:, -1] = T_dot[:, -2]
        
        time_steps = torch.arange(0, seq_len, device=device, dtype=torch.float32) * self.delta_t
        time_steps[0] = self.delta_t
        t_inv_sqrt = 1.0 / torch.sqrt(time_steps.clamp_min(1e-12))
        t_inv_sqrt = t_inv_sqrt.unsqueeze(0).expand(batch_size, -1)
        
        delta_t_normalized = torch.full_like(temp_sequence, self.delta_t / self.total_time)
        
        features = torch.stack([T_tilde, T_dot, t_inv_sqrt, delta_t_normalized], dim=-1)
        return features
    
    def forward(self, temp_sequence):
        # Physics features
        physics_features = self._preprocess_sequence(temp_sequence)
        physics_stats = physics_features.mean(dim=1)
        physics_direct = self.physics_direct_path(physics_stats)
        
        # Project to d_model
        x = self.input_projection(physics_features)
        x = x.transpose(1, 2)
        x = self.input_bn(x)
        x = x.transpose(1, 2)
        
        # Mamba layers
        for mamba_layer, norm in zip(self.mamba_layers, self.mamba_norms):
            x_residual = x
            x = mamba_layer(x)
            x = norm(x + x_residual)
        
        # Transformer layers (if any)
        if self.transformer_encoder is not None:
            x = self.pos_encoder(x)
            x = self.transformer_encoder(x)
        
        # Global pooling
        x_pooled = x.mean(dim=1)
        
        # Fusion and output
        x_fused = torch.cat([x_pooled, physics_direct], dim=-1)
        e_pred = self.fc(x_fused)
        
        return e_pred


class MultiScalePhysicsTransformer(nn.Module):
    """
    多尺度物理信息Transformer (Multi-Scale Physics-Informed Transformer)
    
    核心创新点：
    1. 多尺度时间卷积：同时捕获短期（早期）、中期、长期（后期）特征
    2. 自适应物理特征权重：动态学习每个时间步物理特征的重要性
    3. 层次化特征融合：局部特征（Mamba） + 全局特征（Transformer）
    4. 双向时序建模：结合前向和后向信息
    5. 密集残差连接：增强梯度流动
    
    适用于：
    - 热扩散过程的不同阶段（快速升温 → 稳定扩散 → 渐近平衡）
    - 样本分布不均（高e值样本少）
    - 噪声敏感的场景
    """
    
    def __init__(self, d_model=256, n_mamba=3, n_transformer=3, nhead=8,
                 d_state=32, d_conv=4, expand=2, dropout=0.2,
                 total_time=5.0, delta_t=0.02, use_bidirectional=True):
        """
        Args:
            d_model: 模型维度
            n_mamba: Mamba层数（底层，快速局部建模）
            n_transformer: Transformer层数（顶层，全局精细建模）
            nhead: 注意力头数
            d_state: SSM状态维度（增大以增强记忆能力）
            d_conv: 局部卷积宽度
            expand: 扩展因子
            dropout: Dropout率
            total_time: 总采样时间
            delta_t: 采样间隔
            use_bidirectional: 是否使用双向时序建模
        """
        super().__init__()
        
        self.total_time = total_time
        self.delta_t = delta_t
        self.d_model = d_model
        self.use_bidirectional = use_bidirectional
        
        # ========== 多尺度时间卷积 ==========
        # 捕获不同时间尺度的特征（早期/中期/后期）
        self.multiscale_convs = nn.ModuleList([
            # 短期特征（早期快速变化）：小卷积核，捕获高频细节
            nn.Conv1d(4, d_model // 4, kernel_size=3, padding=1),
            # 中期特征（稳定扩散）：中等卷积核
            nn.Conv1d(4, d_model // 4, kernel_size=7, padding=3),
            # 长期特征（渐近平衡）：大卷积核，捕获趋势
            nn.Conv1d(4, d_model // 2, kernel_size=15, padding=7),
        ])
        self.multiscale_bn = nn.ModuleList([
            nn.BatchNorm1d(d_model // 4) for _ in range(2)
        ] + [nn.BatchNorm1d(d_model // 2)])
        
        # ========== 自适应物理特征权重 ==========
        # 学习每个时间步的物理特征重要性（早期可能T_dot更重要，后期可能t^{-1/2}更重要）
        self.physics_attention = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 4),  # 输出4个物理通道的权重
            nn.Softmax(dim=-1)
        )
        
        # ========== 输入投影 ==========
        # 多尺度特征融合后的投影
        self.input_projection = nn.Linear(d_model, d_model)
        self.input_bn = nn.BatchNorm1d(d_model)
        
        # ========== 位置编码 ==========
        self.pos_encoder = PositionalEncoding(d_model)
        
        # ========== Mamba层（底层：局部时序建模） ==========
        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_mamba)
        ])
        self.mamba_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_mamba)])
        self.mamba_dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_mamba)])
        
        # ========== Transformer层（顶层：全局精细建模） ==========
        if n_transformer > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer)
        else:
            self.transformer_encoder = None
        
        # ========== 双向时序建模 ==========
        if use_bidirectional:
            self.bidirectional_projection = nn.Linear(d_model * 2, d_model)
        else:
            self.bidirectional_projection = None
        
        # ========== 层次化池化 ==========
        # 局部池化（窗口池化）
        self.local_pool = nn.AdaptiveAvgPool1d(8)  # 压缩到8个时间步
        self.local_pool_proj = nn.Linear(d_model, d_model // 2)
        
        # 全局池化（全部时间步）
        self.global_pool_proj = nn.Linear(d_model, d_model // 2)
        
        # ========== 物理直连路径（跳跃连接） ==========
        self.physics_direct_path = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32)
        )
        
        # ========== 最终融合和输出 ==========
        # 融合：局部特征 + 全局特征 + 物理直连
        fusion_dim = d_model // 2 + d_model // 2 + 32  # local + global + physics
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 输出层（主路径）
        self.fc_out = nn.Linear(64, 1)
        
        # 辅助预测器（物理直连路径）
        self.aux_predictor = nn.Linear(32, 1)
    
    def _preprocess_sequence(self, temp_sequence):
        """
        多尺度物理特征预处理
        
        Args:
            temp_sequence: [batch, seq_len]
        Returns:
            multiscale_features: [batch, seq_len, d_model] 多尺度融合特征
            physics_features: [batch, seq_len, 4] 原始物理特征
            physics_weights: [batch, seq_len, 4] 自适应权重
        """
        batch_size, seq_len = temp_sequence.shape
        device = temp_sequence.device
        
        # 高斯滤波平滑
        temp_np = temp_sequence.cpu().numpy()
        T_smooth_np = np.array([gaussian_filter1d(temp_np[i], sigma=2.0) for i in range(batch_size)])
        T_smooth = torch.from_numpy(T_smooth_np).to(device).float()
        
        # 构建4通道物理特征
        T_tilde = T_smooth - T_smooth[:, 0:1]
        
        T_dot = torch.zeros_like(T_smooth)
        T_dot[:, :-1] = (T_smooth[:, 1:] - T_smooth[:, :-1]) / self.delta_t
        T_dot[:, -1] = T_dot[:, -2]
        
        time_steps = torch.arange(0, seq_len, device=device, dtype=torch.float32) * self.delta_t
        time_steps[0] = self.delta_t
        t_inv_sqrt = 1.0 / torch.sqrt(time_steps.clamp_min(1e-12))
        t_inv_sqrt = t_inv_sqrt.unsqueeze(0).expand(batch_size, -1)
        
        delta_t_normalized = torch.full_like(temp_sequence, self.delta_t / self.total_time)
        
        physics_features = torch.stack([T_tilde, T_dot, t_inv_sqrt, delta_t_normalized], dim=-1)  # [batch, seq_len, 4]
        
        # ========== 自适应物理特征权重 ==========
        # 学习每个时间步的物理通道重要性
        physics_weights = self.physics_attention(physics_features)  # [batch, seq_len, 4]
        # 加权后的物理特征
        physics_weighted = physics_features * physics_weights  # [batch, seq_len, 4]
        
        # ========== 多尺度时间卷积 ==========
        # 转换为卷积格式：[batch, 4, seq_len]
        physics_conv = physics_weighted.transpose(1, 2)  # [batch, 4, seq_len]
        
        multiscale_outputs = []
        for i, (conv, bn) in enumerate(zip(self.multiscale_convs, self.multiscale_bn)):
            feat = conv(physics_conv)  # [batch, d_model//4 or d_model//2, seq_len]
            feat = bn(feat)
            feat = F.relu(feat)
            multiscale_outputs.append(feat)
        
        # 拼接多尺度特征：[batch, d_model, seq_len]
        multiscale_concat = torch.cat(multiscale_outputs, dim=1)  # [batch, d_model, seq_len]
        
        # 转回序列格式：[batch, seq_len, d_model]
        multiscale_features = multiscale_concat.transpose(1, 2)
        
        return multiscale_features, physics_features, physics_weights
    
    def forward(self, temp_sequence, return_weights=False):
        """
        Forward pass
        
        Args:
            temp_sequence: [batch, seq_len]
            return_weights: 是否返回物理特征权重（用于可视化）
        Returns:
            e_pred: [batch, 1]
            (optional) physics_weights: [batch, seq_len, 4]
        """
        # 1. 多尺度物理特征预处理
        x, physics_features, physics_weights = self._preprocess_sequence(temp_sequence)
        # x: [batch, seq_len, d_model]
        
        # 2. 输入投影和归一化
        x = self.input_projection(x)  # [batch, seq_len, d_model]
        x = x.transpose(1, 2)  # [batch, d_model, seq_len]
        x = self.input_bn(x)
        x = x.transpose(1, 2)  # [batch, seq_len, d_model]
        
        # 3. 位置编码
        x = self.pos_encoder(x)
        
        # 4. Mamba层（局部时序建模，带残差连接）
        x_residual = x
        for mamba_layer, norm, dropout in zip(self.mamba_layers, self.mamba_norms, self.mamba_dropouts):
            x_mamba = mamba_layer(x)
            x_mamba = dropout(x_mamba)
            x = norm(x + x_mamba + x_residual)  # 密集残差连接
            x_residual = x
        
        # 5. 双向时序建模（可选）
        if self.use_bidirectional and self.bidirectional_projection is not None:
            # 前向序列
            x_forward = x
            # 后向序列（反转）
            x_backward = torch.flip(x, dims=[1])
            # 拼接
            x_bidirectional = torch.cat([x_forward, x_backward], dim=-1)  # [batch, seq_len, d_model*2]
            x = self.bidirectional_projection(x_bidirectional)  # [batch, seq_len, d_model]
        
        # 6. Transformer层（全局精细建模）
        if self.transformer_encoder is not None:
            x = self.transformer_encoder(x)  # [batch, seq_len, d_model]
        
        # 7. 层次化池化
        # 局部池化（窗口特征）
        x_local = x.transpose(1, 2)  # [batch, d_model, seq_len]
        x_local = self.local_pool(x_local)  # [batch, d_model, 8]
        x_local = x_local.transpose(1, 2)  # [batch, 8, d_model]
        x_local = x_local.mean(dim=1)  # [batch, d_model]
        x_local = self.local_pool_proj(x_local)  # [batch, d_model//2]
        
        # 全局池化（全部时间步）
        x_global = x.mean(dim=1)  # [batch, d_model]
        x_global = self.global_pool_proj(x_global)  # [batch, d_model//2]
        
        # 8. 物理直连路径（跳跃连接）
        physics_stats = physics_features.mean(dim=1)  # [batch, 4]
        physics_direct = self.physics_direct_path(physics_stats)  # [batch, 32]
        
        # 9. 特征融合
        x_fused = torch.cat([x_local, x_global, physics_direct], dim=-1)  # [batch, d_model//2 + d_model//2 + 32]
        x_fused = self.fusion(x_fused)  # [batch, 64]
        
        # 10. 主预测路径
        e_pred_main = self.fc_out(x_fused)  # [batch, 1]
        
        # 11. 辅助预测路径（物理直连）+ 残差连接
        e_pred_aux = self.aux_predictor(physics_direct)  # [batch, 1]
        
        # 残差组合
        e_pred = e_pred_main + 0.15 * e_pred_aux
        
        if return_weights:
            return e_pred, physics_weights
        return e_pred


class DilatedTCNBlock(nn.Module):
    """
    Dilated Temporal Convolutional Network Block
    核心组件：Depthwise Dilated Conv + Pointwise Conv + GLU/SiLU + Residual + LayerNorm
    """
    def __init__(self, channels, dilation, kernel_size=7, dropout=0.2, activation='glu'):
        """
        Args:
            channels: 通道数
            dilation: 空洞率
            kernel_size: 卷积核大小
            dropout: Dropout率
            activation: 'glu' 或 'silu'
        """
        super().__init__()
        
        # Padding确保输出长度不变
        padding = ((kernel_size - 1) * dilation) // 2
        
        # Depthwise dilated convolution
        self.depthwise_conv = nn.Conv1d(
            channels, channels, 
            kernel_size=kernel_size, 
            dilation=dilation, 
            padding=padding,
            groups=channels  # Depthwise: 每个输入通道独立卷积
        )
        
        # Pointwise 1x1 convolution (通道混合)
        self.pointwise_conv = nn.Conv1d(channels, channels, kernel_size=1)
        
        # 激活函数
        if activation == 'glu':
            # GLU: Gated Linear Unit (更高效的激活)
            self.glu = nn.GLU(dim=1)
            # GLU需要输入通道数是2的倍数
            self.pointwise_conv_gate = nn.Conv1d(channels, channels * 2, kernel_size=1)
        elif activation == 'silu':
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        self.activation_type = activation
        
        # Normalization & Dropout
        self.layer_norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: [batch, channels, seq_len]
        Returns:
            out: [batch, channels, seq_len]
        """
        residual = x
        
        # Depthwise dilated convolution
        x = self.depthwise_conv(x)  # [batch, channels, seq_len]
        
        # Pointwise convolution
        if self.activation_type == 'glu':
            x = self.pointwise_conv_gate(x)  # [batch, channels*2, seq_len]
            x = self.glu(x)  # [batch, channels, seq_len]
        else:
            x = self.pointwise_conv(x)  # [batch, channels, seq_len]
            x = self.activation(x)
        
        # Dropout
        x = self.dropout(x)
        
        # Residual connection + LayerNorm
        # 需要转置进行LayerNorm（LayerNorm在最后一个维度）
        x = x.transpose(1, 2)  # [batch, seq_len, channels]
        x = self.layer_norm(x + residual.transpose(1, 2))
        x = x.transpose(1, 2)  # [batch, channels, seq_len]
        
        return x


class LearnableQueryPooling(nn.Module):
    """
    可学习查询的全局池化（替代简单均值池化）
    使用可学习的查询向量来加权聚合时序特征
    """
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
        # 可学习的查询向量
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        
        # 缩放因子
        self.scale = math.sqrt(d_model)
        
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model] 或 [batch, d_model, seq_len]
        Returns:
            pooled: [batch, d_model] 加权聚合后的特征
        """
        # 处理两种输入格式
        if x.dim() == 3 and x.size(-1) == self.d_model:
            # [batch, seq_len, d_model]
            pass
        elif x.dim() == 3 and x.size(1) == self.d_model:
            # [batch, d_model, seq_len] -> [batch, seq_len, d_model]
            x = x.transpose(1, 2)
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        batch_size, seq_len, d_model = x.shape
        
        # 计算注意力权重
        # [1, 1, d_model] @ [batch, d_model, seq_len] -> [batch, 1, seq_len]
        scores = torch.matmul(self.query, x.transpose(1, 2)) / self.scale
        attn_weights = torch.softmax(scores, dim=-1)  # [batch, 1, seq_len]
        
        # 加权求和
        # [batch, 1, seq_len] @ [batch, seq_len, d_model] -> [batch, 1, d_model]
        pooled = torch.matmul(attn_weights, x).squeeze(1)  # [batch, d_model]
        
        return pooled


class HeteroscedasticHead(nn.Module):
    """
    异方差输出头
    同时预测均值和方差（不确定性估计）
    """
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        
        # 均值预测
        self.mean_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 方差预测（log方差，确保为正）
        self.logvar_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        """
        Args:
            x: [batch, input_dim]
        Returns:
            mean: [batch, 1] 预测均值
            logvar: [batch, 1] 预测的对数方差
        """
        mean = self.mean_head(x)
        logvar = self.logvar_head(x)
        
        return mean, logvar


class PhysTCN(nn.Module):
    """
    Physics-Informed Temporal Convolutional Network
    
    核心特点：
    - Depthwise Dilated TCN堆叠（O(L)复杂度，多感受野）
    - 可学习查询全局池化（优于简单均值池化）
    - 物理直连路径（统计特征跳跃连接）
    - 异方差输出头（均值+不确定性估计）
    
    优势：
    - 比Transformer快（O(L) vs O(L²)）
    - 对噪声更鲁棒（不依赖大注意力矩阵）
    - 更适合热扩散的平滑-单调特性
    """
    
    def __init__(self, channels=128, num_layers=6, dilations=None, kernel_size=7, 
                 dropout=0.2, activation='glu', total_time=5.0, delta_t=0.02):
        """
        Args:
            channels: TCN通道数（推荐128）
            num_layers: TCN层数（推荐6）
            dilations: 空洞率序列（默认[1,2,4,8,16,32]）
            kernel_size: 卷积核大小（推荐7）
            dropout: Dropout率（推荐0.1）
            activation: 激活函数类型 'glu' 或 'silu'（推荐'glu'）
            total_time: 总采样时间
            delta_t: 采样间隔
        """
        super().__init__()
        
        self.total_time = total_time
        self.delta_t = delta_t
        self.channels = channels
        
        # 默认空洞率序列
        if dilations is None:
            dilations = [2 ** i for i in range(num_layers)]  # [1,2,4,8,16,32] for 6 layers
        else:
            num_layers = len(dilations)
        
        # 4通道物理特征 -> TCN通道数
        self.input_projection = nn.Conv1d(4, channels, kernel_size=1)
        self.input_bn = nn.BatchNorm1d(channels)
        
        # Dilated TCN层堆叠
        self.tcn_layers = nn.ModuleList([
            DilatedTCNBlock(channels, dilation=d, kernel_size=kernel_size, 
                          dropout=dropout, activation=activation)
            for d in dilations
        ])
        
        # 可学习查询全局池化
        self.query_pooling = LearnableQueryPooling(channels)
        
        # 物理直连路径（统计特征）
        self.physics_direct_path = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32)
        )
        
        # 特征融合（TCN特征 + 物理直连特征）
        fusion_dim = channels + 32
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 异方差输出头
        self.heteroscedastic_head = HeteroscedasticHead(64, hidden_dim=64)
        
    def _preprocess_sequence(self, temp_sequence):
        """
        物理特征预处理（复用现有实现）
        
        Args:
            temp_sequence: [batch, seq_len]
        Returns:
            features: [batch, 4, seq_len] 物理特征
        """
        batch_size, seq_len = temp_sequence.shape
        device = temp_sequence.device
        
        # 高斯滤波平滑 (sigma=2)
        temp_np = temp_sequence.cpu().numpy()
        T_smooth_np = np.array([gaussian_filter1d(temp_np[i], sigma=2.0) for i in range(batch_size)])
        T_smooth = torch.from_numpy(T_smooth_np).to(device).float()
        
        # Channel 0: T_tilde (去除初值)
        T_tilde = T_smooth - T_smooth[:, 0:1]  # [batch, seq_len]
        
        # Channel 1: T_dot (数值微分)
        T_dot = torch.zeros_like(T_smooth)
        T_dot[:, :-1] = (T_smooth[:, 1:] - T_smooth[:, :-1]) / self.delta_t
        T_dot[:, -1] = T_dot[:, -2]
        
        # Channel 2: t^{-1/2}
        time_steps = torch.arange(0, seq_len, device=device, dtype=torch.float32) * self.delta_t
        time_steps[0] = self.delta_t
        t_inv_sqrt = 1.0 / torch.sqrt(time_steps.clamp_min(1e-12))
        t_inv_sqrt = t_inv_sqrt.unsqueeze(0).expand(batch_size, -1)
        
        # Channel 3: Delta_t 归一化
        delta_t_normalized = torch.full_like(temp_sequence, self.delta_t / self.total_time)
        
        # Stack: [batch, 4, seq_len]
        features = torch.stack([T_tilde, T_dot, t_inv_sqrt, delta_t_normalized], dim=1)
        return features
    
    def _extract_physics_stats(self, physics_features):
        """
        提取物理特征统计信息
        
        Args:
            physics_features: [batch, 4, seq_len]
        Returns:
            stats: [batch, 4] 每个通道的均值
        """
        return physics_features.mean(dim=2)  # [batch, 4]
    
    def forward(self, temp_sequence, return_uncertainty=False):
        """
        Forward pass
        
        Args:
            temp_sequence: [batch, seq_len] 原始温度序列
            return_uncertainty: 是否返回不确定性（方差）
        Returns:
            e_pred: [batch, 1] 预测的e值（均值）
            (可选) variance: [batch, 1] 预测方差（如果return_uncertainty=True）
        """
        # 1. 物理特征预处理
        physics_features = self._preprocess_sequence(temp_sequence)  # [batch, 4, seq_len]
        
        # 2. 物理直连路径（统计特征）
        physics_stats = self._extract_physics_stats(physics_features)  # [batch, 4]
        physics_direct = self.physics_direct_path(physics_stats)  # [batch, 32]
        
        # 3. 输入投影
        x = self.input_projection(physics_features)  # [batch, channels, seq_len]
        x = self.input_bn(x)
        
        # 4. Dilated TCN层堆叠
        for tcn_layer in self.tcn_layers:
            x = tcn_layer(x)  # [batch, channels, seq_len]
        
        # 5. 可学习查询全局池化
        x_pooled = self.query_pooling(x)  # [batch, channels]
        
        # 6. 特征融合
        x_fused = torch.cat([x_pooled, physics_direct], dim=-1)  # [batch, channels + 32]
        x_fused = self.fusion_mlp(x_fused)  # [batch, 64]
        
        # 7. 异方差输出头
        e_mean, e_logvar = self.heteroscedastic_head(x_fused)  # [batch, 1] each
        
        if return_uncertainty:
            e_var = torch.exp(e_logvar)  # 转换为方差
            return e_mean, e_var
        
        return e_mean


# ==================== Mamba-based VAE Model ====================

class PreNormResidual(nn.Module):
    """
    Pre-Norm残差块
    先归一化，再执行子模块，最后加上残差连接
    """
    def __init__(self, d_model, block):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.block = block
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            out: [batch, seq_len, d_model]
        """
        return x + self.block(self.norm(x))


class MambaEncoder(nn.Module):
    """
    基于Mamba的编码器
    输入: [batch, C_in, seq_len]，先投影到 [batch, seq_len, d_model]，再通过Mamba堆叠
    """
    def __init__(self, C_in=1, d_model=128, depth=4, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.proj_in = nn.Conv1d(C_in, d_model, kernel_size=1)  # [batch, d_model, seq_len]
        blocks = []
        for _ in range(depth):
            blocks.append(PreNormResidual(
                d_model,
                Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            ))
        self.blocks = nn.ModuleList(blocks)
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: [batch, C_in, seq_len]
        Returns:
            h: [batch, seq_len, d_model] token表示
            h_global: [batch, d_model] 全局表示
        """
        h = self.proj_in(x).transpose(1, 2)   # -> [batch, seq_len, d_model]
        for blk in self.blocks:
            h = blk(h)                         # Mamba堆叠
        h = self.norm_out(h)                   # [batch, seq_len, d_model]
        # 全局时间池化，兼容变长序列（掩码可选）
        h_global = h.mean(dim=1)               # [batch, d_model]
        return h, h_global


class TimeVAE1D_Mamba(nn.Module):
    """
    基于Mamba的时间序列VAE (Variational Autoencoder)
    用于单像素/一维时间序列的重建和参数预测
    
    优势：
    - 使用Mamba作为编码器，O(L)复杂度，高效处理长序列
    - 变分推断学习紧凑的隐空间表示
    - 同时支持序列重建和参数e预测
    - 适合无监督/半监督学习场景
    
    输入: [batch, C_in, seq_len]
    输出: 重建 [batch, 1, seq_len]、参数e预测 [batch, 1]
    """
    def __init__(self, seq_len=200, C_in=1, latent_dim=64,
                 d_model=128, depth=4, d_state=16, d_conv=4, expand=2,
                 decoder_base=64):
        """
        Args:
            seq_len: 输入序列长度
            C_in: 输入通道数（默认1，原始温度序列）
            latent_dim: 隐空间维度
            d_model: Mamba模型维度
            depth: Mamba层数
            d_state: SSM状态维度
            d_conv: 局部卷积宽度
            expand: 扩展因子
            decoder_base: 解码器基础通道数
        """
        super().__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim

        # ------ Encoder: Mamba ------
        self.encoder = MambaEncoder(C_in=C_in, d_model=d_model, depth=depth,
                                    d_state=d_state, d_conv=d_conv, expand=expand)
        self.fc_mu     = nn.Linear(d_model, latent_dim)
        self.fc_logvar = nn.Linear(d_model, latent_dim)

        # ------ Decoder: 使用自适应上采样回到seq_len ------
        # 避免反卷积的长度计算问题，使用上采样+卷积
        self.fc_dec = nn.Linear(latent_dim, decoder_base * 64)  # 固定起始长度64
        
        self.decoder = nn.Sequential(
            # [batch, decoder_base, 64]
            nn.Conv1d(decoder_base, decoder_base*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),  # -> 128
            
            nn.Conv1d(decoder_base*2, decoder_base, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(size=seq_len, mode='linear', align_corners=False),  # -> seq_len (精确)
            
            nn.Conv1d(decoder_base, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=3, padding=1)  # -> [batch, 1, seq_len]
        )
        

        # ------ e参数预测头（可选正值约束Softplus）------
        self.param_head = nn.Sequential(
            nn.Linear(latent_dim, 128), 
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        


    def encode(self, x):
        """
        编码器：输入序列 -> 隐空间均值和方差
        
        Args:
            x: [batch, C_in, seq_len]
        Returns:
            mu: [batch, latent_dim] 隐变量均值
            logvar: [batch, latent_dim] 隐变量对数方差
        """
        _, h_global = self.encoder(x)        # [batch, d_model]
        mu, logvar = self.fc_mu(h_global), self.fc_logvar(h_global)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        重参数化技巧：使梯度能够反向传播
        
        Args:
            mu: [batch, latent_dim] 均值
            logvar: [batch, latent_dim] 对数方差
        Returns:
            z: [batch, latent_dim] 采样的隐变量
        """
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        """
        解码器：隐变量 -> 重建序列
        
        Args:
            z: [batch, latent_dim]
        Returns:
            xhat: [batch, 1, seq_len] 重建的序列
        """
        B = z.size(0)
        decoder_base = 64  # 与 __init__ 中的 decoder_base 参数一致
        
        # 线性投影到固定长度64
        h = self.fc_dec(z).view(B, decoder_base, 64)  # [batch, decoder_base, 64]
        
        # 通过解码器上采样到目标长度
        xhat = self.decoder(h)  # [batch, 1, seq_len]
        return xhat

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: [batch, C_in, seq_len]
        Returns:
            recon: [batch, 1, seq_len] 重建序列
            e_pred: [batch, 1] 预测的e参数
            (mu, logvar): 隐变量分布参数（用于计算KL散度）
        """
        assert x.dim() == 3 and x.size(2) == self.seq_len, \
            f"输入应为 [batch, C_in, seq_len]，且seq_len={self.seq_len}，实际得到 {x.shape}"
        
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        e_pred = self.param_head(z)
        # 可选：强制e为正数



        # e_pred = self.softplus(self.param_head(z)) + 1e-6
        
        return recon, e_pred, (mu, logvar)


class TimeVAE1D_Mamba_PhysicsDecoder(nn.Module):
    """
    基于Mamba的时间序列VAE，使用"物理核 + 残差"解码器
    
    解码器架构：
    x^(t) = Σ_k w_k * g_k(t) + Conv1D-Residual(t)
    
    其中：
    - g_k(t) 是预定义的物理基函数（t^{-1/2}, exp(-t/τ), 等）
    - w_k 是从隐变量学习的权重系数
    - Conv1D-Residual 是一个小型卷积网络，学习细节修正
    
    优势：
    - 使用Mamba作为编码器，O(L)复杂度
    - 物理先验指导重建，更符合热扩散规律
    - 残差网络学习细节，兼顾物理正确性和灵活性
    - 参数更少，训练更稳定
    
    输入: [batch, C_in, seq_len]
    输出: 重建 [batch, 1, seq_len]、参数e预测 [batch, 1]
    """
    def __init__(self, seq_len=200, C_in=1, latent_dim=64,
                 d_model=128, depth=4, d_state=16, d_conv=4, expand=2,
                 delta_t=0.02, total_time=None, num_physics_basis=8):
        """
        Args:
            seq_len: 输入序列长度
            C_in: 输入通道数（默认1，原始温度序列）
            latent_dim: 隐空间维度
            d_model: Mamba模型维度
            depth: Mamba层数
            d_state: SSM状态维度
            d_conv: 局部卷积宽度
            expand: 扩展因子
            delta_t: 采样间隔（秒），用于构建物理基函数
            total_time: 总采样时间（秒），如果为None则从seq_len*delta_t推导
            num_physics_basis: 物理基函数数量
        """
        super().__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.delta_t = delta_t
        if total_time is None:
            self.total_time = seq_len * delta_t
        else:
            self.total_time = total_time
        self.num_physics_basis = num_physics_basis

        # ------ Encoder: Mamba (与原模型相同) ------
        self.encoder = MambaEncoder(C_in=C_in, d_model=d_model, depth=depth,
                                    d_state=d_state, d_conv=d_conv, expand=expand)
        self.fc_mu     = nn.Linear(d_model, latent_dim)
        self.fc_logvar = nn.Linear(d_model, latent_dim)

        # ------ Decoder: 物理核 + 残差解码器 ------
        # 从隐变量学习物理核权重
        self.physics_weight_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_physics_basis)  # 输出每个基函数的权重
        )
        
        # Conv1D残差网络（学习细节修正）
        self.residual_decoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=5, padding=2)  # 残差输出
        )
        
        # 注册物理基函数（缓存，在第一次forward时构建）
        self.register_buffer('_physics_basis', None)

        # ------ e参数预测头 ------
        self.param_head = nn.Sequential(
            nn.Linear(latent_dim, 128), 
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def _build_physics_basis(self, seq_len, delta_t, device):
        """
        构建物理基函数 g_k(t)
        
        基函数包括：
        1. t^{-1/2} (半无限体扩散，最常用的物理先验)
        2. exp(-t/τ_k) (多个时间常数的指数衰减)
        3. t^β (幂律形式，补充不同衰减模式)
        
        Args:
            seq_len: 序列长度
            delta_t: 采样间隔
            device: 设备
        
        Returns:
            basis: [num_physics_basis, seq_len] 物理基函数（已归一化）
        """
        # 构建时间序列 t = [delta_t, 2*delta_t, ..., seq_len*delta_t]
        time_steps = torch.arange(1, seq_len + 1, device=device, dtype=torch.float32) * delta_t
        
        basis_list = []
        
        # 1. t^{-1/2} (半无限体扩散，最常用的物理先验)
        basis_list.append(1.0 / torch.sqrt(time_steps.clamp_min(1e-12)))
        
        # 2. exp(-t/τ_k) 多个时间常数
        tau_values = [0.1, 0.5, 1.0, 2.0, 5.0]  # 不同的时间常数
        for tau in tau_values[:min(len(tau_values), self.num_physics_basis - 1)]:
            basis_list.append(torch.exp(-time_steps / tau))
        
        # 3. 如果需要更多基函数，添加 t^β 形式（幂律衰减）
        while len(basis_list) < self.num_physics_basis:
            beta = 0.5 + (len(basis_list) - 5) * 0.3  # 不同的幂指数
            basis_list.append(time_steps.pow(beta))
        
        # 堆叠为 [num_physics_basis, seq_len]
        basis = torch.stack(basis_list[:self.num_physics_basis], dim=0)
        
        # 归一化每个基函数（使其L2范数为1，便于学习权重）
        basis = basis / (basis.norm(dim=1, keepdim=True) + 1e-8)
        
        return basis
    
    def encode(self, x):
        """
        编码器：输入序列 -> 隐空间均值和方差
        
        Args:
            x: [batch, C_in, seq_len]
        Returns:
            mu: [batch, latent_dim] 隐变量均值
            logvar: [batch, latent_dim] 隐变量对数方差
        """
        _, h_global = self.encoder(x)        # [batch, d_model]
        mu, logvar = self.fc_mu(h_global), self.fc_logvar(h_global)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        重参数化技巧：使梯度能够反向传播
        
        Args:
            mu: [batch, latent_dim] 均值
            logvar: [batch, latent_dim] 对数方差
        Returns:
            z: [batch, latent_dim] 采样的隐变量
        """
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z, delta_t=None):
        """
        解码器：隐变量 -> 重建序列（物理核 + 残差）
        
        x^(t) = Σ_k w_k * g_k(t) + Conv1D-Residual(t)
        
        Args:
            z: [batch, latent_dim] 隐变量
            delta_t: 采样间隔（可选，使用默认值如果为None）
        Returns:
            xhat: [batch, 1, seq_len] 重建的序列
        """
        device = z.device
        
        # 使用传入的delta_t或默认值
        if delta_t is None:
            delta_t = self.delta_t
        
        # 1. 构建物理基函数（如果未缓存或长度不匹配则重新构建）
        if (self._physics_basis is None or 
            self._physics_basis.shape[1] != self.seq_len or
            self._physics_basis.device != device):
            self._physics_basis = self._build_physics_basis(
                self.seq_len, delta_t, device
            )  # [num_physics_basis, seq_len]
        
        # 2. 从隐变量学习物理核权重 w_k
        weights = self.physics_weight_head(z)  # [batch, num_physics_basis]
        
        # 3. 线性组合物理基函数：Σ_k w_k * g_k(t)
        # [batch, num_physics_basis] @ [num_physics_basis, seq_len] -> [batch, seq_len]
        physics_reconstruction = torch.matmul(weights, self._physics_basis)  # [batch, seq_len]
        physics_reconstruction = physics_reconstruction.unsqueeze(1)  # [batch, 1, seq_len]
        
        # 4. Conv1D残差网络：学习细节修正
        # 使用物理重建作为输入，学习残差
        residual = self.residual_decoder(physics_reconstruction)  # [batch, 1, seq_len]
        
        # 5. 组合：物理重建 + 残差
        xhat = physics_reconstruction + residual  # [batch, 1, seq_len]
        
        return xhat

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: [batch, C_in, seq_len]
        Returns:
            recon: [batch, 1, seq_len] 重建序列
            e_pred: [batch, 1] 预测的e参数
            (mu, logvar): 隐变量分布参数（用于计算KL散度）
        """
        assert x.dim() == 3 and x.size(2) == self.seq_len, \
            f"输入应为 [batch, C_in, seq_len]，且seq_len={self.seq_len}，实际得到 {x.shape}"
        
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        e_pred = self.param_head(z)
        
        return recon, e_pred, (mu, logvar)


class TransformerEncoder(nn.Module):
    """
    基于Transformer的编码器
    输入: [batch, C_in, seq_len]，先投影到 [batch, seq_len, d_model]，再通过Transformer堆叠
    """
    def __init__(self, C_in=1, d_model=128, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.2):
        super().__init__()
        self.d_model = d_model
        
        # 输入投影: [batch, C_in, seq_len] -> [batch, seq_len, d_model]
        self.input_projection = nn.Linear(C_in, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出归一化
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: [batch, C_in, seq_len]
        Returns:
            h: [batch, seq_len, d_model] token表示
            h_global: [batch, d_model] 全局表示
        """
        # [batch, C_in, seq_len] -> [batch, seq_len, C_in]
        x = x.transpose(1, 2)
        
        # 投影到d_model维度
        h = self.input_projection(x)  # [batch, seq_len, d_model]
        
        # 位置编码
        h = self.pos_encoder(h)
        
        # Transformer编码
        h = self.transformer_encoder(h)  # [batch, seq_len, d_model]
        
        # 归一化
        h = self.norm_out(h)
        
        # 全局时间池化
        h_global = h.mean(dim=1)  # [batch, d_model]
        
        return h, h_global


class TimeVAE1D_Transformer(nn.Module):
    """
    基于Transformer的时间序列VAE (Variational Autoencoder)
    用于单像素/一维时间序列的重建和参数预测
    
    架构：
    - Encoder: Transformer (全局注意力，O(L²)复杂度)
    - Decoder: Conv1D (上采样卷积)
    
    优势：
    - Transformer编码器捕获全局依赖关系
    - 自注意力机制能够关注序列中任意位置
    - 适合需要全局上下文的场景
    - 卷积解码器高效生成序列
    
    输入: [batch, C_in, seq_len]
    输出: 重建 [batch, 1, seq_len]、参数e预测 [batch, 1]
    """
    def __init__(self, seq_len=200, C_in=1, latent_dim=64,
                 d_model=128, nhead=8, num_layers=4, dim_feedforward=512, 
                 dropout=0.2, decoder_base=64):
        """
        Args:
            seq_len: 输入序列长度
            C_in: 输入通道数（默认1，原始温度序列）
            latent_dim: 隐空间维度
            d_model: Transformer模型维度
            nhead: 注意力头数
            num_layers: Transformer层数
            dim_feedforward: 前馈网络维度
            dropout: Dropout率
            decoder_base: 解码器基础通道数
        """
        super().__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim

        # ------ Encoder: Transformer ------
        self.encoder = TransformerEncoder(
            C_in=C_in, 
            d_model=d_model, 
            nhead=nhead, 
            num_layers=num_layers,
            dim_feedforward=dim_feedforward, 
            dropout=dropout
        )
        self.fc_mu     = nn.Linear(d_model, latent_dim)
        self.fc_logvar = nn.Linear(d_model, latent_dim)

        # ------ Decoder: 使用自适应上采样回到seq_len (与Mamba VAE相同) ------
        self.fc_dec = nn.Linear(latent_dim, decoder_base * 64)  # 固定起始长度64
        
        self.decoder = nn.Sequential(
            # [batch, decoder_base, 64]
            nn.Conv1d(decoder_base, decoder_base*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),  # -> 128
            
            nn.Conv1d(decoder_base*2, decoder_base, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(size=seq_len, mode='linear', align_corners=False),  # -> seq_len (精确)
            
            nn.Conv1d(decoder_base, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=3, padding=1)  # -> [batch, 1, seq_len]
        )

        # ------ e参数预测头 ------
        self.param_head = nn.Sequential(
            nn.Linear(latent_dim, 128), 
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def encode(self, x):
        """
        编码器：输入序列 -> 隐空间均值和方差
        
        Args:
            x: [batch, C_in, seq_len]
        Returns:
            mu: [batch, latent_dim] 隐变量均值
            logvar: [batch, latent_dim] 隐变量对数方差
        """
        _, h_global = self.encoder(x)        # [batch, d_model]
        mu, logvar = self.fc_mu(h_global), self.fc_logvar(h_global)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        重参数化技巧：使梯度能够反向传播
        
        Args:
            mu: [batch, latent_dim] 均值
            logvar: [batch, latent_dim] 对数方差
        Returns:
            z: [batch, latent_dim] 采样的隐变量
        """
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        """
        解码器：隐变量 -> 重建序列
        
        Args:
            z: [batch, latent_dim]
        Returns:
            xhat: [batch, 1, seq_len] 重建的序列
        """
        B = z.size(0)
        decoder_base = 64  # 与 __init__ 中的 decoder_base 参数一致
        
        # 线性投影到固定长度64
        h = self.fc_dec(z).view(B, decoder_base, 64)  # [batch, decoder_base, 64]
        
        # 通过解码器上采样到目标长度
        xhat = self.decoder(h)  # [batch, 1, seq_len]
        return xhat

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: [batch, C_in, seq_len]
        Returns:
            recon: [batch, 1, seq_len] 重建序列
            e_pred: [batch, 1] 预测的e参数
            (mu, logvar): 隐变量分布参数（用于计算KL散度）
        """
        assert x.dim() == 3 and x.size(2) == self.seq_len, \
            f"输入应为 [batch, C_in, seq_len]，且seq_len={self.seq_len}，实际得到 {x.shape}"
        
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        e_pred = self.param_head(z)
        
        return recon, e_pred, (mu, logvar)


class HybridMambaTransformerEncoder(nn.Module):
    """
    混合 Mamba-Transformer 编码器
    - 底层: Mamba (快速局部时序建模, O(L))
    - 顶层: Transformer (全局注意力精细化, O(L²))
    
    输入: [batch, C_in, seq_len]
    输出: [batch, seq_len, d_model] 和 [batch, d_model] (全局表示)
    """
    def __init__(self, C_in=1, d_model=128, n_mamba=2, n_transformer=2, 
                 nhead=8, d_state=16, d_conv=4, expand=2, dropout=0.2):
        super().__init__()
        self.d_model = d_model
        
        # 输入投影
        self.proj_in = nn.Conv1d(C_in, d_model, kernel_size=1)
        
        # Mamba 层 (底层)
        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_mamba)
        ])
        self.mamba_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_mamba)])
        self.mamba_dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_mamba)])
        
        # 位置编码 (用于 Transformer)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer 层 (顶层)
        if n_transformer > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer)
        else:
            self.transformer_encoder = None
        
        # 输出归一化
        self.norm_out = nn.LayerNorm(d_model)
    
    def forward(self, x):
        """
        Args:
            x: [batch, C_in, seq_len]
        Returns:
            h: [batch, seq_len, d_model] token表示
            h_global: [batch, d_model] 全局表示
        """
        # 投影: [batch, C_in, seq_len] -> [batch, d_model, seq_len]
        h = self.proj_in(x)
        
        # 转换为序列格式: [batch, d_model, seq_len] -> [batch, seq_len, d_model]
        h = h.transpose(1, 2)
        
        # Mamba 层 (局部时序建模)
        for mamba_layer, norm, dropout in zip(self.mamba_layers, self.mamba_norms, self.mamba_dropouts):
            h_residual = h
            h = mamba_layer(h)
            h = dropout(h)
            h = norm(h + h_residual)  # 残差连接
        
        # Transformer 层 (全局精细化)
        if self.transformer_encoder is not None:
            h = self.pos_encoder(h)
            h = self.transformer_encoder(h)
        
        # 输出归一化
        h = self.norm_out(h)
        
        # 全局池化
        h_global = h.mean(dim=1)  # [batch, d_model]
        
        return h, h_global


class TimeVAE1D_HybridMambaTransformer(nn.Module):
    """
    基于 Hybrid Mamba-Transformer 的时间序列 VAE
    
    架构：
    - Encoder: Hybrid (Mamba + Transformer)
      * 底层 Mamba: 快速捕获局部时序依赖 (O(L))
      * 顶层 Transformer: 全局注意力精细化 (O(L²))
    - Decoder: Conv1D (上采样卷积)
    
    优势：
    - 结合 Mamba 的效率和 Transformer 的全局建模能力
    - Mamba 处理长序列高效，Transformer 提供全局精细化
    - 适合需要多层次特征表示的场景
    - 卷积解码器高效生成序列
    
    输入: [batch, C_in, seq_len]
    输出: 重建 [batch, 1, seq_len]、参数e预测 [batch, 1]
    """
    def __init__(self, seq_len=200, C_in=1, latent_dim=64,
                 d_model=128, n_mamba=2, n_transformer=2, nhead=8,
                 d_state=16, d_conv=4, expand=2, dropout=0.2, decoder_base=64):
        """
        Args:
            seq_len: 输入序列长度
            C_in: 输入通道数（默认1，原始温度序列）
            latent_dim: 隐空间维度
            d_model: 模型维度
            n_mamba: Mamba层数 (底层，快速)
            n_transformer: Transformer层数 (顶层，精细)
            nhead: Transformer注意力头数
            d_state: SSM状态维度
            d_conv: Mamba局部卷积宽度
            expand: Mamba扩展因子
            dropout: Dropout率
            decoder_base: 解码器基础通道数
        """
        super().__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim

        # ------ Encoder: Hybrid Mamba-Transformer ------
        self.encoder = HybridMambaTransformerEncoder(
            C_in=C_in,
            d_model=d_model,
            n_mamba=n_mamba,
            n_transformer=n_transformer,
            nhead=nhead,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout
        )
        self.fc_mu     = nn.Linear(d_model, latent_dim)
        self.fc_logvar = nn.Linear(d_model, latent_dim)

        # ------ Decoder: 使用自适应上采样回到seq_len ------
        self.fc_dec = nn.Linear(latent_dim, decoder_base * 64)
        
        self.decoder = nn.Sequential(
            # [batch, decoder_base, 64]
            nn.Conv1d(decoder_base, decoder_base*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),  # -> 128
            
            nn.Conv1d(decoder_base*2, decoder_base, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(size=seq_len, mode='linear', align_corners=False),  # -> seq_len
            
            nn.Conv1d(decoder_base, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=3, padding=1)  # -> [batch, 1, seq_len]
        )

        # ------ e参数预测头 ------
        self.param_head = nn.Sequential(
            nn.Linear(latent_dim, 128), 
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def encode(self, x):
        """
        编码器：输入序列 -> 隐空间均值和方差
        
        Args:
            x: [batch, C_in, seq_len]
        Returns:
            mu: [batch, latent_dim] 隐变量均值
            logvar: [batch, latent_dim] 隐变量对数方差
        """
        _, h_global = self.encoder(x)  # [batch, d_model]
        mu = self.fc_mu(h_global)
        logvar = self.fc_logvar(h_global)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        重参数化技巧：使梯度能够反向传播
        
        Args:
            mu: [batch, latent_dim] 均值
            logvar: [batch, latent_dim] 对数方差
        Returns:
            z: [batch, latent_dim] 采样的隐变量
        """
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        """
        解码器：隐变量 -> 重建序列
        
        Args:
            z: [batch, latent_dim]
        Returns:
            xhat: [batch, 1, seq_len] 重建的序列
        """
        B = z.size(0)
        decoder_base = 64
        
        # 线性投影到固定长度64
        h = self.fc_dec(z).view(B, decoder_base, 64)
        
        # 通过解码器上采样到目标长度
        xhat = self.decoder(h)
        return xhat

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: [batch, C_in, seq_len]
        Returns:
            recon: [batch, 1, seq_len] 重建序列
            e_pred: [batch, 1] 预测的e参数
            (mu, logvar): 隐变量分布参数（用于计算KL散度）
        """
        assert x.dim() == 3 and x.size(2) == self.seq_len, \
            f"输入应为 [batch, C_in, seq_len]，且seq_len={self.seq_len}，实际得到 {x.shape}"
        
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        e_pred = self.param_head(z)
        
        return recon, e_pred, (mu, logvar)



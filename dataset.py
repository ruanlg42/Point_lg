import torch
from torch.utils.data import Dataset


class ThermalDataset(Dataset):
    """热传导数据集"""
    
    def __init__(self, pth_path='packed_dataset/thermal_dataset.pth', 
                 normalize_temp=True, use_log_e=True):
        """
        Args:
            pth_path: 数据集路径
            normalize_temp: 是否归一化温度
            use_log_e: 是否对 e 取对数（处理偏斜分布）
        """
        self.data = torch.load(pth_path)
        self.normalize_temp = normalize_temp
        self.use_log_e = use_log_e
        
        # 获取温度序列和目标参数 e
        self.temperature = self.data['temperature']  # [N, T]
        self.e_values = self.data['parameters'][:, 4]  # e 是第5个参数 (索引4)
        
        # 温度归一化
        if normalize_temp:
            self.temp_mean = self.temperature.mean()
            self.temp_std = self.temperature.std()
            self.temperature = (self.temperature - self.temp_mean) / self.temp_std
        else:
            self.temp_mean = None
            self.temp_std = None
        
        # e 值处理：取对数 + 归一化
        if use_log_e:
            self.e_log = torch.log(self.e_values)  # 对数变换
            self.e_mean = self.e_log.mean()
            self.e_std = self.e_log.std()
            self.e_transformed = (self.e_log - self.e_mean) / self.e_std
        else:
            self.e_log = None
            self.e_mean = self.e_values.mean()
            self.e_std = self.e_values.std()
            self.e_transformed = (self.e_values - self.e_mean) / self.e_std
        
        print(f"数据集加载完成: {len(self)} 个样本")
        print(f"温度序列形状: {self.temperature.shape} (样本数 × 时间步)")
        print(f"每个样本时间步数: {self.temperature.shape[1]}")
        print(f"温度范围: [{self.data['temperature'].min():.2f}, {self.data['temperature'].max():.2f}]")
        print(f"e 值范围: [{self.e_values.min():.4f}, {self.e_values.max():.4f}]")
        if use_log_e:
            print(f"log(e) 范围: [{self.e_log.min():.4f}, {self.e_log.max():.4f}]")
        print(f"温度归一化: {normalize_temp}")
        print(f"e 对数变换: {use_log_e}")
    
    def __len__(self):
        return len(self.temperature)
    
    def __getitem__(self, idx):
        return {
            'temperature': self.temperature[idx],  # [T]
            'e': self.e_transformed[idx]  # 标量（已归一化）
        }
    
    def denormalize_e(self, e_norm):
        """反归一化 e 值"""
        if self.use_log_e:
            # 先反归一化，再取指数
            e_log = e_norm * self.e_std + self.e_mean
            return torch.exp(e_log)
        else:
            return e_norm * self.e_std + self.e_mean



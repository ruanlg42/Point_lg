import numpy as np
from PIL import Image
from pathlib import Path

# ==================== 配置区域 ====================
GIF_PATH = '/home/ziwu/Newpython/lg_exp/Point_lg/gif_test/newnewnew_30hz/4730_video.gif'
ORIGINAL_TEMP_TXT = '/home/ziwu/Newpython/lg_exp/Point_lg/gif_test/newnewnew_30hz/4730_mph.txt'  # 原始温度文件，如果没有设为 None
MIN_TEMP = 280.0  # 最黑对应的温度 (K)
MAX_TEMP = 320.0  # 最亮对应的温度 (K)
# ================================================


def load_original_temperature(txt_path):
    """读取原始温度txt文件"""
    if txt_path is None or not Path(txt_path).exists():
        return None
    
    temps = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            # 跳过注释行和空行
            if not line or line.startswith('%') or line.startswith('#'):
                continue
            # 分割数据（时间 温度）
            parts = line.split()
            if len(parts) >= 2:
                try:
                    temp = float(parts[1])
                    temps.append(temp)
                except ValueError:
                    continue
    
    return np.array(temps) if temps else None


def gif_to_temperature(gif_path, min_temp=280.0, max_temp=320.0):
    """将 GIF 转换为温度序列"""
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
    # 像素值 0-255 映射到 min_temp-max_temp
    temperature = (frames / 255.0) * (max_temp - min_temp) + min_temp
    return temperature, frames


def main():
    print(f"Reading GIF: {GIF_PATH}")
    print(f"Temperature mapping: {MIN_TEMP}K (black) -> {MAX_TEMP}K (white)")
    print("=" * 60)
    
    # 读取 GIF
    temperature_map, pixel_values = gif_to_temperature(GIF_PATH, MIN_TEMP, MAX_TEMP)
    H, W, T = temperature_map.shape
    
    print(f"GIF shape: {H}x{W} pixels, {T} frames")
    print(f"Temperature range: {temperature_map.min():.2f} - {temperature_map.max():.2f} K")
    print("=" * 60)
    
    # 计算正中心位置
    center_h, center_w = H // 2, W // 2
    
    print(f"\nCenter pixel position: ({center_h}, {center_w})")
    print("=" * 60)
    
    # 提取中心像素点的温度序列
    center_temps = []
    center_pixels = []
    
    for frame_idx in range(T):
        pixel_val = pixel_values[center_h, center_w, frame_idx]
        temp = temperature_map[center_h, center_w, frame_idx]
        center_temps.append(temp)
        center_pixels.append(pixel_val)
    
    # 加载原始温度数据（如果有）
    original_temps = load_original_temperature(ORIGINAL_TEMP_TXT)
    
    # 显示统计信息
    print(f"\nCenter pixel temperature statistics (from GIF):")
    print(f"  Frame 0 (first):    Pixel={center_pixels[0]:6.2f}, Temp={center_temps[0]:6.2f} K")
    print(f"  Frame {T-1} (last):     Pixel={center_pixels[-1]:6.2f}, Temp={center_temps[-1]:6.2f} K")
    print(f"  Mean across frames: {np.mean(center_temps):6.2f} K")
    print(f"  Min:                {np.min(center_temps):6.2f} K")
    print(f"  Max:                {np.max(center_temps):6.2f} K")
    print(f"  Std:                {np.std(center_temps):6.2f} K")
    
    if original_temps is not None:
        print(f"\nOriginal temperature data loaded: {len(original_temps)} points")
        min_len = min(T, len(original_temps))
        diffs = [center_temps[i] - original_temps[i] for i in range(min_len)]
        print(f"  Comparison (first {min_len} frames):")
        print(f"    Mean difference:  {np.mean(diffs):6.2f} K")
        print(f"    Max difference:   {np.max(np.abs(diffs)):6.2f} K")
        print(f"    RMSE:             {np.sqrt(np.mean(np.array(diffs)**2)):6.2f} K")
    print("=" * 60)
    
    # 保存到 txt 文件
    gif_path = Path(GIF_PATH)
    output_txt = gif_path.parent / f"{gif_path.stem}_center_temp.txt"
    
    with open(output_txt, 'w') as f:
        f.write(f"GIF: {gif_path.name}\n")
        f.write(f"Center pixel position: ({center_h}, {center_w})\n")
        f.write(f"Temperature mapping: {MIN_TEMP}K (black) -> {MAX_TEMP}K (white)\n")
        f.write(f"Total frames: {T}\n")
        if original_temps is not None:
            f.write(f"Original temperature file: {Path(ORIGINAL_TEMP_TXT).name}\n")
        f.write("=" * 60 + "\n\n")
        
        # 根据是否有原始数据决定列格式
        if original_temps is not None:
            min_len = min(T, len(original_temps))
            f.write("Frame    Pixel Value    Temp_GIF (K)    Temp_Original (K)    Difference (K)\n")
            f.write("-" * 80 + "\n")
            for i in range(min_len):
                diff = center_temps[i] - original_temps[i]
                f.write(f"{i:5d}    {center_pixels[i]:11.2f}    {center_temps[i]:13.2f}    {original_temps[i]:19.2f}    {diff:15.2f}\n")
            # 如果GIF帧数多于原始数据
            for i in range(min_len, T):
                f.write(f"{i:5d}    {center_pixels[i]:11.2f}    {center_temps[i]:13.2f}    {'N/A':>19}    {'N/A':>15}\n")
        else:
            f.write("Frame    Pixel Value    Temperature (K)\n")
            f.write("-" * 45 + "\n")
            for i in range(T):
                f.write(f"{i:5d}    {center_pixels[i]:11.2f}    {center_temps[i]:15.2f}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("Statistics (GIF temperature):\n")
        f.write(f"  Mean: {np.mean(center_temps):.2f} K\n")
        f.write(f"  Min:  {np.min(center_temps):.2f} K\n")
        f.write(f"  Max:  {np.max(center_temps):.2f} K\n")
        f.write(f"  Std:  {np.std(center_temps):.2f} K\n")
        
        if original_temps is not None:
            min_len = min(T, len(original_temps))
            diffs = [center_temps[i] - original_temps[i] for i in range(min_len)]
            f.write("\nComparison with original data:\n")
            f.write(f"  Mean difference:  {np.mean(diffs):.2f} K\n")
            f.write(f"  Max abs diff:     {np.max(np.abs(diffs)):.2f} K\n")
            f.write(f"  RMSE:             {np.sqrt(np.mean(np.array(diffs)**2)):.2f} K\n")
    
    print(f"\nSaved to: {output_txt}")
    print("=" * 60)


if __name__ == '__main__':
    main()


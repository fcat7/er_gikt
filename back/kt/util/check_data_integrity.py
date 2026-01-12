import numpy as np
import os

# 检查.npy文件是否存在，并进行描述性统计，检查形状、缺失值（如 NaN)等
def check_npy_file(file_path):
    print(f"Checking file: {file_path}")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    try:
        data = np.load(file_path)
        print(f"  Shape: {data.shape}")
        print(f"  Dtype: {data.dtype}")
        
        has_nan = np.isnan(data).any()
        has_inf = np.isinf(data).any()
        
        print(f"  Has NaN: {has_nan}")
        print(f"  Has Inf: {has_inf}")
        
        if has_nan:
            nan_indices = np.where(np.isnan(data))
            print(f"  NaN Count: {len(nan_indices[0])}")
            print(f"  First 5 NaN indices: {list(zip(*[x[:5] for x in nan_indices]))}")
            
        if has_inf:
            inf_indices = np.where(np.isinf(data))
            print(f"  Inf Count: {len(inf_indices[0])}")
        
        # 排除 NaN/Inf 后统计数值范围
        valid_mask = ~(np.isnan(data) | np.isinf(data))
        if valid_mask.any():
            valid_data = data[valid_mask]
            print(f"  Min: {valid_data.min()}")
            print(f"  Max: {valid_data.max()}")
            print(f"  Mean: {valid_data.mean()}")
        else:
            print("  No valid data to calculate stats.")
            
    except Exception as e:
        print(f"Error checking file: {e}")
    print("-" * 30)

# Check assist09 files (which the user indicated were problematic)

# 获取项目根目录
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

base_dir = root_dir + '/data/assist09'
check_npy_file(os.path.join(base_dir, 'user_interval_time.npy'))
check_npy_file(os.path.join(base_dir, 'user_response_time.npy'))

# Also check ednet_kt1 just in case, since it was in the previous file
base_dir_ednet = root_dir + '/data/ednet_kt1'
check_npy_file(os.path.join(base_dir_ednet, 'user_interval_time.npy'))
check_npy_file(os.path.join(base_dir_ednet, 'user_response_time.npy'))

"""
数据集加载策略
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset

class UserDataset(Dataset):

    def __init__(self, config):
        processed_dir = config.PROCESSED_DATA_DIR
        
        # 输入数据
        self.user_seq = torch.tensor(np.load(os.path.join(processed_dir, 'user_seq.npy')), dtype=torch.int64)
        # [num_user, max_seq_len] 输入数据
        self.user_res = torch.tensor(np.load(os.path.join(processed_dir, 'user_res.npy')), dtype=torch.int64)
        # [num_user, max_seq_len] 输入标签
        self.user_mask = torch.tensor(np.load(os.path.join(processed_dir, 'user_mask.npy')), dtype=torch.bool)
        # [num_user, max_seq_len] 有值效记录
        
        # 新增时间特征
        # 检查文件是否存在，如果不存在则使用全0初始化 (兼容旧代码)
        try:
            self.user_interval_time = torch.tensor(np.load(os.path.join(processed_dir, 'user_interval_time.npy')), dtype=torch.float32)
            self.user_response_time = torch.tensor(np.load(os.path.join(processed_dir, 'user_response_time.npy')), dtype=torch.float32)
        except FileNotFoundError:
            print("Warning: Time feature files not found. Using zeros.")
            self.user_interval_time = torch.zeros_like(self.user_seq, dtype=torch.float32)
            self.user_response_time = torch.zeros_like(self.user_seq, dtype=torch.float32)

    # # 假设原始数据：
    # user_seq = [1, 2, 3, …]      # 问题ID序列
    # user_res = [1, 0, 1, …]      # 答题结果序列
    # user_mask = [1, 1, 1, …]     # 掩码序列
    # user_interval = [0.1, 0.5, ...] # 间隔时间序列
    # user_response = [1.2, 0.8, ...] # 作答时间序列
    # # 堆叠后：
    # stacked = [
    #     [1, 1, 1, 0.1, 1.2],  # 第一个时间步：[问题ID, 答题结果, 掩码, 间隔, 作答时间]
    #     ...
    # ]
    def __getitem__(self, index): # index: 用户索引，指定要获取哪个用户的数据
        return torch.stack([
            self.user_seq[index], 
            self.user_res[index], 
            self.user_mask[index],
            self.user_interval_time[index],
            self.user_response_time[index]
        ], dim=-1)
        # 将5个一维张量沿着最后一个维度（dim=-1）堆叠，结果是一个二维张量，形状为 [max_seq_len, 5]

    def __len__(self):
        return self.user_seq.shape[0]

# # 示例说明 [num_user=3, max_seq_len=5, 3]
# [
#     # 用户1的答题序列
#     [
#         [1, 1, 1],  # 第1题：问题ID=1, 答对(1), 有效数据(1)
#         [2, 0, 1],  # 第2题：问题ID=2, 答错(0), 有效数据(1)
#         [3, 1, 1],  # 第3题：问题ID=3, 答对(1), 有效数据(1)
#         [0, 0, 0],  # 填充：无效数据
#         [0, 0, 0]   # 填充：无效数据
#     ],
    
#     # 用户2的答题序列
#     [
#         [1, 1, 1],  # 第1题：问题ID=1, 答对(1), 有效数据(1)
#         [4, 1, 1],  # 第2题：问题ID=4, 答对(1), 有效数据(1)
#         [2, 0, 1],  # 第3题：问题ID=2, 答错(0), 有效数据(1)
#         [5, 1, 1],  # 第4题：问题ID=5, 答对(1), 有效数据(1)
#         [0, 0, 0]   # 填充：无效数据
#     ],
    
#     # 用户3的答题序列
#     [
#         [3, 0, 1],  # 第1题：问题ID=3, 答错(0), 有效数据(1)
#         [1, 1, 1],  # 第2题：问题ID=1, 答对(1), 有效数据(1)
#         [0, 0, 0],  # 填充：无效数据
#         [0, 0, 0],  # 填充：无效数据
#         [0, 0, 0]   # 填充：无效数据
#     ]
# ]
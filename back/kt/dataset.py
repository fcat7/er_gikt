"""
数据集加载策略
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import sparse

class UserDataset(Dataset):

    def __init__(self, config, augment=False, prob_mask=0.1, mode='train'):
        """
        Args:
            mode: 'train' or 'test'. Loads {mode}_seq.npy etc.
        """
        processed_dir = config.PROCESSED_DATA_DIR
        self.augment = augment
        self.prob_mask = prob_mask # 随机掩码概率
        
        prefix = f"{mode}_"
        
        # 输入数据
        try:
            self.user_seq = torch.tensor(np.load(os.path.join(processed_dir, prefix + 'seq.npy')), dtype=torch.int64)
            # [num_user, max_seq_len] 输入数据
            self.user_res = torch.tensor(np.load(os.path.join(processed_dir, prefix + 'res.npy')), dtype=torch.int64)
            # [num_user, max_seq_len] 输入标签
            self.user_mask = torch.tensor(np.load(os.path.join(processed_dir, prefix + 'mask.npy')), dtype=torch.bool)
            # [num_user, max_seq_len] 有值效记录
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find dataset files with prefix '{prefix}' in {processed_dir}. Did you run data_process.py?")
        
        # 新增时间特征
        # 检查文件是否存在，如果不存在则使用全0初始化 (兼容旧代码)
        try:
            self.user_interval_time = torch.tensor(np.load(os.path.join(processed_dir, prefix + 'interval_time.npy')), dtype=torch.float32)
            self.user_response_time = torch.tensor(np.load(os.path.join(processed_dir, prefix + 'response_time.npy')), dtype=torch.float32)
        except FileNotFoundError:
            print(f"Warning: Time feature files for '{mode}' not found. Using zeros.")
            self.user_interval_time = torch.zeros_like(self.user_seq, dtype=torch.float32)
            self.user_response_time = torch.zeros_like(self.user_seq, dtype=torch.float32)

        # @fix_fzq: 加载 Eval Mask (用于区分评估区间，避免冷启动与泄露)
        try:
            self.user_eval_mask = torch.tensor(np.load(os.path.join(processed_dir, prefix + 'eval_mask.npy')), dtype=torch.bool)
        except FileNotFoundError:
            # 兼容旧数据：若不存在 eval_mask，则默认与 user_mask 一致
            self.user_eval_mask = self.user_mask.clone()

        # @fix_fzq: 加载 Group 信息 (用于 GroupKFold 防泄露)
        try:
            self.groups = np.load(os.path.join(processed_dir, prefix + 'window_groups.npy'))
            # print("Loaded user groups for leakage-free splitting.")
        except FileNotFoundError:
            self.groups = None

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
        seq = self.user_seq[index]
        res = self.user_res[index]
        mask = self.user_mask[index]
        interval = self.user_interval_time[index]
        response_time = self.user_response_time[index]
        eval_mask = self.user_eval_mask[index]

        if self.augment:
            # 随机掩码 (Random Masking)
            if self.prob_mask > 0:
                # 创建一个随机概率掩码 (prob < prob_mask)
                prob_random = torch.rand(seq.shape) < self.prob_mask
                
                # 不能掩盖所有数据，至少保留一个
                # 只有原来是 True 的地方才允许被 mask 掉
                mask_to_drop = prob_random & mask
                
                # 如果全部被 drop 了，强行保留至少一个（防止报错）
                if mask_to_drop.sum() == mask.sum() and mask.sum() > 0:
                    # 随机选一个不 drop
                    valid_indices = torch.nonzero(mask, as_tuple=True)[0]
                    keep_idx = valid_indices[torch.randint(0, len(valid_indices), (1,))]
                    mask_to_drop[keep_idx] = False
                    
                # 应用由于数据增强导致的 mask 变更 (设置为 0)
                # 克隆 mask 以避免修改原始数据 (Tensor切片可能是视图)
                mask = mask.clone() 
                mask[mask_to_drop] = False
                # eval_mask 同步更新，避免对被丢弃位置进行评估
                eval_mask = eval_mask.clone()
                eval_mask[mask_to_drop] = False

        return torch.stack([
            seq, 
            res, 
            mask,
            interval,
            response_time,
            eval_mask
        ], dim=-1)
        # 将6个一维张量沿着最后一个维度（dim=-1）堆叠，结果是一个二维张量，形状为 [max_seq_len, 6]

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
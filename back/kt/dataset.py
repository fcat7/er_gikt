import os
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class SeqFeatureKey:
    """
    统一管理序列特征键名，避免魔法字符串 / 魔法下标。
    """
    Q = "q_seq"
    R = "r_seq"
    MASK = "mask"
    EVAL_MASK = "eval_mask"
    T_INTERVAL = "t_interval"
    T_RESPONSE = "t_response"


class UnifiedParquetDataset(Dataset):
    """
    统一的 Parquet 数据集加载器
    支持动态 Padding 和数据增强

    __getitem__ 返回字典视图：
        {
            SeqFeatureKey.Q:         Tensor[max_seq_len],
            SeqFeatureKey.R:         Tensor[max_seq_len],
            SeqFeatureKey.MASK:      Tensor[max_seq_len] (bool),
            SeqFeatureKey.EVAL_MASK: Tensor[max_seq_len] (bool),
            SeqFeatureKey.T_INTERVAL:Tensor[max_seq_len],
            SeqFeatureKey.T_RESPONSE:Tensor[max_seq_len],
        }
    DataLoader 默认 collate 后，batch 形状为 [batch_size, max_seq_len]。
    """

    def __init__(self, config, augment: bool = False, prob_mask: float = 0.1, mode: str = 'train'):
        self.config = config
        self.augment = augment
        self.prob_mask = prob_mask
        self.mode = mode

        data_dir = config.PROCESSED_DATA_DIR
        file_path = os.path.join(data_dir, f"{mode}.parquet")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Parquet file not found: {file_path}. Please run data_process.py first.")

        # 加载元数据
        try:
            with open(os.path.join(data_dir, "metadata.json"), 'r') as f:
                self.meta = json.load(f)
        except Exception:
            self.meta = {}

        self.max_seq_len = self.meta.get('config_at_processing', {}).get('max_seq_len', 200)

        # 加载主数据
        self.df = pd.read_parquet(file_path)

        # 提取 groups 用于 GroupKFold
        if 'group_id' in self.df.columns:
            self.groups = self.df['group_id'].values
        else:
            self.groups = None

    def __len__(self):
        return len(self.df)

    def _pad_sequence(self, seq, dtype=np.int64, pad_val=0):
        """动态 Padding 到 max_seq_len"""
        arr = np.array(seq, dtype=dtype)
        if len(arr) >= self.max_seq_len:
            return arr[:self.max_seq_len]
        padded = np.full(self.max_seq_len, pad_val, dtype=dtype)
        padded[:len(arr)] = arr
        return padded

    def __getitem__(self, index):
        row = self.df.iloc[index]

        # 动态 Padding 并转换为 Tensor
        q_seq = torch.from_numpy(self._pad_sequence(row['q_seq'], dtype=np.int64))
        r_seq = torch.from_numpy(self._pad_sequence(row['r_seq'], dtype=np.int64))
        mask = torch.from_numpy(self._pad_sequence(row['mask'], dtype=bool, pad_val=False))
        # 评估掩码
        if 'eval_mask' in row and row['eval_mask'] is not None:
            eval_mask = torch.from_numpy(self._pad_sequence(row['eval_mask'], dtype=bool, pad_val=False))
        else:
            eval_mask = mask.clone()
        # 可选时间特征
        if 't_interval' in row and row['t_interval'] is not None:
            t_interval = torch.from_numpy(self._pad_sequence(row['t_interval'], dtype=np.float32))
        else:
            t_interval = torch.zeros_like(q_seq, dtype=torch.float32)

        if 't_response' in row and row['t_response'] is not None:
            t_response = torch.from_numpy(self._pad_sequence(row['t_response'], dtype=np.float32))
        else:
            t_response = torch.zeros_like(q_seq, dtype=torch.float32)

        # 数据增强 (逻辑完全复用，作用在 mask / eval_mask 上)
        # 随机丢弃一部分交互，模拟缺失数据或子采样
        if self.augment and self.prob_mask > 0:
            prob_random = torch.rand(q_seq.shape) < self.prob_mask
            mask_to_drop = prob_random & mask

            if mask_to_drop.sum() == mask.sum() and mask.sum() > 0:
                valid_indices = torch.nonzero(mask, as_tuple=True)[0]
                keep_idx = valid_indices[torch.randint(0, len(valid_indices), (1,))]
                mask_to_drop[keep_idx] = False

            mask = mask.clone()
            mask[mask_to_drop] = False
            eval_mask = eval_mask.clone()
            eval_mask[mask_to_drop] = False

        # 返回“具名字段”的字典视图，便于不同模型按需取用
        return {
            SeqFeatureKey.Q: q_seq,
            SeqFeatureKey.R: r_seq,
            SeqFeatureKey.MASK: mask,
            SeqFeatureKey.EVAL_MASK: eval_mask,
            SeqFeatureKey.T_INTERVAL: t_interval,
            SeqFeatureKey.T_RESPONSE: t_response,
        }

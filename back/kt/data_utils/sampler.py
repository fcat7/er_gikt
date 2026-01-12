import pandas as pd
import numpy as np
from config import MIN_SEQ_LEN, RANDOM_SEED

try:
    from icecream import ic
except ImportError:
    ic = print

class KTDataSampler:
    """
    数据采样器
    负责从总体数据集中抽取子集，并保持分布一致性
    """
    def __init__(self, random_seed=RANDOM_SEED):
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def stratified_sample(self, df, ratio=0.1, min_seq_len=MIN_SEQ_LEN):
        """
        分层采样：根据用户的交互数量进行分层
        
        Args:
            df: 总体数据集 DataFrame
            ratio: 采样比例 (0.0 - 1.0)
            min_seq_len: 最小交互数限制 (先过滤再采样)
        
        Returns:
            sampled_df: 采样后的 DataFrame
        """
        ic(f"开始分层采样，目标比例: {ratio}")
        
        # 1. 预过滤：只考虑交互数足够的学生
        user_counts = df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= min_seq_len].index
        valid_df = df[df['user_id'].isin(valid_users)]
        
        if valid_df.empty:
            raise ValueError("没有满足最小交互数要求的用户")

        # 2. 定义分层策略
        # 将用户按交互数量分为 5 个桶 (Quantiles)
        # 这样可以确保我们采样到的用户既有活跃的，也有不活跃的
        user_interaction_counts = valid_df.groupby('user_id').size().reset_index(name='counts')
        
        # 使用 qcut 分桶，如果数据太少可能无法分5桶，尝试减少桶数
        try:
            user_interaction_counts['bucket'] = pd.qcut(user_interaction_counts['counts'], q=5, labels=False, duplicates='drop')
        except ValueError:
            # 如果数据量太小或分布太集中，直接随机采样
            ic("警告：无法进行分桶（可能数据量太小），降级为简单随机采样")
            user_interaction_counts['bucket'] = 0

        # 3. 从每个桶中抽取指定比例的用户
        sampled_users = []
        for bucket in user_interaction_counts['bucket'].unique():
            bucket_users = user_interaction_counts[user_interaction_counts['bucket'] == bucket]['user_id']
            n_sample = int(len(bucket_users) * ratio)
            if n_sample > 0:
                sampled = np.random.choice(bucket_users, n_sample, replace=False)
                sampled_users.extend(sampled)
        
        # 4. 提取数据
        sampled_df = valid_df[valid_df['user_id'].isin(sampled_users)]
        
        ic(f"采样完成: {len(valid_df)} -> {len(sampled_df)} (Users: {len(valid_users)} -> {len(sampled_users)})")
        return sampled_df

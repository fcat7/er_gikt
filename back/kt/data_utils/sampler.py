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
        分层采样：同时根据用户的交互数量(Sequence Length)和平均正确率(Accuracy)进行分层
        策略与 Splitter 保持一致。
        
        Args:
            df: 总体数据集 DataFrame
            ratio: 采样比例 (0.0 - 1.0)
            min_seq_len: 最小交互数限制 (先过滤再采样)
        
        Returns:
            sampled_df: 采样后的 DataFrame
        """
        ic(f"开始双重分层采样 (Length & Acc)，目标比例: {ratio}")
        
        # 1. 预过滤：只考虑交互数足够的学生
        # 如果 df 中没有 correct 列，降级为仅 Len 分层
        if 'correct' not in df.columns:
            ic("Warning: 'correct' column missing. Fallback to Length-only stratification.")
            return self._stratified_sample_legacy(df, ratio, min_seq_len)

        # 聚合用户统计
        user_stats = df.groupby('user_id').agg(
            count=('correct', 'count'),
            acc=('correct', 'mean')
        ).reset_index()
        
        valid_users = user_stats[user_stats['count'] >= min_seq_len].copy()
        
        if valid_users.empty:
            raise ValueError("没有满足最小交互数要求的用户")

        # 2. 定义分层策略 (Length (5) * Accuracy (5))
        n_bins = 5
        try:
            valid_users['len_bin'] = pd.qcut(valid_users['count'], q=n_bins, labels=False, duplicates='drop')
        except ValueError:
            valid_users['len_bin'] = 0
            
        try:
            valid_users['acc_bin'] = pd.qcut(valid_users['acc'], q=n_bins, labels=False, duplicates='drop')
        except ValueError:
            valid_users['acc_bin'] = 0

        valid_users['stratum'] = valid_users['len_bin'].astype(str) + "_" + valid_users['acc_bin'].astype(str)

        # 3. 从每个分层中抽取指定比例的用户
        sampled_user_ids = []
        for stratum, group in valid_users.groupby('stratum'):
            u_ids = group['user_id'].values
            n_sample = int(len(u_ids) * ratio)
            
            # 即使比例很小，如果 bucket 有人，是否至少取 1 个？
            # 严格按照比例：如果 0.05 * 10 = 0.5 -> 0. 
            # 如果样本非常重要，可以做 n_sample = max(1, ...) 但这会偏离 ratio。
            # 这里保持严格比例。
            
            if n_sample > 0:
                sampled = np.random.choice(u_ids, n_sample, replace=False)
                sampled_user_ids.extend(sampled)
        
        # 4. 提取数据
        sampled_df = df[df['user_id'].isin(sampled_user_ids)].copy()
        
        ic(f"采样完成: {len(df)} -> {len(sampled_df)} (Users: {len(valid_users)} -> {len(sampled_user_ids)})")
        return sampled_df

    def _stratified_sample_legacy(self, df, ratio=0.1, min_seq_len=MIN_SEQ_LEN):
        """
        [旧版] 仅基于交互长度采样
        """
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
        
        ic(f"采样完成 (Legacy): {len(valid_df)} -> {len(sampled_df)} (Users: {len(valid_users)} -> {len(sampled_users)})")
        return sampled_df

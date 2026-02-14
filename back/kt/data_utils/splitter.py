import pandas as pd
import numpy as np

try:
    from icecream import ic
except ImportError:
    ic = print

class KTDataSplitter:
    """
    数据集划分器 (Stratified Splitter)
    负责根据用户的属性（序列长度、平均正确率）进行分层采样，
    将用户划分为训练集和测试集，确保两者分布一致。
    """
    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def split_users(self, df, test_ratio=0.2, min_seq_len=5):
        """
        基于分层采样划分用户。
        
        Args:
            df: 包含 'user_id', 'correct' 的 DataFrame。
            test_ratio: 测试集占比 (0.2 表示 20%)。
            min_seq_len: 用户最小交互数，少于此数的用户将被忽略。
            
        Returns:
            train_user_ids (set): 训练集用户ID集合。
            test_user_ids (set): 测试集用户ID集合。
        """
        ic(f"开始分层划分 (Test Ratio={test_ratio}, Seed={self.random_seed})...")
        
        # 1. 聚合：计算每个用户的统计特征
        # 必须列: correct, user_id
        if 'correct' not in df.columns:
            raise ValueError("Dataframe must contain 'correct' column for stratification.")

        user_stats = df.groupby('user_id').agg(
            count=('correct', 'count'),
            acc=('correct', 'mean')
        ).reset_index()
        
        # 过滤过短序列
        valid_users = user_stats[user_stats['count'] >= min_seq_len].copy()
        
        ignored = len(user_stats) - len(valid_users)
        if ignored > 0:
            ic(f"忽略 {ignored} 个交互数 < {min_seq_len} 的用户。")
        
        if valid_users.empty:
            raise ValueError("过滤后无有效用户，请检查数据或 min_seq_len 设置。")

        # 2. 分桶 (Stratification Bins)
        # 序列长度分桶 (Quantiles)
        # 如果用户数太少，qcut 可能失败，回退到 cut 或仅用 1 个桶
        n_samples = len(valid_users)
        n_bins = 5
        
        # Bin 1: Interaction Count (Length)
        try:
            valid_users['len_bin'] = pd.qcut(valid_users['count'], q=n_bins, labels=False, duplicates='drop')
        except ValueError:
            valid_users['len_bin'] = 0 # 无法分桶
            
        # Bin 2: Accuracy
        try:
            valid_users['acc_bin'] = pd.qcut(valid_users['acc'], q=n_bins, labels=False, duplicates='drop')
        except ValueError:
            valid_users['acc_bin'] = 0

        # 组合分层键
        valid_users['stratum'] = valid_users['len_bin'].astype(str) + "_" + valid_users['acc_bin'].astype(str)
        
        # 3. 分层抽样
        train_ids = []
        test_ids = []
        
        grouped = valid_users.groupby('stratum')
        # ic(f"分层桶数量: {len(grouped)}")

        for stratum, group in grouped:
            u_ids = group['user_id'].values
            np.random.shuffle(u_ids)
            
            n_total = len(u_ids)
            n_test = int(n_total * test_ratio)
            
            # 确保即使组很小，分配也尽量合理。
            # 如果 n_test 为 0 但 ratio > 0，且组内有数据，通常全归 train (或者随机给一个test，但这违反比例)
            # 这里严格按照比例截断
            
            test_subset = u_ids[:n_test]
            train_subset = u_ids[n_test:]
            
            test_ids.extend(test_subset)
            train_ids.extend(train_subset)
            
        ic(f"划分结果: Train Users={len(train_ids)}, Test Users={len(test_ids)}")
        
        # 最终验证
        train_set = set(train_ids)
        test_set = set(test_ids)
        assert len(train_set & test_set) == 0, "训练集和测试集用户重叠！"
        
        return train_set, test_set

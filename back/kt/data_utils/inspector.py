import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
try:
    from icecream import ic
except ImportError:
    ic = print

class KTDataInspector:
    """
    数据探查器
    负责统计数据分布、生成报告
    """
    def __init__(self, df, config):
        self.df = df
        self.config = config
        self.report_dir = config.REPORT_DIR
        if not os.path.exists(self.report_dir):
            os.makedirs(self.report_dir)

    def get_stats(self):
        """
        获取基础统计信息
        """
        stats = {
            'n_users': self.df['user_id'].nunique(),
            'n_items': self.df['problem_id'].nunique(),
            'n_skills': self.df['skill_id'].nunique() if 'skill_id' in self.df.columns else 0,
            'n_interactions': len(self.df),
            'avg_seq_len': self.df.groupby('user_id').size().mean(),
            'sparsity': 1 - len(self.df) / (self.df['user_id'].nunique() * self.df['problem_id'].nunique())
        }
        
        ic("数据集统计信息:")
        for k, v in stats.items():
            ic(f"{k}: {v}")
            
        return stats

    def analyze_distributions(self):
        """
        分析数据分布并绘图
        """
        # 1. 用户交互长度分布
        user_counts = self.df['user_id'].value_counts()
        plt.figure(figsize=(10, 6))
        sns.histplot(user_counts, bins=50, kde=True)
        plt.title('User Interaction Count Distribution')
        plt.xlabel('Number of Interactions')
        plt.ylabel('Count')
        plt.savefig(os.path.join(self.report_dir, 'user_interaction_dist.png'))
        plt.close()
        
        # 2. 题目被做次数分布
        item_counts = self.df['problem_id'].value_counts()
        plt.figure(figsize=(10, 6))
        sns.histplot(item_counts, bins=50, kde=True)
        plt.title('Problem Frequency Distribution')
        plt.xlabel('Frequency')
        plt.ylabel('Count')
        plt.savefig(os.path.join(self.report_dir, 'problem_frequency_dist.png'))
        plt.close()
        
        ic(f"分布图已保存至: {self.report_dir}")

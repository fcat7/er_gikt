import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import logging
from config import Config
from .standard_columns import StandardColumns

try:
    from icecream import ic
except ImportError:
    ic = print

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class KTDataInspector:
    """
    严肃的数据探查器 (Data Sanity Check Guardrails)
    不仅仅是统计分布，而是强制暴露稀疏性、信息穿越和异常冷启动。
    如果数据质量不达标，将抛出严重警告甚至阻断。
    """
    def __init__(self, df: pd.DataFrame, config: Config):
        self.df = df
        self.config = config
        self.report_dir = config.path.REPORT_DIR
        if not os.path.exists(self.report_dir):
            os.makedirs(self.report_dir)

    def run_full_sanity_check(self):
        """执行全体检流程"""
        ic("🚀 启动数据质量强制体检 (Data Sanity Check)...")
        self.check_sparsity()
        self.check_logical_consistency()
        self.check_temporal_leakage()
        self.analyze_distributions()
        ic("✅ 数据自身一致性体检完成。")

    def check_sparsity(self, threshold_count: int = 5, alert_ratio: float = 0.2):
        """长尾与稀疏度扫描"""
        item_counts = self.df[StandardColumns.QUESTION_ID].value_counts()
        sparse_items = (item_counts < threshold_count).sum()
        sparse_ratio = sparse_items / len(item_counts)
        
        ic(f"[{StandardColumns.QUESTION_ID} 稀疏度检测] 出现不足 {threshold_count} 次的题目占比: {sparse_ratio:.2%}")
        if sparse_ratio > alert_ratio:
            ic(f"\033[91m🚨 [高度稀疏警告] 你的数据集中有 {sparse_ratio:.2%} 的题目全量出现次数少于 {threshold_count} 次！这会导致严重的 Embedding 记忆过拟合。建议进行 K-core 过滤或映射为 UNK。\033[0m")

        if hasattr(StandardColumns, 'SKILL_IDS') and StandardColumns.SKILL_IDS in self.df.columns:
            skill_counts = self.df[StandardColumns.SKILL_IDS].value_counts()
            sparse_skills = (skill_counts < threshold_count).sum()
            if len(skill_counts) > 0 and (sparse_skills / len(skill_counts)) > alert_ratio:
                ic(f"\033[91m🚨 [技能稀疏警告] 你的数据集中有 {sparse_skills / len(skill_counts):.2%} 的技能出现极少。\033[0m")

    def check_logical_consistency(self):
        """逻辑自洽性检查"""
        # 1. 检查异常准确率
        q_mean = self.df.groupby(StandardColumns.QUESTION_ID)[StandardColumns.LABEL].mean()
        all_1 = (q_mean == 1.0).sum()
        all_0 = (q_mean == 0.0).sum()
        total_q = len(q_mean)
        
        if (all_1 + all_0) / total_q > 0.1:
            ic(f"\033[91m🤔 [逻辑异常] 高达 {(all_1 + all_0) / total_q:.2%} 的题目通过率是绝对的 100% 或 0%！请检查是否把登陆操作等非做题行为卷入了。\033[0m")

    def check_temporal_leakage(self, rapid_threshold_sec: float = 3.0):
        """隐式序列作弊感知 (Temporal Leakage)"""
        if StandardColumns.TIMESTAMP not in self.df.columns:
            return
            
        # 强制转换为数值类型进行时间推算
        self.df[StandardColumns.TIMESTAMP] = pd.to_numeric(self.df[StandardColumns.TIMESTAMP], errors='coerce')
        sorted_df = self.df.sort_values([StandardColumns.USER_ID, StandardColumns.TIMESTAMP])
        
        time_diffs = sorted_df.groupby(StandardColumns.USER_ID)[StandardColumns.TIMESTAMP].diff()
        # 很多 timestamp 是以秒为单位，如果是毫秒则自行调整
        small_diffs = (time_diffs < rapid_threshold_sec).sum()
        total_transitions = len(time_diffs.dropna())
        
        if total_transitions > 0:
            rapid_ratio = small_diffs / total_transitions
            ic(f"[快速答题感知] 两次交互间隔小于 {rapid_threshold_sec}秒 占比: {rapid_ratio:.2%}")
            if rapid_ratio > 0.1:
                ic(f"\033[91m🚨 [泄漏时间序列警告] 有 {rapid_ratio:.2%} 的记录是瞬间连答！极可能混入了由系统自动生成的脚手架(Scaffold)或无效打点记录，会导致模型学到虚假序列规律。\033[0m")

    @staticmethod
    def check_train_test_leakage_and_cold_start(train_df: pd.DataFrame, test_df: pd.DataFrame):
        """
        Train/Test 绝对泄漏与冷启动计算
        注意：此处传入的应当是切分好的 Train / Test DataFrame 或者 Parquet 解开后的序列。
        """
        ic("🔍 分析 Train / Test 集合划分泄漏度...")
        
        def _extract_unique(df_parquet, col_name):
            if col_name not in df_parquet.columns: return set()
            items = []
            for seq in df_parquet[col_name]: items.extend(seq)
            return set(items)

        if 'q_seq' in train_df.columns:
            train_qs = _extract_unique(train_df, 'q_seq')
            test_qs = _extract_unique(test_df, 'q_seq')
            
            if len(test_qs) > 0:
                unseen = test_qs - train_qs
                unseen_ratio = len(unseen) / len(test_qs)
                ic(f"[冷启动题比例] Test集中有 {unseen_ratio:.2%} ({len(unseen)}/{len(test_qs)}) 的题目在Train集从未出现。")
                if unseen_ratio > 0.05:
                    ic(f"\033[91m🛑 [严重划分警告] 测试集中超过 5% 的题目对模型是完全瞎猜状态。你的切分方法或者数据量有问题。\033[0m")

    def analyze_distributions(self):
        """
        分析数据分布并绘图
        """
        # 1. 用户交互长度分布
        user_counts = self.df[StandardColumns.USER_ID].value_counts()
        plt.figure(figsize=(10, 6))
        sns.histplot(user_counts, bins=50, kde=True)
        plt.title('User Interaction Count Distribution')
        plt.xlabel('Number of Interactions')
        plt.ylabel('Count')
        plt.savefig(os.path.join(self.report_dir, 'user_interaction_dist.png'))
        plt.close()
        
        # 2. 题目被做次数分布
        item_counts = self.df[StandardColumns.QUESTION_ID].value_counts()
        plt.figure(figsize=(10, 6))
        sns.histplot(item_counts, bins=50, kde=True)
        plt.title('Question Frequency Distribution')
        plt.xlabel('Frequency')
        plt.ylabel('Count')
        plt.savefig(os.path.join(self.report_dir, 'question_frequency_dist.png'))
        plt.close()
        
        ic(f"分布图已保存至: {self.report_dir}")

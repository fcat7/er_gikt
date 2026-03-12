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
        分析数据分布并生成高阶诊断绘图 (照妖镜级别)
        """
        ic("🎨 正在生成深度数据诊断图表...")
        sns.set_theme(style="whitegrid")
        
        # ==========================================
        # 1. 基础分布: 用户交互长度 & 题目频率
        # ==========================================
        user_counts = self.df[StandardColumns.USER_ID].value_counts()
        plt.figure(figsize=(10, 5))
        sns.histplot(user_counts, bins=50, kde=True, color="skyblue")
        plt.title('User Interaction Count Distribution')
        plt.xlabel('Number of Interactions per User')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(self.report_dir, '01_user_interaction_dist.png'))
        plt.close()
        
        # ==========================================
        # 2. 图二：长尾效应帕累托图 (Sparsity Pareto Curve)
        #  实现逻辑：对题目的交互热度进行降序排列，求出前 X% 题目吃掉了累计多少比例的交互流量。
        # 图表特征：横轴为题目百位比，纵轴为累积交互百分比。我们标记出了典型的 80/20 分界图。
        # 业务意义：如果图上的曲线非常陡峭（比如前 5% 的题目吃掉了 90% 的交互），此时模型就算 AUC 再高，也代表这其实是一个“纯猜高人气题目”的无脑模型，而不是在做真正的知识追踪。同时这图能帮你论证 “K-Core=5 过滤机制”引入的必要性。
        # ==========================================
        item_counts = self.df[StandardColumns.QUESTION_ID].value_counts().sort_values(ascending=False)
        cumulative_pct = item_counts.cumsum() / item_counts.sum() * 100
        x_pct = np.arange(1, len(item_counts) + 1) / len(item_counts) * 100
        
        plt.figure(figsize=(10, 5))
        plt.plot(x_pct, cumulative_pct, linewidth=2.5, color='#ff7f0e')
        plt.axhline(80, color='r', linestyle='--', alpha=0.5, label='80% Interactions')
        plt.axvline(20, color='b', linestyle='--', alpha=0.5, label='20% Questions')
        plt.fill_between(x_pct, cumulative_pct, alpha=0.2, color='#ff7f0e')
        plt.title('Sparsity Pareto Curve (Question Long-tail)')
        plt.xlabel('Percentage of Questions (%) - Sorted by Popularity')
        plt.ylabel('Cumulative Interactions (%)')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.report_dir, '02_sparsity_pareto_curve.png'))
        plt.close()

        # ==========================================
        # 3. 图三：两次交互时间间隔对数分布 (Log-Time Interval)
        # 实现逻辑：计算同一用户答题的严格为正的时间差 diff，映射到 log 空间（因为答题时间的分布可能是跨度数个小时到几天的肥尾分布）。
        # 图表特征：通过 KDE（核心密度估计）画出时间差频率的分布，并在图中打上了两根警戒线 —— 3 Seconds 的红色警报线和 1 Minute 的绿色代表线。
        # 业务意义：正常的答题分布应该是单峰（围绕在几十秒到几分钟之间）。如果在 log10(X)<0.47（即小于3秒）的区域出现了一个明显的“死峭侧峰”，这意味着严重的数据造假！这里实锤了有系统级脚手架自动填充、浏览器刷单等机器操作。
        # ==========================================
        if StandardColumns.TIMESTAMP in self.df.columns:
            df_time = self.df[[StandardColumns.USER_ID, StandardColumns.TIMESTAMP]].copy()
            df_time[StandardColumns.TIMESTAMP] = pd.to_numeric(df_time[StandardColumns.TIMESTAMP], errors='coerce')
            df_time = df_time.dropna(subset=[StandardColumns.TIMESTAMP]).sort_values([StandardColumns.USER_ID, StandardColumns.TIMESTAMP])
            
            time_diffs = df_time.groupby(StandardColumns.USER_ID)[StandardColumns.TIMESTAMP].diff().dropna()
            # 仅保留严格大于0的时间差
            time_diffs_pos = time_diffs[time_diffs > 0]
            
            if not time_diffs_pos.empty:
                # 转换为以 10 为底的对数 (秒)
                log_diffs = np.log10(time_diffs_pos)
                
                plt.figure(figsize=(10, 5))
                sns.histplot(log_diffs, bins=100, kde=True, color='purple')
                
                # 标记异常时间线: 3秒 (log10(3) ≈ 0.477)
                plt.axvline(np.log10(3), color='red', linestyle='--', linewidth=2, label='3 Seconds (Scaffold / Bot Danger Zone)')
                plt.axvline(np.log10(60), color='green', linestyle=':', label='1 Minute (Normal Bound)')
                
                plt.title('Log-Time Interval Histogram')
                plt.xlabel('Log10 (Time Interval in Seconds)')
                plt.ylabel('Interaction Transitions Count')
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(self.report_dir, '03_log_time_interval_hist.png'))
                plt.close()

        # ==========================================
        # 4. 图四：序列位置与正确率关系图 (Accuracy Over Sequence Position)
        # 实现逻辑：按照时间对用户序列进行严格顺排序。提取每个学生的第 1 题至第 N 题。算整个数据集上，处于“第 1 题阶段”的整体正确率、“第 10 题阶段”的正确率。
        # 图表特征：描绘一条随 Sequence Position 递进的准确率折线，并拟合了一条学习率趋势线（一元线性回归极线，带有 Slope 斜率）。并且过滤了某些尾部因为只有十几个人答到（极易被 100% 拉高）的过度抖动位。
        # 模型斜率意义：
        # 正常情况应呈现一个比较平缓的向上斜率曲线（代表能力提升）。
        # **左侧截断检查：**如果“序列中第一题”的正确率异常得高甚至达到颠峰（比如接近 70%），这说明数据集里记录的第一题并不是学生的“破冰题”，说明存在前序交互被强行裁切丢弃带入了极其严重的冷启动漏检（Cold Start Leakage）。
        # ==========================================
        if StandardColumns.LABEL in self.df.columns:
            df_seq = self.df.copy()
            if StandardColumns.TIMESTAMP in df_seq.columns:
                df_seq[StandardColumns.TIMESTAMP] = pd.to_numeric(df_seq[StandardColumns.TIMESTAMP], errors='coerce')
                df_seq = df_seq.sort_values([StandardColumns.USER_ID, StandardColumns.TIMESTAMP])
                
            # 计算每条记录在用户历史中的序列位置
            df_seq['seq_pos'] = df_seq.groupby(StandardColumns.USER_ID).cumcount() + 1
            
            # 为了防止分布极长尾导致大幅波动，截断绘制到 95% 分位数长度 或 最小保留到 30
            max_pos = int(df_seq['seq_pos'].quantile(0.95))
            max_pos = max(max_pos, 30) 
            
            df_seq_filtered = df_seq[df_seq['seq_pos'] <= max_pos]
            
            # 计算每个位置的平均正确率和样本量
            acc_by_pos = df_seq_filtered.groupby('seq_pos')[StandardColumns.LABEL].mean()
            counts_by_pos = df_seq_filtered.groupby('seq_pos')[StandardColumns.LABEL].count()
            
            # 过滤掉偶然性极高（该位置作答用户少于20人）的数据点
            valid_pos = counts_by_pos[counts_by_pos > 20].index
            acc_by_pos = acc_by_pos.loc[valid_pos]
            
            if not acc_by_pos.empty:
                plt.figure(figsize=(12, 5))
                plt.plot(acc_by_pos.index, acc_by_pos.values, marker='o', markersize=3, linestyle='-', alpha=0.6, color='#2ca02c')
                
                # 拟合一条学习趋势线 (一元线性)
                if len(acc_by_pos) > 1:
                    z = np.polyfit(acc_by_pos.index, acc_by_pos.values, 1)
                    p = np.poly1d(z)
                    plt.plot(acc_by_pos.index, p(acc_by_pos.index), "r--", linewidth=2.5, alpha=0.8, label=f"Learning Trend (Slope: {z[0]:.5f})")
                    
                plt.title('Accuracy Over Sequence Position (Learning Curve Indicator)')
                plt.xlabel('Sequence Position (N-th Question Attempted)')
                plt.ylabel('Average Global Accuracy')
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(self.report_dir, '04_accuracy_seq_position.png'))
                plt.close()

        ic(f"高阶诊断图表(照妖镜)已生成并保存至: {self.report_dir}")

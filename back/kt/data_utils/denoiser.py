import pandas as pd
from .standard_columns import StandardColumns

try:
    from icecream import ic
except ImportError:
    ic = print

class KTDataDenoiser:
    """
    负责执行数据管线的“清洗/去噪”作业，从逻辑、网络结构、时序上剔除脏数据。
    支持在数据处理主流程中动态开闭。
    """
    @staticmethod
    def filter_extreme_acc(df: pd.DataFrame, min_attempts: int = 10) -> pd.DataFrame:
        """
        应对“逻辑异常警告”：对于作答次数达到规模，但全部答对或全部答错的题目进行抛弃（如系统测试用例、引导点击）
        """
        ic(f"--> 执行异常正确率过滤 (阀值={min_attempts}次作答)...")
        q_stats = df.groupby(StandardColumns.QUESTION_ID).agg(
            cnt=(StandardColumns.LABEL, 'count'),
            acc=(StandardColumns.LABEL, 'mean')
        )
        bad_qs = q_stats[(q_stats['cnt'] >= min_attempts) & ((q_stats['acc'] == 1.0) | (q_stats['acc'] == 0.0))].index
        
        if len(bad_qs) > 0:
            ic(f"    * 识别并移除 {len(bad_qs)} 道全对/全错逻辑异常题目")
            df = df[~df[StandardColumns.QUESTION_ID].isin(bad_qs)].copy()
        return df

    @staticmethod
    def filter_rapid_actions(df: pd.DataFrame, threshold_sec: float = 3.0) -> pd.DataFrame:
        """
        应对“时间序列泄漏”：当同一用户连续答题时间差极短时，抛弃后续多余动作（合并），防脚手架记忆。
        """
        ic(f"--> 执行时序脚手架/连答过滤 (瞬间动作判定阀值={threshold_sec}秒)...")
        if StandardColumns.TIMESTAMP not in df.columns:
            ic("    * 缺乏时间戳信息，策略略过")
            return df
            
        df[StandardColumns.TIMESTAMP] = pd.to_numeric(df[StandardColumns.TIMESTAMP], errors='coerce')
        # 必须排序后才能计算diff
        df = df.sort_values([StandardColumns.USER_ID, StandardColumns.TIMESTAMP])
        
        time_diffs = df.groupby(StandardColumns.USER_ID)[StandardColumns.TIMESTAMP].diff()
        # 保留第一次交互（diff为NaN）以及时间差达到阈值的记录
        mask = (time_diffs >= threshold_sec) | time_diffs.isna()
        
        dropped_n = (~mask).sum()
        if dropped_n > 0:
            ic(f"    * 移除 {dropped_n} 条瞬间连答动作记录 (斩断时间泄漏)")
            df = df[mask].copy()
        return df

    @staticmethod
    def k_core_pruning(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
        """
        应对“稀疏警告”：K-Core 过滤
        迭代过滤少于 k 次作答的题目和用户，直到不存在长尾游离节点。消除 Embedding 孤岛并增强图密度。
        """
        ic(f"--> 执行稠密网络 K-Core 计算裁剪 (K={k})...")
        iteration = 0
        while True:
            iteration += 1
            start_len = len(df)
            
            # 过滤题目网络
            q_counts = df[StandardColumns.QUESTION_ID].value_counts()
            valid_q = q_counts[q_counts >= k].index
            df = df[df[StandardColumns.QUESTION_ID].isin(valid_q)]
            
            # 过滤用户网络
            u_counts = df[StandardColumns.USER_ID].value_counts()
            valid_u = u_counts[u_counts >= k].index
            df = df[df[StandardColumns.USER_ID].isin(valid_u)]
            
            # 图达到稳态（本轮无任何抛弃）
            if len(df) == start_len:
                break
        
        ic(f"    * K-Core 清洗经过 {iteration} 轮震荡达到网络稳态，现存 {len(df)} 条高质量边")
        return df

    @classmethod
    def run_denoise_pipeline(cls, df: pd.DataFrame, k_core: int=5, extreme_acc_min: int=10, rapid_merge_sec: float=3.0) -> pd.DataFrame:
        ic("=== 🔧 启动数据深度清洗与去噪管线 (Denoiser Pipeline) ===")
        start_len = len(df)
        
        if rapid_merge_sec > 0:
            df = cls.filter_rapid_actions(df, rapid_merge_sec)
            
        if extreme_acc_min > 0:
            df = cls.filter_extreme_acc(df, extreme_acc_min)
            
        if k_core > 0:
            df = cls.k_core_pruning(df, k_core)
            
        dropped_total = start_len - len(df)
        ic(f"=== ✅ 去噪管线结束: 总量 {start_len} -> {len(df)} (剔除了 {dropped_total} 条无效脏记录) ===")
        return df

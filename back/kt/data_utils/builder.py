
import os
import numpy as np
import pandas as pd
from scipy import sparse
try:
    from icecream import ic
except ImportError:
    ic = print

class KTDataBuilder:
    """
    数据构建器
    负责将 DataFrame 转换为模型所需的 .npy / .npz 格式
    包括：ID映射、序列生成、邻接矩阵构建
    """
    def __init__(self, config):
        self.config = config
        self.output_dir = config.PROCESSED_DATA_DIR
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def build_dataset(self, df, dataset_type='train'):
        """
        构建数据集
        Args:
            df: 输入 DataFrame
            dataset_type: 'train' or 'test' or 'full' (用于文件命名区分，如果需要)
        """
        ic("开始构建数据集...")
        
        # 1. 构建并保存映射字典 (ID Maps)
        # 注意：通常我们应该基于全量数据构建 ID 映射，否则测试集可能会有未知 ID
        # 这里假设传入的 df 包含了所有可能的 ID (或者我们在外部先构建好 ID Map)
        self._build_id_maps(df)
        
        # 2. 构建邻接矩阵 (Adjacency Matrices)
        self._build_adjacency_matrices(df)
        
        # 3. 构建序列数据 (Sequences)
        self._build_sequences(df)
        
        # 4. 构建问题特征 (Question Features)
        self._build_q_features(df)
        
        ic("数据集构建完成！")

    def _parse_skills(self, raw_skill):
        """解析技能ID，支持 int, float, str (逗号或下划线分隔)"""
        if pd.isna(raw_skill):
            return []
        if isinstance(raw_skill, (int, float)):
            return [int(raw_skill)]
        
        # 字符串处理
        s = str(raw_skill)
        parts = s.split('_')
        
        skills = []
        for p in parts:
            try:
                skills.append(int(p))
            except ValueError:
                pass
        return skills

    def _build_id_maps(self, df):
        """构建 ID 映射字典"""
        # 获取所有唯一值
        users = df['user_id'].unique()
        problems = df['problem_id'].unique()
        
        # 处理技能 ID (可能是 '123' 或 '123_456' 或 '123,456')
        skills = set()
        for s in df['skill_id'].dropna().unique():
            parsed = self._parse_skills(s)
            skills.update(parsed)
        skills = sorted(list(skills))
        
        # 构建映射
        # problem_id 从 1 开始 (0 用于 padding)
        self.problem2idx = {problem: i + 1 for i, problem in enumerate(problems)}
        self.problem2idx[0] = 0 # Padding
        
        self.skill2idx = {skill: i for i, skill in enumerate(skills)}
        self.user2idx = {user: i for i, user in enumerate(users)}
        
        # 保存
        np.save(os.path.join(self.output_dir, 'question2idx.npy'), self.problem2idx)
        np.save(os.path.join(self.output_dir, 'skill2idx.npy'), self.skill2idx)
        np.save(os.path.join(self.output_dir, 'user2idx.npy'), self.user2idx)
        
        # 反向映射
        self.idx2problem = {v: k for k, v in self.problem2idx.items()}
        self.idx2skill = {v: k for k, v in self.skill2idx.items()}
        self.idx2user = {v: k for k, v in self.user2idx.items()}
        
        np.save(os.path.join(self.output_dir, 'idx2question.npy'), self.idx2problem)
        np.save(os.path.join(self.output_dir, 'idx2skill.npy'), self.idx2skill)
        np.save(os.path.join(self.output_dir, 'idx2user.npy'), self.idx2user)
        
        ic(f"ID 映射构建完成: Users={len(users)}, Problems={len(problems)}, Skills={len(skills)}")

    def _build_adjacency_matrices(self, df):
        """构建 Q-S, Q-Q, S-S 邻接矩阵"""
        num_q = len(self.problem2idx) # 包含 0
        num_s = len(self.skill2idx)
        
        qs_table = np.zeros([num_q, num_s], dtype=int)
        
        # 遍历每个题目，填充 Q-S 表
        # 为了效率，我们先去重
        problem_skill_df = df[['problem_id', 'skill_id']].drop_duplicates()
        
        for row in problem_skill_df.itertuples(index=False):
            q_idx = self.problem2idx.get(row.problem_id)
            if q_idx is None: continue
            
            current_skills = self._parse_skills(row.skill_id)
            
            for s in current_skills:
                if s in self.skill2idx:
                    s_idx = self.skill2idx[s]
                    qs_table[q_idx, s_idx] = 1
                    
        # 计算 Q-Q 和 S-S
        qq_table = np.matmul(qs_table, qs_table.T)
        ss_table = np.matmul(qs_table.T, qs_table)
        
        # 保存为稀疏矩阵
        sparse.save_npz(os.path.join(self.output_dir, 'qs_table.npz'), sparse.coo_matrix(qs_table))
        sparse.save_npz(os.path.join(self.output_dir, 'qq_table.npz'), sparse.coo_matrix(qq_table))
        sparse.save_npz(os.path.join(self.output_dir, 'ss_table.npz'), sparse.coo_matrix(ss_table))
        
        ic("邻接矩阵构建完成")

    def _build_sequences(self, df):
        """构建用户交互序列"""
        num_users = len(self.user2idx)
        max_len = self.config.MAX_SEQ_LEN
        
        user_seq = np.zeros([num_users, max_len], dtype=int)
        user_res = np.zeros([num_users, max_len], dtype=int)
        user_mask = np.zeros([num_users, max_len], dtype=int)
        
        # 预计算平均作答时间 (用于归一化)
        if 'response_time' in df.columns:
            # @fix_fzq 2026-01-08: 增强版数据清洗 - 防止异常值污染基准统计
            # 1. 强制转为数值，无法转换的变为 NaN
            df['response_time'] = pd.to_numeric(df['response_time'], errors='coerce')
            
            # 2. 定义有效时间范围用于统计 (单位ms)
            # 排除 <=0 的错误数据
            # 排除 > 3600000 (1小时) 的挂机/异常数据，防止拉大平均值
            valid_rt_mask = (df['response_time'] > 0) & (df['response_time'] < 3600 * 1000)
            clean_df = df[valid_rt_mask]
            
            # 3. 使用中位数 (median) 代替平均值，对异常值更鲁棒
            problem_avg_time = clean_df.groupby('problem_id')['response_time'].median().to_dict()
            
            user_interval_time = np.zeros([num_users, max_len], dtype=float)
            user_response_time = np.zeros([num_users, max_len], dtype=float)
        else:
            problem_avg_time = {}
            user_interval_time = None
            user_response_time = None

        # 记录每个用户每个技能上次出现的 order_id (用于计算 interval)
        # user_last_skill_time = {user_idx: {skill_idx: last_order_id}}
        user_last_skill_time = {}

        # 按用户分组处理
        # 必须保证 df 已经按 timestamp 排序
        grouped = df.groupby('user_id')
        
        for user_id, group in grouped:
            u_idx = self.user2idx.get(user_id)
            if u_idx is None: continue
            
            # 初始化该用户的技能时间记录
            if u_idx not in user_last_skill_time:
                user_last_skill_time[u_idx] = {}

            # 截取最后 MAX_SEQ_LEN 个交互 (或者最开始，取决于需求，通常是最后)
            # 这里我们采取截断策略：如果超过 MAX_SEQ_LEN，取后段；如果不足，前面补0 (已经在初始化时全0了)
            # 但 GIKT 原代码逻辑似乎是：填满为止，超过截断。
            # 我们采用：取前 MAX_SEQ_LEN 个 (因为通常按时间排序，前面的也是历史)
            # 或者：滑动窗口？
            # 参照原代码：if num_seq < MAX_SEQ_LEN - 1
            
            interactions = group.head(max_len) # 取前 max_len 个
            
            for i, row in enumerate(interactions.itertuples()):
                q_idx = self.problem2idx.get(row.problem_id, 0)
                user_seq[u_idx, i] = q_idx
                user_res[u_idx, i] = int(row.correct)
                user_mask[u_idx, i] = 1
                
                # 计算时间特征
                if user_interval_time is not None:
                    # --- 1. 计算 Interval Time (距离上次同技能练习的间隔) ---
                    # @fix_fzq 2026-01-08: 对 timestamp 进行清洗
                    current_timestamp = float(row.timestamp) if pd.notna(row.timestamp) else 0.0
                    if current_timestamp < 0: current_timestamp = 0.0 # 修正负时间戳

                    # 解析当前题目的技能
                    curr_skills = []
                    raw_skill = row.skill_id
                    if pd.notna(raw_skill):
                        parsed_skills = self._parse_skills(raw_skill)
                        for s_id_raw in parsed_skills:
                            s_id = self.skill2idx.get(s_id_raw)
                            if s_id is not None: curr_skills.append(s_id)
                    
                    # 计算间隔 (取所有相关技能间隔的平均值，如果没见过则间隔为0)
                    intervals = []
                    for s_idx in curr_skills:
                        last_time = user_last_skill_time[u_idx].get(s_idx, current_timestamp) # 没见过则认为就在刚刚(间隔0)
                        diff = current_timestamp - last_time
                        if diff < 0: diff = 0.0 # 防止时间倒流
                        intervals.append(diff)
                        # 更新该技能的最后时间
                        user_last_skill_time[u_idx][s_idx] = current_timestamp
                    
                    avg_interval = sum(intervals) / len(intervals) if intervals else 0
                    # 对间隔进行 Log 归一化: log(interval + 1)
                    # 并使用 tanh 压缩到 [0, 1) 区间
                    val_log = np.log(avg_interval + 1)
                    user_interval_time[u_idx, i] = np.tanh(val_log)

                    # --- 2. 计算 Response Time (归一化作答时间) ---
                    # RT_norm = RT_user / RT_avg_problem
                    # @fix_fzq 2026-01-08: 增加对单行 response_time 的清洗
                    avg_t = problem_avg_time.get(row.problem_id, 1.0)
                    if pd.isna(avg_t) or avg_t <= 0: avg_t = 1.0
                    
                    raw_rt = row.response_time
                    # 检查是否为 NaN 或者 负数 或者 极大值(此时截断为 avg_t * 10)
                    if pd.isna(raw_rt) or raw_rt < 0:
                        raw_rt = avg_t # 用平均值填充缺失数据，或者填0? 这里选平均值(中性)或0
                        # 考虑到 GIKT 认知门，填0表示“瞬间作答/无思考”，填 avg_t 表示“普通作答”
                        # 填 0 比较安全，避免引入不该有的权重
                        raw_rt = 0.0 
                    
                    rt_norm = raw_rt / avg_t
                    rt_norm = min(rt_norm, 10.0) # 截断异常值
                    user_response_time[u_idx, i] = np.tanh(rt_norm)
        
        np.save(os.path.join(self.output_dir, 'user_seq.npy'), user_seq)
        np.save(os.path.join(self.output_dir, 'user_res.npy'), user_res)
        np.save(os.path.join(self.output_dir, 'user_mask.npy'), user_mask)
        
        if user_interval_time is not None:
            np.save(os.path.join(self.output_dir, 'user_interval_time.npy'), user_interval_time)
            np.save(os.path.join(self.output_dir, 'user_response_time.npy'), user_response_time)
            
        ic("序列数据构建完成")

    def _build_q_features(self, df):
        """构建问题特征向量"""
        # 简单实现，仅占位
        pass

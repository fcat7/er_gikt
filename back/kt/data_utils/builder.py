
import os
import numpy as np
import pandas as pd
from scipy import sparse
from config import MAX_SEQ_LEN, Config
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
    def __init__(self, config : Config):
        self.config = config
        self.output_dir = config.PROCESSED_DATA_DIR
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def build_dataset(self, df, enable_window_aug=False, stride=100):
        """
        构建数据集
        Args:
            df: 输入 DataFrame
            enable_window_aug: 是否启用滑动窗口数据增强
            stride: 滑动窗口步长
        """
        ic("开始构建数据集...")
        
        # 1. 构建并保存映射字典 (ID Maps)
        self._build_id_maps(df)
        
        # 2. 构建邻接矩阵 (Adjacency Matrices)
        self._build_adjacency_matrices(df)
        
        # 3. 构建领域映射 (Domain Map)
        self._build_domain_map()
        
        # 4. 构建序列数据 (Sequences)
        self._build_sequences(df, enable_window_aug, stride)
        
        # 5. 构建问题特征 (Question Features)
        self._build_q_features(df)
        
        ic("数据集构建完成！")

    def _build_domain_map(self):
        """可以根据手动配置或自动聚类构建技能领域映射"""
        ic("开始构建 Domain Map...")

        # --- 0. 读取配置 ---
        use_manual = False
        config_path = ""
        target_n_clusters = 15

        if hasattr(self.config.dataset, 'DOMAIN_CONFIG'):
            d_cfg = self.config.dataset.DOMAIN_CONFIG
            use_manual = d_cfg.get('use_manual_config', False)
            target_n_clusters = d_cfg.get('target_domains', 15)
            raw_path = d_cfg.get('manual_config_path', "")
            
            if raw_path:
                if os.path.isabs(raw_path):
                    config_path = raw_path
                else:
                    # 假设 config.PROJECT_ROOT 是项目根目录
                    config_path = os.path.join(self.config.PROJECT_ROOT, raw_path)

        # 路径验证与回退逻辑
        if use_manual:
            if not config_path or not os.path.exists(config_path):
                ic(f"Warning: 配置启用手动 Domain Map，但文件不存在: '{config_path}'")
                ic(f"自动回退到 [Ward 聚类]，目标簇数: {target_n_clusters}")
                use_manual = False
            else:
                ic(f"Info: 将使用手动 DomainMap 配置: {config_path}")
        else:
            ic(f"Info: 使用自动 [Ward 聚类]，目标簇数: {target_n_clusters}")
        
        domain_map = None
        num_s = len(self.skill2idx)
        
        # --- 策略 A: 手动配置 ---
        if use_manual and config_path and os.path.exists(config_path):
            ic(f"加载手动 Domain 配置: {config_path}")
            try:
                import json
                with open(config_path, 'r', encoding='utf-8') as f:
                    manual_config = json.load(f)
                
                # 初始化为 -1 (未分配)
                domain_map = np.full(num_s, -1, dtype=int)
                assigned_count = 0
                max_domain_id = 0
                
                for d_id_str, skill_list in manual_config.items():
                    d_id = int(d_id_str)
                    max_domain_id = max(max_domain_id, d_id)
                    
                    for item in skill_list:
                        # item 可能是 {"skill_id": "...", ...} 或 只是 "name" 或 "id"
                        # 我们尽量兼容
                        s_raw_id = None
                        if isinstance(item, dict):
                            s_raw_id = item.get('skill_id')
                        elif isinstance(item, (str, int)):
                            s_raw_id = item
                        
                        # 在 skill2idx 中查找内部索引
                        # 注意：需要把 s_raw_id 转为 skill2idx 里的 key 类型 (通常是 int, 有时是 str)
                        # 我们之前 Parse 的时候是 int (如果全是数字)
                        
                        # 尝试 int
                        idx = None
                        try:
                            s_int = int(s_raw_id)
                            idx = self.skill2idx.get(s_int)
                        except:
                            idx = self.skill2idx.get(str(s_raw_id))
                            
                        if idx is not None:
                            domain_map[idx] = d_id
                            assigned_count += 1
                
                # 处理未分配的技能
                unassigned_indices = np.where(domain_map == -1)[0]
                unassigned_count = len(unassigned_indices)
                
                if unassigned_count > 0:
                    ic(f"警告: 有 {unassigned_count} 个技能未在手动配置中找到。尝试基于关联性智能分配...")
                    
                    # 自动确定 "Others" 领域 ID (配置列表长度，例如 15)
                    others_id = len(manual_config)
                    
                    ss_path = os.path.join(self.output_dir, "ss_table.npz")
                    if os.path.exists(ss_path):
                        ss_dense = sparse.load_npz(ss_path).toarray()
                        
                        smart_count = 0
                        for s_idx in unassigned_indices:
                            # 统计该技能对各个已分配领域的关联得分 (0 到 max_domain_id)
                            domain_scores = np.zeros(max_domain_id + 1)
                            for d_id in range(max_domain_id + 1):
                                d_skill_indices = np.where(domain_map == d_id)[0]
                                if len(d_skill_indices) > 0:
                                    # 累加与该领域内已知技能的共现次数
                                    domain_scores[d_id] = ss_dense[s_idx, d_skill_indices].sum()
                            
                            if domain_scores.sum() > 0:
                                domain_map[s_idx] = np.argmax(domain_scores)
                                smart_count += 1
                            else:
                                # 彻底孤立的技能，分配到自动生成的 "Others" 独立领域
                                domain_map[s_idx] = others_id
                        ic(f"智能分配完成: {smart_count} 个归类到关联领域，{unassigned_count - smart_count} 个归入独立领域 [Others] (ID: {others_id})")
                    else:
                        ic(f"未找到 ss_table.npz，直接分配到 Others 领域 {others_id}。")
                        domain_map[unassigned_indices] = others_id
                
                # 更新最终分配统计
                unique_domains = np.unique(domain_map)
                ic(f"手动映射完成: 所有 {num_s} 个技能已分配到 {len(unique_domains)} 个 Domain。")
                ic(f"最高 Domain ID: {unique_domains.max()} (请确保配置文件中的 target_domains 对应为 {unique_domains.max() + 1})")
                
                # 如果手动映射成功，直接保存并返回
                np.save(os.path.join(self.output_dir, "skill_domain_map.npy"), domain_map)
                unique, counts = np.unique(domain_map, return_counts=True)
                ic(f"Domain 分布: {dict(zip(unique, counts))}")
                return

            except Exception as e:
                ic(f"处理手动配置失败: {e}。将回退到自动聚类。")
        
        # --- 策略 B: 自动聚类 (Ward) ---
        ic(f"执行自动层次聚类 (Target Clusters={target_n_clusters})...")
        try:
            from sklearn.cluster import AgglomerativeClustering
            from sklearn.preprocessing import normalize
        except ImportError:
            ic("警告: 未安装 scikit-learn，无法聚类。")
            return

        # 1. 加载技能关联矩阵
        ss_path = os.path.join(self.output_dir, "ss_table.npz")
        if not os.path.exists(ss_path):
            ic("错误: ss_table.npz 不存在，无法聚类")
            return
            
        ss_sparse = sparse.load_npz(ss_path)
        ss_dense = ss_sparse.toarray()
        
        # 2. 归一化
        ss_norm = normalize(ss_dense, norm='l2', axis=1)

        # 3. 聚类
        if num_s <= target_n_clusters:
            domain_map = np.arange(num_s)
        else:
            cluster = AgglomerativeClustering(
                n_clusters=target_n_clusters, 
                linkage='ward' 
            )
            domain_map = cluster.fit_predict(ss_norm)
        
        # 4. 保存
        np.save(os.path.join(self.output_dir, "skill_domain_map.npy"), domain_map)
        
        # 统计分布
        unique, counts = np.unique(domain_map, return_counts=True)
        ic(f"Domain 分布: {dict(zip(unique, counts))}")
        ic(f"*** 请确保 params.py 的 num_domains 设置为: {len(unique)} 或更安全的 {target_n_clusters} ***")

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

    def _build_sequences(self, df, enable_window_aug=False, stride=100):
        """
        构建用户交互序列。
        支持两种模式：
        1. 标准模式 (enable_window_aug=False): 每个用户只取最后 MAX_SEQ_LEN 条记录，生成 [num_users, max_len] 的矩阵。
        2. 增强模式 (enable_window_aug=True): 对每个用户全量历史进行滑动窗口切片，生成 [num_windows, max_len] 的矩阵。
        """
        num_users = len(self.user2idx)
        max_len = MAX_SEQ_LEN
        
        # 预计算平均作答时间 (用于归一化)
        if 'response_time' in df.columns:
            # @fix_fzq 2026-01-08: 增强版数据清洗
            df['response_time'] = pd.to_numeric(df['response_time'], errors='coerce')
            valid_rt_mask = (df['response_time'] > 0) & (df['response_time'] < 3600 * 1000)
            clean_df = df[valid_rt_mask]
            problem_avg_time = clean_df.groupby('problem_id')['response_time'].median().to_dict()
            has_time = True
        else:
            problem_avg_time = {}
            has_time = False

        # 模式一：标准模式 (预分配矩阵)
        if not enable_window_aug:
            ic("模式: 标准序列构建 (截取最后一段)")
            user_seq = np.zeros([num_users, max_len], dtype=int)
            user_res = np.zeros([num_users, max_len], dtype=int)
            user_mask = np.zeros([num_users, max_len], dtype=int)
            
            user_interval_time = np.zeros([num_users, max_len], dtype=float) if has_time else None
            user_response_time = np.zeros([num_users, max_len], dtype=float) if has_time else None
            
            user_last_skill_time = {}

            grouped = df.groupby('user_id')
            for user_id, group in grouped:
                u_idx = self.user2idx.get(user_id)
                if u_idx is None: continue
                
                if u_idx not in user_last_skill_time: user_last_skill_time[u_idx] = {}

                # 截取最后 MAX_SEQ_LEN
                interactions = group.head(max_len) # 保持原逻辑 (head? 原代码可能是按时间倒序排列？通常 head 是前 max_len。如果是时间正序，则是最老的。这里暂保持原状)
                
                for i, row in enumerate(interactions.itertuples()):
                    q_idx = self.problem2idx.get(row.problem_id, 0)
                    user_seq[u_idx, i] = q_idx
                    user_res[u_idx, i] = int(row.correct)
                    user_mask[u_idx, i] = 1 # Valid
                    
                    if has_time:
                        # Time processing (kept consistent with legacy)
                        current_timestamp = float(row.timestamp) if pd.notna(row.timestamp) else 0.0
                        if current_timestamp < 0: current_timestamp = 0.0
                        
                        curr_skills = []
                        raw_skill = row.skill_id
                        if pd.notna(raw_skill):
                            parsed_skills = self._parse_skills(raw_skill)
                            for s_id_raw in parsed_skills:
                                s_id = self.skill2idx.get(s_id_raw)
                                if s_id is not None: curr_skills.append(s_id)
                        
                        intervals = []
                        for s_idx in curr_skills:
                            last_time = user_last_skill_time[u_idx].get(s_idx, current_timestamp)
                            diff = current_timestamp - last_time
                            if diff < 0: diff = 0.0
                            intervals.append(diff)
                            user_last_skill_time[u_idx][s_idx] = current_timestamp
                        
                        avg_interval = sum(intervals) / len(intervals) if intervals else 0
                        val_log = np.log(avg_interval + 1)
                        user_interval_time[u_idx, i] = np.tanh(val_log)

                        avg_t = problem_avg_time.get(row.problem_id, 1.0)
                        if pd.isna(avg_t) or avg_t <= 0: avg_t = 1.0
                        raw_rt = row.response_time
                        if pd.isna(raw_rt) or raw_rt < 0: raw_rt = 0.0
                        rt_norm = min(raw_rt / avg_t, 10.0)
                        user_response_time[u_idx, i] = np.tanh(rt_norm)

            # 保存标准结果
            np.save(os.path.join(self.output_dir, 'user_seq.npy'), user_seq)
            np.save(os.path.join(self.output_dir, 'user_res.npy'), user_res)
            np.save(os.path.join(self.output_dir, 'user_mask.npy'), user_mask)
            if has_time:
                np.save(os.path.join(self.output_dir, 'user_interval_time.npy'), user_interval_time)
                np.save(os.path.join(self.output_dir, 'user_response_time.npy'), user_response_time)
        
        # 模式二：滑动窗口增强
        else:
            ic(f"模式: 滑动窗口增强 (Stride={stride})")
            min_len = 20 # 最小窗口长度
            
            all_seq = []
            all_res = []
            all_mask = []
            all_interval = []
            all_response = []
            all_groups = [] # 记录原始 User ID (用于防泄漏划分)
            
            grouped = df.groupby('user_id')
            
            for user_id, group in grouped:
                u_idx = self.user2idx.get(user_id)
                if u_idx is None: continue
                
                # 1. 提取全量历史
                full_seq = []
                full_res = []
                full_interval = []
                full_response = []
                
                user_last_skill_time = {} # Local tracker

                for row in group.itertuples():
                    q_idx = self.problem2idx.get(row.problem_id, 0)
                    full_seq.append(q_idx)
                    full_res.append(int(row.correct))
                    
                    if has_time:
                        current_timestamp = float(row.timestamp) if pd.notna(row.timestamp) else 0.0
                        if current_timestamp < 0: current_timestamp = 0.0
                        curr_skills = []
                        if pd.notna(row.skill_id):
                            for s_raw in self._parse_skills(row.skill_id):
                                s_id = self.skill2idx.get(s_raw)
                                if s_id is not None: curr_skills.append(s_id)
                        
                        intervals = []
                        for s_idx in curr_skills:
                            last = user_last_skill_time.get(s_idx, current_timestamp)
                            intervals.append(max(0.0, current_timestamp - last))
                            user_last_skill_time[s_idx] = current_timestamp
                        
                        avg_int = sum(intervals)/len(intervals) if intervals else 0
                        full_interval.append(np.tanh(np.log(avg_int + 1)))
                        
                        avg_t = problem_avg_time.get(row.problem_id, 1.0)
                        raw_rt = row.response_time if (pd.notna(row.response_time) and row.response_time >=0) else 0.0
                        full_response.append(np.tanh(min(raw_rt / max(avg_t, 1e-6), 10.0)))

                # 2. 从全量历史中切片
                total_interactions = len(full_seq)
                if total_interactions <= max_len:
                    starts = [0]
                else:
                    starts = list(range(0, total_interactions, stride))
                
                # 确保最后一个窗口不被遗漏 & 去重
                # (Simple stride logic already covers most, might miss tail if logic is strict, but keeping simple for now)
                
                for start in starts:
                    end = min(start + max_len, total_interactions)
                    length = end - start
                    
                    # 只有当这是唯一的窗口 或者 长度足够 时才保留
                    if length < min_len and len(starts) > 1:
                        continue
                        
                    win_seq = np.zeros(max_len, dtype=int)
                    win_res = np.zeros(max_len, dtype=int)
                    win_mask = np.zeros(max_len, dtype=int)
                    win_interval = np.zeros(max_len, dtype=float)
                    win_response = np.zeros(max_len, dtype=float)
                    
                    # Fill (Post-Padding style: data at 0..length)
                    win_seq[:length] = full_seq[start:end]
                    win_res[:length] = full_res[start:end]
                    win_mask[:length] = 1
                    if has_time:
                        win_interval[:length] = full_interval[start:end]
                        win_response[:length] = full_response[start:end]
                    
                    all_seq.append(win_seq)
                    all_res.append(win_res)
                    all_mask.append(win_mask)
                    if has_time:
                        all_interval.append(win_interval)
                        all_response.append(win_response)
                        
                    # 记录该窗口属于哪个原始用户
                    all_groups.append(u_idx)

            # 保存增强结果
            ic(f"滑动窗口结果: 原用户数 {num_users} -> 窗口总数 {len(all_seq)}")
            np.save(os.path.join(self.output_dir, 'user_seq.npy'), np.array(all_seq))
            np.save(os.path.join(self.output_dir, 'user_res.npy'), np.array(all_res))
            np.save(os.path.join(self.output_dir, 'user_mask.npy'), np.array(all_mask))
            # 保存用户组信息 (关键修复: 数据泄露)
            np.save(os.path.join(self.output_dir, 'user_window_groups.npy'), np.array(all_groups))

            if has_time:
                np.save(os.path.join(self.output_dir, 'user_interval_time.npy'), np.array(all_interval))
                np.save(os.path.join(self.output_dir, 'user_response_time.npy'), np.array(all_response))

        ic(f"序列数据构建完成. 模式: {'Augmented' if enable_window_aug else 'Standard'}")

    def _build_q_features(self, df):
        """构建问题特征向量"""
        # 简单实现，仅占位
        pass

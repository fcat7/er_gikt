
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

    def build_maps(self, df):
        """
        构建所有静态映射和图结构 (不生成序列)
        这应该在全量数据上运行，以确保 ID 统一。
        """
        ic("开始构建 ID 映射和图结构...")
        
        # 1. 构建并保存映射字典 (ID Maps)
        self._build_id_maps(df)
        
        # 2. 构建邻接矩阵 (Adjacency Matrices)
        self._build_adjacency_matrices(df)
        
        # 3. 构建领域映射 (Domain Map)
        self._build_domain_map(df)
        
        # 4. 构建问题特征 (Question Features)
        self._build_q_features(df)
        
        ic("静态图结构构建完成！")

    def build_sequences(self, df, stride=100, save_prefix="train"):
        """
        构建并保存序列数据
        Args:
            df: 数据子集 (Train 或 Test)
            stride: 滑动窗口步长 (Train用较小值, Test用超大值或None)
            save_prefix: 保存文件的前缀 (e.g. 'train', 'test')
        """
        ic(f"开始构建序列 [{save_prefix}] (Stride={stride})...")
        num_users = len(self.user2idx)
        max_len = MAX_SEQ_LEN
        
        # 预计算平均作答时间
        problem_avg_time = {}
        has_time = False
        if 'response_time' in df.columns:
            # 简单的清洗检查，假设外部已经做过深度清洗
            # 这里的统计应该基于 全局数据 还是 当前分片数据？
            # 理想情况下，平均作答时间等统计量最好来自训练集或由于其无监督性来自全集。
            # 为简单起见，这里只基于当前输入 df 统计，或者忽略差异。
            # 更好的做法是在 build_maps 阶段计算全局统计量。但这里暂且局部计算。
            df['response_time'] = pd.to_numeric(df['response_time'], errors='coerce')
            valid_rt = df[(df['response_time'] > 0) & (df['response_time'] < 3600000)]
            if not valid_rt.empty:
                problem_avg_time = valid_rt.groupby('problem_id')['response_time'].median().to_dict()
                has_time = True

        min_len = 5 # 最小窗口长度 (Train set constraint, Test set might usually be longer)
        
        all_seq = []
        all_res = []
        all_mask = []
        all_eval_mask = []  # ✅ 标记评估区域：1=需要评估, 0=历史上下文（只用于冷启动预热）
        all_interval = []
        all_response = []
        all_groups = [] # 原始 User ID，用于防止数据泄露
        
        grouped = df.groupby('user_id')
        
        count = 0
        for user_id, group in grouped:
            u_idx = self.user2idx.get(user_id)
            if u_idx is None: 
                # 这种情况可能发生：如果该用户在 build_maps 的 df 中不存在 (ID Map没包含他)
                # 但一般流程是 build_maps(full) -> split -> build_seq(subset)，所以只要 keys 也是 int/str 对应即可
                continue
            
            # --- 提取全量历史 ---
            full_seq = []
            full_res = []
            full_interval = []
            full_response = []
            
            user_last_skill_time = {} 

            # 注意：DataFrame 迭代较慢，但比之前的 list comprehension 更清晰
            # 假设 df 已经按 timestamp 排序
            # itertuples 确实快一些
            
            for row in group.itertuples(index=False):
                q_idx = self.problem2idx.get(row.problem_id, 0)
                full_seq.append(q_idx)
                full_res.append(int(row.correct))
                
                if has_time:
                    try:
                        ts = float(row.timestamp) if pd.notna(row.timestamp) else 0.0
                    except: ts = 0.0
                    if ts < 0: ts = 0.0
                    
                    curr_skills = []
                    # 技能解析可能耗时，考虑优化？暂保持原样
                    if hasattr(row, 'skill_id') and pd.notna(row.skill_id):
                        # 这里调用 self._parse_skills 可能会有性能瓶颈
                        # 但如果 builder._parse_skills 足够快...
                        s_raw_list = self._parse_skills(row.skill_id)
                        for s_raw in s_raw_list:
                            s_id = self.skill2idx.get(s_raw)
                            if s_id is not None: curr_skills.append(s_id)
                    
                    intervals = []
                    for s_idx in curr_skills:
                        last = user_last_skill_time.get(s_idx, ts)
                        intervals.append(max(0.0, ts - last))
                        user_last_skill_time[s_idx] = ts
                    
                    avg_int = sum(intervals)/len(intervals) if intervals else 0.0
                    # Tanh-Log scaling
                    full_interval.append(np.tanh(np.log(avg_int + 1)))
                    
                    avg_t = problem_avg_time.get(row.problem_id, 1.0)
                    raw_rt = row.response_time if (hasattr(row, 'response_time') and pd.notna(row.response_time) and row.response_time >=0) else 0.0
                    # Tanh-Norm scaling
                    full_response.append(np.tanh(min(raw_rt / max(avg_t, 1e-6), 10.0)))

            # --- 窗口切分 ---
            total_interactions = len(full_seq)
            if total_interactions == 0: continue

            # 区分训练和测试的切分策略
            is_train_mode = 'train' in save_prefix.lower()
            history_len = 0  # 历史上下文长度（用于标记eval_mask）
            
            if stride is None or stride >= total_interactions:
                # 非重叠模式：区分训练和测试
                if not is_train_mode and total_interactions > max_len:
                    # 测试集超长序列：使用50%重叠，保留历史预热（避免冷启动）
                    # 窗口1: [0-200]，全部评估
                    # 窗口2: [100-300]，前100是历史，后100是新评估数据
                    current_stride = max_len // 2
                    history_len = max_len // 2
                    ic(f"  [Test-Long] 重叠窗口 stride={current_stride}, history={history_len}")
                else:
                    # 短序列或训练集（非重叠）：全部参与评估
                    current_stride = max_len
                    history_len = 0
            else:
                # 重叠模式（stride < total_interactions）
                current_stride = stride
                # 🔧 改进：即使是训练集，如果启用了重叠窗口，也应该标记历史上下文
                # 这样才能公平评估（避免后续窗口被历史数据过度"预热"）
                history_len = max_len // 2 if total_interactions > max_len else 0
                if history_len > 0:
                    ic(f"  [Train-Overlap] stride={current_stride}, history={history_len} (避免后续窗口过度预热)")
            
            starts = list(range(0, total_interactions, current_stride))
            
            for start in starts:
                end = min(start + max_len, total_interactions)
                length = end - start
                
                # 过滤过短序列（Train模式下）
                if is_train_mode and length < min_len and len(starts) > 1:
                    continue
                if length <= 0: 
                    continue

                # ====== 核心修改：生成eval_mask ======
                # eval_mask: 1表示需要计算loss, 0表示只作为历史上下文
                win_eval_mask = np.zeros(max_len, dtype=int)
                
                if history_len > 0 and start > 0:
                    # 这不是第一个窗口，前history_len是历史上下文，不计算loss
                    eval_start = min(history_len, length)  # 防御：确保不超过序列长度
                    eval_end = length
                else:
                    # 第一个窗口或非重叠模式：全部参与评估
                    eval_start = 0
                    eval_end = length
                
                # 防御性检查：确保至少有一些数据被评估
                if eval_end > eval_start and length > 0:
                    win_eval_mask[eval_start:eval_end] = 1
                elif length > 0 and start == 0:
                    # 如果是第一个窗口但没有评估数据，至少标记有效数据为评估
                    win_eval_mask[:length] = 1
                # ====== 核心修改结束 ======

                win_seq = np.zeros(max_len, dtype=int)
                win_res = np.zeros(max_len, dtype=int)
                win_mask = np.zeros(max_len, dtype=int)
                
                win_seq[:length] = full_seq[start:end]
                win_res[:length] = full_res[start:end]
                win_mask[:length] = 1
                
                all_seq.append(win_seq)
                all_res.append(win_res)
                all_mask.append(win_mask)
                all_eval_mask.append(win_eval_mask)  # ✅ 保存eval_mask
                all_groups.append(u_idx)
                
                if has_time:
                    win_int = np.zeros(max_len, dtype=float)
                    win_rt = np.zeros(max_len, dtype=float)
                    win_int[:length] = full_interval[start:end]
                    win_rt[:length] = full_response[start:end]
                    all_interval.append(win_int)
                    all_response.append(win_rt)
            
            count += 1
            if count % 1000 == 0:
                ic(f"Processing users: {count}...")

        # --- 保存 ---
        ic(f"序列构建完成: Users={count}, Sequences={len(all_seq)}")
        np.save(os.path.join(self.output_dir, f'{save_prefix}_seq.npy'), np.array(all_seq))
        np.save(os.path.join(self.output_dir, f'{save_prefix}_res.npy'), np.array(all_res))
        np.save(os.path.join(self.output_dir, f'{save_prefix}_mask.npy'), np.array(all_mask))
        np.save(os.path.join(self.output_dir, f'{save_prefix}_eval_mask.npy'), np.array(all_eval_mask))  # ✅ 新增
        np.save(os.path.join(self.output_dir, f'{save_prefix}_window_groups.npy'), np.array(all_groups))
        
        if has_time and len(all_interval) > 0:
            np.save(os.path.join(self.output_dir, f'{save_prefix}_interval_time.npy'), np.array(all_interval))
            np.save(os.path.join(self.output_dir, f'{save_prefix}_response_time.npy'), np.array(all_response))

    def _build_domain_map(self, df=None):
        """
        可以根据手动配置或自动聚类构建技能领域映射
        Args:
            df: (Optional) 全量数据，用于在手动配置缺失时计算 User-Level 共现来智能分配技能
        """
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
                    
                    # 优先加载题目共现 (SS Table from Questions)
                    ss_path = os.path.join(self.output_dir, "ss_table.npz")
                    ss_dense = None
                    if os.path.exists(ss_path):
                        ss_dense = sparse.load_npz(ss_path).toarray()

                    smart_count = 0
                    
                    for s_idx in unassigned_indices:
                        # 统计该技能对各个已分配领域的关联得分 (0 到 max_domain_id)
                        domain_scores = np.zeros(max_domain_id + 1)
                        
                        # 策略 1: 基于 Question 共现 (Strict)
                        if ss_dense is not None:
                            for d_id in range(max_domain_id + 1):
                                d_skill_indices = np.where(domain_map == d_id)[0]
                                if len(d_skill_indices) > 0:
                                    domain_scores[d_id] += ss_dense[s_idx, d_skill_indices].sum()
                        
                        # 策略 2 (Disabled): 基于 User 共现
                        # if us_cooc is not None: ...

                        if domain_scores.sum() > 0:
                            domain_map[s_idx] = np.argmax(domain_scores)
                            smart_count += 1
                        else:
                            # 彻底孤立的技能，分配到自动生成的 "Others" 独立领域
                            domain_map[s_idx] = others_id
                    
                    ic(f"智能分配完成: {smart_count} 个归类到关联领域，{unassigned_count - smart_count} 个归入独立领域 [Others] (ID: {others_id})")
                
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
    def _build_q_features(self, df):
        """构建问题特征向量"""
        # 简单实现，仅占位
        pass

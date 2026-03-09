import os
import json
import numpy as np
import pandas as pd
from scipy import sparse
from config import MAX_SEQ_LEN, Config
from .standard_columns import StandardColumns
try:
    from icecream import ic
except ImportError:
    ic = print

class KTDataBuilder:
    """
    数据构建器 V2
    负责将 DataFrame 转换为模型所需的 List of Dicts 格式 (用于 Parquet)
    包括：ID映射、序列生成、邻接矩阵构建
    
    支持标准列名: uid, qid, sid, label, timestamp, rt
    """
    def __init__(self, config: Config):
        self.config = config
        self.output_dir = config.PROCESSED_DATA_DIR
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # 使用标准列名
        self.user_col = StandardColumns.USER_ID
        self.question_col = StandardColumns.QUESTION_ID
        self.skill_col = StandardColumns.SKILL_IDS
        self.label_col = StandardColumns.LABEL
        self.timestamp_col = StandardColumns.TIMESTAMP
        self.rt_col = StandardColumns.RESPONSE_TIME
        
        self.user2idx = {}
        self.question2idx = {}
        self.skill2idx = {}
        self.idx2user = {}
        self.idx2question = {}
        self.idx2skill = {}
        self.skill_domain_map = {}

    def _parse_skills(self, raw_skill):
        """解析技能ID，支持 int, float, str (标准分隔符分割)"""
        if pd.isna(raw_skill):
            return []
        if isinstance(raw_skill, (int, float)):
            return [int(raw_skill)]

        # 字符串处理，已由 adapter.py 标准化为统一分隔符
        s = str(raw_skill).strip()
        parts = s.split(StandardColumns.SKILL_IDS_STD_SEP)

        skills = []
        for p in parts:
            p_clean = p.strip()
            if not p_clean:
                continue
            try:
                skills.append(int(p_clean))
            except ValueError:
                # 尝试提取数字部分（如 "skill_123" -> 123）
                import re
                match = re.search(r'\d+', p_clean)
                if match:
                    skills.append(int(match.group()))
                else:
                    ic(f"Warning: 无法解析技能 ID: '{p_clean}'")
        return skills

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

    def _build_id_maps(self, df):
        # User ID
        users = df[self.user_col].unique()
        self.user2idx = {u: i for i, u in enumerate(users)}
        self.idx2user = {i: u for u, i in self.user2idx.items()}
        
        # Question ID (0 is reserved for padding/unknown)
        question = df[self.question_col].unique()
        self.question2idx = {p: i+1 for i, p in enumerate(question)}
        self.question2idx[0] = 0 # Padding
        self.idx2question = {i: p for p, i in self.question2idx.items()}
        
        # Skill ID (0 is reserved for padding/unknown)
        skills = set()
        if self.skill_col in df.columns:
            for s_str in df[self.skill_col].dropna().unique():
                skills.update(self._parse_skills(s_str))
        self.skill2idx = {s: i+1 for i, s in enumerate(sorted(list(skills)))}
        self.skill2idx[0] = 0 # Padding
        self.idx2skill = {i: s for s, i in self.skill2idx.items()}
        
        ic(f"  - Users: {len(self.user2idx)}")
        ic(f"  - Questions: {max(self.question2idx.values()) + 1}")
        ic(f"  - Skills: {max(self.skill2idx.values()) + 1}")

        # Save JSON maps
        # Skill-ID Map
        with open(os.path.join(self.output_dir, 'skill2idx.json'), 'w', encoding='utf-8') as f:
            # json key must be str
            str_skill2idx = {str(k): int(v) for k, v in self.skill2idx.items()}
            json.dump(str_skill2idx, f, indent=4)
            
        # Question-ID Map (question2idx.json)
        with open(os.path.join(self.output_dir, 'question2idx.json'), 'w', encoding='utf-8') as f:
            str_question2idx = {str(k): int(v) for k, v in self.question2idx.items()}
            json.dump(str_question2idx, f, indent=4)
        
        # User-ID Map
        with open(os.path.join(self.output_dir, 'user2idx.json'), 'w', encoding='utf-8') as f:
            str_user2idx = {str(k): int(v) for k, v in self.user2idx.items()}
            json.dump(str_user2idx, f, indent=4)
            
        ic("ID映射已保存至 JSON 文件 (skill2idx.json, question2idx.json, user2idx.json)")

    def _build_adjacency_matrices(self, df):
        """构建 Q-S, Q-Q, S-S 邻接矩阵"""
        num_q = max(self.question2idx.values()) + 1 # 包含 0
        num_s = max(self.skill2idx.values()) + 1 # 包含 0
        
        qs_table = np.zeros([num_q, num_s], dtype=int)
        
        # 遍历每个题目，填充 Q-S 表
        # 为了效率，我们先去重
        question_skill_df = df[[self.question_col, self.skill_col]].drop_duplicates()
        
        for row in question_skill_df.itertuples(index=False):
            q_idx = self.question2idx.get(getattr(row, self.question_col))
            if q_idx is None or q_idx == 0: continue
            
            current_skills = self._parse_skills(getattr(row, self.skill_col))
            
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
        num_s = max(self.skill2idx.values()) + 1
        
        # --- 策略 A: 手动配置 ---
        if use_manual and config_path and os.path.exists(config_path):
            ic(f"加载手动 Domain 配置: {config_path}")
            try:
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
                        s_raw_id = None
                        if isinstance(item, dict):
                            s_raw_id = item.get('skill_id')
                        elif isinstance(item, (str, int)):
                            s_raw_id = item
                        
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
                    
                    others_id = len(manual_config)
                    
                    ss_path = os.path.join(self.output_dir, "ss_table.npz")
                    ss_dense = None
                    if os.path.exists(ss_path):
                        ss_dense = sparse.load_npz(ss_path).toarray()

                    smart_count = 0
                    
                    for s_idx in unassigned_indices:
                        domain_scores = np.zeros(max_domain_id + 1)
                        
                        if ss_dense is not None:
                            for d_id in range(max_domain_id + 1):
                                d_skill_indices = np.where(domain_map == d_id)[0]
                                if len(d_skill_indices) > 0:
                                    domain_scores[d_id] += ss_dense[s_idx, d_skill_indices].sum()
                        
                        if domain_scores.sum() > 0:
                            domain_map[s_idx] = np.argmax(domain_scores)
                            smart_count += 1
                        else:
                            domain_map[s_idx] = others_id
                    
                    ic(f"智能分配完成: {smart_count} 个归类到关联领域，{unassigned_count - smart_count} 个归入独立领域 [Others] (ID: {others_id})")
                
                unique_domains = np.unique(domain_map)
                ic(f"手动映射完成: 所有 {num_s} 个技能已分配到 {len(unique_domains)} 个 Domain。")
                
                # 保存内部索引 -> 领域映射供后续使用
                self.skill_domain_map = {i: int(d) for i, d in enumerate(domain_map)}
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

        ss_path = os.path.join(self.output_dir, "ss_table.npz")
        if not os.path.exists(ss_path):
            ic("错误: ss_table.npz 不存在，无法聚类")
            return
            
        ss_sparse = sparse.load_npz(ss_path)
        ss_dense = ss_sparse.toarray()
        
        ss_norm = normalize(ss_dense, norm='l2', axis=1)

        if num_s <= target_n_clusters:
            domain_map = np.arange(num_s)
        else:
            cluster = AgglomerativeClustering(
                n_clusters=target_n_clusters, 
                linkage='ward' 
            )
            domain_map = cluster.fit_predict(ss_norm)
        
        # 构造内部索引->领域映射
        self.skill_domain_map = {i: int(d) for i, d in enumerate(domain_map)}
        
        # Save Domain Map
        with open(os.path.join(self.output_dir, 'skill_domain_map.json'), 'w', encoding='utf-8') as f:
            str_domain_map = {str(k): int(v) for k, v in self.skill_domain_map.items()}
            json.dump(str_domain_map, f, indent=4, ensure_ascii=False)
        ic(f"Domain Map 已保存: {os.path.join(self.output_dir, 'skill_domain_map.json')}")
        
        unique, counts = np.unique(domain_map, return_counts=True)
        ic(f"Domain 分布: {dict(zip(unique, counts))}")

    def _build_q_features(self, df):
        """构建问题特征向量"""
        # 简单实现，仅占位
        pass

    def build_sequences(self, df, stride=None, save_prefix="train"):
        """
        构建序列数据，返回 List of Dicts
        """
        ic(f"开始构建序列 [{save_prefix}] (Stride={stride})...")
        max_len = MAX_SEQ_LEN
        min_len = 5
        
        has_time = self.rt_col in df.columns and self.timestamp_col in df.columns
        question_avg_time = {}
        if has_time:
            df[self.rt_col] = pd.to_numeric(df[self.rt_col], errors='coerce')
            valid_rt = df[(df[self.rt_col] > 0) & (df[self.rt_col] < 3600000)]
            if not valid_rt.empty:
                question_avg_time = valid_rt.groupby(self.question_col)[self.rt_col].median().to_dict()

        records = []
        grouped = df.groupby(self.user_col)
        
        for user_id, group in grouped:
            u_idx = self.user2idx.get(user_id)
            if u_idx is None: continue
            
            full_seq = []
            full_c_seq = []
            full_res = []
            full_interval = []
            full_response = []
            
            user_last_skill_time = {} 
            
            for row in group.itertuples(index=False):
                q_idx = self.question2idx.get(getattr(row, self.question_col), 0)
                full_seq.append(q_idx)
                full_res.append(int(getattr(row, self.label_col)))
                
                curr_skills = []
                if hasattr(row, self.skill_col) and pd.notna(getattr(row, self.skill_col)):
                    s_raw_list = self._parse_skills(getattr(row, self.skill_col))
                    for s_raw in s_raw_list:
                        s_id = self.skill2idx.get(s_raw)
                        if s_id is not None: curr_skills.append(s_id)
                
                # 保存所有关联的 skill，如果没有则为 [0]
                full_c_seq.append(curr_skills if curr_skills else [0])
                
                if has_time:
                    try:
                        ts = float(getattr(row, self.timestamp_col)) if pd.notna(getattr(row, self.timestamp_col)) else 0.0
                    except: ts = 0.0
                    if ts < 0: ts = 0.0
                    
                    intervals = []
                    for s_idx in curr_skills:
                        last = user_last_skill_time.get(s_idx, ts)
                        intervals.append(max(0.0, ts - last))
                        user_last_skill_time[s_idx] = ts
                    
                    avg_int = sum(intervals)/len(intervals) if intervals else 0.0
                    full_interval.append(float(np.tanh(np.log(avg_int + 1))))
                    
                    avg_t = question_avg_time.get(getattr(row, self.question_col), 1.0)
                    raw_rt = getattr(row, self.rt_col) if (hasattr(row, self.rt_col) and pd.notna(getattr(row, self.rt_col)) and getattr(row, self.rt_col) >=0) else 0.0
                    full_response.append(float(np.tanh(min(raw_rt / max(avg_t, 1e-6), 10.0))))

            total_interactions = len(full_seq)
            if total_interactions == 0: continue

            is_train_mode = 'train' in save_prefix.lower()
            
            history_len = 0
            
            # 如果 stride=None 或者 stride >= max_len，则使用非重叠模式；否则使用指定的 stride 进行切分
            if stride is None or stride >= total_interactions:
                # 测试集且序列过长：使用重叠窗口，保留历史预热
                if not is_train_mode and total_interactions > max_len:
                    # 测试集超长序列：使用50%重叠，保留历史预热（避免冷启动）
                    # 窗口1: [0-200]，全部评估
                    # 窗口2: [100-300]，前100是历史，后100是新评估数据
                    current_stride = max_len // 2
                    history_len = max_len // 2
                else: # 训练集或短序列：直接使用非重叠窗口，全部评估
                    current_stride = max_len
                    history_len = 0
            else: # 否则使用指定的 stride 进行切分，训练集允许更密集的切分，测试集则保持较大的重叠以保留历史预热
                current_stride = stride
                history_len = max_len // 2 if total_interactions > max_len else 0
            
            starts = list(range(0, total_interactions, current_stride))
            
            for i, start in enumerate(starts):
                end = min(start + max_len, total_interactions)
                length = end - start
                
                if is_train_mode and length < min_len and len(starts) > 1:
                    continue
                if length <= 0: 
                    continue

                # eval_mask: True表示需要计算loss, False表示只作为历史上下文
                eval_mask = [False] * length
                
                if history_len > 0 and start > 0:
                    eval_start = min(history_len, length)
                    eval_end = length
                else:
                    eval_start = 0
                    eval_end = length
                
                if eval_end > eval_start and length > 0:
                    for j in range(eval_start, eval_end):
                        eval_mask[j] = True
                elif length > 0 and start == 0:
                    for j in range(length):
                        eval_mask[j] = True
                
                record = {
                    'uid': u_idx,
                    'sequence_id': f"{u_idx}_{i}",
                    'q_seq': full_seq[start:end],
                    'c_seq': full_c_seq[start:end],
                    'r_seq': full_res[start:end],
                    'mask': [True] * length,
                    'eval_mask': eval_mask,
                    'group_id': u_idx # 使用 user_idx 作为 group_id 防止泄露
                }
                
                if has_time:
                    record['t_interval'] = full_interval[start:end]
                    record['t_response'] = full_response[start:end]
                    
                records.append(record)
                
        return records

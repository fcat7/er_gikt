import pandas as pd
from config import Config
try:
    from icecream import ic
except ImportError:
    ic = print

class KTDataAdapter:
    """
    通用知识追踪数据适配器
    负责数据的加载、列名映射和基础清洗
    """
    def __init__(self, config : Config):
        self.config = config
        
        self.col_map = config.dataset.COLUMN_MAP
        
        # 构造 {Raw: Standard} 映射用于 pandas rename
        # 如果 raw 不唯一，会有问题，但假设数据集设计良好 
        self.raw_to_std_map = {raw_col: std_col for std_col, raw_col in self.col_map.items()}

        self.required_cols = ['user_id', 'problem_id', 'correct']

    def load_data(self):
        """
        加载CSV文件并重命名列
        """
        filepath = self.config.dataset.FILE_PATH
        ic(f"正在加载数据: {filepath}")
        try:
            df = pd.read_csv(filepath, encoding=self.config.dataset.ENCODING)
        except FileNotFoundError:
            raise FileNotFoundError(f"文件未找到: {filepath}")
        
        # 检查原始列名是否存在
        # raw_to_std_map.keys() 是原始列名
        missing_raw_cols = [raw_col for raw_col in self.raw_to_std_map.keys() if raw_col not in df.columns]
        
        # 允许部分非必须列缺失 (比如 timestamp)
        # 检查核心列对应的原始列名是否缺失
        # 需要反查得到 Raw Name
        raw_uid = self.col_map.get('user_id')
        raw_pid = self.col_map.get('problem_id')
        
        if (raw_uid and raw_uid in missing_raw_cols) or (raw_pid and raw_pid in missing_raw_cols):
            raise ValueError(f"原始数据中缺少关键列: {missing_raw_cols}")

        # 重命名列
        # 只重命名存在的列
        actual_rename_map = {raw: std for raw, std in self.raw_to_std_map.items() if raw in df.columns}
        df = df.rename(columns=actual_rename_map)
        
        ic(f"数据加载成功，形状: {df.shape}")
        return df

    def clean_data(self, df):
        """
        基础数据清洗
        1. 关键列 (Required): 去除空值(含空字符串)，验证唯一性/有效性
        2. 可选列 (Optional): 清洗异常值 (NaN, 负数, 过大值等)
        3. 排序: 按 user_id 和 timestamp
        """
        import numpy as np
        original_shape = df.shape
        ic(f"开始清洗数据, 原始形状: {original_shape}")
        
        # --- 1. 必选列清洗 (user_id, problem_id, correct) ---
        # 策略: 发现缺失或非法，直接丢弃整行
        subset_cols = [col for col in self.required_cols if col in df.columns]
        
        if 'skill_id' in df.columns:
            subset_cols.append('skill_id')
        
        # 预处理：清理空字符串 (针对 object 列)
        # 防止 " ", "" 等被视作有效值不被 dropna 捕获
        for col in subset_cols:
            if df[col].dtype == 'object':
                df[col] = df[col].replace(r'^\s*$', np.nan, regex=True)

        # 去除 NaN (仅针对 user_id, problem_id, correct)
        df = df.dropna(subset=subset_cols)
        
        # --- 1.5 Skill ID 特殊清洗 ---
        if 'skill_id' in df.columns:
            # 统计并去除 -1
            # 转换为字符串进行匹配，兼容数值型和字符型
            # 注意：不处理 NaN (空值保留)
            
            # 创建掩码：值为 -1 或 "-1" 或 "-1.0"
            # astype(str) 会将 NaN 转换为 'nan'，不用担心报错
            is_invalid_skill = df['skill_id'].astype(str).isin(['-1', '-1.0', '-1.00'])
            
            removed_count = is_invalid_skill.sum()
            if removed_count > 0:
                ic(f"检测到并移除 skill_id 为 -1 的数据: {removed_count} 条")
                df = df[~is_invalid_skill]

        # 验证 correct 列 (允许 >= 0 的数值，支持分数)
        if 'correct' in df.columns:
            # 确保是数值类型
            df['correct'] = pd.to_numeric(df['correct'], errors='coerce')
            df = df.dropna(subset=['correct']) 
            
            # @change_fzq: 允许分数 (>=0)，不再强制 0/1
            # 过滤掉负分 (通常是异常值)
            df = df[df['correct'] >= 0]
            
            # 简单的检查报警，防止误用
            if (df['correct'] > 1).any():
                ic("Warning: 检测到 correct 值大于 1，将被保留作为原始分数。")
            
        # --- 2. 可选列清洗 (timestamp, response_time) ---
        
        # A. Timestamp (如果存在，必须有效，否则无法排序)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['timestamp']) # 时间戳缺失则丢弃
            
        # B. Response Time (存在脏数据的重灾区)
        if 'response_time' in df.columns:
            # 转数值
            df['response_time'] = pd.to_numeric(df['response_time'], errors='coerce')
            
            # 策略: 
            # 1. NaN -> 填充 0 (表示未知或极快)
            # 2. 负数 -> 截断为 0
            # 3. 过大 (挂机) -> 截断 (例如 3600*1000 ms = 1小时)
            # 4. 过小 (频繁点击) -> 保留 (用户认知风格)
            
            # 处理 NaN
            df['response_time'] = df['response_time'].fillna(0.0)
            
            # 处理负数
            df.loc[df['response_time'] < 0, 'response_time'] = 0.0
            
            # 处理挂机数据 (单位统一假设为 ms, 阈值设为 1小时)
            # Assist09: ms_first_response (ms)
            # EdNet: elapsed_time (ms)
            THRESHOLD_AFK = 3600 * 1000  # 1 hour
            
            # 统计一下有多少挂机数据被截断
            afk_count = (df['response_time'] > THRESHOLD_AFK).sum()
            if afk_count > 0:
                ic(f"检测到 {afk_count} 条 response_time > 1h (挂机数据), 执行截断处理")
                df.loc[df['response_time'] > THRESHOLD_AFK, 'response_time'] = THRESHOLD_AFK
                
            # 处理 "过零" (严格等于0的数据)
            # 虽然 0ms 不太可能，但如果是点击间隔为0，可能意味着重复提交或系统错误
            # 这里暂时不处理，或者视作极快
                
        # 3. 排序
        sort_cols = []
        if 'user_id' in df.columns:
            sort_cols.append('user_id')
        if 'timestamp' in df.columns:
            sort_cols.append('timestamp')
            
        if sort_cols:
            df = df.sort_values(by=sort_cols)
            
        ic(f"基础清洗完成: {original_shape} -> {df.shape}")
        return df

    def save_standard_csv(self, df, output_path):
        """
        将清洗后的数据保存为标准 CSV
        """
        ic(f"正在保存标准数据集至: {output_path}")
        # 只保存配置中映射过的列，确保输出整洁
        # col_map 是 {Standard: Raw}，所以 .keys() 是标准列名
        standard_cols = list(self.col_map.keys())
        # 过滤掉 DataFrame 中不存在的列
        cols_to_save = [col for col in standard_cols if col in df.columns]
        
        df[cols_to_save].to_csv(output_path, index=False, encoding='utf-8')
        ic("保存完成")

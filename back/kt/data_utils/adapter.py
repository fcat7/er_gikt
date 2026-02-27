import pandas as pd
import numpy as np
from .standard_columns import StandardColumns

try:
    from icecream import ic
except ImportError:
    ic = print

class KTDataAdapter:
    """
    数据适配器 V2（配置驱动）
    职责：原始数据源 → 标准化 DataFrame
    
    架构定位：
    - 输入：TOML 配置 + 原始 CSV/Parquet
    - 输出：标准化的 DataFrame（列名：uid, qid, sid, label, timestamp, rt）
    - 功能：加载、重命名、过滤、清洗、验证
    """
    
    @staticmethod
    def load_from_toml(toml_config):
        """
        从 TOML 配置加载数据并完成标准化处理
        
        Args:
            toml_config: dict, 包含 file_path, encoding, column_map, filters
            
        Returns:
            df: pd.DataFrame, 标准化后的数据（列名：uid, qid, sid, label, timestamp, rt）
        """
        ic("=== 开始数据适配流程 (Adapter) ===")
        
        # Step 1: 加载原始数据
        df = KTDataAdapter._load_raw_data(
            toml_config['file_path'], 
            toml_config.get('encoding', 'utf-8')
        )
        
        # Step 2: 列名标准化（核心：Normalization Wall）
        df = KTDataAdapter._standardize_columns(df, toml_config.get('column_map', {}))
        
        # Step 3: 应用过滤器
        df = KTDataAdapter._apply_filters(df, toml_config.get('filters', {}))
        
        # Step 4: skill_ids 分隔符标准化（可选）
        skill_ids_col = StandardColumns.SKILL_IDS
        skill_ids_sep = toml_config.get('skill_ids_sep', ",;_")
        std_sep = StandardColumns.SKILL_IDS_STD_SEP
        if skill_ids_col in df.columns and skill_ids_sep:
            ic(f"Skill IDs 分隔符标准化: {skill_ids_sep} → {std_sep}")
            # 支持多个分隔符及其前后空白
            import re
            pattern = f"\\s*[{re.escape(skill_ids_sep)}]\\s*"
            df[skill_ids_col] = df[skill_ids_col].astype(str).replace(pattern, std_sep, regex=True)

        # Step 5: 数据清洗
        df = KTDataAdapter._clean_data(df)

        ic("=== 数据适配完成 ===")
        return df
    
    @staticmethod
    def _load_raw_data(file_path, encoding='utf-8'):
        """加载原始 CSV 文件"""
        ic(f"正在加载数据: {file_path}")
        
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            ic(f"Raw Data Loaded: {df.shape}")
            return df
        except FileNotFoundError:
            raise FileNotFoundError(f"文件未找到: {file_path}")
        except Exception as e:
            raise RuntimeError(f"数据加载失败: {e}")
    
    @staticmethod
    def _standardize_columns(df, column_map):
        """
        列名标准化（Normalization Wall）
        
        将原始列名映射为框架标准列名：
        - uid: 用户ID
        - qid: 题目ID
        - sid: 技能ID
        - label: 答题正确性
        - timestamp: 时间戳
        - rt: 答题时长
        """
        if not column_map:
            raise ValueError("column_map 不能为空")

        # 检查 column_map key 是否全部在标准列名内，否则 warning
        extra_keys = [k for k in column_map.keys() if k not in StandardColumns.ALL]
        if extra_keys:
            ic(f"[Warning] column_map 包含未在 StandardColumns.ALL 中定义的列: {extra_keys}")

        # column_map 格式: {标准名: 原始名}
        # 需要反转为 {原始名: 标准名} 用于 pandas.rename
        rename_map = {raw_col: std_col for std_col, raw_col in column_map.items() if raw_col in df.columns}

        if not rename_map:
            raise ValueError("column_map 中没有匹配到任何列，请检查配置文件")

        df = df.rename(columns=rename_map)
        ic(f"列名标准化完成，映射了 {len(rename_map)} 列")

        # 验证必需列
        required_cols = [StandardColumns.USER_ID, StandardColumns.QUESTION_ID, StandardColumns.LABEL]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"标准化后缺少必需列: {missing_cols}")

        return df
    
    @staticmethod
    def _apply_filters(df, filters):
        """
        动态过滤器
        
        根据配置文件的 filters 字段过滤数据
        filters 格式: {列名: 保留值}
        示例: {"original": 1, "is_valid": True}
        """
        if not filters:
            return df
        
        original_len = len(df)
        
        for col, value in filters.items():
            if col not in df.columns:
                ic(f"警告: 过滤列 '{col}' 不存在，跳过")
                continue
            df = df[df[col] == value]
            ic(f"过滤 {col}={value}: {original_len} → {len(df)}")
            original_len = len(df)
        
        return df
    
    @staticmethod
    def _clean_data(df):
        """
        基础数据清洗
        1. 关键列 (Required): 去除空值(含空字符串)，验证唯一性/有效性
        2. 可选列 (Optional): 清洗异常值 (NaN, 负数, 过大值等)
        3. 排序: 按 user_id 和 timestamp
        """
        original_shape = df.shape
        ic(f"开始数据清洗, 原始形状: {original_shape}")

        # --- 1. 必选列清洗 (user_id, quesion_id, label) 和 skill_ids 列清洗（若存在） ---
        # 策略: 发现缺失或非法，直接丢弃整行
        required_cols = [StandardColumns.USER_ID, StandardColumns.QUESTION_ID, StandardColumns.LABEL]
        subset_cols = [col for col in required_cols if col in df.columns]
        if StandardColumns.SKILL_IDS in df.columns:
            subset_cols.append(StandardColumns.SKILL_IDS)

        # 清理空字符串（针对 object 类型列）
        for col in subset_cols:
            if df[col].dtype == 'object':
                df[col] = df[col].replace(r'^\s*$', np.nan, regex=True)

        # 去除 NaN
        df = df.dropna(subset=subset_cols).copy()

        # 2. skill_ids 特殊清洗
        if StandardColumns.SKILL_IDS in df.columns:
            # 统计并去除 -1
            # 转换为字符串进行匹配，兼容数值型和字符型
            # 注意：不处理 NaN (空值保留)
            
            # 创建掩码：值为 -1 或 "-1" 或 "-1.0"
            # astype(str) 会将 NaN 转换为 'nan'，不用担心报错
            is_invalid_skill = df[StandardColumns.SKILL_IDS].astype(str).isin(['-1', '-1.0', '-1.00'])
            removed_count = is_invalid_skill.sum()
            if removed_count > 0:
                ic(f"检测到并移除 {StandardColumns.SKILL_IDS} 为 -1 的数据: {removed_count} 条")
                df = df[~is_invalid_skill]

        # 2. 标签列(correct)验证(允许 >= 0 的数值，支持分数)
        if StandardColumns.LABEL in df.columns:
            df[StandardColumns.LABEL] = pd.to_numeric(df[StandardColumns.LABEL], errors='coerce')
            df = df.dropna(subset=[StandardColumns.LABEL])
            df = df[df[StandardColumns.LABEL] >= 0] # 过滤掉负分 (通常是异常值),允许分数 (>=0)，不强制 0/1
            if (df[StandardColumns.LABEL] > 1).any():
                ic(f"Warning: 检测到 {StandardColumns.LABEL} 值大于 1，将被保留作为原始分数。")

        # 3. 可选列清洗 (timestamp, response_time)
        # 3.A. Timestamp (如果存在，必须有效，否则无法排序)
        if StandardColumns.TIMESTAMP in df.columns:
            df[StandardColumns.TIMESTAMP] = pd.to_numeric(df[StandardColumns.TIMESTAMP], errors='coerce')
            df = df.dropna(subset=[StandardColumns.TIMESTAMP]) # 时间戳缺失则丢弃

        # 3.B. Response Time (存在脏数据的重灾区)
        if StandardColumns.RESPONSE_TIME in df.columns:
            df[StandardColumns.RESPONSE_TIME] = pd.to_numeric(df[StandardColumns.RESPONSE_TIME], errors='coerce') # 转数值
            # 策略: 
            # 1. NaN -> 填充 0 (表示未知或极快)
            # 2. 负数 -> 截断为 0
            # 3. 过大 (挂机) -> 截断 (例如 3600*1000 ms = 1小时)
            # 4. 过小 (频繁点击) -> 保留 (用户认知风格)
            df[StandardColumns.RESPONSE_TIME] = df[StandardColumns.RESPONSE_TIME].fillna(0.0)
            df.loc[df[StandardColumns.RESPONSE_TIME] < 0, StandardColumns.RESPONSE_TIME] = 0.0
            THRESHOLD_AFK = 3600 * 1000  # 1 hour
            afk_count = (df[StandardColumns.RESPONSE_TIME] > THRESHOLD_AFK).sum()
            if afk_count > 0:
                ic(f"检测到 {afk_count} 条 {StandardColumns.RESPONSE_TIME} > 1h (挂机数据), 执行截断处理")
                df.loc[df[StandardColumns.RESPONSE_TIME] > THRESHOLD_AFK, StandardColumns.RESPONSE_TIME] = THRESHOLD_AFK

        # 4. 排序
        sort_cols = []
        if StandardColumns.USER_ID in df.columns:
            sort_cols.append(StandardColumns.USER_ID)
        if StandardColumns.TIMESTAMP in df.columns:
            sort_cols.append(StandardColumns.TIMESTAMP)
        if sort_cols:
            df = df.sort_values(by=sort_cols).reset_index(drop=True)

        ic(f"数据清洗完成: {original_shape} -> {df.shape}")
        return df

    @staticmethod
    def save_standard_csv(df, output_path):
        """
        保存标准化后的数据为 CSV，仅包含 StandardColumns.ALL 中存在于 df 的列，顺序与 StandardColumns.ALL 保持一致。
        """
        ic(f"正在保存标准数据集至: {output_path}")
        cols_to_save = [col for col in StandardColumns.ALL if col in df.columns]
        df[cols_to_save].to_csv(output_path, index=False, encoding='utf-8')
        ic("保存完成")

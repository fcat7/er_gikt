import sys
import os
# @fix_fzq: 解决 OpenMP 多副本冲突报错 (放在其他 import 之前)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
from config import Config, RANDOM_SEED
try:
    from icecream import ic
except ImportError:
    ic = print

# Add current directory to sys.path to ensure modules can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from data_utils.adapter import KTDataAdapter
from data_utils.inspector import KTDataInspector
from data_utils.sampler import KTDataSampler
from data_utils.builder import KTDataBuilder
import numpy as np
from tqdm import tqdm
import os

def calculate_question_features(df):
    """基于全量 DataFrame 计算题目统计特征"""
    ic(f"开始计算题目特征，数据量: {df.shape}")
    
    # 预计算用户平均分 (用于区分度计算)
    if 'correct' in df.columns and 'user_id' in df.columns:
        user_scores = df.groupby('user_id')['correct'].mean()
        df = df.copy() # 避免 SettingWithCopyWarning
        df['user_score'] = df['user_id'].map(user_scores)
    else:
        ic("缺少 correct 或 user_id 列，跳过特征计算")
        return None

    grouped = df.groupby('problem_id')
    feature_data = []
    
    for pid, group in tqdm(grouped, desc="提取题目特征"):
        # A. 平均正确率
        avg_correct = group['correct'].mean()
        # B. 难度 (1 - Correct)
        difficulty = 1.0 - avg_correct
        # C. 平均响应时间
        avg_rt = group['response_time'].mean() if 'response_time' in group.columns else 0.0
        # D. 区分度
        if len(group) > 1 and group['correct'].nunique() > 1:
            disc = group['correct'].corr(group['user_score'])
            if pd.isna(disc): disc = 0.0
        else:
            disc = 0.0
        
        feature_data.append({
            'problem_id': pid,
            'difficulty': difficulty,
            'discrimination': disc,
            'avg_rt': avg_rt,
            'avg_correct': avg_correct,
            'count': len(group)
        })
        
    return pd.DataFrame(feature_data)

def build_feature_matrix(features_df, q2idx_path, output_path, feature_cols=['difficulty', 'discrimination', 'avg_rt']):
    """将特征 DataFrame 映射到 question2idx 并归一化保存"""
    # 用户决定移除 'avg_correct' 因为它与 'difficulty' 高度负相关
    if features_df is None or features_df.empty:
        return

    ic("构建特征矩阵 (.npy)...")
    # 1. 预处理与归一化 (使用 Log1p 处理 RT)
    df = features_df.copy()
    if 'avg_rt' in df.columns:
        df['avg_rt'] = df['avg_rt'].apply(lambda x: max(0, x)) # Clip neg
        df['avg_rt_log'] = np.log1p(df['avg_rt'])
        # 替换列名用于后续处理
        cols_to_use = [c if c != 'avg_rt' else 'avg_rt_log' for c in feature_cols]
    else:
        cols_to_use = feature_cols
    
    # 2. Z-Score 归一化 (在全量特征分布上进行)
    stats = {}
    for col in cols_to_use:
        if col not in df.columns: continue
        mean_val = df[col].mean()
        std_val = df[col].std()
        if std_val == 0: df[col] = 0.0
        else: df[col] = (df[col] - mean_val) / std_val
        stats[col] = {'mean': mean_val, 'std': std_val}
    
    ic("特征归一化统计:", stats)
    
    # 3. 映射到 question2idx
    if not os.path.exists(q2idx_path):
        ic(f"Error: 找不到 ID 映射文件 {q2idx_path}")
        return

    q2idx = np.load(q2idx_path, allow_pickle=True).item()
    num_q = len(q2idx)
    final_matrix = np.zeros((num_q, len(feature_cols)), dtype=np.float32)
    
    df.set_index('problem_id', inplace=True)
    
    found = 0
    for original_id, mapped_id in q2idx.items():
        if mapped_id == 0: continue
        # 尝试匹配 ID (int/str)
        oid_candidates = [original_id]
        try: oid_candidates.append(int(original_id))
        except: pass
        
        for oid in oid_candidates:
            if oid in df.index:
                final_matrix[mapped_id] = df.loc[oid, cols_to_use].values
                found += 1
                break
    
    ic(f"特征矩阵构建完成: {found}/{num_q-1} matched. Shape: {final_matrix.shape}")
    np.save(output_path, final_matrix)
    ic(f"Saved to {output_path}")

def main():
    # 1. 初始化配置
    ic("初始化配置...")
    # Available: ['assist09', 'assist12', 'ednet_kt1'] 
    base_dataset_name = 'assist09'  # 基础数据集名称 (如 assist09, assist12)
    ratio = 0.1                     # 采样率 (0.0 < ratio <= 1.0)
    min_seq_len = 20
    reandom_seed = RANDOM_SEED
    
    # 校验 ratio
    if not (0 < ratio <= 1.0):
        ic(f"Error: ratio must be in range (0, 1]. Current: {ratio}")
        return

    # 根据 ratio 生成中间数据集名称 (不含窗口后缀)
    if ratio == 1.0:
        current_dataset_name = base_dataset_name
    else:
        current_dataset_name = f"{base_dataset_name}-sample_{int(ratio*100)}%"

    # 兼容后续代码使用的 dataset_name 变量
    dataset_name = current_dataset_name 
    
    # @add_fzq: 数据增强配置开关
    ENABLE_SLIDING_WINDOW = True  # <--- 在这里控制是否启用滑动窗口！
    WINDOW_STRIDE = 100            # 滑动步长

    # 1. 配置加载策略 
    # src_config 始终读取基础数据集 (确保能正确加载原始数据)
    src_config = Config(dataset_name=base_dataset_name) 
    
    # target_config 决定输出目录
    if ENABLE_SLIDING_WINDOW:
        # 如果启用增强，在 dataset_name 基础上增加后缀
        target_dataset_name = f"{dataset_name}_window"
        ic(f"模式: 滑动窗口增强开启。")
    else:
        target_dataset_name = dataset_name
        ic(f"模式: 标准/采样处理。")

    target_config = Config(dataset_name=target_dataset_name)
    ic(f"配置生成 -> 源: {base_dataset_name}, 目标: {target_dataset_name}")

    # 2. 数据加载与基础清洗 (使用源配置读取)
    ic("Step 1: 数据加载与标准化...")
    adapter = KTDataAdapter(src_config)
    try:
        df = adapter.load_data()
        df = adapter.clean_data(df)
    except Exception as e:
        ic(f"数据加载失败: {e}")
        return
    
    # 3. 数据集特定过滤
    if src_config.dataset.DATA_NAME == 'assist09':
        if 'original' in df.columns:
            df = df[df['original'] == 1]
        if 'skill_id' in df.columns:
            df = df[df['skill_id'] != 'NA']

    # 4. 过滤过短序列
    ic(f"过滤交互数小于 {min_seq_len} 的用户...")
    user_counts = df['user_id'].value_counts()
    valid_users = user_counts[user_counts >= min_seq_len].index
    df = df[df['user_id'].isin(valid_users)]
    
    # 将 skill_id 中的分隔符统一处理
    if 'skill_id' in df.columns:
        df['skill_id'] = df['skill_id'].apply(lambda x: str(x).replace(',', '_').replace(';', '_'))
    
    # 保存标准全量数据集
    standard_csv_path = os.path.join(target_config.PROCESSED_DATA_DIR, f'{dataset_name}_standard.csv')
    if not os.path.exists(target_config.PROCESSED_DATA_DIR):
        os.makedirs(target_config.PROCESSED_DATA_DIR)
    adapter.save_standard_csv(df, standard_csv_path)

    # ==========================================
    # Step 2: 提取全量题目特征 (新增集成)
    # ==========================================
    ic("Step 2: 提取题目统计特征 (Difficulty, Disc, etc.)...")
    q_features_df = calculate_question_features(df)
    
    if q_features_df is not None:
        feat_csv_path = os.path.join(target_config.PROCESSED_DATA_DIR, 'q_features_analysis.csv')
        q_features_df.to_csv(feat_csv_path, index=False)
        ic(f"特征分析表已保存: {feat_csv_path}")
    
    # ==========================================
    # Step 3: 数据探查
    # ==========================================
    ic("Step 3: 数据探查...")
    inspector = KTDataInspector(df, target_config)
    # inspector.get_stats()
    
    # ==========================================
    # Step 4: 数据采样 (可选)
    # ==========================================
    # 假设我们想抽取 10% 的数据进行快速测试
    # 注意：如果不需要采样，请将 ratio 设为 1.0 或直接使用 df
    ic(f"Step 4: 数据采样{ratio}...")
    sampler = KTDataSampler(random_seed=reandom_seed)
    
    # [修改] 使用采样数据，或者如果需要全量，可以注释掉下面这行用 sampled_df = df
    sampled_df = sampler.stratified_sample(df, ratio=ratio, min_seq_len=min_seq_len)

    # 保存抽样数据集
    sampled_csv_path = os.path.join(target_config.PROCESSED_DATA_DIR, f'{dataset_name}_sampled.csv')
    adapter.save_standard_csv(sampled_df, sampled_csv_path)
    
    # ==========================================
    # Step 5: 构建模型输入 (.npy)
    # ==========================================
    ic("Step 5: 构建模型序列数据...")
    builder = KTDataBuilder(target_config)
    # build_dataset 会生成 train/test/valid.npy 以及 question2idx.npy
    # 传递新的参数以控制领域构建
    builder.build_dataset(sampled_df, enable_window_aug=ENABLE_SLIDING_WINDOW, stride=WINDOW_STRIDE) 

    # ==========================================
    # Step 6: 生成特征矩阵 (.npy)
    # ==========================================
    if q_features_df is not None:
        ic("Step 6: 生成题目特征矩阵 q_features.npy...")
        q2idx_path = os.path.join(target_config.PROCESSED_DATA_DIR, 'question2idx.npy')
        output_matrix_path = os.path.join(target_config.PROCESSED_DATA_DIR, 'q_features.npy')
        
        build_feature_matrix(q_features_df, q2idx_path, output_matrix_path)
    else:
        ic("Warning: 未能生成特征矩阵 (q_features_df is None)")

if __name__ == '__main__':
    main()

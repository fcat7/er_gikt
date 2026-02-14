import sys
import os
import argparse
# @fix_fzq: 解决 All-In-One 多库冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import numpy as np
from tqdm import tqdm

# Self modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import get_config, RANDOM_SEED, MAX_SEQ_LEN
from data_utils.adapter import KTDataAdapter
from data_utils.sampler import KTDataSampler
from data_utils.builder import KTDataBuilder
from data_utils.splitter import KTDataSplitter

try:
    from icecream import ic
except ImportError:
    ic = print

def calculate_question_features_full(df):
    """
    全量数据计算题目特征 (Difficulty, Discrimination, RT)
    注意：为了特征稳定性，这里使用全量数据计算。严谨场景下应仅使用训练集。
    """
    # 辅助 Score 计算区分度
    # 简单计算：每个用户的全局平均分
    user_scores = df.groupby('user_id')['correct'].mean()
    df_aug = df.copy()
    df_aug['user_score'] = df_aug['user_id'].map(user_scores)
    
    feature_list = []
    
    # 分组计算
    # 必须列: problem_id, correct.可选: response_time
    if 'response_time' not in df.columns:
        df_aug['response_time'] = 0.0
        
    grouped = df_aug.groupby('problem_id')
    
    for pid, group in tqdm(grouped, desc="Extracting Features"):
        avg_correct = group['correct'].mean()
        difficulty = 1.0 - avg_correct
        
        # RT
        avg_rt = group['response_time'].median() # Median usually better for RT
        
        # Disc
        if len(group) > 5 and group['correct'].nunique() > 1:
            disc = group['correct'].corr(group['user_score'])
            if pd.isna(disc): disc = 0.0
        else:
            disc = 0.0
            
        feature_list.append({
            'problem_id': pid,
            'difficulty': difficulty,
            'discrimination': disc,
            'avg_rt': avg_rt
        })
        
    f_df = pd.DataFrame(feature_list)
    return f_df

def normalize_and_save_features(f_df, builder, output_dir):
    """
    归一化特征并根据 problem2idx 保存为 npy
    """
    if f_df is None or f_df.empty: return
    
    # 1. Norm
    cols = ['difficulty', 'discrimination', 'avg_rt']
    if 'avg_rt' in f_df.columns:
        # Log RT
        f_df['avg_rt'] = np.log1p(np.clip(f_df['avg_rt'], 0, None))
    
    # Z-Score
    for c in cols:
        if c in f_df.columns:
            m = f_df[c].mean()
            s = f_df[c].std()
            if s == 0: s = 1.0
            f_df[c] = (f_df[c] - m) / s
            
    # 2. Map
    num_q = len(builder.problem2idx) # includes 0
    final_mat = np.zeros((num_q, len(cols)), dtype=np.float32)
    
    f_df = f_df.set_index('problem_id')
    
    cnt = 0
    for pid_raw, idx in builder.problem2idx.items():
        if idx == 0: continue
        # pid_raw could be int or str
        # f_df index might be inferred type
        
        # Try direct
        vals = None
        if pid_raw in f_df.index:
            vals = f_df.loc[pid_raw, cols].values
        else:
            # Try type cast
            try:
                pid_int = int(pid_raw)
                if pid_int in f_df.index:
                    vals = f_df.loc[pid_int, cols].values
            except:
                pass
                
        if vals is not None:
            final_mat[idx] = vals
            cnt += 1
            
    save_path = os.path.join(output_dir, 'problem_features.npy')
    np.save(save_path, final_mat)
    ic(f"特征矩阵已保存: {save_path} (Matched {cnt}/{num_q-1})")

def main():
    parser = argparse.ArgumentParser(description="KT Data Processing Pipeline (Split Train/Test)")
    parser.add_argument('--dataset', type=str, default='assist09', help='Base dataset name')
    parser.add_argument('--ratio', type=float, default=1.0, help='Sample ratio (1.0 for full)')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED, help='Random seed')
    parser.add_argument('--min_seq_len', type=int, default=5, help='Min interactions per user')
    parser.add_argument('--stride', type=int, default=0, help=f'Sliding window stride for TRAIN set. 0 = disable, range: [0, {MAX_SEQ_LEN}]') # 设置滑动窗口步长，0表示不启用滑动窗口，建议设置为 50，100，150 以增强数据集
    
    args = parser.parse_args()
    
    # 1. Config Setup
    base_name = args.dataset
    if args.ratio < 1.0:
        dataset_name = f"{base_name}-sample_{int(args.ratio*100)}%"
    else:
        dataset_name = base_name

    # Sliding window control (default: disabled)
    if args.stride < 0 or args.stride > MAX_SEQ_LEN:
        raise ValueError(f"stride must be in range [0, {MAX_SEQ_LEN}], got {args.stride}")
    enable_window = args.stride > 0
    target_dataset_name = f"{dataset_name}_window" if enable_window else dataset_name
    train_stride = args.stride if enable_window else None

    ic(f"=== Pipeline Start: {dataset_name} ===")
    ic(f"滑动窗口: {'ON' if enable_window else 'OFF'} | stride={args.stride}")

    # 输入配置用于加载原始数据，输出配置用于保存处理后的数据
    src_config = get_config(dataset_name)
    target_config = get_config(target_dataset_name)
    
    # Step 1: 数据加载与基础清洗
    ic("Step 1: 数据加载与标准化...")
    adapter = KTDataAdapter(src_config)
    try:
        df = adapter.load_data()
        ic(f"Raw Data Loaded: {df.shape}")
        df = adapter.clean_data(df)
        ic(f"After Clean: {df.shape}, Users: {df['user_id'].nunique()}")
    except Exception as e:
        ic(f"数据加载失败: {e}")
        return
    
    # Step 2: 数据集特定过滤 (去除脚手架题目等)
    ic("Step 2: 数据集特定过滤...")
    if src_config.dataset.DATA_NAME == 'assist09':
        before_filter = len(df)
        if 'original' in df.columns:
            df = df[df['original'] == 1]
            ic(f"  - Filtered non-original: {before_filter} → {len(df)}")
    ic(f"After Dataset-Specific Filter: {df.shape}, Users: {df['user_id'].nunique()}")
    
    # Step 3: 过滤短序列 (在采样前统一过滤)
    ic(f"Step 3: 过滤交互数 < {args.min_seq_len} 的用户...")
    user_counts = df['user_id'].value_counts()
    valid_users = user_counts[user_counts >= args.min_seq_len].index
    df = df[df['user_id'].isin(valid_users)].copy()
    ic(f"After Filter Short Seqs: {df.shape}, Users: {df['user_id'].nunique()}")
    
    # Step 4: 特殊字符处理 (统一 skill_id 分隔符)
    ic("Step 4: 处理 skill_id 分隔符...")
    if 'skill_id' in df.columns:
        df['skill_id'] = df['skill_id'].apply(lambda x: str(x).replace(',', '_').replace(';', '_'))
        ic("  - skill_id 分隔符统一为 '_'")
    

    # Step 5: 分层采样 (如果 ratio < 1.0)
    if args.ratio < 1.0:
        ic(f"Step 5: 分层采样 (ratio={args.ratio}, 基于序列长度和准确率)...")
        sampler = KTDataSampler(random_seed=args.seed)
        df = sampler.stratified_sample(df, ratio=args.ratio)
        ic(f"After Sampling: {df.shape}, Users: {df['user_id'].nunique()}")
        
        # 保存采样后的数据集
        if not os.path.exists(target_config.PROCESSED_DATA_DIR):
            os.makedirs(target_config.PROCESSED_DATA_DIR)
        sampled_csv_path = os.path.join(target_config.PROCESSED_DATA_DIR, f'{dataset_name}_standard.csv')
        adapter.save_standard_csv(df, sampled_csv_path)

    # Step 6: 用户级分割 (Train / Test)
    ic("Step 6: 用户级分割 (Train 80% / Test 20%)...")
    splitter = KTDataSplitter(random_seed=args.seed)
    # 注意：这里不传 min_seq_len，因为数据已经过滤过了
    train_users, test_users = splitter.split_users(df, test_ratio=0.2)
    
    ic(f"Train Users: {len(train_users)}, Test Users: {len(test_users)}")
    
    # Step 7: 构建映射表 (基于全量数据确保 ID 一致性)
    ic("Step 7: 构建 ID 映射表 (problem/skill/domain)...")
    builder = KTDataBuilder(target_config)
    builder.build_maps(df)
    ic("  - ID mappings built successfully")
    
    # Step 8: 构建序列数据 (Train & Test 分别处理)
    ic("Step 8: 构建序列数据...")
    
    # Train Set: 可选滑动窗口增强
    ic(f"  [Train] Building sequences (stride={train_stride})...")
    df_train = df[df['user_id'].isin(train_users)].copy()
    if 'timestamp' in df_train.columns:
        df_train.sort_values(['user_id', 'timestamp'], inplace=True)
    builder.build_sequences(df_train, stride=train_stride, save_prefix='train')
    
    # Test Set: 非重叠切分
    ic("  [Test] Building sequences (non-overlapping)...")
    df_test = df[df['user_id'].isin(test_users)].copy()
    if 'timestamp' in df_test.columns:
        df_test.sort_values(['user_id', 'timestamp'], inplace=True)
    builder.build_sequences(df_test, stride=None, save_prefix='test')
    
    # Step 9: 计算题目统计特征
    ic("Step 9: 计算题目统计特征 (Difficulty, Discrimination, RT)...")
    f_df = calculate_question_features_full(df)
    normalize_and_save_features(f_df, builder, target_config.PROCESSED_DATA_DIR)
    

    ic("=== Pipeline Complete ===")
    ic(f"=== Output Directory: {target_config.PROCESSED_DATA_DIR} ===")


# ============================================================================
# 使用示例
# ============================================================================
# 1. 全量数据 + 不增强 (默认)
#    python data_process.py --dataset assist09 --ratio 1.0 --stride 0
#
# 2. 采样 10% + 不增强
#    python data_process.py --dataset assist09 --ratio 0.1 --stride 0
#
# 3. 采样 20% + 滑动窗口 (50% 重叠)
#    python data_process.py --dataset assist09 --ratio 0.2 --stride 100
#
# 4. 采样 10% + 滑动窗口 (75% 重叠，强力增强)
#    python data_process.py --dataset assist09 --ratio 0.1 --stride 50
#
# 5. 采样 10% + 最小重叠 (仅切分超长序列)
#    python data_process.py --dataset assist09 --ratio 0.1 --stride 200
# ============================================================================

if __name__ == "__main__":
    main()

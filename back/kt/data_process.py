import sys
import os
import argparse
import json
import toml
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
from data_utils.standard_columns import StandardColumns

try:
    from icecream import ic
except ImportError:
    ic = print

def load_dataset_config(dataset_name):
    """
    加载数据集的 TOML 配置文件
    """
    config_path = os.path.join('config', 'datasets', f'{dataset_name}.toml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return toml.load(f)

def calculate_question_features_full(df):
    """
    计算题目特征 (Difficulty, Discrimination, response_time)
    注意：调用方应传入训练集数据以避免信息泄露；不要包含测试集或验证集交互。
    
    使用标准列名: user_id, question_id, label, response_time
    """
    # 辅助 Score 计算区分度
    # 简单计算：每个用户的全局平均分
    user_scores = df.groupby(StandardColumns.USER_ID)[StandardColumns.LABEL].mean()
    df_aug = df.copy()
    df_aug['user_score'] = df_aug[StandardColumns.USER_ID].map(user_scores)
    
    feature_list = []
    
    # 分组计算
    # 必须列: question_id, label. 可选: response_time
    if StandardColumns.RESPONSE_TIME not in df.columns:
        df_aug[StandardColumns.RESPONSE_TIME] = 0.0

    grouped = df_aug.groupby(StandardColumns.QUESTION_ID)

    for qid, group in tqdm(grouped, desc="Extracting Features"):
        avg_correct = group[StandardColumns.LABEL].mean()
        difficulty = 1.0 - avg_correct

        # response_time
        avg_rt = group[StandardColumns.RESPONSE_TIME].median() # Median usually better for RT

        # Disc
        if len(group) > 5 and group[StandardColumns.LABEL].nunique() > 1:
            disc = group[StandardColumns.LABEL].corr(group['user_score'])
            if pd.isna(disc): disc = 0.0
        else:
            disc = 0.0

        feature_list.append({
            StandardColumns.QUESTION_ID: qid,  # 使用标准列名
            'difficulty': difficulty,
            'discrimination': disc,
            'avg_rt': avg_rt
        })
        
    f_df = pd.DataFrame(feature_list)
    return f_df

def normalize_and_save_features(f_df, builder, output_dir):
    """
    归一化特征并根据 question2idx 保存为 npy
    使用标准列名: qid
    """
    if f_df is None or f_df.empty: return None
    
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
    num_q = len(builder.question2idx) # includes 0
    final_mat = np.zeros((num_q, len(cols)), dtype=np.float32)
    
    f_df = f_df.set_index(StandardColumns.QUESTION_ID)  # 使用标准列名
    
    cnt = 0
    for qid_raw, idx in builder.question2idx.items():
        if idx == 0: continue

        vals = None
        if qid_raw in f_df.index:
            vals = f_df.loc[qid_raw]
        else:
            # Try type cast
            try:
                qid_int = int(qid_raw)
                if qid_int in f_df.index:
                    vals = f_df.loc[qid_int]
            except:
                pass

        if vals is not None:
            if isinstance(vals, pd.DataFrame):
                vals = vals.iloc[0]
            final_mat[idx, 0] = vals.get('difficulty', 0.0)
            final_mat[idx, 1] = vals.get('discrimination', 0.0)
            final_mat[idx, 2] = vals.get('avg_rt', 0.0)
            cnt += 1
            
    save_path = os.path.join(output_dir, 'q_features.npy')
    np.save(save_path, final_mat)
    ic(f"特征矩阵已保存: {save_path} (Matched {cnt}/{num_q-1})")
    
    return {
        "file": "q_features.npy",
        "columns": cols,
        "shape": list(final_mat.shape)
    }

def save_json_mapping(data_dict, output_dir, filename):
    """
    保存映射表为 JSON，兼容 Numpy 类型和字符串 Key
    """
    path = os.path.join(output_dir, filename)
    clean_dict = {}
    for k, v in data_dict.items():
        if hasattr(k, 'item'): k = k.item()
        if isinstance(k, (np.integer, int)): k = str(k)
        if hasattr(v, 'item'): v = v.item()
        clean_dict[k] = v
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(clean_dict, f, indent=4)
    return filename

def main():
    parser = argparse.ArgumentParser(description="KT Data Processing Pipeline (Parquet + Metadata)")
    parser.add_argument('--dataset', type=str, default='assist09', help='Base dataset name')
    parser.add_argument('--ratio', type=float, default=1.0, help='Sample ratio (1.0 for full)')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED, help='Random seed')
    parser.add_argument('--min_seq_len', type=int, default=5, help='Min interactions per user')
    parser.add_argument('--stride', type=int, default=0, help=f'Sliding window stride for TRAIN set. 0 = disable, range: [0, {MAX_SEQ_LEN}]')
    
    args = parser.parse_args()
    # 检查 min_seq_len 合法性
    if args.min_seq_len < 0 or args.min_seq_len >= 200:
        raise ValueError(f"min_seq_len 必须在 [0, 200) 范围内，当前为 {args.min_seq_len}")
    
    # ========== 配置加载 ==========
    base_name = args.dataset
    if args.ratio < 1.0:
        dataset_name = f"{base_name}-sample_{int(args.ratio*100)}%"
    else:
        dataset_name = base_name

    if args.stride < 0 or args.stride > MAX_SEQ_LEN:
        raise ValueError(f"stride must be in range [0, {MAX_SEQ_LEN}], got {args.stride}")
    enable_window = args.stride > 0
    target_dataset_name = f"{dataset_name}_window" if enable_window else dataset_name
    train_stride = args.stride if enable_window else None

    ic(f"=== Pipeline Start: {dataset_name} ===")
    ic(f"滑动窗口: {'ON' if enable_window else 'OFF'} | stride={args.stride}")

    # 加载数据集的 TOML 配置
    dataset_config = load_dataset_config(base_name)
    
    # 获取输出目录配置
    target_config = get_config(target_dataset_name)
    output_dir = target_config.PROCESSED_DATA_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    # ========== Step 1: 数据加载与标准化（配置驱动） ========== 
    ic("Step 1: 数据加载与标准化（Adapter）...")
    try:
        df = KTDataAdapter.load_from_toml(dataset_config)
        ic(f"标准化后数据: {df.shape}")
        # 保存标准化后的数据（全量）
        standard_csv_path = os.path.join(output_dir, f'{dataset_name}_standard.csv')
        KTDataAdapter.save_standard_csv(df, standard_csv_path)
        ic(f"标准化数据已保存: {standard_csv_path}")
    except Exception as e:
        ic(f"数据加载失败: {e}")
        return
    

    # ========== Step 2: 过滤短序列 ========== 
    ic(f"Step 2: 过滤交互数 < {args.min_seq_len} 的用户...")
    user_counts = df[StandardColumns.USER_ID].value_counts()  # 使用标准列名
    valid_users = user_counts[user_counts >= args.min_seq_len].index
    df = df[df[StandardColumns.USER_ID].isin(valid_users)].copy()
    ic(f"过滤后用户数: {df[StandardColumns.USER_ID].nunique()}")


    # ========== Step 3: 分层采样 ========== 
    if args.ratio < 1.0:
        ic(f"Step 3: 分层采样 (ratio={args.ratio})...")
        sampler = KTDataSampler(random_seed=args.seed)
        df = sampler.stratified_sample(df, ratio=args.ratio, user_col=StandardColumns.USER_ID)  # 传入标准列名
        # 保存采样后的标准化 CSV
        sampled_csv_path = os.path.join(output_dir, f'{dataset_name}_standard.csv')
        df.to_csv(sampled_csv_path, index=False, encoding='utf-8')
        ic(f"采样后数据已保存: {sampled_csv_path}")

    # ========== Step 4: 用户级分割 ========== 
    ic("Step 4: 用户级分割 (Train 80% / Test 20%)...")
    splitter = KTDataSplitter(random_seed=args.seed)
    train_users, test_users = splitter.split_users(df, test_ratio=0.2, user_col=StandardColumns.USER_ID)  # 传入标准列名
    ic(f"Train Users: {len(train_users)}, Test Users: {len(test_users)}")
    
    # ========== Step 5: 全景图结构构建 (DataBuilder) ========== 
    ic("Step 5: 全景图结构构建 (DataBuilder)...")
    builder = KTDataBuilder(target_config)
    builder.build_maps(df)  # Builder 内部需要适配标准列名
    
    # ========== Step 6: 序列构建 (Parquet) ========== 
    ic("Step 6: 序列构建 (Parquet)...")
    df_train = df[df[StandardColumns.USER_ID].isin(train_users)].copy()
    if StandardColumns.TIMESTAMP in df_train.columns:
        df_train.sort_values([StandardColumns.USER_ID, StandardColumns.TIMESTAMP], inplace=True)
    train_records = builder.build_sequences(df_train, stride=train_stride, save_prefix='train')
    pd.DataFrame(train_records).to_parquet(os.path.join(output_dir, 'train.parquet'), engine='pyarrow', index=False)

    df_test = df[df[StandardColumns.USER_ID].isin(test_users)].copy()
    if StandardColumns.TIMESTAMP in df_test.columns:
        df_test.sort_values([StandardColumns.USER_ID, StandardColumns.TIMESTAMP], inplace=True)
    test_records = builder.build_sequences(df_test, stride=None, save_prefix='test')
    pd.DataFrame(test_records).to_parquet(os.path.join(output_dir, 'test.parquet'), engine='pyarrow', index=False)
    
    # ========== Step 7: 题目统计特征计算 ========== 
    ic("Step 7: 题目统计特征计算（仅基于训练集）...")
    # 为避免信息泄露，这里仅使用训练集用户的交互数据来计算题目难度/区分度/答题时间等特征
    f_df = calculate_question_features_full(df_train)
    features_meta = normalize_and_save_features(f_df, builder, output_dir)
    
    # ========== Step 8: 统一元数据生成 (metadata.json) ========== 
    ic("Step 8: 统一元数据生成 (metadata.json)...")
    mappings = {
        "skill2idx": save_json_mapping(builder.skill2idx, output_dir, "skill2idx.json"),
        "question2idx": save_json_mapping(builder.question2idx, output_dir, "question2idx.json"),
        "user2idx": save_json_mapping(builder.user2idx, output_dir, "user2idx.json"),
        "skill_domain_map": save_json_mapping(builder.skill_domain_map, output_dir, "skill_domain_map.json")
    }
    
    metadata = {
        "dataset_name": target_dataset_name,
        "metrics": {
            "n_user": len(builder.user2idx),
            "n_question": len(builder.question2idx),
            "n_skill": len(builder.skill2idx),
            "n_domain": len(set(builder.skill_domain_map.values())) if builder.skill_domain_map else 0,
            "avg_seq_len": float(df.groupby(StandardColumns.USER_ID).size().mean()),
            "max_seq_len": MAX_SEQ_LEN
        },
        "config_at_processing": {
            "min_seq_len": args.min_seq_len,
            "max_seq_len": MAX_SEQ_LEN,
            "window_size": args.stride
        },
        "mappings": mappings,
        "features": {
            "q_features": features_meta
        }
    }
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)

    ic("=== Pipeline Complete ===")
    ic(f"=== Output Directory: {output_dir} ===")


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

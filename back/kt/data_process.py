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
from data_utils.inspector import KTDataInspector
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
    
    if StandardColumns.RESPONSE_TIME not in df.columns:
        df_aug[StandardColumns.RESPONSE_TIME] = 0.0

    ic("Extracting Features (Vectorized)...")
    grouped = df_aug.groupby(StandardColumns.QUESTION_ID)
    
    # 1. 向量化计算基础指标（极大提升速度）
    agg_df = grouped.agg(
        difficulty=(StandardColumns.LABEL, lambda x: 1.0 - x.mean()),
        avg_rt=(StandardColumns.RESPONSE_TIME, 'median'),
        count=(StandardColumns.LABEL, 'count'),
        nunique_label=(StandardColumns.LABEL, 'nunique')
    )
    
    # 2. 区分度的计算：只对符合条件的题目操作
    valid_mask = (agg_df['count'] > 5) & (agg_df['nunique_label'] > 1)
    valid_qids = agg_df[valid_mask].index

    # 只过滤出多于5条记录且有不同label的题目进行相关性计算
    if len(valid_qids) > 0:
        valid_df = df_aug[df_aug[StandardColumns.QUESTION_ID].isin(valid_qids)]
        # 用 lambda 过滤计算 corr 还是会比循环所有的快几十倍
        disc_series = valid_df.groupby(StandardColumns.QUESTION_ID).apply(
            lambda g: g[StandardColumns.LABEL].corr(g['user_score'])
        )
        agg_df.loc[valid_qids, 'discrimination'] = disc_series
    
    agg_df['discrimination'] = agg_df.get('discrimination', 0.0).fillna(0.0)
    agg_df = agg_df.reset_index()
    return agg_df[[StandardColumns.QUESTION_ID, 'difficulty', 'discrimination', 'avg_rt']]

def normalize_and_save_features(f_df, builder, output_dir):
    """
    归一化特征并根据 question2idx 保存为 npy
    使用标准列名: qid
    """
    if f_df is None or f_df.empty: return None
    
    # 1. Norm
    cols = ['difficulty', 'discrimination', 'avg_rt']
    
    # 防止 SettingWithCopyWarning，使用显式的复制
    f_df = f_df.copy()
    
    if 'avg_rt' in f_df.columns:
        # Log RT
        f_df.loc[:, 'avg_rt'] = np.log1p(np.clip(f_df['avg_rt'], 0, None))
    
    # Z-Score
    for c in cols:
        if c in f_df.columns:
            m = f_df[c].mean()
            s = f_df[c].std()
            if s == 0: s = 1.0
            f_df.loc[:, c] = (f_df[c] - m) / s
            
    # 2. Map
    num_q = max(builder.question2idx.values()) + 1 # includes 0
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
    parser.add_argument('--train_window_stride', type=int, default=0, help=f'Sliding window stride for TRAIN set. 0 = disable (no overlap), range: [1, {MAX_SEQ_LEN}]')
    parser.add_argument('--test_window_stride', type=int, default=0, help=f'Sliding window stride for TEST set. 0 = default (50% overlap), range: [1, {MAX_SEQ_LEN}]')
    parser.add_argument('--k_core', type=int, default=5, help='K-Core过滤稀疏节点阈值 (默认5, <=0表示关闭)')
    parser.add_argument('--extreme_acc_min', type=int, default=10, help='触发异常正确率100%/0%过滤的最少作答数阈值 (默认10, <=0表示关闭)')
    parser.add_argument('--rapid_merge_sec', type=float, default=3.0, help='过滤连答/脚手架的最短合法秒数间隔 (默认3.0, <=0表示关闭)')
    parser.add_argument('--no_denoise', action='store_true', help='一键关闭所有降噪清洗规则 (K-Core, 逻辑异常, 防时序泄漏)')
    
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

    # 兼容性检查
    if args.train_window_stride < 0 or args.train_window_stride > MAX_SEQ_LEN:
        raise ValueError(f"train_window_stride must be in range [0, {MAX_SEQ_LEN}], got {args.train_window_stride}")
        
    enable_window = args.train_window_stride > 0
    target_dataset_name = f"{dataset_name}_window" if enable_window else dataset_name
    
    # 训练集 Stride 策略：
    # 0 或 >= MAX_SEQ_LEN: 不重叠 (pykt standard)
    # < MAX_SEQ_LEN: 重叠增强 (Ours)
    train_stride_val = args.train_window_stride if enable_window else None

    ic(f"=== Pipeline Start: {dataset_name} ===")
    ic(f"Train Window: {'Overlap' if enable_window else 'Non-Overlap'} | Stride={args.train_window_stride}")
    ic(f"Test Window Stride: {args.test_window_stride} (0=Auto 50%)")

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
    except Exception as e:
        ic(f"数据加载失败: {e}")
        return
    

    # ========== Step 2: 首次过滤短序列 ========== 
    ic(f"Step 2: 过滤交互数 < {args.min_seq_len} 的用户...")
    user_counts = df[StandardColumns.USER_ID].value_counts()  # 使用标准列名
    valid_users = user_counts[user_counts >= args.min_seq_len].index
    df = df[df[StandardColumns.USER_ID].isin(valid_users)].copy()
    ic(f"首次过滤后用户数: {df[StandardColumns.USER_ID].nunique()}")


    # ========== Step 3: 分层采样 ========== 
    if args.ratio < 1.0:
        ic(f"Step 3: 分层采样 (ratio={args.ratio})...")
        sampler = KTDataSampler(random_seed=args.seed)
        df = sampler.stratified_sample(df, ratio=args.ratio, user_col=StandardColumns.USER_ID)  # 传入标准列名

    # ========== Step 3.5: 采样后数据彻底去噪 (Denoiser) ========== 
    # ⚠️ 学术声明 (关于为何在划分前进行全局去噪):
    # 此处在划分 Train/Test 之前执行了全量数据的 K-Core 算法和规则过滤。
    # 严格的学院派理论中，任何依赖 Test 数据分布的超前修剪都可能构成轻微“信息泄漏”。
    # 但在工程与经典学术研究中（如 DKT, GKT 等），通常选择在统一数据可访问域 (Universe of Accessible Interactions) 
    # 中建立稳定的知识图谱拓扑基线。本管线的去噪目的是为了剥离系统本身的物理缺陷噪音（机器刷单、100%正确作弊题）。
    # 防止被采样砍出的网络破尾会导致极多新孤岛和新 100%正确率题目。
    from data_utils.denoiser import KTDataDenoiser
    if not args.no_denoise:
        df = KTDataDenoiser.run_denoise_pipeline(
            df, 
            k_core=args.k_core, 
            extreme_acc_min=args.extreme_acc_min, 
            rapid_merge_sec=args.rapid_merge_sec
        )
    else:
        ic("⚠️ Warning: 所有降噪(Denoiser)规则已通过 --no_denoise 参数一键禁用")
        
    # ========== Step 3.6: 去噪后再次过滤短序列 (防破坏兜底) ==========
    if not args.no_denoise:
        user_counts_after = df[StandardColumns.USER_ID].value_counts()
        valid_users_after = user_counts_after[user_counts_after >= args.min_seq_len].index
        dropped_count = len(df[StandardColumns.USER_ID].unique()) - len(valid_users_after)
        if dropped_count > 0:
            df = df[df[StandardColumns.USER_ID].isin(valid_users_after)].copy()
            ic(f"⚠️ 兜底清理: 去噪管线导致 {dropped_count} 名用户序列在清洗后过短 (<{args.min_seq_len})，已被移除。")
        else:
            ic("✅ 兜底检测: 未发现因去噪导致的短序列残缺。")
            
    # 为了防止后续 builder 和 diff 的随机时序异常，此节点强制开启全局排序
    if StandardColumns.TIMESTAMP in df.columns:
        ic("-> 强制时序全局对齐 (按 UserID -> Timestamp)...")
        # 此时强行转换为数值以防万一
        df[StandardColumns.TIMESTAMP] = pd.to_numeric(df[StandardColumns.TIMESTAMP], errors='coerce')
        df.sort_values([StandardColumns.USER_ID, StandardColumns.TIMESTAMP], inplace=True)
    
    # 将标准化的数据只在处理好后（即短序列过滤/采样完成后）进行统一落盘，避免百万级 IO 导致缓慢
    standard_csv_path = os.path.join(output_dir, f'{dataset_name}_standard.csv')
    KTDataAdapter.save_standard_csv(df, standard_csv_path)
    ic(f"核心清洗后数据集已落盘: {standard_csv_path}")

    # ========== Step 4: 用户级分割 (序列级无偏划分) ========== 
    ic("Step 4: 用户级分割 (Train 80% / Test 20%)...")
    # ⚠️ 关于冷启动泄漏/模型瞎猜: 此处采用了纯净的以用户为单位的割接 (User-Level Split)。
    # 目的：严格考察模型对于没见过的陌生人的推理和部分题目零样本(Zero-Shot)的能力。这也是知识追踪领域测试冷启动的标准做法。
    splitter = KTDataSplitter(random_seed=args.seed)
    train_users, test_users = splitter.split_users(df, test_ratio=0.2, min_seq_len=args.min_seq_len, user_col=StandardColumns.USER_ID)  # 传入管线的 min_seq_len 保持一致
    ic(f"Train Users: {len(train_users)}, Test Users: {len(test_users)}")
    
    # ========== Step 5: 全景图结构构建 (DataBuilder) ========== 
    ic("Step 5: 全景图结构构建 (DataBuilder)...")
    builder = KTDataBuilder(target_config)
    builder.build_maps(df)  # Builder 内部需要适配标准列名
    
    # ========== Step 6: 序列构建 (Parquet) ========== 
    ic("Step 6: 序列构建 (Parquet) 并计算相对时序特征...")
    df_train = df[df[StandardColumns.USER_ID].isin(train_users)].copy()
    # 训练集: 如果 train_window_stride=0，则传入 None，Builder 内部会自动处理为 Non-overlap
    train_records = builder.build_sequences(df_train, stride=train_stride_val, save_prefix='train')
    pd.DataFrame(train_records).to_parquet(os.path.join(output_dir, 'train.parquet'), engine='pyarrow', index=False)

    # ⚠️ 防特征泄漏：提取训练集的 question_avg_time，供测试集复用
    # 测试集的 t_response 特征必须基于训练集的中位答题时间，而非测试集自身的统计量
    train_question_avg_time = builder.get_question_avg_time(df_train)
    
    df_test = df[df[StandardColumns.USER_ID].isin(test_users)].copy()
    
    # 解析测试集 stride
    # 如果 args.test_window_stride 为 0，则默认使用 50% 重叠 (MAX_SEQ_LEN // 2) - 工程默认
    # 如果想要类似 pykt 的 "window" 模式（密集评估），请传入 1
    # 如果想要类似 pykt 的 "original" 模式（无重叠），请传入 200 (MAX_SEQ_LEN)
    real_test_stride = args.test_window_stride if args.test_window_stride > 0 else (MAX_SEQ_LEN // 2)
    ic(f"Testing Stride Config: {real_test_stride}")

    # 按照严格的评测标准，测试集应将超长用户按 50% 重叠的滑动窗口(MAX_SEQ_LEN//2) 切开，依靠 builder 内部的 eval_mask 解决状态截断导致的冷启动。
    test_records = builder.build_sequences(df_test, stride=real_test_stride, save_prefix='test', question_avg_time_override=train_question_avg_time)
    pd.DataFrame(test_records).to_parquet(os.path.join(output_dir, 'test.parquet'), engine='pyarrow', index=False)
    

    # ========== 数据深度体检 (Sanity Check) ==========
    ic("执行数据深度体检...")
    # NOTE: config is passed as None temporarily, or create a mock. The script uses no config.
    class MockConfig:
        class path:
            REPORT_DIR = os.path.join(output_dir, "reports")
    inspector = KTDataInspector(df, MockConfig())
    inspector.run_full_sanity_check()
    # 判断泄漏
    train_pq = pd.DataFrame(train_records)
    test_pq = pd.DataFrame(test_records)
    KTDataInspector.check_train_test_leakage_and_cold_start(train_pq, test_pq)

    # ========== Step 7: 题目统计特征计算 ========== 
    ic("Step 7: 题目统计特征计算（仅基于训练集）...")
    # 为避免信息泄露，这里仅使用训练集用户的交互数据来计算题目难度/区分度/答题时间等特征
    f_df = calculate_question_features_full(df_train)
    features_meta = normalize_and_save_features(f_df, builder, output_dir)
    
    # ========== Step 8: 统一元数据生成 (metadata.json) ========== 
    ic("Step 8: 统一元数据生成 (metadata.json)...")
    
    # --- 1. 计算扩展统计指标 ---
    n_interactions = len(df)
    n_unique_q = df[StandardColumns.QUESTION_ID].nunique()
    
    avg_attempts_per_question = float(n_interactions / n_unique_q) if n_unique_q > 0 else 0.0
    global_acc = float(df[StandardColumns.LABEL].mean()) if StandardColumns.LABEL in df.columns else 0.0
    sparsity = float(1.0 - n_interactions / (len(builder.user2idx) * max(builder.question2idx.values()))) if len(builder.user2idx) > 0 and max(builder.question2idx.values()) > 0 else 1.0

    avg_skills_per_q = 0.0
    avg_q_per_skill = 0.0
    avg_attempts_per_skill = 0.0

    if StandardColumns.SKILL_IDS in df.columns:
        valid_skills_df = df.dropna(subset=[StandardColumns.SKILL_IDS])
        if not valid_skills_df.empty:
            # 扁平化技能列
            skill_series = valid_skills_df[StandardColumns.SKILL_IDS].astype(str).str.split(StandardColumns.SKILL_IDS_STD_SEP)
            df_exploded = pd.DataFrame({
                'q': valid_skills_df[StandardColumns.QUESTION_ID],
                's': skill_series
            }).explode('s')
            # 过滤无效空串
            df_exploded = df_exploded[df_exploded['s'].str.strip() != '']
            df_exploded = df_exploded[df_exploded['s'].notna()]
            
            if not df_exploded.empty:
                n_unique_s = df_exploded['s'].nunique()
                avg_skills_per_q = float(df_exploded.groupby('q')['s'].nunique().mean())
                avg_q_per_skill = float(df_exploded.groupby('s')['q'].nunique().mean())
                avg_attempts_per_skill = float(len(df_exploded) / n_unique_s) if n_unique_s > 0 else 0.0

    # --- 2. 获取源文件路径 ---
    src_file_path = dataset_config.get("file_path", "")

    mappings = {
        "skill2idx": save_json_mapping(builder.skill2idx, output_dir, "skill2idx.json"),
        "question2idx": save_json_mapping(builder.question2idx, output_dir, "question2idx.json"),
        "user2idx": save_json_mapping(builder.user2idx, output_dir, "user2idx.json"),
        "skill_domain_map": save_json_mapping(builder.skill_domain_map, output_dir, "skill_domain_map.json")
    }
    
    metadata = {
        "dataset_name": target_dataset_name,
        "src_file_path": src_file_path,
        "metrics": {
            "n_interactions": n_interactions,
            "n_user": len(builder.user2idx),
            "n_question": max(builder.question2idx.values()) + 1,
            "n_skill": max(builder.skill2idx.values()) + 1,
            "n_domain": len(set(builder.skill_domain_map.values())) if builder.skill_domain_map else 0,
            "avg_seq_len": float(df.groupby(StandardColumns.USER_ID).size().mean()),
            "max_seq_len": MAX_SEQ_LEN,
            "global_accuracy": global_acc,
            "sparsity_ratio": sparsity,
            "avg_attempts_per_question": avg_attempts_per_question,
            "avg_attempts_per_skill": avg_attempts_per_skill,
            "avg_skills_per_question": avg_skills_per_q,
            "avg_questions_per_skill": avg_q_per_skill
        },
        "config_at_processing": {
            "min_seq_len": args.min_seq_len,
            "max_seq_len": MAX_SEQ_LEN,
            "train_window_stride": args.train_window_stride,
            "test_window_stride": args.test_window_stride,
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
#    python data_process.py --dataset assist09 --ratio 1.0 --stride 0 --min_seq_len 20
#
# 2. 采样 10% + 不增强
#    python data_process.py --dataset assist09 --ratio 0.1 --stride 0 --min_seq_len 20
#
# 3. 采样 20% + 滑动窗口 (50% 重叠)
#    python data_process.py --dataset assist09 --ratio 0.2 --stride 100
#
# 4. 采样 10% + 滑动窗口 (75% 重叠，强力增强)
#    python data_process.py --dataset assist09 --ratio 0.1 --stride 50
#
# 5. 采样 10% + 最小重叠 (仅切分超长序列)
#    python data_process.py --dataset assist09 --ratio 0.1 --stride 200
# 
# 6. 一键关闭数据降噪 (用于对比降噪前后的性能差异)
#    python data_process.py --dataset assist09 --no_denoise
# ============================================================================

# === 1. 复现 pykt Baseline (0.75) ===
# 训练无重叠，测试无重叠
# python data_process.py --dataset assist09 --train_window_stride 0 --test_window_stride 200

# === 2. 复现 pykt Window Evaluaton ===
# 训练无重叠，测试如其名 "Window" (极其密集，慎用)
# python data_process.py --dataset assist09 --train_window_stride 0 --test_window_stride 1

# === 3. 你的 SOTA 配置 (0.83) ===
# 训练 50% 重叠增强，测试 50% 重叠评估
# 这种配置在工程落地和学术竞赛中是最常用的 Trick
# python data_process.py --dataset assist09 --train_window_stride 100 --test_window_stride 100
if __name__ == "__main__":
    main()

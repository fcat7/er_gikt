import sys
import os
import argparse
import json
import toml
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import numpy as np

# Self modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import get_config, RANDOM_SEED, MAX_SEQ_LEN
from data_utils.adapter import KTDataAdapter
from data_utils.builder import KTDataBuilder
from data_utils.inspector import KTDataInspector
from data_utils.standard_columns import StandardColumns
from data_process import calculate_question_features_full, normalize_and_save_features, save_json_mapping, load_dataset_config

try:
    from icecream import ic
except ImportError:
    ic = print

def main():
    parser = argparse.ArgumentParser(description="Process GIKT 原版划分数据并打入 Parquet 流程")
    parser.add_argument('--dataset', type=str, default='assist09_gikt_old', help='Dataset config (e.g. assist09_gikt_old)')
    parser.add_argument('--stride', type=int, default=0, help=f'Sliding window stride for TRAIN set.')
    
    args = parser.parse_args()
    
    dataset_name = args.dataset
    enable_window = args.stride > 0
    target_dataset_name = f"{dataset_name}_window" if enable_window else dataset_name
    train_stride = args.stride if enable_window else None

    ic(f"=== GIKT Raw Process Start: {dataset_name} ===")
    
    # 1. 加载配置
    dataset_config = load_dataset_config(dataset_name)
    target_config = get_config(target_dataset_name)
    temp_config = dataset_config.copy()
    
    # 提取 train 和 test paths
    train_path = temp_config.get('train_file_path')
    test_path = temp_config.get('test_file_path')
    if not train_path or not os.path.exists(train_path):
        raise ValueError(f"未找到训练文件 {train_path}")
    if not test_path or not os.path.exists(test_path):
        raise ValueError(f"未找到测试文件 {test_path}")
    
    # 使用 KTDataAdapter 完成列名修正与类型解析
    def load_mapped_df(csv_path):
        _config = temp_config.copy()
        _config['file_path'] = csv_path
        
        # 修复：KTDataAdapter 的 load_from_toml 是静态方法，且内部包含了 load -> rename -> filters -> convert 完整链路
        # 为了避免触发原有 toml 中的 [filters]（因为原版数据可能已经处理过，或者列名不一致），
        # 我们这里直接调用 load_from_toml。
        df = KTDataAdapter.load_from_toml(_config)
        return df
        
    ic("正在加载原版训练集...")
    df_train = load_mapped_df(train_path)
    ic("正在加载原版测试集...")
    df_test = load_mapped_df(test_path)
    
    # 添加强标记方便合并后拆解
    df_train['split_flag'] = 'train'
    df_test['split_flag'] = 'test'
    
    # 纵向合并以共享映射空间和全局属性
    df = pd.concat([df_train, df_test], ignore_index=True)
    ic(f"合并后总交互条数: {len(df)}")
    
    output_dir = target_config.PROCESSED_DATA_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    # 【重点：不能使用时序排序，必须维持原始提取的行列顺序，否则影响原版数据序列化逻辑】
    standard_csv_path = os.path.join(output_dir, f'{dataset_name}_standard.csv')
    KTDataAdapter.save_standard_csv(df.drop(columns=['split_flag']), standard_csv_path)
    ic(f"记录统一存档: {standard_csv_path}")

    # ========== 全景图结构构建 (DataBuilder) ========== 
    ic("构建全局拓扑映射 (Builder)...")
    builder = KTDataBuilder(target_config)
    builder.build_maps(df)
    
    # ========== 切分回 Train / Test ==========
    df_train_final = df[df['split_flag'] == 'train'].drop(columns=['split_flag']).copy()
    df_test_final  = df[df['split_flag'] == 'test'].drop(columns=['split_flag']).copy()
    
    ic("构建 Parquet 序列文件...")
    train_records = builder.build_sequences(df_train_final, stride=train_stride, save_prefix='train')
    pd.DataFrame(train_records).to_parquet(os.path.join(output_dir, 'train.parquet'), engine='pyarrow', index=False)
    
    train_question_avg_time = builder.get_question_avg_time(df_train_final)
    
    test_records = builder.build_sequences(df_test_final, stride=MAX_SEQ_LEN // 2, save_prefix='test', question_avg_time_override=train_question_avg_time)
    pd.DataFrame(test_records).to_parquet(os.path.join(output_dir, 'test.parquet'), engine='pyarrow', index=False)
    
    # ========== 计算题目特征 ==========
    ic("计算辅助特征 (基于纯训练集)...")
    f_df = calculate_question_features_full(df_train_final)
    features_meta = normalize_and_save_features(f_df, builder, output_dir)
    
    # ========== 生成统一元数据字典 ==========
    n_interactions = len(df)
    n_unique_q = df[StandardColumns.QUESTION_ID].nunique()
    global_acc = float(df[StandardColumns.LABEL].mean()) if StandardColumns.LABEL in df.columns else 0.0

    avg_skills_per_q, avg_q_per_skill, avg_attempts_per_skill = 0.0, 0.0, 0.0
    if StandardColumns.SKILL_IDS in df.columns:
        valid_skills_df = df.dropna(subset=[StandardColumns.SKILL_IDS])
        if not valid_skills_df.empty:
            skill_series = valid_skills_df[StandardColumns.SKILL_IDS].astype(str).str.split(StandardColumns.SKILL_IDS_STD_SEP)
            df_exploded = pd.DataFrame({'q': valid_skills_df[StandardColumns.QUESTION_ID], 's': skill_series}).explode('s')
            df_exploded = df_exploded[df_exploded['s'].str.strip() != '']
            df_exploded = df_exploded[df_exploded['s'].notna()]
            
            if not df_exploded.empty:
                n_unique_s = df_exploded['s'].nunique()
                avg_skills_per_q = float(df_exploded.groupby('q')['s'].nunique().mean())
                avg_q_per_skill = float(df_exploded.groupby('s')['q'].nunique().mean())
                avg_attempts_per_skill = float(len(df_exploded) / n_unique_s) if n_unique_s > 0 else 0.0

    mappings = {
        "skill2idx": save_json_mapping(builder.skill2idx, output_dir, "skill2idx.json"),
        "question2idx": save_json_mapping(builder.question2idx, output_dir, "question2idx.json"),
        "user2idx": save_json_mapping(builder.user2idx, output_dir, "user2idx.json"),
        "skill_domain_map": save_json_mapping(builder.skill_domain_map, output_dir, "skill_domain_map.json")
    }
    
    metadata = {
        "dataset_name": target_dataset_name,
        "src_file_path": train_path + " | " + test_path,
        "metrics": {
            "n_interactions": n_interactions,
            "n_user": len(builder.user2idx),
            "n_question": max(builder.question2idx.values()) + 1,
            "n_skill": max(builder.skill2idx.values()) + 1,
            "n_domain": len(set(builder.skill_domain_map.values())) if builder.skill_domain_map else 0,
            "global_accuracy": global_acc,
            "avg_skills_per_question": avg_skills_per_q,
            "avg_questions_per_skill": avg_q_per_skill
        },
        "config_at_processing": {
            "max_seq_len": MAX_SEQ_LEN,
            "window_size": args.stride,
            "mode": "Raw Split Preserved & Parquet Standardized"
        },
        "mappings": mappings,
        "features": {
            "q_features": features_meta
        }
    }
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)

    ic("=== GIKT 原版划分打入 Parquet 管道处理完全结束 ===")
    ic(f"输出路径: {output_dir}")

# python data_process_gikt_raw.py --dataset ednet_kt1_gikt_old
if __name__ == "__main__":
    main()

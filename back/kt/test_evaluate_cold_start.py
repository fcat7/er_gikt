#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型冷启动（序列长度分桶）测试评价脚本
用于第三章实验分析：探讨图结构对短序列（冷启动）的兜底鲁棒性
python test_evaluate_cold_start.py --auto --checkpoint_dir "H:\er_gikt\back\kt\checkpoint-min20" --data_dir H:\er_gikt\back\kt\data\bak
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from sklearn import metrics
import argparse
from collections import defaultdict
import datetime
from torch.utils.data import DataLoader
import toml

# 确保能导入内部模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from dataset import UnifiedParquetDataset, SeqFeatureKey
from models.factory import ModelFactory
def get_available_checkpoints(checkpoint_dir=None):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(current_dir, 'checkpoint')
    available_combos = []
    known_models = [
        'deep_irt', 'simplekt', 'gikt_old',
        'dkvmn', 'qikt', 'gikt', 'dkt', 'akt'
    ]
    if os.path.exists(checkpoint_dir):
        for f in os.listdir(checkpoint_dir):
            if f.endswith('_best.pt'):
                prefix = f.replace('_best.pt', '')
                matched_model = None
                for m in sorted(known_models, key=len, reverse=True):
                    if prefix.startswith(m + '_'):
                        matched_model = m
                        break
                if matched_model:
                    dataset = prefix[len(matched_model)+1:]
                    available_combos.append((matched_model, dataset))
    return available_combos

# ==========================================
# 专家推荐的序列长度划分 (Cold-Start Bins)
# ==========================================
# 1. 极端冷启动 (3-10): 交互极少，传统RNN/Transformer模型极易崩溃。
# 2. 中度冷启动 (11-30): 具备部分历史，但仍不足以精准刻画。
# 3. 常规长度 (31-100): 数据相对丰富，模型性能通常在这个区间达到平台期。
# 4. 丰富序列 (>100): 充分交互数据。
BINS = [
    (3, 10, "3-10 (Extreme Cold)"),
    (11, 30, "11-30 (Moderate Cold)"),
    (31, 100, "31-100 (Normal)"),
    (101, float('inf'), ">100 (Data Rich)")
]


def resolve_processed_data_dir(data_dir, dataset_name):
    """解析实际的数据目录。

    支持两种传参方式：
    1. 传入数据根目录，如 H:\\er_gikt\\back\\kt\\data\\bak
       -> 自动拼接为 H:\\...\\bak\\<dataset_name>
    2. 直接传入某个数据集目录，如 H:\\...\\bak\\assist09-min20
       -> 直接使用该目录
    """
    if not data_dir:
        return None

    data_dir = os.path.abspath(data_dir)
    direct_metadata = os.path.join(data_dir, 'metadata.json')
    nested_dir = os.path.join(data_dir, dataset_name)
    nested_metadata = os.path.join(nested_dir, 'metadata.json')

    if os.path.exists(direct_metadata):
        return data_dir
    if os.path.exists(nested_metadata):
        return nested_dir

    # 即使 metadata 还不存在，也优先按“root/dataset_name”语义兜底
    if os.path.basename(data_dir).lower() == dataset_name.lower():
        return data_dir
    return nested_dir

def calculate_auc(targets, preds):
    """单独计算 AUC 且防报错处理"""
    if len(preds) < 2 or len(set(targets)) < 2:
        return 0.0
    targets = np.array(targets)
    preds = np.array(preds)
    return metrics.roc_auc_score(targets, preds)

def evaluate_cold_start(model, dataloader, device, use_amp=True):
    """按序列长度分桶验证循环"""
    model.eval()
    
    bin_preds = {b_label: [] for _, _, b_label in BINS}
    bin_targets = {b_label: [] for _, _, b_label in BINS}
    bin_counts = {b_label: 0 for _, _, b_label in BINS}
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            features = {k: v.to(device) for k, v in batch.items()}
            question = features[SeqFeatureKey.Q].to(torch.long)
            response = features[SeqFeatureKey.R].to(torch.long)
            mask = features[SeqFeatureKey.MASK].to(torch.bool)
            eval_mask = features.get(SeqFeatureKey.EVAL_MASK, mask).to(torch.bool)
            interval = features[SeqFeatureKey.T_INTERVAL].to(torch.float32)
            r_time = features[SeqFeatureKey.T_RESPONSE].to(torch.float32)
            
            interval = torch.nan_to_num(interval, nan=0.0)
            r_time = torch.nan_to_num(r_time, nan=0.0)

            model_name = getattr(model, 'model_name', getattr(model, '_get_name', lambda: '')()).lower()
            cognitive_mode = getattr(model, 'cognitive_mode', None)
            
            if not model_name or model_name == 'module':
                model_name = model.__class__.__name__.lower()

            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                if model_name == 'dkt' or 'dkt' in model_name and 'lbkt' not in model_name:
                    y_hat = model(question, response, mask)
                    preds = torch.sigmoid(y_hat[:, :-1])
                    preds = torch.nan_to_num(preds, nan=0.0)
                elif model_name in ['dkvmn', 'akt', 'simplekt', 'qikt', 'lbkt']:
                    y_hat = model(question, response, mask, interval, r_time)
                    y_hat = torch.sigmoid(y_hat)
                    preds = y_hat[:, :-1] if y_hat.shape[1] == question.shape[1] else y_hat
                elif model_name in ['gikt', 'gikt_old'] or cognitive_mode == 'classic':
                    y_hat = model(question, response, mask, interval, r_time)
                    y_hat = torch.sigmoid(y_hat)
                    preds = y_hat[:, 1:]
                else:
                    try:
                        y_hat = model(question, response, mask, interval, r_time)
                    except:
                        y_hat = model(question, response, mask)
                    y_hat = torch.sigmoid(y_hat)
                    preds = y_hat[:, :-1] if y_hat.shape[1] == question.shape[1] else y_hat

            targets = response[:, 1:].float()
            mask_valid = mask[:, 1:]
            eval_mask_valid = eval_mask[:, 1:]
            
            # 按 Batch 里的每条学生序列拆解
            current_batch_size = question.shape[0]
            for i in range(current_batch_size):
                # 利用原始 mask (包含起始位) 计算此学生的真实序列交互长度
                actual_len = mask[i].sum().item()
                
                # 寻找匹配的区间
                matched_label = None
                for b_min, b_max, b_label in BINS:
                    if b_min <= actual_len <= b_max:
                        matched_label = b_label
                        break
                
                if matched_label is None:
                    continue
                    
                student_final_mask = mask_valid[i] & eval_mask_valid[i]
                if student_final_mask.sum() > 0:
                    p = preds[i][student_final_mask].cpu().numpy()
                    t = targets[i][student_final_mask].cpu().numpy()
                    
                    if not (np.isnan(p).any() or np.isinf(p).any()):
                        bin_preds[matched_label].extend(p)
                        bin_targets[matched_label].extend(t)
                        bin_counts[matched_label] += 1

    # 计算各区间的 AUC
    results = {}
    for _, _, b_label in BINS:
        auc_val = calculate_auc(bin_targets[b_label], bin_preds[b_label])
        results[b_label] = {
            'auc': auc_val,
            'samples': len(bin_targets[b_label]),
            'users': bin_counts[b_label]
        }
    return results

def test_cold_start_and_report(tasks, batch_size=128, result_csv="cold_start_results.csv", force=False, checkpoint_dir=None, data_dir=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 开始切入第三章冷启动实验。使用设备: {device}")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results = []
    
    output_path = os.path.join(current_dir, 'output', result_csv)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    existing_records = set()
    if os.path.exists(output_path) and not force:
        try:
            df_old = pd.read_csv(output_path)
            for _, row in df_old.iterrows():
                existing_records.add((str(row.get('Dataset', '')).lower(), str(row.get('Model', '')).lower()))
        except Exception:
            pass

    tasks_by_dataset = defaultdict(list)
    for t in tasks:
        if t['model'] not in tasks_by_dataset[t['dataset']]:
            tasks_by_dataset[t['dataset']].append(t['model'])

    for dataset_name, models_to_test in tasks_by_dataset.items():
        print(f"\n{'='*60}")
        print(f"📦 测试集: {dataset_name} (自动解析序列长度)")
        print(f"{'='*60}")
        
        try:
            config = get_config(dataset_name)
            if data_dir:
                resolved_data_dir = resolve_processed_data_dir(data_dir, dataset_name)
                config.path.PROCESSED_DATA_ROOT = os.path.dirname(resolved_data_dir)
                config.DATASET_NAME = os.path.basename(resolved_data_dir)
                print(f"📁 强制覆盖数据目录: {config.PROCESSED_DATA_DIR}")
            metadata_path = os.path.join(config.PROCESSED_DATA_DIR, 'metadata.json')
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"❌ 加载配置文件或 metadata 失败: {e}")
            continue
            
        num_q = metadata['metrics']['n_question']
        num_c = metadata['metrics']['n_skill']
        test_loader = None

        for model_name in models_to_test:
            print(f"\n  >> 正在评估: {model_name.upper()}")
            if not force and (dataset_name.lower(), model_name.lower()) in existing_records:
                print(f"    🌟 该组合已有分桶成绩，跳过 (使用 --force 覆盖)")
                continue
            
            ckpt_base = checkpoint_dir if checkpoint_dir else os.path.join(current_dir, 'checkpoint')
            ckpt_path = os.path.join(ckpt_base, f"{model_name.lower()}_{dataset_name.lower()}_best.pt")
            if not os.path.exists(ckpt_path):
                print(f"    ✗ 错误: 找不到权重 {ckpt_path}")
                continue
                
            if test_loader is None:
                print("    ⏳ 加载并初始化测试集 DataLoader (不需要剔除小序列)...")
                test_dataset = UnifiedParquetDataset(config, augment=False, mode='test')
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            try:
                # ----------------加载模型字典与超参----------------
                kwargs = {}
                if model_name.lower() in ['gikt', 'gikt_old']:
                    exp_config_path = os.path.join(current_dir, 'config', 'experiments', 'exp_full_default.toml')
                    if os.path.exists(exp_config_path):
                        gikt_config = toml.load(exp_config_path)
                        if 'model' in gikt_config:
                            kwargs.update(gikt_config['model'])
                if data_dir:
                    kwargs['data_dir'] = config.PROCESSED_DATA_DIR
                
                best_params_path = os.path.join(current_dir, 'config', 'best_params', f"{model_name.lower()}_best_params.json")
                if os.path.exists(best_params_path):
                    with open(best_params_path, 'r', encoding='utf-8') as f:
                        best_params = json.load(f)
                    matched_key = next((k for k in best_params.keys() if k in dataset_name.lower() or dataset_name.lower() in k), 'default')
                    if matched_key in best_params:
                        kwargs.update(best_params[matched_key])

                model = ModelFactory.get_model(model_name=model_name, num_question=num_q, num_skill=num_c, device=device, config=config, **kwargs)
                state_dict = torch.load(ckpt_path, map_location=device)
                if isinstance(state_dict, torch.nn.Module):
                    state_dict = state_dict.state_dict()
                
                clean_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
                model.load_state_dict(clean_state_dict, strict=False)
                
                # ----------------执行区段评测----------------
                print("    🚀 开始进行区间穿越推断...")
                bin_res = evaluate_cold_start(model, test_loader, device=device)
                
                row_data = {
                    "Dataset": dataset_name,
                    "Model": model_name.upper(),
                    "Update_Time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                }
                
                print(f"    📊 成绩拆解:")
                for _, _, b_label in BINS:
                    info = bin_res[b_label]
                    auc_val = info['auc']
                    row_data[b_label] = round(auc_val, 4) if auc_val > 0 else 0.0
                    print(f"       + {b_label:<25} | AUC: {auc_val:.4f}  (答题对数: {info['samples']}, 涵盖学生数: {info['users']})")
                
                results.append(row_data)

            except Exception as e:
                print(f"    ✗ 错误记录: {e}")

    # 合并输出
    if results:
        df_new = pd.DataFrame(results)
        if os.path.exists(output_path) and not force:
            df_old = pd.read_csv(output_path)
            df_merged = pd.concat([df_old, df_new], ignore_index=True)
            df_merged.drop_duplicates(subset=['Dataset', 'Model'], keep='last', inplace=True)
            df_merged.to_csv(output_path, index=False)
            print("\n📈 [冷启动评估] 最新全景排行榜:")
            print(df_merged.to_string(index=False))
        else:
            df_new.to_csv(output_path, index=False)
            print("\n📈 [冷启动评估] 成绩单:")
            print(df_new.to_string(index=False))
            
        print(f"\n✅ 评测报告存至: {output_path}。可用于第三章《极端冷启动场景下模型鲁棒性与先验图结构分析》的数据支撑。")

def parse_args():
    parser = argparse.ArgumentParser(description="冷启动(按序列长度段)分析测试")
    parser.add_argument('--models', type=str, nargs='+', default=[], help='要测试的模型列表')
    parser.add_argument('--datasets', type=str, nargs='+', default=[], help='涉及数据集')
    parser.add_argument('--batch_size', type=int, default=128, help='批量')
    parser.add_argument('--output', type=str, default='cold_start_results.csv', help='输出文件名')
    parser.add_argument('--auto', action='store_true', help='自动拉取 checkpoint 跑分')
    parser.add_argument('--force', action='store_true', help='强制覆盖已有成绩')
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='指定模型权重目录')
    parser.add_argument('--data_dir', type=str, default=None, help='指定处理后的数据目录，如 test.parquet / metadata.json 所在目录')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    tasks = []
    
    if args.auto:
        for m, d in get_available_checkpoints(checkpoint_dir=args.checkpoint_dir):
            tasks.append({'model': m, 'dataset': d})
            
    if args.models and args.datasets:
        for d in args.datasets:
            for m in args.models:
                tasks.append({'model': m, 'dataset': d})
                
    if not tasks:
        print("💡 请使用 --auto 或者传入 --models和--datasets！")
        sys.exit(1)
        
    test_cold_start_and_report(
        tasks=tasks,
        batch_size=args.batch_size,
        result_csv=args.output,
        force=args.force,
        checkpoint_dir=args.checkpoint_dir,
        data_dir=args.data_dir,
    )
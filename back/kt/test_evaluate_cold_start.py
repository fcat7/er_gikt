#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型冷启动（序列长度分桶）测试评价脚本（V2）

升级点（对齐 test_evaluate.py）：
1) 路径与模型加载更稳健：严格按评估 JSON 清单加载。
2) 评估口径统一：固定协议、eval_mask、生效 batch 统计、AMP 开关可控。
3) 输出增强：CSV 记录时间、吞吐、分桶样本/用户与 AUC，支持增量去重。
"""

import argparse
import datetime
import json
import os
import random
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import toml
import torch
from sklearn import metrics
from torch.utils.data import DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from config import get_config
from dataset import UnifiedParquetDataset, SeqFeatureKey
from models.factory import ModelFactory


KNOWN_MODEL_PREFIXES = [
    'deep_irt', 'simplekt', 'gikt_old', 'dkvmn', 'qikt', 'gikt', 'dkt', 'akt', 'lbkt'
]
BASELINE_MODELS = {'dkt', 'dkvmn', 'akt', 'simplekt', 'deep_irt', 'qikt', 'lbkt'}
GIKT_MODELS = {'gikt', 'gikt_old'}

DISPLAY_NAME_MAP = {
    'dkt': 'DKT', 'dkvmn': 'DKVMN', 'akt': 'AKT', 'simplekt': 'SimpleKT',
    'deep_irt': 'Deep-IRT', 'qikt': 'QIKT', 'lbkt': 'LBKT', 'gikt_old': 'GIKT', 'gikt': 'FA-GIKT'
}

ABLATION_OVERRIDES = {
    'A_Baseline': {},
    'B_Remove_PID': {'use_pid': False},
    'C_Remove_Cognitive': {'use_cognitive_model': False},
    'D_Remove_IRT': {'use_4pl_irt': False},
    'E_agg_method-kk_gat': {'agg_method': 'kk_gat'},
    'F_old_gikt': {'use_pid': False, 'use_cognitive_model': False, 'use_4pl_irt': False},
    'G_remove_PID_Cognitive': {'use_pid': False, 'use_cognitive_model': False},
    'H_remove_PID_IRT': {'use_pid': False, 'use_4pl_irt': False},
    'I_remove_Cognitive_IRT': {'use_cognitive_model': False, 'use_4pl_irt': False},
}

BINS = [
    (3, 10, '3-10 (Extreme Cold)', 'bin_3_10'),
    (11, 30, '11-30 (Moderate Cold)', 'bin_11_30'),
    (31, 100, '31-100 (Normal)', 'bin_31_100'),
    (101, float('inf'), '>100 (Data Rich)', 'bin_100p'),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='冷启动(按序列长度分桶)统一评估脚本 V2')
    parser.add_argument('--config', type=str, default=os.path.join(CURRENT_DIR, 'config', 'evaluation', 'unified_acc_reeval_models.json'), help='评估清单 JSON 路径')
    parser.add_argument('--dataset', type=str, default='', help='仅评估指定数据集')
    parser.add_argument('--model_name', type=str, default='', help='仅评估指定模型')
    parser.add_argument('--ablation_group', type=str, default='', help='仅评估指定消融组（FA-GIKT）')
    parser.add_argument('--family', type=str, choices=['all', 'baseline', 'gikt'], default='all', help='模型家族过滤')
    parser.add_argument('--batch_size', type=int, default=128, help='测试批大小')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader num_workers')
    parser.add_argument('--use_amp', action='store_true', help='测试时启用 AMP（仅 CUDA 生效）')
    parser.add_argument('--output', type=str, default='', help='输出文件名或路径；默认自动追加时间戳，避免覆盖')
    parser.add_argument('--force', action='store_true', help='强制覆盖（不跳过已有记录）')
    parser.add_argument('--data_dir', type=str, default=None, help='处理后数据目录（root 或 dataset 目录）')
    parser.add_argument('--seed', type=int, default=42, help='评估随机种子')
    parser.add_argument('--shard_index', type=int, default=0, help='并行分片编号（从 0 开始）')
    parser.add_argument('--num_shards', type=int, default=1, help='总分片数')
    return parser.parse_args()


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def normalize_dataset_name(name: str) -> str:
    return name.lower().replace('2009', '09').replace('2012', '12').replace('2015', '15').replace('2017', '17')


def resolve_path(base_dir: str, path_str: str) -> str:
    if os.path.isabs(path_str):
        return os.path.normpath(path_str)
    return os.path.normpath(os.path.join(base_dir, path_str))


def resolve_output_path(output_arg: str) -> str:
    if not output_arg:
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_arg = f'cold_start_results_v2_{ts}.csv'
    out = output_arg if os.path.isabs(output_arg) else os.path.join(CURRENT_DIR, 'output', output_arg)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    return out


def load_json(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def resolve_processed_data_dir(data_dir: Optional[str], dataset_name: str) -> Optional[str]:
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
    if os.path.basename(data_dir).lower() == dataset_name.lower():
        return data_dir
    return nested_dir


def get_metadata(processed_data_dir: str) -> Tuple[int, int]:
    metadata_path = os.path.join(processed_data_dir, 'metadata.json')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f'Metadata file not found: {metadata_path}')
    with open(metadata_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    return int(meta['metrics'].get('n_question', 0)), int(meta['metrics'].get('n_skill', 0))


def select_best_params(model_name: str, dataset_name: str) -> Dict:
    best_params_path = os.path.join(CURRENT_DIR, 'config', 'best_params', f'{model_name}_best_params.json')
    if not os.path.exists(best_params_path):
        return {}
    with open(best_params_path, 'r', encoding='utf-8') as f:
        best_params_dict = json.load(f)
    ds_name = dataset_name.lower()
    base_ds_name = ds_name.split('-')[0]
    matched_key = None
    for key in best_params_dict.keys():
        if key == 'default':
            continue
        key_norm = normalize_dataset_name(key)
        if key_norm == base_ds_name or key_norm in base_ds_name:
            matched_key = key
            break
    return dict(best_params_dict.get(matched_key, best_params_dict.get('default', {})))


def safe_torch_load(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def parse_model_dataset_from_checkpoint_name(filename: str) -> Tuple[Optional[str], Optional[str]]:
    name = os.path.basename(filename)
    if name.endswith('.pt'):
        name = name[:-3]
    for suffix in ['_best', '_last_fold', '_global_best']:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    matched_model = None
    for model_name in sorted(KNOWN_MODEL_PREFIXES, key=len, reverse=True):
        if name.startswith(model_name + '_'):
            matched_model = model_name
            break
    if matched_model is None:
        return None, None
    dataset_name = name[len(matched_model) + 1:]
    return (matched_model, dataset_name) if dataset_name else (None, None)


def parse_baseline_filename_from_config(filename: str) -> Tuple[Optional[str], Optional[str]]:
    name = os.path.basename(filename)
    if name.endswith('.pt'):
        name = name[:-3]

    matched_model = None
    for model_name in sorted(KNOWN_MODEL_PREFIXES, key=len, reverse=True):
        if name.startswith(model_name + '_'):
            matched_model = model_name
            break
    if matched_model is None:
        return None, None

    suffix = name[len(matched_model) + 1:]
    if '_best_' in suffix:
        dataset = suffix.split('_best_', 1)[0]
    elif suffix.endswith('_best'):
        dataset = suffix[:-5]
    else:
        dataset = suffix
    return matched_model, dataset


def should_keep_by_family(model_name: str, family: str) -> bool:
    m = model_name.lower()
    if family == 'all':
        return True
    if family == 'baseline':
        return m in BASELINE_MODELS
    if family == 'gikt':
        return m in GIKT_MODELS
    return True


def should_keep_by_shard(item_index: int, shard_index: int, num_shards: int) -> bool:
    return True if num_shards <= 1 else (item_index % num_shards == shard_index)


def calculate_auc(targets: List[float], preds: List[float]) -> float:
    if len(preds) < 2 or len(set(targets)) < 2:
        return float('nan')
    return float(metrics.roc_auc_score(np.array(targets), np.array(preds)))


def build_test_loader(config, batch_size: int, num_workers: int) -> DataLoader:
    test_dataset = UnifiedParquetDataset(config, augment=False, mode='test')
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())


def instantiate_model(model_name: str, dataset_name: str, checkpoint_path: str, device: torch.device, data_dir: Optional[str] = None, ablation_group: str = ''):
    config = get_config(dataset_name)
    if data_dir:
        resolved_data_dir = resolve_processed_data_dir(data_dir, dataset_name)
        config.path.PROCESSED_DATA_ROOT = os.path.dirname(resolved_data_dir)
        config.DATASET_NAME = os.path.basename(resolved_data_dir)
    num_question, num_skill = get_metadata(config.PROCESSED_DATA_DIR)
    kwargs = select_best_params(model_name.lower(), dataset_name)
    if model_name.lower() in {'gikt', 'gikt_old'}:
        exp_config_path = os.path.join(CURRENT_DIR, 'config', 'experiments', 'exp_full_default.toml')
        if os.path.exists(exp_config_path):
            gikt_config = toml.load(exp_config_path)
            if 'model' in gikt_config and isinstance(gikt_config['model'], dict):
                merged = dict(gikt_config['model'])
                merged.update(kwargs)
                kwargs = merged
        if ablation_group:
            kwargs.update(ABLATION_OVERRIDES.get(ablation_group, {}))
        if data_dir:
            kwargs['data_dir'] = config.PROCESSED_DATA_DIR
    loaded_obj = safe_torch_load(checkpoint_path, device)
    load_info = {'checkpoint_kind': type(loaded_obj).__name__, 'missing_keys_count': 0, 'unexpected_keys_count': 0}
    if isinstance(loaded_obj, torch.nn.Module):
        model = loaded_obj.to(device)
    else:
        model = ModelFactory.get_model(model_name=model_name, num_question=num_question, num_skill=num_skill, device=device, config=config, **kwargs)
        clean_state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in loaded_obj.items()}
        incompatible = model.load_state_dict(clean_state_dict, strict=False)
        load_info['missing_keys_count'] = len(getattr(incompatible, 'missing_keys', []))
        load_info['unexpected_keys_count'] = len(getattr(incompatible, 'unexpected_keys', []))
    model.model_name = model_name.lower()
    return model, config, load_info


def evaluate_cold_start(model, dataloader: DataLoader, device: torch.device, use_amp: bool = False) -> Dict:
    model.eval()
    amp_enabled = bool(use_amp and device.type == 'cuda')
    bin_preds = {b_label: [] for _, _, b_label, _ in BINS}
    bin_targets = {b_label: [] for _, _, b_label, _ in BINS}
    bin_user_counts = {b_label: 0 for _, _, b_label, _ in BINS}
    total_batches = 0
    effective_batches = 0
    invalid_prob_count = 0

    with torch.inference_mode():
        for batch in dataloader:
            total_batches += 1
            features = {k: v.to(device) for k, v in batch.items()}
            question = features[SeqFeatureKey.Q].to(torch.long)
            skill = features.get(SeqFeatureKey.C, torch.zeros_like(question)).to(torch.long)
            response = features[SeqFeatureKey.R].to(torch.long)
            mask = features[SeqFeatureKey.MASK].to(torch.bool)
            eval_mask = features.get(SeqFeatureKey.EVAL_MASK, mask).to(torch.bool)
            interval = torch.nan_to_num(features[SeqFeatureKey.T_INTERVAL].to(torch.float32), nan=0.0)
            r_time = torch.nan_to_num(features[SeqFeatureKey.T_RESPONSE].to(torch.float32), nan=0.0)
            model_name = getattr(model, 'model_name', '').lower()
            cognitive_mode = getattr(model, 'cognitive_mode', None)

            with torch.amp.autocast(device_type='cuda', enabled=amp_enabled):
                if model_name == 'dkt':
                    y_hat = model(question, response, mask, skill=skill)
                    preds = torch.sigmoid(y_hat[:, :-1])
                elif model_name in ['dkvmn', 'akt', 'simplekt', 'qikt', 'deep_irt', 'lbkt', 'gkt']:
                    try:
                        y_hat = model(question, response, mask, interval, r_time, skill=skill)
                    except TypeError:
                        y_hat = model(question, response, mask, interval, r_time)
                    y_hat = torch.sigmoid(y_hat)
                    preds = y_hat if y_hat.shape[1] != question.shape[1] else y_hat[:, :-1]
                elif model_name in ['gikt', 'gikt_old'] or cognitive_mode == 'classic':
                    y_hat = model(question, response, mask, interval, r_time)
                    y_hat = torch.sigmoid(y_hat)
                    preds = y_hat[:, 1:]
                else:
                    try:
                        y_hat = model(question, response, mask, interval, r_time)
                    except TypeError:
                        y_hat = model(question, response, mask)
                    y_hat = torch.sigmoid(y_hat)
                    preds = y_hat[:, 1:] if y_hat.shape[1] == question.shape[1] else y_hat

            targets = response[:, 1:].float()
            mask_valid = mask[:, 1:]
            eval_mask_valid = eval_mask[:, 1:]
            batch_effective = False

            for i in range(question.shape[0]):
                actual_len = int(mask[i].sum().item())
                matched_label = None
                for b_min, b_max, b_label, _ in BINS:
                    if b_min <= actual_len <= b_max:
                        matched_label = b_label
                        break
                if matched_label is None:
                    continue
                student_final_mask = mask_valid[i] & eval_mask_valid[i]
                if student_final_mask.sum() <= 0:
                    continue
                p = preds[i][student_final_mask]
                t = targets[i][student_final_mask]
                finite_mask = torch.isfinite(p) & torch.isfinite(t)
                if finite_mask.sum() <= 0:
                    invalid_prob_count += int(p.numel())
                    continue
                p_np = p[finite_mask].detach().cpu().numpy()
                t_np = t[finite_mask].detach().cpu().numpy()
                invalid_prob_count += int((~finite_mask).sum().item())
                bin_preds[matched_label].extend(p_np.tolist())
                bin_targets[matched_label].extend(t_np.tolist())
                bin_user_counts[matched_label] += 1
                batch_effective = True

            if batch_effective:
                effective_batches += 1

    results = {}
    total_samples = 0
    total_users = 0
    weighted_auc_num = 0.0
    for _, _, b_label, b_key in BINS:
        auc_val = calculate_auc(bin_targets[b_label], bin_preds[b_label])
        samples = len(bin_targets[b_label])
        users = int(bin_user_counts[b_label])
        total_samples += samples
        total_users += users
        if samples > 0 and not np.isnan(auc_val):
            weighted_auc_num += auc_val * samples
        results[b_key] = {'label': b_label, 'auc': auc_val, 'samples': samples, 'users': users}
    weighted_auc = weighted_auc_num / total_samples if total_samples > 0 else float('nan')

    return {
        'bins': results,
        'total_samples': int(total_samples),
        'total_users': int(total_users),
        'weighted_auc': float(weighted_auc) if not np.isnan(weighted_auc) else np.nan,
        'total_batches': int(total_batches),
        'effective_batches': int(effective_batches),
        'invalid_prob_count': int(invalid_prob_count),
    }


def get_device_name(device: torch.device) -> str:
    if device.type == 'cuda' and torch.cuda.is_available():
        try:
            return torch.cuda.get_device_name(device)
        except Exception:
            return 'cuda'
    return 'cpu'


def summarize_row(row: Dict):
    print(
        f"[Done] {row['model']}@{row['dataset']} | "
        f"samples={row['total_samples']} users={row['total_users']} | "
        f"load={row['load_time_sec']:.3f}s eval={row['eval_time_sec']:.3f}s total={row['total_time_sec']:.3f}s | "
        f"throughput={row['samples_per_sec']:.2f} samples/s | "
        f"per_sample_eval={row['eval_time_per_sample_ms']:.6f} ms"
    )


def to_float_or_nan(v):
    try:
        return float(v)
    except Exception:
        return np.nan


def build_tasks_from_config(args: argparse.Namespace) -> List[Dict[str, str]]:
    config_path = resolve_path(CURRENT_DIR, args.config)
    config_data = load_json(config_path)
    base_dir = os.path.dirname(config_path)

    baseline_root = resolve_path(base_dir, config_data.get('baseline_root', 'checkpoint'))
    fa_gikt_root = resolve_path(base_dir, config_data.get('fa_gikt_root', 'checkpoint'))

    tasks = []

    for ckpt_name in config_data.get('baseline_files', []):
        model_name, dataset_name = parse_baseline_filename_from_config(ckpt_name)
        if not model_name or not dataset_name:
            continue
        tasks.append({
            'model': model_name.lower(),
            'dataset': dataset_name.lower(),
            'checkpoint_path': os.path.join(baseline_root, ckpt_name),
            'checkpoint_source': 'json_baseline_files',
            'ablation_group': '',
        })

    for run in config_data.get('fa_gikt_runs', []):
        dataset_name = str(run.get('dataset', '')).lower()
        model_name = str(run.get('model_name', ''))
        ablation_group = str(run.get('ablation_group', ''))
        if not dataset_name or not model_name:
            continue
        if not model_name.endswith('.pt'):
            model_name = f'{model_name}.pt'
        tasks.append({
            'model': 'gikt',
            'dataset': dataset_name,
            'checkpoint_path': os.path.join(fa_gikt_root, model_name),
            'checkpoint_source': 'json_fa_gikt_runs',
            'ablation_group': ablation_group,
        })

    uniq = []
    seen = set()
    for t in tasks:
        key = (t['model'], t['dataset'], t['checkpoint_path'])
        if key not in seen:
            seen.add(key)
            uniq.append(t)
    return uniq


def test_cold_start_and_report(tasks: List[Dict[str, str]], args: argparse.Namespace):
    if args.num_shards < 1:
        raise ValueError('--num_shards 必须 >= 1')
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError('--shard_index 必须满足 0 <= shard_index < num_shards')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_name = get_device_name(device)
    print(f'🔥 冷启动评估启动，设备: {device} ({device_name})')
    output_path = resolve_output_path(args.output)

    existing_records = set()
    if os.path.exists(output_path) and not args.force:
        try:
            df_old = pd.read_csv(output_path)
            for _, row in df_old.iterrows():
                existing_records.add((
                    str(row.get('dataset', '')).lower(),
                    str(row.get('model', '')).lower(),
                    str(row.get('checkpoint_path', '')).lower(),
                ))
        except Exception:
            pass

    filtered_tasks = []
    for idx, t in enumerate(tasks):
        m, d = t['model'].lower(), t['dataset'].lower()
        if args.dataset and d != args.dataset.lower():
            continue
        if args.model_name and m != args.model_name.lower():
            continue
        if args.ablation_group and str(t.get('ablation_group', '')) != args.ablation_group:
            continue
        if not should_keep_by_family(m, args.family):
            continue
        if not should_keep_by_shard(idx, args.shard_index, args.num_shards):
            continue
        filtered_tasks.append(t)

    if not filtered_tasks:
        print('[WARN] 过滤后无可执行任务。')
        return

    tasks_by_dataset = defaultdict(list)
    for t in filtered_tasks:
        tasks_by_dataset[t['dataset']].append(t)

    new_rows = []
    script_start_time = time.perf_counter()

    for dataset_name, task_list in tasks_by_dataset.items():
        print('\n' + '=' * 72)
        print(f'📦 数据集: {dataset_name}')
        print('=' * 72)

        try:
            config = get_config(dataset_name)
            if args.data_dir:
                resolved_data_dir = resolve_processed_data_dir(args.data_dir, dataset_name)
                config.path.PROCESSED_DATA_ROOT = os.path.dirname(resolved_data_dir)
                config.DATASET_NAME = os.path.basename(resolved_data_dir)
                print(f'📁 覆盖数据目录: {config.PROCESSED_DATA_DIR}')
            metadata_path = os.path.join(config.PROCESSED_DATA_DIR, 'metadata.json')
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            test_loader = build_test_loader(config, args.batch_size, args.num_workers)
            metadata_n_question = int(metadata.get('metrics', {}).get('n_question', 0))
            metadata_n_skill = int(metadata.get('metrics', {}).get('n_skill', 0))
        except Exception as e:
            print(f'❌ 初始化数据集失败: {e}')
            continue

        for task in task_list:
            model_name = task['model']
            ckpt_path = task['checkpoint_path']
            ckpt_source = task.get('checkpoint_source', 'json')
            ablation_group = task.get('ablation_group', '')
            if not ckpt_path or not os.path.exists(ckpt_path):
                print(f'  ✗ checkpoint 未找到: {model_name}@{dataset_name} (source={ckpt_source})')
                continue
            dedup_key = (dataset_name.lower(), model_name.lower(), ckpt_path.lower())
            if (not args.force) and (dedup_key in existing_records):
                print(f'  ⏭️ 已存在记录，跳过: {model_name}@{dataset_name} | {ckpt_path}')
                continue

            print(f'\n  >> Evaluating {model_name.upper()} on {dataset_name} -> {ckpt_path}')
            try:
                load_start = time.perf_counter()
                model, _, load_info = instantiate_model(
                    model_name=model_name,
                    dataset_name=dataset_name,
                    checkpoint_path=ckpt_path,
                    device=device,
                    data_dir=args.data_dir,
                    ablation_group=ablation_group,
                )
                load_time_sec = time.perf_counter() - load_start

                eval_start = time.perf_counter()
                eval_res = evaluate_cold_start(model, test_loader, device, use_amp=args.use_amp)
                eval_time_sec = time.perf_counter() - eval_start
                total_time_sec = load_time_sec + eval_time_sec
                total_samples = int(eval_res['total_samples'])
                samples_per_sec = (total_samples / eval_time_sec) if eval_time_sec > 0 and total_samples > 0 else np.nan
                eval_time_per_sample_ms = (1000.0 * eval_time_sec / total_samples) if total_samples > 0 else np.nan
                checkpoint_mtime = datetime.datetime.fromtimestamp(os.path.getmtime(ckpt_path)).strftime('%Y-%m-%d %H:%M:%S')

                row = {
                    'dataset': dataset_name,
                    'model': model_name.lower(),
                    'display_name': (ablation_group if ablation_group else DISPLAY_NAME_MAP.get(model_name.lower(), model_name.upper())),
                    'family': 'gikt' if model_name.lower() in GIKT_MODELS else 'baseline',
                    'ablation_group': ablation_group,
                    'checkpoint_path': ckpt_path,
                    'checkpoint_source': ckpt_source,
                    'checkpoint_mtime': checkpoint_mtime,
                    'update_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'seed': int(args.seed),
                    'device': device.type,
                    'device_name': device_name,
                    'amp_enabled': bool(args.use_amp and device.type == 'cuda'),
                    'batch_size': int(args.batch_size),
                    'num_workers': int(args.num_workers),
                    'total_samples': total_samples,
                    'total_users': int(eval_res['total_users']),
                    'weighted_auc': to_float_or_nan(eval_res['weighted_auc']),
                    'total_batches': int(eval_res['total_batches']),
                    'effective_batches': int(eval_res['effective_batches']),
                    'invalid_prob_count': int(eval_res['invalid_prob_count']),
                    'load_time_sec': float(load_time_sec),
                    'eval_time_sec': float(eval_time_sec),
                    'total_time_sec': float(total_time_sec),
                    'samples_per_sec': float(samples_per_sec) if not np.isnan(samples_per_sec) else np.nan,
                    'eval_time_per_sample_ms': float(eval_time_per_sample_ms) if not np.isnan(eval_time_per_sample_ms) else np.nan,
                    'metadata_n_question': metadata_n_question,
                    'metadata_n_skill': metadata_n_skill,
                    'load_missing_keys_count': int(load_info.get('missing_keys_count', 0)),
                    'load_unexpected_keys_count': int(load_info.get('unexpected_keys_count', 0)),
                    'load_checkpoint_kind': str(load_info.get('checkpoint_kind', 'unknown')),
                }
                for _, _, b_label, b_key in BINS:
                    b = eval_res['bins'][b_key]
                    row[f'{b_key}_label'] = b_label
                    row[f'{b_key}_auc'] = to_float_or_nan(b['auc'])
                    row[f'{b_key}_samples'] = int(b['samples'])
                    row[f'{b_key}_users'] = int(b['users'])

                summarize_row(row)
                new_rows.append(row)
            except Exception as e:
                print(f'  ✗ 评估失败: {model_name}@{dataset_name} | {e}')

    if not new_rows:
        print('[WARN] 没有新增结果。')
        return

    df_new = pd.DataFrame(new_rows)
    # 重要：即使 --force，也不应清空历史结果文件。
    # --force 仅用于“重新评估并覆盖同 key 记录”，而不是“覆盖整个输出文件”。
    if os.path.exists(output_path):
        try:
            df_old = pd.read_csv(output_path)
            df_merged = pd.concat([df_old, df_new], ignore_index=True)
        except Exception:
            df_merged = df_new.copy()
    else:
        df_merged = df_new.copy()

    if {'dataset', 'model', 'checkpoint_path'}.issubset(set(df_merged.columns)):
        df_merged.drop_duplicates(subset=['dataset', 'model', 'checkpoint_path'], keep='last', inplace=True)

    df_merged = df_merged.sort_values(['dataset', 'family', 'model', 'update_time']).reset_index(drop=True)
    df_merged.to_csv(output_path, index=False, encoding='utf-8-sig')

    total_script_time = time.perf_counter() - script_start_time
    print('\n' + '=' * 80)
    print('✅ 冷启动分桶评估完成')
    print(f'结果文件: {output_path}')
    print(f'新增记录: {len(df_new)} 条 | 累计记录: {len(df_merged)} 条')
    print(f'脚本总耗时: {total_script_time:.3f}s')
    print('=' * 80)


def main():
    args = parse_args()
    set_global_seed(args.seed)
    tasks = build_tasks_from_config(args)
    if not tasks:
        print('💡 配置中未解析到可评估任务，请检查 --config 文件')
        sys.exit(1)
    test_cold_start_and_report(tasks, args)


if __name__ == '__main__':
    main()
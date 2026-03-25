#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
统一 ACC 重评脚本（Fixed Threshold = 0.5）

设计目标：
1. 使用固定阈值 0.5 对所有模型重新计算测试集 ACC，避免验证集调阈值带来的口径不一致。
2. 评估协议与学术论文常见做法保持一致：
   - 仅在独立测试集上评估；
   - 不在测试阶段进行阈值搜索；
   - 保留 AUC 作为阈值无关指标，同时输出 ACC 与 BCE loss；
   - 严格遵循 eval_mask，避免重叠窗口标签重复暴露。
3. 同时支持：
   - baseline 模型（通常保存为 state_dict）
   - FA-GIKT / 消融模型（通常保存为完整 torch.nn.Module）

默认配置文件：
	back/kt/config/evaluation/unified_acc_reeval_models.json

示例：
	python test_evaluate.py
	python test_evaluate.py --config back/kt/config/evaluation/unified_acc_reeval_models.json
	python test_evaluate.py --family baseline
	python test_evaluate.py --dataset assist12
"""

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn import metrics
from torch.utils.data import DataLoader


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
	sys.path.append(CURRENT_DIR)

from config import get_config  # noqa: E402
from dataset import UnifiedParquetDataset, SeqFeatureKey  # noqa: E402
from models.factory import ModelFactory  # noqa: E402


KNOWN_MODEL_PREFIXES = [
	'deep_irt',
	'simplekt',
	'gikt_old',
	'dkvmn',
	'gikt',
	'dkt',
	'akt',
]

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

DISPLAY_NAME_MAP = {
	'dkt': 'DKT',
	'dkvmn': 'DKVMN',
	'akt': 'AKT',
	'simplekt': 'SimpleKT',
	'deep_irt': 'Deep-IRT',
	'gikt_old': 'GIKT',
	'gikt': 'FA-GIKT',
}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description='统一 ACC 重评脚本（固定阈值 0.5）')
	parser.add_argument(
		'--config',
		type=str,
		default=os.path.join(CURRENT_DIR, 'config', 'evaluation', 'unified_acc_reeval_models.json'),
		help='评估配置文件路径',
	)
	parser.add_argument('--family', type=str, choices=['all', 'baseline', 'fa_gikt'], default='all', help='仅评估某一类模型')
	parser.add_argument('--dataset', type=str, default='', help='仅评估指定数据集')
	parser.add_argument('--batch_size', type=int, default=256, help='测试批大小')
	parser.add_argument('--num_workers', type=int, default=0, help='DataLoader num_workers')
	parser.add_argument('--use_amp', action='store_true', help='测试时启用 AMP（默认关闭，保证数值口径更稳定）')
	parser.add_argument('--output_csv', type=str, default='', help='若提供，则覆盖配置文件中的输出路径')
	parser.add_argument('--model_name', type=str, default='', help='仅评估指定 baseline 模型名，例如 dkt、akt、simplekt')
	parser.add_argument('--ablation_group', type=str, default='', help='仅评估指定消融组，例如 A_Baseline')
	parser.add_argument('--shard_index', type=int, default=0, help='并行分片编号，从 0 开始')
	parser.add_argument('--num_shards', type=int, default=1, help='总分片数，用于多卡并行切分任务')
	parser.add_argument('--seed', type=int, default=42, help='评估随机种子，用于保证复现实验环境')
	return parser.parse_args()


def load_json(path: str) -> dict:
	with open(path, 'r', encoding='utf-8') as f:
		return json.load(f)


def resolve_path(base_dir: str, path_str: str) -> str:
	if os.path.isabs(path_str):
		return path_str
	return os.path.normpath(os.path.join(base_dir, path_str))


def safe_torch_load(path: str, device: torch.device):
	try:
		return torch.load(path, map_location=device, weights_only=False)
	except TypeError:
		return torch.load(path, map_location=device)


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


def get_device_name(device: torch.device) -> str:
	if device.type == 'cuda' and torch.cuda.is_available():
		try:
			return torch.cuda.get_device_name(device)
		except Exception:
			return 'cuda'
	return 'cpu'


def get_metadata(processed_data_dir: str) -> Tuple[int, int]:
	metadata_path = os.path.join(processed_data_dir, 'metadata.json')
	if not os.path.exists(metadata_path):
		raise FileNotFoundError(f'Metadata file not found: {metadata_path}')

	with open(metadata_path, 'r', encoding='utf-8') as f:
		meta = json.load(f)

	num_question = int(meta['metrics'].get('n_question', 0))
	num_skill = int(meta['metrics'].get('n_skill', 0))
	return num_question, num_skill


def normalize_dataset_name(name: str) -> str:
	return name.lower().replace('2009', '09').replace('2012', '12').replace('2015', '15').replace('2017', '17')


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

	selected = best_params_dict.get(matched_key, best_params_dict.get('default', {}))
	return dict(selected)


def parse_baseline_filename(filename: str) -> Tuple[str, str]:
	name = os.path.basename(filename)
	if name.endswith('.pt'):
		name = name[:-3]

	marker = '_best_'
	if marker not in name:
		raise ValueError(f'Invalid baseline checkpoint filename: {filename}')

	prefix = name.split(marker)[0]
	matched_model = None
	for model_name in sorted(KNOWN_MODEL_PREFIXES, key=len, reverse=True):
		if prefix.startswith(model_name + '_'):
			matched_model = model_name
			break

	if matched_model is None:
		raise ValueError(f'Unable to infer model name from filename: {filename}')

	dataset_name = prefix[len(matched_model) + 1:]
	return matched_model, dataset_name


def build_test_loader(config, batch_size: int, num_workers: int) -> DataLoader:
	dataset_test = UnifiedParquetDataset(config, augment=False, mode='test')
	return DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())


def instantiate_baseline_model(model_name: str, dataset_name: str, checkpoint_path: str, device: torch.device):
	config = get_config(dataset_name)
	num_question, num_skill = get_metadata(config.PROCESSED_DATA_DIR)
	model_kwargs = select_best_params(model_name, dataset_name)

	model = ModelFactory.get_model(
		model_name=model_name,
		num_question=num_question,
		num_skill=num_skill,
		device=device,
		config=config,
		**model_kwargs,
	)

	loaded_obj = safe_torch_load(checkpoint_path, device)
	if isinstance(loaded_obj, torch.nn.Module):
		model = loaded_obj.to(device)
	else:
		model.load_state_dict(loaded_obj)

	model.model_name = model_name.lower()
	return model, config


def instantiate_fa_gikt_model(dataset_name: str, checkpoint_path: str, device: torch.device, ablation_group: str):
	config = get_config(dataset_name)
	loaded_obj = safe_torch_load(checkpoint_path, device)

	if isinstance(loaded_obj, torch.nn.Module):
		model = loaded_obj.to(device)
	else:
		num_question, num_skill = get_metadata(config.PROCESSED_DATA_DIR)
		model_kwargs = select_best_params('gikt', dataset_name)
		model_kwargs.update(ABLATION_OVERRIDES.get(ablation_group, {}))
		model = ModelFactory.get_model(
			model_name='gikt',
			num_question=num_question,
			num_skill=num_skill,
			device=device,
			config=config,
			**model_kwargs,
		)
		model.load_state_dict(loaded_obj)

	model.model_name = 'gikt'
	return model, config


def evaluate_model(model, dataloader: DataLoader, device: torch.device, threshold: float = 0.5, use_amp: bool = False) -> Dict[str, float]:
	criterion = nn.BCEWithLogitsLoss()
	model.eval()

	all_preds: List[float] = []
	all_targets: List[float] = []
	total_loss = 0.0
	total_samples = 0
	total_batches = 0
	effective_batches = 0

	amp_enabled = use_amp and device.type == 'cuda'

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
					preds = y_hat[:, :-1]
				elif model_name in ['dkvmn', 'akt', 'simplekt', 'qikt', 'deep_irt', 'gkt']:
					try:
						y_hat = model(question, response, mask, interval, r_time, skill=skill)
					except TypeError:
						y_hat = model(question, response, mask, interval, r_time)
					preds = y_hat if y_hat.shape[1] != question.shape[1] else y_hat[:, :-1]
				elif model_name in ['gikt', 'gikt_old'] or cognitive_mode == 'classic':
					y_hat = model(question, response, mask, interval, r_time)
					preds = y_hat[:, 1:]
				else:
					try:
						y_hat = model(question, response, mask, interval, r_time)
					except TypeError:
						y_hat = model(question, response, mask)
					preds = y_hat[:, 1:]

			targets = response[:, 1:].float()
			mask_valid = mask[:, 1:]
			eval_mask_valid = eval_mask[:, 1:]
			final_mask = mask_valid & eval_mask_valid

			if final_mask.sum() == 0:
				continue

			effective_batches += 1

			logits = preds[final_mask]
			selected_targets = targets[final_mask]
			probs = torch.sigmoid(logits)
			loss = criterion(logits, selected_targets)

			total_loss += loss.item() * selected_targets.size(0)
			total_samples += int(selected_targets.size(0))

			finite_mask = torch.isfinite(probs) & torch.isfinite(selected_targets)
			if finite_mask.sum() > 0:
				all_preds.extend(probs[finite_mask].detach().cpu().numpy().tolist())
				all_targets.extend(selected_targets[finite_mask].detach().cpu().numpy().tolist())

	if total_samples == 0 or not all_preds:
		return {
			'auc': np.nan,
			'acc': np.nan,
			'loss': np.nan,
			'samples': 0,
			'positive_rate': np.nan,
			'prediction_mean': np.nan,
			'total_batches': int(total_batches),
			'effective_batches': int(effective_batches),
		}

	preds_np = np.array(all_preds)
	targets_np = np.array(all_targets)
	avg_loss = total_loss / total_samples

	try:
		auc = metrics.roc_auc_score(targets_np, preds_np)
	except ValueError:
		auc = np.nan

	acc = metrics.accuracy_score(targets_np, (preds_np >= threshold).astype(np.int64))
	return {
		'auc': float(auc) if not np.isnan(auc) else np.nan,
		'acc': float(acc),
		'loss': float(avg_loss),
		'samples': int(total_samples),
		'positive_rate': float(np.mean(targets_np)),
		'prediction_mean': float(np.mean(preds_np)),
		'total_batches': int(total_batches),
		'effective_batches': int(effective_batches),
	}


def ensure_output_path(output_csv: str) -> str:
	if os.path.isabs(output_csv):
		os.makedirs(os.path.dirname(output_csv), exist_ok=True)
		return output_csv

	resolved = os.path.join(CURRENT_DIR, output_csv)
	os.makedirs(os.path.dirname(resolved), exist_ok=True)
	return resolved


def should_keep_by_shard(item_index: int, shard_index: int, num_shards: int) -> bool:
	if num_shards <= 1:
		return True
	return item_index % num_shards == shard_index


def summarize_runtime(model_label: str, dataset_name: str, checkpoint_path: str, load_time_sec: float, eval_time_sec: float, metrics_dict: Dict[str, float]):
	total_time_sec = load_time_sec + eval_time_sec
	samples = int(metrics_dict.get('samples', 0))
	throughput = (samples / eval_time_sec) if eval_time_sec > 0 and samples > 0 else np.nan
	per_sample_eval_ms = (1000.0 * eval_time_sec / samples) if samples > 0 else np.nan
	print(
		f'[Done] {model_label} | dataset={dataset_name} | samples={samples} | '
		f'load={load_time_sec:.3f}s | eval={eval_time_sec:.3f}s | total={total_time_sec:.3f}s | '
		f'throughput={throughput:.2f} samples/s | per_sample_eval={per_sample_eval_ms:.6f} ms | ckpt={checkpoint_path}'
	)


def run_baseline_evaluation(config_data: dict, device: torch.device, args: argparse.Namespace, threshold: float) -> List[Dict]:
	baseline_root = config_data['baseline_root']
	baseline_files = list(dict.fromkeys(config_data.get('baseline_files', [])))
	results = []
	device_name = get_device_name(device)

	for item_index, checkpoint_name in enumerate(baseline_files):
		model_name, dataset_name = parse_baseline_filename(checkpoint_name)
		if args.dataset and dataset_name != args.dataset:
			continue
		if args.model_name and model_name != args.model_name:
			continue
		if not should_keep_by_shard(item_index, args.shard_index, args.num_shards):
			continue

		checkpoint_path = os.path.join(baseline_root, checkpoint_name)
		if not os.path.exists(checkpoint_path):
			print(f'[WARN] Baseline checkpoint not found: {checkpoint_path}')
			continue

		print(f'[Baseline] Evaluating {model_name} on {dataset_name} -> {checkpoint_path}')
		load_start_time = time.perf_counter()
		model, dataset_config = instantiate_baseline_model(model_name, dataset_name, checkpoint_path, device)
		test_loader = build_test_loader(dataset_config, args.batch_size, args.num_workers)
		load_time_sec = time.perf_counter() - load_start_time
		eval_start_time = time.perf_counter()
		metrics_dict = evaluate_model(model, test_loader, device, threshold=threshold, use_amp=args.use_amp)
		eval_time_sec = time.perf_counter() - eval_start_time
		total_time_sec = load_time_sec + eval_time_sec
		samples_per_sec = metrics_dict['samples'] / eval_time_sec if eval_time_sec > 0 and metrics_dict['samples'] > 0 else np.nan
		eval_time_per_sample_ms = (1000.0 * eval_time_sec / metrics_dict['samples']) if metrics_dict['samples'] > 0 else np.nan
		total_time_per_sample_ms = (1000.0 * total_time_sec / metrics_dict['samples']) if metrics_dict['samples'] > 0 else np.nan
		eval_time_per_batch_sec = (eval_time_sec / metrics_dict['effective_batches']) if metrics_dict.get('effective_batches', 0) > 0 else np.nan
		avg_samples_per_batch = (metrics_dict['samples'] / metrics_dict['effective_batches']) if metrics_dict.get('effective_batches', 0) > 0 else np.nan
		summarize_runtime(model_name, dataset_name, checkpoint_path, load_time_sec, eval_time_sec, metrics_dict)

		results.append({
			'family': 'baseline',
			'model_name': model_name,
			'display_name': DISPLAY_NAME_MAP.get(model_name, model_name),
			'ablation_group': '',
			'dataset': dataset_name,
			'checkpoint_path': checkpoint_path,
			'threshold_strategy': f'fixed_{threshold}',
			'threshold': threshold,
			'auc': metrics_dict['auc'],
			'acc': metrics_dict['acc'],
			'loss': metrics_dict['loss'],
			'samples': metrics_dict['samples'],
			'positive_rate': metrics_dict['positive_rate'],
			'prediction_mean': metrics_dict['prediction_mean'],
			'total_batches': metrics_dict['total_batches'],
			'effective_batches': metrics_dict['effective_batches'],
			'load_time_sec': load_time_sec,
			'eval_time_sec': eval_time_sec,
			'total_time_sec': total_time_sec,
			'samples_per_sec': samples_per_sec,
			'eval_time_per_sample_ms': eval_time_per_sample_ms,
			'total_time_per_sample_ms': total_time_per_sample_ms,
			'eval_time_per_batch_sec': eval_time_per_batch_sec,
			'avg_samples_per_effective_batch': avg_samples_per_batch,
			'device': device.type,
			'device_name': device_name,
			'amp_enabled': bool(args.use_amp),
			'seed': int(args.seed),
		})

	return results


def run_fa_gikt_evaluation(config_data: dict, device: torch.device, args: argparse.Namespace, threshold: float) -> List[Dict]:
	fa_root = config_data['fa_gikt_root']
	results = []
	device_name = get_device_name(device)

	for item_index, run in enumerate(config_data.get('fa_gikt_runs', [])):
		dataset_name = run['dataset']
		if args.dataset and dataset_name != args.dataset:
			continue

		ablation_group = run['ablation_group']
		if args.ablation_group and ablation_group != args.ablation_group:
			continue
		if not should_keep_by_shard(item_index, args.shard_index, args.num_shards):
			continue

		checkpoint_name = run.get('model_name') or run.get('mode', '')
		if not checkpoint_name:
			print(f'[WARN] Missing model_name in fa_gikt_runs item: {run}')
			continue
		if not checkpoint_name.endswith('.pt'):
			checkpoint_name = f'{checkpoint_name}.pt'
		checkpoint_path = os.path.join(fa_root, checkpoint_name)
		if not os.path.exists(checkpoint_path):
			print(f'[WARN] FA-GIKT checkpoint not found: {checkpoint_path}')
			continue

		print(f'[FA-GIKT] Evaluating {ablation_group} on {dataset_name} -> {checkpoint_path}')
		load_start_time = time.perf_counter()
		model, dataset_config = instantiate_fa_gikt_model(dataset_name, checkpoint_path, device, ablation_group)
		test_loader = build_test_loader(dataset_config, args.batch_size, args.num_workers)
		load_time_sec = time.perf_counter() - load_start_time
		eval_start_time = time.perf_counter()
		metrics_dict = evaluate_model(model, test_loader, device, threshold=threshold, use_amp=args.use_amp)
		eval_time_sec = time.perf_counter() - eval_start_time
		total_time_sec = load_time_sec + eval_time_sec
		samples_per_sec = metrics_dict['samples'] / eval_time_sec if eval_time_sec > 0 and metrics_dict['samples'] > 0 else np.nan
		eval_time_per_sample_ms = (1000.0 * eval_time_sec / metrics_dict['samples']) if metrics_dict['samples'] > 0 else np.nan
		total_time_per_sample_ms = (1000.0 * total_time_sec / metrics_dict['samples']) if metrics_dict['samples'] > 0 else np.nan
		eval_time_per_batch_sec = (eval_time_sec / metrics_dict['effective_batches']) if metrics_dict.get('effective_batches', 0) > 0 else np.nan
		avg_samples_per_batch = (metrics_dict['samples'] / metrics_dict['effective_batches']) if metrics_dict.get('effective_batches', 0) > 0 else np.nan

		display_name = 'FA-GIKT' if ablation_group == 'A_Baseline' else ablation_group
		summarize_runtime(display_name, dataset_name, checkpoint_path, load_time_sec, eval_time_sec, metrics_dict)
		results.append({
			'family': 'fa_gikt',
			'model_name': 'gikt',
			'display_name': display_name,
			'ablation_group': ablation_group,
			'dataset': dataset_name,
			'checkpoint_path': checkpoint_path,
			'threshold_strategy': f'fixed_{threshold}',
			'threshold': threshold,
			'auc': metrics_dict['auc'],
			'acc': metrics_dict['acc'],
			'loss': metrics_dict['loss'],
			'samples': metrics_dict['samples'],
			'positive_rate': metrics_dict['positive_rate'],
			'prediction_mean': metrics_dict['prediction_mean'],
			'total_batches': metrics_dict['total_batches'],
			'effective_batches': metrics_dict['effective_batches'],
			'load_time_sec': load_time_sec,
			'eval_time_sec': eval_time_sec,
			'total_time_sec': total_time_sec,
			'samples_per_sec': samples_per_sec,
			'eval_time_per_sample_ms': eval_time_per_sample_ms,
			'total_time_per_sample_ms': total_time_per_sample_ms,
			'eval_time_per_batch_sec': eval_time_per_batch_sec,
			'avg_samples_per_effective_batch': avg_samples_per_batch,
			'device': device.type,
			'device_name': device_name,
			'amp_enabled': bool(args.use_amp),
			'seed': int(args.seed),
		})

	return results


def build_paper_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
	paper_df = df[(df['family'] == 'baseline') | (df['ablation_group'] == 'A_Baseline')].copy()
	if paper_df.empty:
		return paper_df

	paper_df['paper_model'] = paper_df['display_name']
	# 论文主表默认增加时间口径，便于直接比较推理效率
	# - eval_time_per_sample_ms: 每个有效样本平均评估耗时（ms）
	# - eval_time_sec: 模型评估阶段总耗时（不含加载）
	# - total_time_sec: 装载+评估总耗时
	paper_df = paper_df[[
		'paper_model', 'dataset', 'auc', 'acc', 'loss',
		'eval_time_per_sample_ms', 'eval_time_sec', 'total_time_sec',
		'samples_per_sec', 'samples',
	]]
	paper_df = paper_df.sort_values(['paper_model', 'dataset'])
	return paper_df


def main():
	args = parse_args()
	if args.num_shards < 1:
		raise ValueError('--num_shards 必须 >= 1')
	if args.shard_index < 0 or args.shard_index >= args.num_shards:
		raise ValueError('--shard_index 必须满足 0 <= shard_index < num_shards')

	set_global_seed(args.seed)

	script_start_time = time.perf_counter()
	config_path = os.path.abspath(args.config)
	config_data = load_json(config_path)
	config_base_dir = os.path.dirname(config_path)

	threshold = float(config_data.get('threshold', 0.5))
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	config_data['baseline_root'] = resolve_path(config_base_dir, config_data['baseline_root'])
	config_data['fa_gikt_root'] = resolve_path(config_base_dir, config_data['fa_gikt_root'])
	output_csv = args.output_csv if args.output_csv else config_data.get('output_csv', 'output/evaluation/unified_acc_fixed_threshold_0_5.csv')
	output_csv = ensure_output_path(output_csv)

	all_results: List[Dict] = []
	if args.family in ['all', 'baseline']:
		all_results.extend(run_baseline_evaluation(config_data, device, args, threshold))
	if args.family in ['all', 'fa_gikt']:
		all_results.extend(run_fa_gikt_evaluation(config_data, device, args, threshold))

	if not all_results:
		print('[WARN] No evaluation results were produced. Please check config paths and filters.')
		return

	df = pd.DataFrame(all_results)
	df = df.sort_values(['dataset', 'family', 'display_name']).reset_index(drop=True)
	df.to_csv(output_csv, index=False, encoding='utf-8-sig')

	paper_df = build_paper_comparison_table(df)
	paper_csv = output_csv.replace('.csv', '_paper_table.csv')
	if not paper_df.empty:
		paper_df.to_csv(paper_csv, index=False, encoding='utf-8-sig')

	total_script_time_sec = time.perf_counter() - script_start_time
	if 'total_time_sec' in df.columns:
		print(f"[Summary] 模型累计装载+推理时间: {df['total_time_sec'].sum():.3f}s")
	print(f'[Summary] 脚本总耗时: {total_script_time_sec:.3f}s')

	print('\n' + '=' * 80)
	print('统一 ACC 重评完成（固定阈值 = 0.5）')
	print(f'主结果文件: {output_csv}')
	if not paper_df.empty:
		print(f'论文主表候选文件: {paper_csv}')
	print('=' * 80)
	print(df.to_string(index=False))


if __name__ == '__main__':
	main()

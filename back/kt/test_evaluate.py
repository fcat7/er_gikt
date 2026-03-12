#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型测试评价脚本 (评估保存在 checkpoint 下的最佳模型)
"""

import os
import sys
import time
import json
import torch
import numpy as np
import pandas as pd
from sklearn import metrics
from torch.utils.data import DataLoader
import argparse

# 确保能导入内部模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from dataset import UnifiedParquetDataset, SeqFeatureKey
from models.factory import ModelFactory


def calculate_metrics(targets, preds):
    """计算评估指标"""
    if len(preds) == 0:
        return 0.0, 0.0, 0.0
        
    targets = np.array(targets)
    preds = np.array(preds)
    
    # 防止 NaN 导致的报错
    if np.isnan(preds).any() or np.isinf(preds).any():
        print("警告: 预测结果包含 NaN 或 Inf，指标将被设为 0。")
        return 0.0, 0.0, 0.0
        
    auc = metrics.roc_auc_score(targets, preds)
    acc = metrics.accuracy_score(targets, [1 if p >= 0.5 else 0 for p in preds])
    rmse = np.sqrt(metrics.mean_squared_error(targets, preds))
    
    return auc, acc, rmse


def evaluate_model(model, dataloader, device, use_amp=True):
    """验证循环"""
    model.eval()
    all_preds = []
    all_targets = []
    
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
            
            # TODO: 有些包装使得模型名变成外层类，简单用类名猜测
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
                    if y_hat.shape[1] == question.shape[1]:
                        preds = y_hat[:, :-1]
                    else:
                        preds = y_hat
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
                    
                    if y_hat.shape[1] == question.shape[1]:
                        preds = y_hat[:, :-1]
                    else:
                        preds = y_hat

            targets = response[:, 1:].float()
            mask_valid = mask[:, 1:]
            eval_mask_valid = eval_mask[:, 1:]
            
            final_mask = mask_valid & eval_mask_valid
            
            if final_mask.sum() > 0:
                p = preds[final_mask].cpu().numpy()
                t = targets[final_mask].cpu().numpy()
                
                if not (np.isnan(p).any() or np.isinf(p).any()):
                    all_preds.extend(p)
                    all_targets.extend(t)

    return calculate_metrics(all_targets, all_preds)


import toml
import datetime
from collections import defaultdict

def get_available_checkpoints():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(current_dir, 'checkpoint')
    available_combos = []
    known_models = ['dkt', 'dkvmn', 'akt', 'simplekt', 'qikt', 'lbkt', 'gikt_old', 'gikt', 'dtransformer']
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

def test_and_report(tasks, batch_size=64, result_csv="test_results.csv", force=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用的计算设备: {device}")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results = []
    existing_records = set()
    output_path_fixed = os.path.join(current_dir, 'output', result_csv)
    if os.path.exists(output_path_fixed):
        try:
            import pandas as pd
            df_old = pd.read_csv(output_path_fixed)
            for _, row in df_old.iterrows():
                existing_records.add((str(row.get('Dataset', '')).lower(), str(row.get('Model', '')).lower()))
        except Exception:
            pass

    tasks_by_dataset = defaultdict(list)
    for t in tasks:
        if t['model'] not in tasks_by_dataset[t['dataset']]:
            tasks_by_dataset[t['dataset']].append(t['model'])

    for dataset_name, models_to_test in tasks_by_dataset.items():
        print(f"\n{'='*50}")
        print(f"评估数据集: {dataset_name}")
        print(f"{'='*50}")
        
        try:
            config = get_config(dataset_name)
        except Exception as e:
            print(f"加载数据集配置失败: {e}")
            continue

        metadata_path = os.path.join(config.PROCESSED_DATA_DIR, 'metadata.json')
        if not os.path.exists(metadata_path):
            print("未找到 metadata.json")
            continue
            
        import json
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            
        num_q = metadata['metrics']['n_question']
        num_c = metadata['metrics']['n_skill']
        test_loader = None

        for model_name in models_to_test:
            print(f"\n  >> 测试模型: {model_name.upper()}")
            if not force and (dataset_name.lower(), model_name.lower()) in existing_records:
                print(f"    🌟 结果已存在于 {result_csv} 中，自动跳过此评估 (如需覆盖请添加 --force 参数)")
                continue
            
            ckpt_path = f"checkpoint/{model_name.lower()}_{dataset_name.lower()}_best.pt"
            if not os.path.exists(ckpt_path):
                print(f"    ✗ 找不到预训练模型权重: {ckpt_path}")
                continue
                
            if test_loader is None:
                try:
                    test_dataset = UnifiedParquetDataset(config, augment=False, mode='test')
                    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                    print("✓ 测试集加载成功")
                except Exception as e:
                    print(f"加载测试集失败: {e}")
                    break
                
            try:
                kwargs = {}
                current_dir = os.path.dirname(os.path.abspath(__file__))
                if model_name.lower() in ['gikt', 'gikt_old']:
                    exp_config_path = os.path.join(current_dir, 'config', 'experiments', 'exp_full_default.toml')
                    if os.path.exists(exp_config_path):
                        gikt_config = toml.load(exp_config_path)
                        if 'model' in gikt_config:
                            for k, v in gikt_config['model'].items():
                                kwargs[k] = v
                
                best_params_path = os.path.join(current_dir, 'config', 'best_params', f"{model_name.lower()}_best_params.json")
                if os.path.exists(best_params_path):
                    with open(best_params_path, 'r', encoding='utf-8') as f:
                        best_params_dict = json.load(f)
                        
                    ds_name = dataset_name.lower()
                    matched_key = None
                    for k in best_params_dict.keys():
                        if k == ds_name or k.replace("2009", "09").replace("2012", "12").replace("2015", "15").replace("2017", "17") in ds_name or k in ds_name:
                            matched_key = k
                            break
                            
                    if matched_key:
                        print(f"    ✓ 从 JSON 自动加载最优参数: {matched_key}")
                        override_params = best_params_dict[matched_key]
                    elif 'default' in best_params_dict:
                        print("    ✓ 从 JSON 自动加载 default 的默认最优参数")
                        override_params = best_params_dict['default']
                    else:
                        override_params = {}
                        
                    for param_k, param_v in override_params.items():
                        kwargs[param_k] = param_v

                model = ModelFactory.get_model(model_name=model_name, num_question=num_q, num_skill=num_c, device=device, config=config, **kwargs)
                state_dict = torch.load(ckpt_path, map_location=device)
                
                if isinstance(state_dict, torch.nn.Module):
                    state_dict = state_dict.state_dict()
                
                model_state_dict = model.state_dict()
                clean_state_dict = {}
                for k, v in state_dict.items():
                    # Handle DDP prefix
                    key = k[7:] if k.startswith('module.') else k
                    
                    if key in model_state_dict:
                        param_shape = model_state_dict[key].shape
                        if v.shape != param_shape:
                            raise ValueError(f"模型不匹配，参数 {key} 的形状不一致。当前网络结构预期 {param_shape}，但权重文件中保存的为 {v.shape}。请检查数据集是否更新。")
                    
                    clean_state_dict[key] = v

                model.load_state_dict(clean_state_dict, strict=False)
                print(f"    ✓ 权重加载成功 ({ckpt_path})")
                
                auc, acc, rmse = evaluate_model(model, test_loader, device=device)
                print(f"    ⭐ 结果 - AUC: {auc:.4f}, ACC: {acc:.4f}, RMSE: {rmse:.4f}")
                
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                results.append({"Dataset": dataset_name, "Model": model_name.upper(), "AUC": round(auc, 4), "ACC": round(acc, 4), "RMSE": round(rmse, 4), "Update_Time": timestamp})
                
            except Exception as e:
                print(f"    ✗ 模型评价错误: {e}")
                import traceback
                traceback.print_exc()

    import pandas as pd
    if len(results) > 0:
        df = pd.DataFrame(results)
        print(f"\n{'='*50}")
        print("最终本次测试执行成绩汇总:")
        print(df.to_string(index=False))
        print("="*50)
        
        os.makedirs(os.path.dirname(output_path_fixed), exist_ok=True)
        if os.path.exists(output_path_fixed):
            df_old = pd.read_csv(output_path_fixed)
            if 'Update_Time' not in df_old.columns:
                df_old['Update_Time'] = ''
            df_new = pd.concat([df_old, df], ignore_index=True)
            df_new.drop_duplicates(subset=['Dataset', 'Model'], keep='last', inplace=True)
            df_new.to_csv(output_path_fixed, index=False)
        else:
            df.to_csv(output_path_fixed, index=False)
            
        print(f"\n测试结果已保存至: {output_path_fixed}")
    else:
        if tasks:
            print("\n✔️ 所有指定的模型和数据集组合均已是最新，无执行或未能生成结果。")
        else:
            print("\n未提供或发现任何需要执行的模型和数据集组合。")

def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-h', '--help', action='store_true', help='显示帮助信息')
    parser.add_argument('--models', type=str, nargs='+', default=[], help='列表')
    parser.add_argument('--datasets', type=str, nargs='+', default=[], help='列表')
    parser.add_argument('--batch_size', type=int, default=128, help='批量')
    parser.add_argument('--output', type=str, default='test_results.csv', help='csv')
    parser.add_argument('--auto', action='store_true', help='自动扫描并补充测试未评估的权重')
    parser.add_argument('--force', action='store_true', help='强制覆盖已有结果并更新时间戳')
    
    args = parser.parse_args()
    
    if args.help:
        parser.print_help()
        import pandas as pd
        print("\n\n" + "="*50)
        print("💡 当前 checkpoint 目录中检测到可用的组合:")
        combos = get_available_checkpoints()
        if combos:
            df_combos = pd.DataFrame(combos, columns=['Model', 'Dataset'])
            df_combos.sort_values(by=['Dataset', 'Model'], inplace=True)
            print(df_combos.to_string(index=False))
            print("\n提示: 可直接运行 `python test_evaluate.py --auto` 快速测完上表所有未测组合！")
            print("提示: 如需强制全量测一遍并更新时间戳，请指明 `python test_evaluate.py --auto --force`")
        else:
            print("  未找到任何 _best.pt 文件。")
        print("="*50 + "\n")
        sys.exit(0)
        
    return args

if __name__ == '__main__':
    args = parse_args()
    tasks = []
    
    if args.auto:
        combos = get_available_checkpoints()
        for m, d in combos:
            tasks.append({'model': m, 'dataset': d})
            
    if args.models and args.datasets:
        for d in args.datasets:
            for m in args.models:
                tasks.append({'model': m, 'dataset': d})
    elif not args.auto:
        print("⚠️ 警告: 未指定 --auto 参数或 models及datasets。")
        sys.exit(1)
        
    test_and_report(tasks=tasks, batch_size=args.batch_size, result_csv=args.output, force=args.force)

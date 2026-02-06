import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy import sparse
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold, StratifiedShuffleSplit
import random
import pandas as pd

# 导入项目模块
from config import Config, DEVICE, COLOR_LOG_B, COLOR_LOG_Y, COLOR_LOG_G, COLOR_LOG_END
from params import HyperParameters
from dataset import UserDataset
from util.utils import gen_gikt_graph, build_adj_list

# 导入模型
from gikt import GIKT
from dkt import DKT

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def get_args():
    parser = argparse.ArgumentParser(description='Compare KT Models (Standard Experiment)')
    parser.add_argument('--dataset', type=str, default='assist09', help='Dataset name')
    parser.add_argument('--models', type=str, default='dkt,gikt', help='Comma-separated list of models to compare')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--name', type=str, default='default', help='Name of the experiment config')
    parser.add_argument('--full', action='store_true', help='Use "full" config type (if not set, auto-detects based on dataset name)')
    parser.add_argument('--enable_tf_alignment', action='store_true', default=True, help='Align GIKT with TF logic')
    parser.add_argument('--verbose', action='store_true', default=True, help='Print detailed logs')
    return parser.parse_args()

def init_model(model_name, num_question, num_skill, config, params):
    if model_name.lower() == 'gikt':
        qs_table = torch.tensor(sparse.load_npz(os.path.join(config.PROCESSED_DATA_DIR, 'qs_table.npz')).toarray(), dtype=torch.int64).to(DEVICE)
        q_neighbors_list, s_neighbors_list = build_adj_list(config.PROCESSED_DATA_DIR)
        q_neighbors, s_neighbors = gen_gikt_graph(q_neighbors_list, s_neighbors_list, params.model.size_q_neighbors, params.model.size_s_neighbors)
        return GIKT(
            num_question=num_question,
            num_skill=num_skill,
            q_neighbors=torch.tensor(q_neighbors, dtype=torch.int64).to(DEVICE),
            s_neighbors=torch.tensor(s_neighbors, dtype=torch.int64).to(DEVICE),
            qs_table=qs_table,
            agg_hops=params.model.agg_hops,
            emb_dim=params.model.emb_dim,
            dropout=tuple(params.model.dropout) if isinstance(params.model.dropout, list) else params.model.dropout,
            hard_recap=params.model.hard_recap,
            use_cognitive_model=params.model.use_cognitive_model,
            pre_train=params.model.pre_train,
            agg_method=params.model.agg_method,
            recap_source='hsei' if params.model.use_input_attention else 'hssi',
            enable_tf_alignment=params.model.enable_tf_alignment
        ).to(DEVICE)
    elif model_name.lower() == 'dkt':
        return DKT(num_question=num_question, num_skill=num_skill, emb_dim=params.model.emb_dim, dropout=params.model.dropout[0] if isinstance(params.model.dropout, list) else 0.1).to(DEVICE)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def get_stratified_split(dataset, test_ratio=0.2, seed=42):
    total_len = len(dataset)
    indices = np.arange(total_len)
    print(f"Generating stratification labels for {total_len} samples...")
    masks = dataset.user_mask.float()
    res = dataset.user_res.float()
    lengths = masks.sum(dim=1).numpy()
    corrects = (res * masks).sum(dim=1).numpy()
    accuracies = np.divide(corrects, lengths, out=np.zeros_like(corrects), where=lengths!=0)
    len_bins = [0, 50, 100, 150, 20000]
    len_labels = pd.cut(lengths, bins=len_bins, labels=False, include_lowest=True)
    acc_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
    acc_labels = pd.cut(accuracies, bins=acc_bins, labels=False, include_lowest=True)
    len_labels = np.nan_to_num(len_labels, nan=0).astype(int)
    acc_labels = np.nan_to_num(acc_labels, nan=0).astype(int)
    strat_labels = len_labels * 10 + acc_labels 
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    try:
        train_index, test_index = next(splitter.split(indices, strat_labels))
    except ValueError:
        print(f"{COLOR_LOG_Y}Warning: Stratification classes have too few members. Falling back to Random Split.{COLOR_LOG_END}")
        np.random.shuffle(indices)
        limit = int(total_len * (1 - test_ratio))
        train_index = indices[:limit]
        test_index = indices[limit:]
    print(f"Stratified Split Done. Train: {len(train_index)}, Test: {len(test_index)}")
    return train_index, test_index

def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    total_samples = 0
    
    for batch in dataloader:
        question = batch[:, :, 0].to(torch.long).to(DEVICE)
        response = batch[:, :, 1].to(torch.long).to(DEVICE)
        mask = batch[:, :, 2].to(torch.bool).to(DEVICE)
        interval = batch[:, :, 3].to(torch.float32).to(DEVICE)
        r_time = batch[:, :, 4].to(torch.float32).to(DEVICE)
        
        interval = torch.nan_to_num(interval, nan=0.0)
        r_time = torch.nan_to_num(r_time, nan=0.0)
        
        optimizer.zero_grad()
        if getattr(model, 'model_name', '').lower() == 'gikt':
            y_hat = model(question, response, mask, interval, r_time)
            preds = y_hat[:, 1:]
            targets = response[:, 1:].float()
            mask_valid = mask[:, 1:]
        else:
            y_hat = model(question, response, mask)
            preds = y_hat[:, :-1]
            targets = response[:, 1:].float()
            mask_valid = mask[:, 1:]
            
        preds = preds[mask_valid == 1]
        targets = targets[mask_valid == 1]
        
        if preds.numel() == 0: continue

        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * preds.size(0)
        total_samples += preds.size(0)
        
    return total_loss / total_samples if total_samples > 0 else 0

def evaluate(model, dataloader):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            question = batch[:, :, 0].to(torch.long).to(DEVICE)
            response = batch[:, :, 1].to(torch.long).to(DEVICE)
            mask = batch[:, :, 2].to(torch.bool).to(DEVICE)
            interval = batch[:, :, 3].to(torch.float32).to(DEVICE)
            r_time = batch[:, :, 4].to(torch.float32).to(DEVICE)
            
            interval = torch.nan_to_num(interval, nan=0.0)
            r_time = torch.nan_to_num(r_time, nan=0.0)
            
            if getattr(model, 'model_name', '').lower() == 'gikt':
                y_hat = model(question, response, mask, interval, r_time)
                y_hat = torch.sigmoid(y_hat)
                preds = y_hat[:, 1:]
                targets = response[:, 1:].float()
                mask_valid = mask[:, 1:]
            else:
                y_hat = model(question, response, mask)
                preds = y_hat[:, :-1]
                targets = response[:, 1:].float()
                mask_valid = mask[:, 1:]

            preds_flat = preds[mask_valid == 1]
            targets_flat = targets[mask_valid == 1]
            
            all_preds.extend(preds_flat.cpu().numpy())
            all_targets.extend(targets_flat.cpu().numpy())
    
    if len(all_targets) == 0: return 0, 0
    auc = roc_auc_score(all_targets, all_preds)
    acc = accuracy_score(all_targets, np.array(all_preds) >= 0.5)
    return auc, acc

def run_comparison():
    args = get_args()
    
    # 配置文件逻辑优化: 自动判断使用 sample 还是 full 配置
    if args.full:
        dataset_type = 'full'
    elif 'sample' in args.dataset:
        dataset_type = 'sample'
    else:
        dataset_type = 'full'
        
    exp_config_path = f"config/experiments/exp_{dataset_type}_{args.name}.toml"
    print(f"Loading experiment config from: {exp_config_path}")
    
    if not os.path.exists(exp_config_path):
        # 尝试回退到 full
        fallback_path = f"config/experiments/exp_full_{args.name}.toml"
        if os.path.exists(fallback_path):
             print(f"{COLOR_LOG_Y}Config {exp_config_path} not found. Fallback to {fallback_path}{COLOR_LOG_END}")
             exp_config_path = fallback_path
             params = HyperParameters.load(exp_config_path=exp_config_path)
        else:
             print(f"{COLOR_LOG_Y}Config not found. Using default params.{COLOR_LOG_END}")
             params = HyperParameters()
    else:
        params = HyperParameters.load(exp_config_path=exp_config_path)

    config = Config(args.dataset)
    dataset = UserDataset(config)
    qs_table = sparse.load_npz(os.path.join(config.PROCESSED_DATA_DIR, 'qs_table.npz'))
    num_question = qs_table.shape[0]
    num_skill = qs_table.shape[1]

    print(f"\n{COLOR_LOG_B}Step 1: Splitting 20% Test Set (Stratified){COLOR_LOG_END}")
    dev_idx, test_idx = get_stratified_split(dataset, test_ratio=0.2)
    test_set = Subset(dataset, test_idx)
    dev_set_base = Subset(dataset, dev_idx)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    
    k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
    model_list = args.models.split(',')
    final_results = {m: [] for m in model_list}
    dev_indices_array = np.arange(len(dev_set_base))
    
    for fold_i, (train_rel_idx, val_rel_idx) in enumerate(k_fold.split(dev_indices_array)):
        print(f"\n{COLOR_LOG_B}=== CV Fold {fold_i+1}/5 ==={COLOR_LOG_END}")
        train_ds = Subset(dev_set_base, train_rel_idx)
        val_ds = Subset(dev_set_base, val_rel_idx)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        
        for model_name in model_list:
            print(f"--> Training {model_name}...")
            model = init_model(model_name, num_question, num_skill, config, params)
            optimizer = torch.optim.Adam(model.parameters(), lr=params.train.lr)
            criterion = nn.BCEWithLogitsLoss() if model_name == 'gikt' else nn.BCELoss()
            best_val_auc = 0.0
            patience = 5
            counter = 0
            best_model_state = None
            
            for epoch in range(args.epochs):
                avg_loss = train_epoch(model, train_loader, optimizer, criterion)
                val_auc, val_acc = evaluate(model, val_loader)
                if args.verbose:
                    print(f"    Epoch {epoch+1}: Loss={avg_loss:.4f}, Val AUC={val_auc:.4f}")
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_model_state = model.state_dict()
                    counter = 0
                else:
                    counter += 1
                    if counter >= patience:
                        print(f"    Early stopping at epoch {epoch+1}")
                        break
            
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            test_auc, test_acc = evaluate(model, test_loader)
            print(f"    {model_name} Fold {fold_i+1} Test AUC: {test_auc:.4f}")
            final_results[model_name].append(test_auc)

    print(f"\n{COLOR_LOG_Y}=== Final Report (Average on 20% Test Set) ==={COLOR_LOG_END}")
    for m, aucs in final_results.items():
        mean_v = np.mean(aucs)
        std_v = np.std(aucs)
        print(f"Model: {m:<10} | Mean AUC: {mean_v:.4f} (+/- {std_v:.4f})")

if __name__ == '__main__':
    run_comparison()

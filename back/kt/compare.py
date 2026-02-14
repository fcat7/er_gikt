import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy import sparse
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold, StratifiedShuffleSplit, GroupKFold, GroupShuffleSplit
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
    parser.add_argument('--verbose', action='store_true', default=True, help='Print detailed logs')
    return parser.parse_args()

def init_model(model_name, num_question, num_skill, config, params):
    if model_name.lower() == 'gikt':
        qs_table = torch.tensor(sparse.load_npz(os.path.join(config.PROCESSED_DATA_DIR, 'qs_table.npz')).toarray(), dtype=torch.int64).to(DEVICE)
        q_neighbors_list, s_neighbors_list = build_adj_list(config.PROCESSED_DATA_DIR)
        q_neighbors, s_neighbors = gen_gikt_graph(q_neighbors_list, s_neighbors_list, params.model.size_q_neighbors, params.model.size_s_neighbors)
        model = GIKT(
            num_question=num_question,
            num_skill=num_skill,
            q_neighbors=torch.tensor(q_neighbors, dtype=torch.int64).to(DEVICE),
            s_neighbors=torch.tensor(s_neighbors, dtype=torch.int64).to(DEVICE),
            qs_table=qs_table,
            agg_hops=params.model.agg_hops,
            emb_dim=params.model.emb_dim,
            dropout_linear=params.model.dropout_linear,
            dropout_gnn=params.model.dropout_gnn,
            drop_edge_rate=params.model.drop_edge_rate,
            feature_noise_scale=params.model.feature_noise_scale,
            hard_recap=params.model.hard_recap,
            use_cognitive_model=params.model.use_cognitive_model,
            pre_train=params.model.pre_train,
            agg_method=params.model.agg_method,
            recap_source=params.model.recap_source,
            use_pid=params.model.use_pid,
            pid_mode=params.model.pid_mode,
            data_dir=config.PROCESSED_DATA_DIR
        ).to(DEVICE)
        model.model_name = 'gikt'
        return model
    elif model_name.lower() == 'dkt':
        model = DKT(num_question=num_question, num_skill=num_skill, emb_dim=params.model.emb_dim, dropout=params.model.dropout_linear).to(DEVICE)
        model.model_name = 'dkt'
        return model
    else:
        raise ValueError(f"Unknown model: {model_name}")

def get_stratified_split(dataset, test_ratio=0.2, seed=42):
    total_len = len(dataset)
    indices = np.arange(total_len)
    groups = dataset.groups
    
    # @fix_fzq: If groups exist, use GroupShuffleSplit instead
    if groups is not None:
        print(f"🔒 Detecting Windowed Dataset. Using Group-based stratified split.")
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
        train_index, test_index = next(splitter.split(indices, groups=groups))
        print(f"Group-aware Stratified Split Done. Train: {len(train_index)}, Test: {len(test_index)}")
        return train_index, test_index
    
    # Standard stratified split (no groups)
    print(f"🔓 Using Standard Stratified splitting (no groups detected).")
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
        eval_mask = batch[:, :, 5].to(torch.bool).to(DEVICE)
        
        interval = torch.nan_to_num(interval, nan=0.0)
        r_time = torch.nan_to_num(r_time, nan=0.0)

        
        optimizer.zero_grad()
        if getattr(model, 'model_name', '').lower() == 'gikt':
            y_hat = model(question, response, mask, interval, r_time)
            preds = y_hat[:, 1:]
            targets = response[:, 1:].float()
            mask_valid = mask[:, 1:]
        elif getattr(model, 'model_name', '').lower() == 'dkt':
            # DKT: Input (Q, R) -> Output (Logits if no sigmoid) -> y_hat[:, k+1] predicts R[k+1]
            y_hat = model(question, response, mask)
            # The current output of DKT is actually probabilities or logits?
            # Standard DKT usually predicts logits -> Sigmoid.
            preds = y_hat[:, :-1]
            targets = response[:, 1:].float()
            mask_valid = mask[:, 1:]
        
        eval_mask_valid = eval_mask[:, 1:]
        final_mask = mask_valid & eval_mask_valid
            
        preds = preds[final_mask == 1]
        targets = targets[final_mask == 1]
        
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
            eval_mask = batch[:, :, 5].to(torch.bool).to(DEVICE)
            
            interval = torch.nan_to_num(interval, nan=0.0)
            r_time = torch.nan_to_num(r_time, nan=0.0)

            if getattr(model, 'model_name', '').lower() == 'gikt':
                y_hat = model(question, response, mask, interval, r_time)
                y_hat = torch.sigmoid(y_hat)
                preds = y_hat[:, 1:]
                targets = response[:, 1:].float()
                mask_valid = mask[:, 1:]
            elif getattr(model, 'model_name', '').lower() == 'dkt':
                y_hat = model(question, response, mask)
                # Ensure DKT output is activated if it's logits!
                # Assuming your DKT outputs logits directly.
                y_hat = torch.sigmoid(y_hat) 
                preds = y_hat[:, :-1]
                targets = response[:, 1:].float()
                mask_valid = mask[:, 1:]
            eval_mask_valid = eval_mask[:, 1:]
            final_mask = mask_valid & eval_mask_valid

            preds_flat = preds[final_mask == 1]
            targets_flat = targets[final_mask == 1]
            
            all_preds.extend(preds_flat.cpu().numpy())
            all_targets.extend(targets_flat.cpu().numpy())

    if len(all_targets) == 0: return 0, 0
    auc = roc_auc_score(all_targets, all_preds)
    acc = accuracy_score(all_targets, np.array(all_preds) >= 0.5)

    # @add_fzq: Quick Diagnostic (Only print if it seems like a validation step)
    # Check if we have access to unfiltered data implies we need to track it.
    # For compare.py, we keep it simple but informative.
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

    # @fix_fzq: Group-aware splitting for initial holdout
    groups = dataset.groups
    if groups is not None:
        print(f"🔒 Detecting Windowed Dataset. Using GroupShuffleSplit for Holdout Split (20%).")
        from sklearn.model_selection import GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        dev_idx, test_idx = next(gss.split(dataset, groups=groups))
    else:
        print(f"\n{COLOR_LOG_B}Step 1: Splitting 20% Test Set (Stratified){COLOR_LOG_END}")
        dev_idx, test_idx = get_stratified_split(dataset, test_ratio=0.2)

    test_set = Subset(dataset, test_idx)
    dev_set_base = Subset(dataset, dev_idx)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    
    # @fix_fzq: Group-aware KFold to prevent data leakage in windowed mode
    groups = None
    if dataset.groups is not None:
        groups = dataset.groups[dev_idx]
        print(f"🔒 Detecting Windowed Dataset with {len(np.unique(groups))} unique groups. Using GroupKFold for CV.")
        k_fold = GroupKFold(n_splits=5)
    else:
        print(f"🔓 Using Standard KFold.")
        groups = None
        k_fold = KFold(n_splits=5, shuffle=True, random_state=42)

    model_list = [m.strip().lower() for m in args.models.split(',')]
    final_results = {m: [] for m in model_list}
    dev_indices_array = np.arange(len(dev_set_base))
    
    for fold_i, (train_rel_idx, val_rel_idx) in enumerate(k_fold.split(dev_indices_array, groups=groups)):
        print(f"\n{COLOR_LOG_B}=== CV Fold {fold_i+1}/5 (Double-Check Leakage: Groups Unique) ==={COLOR_LOG_END}")
        if groups is not None:
            train_groups = groups[train_rel_idx]
            val_groups = groups[val_rel_idx]
            intersect = np.intersect1d(train_groups, val_groups)
            if len(intersect) > 0:
                print(f"{COLOR_LOG_Y}⚠️ CRITICAL LEAKAGE DETECTED: {len(intersect)} users in both train and val!{COLOR_LOG_END}")
            else:
                print(f"{COLOR_LOG_G}✅ No User Leakage Detected.{COLOR_LOG_END}")

        print(f"Train samples: {len(train_rel_idx)}, Val samples: {len(val_rel_idx)}")
        train_ds = Subset(dev_set_base, train_rel_idx)
        val_ds = Subset(dev_set_base, val_rel_idx)
        loader_kwargs = {'batch_size': args.batch_size, 'num_workers': 0, 'pin_memory': True}
        train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

        
        for model_name in model_list:
            print(f"\n  → Training {model_name.upper()}...")
            # For DKT, we should use Skill IDs vs Question IDs. 
            # Current Implementation: Uses Question IDs (Sparse Item-KT). This is OK for fairness if embeddings are same dim.
            model = init_model(model_name, num_question, num_skill, config, params)
            optimizer = torch.optim.Adam(model.parameters(), lr=params.train.lr)
            criterion = nn.BCEWithLogitsLoss() if model_name.lower() in ['gikt', 'dkt'] else nn.BCELoss() # @fix_fzq: DKT also output logits if we use unmodified dkt.py 
            # Wait, dkt.py usually has sigmoid? Current dkt.py (from context) has output linear layer -> need sigmoid?
            # Actually, let's check dkt.py. The attached dkt.py doesn't have sigmoid in forward method final return if it's logits.
            # But the evaluate function applies sigmoid. The training loop does NOT apply sigmoid to preds before criterion if using BCEWithLogitsLoss.
            # But wait, original train_epoch for non-GIKT uses:
            # y_hat = model(...)
            # preds = y_hat
            # loss = criterion(preds, targets)
            # If criterion is BCELoss, then preds must be probabilities (0-1).
            # If criterion is BCEWithLogitsLoss, then preds must be logits.
            # Your DKT implementation seems to match GIKT's style (logits). Update criterion to be safe.
            criterion = nn.BCEWithLogitsLoss() 
            
            best_val_auc = 0.0

            patience = 5
            counter = 0
            best_model_state = None
            epoch_times = []
            
            for epoch in range(args.epochs):
                epoch_start = time.time()
                avg_loss = train_epoch(model, train_loader, optimizer, criterion)
                val_auc, val_acc = evaluate(model, val_loader)
                epoch_time = time.time() - epoch_start
                epoch_times.append(epoch_time)
                
                print(f"    Epoch {epoch+1}/{args.epochs}: Loss={avg_loss:.4f}, Val AUC={val_auc:.4f}, Acc={val_acc:.4f}, Time={epoch_time:.2f}s")
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_model_state = model.state_dict()
                    counter = 0
                else:
                    counter += 1
                    if counter >= patience:
                        print(f"    Early stopping at epoch {epoch+1}")
                        break
            
            avg_epoch_time = np.mean(epoch_times)
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            test_auc, test_acc = evaluate(model, test_loader)
            print(f"  ✓ {model_name.upper()} Fold {fold_i+1} | Best Val AUC: {best_val_auc:.4f} | Test AUC: {test_auc:.4f} | Test Acc: {test_acc:.4f} | Avg Epoch Time: {avg_epoch_time:.2f}s")
            final_results[model_name].append(test_auc)

    print(f"\n{COLOR_LOG_Y}{'='*70}")
    print(f"Final Report: 5-Fold CV Results (Test on 20% Holdout Set)")
    print(f"{'='*70}{COLOR_LOG_END}")
    for m in model_list:
        aucs = final_results[m]
        mean_v = np.mean(aucs)
        std_v = np.std(aucs)
        min_v = np.min(aucs)
        max_v = np.max(aucs)
        print(f"\n{COLOR_LOG_B}{m.upper():<10}{COLOR_LOG_END}")
        print(f"  Per-Fold AUC: {[f'{a:.4f}' for a in aucs]}")
        print(f"  Mean AUC: {mean_v:.4f} ± {std_v:.4f}")
        print(f"  Range: [{min_v:.4f}, {max_v:.4f}]")
    print(f"{COLOR_LOG_G}{'='*70}{COLOR_LOG_END}")

if __name__ == '__main__':
    run_comparison()

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from scipy import sparse
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import ShuffleSplit
import itertools
import random
from copy import deepcopy

# 导入项目模块
from config import Config, DEVICE, COLOR_LOG_B, COLOR_LOG_Y, COLOR_LOG_G, COLOR_LOG_END
from params import HyperParameters
from dataset import UserDataset
from util.utils import gen_gikt_graph, build_adj_list
from gikt import GIKT

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_samples = 0
    epoch_auc_sum = 0
    total_loss = 0
    
    for batch in dataloader:
        question = batch[:, :, 0].to(torch.long).to(DEVICE)
        response = batch[:, :, 1].to(torch.long).to(DEVICE)
        mask = batch[:, :, 2].to(torch.bool).to(DEVICE)
        interval_time = batch[:, :, 3].to(torch.float32).to(DEVICE)
        response_time = batch[:, :, 4].to(torch.float32).to(DEVICE)

        if torch.isnan(interval_time).any():
            interval_time = torch.nan_to_num(interval_time, nan=0.0)
        if torch.isnan(response_time).any():
            response_time = torch.nan_to_num(response_time, nan=0.0)
        
        optimizer.zero_grad()
        y_hat = model(question, response, mask, interval_time, response_time)
        
        preds = y_hat[:, :-1]
        targets = response[:, 1:].float()
        mask_valid = mask[:, 1:]
        
        preds = preds[mask_valid == 1]
        targets = targets[mask_valid == 1]

        if isinstance(criterion, nn.BCEWithLogitsLoss):
            preds_prob = torch.sigmoid(preds)
        else:
            preds_prob = preds

        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * preds.size(0)
        
        try:
            # Fast batch AUC for monitoring
            if len(torch.unique(targets)) > 1:
                batch_auc = roc_auc_score(targets.detach().cpu().numpy(), preds_prob.detach().cpu().numpy())
                epoch_auc_sum += batch_auc * preds.size(0)
            else:
                epoch_auc_sum += 0.5 * preds.size(0)
        except ValueError:
            pass

        total_samples += preds.size(0)
        
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    avg_auc = epoch_auc_sum / total_samples if total_samples > 0 else 0
    return avg_loss, avg_auc

def evaluate(model, dataloader):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            question = batch[:, :, 0].to(torch.long).to(DEVICE)
            response = batch[:, :, 1].to(torch.long).to(DEVICE)
            mask = batch[:, :, 2].to(torch.bool).to(DEVICE)
            interval_time = batch[:, :, 3].to(torch.float32).to(DEVICE)
            response_time = batch[:, :, 4].to(torch.float32).to(DEVICE)

            # Nan Check
            if torch.isnan(interval_time).any():
                interval_time = torch.nan_to_num(interval_time, nan=0.0)
            if torch.isnan(response_time).any():
                response_time = torch.nan_to_num(response_time, nan=0.0)
            
            y_hat = model(question, response, mask, interval_time, response_time)
            
            # TF Alignment: Always convert Logits to Prob because we removed sigmoid in model for BCEWithLogitsLoss
            y_hat = torch.sigmoid(y_hat)

            preds = y_hat[:, :-1][mask[:, 1:] == 1]
            targets = response[:, 1:][mask[:, 1:] == 1].float()
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    auc = roc_auc_score(all_targets, all_preds)
    return auc

def run_optimization_search():
    # 0. 解析参数
    parser = argparse.ArgumentParser(description="Grid Search Optimization")
    parser.add_argument('--full', action='store_true', help='Run on full dataset (adjusts config and parameters)')
    args = parser.parse_args()

    # 1. 基础配置
    if args.full:
        base_config_path = "config/experiments/exp_full_default.toml"
        print(f"{COLOR_LOG_Y}!!! RUNNING ON FULL DATASET !!!{COLOR_LOG_END}")
        
        # 针对 Full Dataset 的优化搜索空间 (基于 Sample 结果调整)
        # Sample 结论: Hops=1 最优, Hops=3 显著下降 (过平滑/噪声). Dropout (0.2, 0.4) 表现稳健.
        # Full 策略: 
        # 1. Hops: 锁定在 [1, 2], 放弃 3.
        # 2. Dropout: 全量数据可能需要较少正则化，增加 (0.1, 0.3) 选项; 保留 Baseline (0.2, 0.4).
        search_space = {
            'dropout_tuple': [
                (0.1, 0.2), # Try lower dropout for larger data
                (0.1, 0.3), # Try lower dropout for larger data
                (0.2, 0.4), # Baseline
                (0.2, 0.5)  # Slightly higher GNN dropout test
            ],
            'agg_hops': [1, 2] # Drop 3
        }
        # 全量数据设置
        epochs_override = 10  # 全量数据5个epoch足以观察收敛趋势
        batch_size_override = 128
    else:
        base_config_path = "config/experiments/exp_sample_default.toml"
        # Sample 默认搜索空间
        search_space = {
            'dropout_tuple': [
                (0.2, 0.4), 
                (0.3, 0.5), 
                (0.2, 0.6), 
            ],
            'agg_hops': [1, 2, 3] 
        }
        epochs_override = 15
        batch_size_override = 64

    base_params = HyperParameters.load(exp_config_path=base_config_path)
    dataset_name = base_params.train.dataset_name 
    
    os.environ['DATASET'] = dataset_name
    config = Config(dataset_name)
    
    print(f"{COLOR_LOG_G}Starting Optimization Search (Hops & Dropout) on: {dataset_name}{COLOR_LOG_END}")
    
    # 2. 准备数据
    qs_table_data = sparse.load_npz(os.path.join(config.PROCESSED_DATA_DIR, 'qs_table.npz')).toarray()
    qs_table = torch.tensor(qs_table_data, dtype=torch.int64).to(DEVICE)
    num_question = qs_table.shape[0]
    num_skill = qs_table.shape[1]
    
    q_neighbors_list, s_neighbors_list = build_adj_list(config.PROCESSED_DATA_DIR)
    
    dataset = UserDataset(config)
    
    # 统一划分为 8:2
    rs = ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(rs.split(dataset))
    
    train_sub = Subset(dataset, train_idx)
    test_sub = Subset(dataset, test_idx)
    
    # 3. 定义精准搜索空间 (已在上方根据 dataset 动态定义)
    
    keys, values = zip(*search_space.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Total combinations to test: {len(combinations)}")
    
    results = [] # Store (params_dict, best_auc)

    for i, combo in enumerate(combinations):
        set_seed(42) # 重置种子，确保每组实验初始化一致
        
        print(f"\n{COLOR_LOG_B}=== Experiment {i+1}/{len(combinations)}: {combo} ==={COLOR_LOG_END}")
        
        # 克隆参数并覆盖
        current_params = deepcopy(base_params)
        
        # 覆盖优化参数
        current_params.model.dropout = combo['dropout_tuple'] # 直接读取 Tuple
        current_params.model.agg_hops = combo['agg_hops']
        
        # 覆盖训练参数
        current_params.train.epochs = epochs_override 
        current_params.train.batch_size = batch_size_override
        
        # Print full config for verification
        print(f"Params: Dropout={current_params.model.dropout}, Hops={current_params.model.agg_hops}")
        
        # DataLoader
        train_loader = DataLoader(train_sub, batch_size=current_params.train.batch_size, shuffle=True)
        test_loader = DataLoader(test_sub, batch_size=current_params.train.batch_size, shuffle=False)
        
        # 生成图
        q_neighbors, s_neighbors = gen_gikt_graph(
            q_neighbors_list, s_neighbors_list, 
            current_params.model.size_q_neighbors, 
            current_params.model.size_s_neighbors
        )
        q_neighbors = torch.tensor(q_neighbors, dtype=torch.int64).to(DEVICE)
        s_neighbors = torch.tensor(s_neighbors, dtype=torch.int64).to(DEVICE)
        
        # 初始化模型
        model = GIKT(
            num_question=num_question,
            num_skill=num_skill,
            q_neighbors=q_neighbors,
            s_neighbors=s_neighbors,
            qs_table=qs_table,
            agg_hops=current_params.model.agg_hops,
            emb_dim=current_params.model.emb_dim,
            dropout=current_params.model.dropout,
            hard_recap=current_params.model.hard_recap,
            use_cognitive_model=current_params.model.use_cognitive_model,
            pre_train=current_params.model.pre_train,
            data_dir=config.PROCESSED_DATA_DIR,
            agg_method='gcn', 
            recap_source=current_params.model.recap_source,
            use_pid=current_params.model.use_pid,
            pid_mode=current_params.model.pid_mode,
            pid_ema_alpha=current_params.model.pid_ema_alpha,
            pid_lambda=current_params.model.pid_lambda,
            guessing_prob_init=current_params.model.guessing_prob_init,
            slipping_prob_init=current_params.model.slipping_prob_init
        ).to(DEVICE)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=current_params.train.lr, weight_decay=current_params.train.weight_decay)
        criterion = nn.BCEWithLogitsLoss()
        
        best_test_auc = 0.0
        best_train_auc = 0.0
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(current_params.train.epochs):
            loss, train_auc = train_epoch(model, train_loader, optimizer, criterion)
            test_auc = evaluate(model, test_loader)
            
            if test_auc > best_test_auc:
                best_test_auc = test_auc
                best_train_auc = train_auc
                best_epoch = epoch + 1
                patience_counter = 0
            else:
                patience_counter += 1
                
            # print(f"  Ep {epoch+1}: L={loss:.4f} T_AUC={train_auc:.4f} V_AUC={test_auc:.4f}")

            if current_params.train.patience > 0 and patience_counter >= current_params.train.patience:
                # print(f"  Early stopping at epoch {epoch+1}")
                break
            
        gap = best_train_auc - best_test_auc
        print(f"{COLOR_LOG_Y}  -> Result: Best Test AUC={best_test_auc:.4f} (Train AUC={best_train_auc:.4f}, Gap={gap:.4f}) at Epoch {best_epoch}{COLOR_LOG_END}")
        
        results.append({
            'dropout': combo['dropout_tuple'],
            'hops': combo['agg_hops'],
            'test_auc': best_test_auc,
            'train_auc': best_train_auc,
            'gap': gap,
            'epoch': best_epoch
        })
        
    # 4. 总结结果
    print(f"\n{COLOR_LOG_G}=== Optimization Results (Sorted by Test AUC) ==={COLOR_LOG_END}")
    results.sort(key=lambda x: x['test_auc'], reverse=True)
    
    print(f"{'Rank':<5} | {'Dropout':<15} | {'Hops':<5} | {'Test AUC':<10} | {'Train AUC':<10} | {'Gap':<10} | {'Epoch':<5}")
    print("-" * 85)
    for rank, res in enumerate(results):
        print(f"{rank+1:<5} | {str(res['dropout']):<15} | {res['hops']:<5} | {res['test_auc']:.4f}     | {res['train_auc']:.4f}     | {res['gap']:.4f}     | {res['epoch']:<5}")

if __name__ == '__main__':
    run_optimization_search()
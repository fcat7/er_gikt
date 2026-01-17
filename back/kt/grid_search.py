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

def evaluate(model, dataloader, enable_tf_alignment=True):
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
            
            # Logits -> Prob
            if enable_tf_alignment:
                y_hat = torch.sigmoid(y_hat)

            preds = y_hat[:, :-1][mask[:, 1:] == 1]
            targets = response[:, 1:][mask[:, 1:] == 1].float()
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    auc = roc_auc_score(all_targets, all_preds)
    return auc

def run_grid_search():
    # 1. 基础配置
    # 加载默认的 sample 配置作为基底
    base_config_path = "config/experiments/exp_sample_default.toml"
    base_params = HyperParameters.load(exp_config_path=base_config_path)
    dataset_name = base_params.train.dataset_name # 'assist09-sample_10%'
    
    os.environ['DATASET'] = dataset_name
    config = Config(dataset_name)
    
    print(f"{COLOR_LOG_G}Starting Grid Search on: {dataset_name}{COLOR_LOG_END}")
    
    # 2. 准备数据 (只加载一次)
    qs_table_data = sparse.load_npz(os.path.join(config.PROCESSED_DATA_DIR, 'qs_table.npz')).toarray()
    qs_table = torch.tensor(qs_table_data, dtype=torch.int64).to(DEVICE)
    num_question = qs_table.shape[0]
    num_skill = qs_table.shape[1]
    
    q_neighbors_list, s_neighbors_list = build_adj_list(config.PROCESSED_DATA_DIR)
    
    dataset = UserDataset(config)
    
    # 使用 ShuffleSplit 固定划分 8:2
    rs = ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(rs.split(dataset))
    
    train_sub = Subset(dataset, train_idx)
    test_sub = Subset(dataset, test_idx)
    
    # 3. 定义搜索空间
    # 根据 pyKT 表格和 GIKT 特性建议的搜索空间
    search_space = {
        'learning_rate': [1e-3, 1e-4, 1e-5],
        'emb_dim': [64],                  # 暂时固定维度以控制搜索规模 (3*1*4*3=36组)
        'dropout': [0.05, 0.1, 0.3, 0.5],
        'agg_hops': [3],
        # 'agg_hops': [1, 2, 3],
    }
    
    keys, values = zip(*search_space.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Total combinations to test: {len(combinations)}")
    
    results = [] # Store (params_dict, best_auc)

    for i, combo in enumerate(combinations):
        set_seed(42) # 重置种子，确保每组实验初始化一致
        
        print(f"\n{COLOR_LOG_B}=== Experiment {i+1}/{len(combinations)}: {combo} ==={COLOR_LOG_END}")
        
        # 克隆参数并覆盖
        current_params = deepcopy(base_params)
        
        # 覆盖 grid 参数
        current_params.train.lr = combo['learning_rate']
        current_params.model.emb_dim = combo['emb_dim']
        current_params.model.dropout = (combo['dropout'], combo['dropout']) # GIKT使用tuple
        current_params.model.agg_hops = combo['agg_hops']
        current_params.train.epochs = 15 # 搜索时可以适当减少 epoch
        
        # Print full config for verification
        print(current_params)
        
        # DataLoader (Batch size 保持 32 或根据 grid 调整)
        train_loader = DataLoader(train_sub, batch_size=current_params.train.batch_size, shuffle=True)
        test_loader = DataLoader(test_sub, batch_size=current_params.train.batch_size, shuffle=False)
        
        # 重新生成图结构 (如果 hops 影响图生成，或者 size_neighbors 变了)
        # GIKT 的 gen_gikt_graph 实际上只做从 list 到 matrix 的转换，hops 是模型内部参数
        # 但为了一致性，这里还是生成一遍
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
            agg_method='gcn', # 这里固定，也可以加入搜索
            recap_source='hsei' if current_params.model.use_input_attention else 'hssi',
            enable_tf_alignment=current_params.model.enable_tf_alignment
        ).to(DEVICE)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=current_params.train.lr)
        criterion = nn.BCEWithLogitsLoss() # GIKT default with alignment
        
        best_auc = 0.0
        
        for epoch in range(current_params.train.epochs):
            loss, train_auc = train_epoch(model, train_loader, optimizer, criterion)
            test_auc = evaluate(model, test_loader, current_params.model.enable_tf_alignment)
            
            if test_auc > best_auc:
                best_auc = test_auc
                
            # 简单的 Early Stopping 打印 (可选)
            # print(f"  Ep {epoch+1}: L={loss:.4f} T_AUC={train_auc:.4f} V_AUC={test_auc:.4f}")
            
        print(f"{COLOR_LOG_Y}  -> Best Test AUC: {best_auc:.4f}{COLOR_LOG_END}")
        results.append((combo, best_auc))
        
    # 4. 总结结果
    print(f"\n{COLOR_LOG_G}=== Grid Search Results (Sorted) ==={COLOR_LOG_END}")
    results.sort(key=lambda x: x[1], reverse=True)
    
    for rank, (combo, auc) in enumerate(results):
        print(f"Rank {rank+1}: AUC={auc:.4f} | Params: {combo}")

if __name__ == '__main__':
    run_grid_search()

import os
import time
import argparse
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy import sparse
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold, ShuffleSplit
import random

# 导入项目模块
from config import Config, DEVICE, COLOR_LOG_B, COLOR_LOG_Y, COLOR_LOG_G, COLOR_LOG_END
from params import HyperParameters
from dataset import UserDataset
from util.utils import gen_gikt_graph, build_adj_list

# 导入不同模型
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
    parser = argparse.ArgumentParser(description='Compare KT Models')
    parser.add_argument('--dataset', type=str, default='assist09', help='Dataset name')
    parser.add_argument('--models', type=str, default='dkt,gikt', help='Comma-separated list of models to compare')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    
    # GIKT specific defaults
    parser.add_argument('--gikt_agg_method', type=str, default='gcn', help='GIKT aggregation method (gcn/gat)')
    parser.add_argument('--gikt_recap', type=str, default='hsei', help='GIKT recap source (hsei/hssi)')
    parser.add_argument('--enable_tf_alignment', action='store_true', default=True, help='Align GIKT with TF logic')
    
    # Dataset control
    parser.add_argument('--full', action='store_true', help='Use full dataset (overrides --mode)')
    parser.add_argument('--name', type=str, default='default', help='Name of the experiment config (e.g. default)')
    parser.add_argument('--verbose', action='store_true', default=True, help='Print detailed logs per epoch')

    return parser.parse_args()

def init_model(model_name, num_question, num_skill, config, params, extra_args):
    """
    统一模型初始化接口
    """
    if model_name.lower() == 'gkt' or model_name.lower() == 'gikt':
        # 加载图数据
        qs_table = torch.tensor(sparse.load_npz(os.path.join(config.PROCESSED_DATA_DIR, 'qs_table.npz')).toarray(), dtype=torch.int64).to(DEVICE)
        q_neighbors_list, s_neighbors_list = build_adj_list(config.PROCESSED_DATA_DIR)
        q_neighbors, s_neighbors = gen_gikt_graph(q_neighbors_list, s_neighbors_list, params.model.size_q_neighbors, params.model.size_s_neighbors)
        q_neighbors = torch.tensor(q_neighbors, dtype=torch.int64).to(DEVICE)
        s_neighbors = torch.tensor(s_neighbors, dtype=torch.int64).to(DEVICE)

        return GIKT(
            num_question=num_question,
            num_skill=num_skill,
            q_neighbors=q_neighbors,
            s_neighbors=s_neighbors,
            qs_table=qs_table,
            agg_hops=params.model.agg_hops,
            emb_dim=params.model.emb_dim,
            dropout=tuple(params.model.dropout) if isinstance(params.model.dropout, list) else params.model.dropout,
            hard_recap=params.model.hard_recap,
            use_cognitive_model=params.model.use_cognitive_model,
            pre_train=params.model.pre_train,
            data_dir=config.PROCESSED_DATA_DIR,
            agg_method=params.model.agg_method, # Align with train_test.py
            recap_source='hsei' if params.model.use_input_attention else 'hssi', # Align with train_test.py logic
            enable_tf_alignment=params.model.enable_tf_alignment # Align with train_test.py
        ).to(DEVICE)
        
    elif model_name.lower() == 'dkt':
        # DKT 作为 Baseline，使用 Question-Level 追踪
        return DKT(
            num_question=num_question,
            num_skill=num_skill,
            emb_dim=params.model.emb_dim,
            dropout=params.model.dropout[0] if isinstance(params.model.dropout, list) else 0.1
        ).to(DEVICE)
        
    else:
        raise ValueError(f"Unknown model: {model_name}")

def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    total_samples = 0
    epoch_auc_sum = 0
    
    for batch in dataloader:
        # question, response, mask, interval_time, response_time = (x.to(DEVICE) for x in batch)
        # 修正数据解包逻辑: [Batch, Seq, Features] -> Slice
        question = batch[:, :, 0].to(torch.long).to(DEVICE)
        response = batch[:, :, 1].to(torch.long).to(DEVICE)
        mask = batch[:, :, 2].to(torch.bool).to(DEVICE)
        interval_time = batch[:, :, 3].to(torch.float32).to(DEVICE)
        response_time = batch[:, :, 4].to(torch.float32).to(DEVICE)

        # 安全性检查: 处理 NaN (针对 Assist09 等含有脏数据的情况)
        if torch.isnan(interval_time).any():
            interval_time = torch.nan_to_num(interval_time, nan=0.0)
        if torch.isnan(response_time).any():
            response_time = torch.nan_to_num(response_time, nan=0.0)
        
        optimizer.zero_grad()
        
        # 兼容不同模型的 forward 签名
        if getattr(model, 'model_name', '').lower() == 'gikt':
            y_hat = model(question, response, mask, interval_time, response_time)
            # GIKT follows train_test.py: Output is already aligned to target (no shifting needed)
            preds = y_hat
            targets = response.float()
            mask_valid = mask
        else:
            # DKT / Other Baselines: Usually predict next token, requiring shift
            y_hat = model(question, response, mask)
            preds = y_hat[:, :-1]
            targets = response[:, 1:].float()
            mask_valid = mask[:, 1:]
        
        # Apply mask
        preds = preds[mask_valid == 1]
        targets = targets[mask_valid == 1]

        # 如果输出是 Logits，需要转为概率用于计算 AUC
        # 注意: 这里的 criterion 可能是 BCEWithLogitsLoss (接受 logits) 或 BCELoss (接受 probs)
        # 但我们为了计算 Training AUC，需要统一拿到概率值 preds_prob
        if isinstance(criterion, nn.BCEWithLogitsLoss):
             preds_prob = torch.sigmoid(preds)
        else:
             preds_prob = preds

        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * preds.size(0)
        
        # Accumulate metrics for Training AUC
        try:
             batch_auc = roc_auc_score(targets.detach().cpu().numpy(), preds_prob.detach().cpu().numpy())
             epoch_auc_sum += batch_auc * preds.size(0)
        except ValueError:
             # Handle cases with only one class in batch
             pass

        total_samples += preds.size(0)
        
    avg_loss = total_loss / total_samples
    avg_auc = epoch_auc_sum / total_samples if total_samples > 0 else 0
    return avg_loss, avg_auc

def evaluate(model, dataloader, params=None):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            # question, response, mask, interval_time, response_time = (x.to(DEVICE) for x in batch)
            # 修正数据解包逻辑
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
            
            if getattr(model, 'model_name', '').lower() == 'gikt':
                y_hat = model(question, response, mask, interval_time, response_time)
                
                # Check alignment logic from params if available, else infer
                use_logits = False
                if params and params.model.enable_tf_alignment:
                    use_logits = True
                elif isinstance(model, nn.Module) and hasattr(model, 'enable_tf_alignment') and model.enable_tf_alignment:
                    use_logits = True

                if use_logits: # Output is Logits
                    y_hat = torch.sigmoid(y_hat)
                
                # GIKT: No shifting
                preds = y_hat
                targets = response.float()
                mask_valid = mask
            else:
                y_hat = model(question, response, mask)
                # DKT / Other: Shift
                preds = y_hat[:, :-1]
                targets = response[:, 1:].float()
                mask_valid = mask[:, 1:]
                
            preds = preds[mask_valid == 1]
            targets = targets[mask_valid == 1]
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    auc = roc_auc_score(all_targets, all_preds)
    acc = accuracy_score(all_targets, np.array(all_preds) >= 0.5)
    return auc, acc

def run_comparison():
    args = get_args()
    
    # 1. Config & Params Setup
    # 构建配置文件路径
    dataset_type = 'full' if args.full else 'sample'
    exp_config_path = f"config/experiments/exp_{dataset_type}_{args.name}.toml"
    print(f"Loading experiment config from: {exp_config_path}")
    
    # 加载超参数
    if not os.path.exists(exp_config_path):
        raise FileNotFoundError(f"Config file not found: {exp_config_path}")
        
    params = HyperParameters.load(exp_config_path=exp_config_path)
    
    # 如果命令行指定了 dataset，覆盖配置文件中的设置 (可选)
    # 但通常我们信赖配置文件。这里为了保持一致性，使用配置文件的 dataset_name
    dataset_name = params.train.dataset_name
    if args.dataset != 'assist09' and args.dataset != dataset_name:
        print(f"{COLOR_LOG_Y}Warning: CLI dataset '{args.dataset}' differs from Config '{dataset_name}'. Using Config value.{COLOR_LOG_END}")
    
    os.environ['DATASET'] = dataset_name
    config = Config(dataset_name)
    
    print(f"{COLOR_LOG_G}Starting Comparison on Dataset: {dataset_name}{COLOR_LOG_END}")
    print(params)
    print(f"Models: {args.models}")

    
    # 2. Data Preparation
    # 加载元数据以获取 num_question/skill
    qs_table = sparse.load_npz(os.path.join(config.PROCESSED_DATA_DIR, 'qs_table.npz'))
    num_question = qs_table.shape[0]
    num_skill = qs_table.shape[1]
    
    dataset = UserDataset(config)
    print(f"Total Users: {len(dataset)}")
    
    # 3. K-Fold Cross Validation (Align with train_test.py logic)
    if params.train.k_fold == 1:
        # 如果 k_fold 为 1，使用 ShuffleSplit 进行单次划分 (80% 训练, 20% 测试)
        k_fold = ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    else:
        k_fold = KFold(n_splits=params.train.k_fold, shuffle=True, random_state=42)
        
    model_list = args.models.split(',')
    
    # 存储结果 {model_name: [auc_fold1, auc_fold2...]}
    results = {m: [] for m in model_list}
    
    for fold, (train_idx, test_idx) in enumerate(k_fold.split(dataset)):
        # 严格对齐 train_test.py 的 shuffle 逻辑
        train_sub = Subset(dataset, train_idx)
        test_sub = Subset(dataset, test_idx)
        
        # train_test.py 中使用了 params.common.num_workers
        train_loader = DataLoader(train_sub, batch_size=args.batch_size, shuffle=True, 
                                num_workers=params.common.num_workers if DEVICE.type != 'cpu' else 0)
        test_loader = DataLoader(test_sub, batch_size=args.batch_size, shuffle=False, 
                                num_workers=params.common.num_workers if DEVICE.type != 'cpu' else 0)
        
        print(f"\n{COLOR_LOG_B}=== Fold {fold+1}/{k_fold.get_n_splits()} ==={COLOR_LOG_END}")        
        for model_name in model_list:
            print(f"--> Training {model_name}...")
            
            # Init Model
            model = init_model(model_name, num_question, num_skill, config, params, args)
            
            # Init Optimizer & Loss
            optimizer = torch.optim.Adam(model.parameters(), lr=params.train.lr)
            # 注意: 如果 GIKT 开启了 enable_tf_alignment，它输出的是 Logits，需要 BCEWithLogitsLoss
            # 但是 DKT 当前实现输出的是 Sigmoid。
            # 为了统一比较，建议标准化所有模型输出：
            # 这里简单起见，假设所有 forward 都返回 Prob (Sigmoid后)。
            # 如果 GIKT 返回 Logits，需要在 forward 补 sigmoid 或者在这里处理。
            # 检查 GIKT: predict 中根据 enable_tf_alignment 分支。如果 True，返回 Logits。
            # 检查 DKT: 返回 Sigmoid。
            
            # ** 关键处理 **
            # 我们统一要求模型在 Evaluation 时输出概率。在 Training 时，根据模型特性选 Loss。
            # 这里简化处理：强制使用 BCELoss，意味着要求所有模型 forward 输出概率。
            # 即使 GIKT hsei_v3.2 倾向于 Logits，为了公平对比基线，我们可以在这里 wrapper 一下，
            # 或者修改 Loss 选择逻辑。
            
            if model_name == 'gikt' and params.model.enable_tf_alignment:
                criterion = nn.BCEWithLogitsLoss()
                # GIKT v3.2 align 返回 Logits
            else:
                criterion = nn.BCELoss()
                # DKT 返回 Sigmoid
            
            best_auc = 0.0
            for epoch in range(args.epochs):
                train_loss, train_auc = train_epoch(model, train_loader, optimizer, criterion)
                
                # Eval
                # Pass params to evaluate for correct Logit handling
                auc, acc = evaluate(model, test_loader, params=params)

                if args.verbose:
                    print(f"    Epoch {epoch+1}/{args.epochs}: Train Loss={train_loss:.4f}, Train AUC={train_auc:.4f} | Test AUC={auc:.4f}")
                
                if auc > best_auc:
                    best_auc = auc
            
            print(f"    {model_name} Best AUC: {best_auc:.4f}")
            results[model_name].append(best_auc)
            
    # Final Report
    print(f"\n{COLOR_LOG_Y}=== Final Comparison Results ==={COLOR_LOG_END}")
    for m, aucs in results.items():
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        print(f"Model: {m:<10} | Mean AUC: {mean_auc:.4f} (+/- {std_auc:.4f})")

if __name__ == '__main__':
    run_comparison()

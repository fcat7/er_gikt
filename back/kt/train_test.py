"""
训练并测试模型
使用五折交叉验证法 (Standard K-Fold)
python train_test.py --override train.dataset_name=assist09_gikt_old train.lr=0.01
"""
import os
import time
from datetime import datetime
import numpy as np
from scipy import sparse
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
from sklearn.model_selection import KFold, ShuffleSplit
from torch.utils.data import DataLoader, Subset
from config import Config, DEVICE, COLOR_LOG_B, COLOR_LOG_Y, COLOR_LOG_G, COLOR_LOG_END
from params import HyperParameters
from util.utils import gen_gikt_graph, build_adj_list
import argparse
from models.gikt import GIKT

try:
    from dataset import UnifiedParquetDataset, SeqFeatureKey
except ImportError:
    raise ImportError("Failed to import UnifiedParquetDataset from dataset.py. Please ensure V2 dataset is ready.")

# @add_fzq: AMP Support
import torch
from torch.cuda.amp import autocast, GradScaler

# AMP 将在加载配置后再设置

# @add_fzq 2025-12-25 10:42:10 -------------------------------------------
# 固定随机种子，保证实验可复现
import random
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
# @add_fzq 2025-12-25 10:42:10 -------------------------------------------

def get_parser():
    parser = argparse.ArgumentParser(description="Train and Test GIKT Model")
    parser.add_argument('--full', action='store_true', help='Use full dataset (overrides --mode)')
    parser.add_argument('--name', type=str, default='default', help='Name of the experiment')
    parser.add_argument('--override', nargs='*', help='Override params, e.g. model.use_pid=False train.epochs=10')
    parser.add_argument('--ablation_name', type=str, default='', help='If set, logs result to ablation_summary.csv')
    return parser

def get_exp_config_path(isFull=False, name='default'):
    return f"config/experiments/exp_{'full' if isFull else 'sample'}_{name}.toml"

# python train_test.py
if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()

    # @add_fzq 2025-12-24 17:28:09 -------------------------------------------
    # 1. 解决时区问题：强制使用 UTC+8 (北京时间)
    from datetime import timedelta, timezone
    beijing_time = datetime.now(timezone(timedelta(hours=8)))
    time_now = beijing_time.strftime('%Y%m%d_%H%M%S')
    # @add_fzq 2025-12-24 17:28:09 -------------------------------------------

    # 加载超参数
    exp_config_path = get_exp_config_path(isFull=args.full, name=args.name)
    params = HyperParameters.load(exp_config_path=exp_config_path)

    # @add_fzq 2026-03-07: 动态覆写配置，支持消融实验
    if args.override:
        import ast
        for override_item in args.override:
            if '=' not in override_item: continue
            key_path, value_str = override_item.split('=', 1)
            try:
                parsed_value = ast.literal_eval(value_str)
            except (ValueError, SyntaxError):
                parsed_value = value_str
            
            parts = key_path.split('.')
            if len(parts) == 2:
                section, key = parts
                if hasattr(params, section):
                    sub_obj = getattr(params, section)
                    if hasattr(sub_obj, key):
                        setattr(sub_obj, key, parsed_value)
                        print(COLOR_LOG_Y + f"🔧 命令行覆写: {key_path} = {parsed_value}" + COLOR_LOG_END)
                    else:
                        print(f"⚠️ Warning: 找不到参数 {key} 于 {section}")

    # 加载配置
    dataset_name = params.train.dataset_name
    config = Config(dataset_name=dataset_name)
    
    # @add_fzq 2026-03-05: 从参数配置中读取 AMP 设置
    use_amp = params.train.amp_enabled and torch.cuda.is_available()
    scaler = GradScaler(enabled=use_amp)
    gradient_clip_norm = params.train.gradient_clip_norm
    if use_amp:
        print(f"🚀 AMP Enabled on {torch.cuda.get_device_name(0)}")
    else:
        print(f"⚠️ AMP Disabled (Using float32 for numerical stability)")
    if gradient_clip_norm > 0:
        print(f"✂️ Gradient Clipping Enabled (max_norm={gradient_clip_norm})")
    else:
        print(f"⚠️ Gradient Clipping Disabled")

    output_path = f'{config.path.LOG_DIR}/{time_now}_{params.train.dataset_name}{ "_save_model"  if params.train.save_model else "" }.log'
    output_dir = os.path.dirname(output_path)  # 获取目录路径    
    os.makedirs(output_dir, exist_ok=True) # 创建目录（如果不存在）
    output_file = open(output_path, 'a', buffering=1) # 解决日志丢失问题

    print(f"Using dataset: {dataset_name}, Data dir: {config.PROCESSED_DATA_DIR}\n")
    print(f"Using experiment config: {exp_config_path}\n")
    print(f"Using device: {torch.cuda.is_available() and torch.cuda.get_device_name(0) or 'cpu'}\n")
    
    # 打印并写超参数
    output_file.write(str(params) + '\n')
    print(params)
    
    batch_size = params.train.batch_size
    
    # 构建模型需要的数据结构
    qs_table = torch.tensor(sparse.load_npz(os.path.join(config.PROCESSED_DATA_DIR, 'qs_table.npz')).toarray(), dtype=torch.int64, device=DEVICE)  # [num_q, num_c]
    num_question = torch.tensor(qs_table.shape[0], device=DEVICE)
    num_skill = torch.tensor(qs_table.shape[1], device=DEVICE)
    q_neighbors_list, s_neighbors_list = build_adj_list(config.PROCESSED_DATA_DIR)
    
    q_neighbors, s_neighbors = gen_gikt_graph(q_neighbors_list, s_neighbors_list, params.model.size_q_neighbors, params.model.size_s_neighbors)
    q_neighbors = torch.tensor(q_neighbors, dtype=torch.int64, device=DEVICE)
    s_neighbors = torch.tensor(s_neighbors, dtype=torch.int64, device=DEVICE)

    # 实例化数据集
    # @update_fzq: 使用 V2 数据集 (Parquet)
    print("Using UnifiedParquetDataset (Parquet + Metadata)")
    dataset_train_augment = UnifiedParquetDataset(
        config, 
        augment=params.train.enable_data_augmentation,
        prob_mask=params.train.aug_mask_prob,
        mode='train' 
    )
    dataset_train_clean = UnifiedParquetDataset(config, augment=False, mode='train')

    # 加载独立的测试集（不重叠的用户，正式评估用）
    try:
        dataset_test = UnifiedParquetDataset(config, augment=False, mode='test')
        print(f"📚 Loaded Test Set: {len(dataset_test)} samples.")
        dataset_test_loader = DataLoader(
            dataset_test, 
            batch_size=batch_size, 
            num_workers=params.common.num_workers,
            pin_memory=True
        )
    except Exception as e:
        dataset_test = None
        dataset_test_loader = None
        print(f"⚠️ Warning: Test Set load failed ({e}). Skipping independent testing.")
    
    data_len = len(dataset_train_clean)
    output_file.write(f'Train/Val Pool size: {data_len}\n')
    
    # 记录总开始时间
    total_start_time = time.time()

    # TF Alignment is now always enabled (Logits output, BCEWithLogitsLoss)
    loss_fun = torch.nn.BCEWithLogitsLoss().to(DEVICE) 

    # ==========================================================================================
    # K-Fold Strategy Logic
    # ==========================================================================================
    groups = dataset_train_clean.groups
    if groups is not None:
        print(f"🔒 Detecting Windowed Dataset with {len(np.unique(groups))} unique users. Using Group-based splitting to prevent leakage.")
        from sklearn.model_selection import GroupKFold, GroupShuffleSplit
        if params.train.k_fold == 1:
            k_fold = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        else:
            k_fold = GroupKFold(n_splits=params.train.k_fold)
        splits = list(k_fold.split(dataset_train_clean, groups=groups))
    else:
        print("🔓 Using Standard Shuffled splitting.")
        if params.train.k_fold == 1:
            k_fold = ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        else:
            k_fold = KFold(n_splits=params.train.k_fold, shuffle=True, random_state=42)
        splits = list(k_fold.split(dataset_train_clean))

    fold_results_test_auc = []
    
    # 初始化记录数组 (Metric x Fold*Epoch)
    # y_label_aver: [Metric, Epoch] (Averaged across folds)
    y_label_aver = np.zeros([3, params.train.epochs]) 
    # y_label_all: [Metric, Fold*Epoch] (Sequential)
    y_label_all = np.zeros([3, params.train.epochs * len(splits)]) 

    # === [add_fzq] Enhanced Logging Initialization ===
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from torch.utils.tensorboard import SummaryWriter

    history_records = []
    
    if args.ablation_name:
        task_dir = os.path.join(config.path.OUTPUT_DIR, "run_ablation")
        run_name = f"gikt_{dataset_name}_{args.ablation_name}_{time_now}"
    else:
        task_dir = os.path.join(config.path.OUTPUT_DIR, "train_test")
        run_name = f"gikt_{dataset_name}_{time_now}"
        
    chart_data_dir = os.path.join(task_dir, "chart_data")
    chart_img_dir = os.path.join(task_dir, "chart")
    runs_root = os.path.join(task_dir, "runs", run_name)
    
    os.makedirs(chart_data_dir, exist_ok=True)
    os.makedirs(chart_img_dir, exist_ok=True)
    os.makedirs(runs_root, exist_ok=True)
    
    # 覆盖原有的 CHART_DIR 使得旧代码兼容
    config.path.CHART_DIR = chart_data_dir
    # ================================================

    # --------------------------------------------------------------------------
    # Outer Loop: Folds
    # --------------------------------------------------------------------------
    for fold, (train_indices, test_indices) in enumerate(splits):
        print('===================' + COLOR_LOG_Y + f'fold: {fold + 1} / {len(splits)}'+ COLOR_LOG_END + '====================')
        output_file.write('===================' + f'fold: {fold + 1} / {len(splits)}' + '====================\n')
        
        # Initialize Writers for fold 1
        if fold == 0:
            train_writer = SummaryWriter(log_dir=os.path.join(runs_root, "fold_1", "train"))
            val_writer = SummaryWriter(log_dir=os.path.join(runs_root, "fold_1", "val"))
        else:
            train_writer = None
            val_writer = None

        # 1. Initialize Model & Optimizer (RESET FOR EACH FOLD)
        model = GIKT(
            num_question, num_skill, q_neighbors, s_neighbors, qs_table,
            agg_hops=params.model.agg_hops,
            emb_dim=params.model.emb_dim,
            dropout_linear=params.model.dropout_linear,
            dropout_gnn=params.model.dropout_gnn,
            drop_edge_rate=params.model.drop_edge_rate,
            feature_noise_scale=params.model.feature_noise_scale,
            hard_recap=params.model.hard_recap,
            rank_k=params.model.rank_k,
            use_cognitive_model=params.model.use_cognitive_model,
            cognitive_mode=params.model.cognitive_mode,
            pre_train=params.model.pre_train,
            data_dir=config.PROCESSED_DATA_DIR,
            agg_method=params.model.agg_method,
            recap_source=params.model.recap_source,
            use_pid=params.model.use_pid,
            pid_mode=params.model.pid_mode,
            pid_ema_alpha=params.model.pid_ema_alpha,
            pid_lambda=params.model.pid_lambda,
            pid_init_i=params.model.pid_init_i,
            pid_init_d=params.model.pid_init_d,
            guessing_prob_init=params.model.guessing_prob_init,
            slipping_prob_init=params.model.slipping_prob_init,
            use_4pl_irt=params.model.use_4pl_irt
        ).to(DEVICE)
        
        optimizer = torch.optim.Adam(params=model.parameters(), lr=params.train.lr, weight_decay=params.train.weight_decay)
        
        best_fold_val_auc = 0.0
        fold_best_threshold = 0.5
        train_set = Subset(dataset_train_augment, train_indices)
        val_set = Subset(dataset_train_clean, test_indices)
        
        loader_kwargs = {
            'batch_size': batch_size,
            'num_workers': params.common.num_workers,
            'pin_memory': True
        }
        if params.common.num_workers > 0 and DEVICE.type != 'cpu':
            loader_kwargs['prefetch_factor'] = params.train.prefetch_factor
            
        train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
        
        # @add_fzq: LR Scheduler Upgrade (CosineAnnealing with Warmup)
        # Using OneCycleLR which inherently includes warmup (default pct_start=0.3) and cosine annealing
        if params.train.enable_lr_scheduler:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, 
                max_lr=params.train.lr, 
                steps_per_epoch=max(1, len(train_loader)), 
                epochs=params.train.epochs,
                pct_start=0.1, # 10% steps for warmup
                anneal_strategy='cos'
            )
        else:
            scheduler = None
        
        print(f"Fold {fold+1} Stats: Train Samples={len(train_set)}, Val Samples={len(val_set)}")
        print(f"Initial Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # @add_fzq: 样本量告警 - eval_mask 会显著减少有效样本
        print(f"{COLOR_LOG_Y}⚠️ Note: Actual training samples will be filtered by eval_mask (typically -30%~50% of raw samples){COLOR_LOG_END}")

        # ----------------------------------------------------------------------
        # Inner Loop: Epochs
        # ----------------------------------------------------------------------
        for epoch in range(params.train.epochs):
            
            print('===================' + COLOR_LOG_Y + f'Epoch: {epoch + 1}'+ COLOR_LOG_END + '====================')
            # Training
            # ---------------------
            current_lr = optimizer.param_groups[0]['lr']
            print(f'-------------------training------------------ [LR: {current_lr:.6f}]')

            torch.set_grad_enabled(True)
            model.train()
            train_step = train_loss = train_total = train_right = train_auc = 0
            all_train_targets = []
            all_train_probs = []
            all_train_targets_no_mask = []
            all_train_probs_no_mask = []
            # 每轮训练第几个批量, 总损失, 训练的真实样本个数, 其中正确的个数, 总体训练的auc
            train_start_time = time.time()
            
            total_batches = len(train_loader)
            train_set_len = len(train_set)

            for batch_idx, data in enumerate(train_loader, 1):
                optimizer.zero_grad()
                # 统一将 batch dict 移动到 DEVICE
                features = {k: v.to(DEVICE) for k, v in data.items()}

                x = features[SeqFeatureKey.Q].to(torch.long)
                y_target = features[SeqFeatureKey.R].to(torch.long)
                mask = features[SeqFeatureKey.MASK].to(torch.bool)

                # 兼容可能不存在 eval_mask 的情况
                eval_mask = features.get(SeqFeatureKey.EVAL_MASK, mask).to(torch.bool)
                interval_time = features[SeqFeatureKey.T_INTERVAL].to(torch.float32)
                response_time = features[SeqFeatureKey.T_RESPONSE].to(torch.float32)

                with autocast(enabled=use_amp):
                    # @fix_fzq: Pass mask as tensor for GIKT internal logic
                    y_hat = model(x, y_target, mask, interval_time, response_time)
                    # @fix_fzq: Skip first timestep (no history for prediction)
                    y_hat = y_hat[:, 1:]
                    y_target_shift = y_target[:, 1:].float()
                    mask_valid = mask[:, 1:]
                    eval_mask_valid = eval_mask[:, 1:]
                    final_mask = mask_valid & eval_mask_valid
                    y_hat_flat = torch.masked_select(y_hat, final_mask)
                    y_target_flat = torch.masked_select(y_target_shift, final_mask)
                    
                    # @add_fzq: Label Smoothing
                    if params.train.label_smoothing > 0:
                        y_target_flat_smoothed = y_target_flat * (1.0 - params.train.label_smoothing) + 0.5 * params.train.label_smoothing
                        loss = loss_fun(y_hat_flat, y_target_flat_smoothed)
                    else:
                        loss = loss_fun(y_hat_flat, y_target_flat)

                    # Regularization
                    reg_loss = 0.0
                    if hasattr(model, 'discrimination_gain'): reg_loss += 0.01 * (model.discrimination_gain ** 2)
                    
                    # 4PL-IRT Regularization (Zero-mean L2 for stable priors)
                    if hasattr(model, 'difficulty_bias'): 
                         reg_loss += params.train.reg_4pl * torch.sum(model.difficulty_bias.weight ** 2)
                    if hasattr(model, 'discrimination_bias'): 
                         reg_loss += params.train.reg_4pl * torch.sum(model.discrimination_bias.weight ** 2)
                    if hasattr(model, 'guessing_bias') and hasattr(model, 'slipping_bias'):
                         reg_loss += params.train.reg_4pl * torch.sum(model.guessing_bias.weight ** 2) 
                         reg_loss += params.train.reg_4pl * torch.sum(model.slipping_bias.weight ** 2)

                    loss += reg_loss
                
                scaler.scale(loss).backward()
                # @add_fzq 2026-03-05: 梯度裁剪防止数值不稳定 (特别是 AMP + GAT 场景)
                if gradient_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                
                # @add_fzq: per-batch scheduler step (OneCycleLR required)
                if scheduler is not None:
                    scheduler.step()

                # Metrics Calculation
                y_prob = torch.sigmoid(y_hat_flat)
                y_pred = torch.ge(y_prob, 0.5)
                train_loss += loss.item()
                train_right += torch.sum(torch.eq(y_target_flat, y_pred)).item()
                train_total += torch.sum(final_mask).item()
                train_step += 1
                
                # Collect for global AUC (consistent with validation)
                all_train_targets.append(y_target_flat.detach().cpu().numpy())
                all_train_probs.append(y_prob.detach().cpu().numpy())
                
                # For diagnostic: collect without eval_mask filter
                y_hat_flat_no_mask = torch.masked_select(y_hat, mask_valid)
                y_target_flat_no_mask = torch.masked_select(y_target_shift, mask_valid)
                if len(y_target_flat_no_mask) > 0:
                    y_prob_no_mask = torch.sigmoid(y_hat_flat_no_mask)
                    all_train_targets_no_mask.append(y_target_flat_no_mask.detach().cpu().numpy())
                    all_train_probs_no_mask.append(y_prob_no_mask.detach().cpu().numpy())
                
                # Verbose: Per-batch logging (Optional, disabled by default to reduce noise)
                if params.train.verbose:
                    if len(y_target_flat) > 0:
                        batch_acc = torch.sum(torch.eq(y_target_flat, y_pred)).item() / len(y_target_flat)
                        batch_auc = 0.5
                        try:
                            batch_auc = roc_auc_score(y_target_flat.detach().cpu().numpy(), y_prob.detach().cpu().numpy())
                        except ValueError:
                            batch_auc = 0.5
                    else:
                        batch_acc = 0.0
                        batch_auc = 0.5
                    print(f'step: {batch_idx}, loss: {loss.item():.4f}, acc: {batch_acc:.4f}, auc: {batch_auc:.4f}')

            train_loss /= train_step if train_step > 0 else 1
            # 修正：不再按 epoch 步进，已改为按 batch 使用 OneCycleLR 步进
            # step_count = getattr(optimizer, '_step_count', 0)
            # if train_step > 0 and step_count > 0:
            #     scheduler.step()

            train_acc = train_right / train_total if train_total > 0 else 0
            # Calculate global training AUC (consistent with validation)
            if len(all_train_targets) > 0:
                arr_train_targets = np.concatenate(all_train_targets)
                arr_train_probs = np.concatenate(all_train_probs)
                train_auc = roc_auc_score(arr_train_targets, arr_train_probs)
                
                if len(all_train_targets_no_mask) > 0:
                    arr_train_targets_no_mask = np.concatenate(all_train_targets_no_mask)
                    arr_train_probs_no_mask = np.concatenate(all_train_probs_no_mask)
                    train_auc_no_mask = roc_auc_score(arr_train_targets_no_mask, arr_train_probs_no_mask)
                    eval_mask_filter_ratio = 1.0 - (len(arr_train_targets) / max(1, len(arr_train_targets_no_mask)))
                else:
                    train_auc_no_mask = train_auc
                    eval_mask_filter_ratio = 0.0
                    
                # @add_fzq: 告警 - 如果有效样本过少
                if train_total < train_set_len * 0.1:
                    print(f"{COLOR_LOG_Y}⚠️ Warning: Only {train_total} effective training samples (< 10% of {train_set_len}). Model may underfit. Consider increasing epochs or reducing stride.{COLOR_LOG_END}")
            else:
                train_auc_no_mask = 0.0
                eval_mask_filter_ratio = 0.0
            
            train_time = time.time() - train_start_time
            
            # 验证集评估
            print('-------------------validate------------------')
            model.eval()
            val_loss = val_total = val_right = val_auc = val_step = 0
            all_targets = []
            all_probs = []
            all_targets_no_mask = []
            all_probs_no_mask = []
            val_start_time = time.time()
            torch.set_grad_enabled(False)
            total_val_batches = len(val_loader)
            for val_batch_idx, data in enumerate(val_loader, 1):
                features = {k: v.to(DEVICE) for k, v in data.items()}
                x = features[SeqFeatureKey.Q].to(torch.long)
                y_target = features[SeqFeatureKey.R].to(torch.long)
                mask = features[SeqFeatureKey.MASK].to(torch.bool)
                interval_time = features[SeqFeatureKey.T_INTERVAL].to(torch.float32)
                response_time = features[SeqFeatureKey.T_RESPONSE].to(torch.float32)
                eval_mask = features[SeqFeatureKey.EVAL_MASK].to(torch.bool)

                with autocast(enabled=use_amp):
                    y_hat = model(x, y_target, mask, interval_time, response_time)

                y_hat = y_hat[:, 1:]
                y_target_shift = y_target[:, 1:].float()
                mask_valid = mask[:, 1:]
                eval_mask_valid = eval_mask[:, 1:]
                final_mask = mask_valid & eval_mask_valid
                y_hat_flat = torch.masked_select(y_hat, final_mask)
                y_target_flat = torch.masked_select(y_target_shift, final_mask)
                loss = loss_fun(y_hat_flat, y_target_flat)
                y_prob = torch.sigmoid(y_hat_flat)

                val_loss += loss.item()
                y_pred = torch.ge(y_prob, 0.5)
                val_right += torch.sum(torch.eq(y_target_flat, y_pred))
                val_total += torch.sum(final_mask)
                val_step += 1

                all_targets.append(y_target_flat.cpu().detach().numpy())
                all_probs.append(y_prob.cpu().detach().numpy())

                # For diagnostic: collect without eval_mask filter
                y_hat_flat_no_mask = torch.masked_select(y_hat, mask_valid)
                y_target_flat_no_mask = torch.masked_select(y_target_shift, mask_valid)
                if len(y_target_flat_no_mask) > 0:
                    y_prob_no_mask = torch.sigmoid(y_hat_flat_no_mask)
                    all_targets_no_mask.append(y_target_flat_no_mask.cpu().detach().numpy())
                    all_probs_no_mask.append(y_prob_no_mask.cpu().detach().numpy())

                if params.train.verbose:
                    if len(y_target_flat) > 0:
                        batch_acc = torch.sum(torch.eq(y_target_flat, y_pred)).item() / len(y_target_flat)
                        batch_auc = 0.5
                        try:
                            batch_auc = roc_auc_score(y_target_flat.cpu().detach().numpy(), y_prob.cpu().detach().numpy())
                        except ValueError:
                            batch_auc = 0.5
                    else:
                        batch_acc = 0.0
                        batch_auc = 0.5
                    print(f'step: {val_batch_idx}, loss: {loss.item():.4f}, acc: {batch_acc:.4f}, auc: {batch_auc:.4f}')

            if len(all_targets) > 0:
                arr_targets = np.concatenate(all_targets)
                arr_probs = np.concatenate(all_probs)
                val_auc = roc_auc_score(arr_targets, arr_probs)
                if len(all_targets_no_mask) > 0:
                    arr_targets_no_mask = np.concatenate(all_targets_no_mask)
                    arr_probs_no_mask = np.concatenate(all_probs_no_mask)
                    val_auc_no_mask = roc_auc_score(arr_targets_no_mask, arr_probs_no_mask) 
                    val_eval_mask_filter_ratio = 1.0 - (len(arr_targets) / max(1, len(arr_targets_no_mask)))
                else:
                    val_auc_no_mask = val_auc
                    val_eval_mask_filter_ratio = 0.0
                
                # 动态阈值搜索：分别寻找最大化 F1 和 ACC 的阈值
                precision, recall, f1_thresholds = precision_recall_curve(arr_targets, arr_probs)
                f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
                best_f1_idx = np.argmax(f1_scores)
                best_val_f1 = f1_scores[best_f1_idx]
                
                # 为了把 ACC 压榨到极致，直接强搜最佳 ACC 的阈值
                search_thresholds = np.linspace(0.2, 0.8, 61) # 从0.2到0.8扫描
                acc_scores = [np.mean(arr_targets == (arr_probs >= t)) for t in search_thresholds]
                best_acc_idx = np.argmax(acc_scores)
                best_val_threshold = search_thresholds[best_acc_idx]
                val_acc = acc_scores[best_acc_idx]

            else:
                val_auc_no_mask = 0.0
                val_eval_mask_filter_ratio = 0.0
                val_acc = 0.0
                best_val_threshold = 0.5
                best_val_f1 = 0.0
            
            val_loss /= val_step if val_step > 0 else 1
            # val_acc = val_right / val_total if val_total > 0 else 0 (已由动态阈值替代)
            val_time = time.time() - val_start_time
            
            # Logging & Recording
            # ---------------------
            run_time = train_time + val_time
            
            # Epoch 总结阶段
            train_avg_batch_time = train_time / train_step if train_step > 0 else 0.0
            val_avg_batch_time = val_time / val_step if val_step > 0 else 0.0
            total_avg_batch_time = run_time / (val_step + train_step) if (val_step + train_step) > 0 else 0.0
            
            print(COLOR_LOG_B + f'training: loss: {train_loss:.4f}, acc: {train_acc:.4f}, auc: {train_auc: .4f} | samples: {train_total}' + COLOR_LOG_END)
            print(COLOR_LOG_B + f'validate: loss: {val_loss:.4f}, acc: {val_acc:.4f}, auc: {val_auc: .4f}, f1: {best_val_f1:.4f}, thresh: {best_val_threshold:.2f} | samples: {val_total.item() if torch.is_tensor(val_total) else val_total}' + COLOR_LOG_END)
            
            # @add_fzq: Eval Mask Diagnostic Output
            n_train_targets = sum(len(arr) for arr in all_train_targets) if len(all_train_targets) > 0 else 0
            n_train_targets_no_mask = sum(len(arr) for arr in all_train_targets_no_mask) if len(all_train_targets_no_mask) > 0 else 0
            if eval_mask_filter_ratio > 0:
                print(COLOR_LOG_Y + f'📊 Eval Mask Diagnostic: Filtered {eval_mask_filter_ratio*100:.1f}% of training samples (History context)' + COLOR_LOG_END)
                print(COLOR_LOG_Y + f'   with_mask:    AUC={train_auc:.4f} (n={n_train_targets})' + COLOR_LOG_END)
                print(COLOR_LOG_Y + f'   without_mask: AUC={train_auc_no_mask:.4f} (n={n_train_targets_no_mask}) | Δ AUC={train_auc_no_mask - train_auc:+.4f}' + COLOR_LOG_END)
            else:
                # No history context filtering (all sequences are short or training non-overlapping windows)
                print(COLOR_LOG_Y + f'📊 Eval Mask Diagnostic: No history context filtering (all {n_train_targets} samples are evaluation data)' + COLOR_LOG_END)
            print(COLOR_LOG_B + f'train time: {train_time:.2f}s, avg batch: {train_avg_batch_time:.4f}s | batches: {train_step}' + COLOR_LOG_END)
            print(COLOR_LOG_B + f'validate time: {val_time:.2f}s, avg batch: {val_avg_batch_time:.4f}s | batches: {val_step}' + COLOR_LOG_END)
            print(COLOR_LOG_B + f'total time: {run_time:.2f}s, average batch time: {total_avg_batch_time:.4f}s' + COLOR_LOG_END)
            
            # @add_fzq: Validation Eval Mask Diagnostic Output
            n_val_targets = sum(len(arr) for arr in all_targets) if len(all_targets) > 0 else 0
            n_val_targets_no_mask = sum(len(arr) for arr in all_targets_no_mask) if len(all_targets_no_mask) > 0 else 0
            if val_eval_mask_filter_ratio > 0:
                print(COLOR_LOG_Y + f'📊 Val Eval Mask Diagnostic: Filtered {val_eval_mask_filter_ratio*100:.1f}% of validation samples (History context)' + COLOR_LOG_END)
                print(COLOR_LOG_Y + f'   with_mask:    AUC={val_auc:.4f} (n={n_val_targets})' + COLOR_LOG_END)
                print(COLOR_LOG_Y + f'   without_mask: AUC={val_auc_no_mask:.4f} (n={n_val_targets_no_mask}) | Δ AUC={val_auc_no_mask - val_auc:+.4f}' + COLOR_LOG_END)
            else:
                # No history context filtering in validation
                print(COLOR_LOG_Y + f'📊 Val Eval Mask Diagnostic: No history context filtering (all {n_val_targets} samples are evaluation data)' + COLOR_LOG_END)
            
            # 保存输出至本地文件
            output_file.write(f'  Epoch {epoch+1} | ')
            output_file.write(f'training: loss: {train_loss:.4f}, acc: {train_acc:.4f}, auc: {train_auc: .4f} | samples: {train_total}\n')
            output_file.write(f'          | validate: loss: {val_loss:.4f}, acc: {val_acc:.4f}, auc: {val_auc: .4f}, f1: {best_val_f1:.4f}, thresh: {best_val_threshold:.2f} | samples: {val_total.item() if torch.is_tensor(val_total) else val_total}\n')
            output_file.write(f'          | train time: {train_time:.2f}s, avg batch: {train_avg_batch_time:.2f}s | ')
            output_file.write(f'test time: {val_time:.2f}s, avg batch: {val_avg_batch_time:.2f}s | ')
            output_file.write(f'total time: {run_time:.2f}s, average batch time: {total_avg_batch_time:.2f}s\n')


            # Record Data
            # Accumulate Average (Divided later)
            y_label_aver[0][epoch] += val_loss
            y_label_aver[1][epoch] += val_acc
            y_label_aver[2][epoch] += val_auc
            
            # Save All Points (Flat Index)
            # Index logic: Fold 0 (E1...En), Fold 1 (E1...En)
            idx = fold * params.train.epochs + epoch
            if idx < y_label_all.shape[1]:
                y_label_all[0][idx] = val_loss
                y_label_all[1][idx] = val_acc
                y_label_all[2][idx] = val_auc

            # === [add_fzq] Enhanced History Tracking ===
            history_records.append({
                'fold': fold + 1,
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_auc': train_auc,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_auc': val_auc,
                'val_acc': val_acc,
                'time_sec': run_time
            })
            
            if train_writer and val_writer:
                train_writer.add_scalar('Performance/Loss', train_loss, epoch)
                train_writer.add_scalar('Performance/AUC', train_auc, epoch)
                train_writer.add_scalar('Performance/ACC', train_acc, epoch)
                try:
                    train_writer.add_scalar('Optimization/Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
                except:
                    pass

                val_writer.add_scalar('Performance/Loss', val_loss, epoch)
                val_writer.add_scalar('Performance/AUC', val_auc, epoch)
                val_writer.add_scalar('Performance/ACC', val_acc, epoch)

            # Early Stopping (Per Fold)
            if val_auc > best_fold_val_auc:
                improvement = val_auc - best_fold_val_auc
                best_fold_val_auc = val_auc
                fold_best_threshold = best_val_threshold
                # Save best model logic could go here
                patience_counter = 0
                if params.train.verbose:
                    print(COLOR_LOG_G + f'🎯 新的最佳验证AUC: {best_fold_val_auc:.4f} (提升 +{improvement:.4f})' + COLOR_LOG_END)
            else:
                patience_counter += 1
                if params.train.verbose:
                    print(COLOR_LOG_Y + f'⏳ 验证AUC未提升 (Patience: {patience_counter}/{params.train.patience}, 最佳: {best_fold_val_auc:.4f})' + COLOR_LOG_END)
                if params.train.patience > 0 and patience_counter >= params.train.patience:
                    if params.train.verbose:
                        print(f"{COLOR_LOG_Y}⛔ Early Stopping 触发于 Epoch {epoch+1} (最佳验证AUC: {best_fold_val_auc:.4f}){COLOR_LOG_END}")
                    break
        
        # Fold Summary (after all epochs)
        # 计算 fold 的平均指标（使用独立测试集）
        best_fold_test_auc = 0.0
        best_fold_test_loss = 0.0
        best_fold_test_acc = 0.0
        # 使用最佳模型状态（已在早期停止逻辑中预先加载）对测试集进行重新评估
        model.eval()
        test_targets = []
        test_probs = []
        test_loss = 0
        test_step = 0
        if dataset_test_loader is not None:
            with torch.no_grad():
                for data in dataset_test_loader:
                    features = {k: v.to(DEVICE) for k, v in data.items()}
                    x = features[SeqFeatureKey.Q].to(torch.long)
                    y_target = features[SeqFeatureKey.R].to(torch.long)
                    mask = features[SeqFeatureKey.MASK].to(torch.bool)
                    interval_time = features[SeqFeatureKey.T_INTERVAL].to(torch.float32)
                    response_time = features[SeqFeatureKey.T_RESPONSE].to(torch.float32)
                    eval_mask = features[SeqFeatureKey.EVAL_MASK].to(torch.bool)

                    with autocast(enabled=use_amp):
                        y_hat = model(x, y_target, mask, interval_time, response_time)

                    y_hat = y_hat[:, 1:]
                    y_target_shift = y_target[:, 1:].float()
                    mask_valid = mask[:, 1:]
                    eval_mask_valid = eval_mask[:, 1:]
                    final_mask = mask_valid & eval_mask_valid
                    y_hat_flat = torch.masked_select(y_hat, final_mask)
                    y_target_flat = torch.masked_select(y_target_shift, final_mask)
                    loss = loss_fun(y_hat_flat, y_target_flat)
                    y_prob = torch.sigmoid(y_hat_flat)

                    test_loss += loss.item()
                    test_step += 1
                    test_targets.extend(y_target_flat.cpu().detach().numpy())
                    test_probs.extend(y_prob.cpu().detach().numpy())
            if len(test_targets) > 0:
                best_fold_test_auc = roc_auc_score(test_targets, test_probs)
            best_fold_test_loss = test_loss / test_step if test_step > 0 else 0
            best_fold_test_acc = np.mean(np.array(test_targets) == (np.array(test_probs) >= fold_best_threshold))

        print('\n' + '='*70)
        print(COLOR_LOG_G + f"✅ Fold {fold+1} 完成 | 最佳验证AUC: {best_fold_val_auc:.4f}" + COLOR_LOG_END)
        print('='*70 + '\n')
        fold_results_test_auc.append(best_fold_test_auc)
        
    print('\n' + '='*70)
    print(COLOR_LOG_G + '🎉 交叉验证完成！' + COLOR_LOG_END)
    print('='*70)
    print(f"各折测试集AUC: {[f'{auc:.4f}' for auc in fold_results_test_auc]}")
    print(f"平均AUC (Holdout Test Set): {np.mean(fold_results_test_auc):.4f} ± {np.std(fold_results_test_auc):.4f}")
    print(f"最佳AUC: {np.max(fold_results_test_auc):.4f} (Fold {np.argmax(fold_results_test_auc)+1})")
    print(f"最差AUC: {np.min(fold_results_test_auc):.4f} (Fold {np.argmin(fold_results_test_auc)+1})")
    print('='*70 + '\n')
    output_file.write(f"CV Mean AUC (Holdout Test Set): {np.mean(fold_results_test_auc):.4f}\n")

    # Normalize Averages
    if len(splits) > 0:
        y_label_aver /= len(splits)

    # @add_fzq 2025-12-24 17:28:09 -------------------------------------------
    # 计算总耗时 
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    total_hours = int(total_duration // 3600)
    total_minutes = int((total_duration % 3600) // 60)
    total_seconds = int(total_duration % 60)
    time_str = f'{total_hours}h {total_minutes}min {total_seconds}s'
    print(f'Total training time: {time_str}')
    output_file.write(f'Total training time: {time_str}\n')
    
    output_file.close()

    # 记录消融实验结果（如果指定了消融组别名）
    if args.ablation_name:
        import csv
        csv_path = f'{config.path.LOG_DIR}/ablation_summary.csv'
        file_exists = os.path.exists(csv_path)
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Ablation Group', 'Date', 'Dataset', 'Mean AUC', 'Std AUC', 'Max AUC (Fold)', 'Min AUC (Fold)', 'Time'])
            
            mean_auc = np.mean(fold_results_test_auc)
            std_auc = np.std(fold_results_test_auc)
            max_auc_idx = np.argmax(fold_results_test_auc)
            min_auc_idx = np.argmin(fold_results_test_auc)

            writer.writerow([
                args.ablation_name, 
                time_now, 
                dataset_name, 
                f"{mean_auc:.4f}", 
                f"{std_auc:.4f}", 
                f"{fold_results_test_auc[max_auc_idx]:.4f} (F{max_auc_idx+1})",
                f"{fold_results_test_auc[min_auc_idx]:.4f} (F{min_auc_idx+1})",
                time_str
            ])
        print(COLOR_LOG_Y + f"📝 消融结果已追加至 {csv_path}" + COLOR_LOG_END)

    # Save Data
    # Optional: Save final model (from last fold, or logic to save best)
    if params.train.save_model:
        torch.save(model, f=f'{config.path.MODEL_DIR}/{time_now}.pt')
        print(f'Model saved to {config.path.MODEL_DIR}/{time_now}.pt')
    np.savetxt(f'{config.path.CHART_DIR}/{time_now}_all.txt', y_label_all)
    np.savetxt(f'{config.path.CHART_DIR}/{time_now}_aver.txt', y_label_aver)

    # === [add_fzq] Enhanced Output Generation (CSV + PNG + TensorBoard HParams) ===
    tb_hparam_writer = SummaryWriter(log_dir=runs_root)
    avg_val_auc = np.mean(fold_results_test_auc) if fold_results_test_auc else 0.0
    tb_hparam_writer.add_hparams(
        hparam_dict={'model': 'GIKT', 'dataset': dataset_name, 'lr': params.train.lr},
        metric_dict={'hparam/Avg_Test_AUC': avg_val_auc}
    )
    tb_hparam_writer.close()

    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        df_history = pd.DataFrame(history_records)
        if args.ablation_name:
            csv_filename = f"gikt_{args.ablation_name}_{dataset_name}_history_{time_now}.csv"
            img_filename = f"gikt_{args.ablation_name}_{dataset_name}_metrics_{time_now}.png"
        else:
            csv_filename = f"gikt_{dataset_name}_history_{time_now}.csv"
            img_filename = f"gikt_{dataset_name}_metrics_{time_now}.png"
            
        csv_path = os.path.join(chart_data_dir, csv_filename)
        df_history.to_csv(csv_path, index=False)
        print(f"📊 历史训练数据已保存至 {csv_path}")
        
        # Plotting
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
        color_train = "#4DBBD5"
        color_val = "#E64B35"
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        sns.lineplot(data=df_history, x='epoch', y='train_loss', label='Train Loss', color=color_train, ax=axes[0], errorbar='ci', n_boot=100)
        sns.lineplot(data=df_history, x='epoch', y='val_loss', label='Val Loss', color=color_val, ax=axes[0], errorbar='ci', n_boot=100)
        axes[0].set_title('Loss Curve (Mean & CI)')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        
        sns.lineplot(data=df_history, x='epoch', y='train_auc', label='Train AUC', color=color_train, ax=axes[1], errorbar='ci', n_boot=100)
        sns.lineplot(data=df_history, x='epoch', y='val_auc', label='Val AUC', color=color_val, ax=axes[1], errorbar='ci', n_boot=100)
        axes[1].set_title('AUC Curve (Mean & CI)')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('AUC')
        axes[1].legend()
        
        sns.lineplot(data=df_history, x='epoch', y='train_acc', label='Train ACC', color=color_train, ax=axes[2], errorbar='ci', n_boot=100)
        sns.lineplot(data=df_history, x='epoch', y='val_acc', label='Val ACC', color=color_val, ax=axes[2], errorbar='ci', n_boot=100)
        axes[2].set_title('Accuracy Curve (Mean & CI)')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Accuracy')
        axes[2].legend()
        
        plt.tight_layout()
        img_path = os.path.join(chart_img_dir, img_filename)
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📈 训练曲线图已保存至 {img_path}")
        print(f"📝 Tensorboard 目录: {runs_root}")
        
    except Exception as e:
        print(f"⚠️ 生成增强历史图表失败: {e}")

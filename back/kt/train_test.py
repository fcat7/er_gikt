"""
训练并测试模型
使用五折交叉验证法 (Standard K-Fold)
"""
import os
import time
from datetime import datetime
import numpy as np
from scipy import sparse
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, ShuffleSplit
from torch.utils.data import DataLoader, Subset
from dataset import UserDataset
from config import Config, DEVICE, COLOR_LOG_B, COLOR_LOG_Y, COLOR_LOG_G, COLOR_LOG_END
from params import HyperParameters
from util.utils import gen_gikt_graph, build_adj_list
import argparse
from gikt import GIKT

# @add_fzq: AMP Support
import torch
from torch.cuda.amp import autocast, GradScaler

# 自动检测 AMP 可用性
use_amp = torch.cuda.is_available() 
scaler = GradScaler(enabled=use_amp)
if use_amp:
    print(f"🚀 AMP Enabled on {torch.cuda.get_device_name(0)}")
else:
    print(f"⚠️ AMP Disabled")

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
    # 加载配置
    dataset_name = params.train.dataset_name
    config = Config(dataset_name=dataset_name)

    output_path = f'{config.path.LOG_DIR}/{time_now}.log'
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
    # @update_fzq: 使用细粒度的数据增强配置
    dataset_full_augment = UserDataset(
        config, 
        augment=params.train.enable_data_augmentation,
        prob_mask=params.train.aug_mask_prob,
        mode='train' 
    )
    dataset_full_clean = UserDataset(config, augment=False, mode='train')

    # 加载独立的 Holdout 测试集 (不重叠的用户)
    try:
        dataset_holdout = UserDataset(config, augment=False, mode='test')
        print(f"📚 Loaded Holdout Test Set: {len(dataset_holdout)} samples.")
        dataset_holdout_loader = DataLoader(
            dataset_holdout, 
            batch_size=batch_size, 
            num_workers=params.common.num_workers,
            pin_memory=True
        )
    except Exception as e:
        dataset_holdout = None
        dataset_holdout_loader = None
        print(f"⚠️ Warning: Holdout Test Set load failed ({e}). Skipping independent testing.")
    
    data_len = len(dataset_full_clean)
    output_file.write(f'Train/Val Pool size: {data_len}\n')
    
    # 记录总开始时间
    total_start_time = time.time()

    # TF Alignment is now always enabled (Logits output, BCEWithLogitsLoss)
    loss_fun = torch.nn.BCEWithLogitsLoss().to(DEVICE) 

    # ==========================================================================================
    # K-Fold Strategy Logic
    # ==========================================================================================
    groups = dataset_full_clean.groups
    if groups is not None:
        print(f"🔒 Detecting Windowed Dataset with {len(np.unique(groups))} unique users. Using Group-based splitting to prevent leakage.")
        from sklearn.model_selection import GroupKFold, GroupShuffleSplit
        if params.train.k_fold == 1:
            k_fold = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        else:
            k_fold = GroupKFold(n_splits=params.train.k_fold)
        splits = list(k_fold.split(dataset_full_clean, groups=groups))
    else:
        print("🔓 Using Standard Shuffled splitting.")
        if params.train.k_fold == 1:
            k_fold = ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        else:
            k_fold = KFold(n_splits=params.train.k_fold, shuffle=True, random_state=42)
        splits = list(k_fold.split(dataset_full_clean))

    fold_results_val_auc = []
    
    # 初始化记录数组 (Metric x Fold*Epoch)
    # y_label_aver: [Metric, Epoch] (Averaged across folds)
    y_label_aver = np.zeros([3, params.train.epochs]) 
    # y_label_all: [Metric, Fold*Epoch] (Sequential)
    y_label_all = np.zeros([3, params.train.epochs * len(splits)]) 

    # --------------------------------------------------------------------------
    # Outer Loop: Folds
    # --------------------------------------------------------------------------
    for fold, (train_indices, test_indices) in enumerate(splits):
        print('===================' + COLOR_LOG_Y + f'fold: {fold + 1} / {len(splits)}'+ COLOR_LOG_END + '====================')
        output_file.write('===================' + f'fold: {fold + 1} / {len(splits)}' + '====================\n')
        
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
            use_cognitive_model=params.model.use_cognitive_model,
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
            slipping_prob_init=params.model.slipping_prob_init
        ).to(DEVICE)
        
        optimizer = torch.optim.Adam(params=model.parameters(), lr=params.train.lr, weight_decay=params.train.weight_decay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, params.train.lr_gamma)
        
        best_fold_val_auc = 0.0
        patience_counter = 0

        # Data Loaders
        train_set = Subset(dataset_full_augment, train_indices)
        test_set = Subset(dataset_full_clean, test_indices)
        
        loader_kwargs = {
            'batch_size': batch_size,
            'num_workers': params.common.num_workers,
            'pin_memory': True
        }
        if params.common.num_workers > 0 and DEVICE.type != 'cpu':
            loader_kwargs['prefetch_factor'] = params.train.prefetch_factor
            
        train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
        test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)
        
        print(f"Fold {fold+1} Stats: Train Samples={len(train_set)}, Val Samples={len(test_set)}")
        print(f"Initial Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # ----------------------------------------------------------------------
        # Inner Loop: Epochs
        # ----------------------------------------------------------------------
        for epoch in range(params.train.epochs):
            
            print('===================' + COLOR_LOG_Y + f'Epoch: {epoch + 1}'+ COLOR_LOG_END + '====================')
            # Training
            # ---------------------
            print('-------------------training------------------')
            
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
                data_gpu = data.to(DEVICE)
                x = data_gpu[:, :, 0].to(torch.long)
                y_target = data_gpu[:, :, 1].to(torch.long)
                mask = data_gpu[:, :, 2].to(torch.bool)
                interval_time = data_gpu[:, :, 3].to(torch.float32)
                response_time = data_gpu[:, :, 4].to(torch.float32)
                eval_mask = data_gpu[:, :, 5].to(torch.bool)

                with autocast(enabled=use_amp):
                    y_hat = model(x, y_target, mask, interval_time, response_time)
                    # @fix_fzq: Skip first timestep (no history for prediction)
                    y_hat = y_hat[:, 1:]
                    y_target_shift = y_target[:, 1:].float()
                    mask_valid = mask[:, 1:]
                    eval_mask_valid = eval_mask[:, 1:]
                    final_mask = mask_valid & eval_mask_valid
                    y_hat_flat = torch.masked_select(y_hat, final_mask)
                    y_target_flat = torch.masked_select(y_target_shift, final_mask)
                    loss = loss_fun(y_hat_flat, y_target_flat)
                    
                    # Regularization
                    reg_loss = 0.0
                    if hasattr(model, 'discrimination_gain'): reg_loss += 0.01 * (model.discrimination_gain ** 2)
                    if hasattr(model, 'discrimination_bias'): reg_loss += params.train.reg_4pl * torch.sum(model.discrimination_bias.weight ** 2)
                    if hasattr(model, 'guessing_bias') and hasattr(model, 'slipping_bias'):
                         reg_loss += params.train.reg_4pl * torch.sum(torch.relu(model.guessing_bias.weight + 2.0)**2) 
                         reg_loss += params.train.reg_4pl * torch.sum(torch.relu(model.slipping_bias.weight + 3.0)**2)
                    loss += reg_loss
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                # Metrics Calculation
                y_prob = torch.sigmoid(y_hat_flat)
                y_pred = torch.ge(y_prob, 0.5)
                train_loss += loss.item()
                train_right += torch.sum(torch.eq(y_target_flat, y_pred)).item()
                train_total += torch.sum(final_mask).item()
                train_step += 1
                
                # Collect for global AUC (consistent with validation)
                all_train_targets.extend(y_target_flat.detach().cpu().numpy())
                all_train_probs.extend(y_prob.detach().cpu().numpy())
                
                # For diagnostic: collect without eval_mask filter
                y_hat_flat_no_mask = torch.masked_select(y_hat, mask_valid)
                y_target_flat_no_mask = torch.masked_select(y_target_shift, mask_valid)
                if len(y_target_flat_no_mask) > 0:
                    y_prob_no_mask = torch.sigmoid(y_hat_flat_no_mask)
                    all_train_targets_no_mask.extend(y_target_flat_no_mask.detach().cpu().numpy())
                    all_train_probs_no_mask.extend(y_prob_no_mask.detach().cpu().numpy())
                
                # Verbose: Per-batch logging (Optional, disabled by default to reduce noise)
                if params.train.verbose:
                    batch_acc = torch.sum(torch.eq(y_target_flat, y_pred)).item() / len(y_target_flat)
                    batch_auc = 0.5  # Default value
                    try:
                        batch_auc = roc_auc_score(y_target_flat.detach().cpu().numpy(), y_prob.detach().cpu().numpy())
                    except ValueError:
                        pass  # Only one class present in batch
                    print(f'step: {batch_idx}, loss: {loss.item():.4f}, acc: {batch_acc:.4f}, auc: {batch_auc:.4f}')

            train_loss /= train_step if train_step > 0 else 1
            # 修正：在 epoch 结束后调用，必须确保 optimizer 已执行过 step
            # 使用 getattr 安全访问 _step_count (PyTorch 内部属性)
            step_count = getattr(optimizer, '_step_count', 0) 
            if train_step > 0 and step_count > 0:
                scheduler.step()
                
            train_acc = train_right / train_total if train_total > 0 else 0
            # Calculate global training AUC (consistent with validation)
            if len(all_train_targets) > 0:
                train_auc = roc_auc_score(all_train_targets, all_train_probs)
                train_auc_no_mask = roc_auc_score(all_train_targets_no_mask, all_train_probs_no_mask) if len(all_train_targets_no_mask) > 0 else train_auc
                eval_mask_filter_ratio = 1.0 - (len(all_train_targets) / max(1, len(all_train_targets_no_mask)))
            else:
                train_auc_no_mask = 0.0
                eval_mask_filter_ratio = 0.0
            
            train_time = time.time() - train_start_time
            
            # Validation
            # ---------------------
            print('-------------------validate------------------')
            model.eval()
            val_loss = val_total = val_right = val_auc = val_step = 0
            all_targets = []
            all_probs = []
            all_targets_no_mask = []
            all_probs_no_mask = []
            val_start_time = time.time()
            
            torch.set_grad_enabled(False)
            total_val_batches = len(test_loader)
            for val_batch_idx, data in enumerate(test_loader, 1):
                data_gpu = data.to(DEVICE)
                x = data_gpu[:, :, 0].to(torch.long)
                y_target = data_gpu[:, :, 1].to(torch.long)
                mask = data_gpu[:, :, 2].to(torch.bool)
                interval_time = data_gpu[:, :, 3].to(torch.float32)
                response_time = data_gpu[:, :, 4].to(torch.float32)
                eval_mask = data_gpu[:, :, 5].to(torch.bool)

                with autocast(enabled=use_amp):
                    y_hat = model(x, y_target, mask, interval_time, response_time)
                
                # @fix_fzq: Skip first timestep (no history for prediction)
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
                
                all_targets.extend(y_target_flat.cpu().detach().numpy())
                all_probs.extend(y_prob.cpu().detach().numpy())
                
                # For diagnostic: collect without eval_mask filter
                y_hat_flat_no_mask = torch.masked_select(y_hat, mask_valid)
                y_target_flat_no_mask = torch.masked_select(y_target_shift, mask_valid)
                if len(y_target_flat_no_mask) > 0:
                    y_prob_no_mask = torch.sigmoid(y_hat_flat_no_mask)
                    all_targets_no_mask.extend(y_target_flat_no_mask.cpu().detach().numpy())
                    all_probs_no_mask.extend(y_prob_no_mask.cpu().detach().numpy())
                
                if params.train.verbose:
                    batch_acc = torch.sum(torch.eq(y_target_flat, y_pred)).item() / len(y_target_flat)
                    batch_auc = 0.5  # Default value
                    try:
                        batch_auc = roc_auc_score(y_target_flat.cpu().detach().numpy(), y_prob.cpu().detach().numpy())
                    except ValueError:
                        pass  # Only one class present in batch
                    print(f'step: {val_batch_idx}, loss: {loss.item():.4f}, acc: {batch_acc:.4f}, auc: {batch_auc:.4f}')

            if len(all_targets) > 0:
                val_auc = roc_auc_score(all_targets, all_probs)
                val_auc_no_mask = roc_auc_score(all_targets_no_mask, all_probs_no_mask) if len(all_targets_no_mask) > 0 else val_auc
                val_eval_mask_filter_ratio = 1.0 - (len(all_targets) / max(1, len(all_targets_no_mask)))
            else:
                val_auc_no_mask = 0.0
                val_eval_mask_filter_ratio = 0.0
            val_loss /= val_step if val_step > 0 else 1
            val_acc = val_right / val_total if val_total > 0 else 0
            val_time = time.time() - val_start_time
            
            # Logging & Recording
            # ---------------------
            run_time = train_time + val_time
            
            # Epoch 总结阶段
            train_avg_batch_time = train_time / train_step if train_step > 0 else 0.0
            val_avg_batch_time = val_time / val_step if val_step > 0 else 0.0
            total_avg_batch_time = run_time / (val_step + train_step) if (val_step + train_step) > 0 else 0.0
            
            print(COLOR_LOG_B + f'training: loss: {train_loss:.4f}, acc: {train_acc:.4f}, auc: {train_auc: .4f} | samples: {train_total}' + COLOR_LOG_END)
            print(COLOR_LOG_B + f'validate: loss: {val_loss:.4f}, acc: {val_acc:.4f}, auc: {val_auc: .4f} | samples: {val_total.item() if torch.is_tensor(val_total) else val_total}' + COLOR_LOG_END)
            
            # @add_fzq: Eval Mask Diagnostic Output
            if eval_mask_filter_ratio > 0:
                print(COLOR_LOG_Y + f'📊 Eval Mask Diagnostic: Filtered {eval_mask_filter_ratio*100:.1f}% of training samples (History context)' + COLOR_LOG_END)
                print(COLOR_LOG_Y + f'   with_mask:    AUC={train_auc:.4f} (n={len(all_train_targets)})' + COLOR_LOG_END)
                print(COLOR_LOG_Y + f'   without_mask: AUC={train_auc_no_mask:.4f} (n={len(all_train_targets_no_mask)}) | Δ AUC={train_auc_no_mask - train_auc:+.4f}' + COLOR_LOG_END)
            else:
                # No history context filtering (all sequences are short or training non-overlapping windows)
                print(COLOR_LOG_Y + f'📊 Eval Mask Diagnostic: No history context filtering (all {len(all_train_targets)} samples are evaluation data)' + COLOR_LOG_END)
            print(COLOR_LOG_B + f'train time: {train_time:.2f}s, avg batch: {train_avg_batch_time:.4f}s | batches: {train_step}' + COLOR_LOG_END)
            print(COLOR_LOG_B + f'validate time: {val_time:.2f}s, avg batch: {val_avg_batch_time:.4f}s | batches: {val_step}' + COLOR_LOG_END)
            print(COLOR_LOG_B + f'total time: {run_time:.2f}s, average batch time: {total_avg_batch_time:.4f}s' + COLOR_LOG_END)
            
            # @add_fzq: Validation Eval Mask Diagnostic Output
            if val_eval_mask_filter_ratio > 0:
                print(COLOR_LOG_Y + f'📊 Val Eval Mask Diagnostic: Filtered {val_eval_mask_filter_ratio*100:.1f}% of validation samples (History context)' + COLOR_LOG_END)
                print(COLOR_LOG_Y + f'   with_mask:    AUC={val_auc:.4f} (n={len(all_targets)})' + COLOR_LOG_END)
                print(COLOR_LOG_Y + f'   without_mask: AUC={val_auc_no_mask:.4f} (n={len(all_targets_no_mask)}) | Δ AUC={val_auc_no_mask - val_auc:+.4f}' + COLOR_LOG_END)
            else:
                # No history context filtering in validation
                print(COLOR_LOG_Y + f'📊 Val Eval Mask Diagnostic: No history context filtering (all {len(all_targets)} samples are evaluation data)' + COLOR_LOG_END)
            
            # 保存输出至本地文件
            output_file.write(f'  Epoch {epoch+1} | ')
            output_file.write(f'training: loss: {train_loss:.4f}, acc: {train_acc:.4f}, auc: {train_auc: .4f} | samples: {train_total}\n')
            output_file.write(f'          | validate: loss: {val_loss:.4f}, acc: {val_acc:.4f}, auc: {val_auc: .4f} | samples: {val_total.item() if torch.is_tensor(val_total) else val_total}\n')
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

            # Early Stopping (Per Fold)
            if val_auc > best_fold_val_auc:
                improvement = val_auc - best_fold_val_auc
                best_fold_val_auc = val_auc
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
        # 计算 fold 的平均指标（使用 holdout 测试集）
        best_fold_train_loss = 0
        best_fold_train_acc = 0
        best_fold_train_auc = 0
        best_fold_val_loss = 0
        best_fold_val_acc = 0
        best_fold_test_auc = 0.0
        
        # 使用最佳模型状态（已在早期停止逻辑中预先加载）对测试集进行重新评估
        model.eval()
        test_targets = []
        test_probs = []
        test_loss = 0
        test_step = 0
        with torch.no_grad():
            for data in test_loader:
                data_gpu = data.to(DEVICE)
                x = data_gpu[:, :, 0].to(torch.long)
                y_target = data_gpu[:, :, 1].to(torch.long)
                mask = data_gpu[:, :, 2].to(torch.bool)
                interval_time = data_gpu[:, :, 3].to(torch.float32)
                response_time = data_gpu[:, :, 4].to(torch.float32)
                eval_mask = data_gpu[:, :, 5].to(torch.bool)

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
        best_fold_test_acc = np.mean(np.array(test_targets) == (np.array(test_probs) >= 0.5))
        
        
        print('\n' + '='*70)
        print(COLOR_LOG_G + f"✅ Fold {fold+1} 完成 | 最佳验证AUC: {best_fold_val_auc:.4f}" + COLOR_LOG_END)
        print('='*70 + '\n')
        fold_results_val_auc.append(best_fold_test_auc)
        
    print('\n' + '='*70)
    print(COLOR_LOG_G + '🎉 交叉验证完成！' + COLOR_LOG_END)
    print('='*70)
    print(f"各折AUC: {[f'{auc:.4f}' for auc in fold_results_val_auc]}")
    print(f"平均AUC: {np.mean(fold_results_val_auc):.4f} ± {np.std(fold_results_val_auc):.4f}")
    print(f"最佳AUC: {np.max(fold_results_val_auc):.4f} (Fold {np.argmax(fold_results_val_auc)+1})")
    print(f"最差AUC: {np.min(fold_results_val_auc):.4f} (Fold {np.argmin(fold_results_val_auc)+1})")
    print('='*70 + '\n')
    output_file.write(f"CV Mean AUC: {np.mean(fold_results_val_auc):.4f}\n")

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

    # Save Data
    # Optional: Save final model (from last fold, or logic to save best)
    # torch.save(model, f=f'{config.path.MODEL_DIR}/{time_now}.pt')
    np.savetxt(f'{config.path.CHART_DIR}/{time_now}_all.txt', y_label_all)
    np.savetxt(f'{config.path.CHART_DIR}/{time_now}_aver.txt', y_label_aver)

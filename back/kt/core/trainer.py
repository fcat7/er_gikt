import time
import os
import logging
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
from sklearn.model_selection import KFold, GroupKFold, ShuffleSplit, GroupShuffleSplit
import numpy as np

from dataset import SeqFeatureKey

class BaseTrainer:
    """
    通用的模型训练器，封装了 5-Fold CV、早停机制和评估逻辑
    """
    def __init__(self, model_factory_func, num_question, num_skill, device, config, kwargs):
        self.model_factory_func = model_factory_func
        self.num_question = num_question
        self.num_skill = num_skill
        self.device = device
        self.config = config
        self.kwargs = kwargs # 包含超参数
        
        self.model_name = kwargs.get('model_name', 'Unknown')
        self.dataset_name = kwargs.get('dataset_name', 'Unknown')
        self.task_name = kwargs.get('task_name', 'train_baseline')
        self.epochs = kwargs.get('epochs', 50)
        self.patience = kwargs.get('patience', 5)
        self.k_fold = kwargs.get('k_fold', 5)
        self.batch_size = kwargs.get('batch_size', 64)
        self.learning_rate = kwargs.get('learning_rate', 1e-3)
        self.reg_4pl = kwargs.get('reg_4pl', 1e-5)
        self.gradient_clip_norm = kwargs.get('gradient_clip_norm', 1.0)
        self.verbose_batch = kwargs.get('verbose_batch', False)  # 是否打印每个 batch 的耗时
        
        self.criterion = nn.BCEWithLogitsLoss()
        
        self._setup_logger()
        self.logger.info(f"====== BaseTrainer Initialized ======")
        self.logger.info(f"Task: {self.task_name} | Model: {self.model_name}, Dataset: {self.dataset_name}")
        self.logger.info(f"Configuration: ")
        self.logger.info(f"  Num Question: {self.num_question}")
        self.logger.info(f"  Num Skill: {self.num_skill}")
        self.logger.info(f"  Epochs: {self.epochs}")
        self.logger.info(f"  Batch Size: {self.batch_size}")
        self.logger.info(f"  Learning Rate: {self.learning_rate}")
        self.logger.info(f"  Patience: {self.patience}")
        self.logger.info(f"  K-Fold: {self.k_fold}")
        self.logger.info(f"  Optimization: AMP={self.kwargs.get('amp_enabled', True)}")
        self.logger.info(f"=====================================")

    def _setup_logger(self):
        self.logger = logging.getLogger("BaseTrainer")
        self.logger.setLevel(logging.INFO)
        
        # 防止重复添加 handler
        if not self.logger.handlers:
            formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', '%Y-%m-%d %H:%M:%S')
            
            # 控制台输出
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
            
            # 文件输出
            try:
                log_dir = getattr(self.config.path, 'LOG_DIR', './logs')
            except AttributeError:
                log_dir = './logs'
            
            os.makedirs(log_dir, exist_ok=True)
            time_now = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 生成日志文件名称
            log_file = os.path.join(log_dir, f"trainer_{time_now}_{self.model_name}_{self.dataset_name}.log")
            
            fh = logging.FileHandler(log_file, encoding='utf-8')
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
            
            self.logger.info(f"Logger initialized. File output: {log_file}")

    def _train_epoch(self, model, dataloader, optimizer, scaler, use_amp, epoch_idx=None, tb_writer=None):
        model.train()
        total_loss = 0
        total_samples = 0
        all_preds = []
        all_targets = []
        
        for batch_idx, batch in enumerate(dataloader):
            batch_start_time = time.time()
            features = {k: v.to(self.device) for k, v in batch.items()}
            question = features[SeqFeatureKey.Q].to(torch.long)
            skill = features.get(SeqFeatureKey.C, torch.zeros_like(question)).to(torch.long)
            response = features[SeqFeatureKey.R].to(torch.long)
            mask = features[SeqFeatureKey.MASK].to(torch.bool)
            eval_mask = features.get(SeqFeatureKey.EVAL_MASK, mask).to(torch.bool)
            interval = features[SeqFeatureKey.T_INTERVAL].to(torch.float32)
            r_time = features[SeqFeatureKey.T_RESPONSE].to(torch.float32)
            
            interval = torch.nan_to_num(interval, nan=0.0)
            r_time = torch.nan_to_num(r_time, nan=0.0)

            optimizer.zero_grad()
            
            model_name = getattr(model, 'model_name', '').lower()
            cognitive_mode = getattr(model, 'cognitive_mode', None)
            
            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                if model_name == 'dkt':
                    y_hat = model(question, response, mask, skill=skill)
                    eps = 1e-6
                    y_hat = torch.clamp(y_hat, eps, 1-eps)
                    preds = torch.log(y_hat[:, :-1] / (1 - y_hat[:, :-1]))
                elif model_name in ['dkvmn', 'akt', 'simplekt', 'qikt', 'lbkt', 'dkt_forget', 'deep_irt']:
                    y_hat = model(question, response, mask, interval, r_time)
                    if y_hat.shape[1] == question.shape[1]: 
                        preds = y_hat[:, :-1]
                    else:
                        preds = y_hat
                elif model_name in ['gikt', 'gikt_old', 'dkt_forget', 'deep_irt'] or cognitive_mode == 'classic':
                    y_hat = model(question, response, mask, interval, r_time)
                    preds = y_hat[:, 1:]
                else:
                    y_hat = model(question, response, mask)
                    preds = y_hat[:, 1:]

                targets = response[:, 1:].float()
                mask_valid = mask[:, 1:]
                eval_mask_valid = eval_mask[:, 1:]
                
                final_mask = mask_valid & eval_mask_valid
                
                if final_mask.sum() == 0:
                    continue
                    
                preds_filtered = preds[final_mask]
                targets_filtered = targets[final_mask]
                
                loss = self.criterion(preds_filtered, targets_filtered)

                # 添加 4PL 正则化 (针对 GIKT)
                reg_loss = 0.0
                if hasattr(model, 'discrimination_gain'): 
                    reg_loss += 0.01 * (model.discrimination_gain ** 2)
                if hasattr(model, 'discrimination_bias'): 
                    reg_loss += self.reg_4pl * torch.sum(model.discrimination_bias.weight ** 2)
                if hasattr(model, 'guessing_bias') and hasattr(model, 'slipping_bias'):
                    reg_loss += self.reg_4pl * torch.sum(torch.relu(model.guessing_bias.weight + 2.0)**2)
                    reg_loss += self.reg_4pl * torch.sum(torch.relu(model.slipping_bias.weight + 3.0)**2)
                loss += reg_loss

            scaler.scale(loss).backward()
            
            if self.gradient_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip_norm)
                
            scaler.step(optimizer)
            scaler.update()

            # Record Gradients and Weights for Deep Monitoring
            if tb_writer and batch_idx == 0:
                try:
                    for name, param in model.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            # 转换成 CPU numpy，并显式指定 float32 类型，解决特定 numpy/tensorboard 版本兼容性报错
                            grad_data = param.grad.detach().cpu().numpy().astype(np.float32).flatten()
                            weight_data = param.detach().cpu().numpy().astype(np.float32).flatten()
                            if grad_data.size > 0:
                                tb_writer.add_histogram(f'Gradients/{name}', grad_data, epoch_idx)
                            if weight_data.size > 0:
                                tb_writer.add_histogram(f'Weights/{name}', weight_data, epoch_idx)
                except TypeError:
                    # 彻底隔离 Numpy>=1.24 与 Pytorch 内置 tensorboard 冲突导致的 np.greater 报错
                    # 如果环境不兼容，直接放弃梯度直方图记录，保全模型主干训练逻辑
                    pass

            total_loss += loss.item() * preds_filtered.size(0)
            total_samples += preds_filtered.size(0)

            # Store predictions after Sigmoid conversion for AUC/ACC calculation
            with torch.no_grad():
                prob_preds = torch.sigmoid(preds_filtered)
                # Ensure no inf/nan before appending
                valid_mask = torch.isfinite(prob_preds) & torch.isfinite(targets_filtered)
                if valid_mask.sum() > 0:
                    all_preds.extend(prob_preds[valid_mask].cpu().numpy())
                    all_targets.extend(targets_filtered[valid_mask].cpu().numpy())
            
            if self.verbose_batch:
                batch_time = time.time() - batch_start_time
                self.logger.info(f"[Train] Epoch {epoch_idx} | Batch {batch_idx+1}/{len(dataloader)} | Loss: {loss.item():.4f} | Time: {batch_time:.3f}s")
            
        epoch_loss = total_loss / total_samples if total_samples > 0 else 0
        
        # Compute Train AUC and Train ACC
        train_auc, train_acc = 0.0, 0.0
        if all_preds and all_targets:
            all_preds_np = np.array(all_preds)
            all_targets_np = np.array(all_targets)
            if not (np.isnan(all_preds_np).any() or np.isinf(all_preds_np).any()):
                try:
                    train_auc = metrics.roc_auc_score(all_targets_np, all_preds_np)
                    train_acc = metrics.accuracy_score(all_targets_np, [1 if p >= 0.5 else 0 for p in all_preds_np])
                except ValueError:
                    pass # Handled single class exception cases mostly in very tiny batches

        return epoch_loss, train_auc, train_acc

    def _evaluate(self, model, dataloader, use_amp, epoch_idx=None):
        model.eval()
        all_preds = []
        all_targets = []
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                batch_start_time = time.time()
                features = {k: v.to(self.device) for k, v in batch.items()}
                question = features[SeqFeatureKey.Q].to(torch.long)
                skill = features.get(SeqFeatureKey.C, torch.zeros_like(question)).to(torch.long)
                response = features[SeqFeatureKey.R].to(torch.long)
                mask = features[SeqFeatureKey.MASK].to(torch.bool)
                eval_mask = features.get(SeqFeatureKey.EVAL_MASK, mask).to(torch.bool)
                interval = features[SeqFeatureKey.T_INTERVAL].to(torch.float32)
                r_time = features[SeqFeatureKey.T_RESPONSE].to(torch.float32)
                
                interval = torch.nan_to_num(interval, nan=0.0)
                r_time = torch.nan_to_num(r_time, nan=0.0)

                model_name = getattr(model, 'model_name', '').lower()
                cognitive_mode = getattr(model, 'cognitive_mode', None)
                
                with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                    if model_name == 'dkt':
                        y_hat = model(question, response, mask, skill=skill)
                        # DKT returns sigmoid probabilities, need to convert back to logits for loss
                        eps = 1e-6
                        y_hat = torch.clamp(y_hat, eps, 1-eps)
                        preds = torch.log(y_hat[:, :-1] / (1 - y_hat[:, :-1]))
                        y_hat_prob = y_hat[:, :-1]
                    elif model_name in ['dkvmn', 'akt', 'simplekt', 'qikt', 'lbkt']:
                        y_hat = model(question, response, mask, interval, r_time)
                        preds = y_hat if y_hat.shape[1] != question.shape[1] else y_hat[:, :-1]
                        y_hat_prob = torch.sigmoid(preds)
                    elif model_name in ['gikt', 'gikt_old', 'dkt_forget', 'deep_irt'] or cognitive_mode == 'classic':
                        y_hat = model(question, response, mask, interval, r_time)
                        preds = y_hat[:, 1:]
                        y_hat_prob = torch.sigmoid(preds)
                    else:
                        y_hat = model(question, response, mask)
                        preds = y_hat[:, 1:]
                        y_hat_prob = torch.sigmoid(preds)

                targets = response[:, 1:].float()
                mask_valid = mask[:, 1:]
                eval_mask_valid = eval_mask[:, 1:]
                final_mask = mask_valid & eval_mask_valid
                
                if final_mask.sum() > 0:
                    p_logits = preds[final_mask]
                    p_prob = y_hat_prob[final_mask].cpu().numpy()
                    t = targets[final_mask]
                    
                    # Calculate Val Loss
                    val_loss = self.criterion(p_logits, t)
                    total_loss += val_loss.item() * t.size(0)
                    total_samples += t.size(0)
                        
                    all_preds.extend(p_prob)
                    all_targets.extend(t.cpu().numpy())
                    
                if self.verbose_batch:
                    batch_time = time.time() - batch_start_time
                    self.logger.info(f"[Eval] Epoch {epoch_idx} | Batch {batch_idx+1}/{len(dataloader)} | Time: {batch_time:.3f}s")
        
        if not all_preds:
            return 0.0, 0.0, 0.0
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        all_targets = np.array(all_targets)
        all_preds = np.array(all_preds)
        
        auc = metrics.roc_auc_score(all_targets, all_preds)
        acc = metrics.accuracy_score(all_targets, [1 if p >= 0.5 else 0 for p in all_preds])
        return auc, acc, avg_loss

    def cross_validate(self, dataset, test_dataset=None):
        """
        执行 K-Fold 交叉验证
        返回: 各折在验证集上的最佳 AUC 列表
        """
        groups = getattr(dataset, 'groups', None)
        
        if groups is not None:
            if self.k_fold == 1:
                k_fold = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            else:
                k_fold = GroupKFold(n_splits=self.k_fold)
            splits = list(k_fold.split(dataset, groups=groups))
        else:
            if self.k_fold == 1:
                k_fold = ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            else:
                k_fold = KFold(n_splits=self.k_fold, shuffle=True, random_state=42)
            splits = list(k_fold.split(dataset))

        fold_aucs = []
        
        # History dict for storing metrics for dataframe conversion
        history_records = []

        # TensorBoard run root: one training call -> one run directory
        time_now = datetime.now().strftime('%Y%m%d_%H%M%S')
        try:
            base_out_dir = self.config.path.OUTPUT_DIR # type: ignore
        except AttributeError:
            base_out_dir = './output'
        runs_root = os.path.join(base_out_dir, self.task_name, "runs", f"{self.model_name}_{self.dataset_name}_{time_now}")

        for fold, (train_indices, val_indices) in enumerate(splits):
            self.logger.info(f"--- Fold {fold + 1}/{len(splits)} ---")
            
            train_set = Subset(dataset, train_indices)
            val_set = Subset(dataset, val_indices)
            
            train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False)
            
            # 实例化模型
            model = self.model_factory_func(
                num_question=self.num_question, 
                num_skill=self.num_skill, 
                device=self.device, 
                config=self.config, 
                **self.kwargs
            )
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.kwargs.get('weight_decay', 1e-5))
            
            # 初始化 AMP
            use_amp = self.kwargs.get('amp_enabled', True) and torch.cuda.is_available()
            if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler"):
                scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
            elif hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
                scaler = torch.amp.GradScaler(enabled=use_amp)
            else:
                # 兼容 DummyScaler
                class DummyScaler:
                    def scale(self, loss): return loss
                    def step(self, optimizer): optimizer.step()
                    def update(self): pass
                scaler = DummyScaler()
            
            best_val_auc = 0.0
            patience_counter = 0
            
            if fold == 0:
                train_writer = SummaryWriter(log_dir=os.path.join(runs_root, "fold_1", "train"))
                val_writer = SummaryWriter(log_dir=os.path.join(runs_root, "fold_1", "val"))
            else:
                train_writer = None
                val_writer = None

            for epoch in range(self.epochs):
                start_time = time.time()
                train_loss, train_auc, train_acc = self._train_epoch(model, train_loader, optimizer, scaler, use_amp, epoch_idx=epoch+1, tb_writer=train_writer)
                val_auc, val_acc, val_loss = self._evaluate(model, val_loader, use_amp, epoch_idx=epoch+1)
                end_time = time.time()

                # Write to TensorBoard
                if train_writer and val_writer:
                    train_writer.add_scalar('Performance/Loss', train_loss, epoch)
                    train_writer.add_scalar('Performance/AUC', train_auc, epoch)
                    train_writer.add_scalar('Performance/ACC', train_acc, epoch)

                    val_writer.add_scalar('Performance/Loss', val_loss, epoch)
                    val_writer.add_scalar('Performance/AUC', val_auc, epoch)
                    val_writer.add_scalar('Performance/ACC', val_acc, epoch)
                    
                    try:
                        current_lr = optimizer.param_groups[0]['lr']
                        train_writer.add_scalar('Optimization/Learning_Rate', current_lr, epoch)
                    except:
                        pass

                # Append to history
                history_records.append({
                    'fold': fold + 1,
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'train_auc': train_auc,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_auc': val_auc,
                    'val_acc': val_acc,
                    'time_sec': end_time - start_time
                })
                
                self.logger.info(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | time: {end_time - start_time:.2f}s")
                
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    patience_counter = 0
                    if self.kwargs.get('save_model') and self.kwargs.get('model_save_path'):
                        torch.save(model.state_dict(), self.kwargs.get('model_save_path'))
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.patience:
                    self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break
                    
            if train_writer and val_writer:
                train_writer.close()
                val_writer.close()

            if test_dataset is not None:
                if self.kwargs.get('save_model') and self.kwargs.get('model_save_path'):
                    # Load best model weights for this fold
                    best_model_path = self.kwargs.get('model_save_path')
                    if os.path.exists(best_model_path):
                        self.logger.info(f"Loading best model for fold {fold+1} test evaluation...")
                        state_dict = torch.load(best_model_path, map_location=self.device)
                        if isinstance(state_dict, torch.nn.Module):
                            model = state_dict
                        else:
                            model.load_state_dict(state_dict)

                test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
                test_auc, test_acc, _ = self._evaluate(model, test_loader, use_amp)
                self.logger.info(f"Fold {fold + 1} Test ACC: {test_acc:.4f} | Test AUC: {test_auc:.4f}\n")
            else:
                self.logger.info("\n")
                
            fold_aucs.append(best_val_auc)
            
        # Save history to CSV and Plot
        try:
            import pandas as pd
            df_history = pd.DataFrame(history_records)
            chart_data_dir = os.path.join(base_out_dir, self.task_name, "chart_data")
            chart_img_dir = os.path.join(base_out_dir, self.task_name, "chart")
            os.makedirs(chart_data_dir, exist_ok=True)
            os.makedirs(chart_img_dir, exist_ok=True)

            csv_path = os.path.join(chart_data_dir, f"{self.model_name}_{self.dataset_name}_history_{time_now}.csv")
            df_history.to_csv(csv_path, index=False)
            self.logger.info(f"Training history saved to {csv_path}")
            
            # Write TB Hparams
            avg_val_auc = sum(fold_aucs) / len(fold_aucs) if fold_aucs else 0.0
            hparam_writer = SummaryWriter(log_dir=runs_root)
            hparam_writer.add_hparams(
                hparam_dict={
                    'model': self.model_name,
                    'dataset': self.dataset_name,
                    'lr': self.learning_rate,
                    'batch_size': self.batch_size,
                    'epochs': self.epochs
                },
                metric_dict={'hparam/Avg_Best_AUC': avg_val_auc}
            )
            hparam_writer.close()

            self._plot_training_history(df_history, chart_img_dir, time_now)
        except Exception as e:
            self.logger.error(f"Failed to save training history: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            
        return fold_aucs

    def _plot_training_history(self, df, save_dir, timestamp):
        """绘制学术级训练历史图: 均值曲线及阴影置信区间"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # 设置学术风的主题
            sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
            color_train = "#4DBBD5"  # 宁静蓝
            color_val = "#E64B35"    # 高光红

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            # 1. Loss Plot
            sns.lineplot(data=df, x='epoch', y='train_loss', label='Train Loss', color=color_train, ax=axes[0], errorbar='ci', n_boot=100)
            sns.lineplot(data=df, x='epoch', y='val_loss', label='Val Loss', color=color_val, ax=axes[0], errorbar='ci', n_boot=100)
            axes[0].set_title(f'Loss Curve (5-Fold Mean & 95% CI)')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].legend()

            # 2. AUC Plot
            sns.lineplot(data=df, x='epoch', y='train_auc', label='Train AUC', color=color_train, ax=axes[1], errorbar='ci', n_boot=100)
            sns.lineplot(data=df, x='epoch', y='val_auc', label='Val AUC', color=color_val, ax=axes[1], errorbar='ci', n_boot=100)
            axes[1].set_title(f'AUC Curve (5-Fold Mean & 95% CI)')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('AUC')
            axes[1].legend()
            
            # 3. ACC Plot
            sns.lineplot(data=df, x='epoch', y='train_acc', label='Train ACC', color=color_train, ax=axes[2], errorbar='ci', n_boot=100)
            sns.lineplot(data=df, x='epoch', y='val_acc', label='Val ACC', color=color_val, ax=axes[2], errorbar='ci', n_boot=100)
            axes[2].set_title(f'Accuracy Curve (5-Fold Mean & 95% CI)')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Accuracy')
            axes[2].legend()

            plt.tight_layout()
            img_path = os.path.join(save_dir, f"{self.model_name}_{self.dataset_name}_metrics_{timestamp}.png")
            plt.savefig(img_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Journal-quality training plots saved to {img_path}")
        except ModuleNotFoundError:
            self.logger.warning("Seaborn module not found. Please `pip install seaborn` for journal quality plots.")
        except Exception as e:
            self.logger.error(f"Failed to plot history: {e}")

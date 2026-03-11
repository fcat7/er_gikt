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
            log_file = os.path.join(log_dir, f"trainer_{time_now}.log")
            
            fh = logging.FileHandler(log_file, encoding='utf-8')
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
            
            self.logger.info(f"Logger initialized. File output: {log_file}")

    def _train_epoch(self, model, dataloader, optimizer, scaler, use_amp, epoch_idx=None):
        model.train()
        total_loss = 0
        total_samples = 0
        
        for batch_idx, batch in enumerate(dataloader):
            batch_start_time = time.time()
            features = {k: v.to(self.device) for k, v in batch.items()}
            question = features[SeqFeatureKey.Q].to(torch.long)
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
                    y_hat = model(question, response, mask)
                    eps = 1e-6
                    y_hat = torch.clamp(y_hat, eps, 1-eps)
                    preds = torch.log(y_hat[:, :-1] / (1 - y_hat[:, :-1]))
                elif model_name in ['dkvmn', 'akt', 'simplekt', 'qikt', 'lbkt']:
                    y_hat = model(question, response, mask, interval, r_time)
                    if y_hat.shape[1] == question.shape[1]: 
                        preds = y_hat[:, :-1]
                    else:
                        preds = y_hat
                elif model_name in ['gikt', 'gikt_old'] or cognitive_mode == 'classic':
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
            
            total_loss += loss.item() * preds_filtered.size(0)
            total_samples += preds_filtered.size(0)
            
            if self.verbose_batch:
                batch_time = time.time() - batch_start_time
                self.logger.info(f"[Train] Epoch {epoch_idx} | Batch {batch_idx+1}/{len(dataloader)} | Loss: {loss.item():.4f} | Time: {batch_time:.3f}s")
            
        return total_loss / total_samples if total_samples > 0 else 0

    def _evaluate(self, model, dataloader, use_amp, epoch_idx=None):
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                batch_start_time = time.time()
                features = {k: v.to(self.device) for k, v in batch.items()}
                question = features[SeqFeatureKey.Q].to(torch.long)
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
                        y_hat = model(question, response, mask)
                        preds = y_hat[:, :-1]
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
                        y_hat = model(question, response, mask)
                        y_hat = torch.sigmoid(y_hat)
                        preds = y_hat[:, 1:]

                targets = response[:, 1:].float()
                mask_valid = mask[:, 1:]
                eval_mask_valid = eval_mask[:, 1:]
                
                final_mask = mask_valid & eval_mask_valid
                
                if final_mask.sum() > 0:
                    p = preds[final_mask].cpu().numpy()
                    t = targets[final_mask].cpu().numpy()
                    
                    if np.isnan(p).any() or np.isinf(p).any():
                        continue
                        
                    all_preds.extend(p)
                    all_targets.extend(t)
                    
                if self.verbose_batch:
                    batch_time = time.time() - batch_start_time
                    self.logger.info(f"[Eval] Epoch {epoch_idx} | Batch {batch_idx+1}/{len(dataloader)} | Time: {batch_time:.3f}s")
        
        if not all_preds:
            return 0, 0
        
        all_targets = np.array(all_targets)
        all_preds = np.array(all_preds)
        
        if np.isnan(all_preds).any() or np.isinf(all_preds).any():
            return 0, 0
            
        auc = metrics.roc_auc_score(all_targets, all_preds)
        acc = metrics.accuracy_score(all_targets, [1 if p >= 0.5 else 0 for p in all_preds])
        return auc, acc

    def cross_validate(self, dataset):
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
            
            # TensorBoard Setup
            trial_number = self.kwargs.get('trial_number', 'unknown')
            log_dir = os.path.join("runs", f"trial_{trial_number}_fold_{fold+1}")
            writer = SummaryWriter(log_dir=log_dir)
            
            for epoch in range(self.epochs):
                start_time = time.time()
                train_loss = self._train_epoch(model, train_loader, optimizer, scaler, use_amp, epoch_idx=epoch+1)
                val_auc, val_acc = self._evaluate(model, val_loader, use_amp, epoch_idx=epoch+1)
                end_time = time.time()
                
                # Write to TensorBoard
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('AUC/val', val_auc, epoch)
                writer.add_scalar('ACC/val', val_acc, epoch)
                
                self.logger.info(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val ACC: {val_acc:.4f} | Val AUC: {val_auc:.4f} | time: {end_time - start_time:.2f}s")
                
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
                    
            writer.close()
            self.logger.info(f"Fold {fold + 1} Best Val AUC: {best_val_auc:.4f}\n")
            fold_aucs.append(best_val_auc)
            
        return fold_aucs

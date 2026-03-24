import time
import os
import logging
from datetime import datetime
import gc
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
        # 模型保存策略：默认保存每折最优；全局最优默认不保存（可通过参数开启）
        self.save_model = kwargs.get('save_model', True)
        self.save_fold_checkpoints = kwargs.get('save_fold_checkpoints', True)
        self.save_global_best = kwargs.get('save_global_best', False)
        
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
        self.logger.info(f"  Save Model: {self.save_model}")
        self.logger.info(f"  Save Fold Checkpoints: {self.save_fold_checkpoints}")
        self.logger.info(f"  Save Global Best: {self.save_global_best}")
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
                    # 当前 DKT 前向返回的是题目级 next-item logits，不是概率。
                    y_hat = model(question, response, mask, skill=skill)
                    preds = y_hat[:, :-1]
                elif model_name in ['dkvmn', 'akt', 'simplekt', 'qikt', 'deep_irt', 'gkt']:
                    # 必须把 skill 显式传给基线模型，否则它们内部会因为获取不到 skill=None 退化为全 0 张量
                    try:
                        y_hat = model(question, response, mask, interval, r_time, skill=skill)
                    except TypeError:
                        # 兼容某些没有 skill 参数的模型 (例如 QIKT 如果没定义 skill)
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

                # 添加 4PL 正则化 (针对 GIKT: Zero-mean L2 for stable priors)
                reg_loss = 0.0
                if hasattr(model, 'discrimination_gain'): 
                    reg_loss += 0.01 * (model.discrimination_gain ** 2)
                if hasattr(model, 'difficulty_bias'): 
                    reg_loss += self.reg_4pl * torch.sum(model.difficulty_bias.weight ** 2)
                if hasattr(model, 'discrimination_bias'): 
                    reg_loss += self.reg_4pl * torch.sum(model.discrimination_bias.weight ** 2)
                if hasattr(model, 'guessing_bias') and hasattr(model, 'slipping_bias'):
                    reg_loss += self.reg_4pl * torch.sum(model.guessing_bias.weight ** 2)
                    reg_loss += self.reg_4pl * torch.sum(model.slipping_bias.weight ** 2)
                
                # 针对 AKT 模型: Rasch difficulty L2 regularization (P2 fix)
                if model_name == 'akt' and hasattr(model, 'q_diff_embed'):
                    rasch_loss = torch.sum(model.q_diff_embed.weight ** 2)
                    # PyKT AKT paper officially uses 1e-5 for rasch l2 regularization
                    reg_loss += 1e-5 * rasch_loss
                
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

        return epoch_loss, train_auc, train_acc, total_samples

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
                        # 当前 DKT 前向返回的是题目级 next-item logits，不是概率。
                        y_hat = model(question, response, mask, skill=skill)
                        preds = y_hat[:, :-1]
                        y_hat_prob = torch.sigmoid(preds)
                    elif model_name in ['dkvmn', 'akt', 'simplekt', 'qikt', 'deep_irt', 'gkt']:
                        try:
                            y_hat = model(question, response, mask, interval, r_time, skill=skill)
                        except TypeError:
                            y_hat = model(question, response, mask, interval, r_time)
                        preds = y_hat if y_hat.shape[1] != question.shape[1] else y_hat[:, :-1]
                        y_hat_prob = torch.sigmoid(preds)
                    elif model_name in ['gikt', 'gikt_old'] or cognitive_mode == 'classic':
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
            return 0.0, 0.0, 0.0, 0
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        all_targets = np.array(all_targets)
        all_preds = np.array(all_preds)
        
        auc = metrics.roc_auc_score(all_targets, all_preds)
        acc = metrics.accuracy_score(all_targets, [1 if p >= 0.5 else 0 for p in all_preds])
        return auc, acc, avg_loss, total_samples

    @staticmethod
    def _format_fold_records(fold_result_records, key, value_format):
        formatted = []
        for record in fold_result_records:
            value = record.get(key, '')
            if isinstance(value, (int, np.integer)):
                value_str = f"{int(value)}"
            elif isinstance(value, (float, np.floating)):
                value_str = format(float(value), value_format)
            else:
                value_str = str(value)
            formatted.append(f"F{record['fold']}:{value_str}")
        return '; '.join(formatted)

    def _append_baseline_summary(self, summary_path, summary_row):
        import csv

        summary_headers = [
            'Date',
            'Task Name',
            'Model',
            'Dataset',
            'K Fold',
            'Run Name',
            'History CSV',
            'Fold Best Epochs',
            'Fold Best Val AUCs',
            'Fold Best Val ACCs',
            'Fold Best Val LOSSes',
            'Fold Test AUCs',
            'Fold Test ACCs',
            'Fold Test LOSSes',
            'Mean Test AUC',
            'Std Test AUC',
            'Mean Test ACC',
            'Std Test ACC',
            'Mean Test LOSS',
            'Std Test LOSS',
            'Max AUC (Fold)',
            'Min AUC (Fold)',
            'Avg Epoch Train Time',
            'Avg Epoch Val Time',
            'Avg Epoch Total Time',
            'Avg Epoch Test Time',
            'Total Train Time',
            'Total Val Time',
            'Total Test Time',
            'Total Wall Time',
        ]

        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        file_exists = os.path.exists(summary_path) and os.path.getsize(summary_path) > 0

        with open(summary_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=summary_headers)
            if not file_exists:
                writer.writeheader()
            writer.writerow(summary_row)

    def cross_validate(self, dataset, test_dataset=None):
        """
        执行 K-Fold 交叉验证
        返回: 各折在验证集上的最佳 AUC 列表
        """
        cv_start_time = time.time()
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
        fold_result_records = []
        
        # History dict for storing metrics for dataframe conversion
        history_records = []

        # TensorBoard run root: one training call -> one run directory
        time_now = datetime.now().strftime('%Y%m%d_%H%M%S')
        try:
            base_out_dir = self.config.path.OUTPUT_DIR # type: ignore
        except AttributeError:
            base_out_dir = './output'
        run_name = f"{self.model_name}_{self.dataset_name}_{time_now}"
        runs_root = os.path.join(base_out_dir, self.task_name, "runs", run_name)

        # 解析 checkpoint 保存路径，并准备“每折保存目录”
        model_save_path = self.kwargs.get('model_save_path') if self.save_model else None
        save_dir = ""
        save_stem = ""
        fold_ckpt_root = ""
        if model_save_path:
            save_dir = os.path.dirname(model_save_path) or '.'
            save_file = os.path.basename(model_save_path)
            save_stem, _ = os.path.splitext(save_file)
            fold_ckpt_root = os.path.join(save_dir, f"{save_stem}_folds")
            if self.save_fold_checkpoints:
                os.makedirs(fold_ckpt_root, exist_ok=True)

        # 全局最优（跨所有折、按验证集 AUC）
        global_best_val_auc = float('-inf')
        global_best_fold = 0

        for fold, (train_indices, val_indices) in enumerate(splits):
            self.logger.info(f"--- Fold {fold + 1}/{len(splits)} ---")

            # 当前折最优 checkpoint 路径
            fold_best_ckpt_path = ""
            if self.save_model and self.save_fold_checkpoints and fold_ckpt_root:
                fold_best_ckpt_path = os.path.join(fold_ckpt_root, f"{save_stem}_fold{fold + 1}_best.pt")
            
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
            wd = float(self.kwargs.get('weight_decay', 1e-5))
            optimizer = optim.Adam(model.parameters(), lr=float(self.learning_rate), weight_decay=wd)
            
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
            best_val_acc = 0.0
            best_val_loss = float('inf')
            best_epoch = 0
            patience_counter = 0
            best_state_dict_cpu = None
            fold_train_time_sec = 0.0
            fold_val_time_sec = 0.0
            fold_test_time_sec = 0.0
            epochs_run = 0
            best_fold_test_auc = 0.0
            best_fold_test_acc = 0.0
            best_fold_test_loss = 0.0
            
            if fold == 0:
                train_writer = SummaryWriter(log_dir=os.path.join(runs_root, "fold_1", "train"))
                val_writer = SummaryWriter(log_dir=os.path.join(runs_root, "fold_1", "val"))
            else:
                train_writer = None
                val_writer = None

            for epoch in range(self.epochs):
                train_start_time = time.time()
                train_loss, train_auc, train_acc, train_samples = self._train_epoch(model, train_loader, optimizer, scaler, use_amp, epoch_idx=epoch+1, tb_writer=train_writer)
                train_time_sec = time.time() - train_start_time

                val_start_time = time.time()
                val_auc, val_acc, val_loss, val_samples = self._evaluate(model, val_loader, use_amp, epoch_idx=epoch+1)
                val_time_sec = time.time() - val_start_time
                epoch_time_sec = train_time_sec + val_time_sec

                fold_train_time_sec += train_time_sec
                fold_val_time_sec += val_time_sec
                epochs_run += 1

                current_lr = optimizer.param_groups[0]['lr']
                is_best_epoch = False

                # Write to TensorBoard
                if train_writer and val_writer:
                    train_writer.add_scalar('Performance/Loss', train_loss, epoch)
                    train_writer.add_scalar('Performance/AUC', train_auc, epoch)
                    train_writer.add_scalar('Performance/ACC', train_acc, epoch)

                    val_writer.add_scalar('Performance/Loss', val_loss, epoch)
                    val_writer.add_scalar('Performance/AUC', val_auc, epoch)
                    val_writer.add_scalar('Performance/ACC', val_acc, epoch)
                    
                    try:
                        train_writer.add_scalar('Optimization/Learning_Rate', current_lr, epoch)
                    except:
                        pass
                
                vram_info = ""
                if torch.cuda.is_available():
                    # 显示最大已分配(Allocated)和最大已保留(Reserved)对显存的使用情况
                    vram_alloc_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                    vram_res_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
                    vram_info = f" | VRAM: {vram_alloc_gb:.2f}G (Res: {vram_res_gb:.2f}G)"

                self.logger.info(
                    f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f} | "
                    f"Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | "
                    f"train: {train_time_sec:.2f}s | val: {val_time_sec:.2f}s | total: {epoch_time_sec:.2f}s{vram_info}"
                )

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_val_acc = val_acc
                    best_val_loss = val_loss
                    best_epoch = epoch + 1
                    is_best_epoch = True
                    patience_counter = 0
                    # Keep an in-memory copy of best weights on CPU for robust test-time loading.
                    # This avoids potential Windows file I/O edge cases during torch.load.
                    try:
                        with torch.no_grad():
                            best_state_dict_cpu = {
                                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                            }
                    except Exception as e:
                        self.logger.warning(f"Failed to snapshot best state_dict to CPU: {e}")
                    # 保存策略：每折最优 + 可选全局最优
                    if self.save_model:
                        # 1) 保存当前折最优 checkpoint
                        if self.save_fold_checkpoints and fold_best_ckpt_path:
                            torch.save(model.state_dict(), fold_best_ckpt_path)

                        # 2) 可选：保存全局最优 checkpoint（跨折）
                        if self.save_global_best and model_save_path and val_auc > global_best_val_auc:
                            global_best_val_auc = val_auc
                            global_best_fold = fold + 1
                            torch.save(model.state_dict(), model_save_path)
                else:
                    patience_counter += 1

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
                    'train_time_sec': train_time_sec,
                    'val_time_sec': val_time_sec,
                    'epoch_time_sec': epoch_time_sec,
                    'train_samples': train_samples,
                    'val_samples': val_samples,
                    'learning_rate': current_lr,
                    'is_best_epoch': is_best_epoch,
                })

                # 语义修正：每折仅保留“最终最佳 epoch”一个 True
                if is_best_epoch:
                    for prev_idx in range(len(history_records) - 2, -1, -1):
                        if history_records[prev_idx]['fold'] != (fold + 1):
                            break
                        if history_records[prev_idx].get('is_best_epoch', False):
                            history_records[prev_idx]['is_best_epoch'] = False
                    
                if patience_counter >= self.patience:
                    self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break
                    
            if train_writer and val_writer:
                train_writer.close()
                val_writer.close()

            if test_dataset is not None:
                # Load best model weights for this fold
                if best_state_dict_cpu is not None:
                    self.logger.info(f"Loading best model for fold {fold+1} test evaluation (in-memory CPU snapshot)...")
                    try:
                        model.load_state_dict(best_state_dict_cpu)
                    except Exception as e:
                        self.logger.error(f"Failed to load in-memory best state_dict: {e}")
                elif self.save_model:
                    # 若内存快照不存在，则优先从“当前折最优文件”回读；其次尝试“全局最优文件”
                    best_model_candidates = []
                    if fold_best_ckpt_path:
                        best_model_candidates.append(fold_best_ckpt_path)
                    if model_save_path:
                        best_model_candidates.append(model_save_path)

                    loaded_from_disk = False
                    for best_model_path in best_model_candidates:
                        if best_model_path and os.path.exists(best_model_path):
                            self.logger.info(f"Loading best model for fold {fold+1} test evaluation (torch.load from disk): {best_model_path}")
                            try:
                                state_dict = torch.load(best_model_path, map_location=self.device)
                                if isinstance(state_dict, torch.nn.Module):
                                    model = state_dict
                                else:
                                    model.load_state_dict(state_dict)
                                loaded_from_disk = True
                                break
                            except Exception as e:
                                self.logger.error(f"Failed to torch.load best model from disk ({best_model_path}): {e}")
                    if not loaded_from_disk:
                        self.logger.warning(f"Fold {fold+1} has no available checkpoint on disk for fallback loading.")

                # Best-effort memory cleanup before long test evaluation.
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
                test_start_time = time.time()
                test_auc, test_acc, test_loss, _ = self._evaluate(model, test_loader, use_amp)
                fold_test_time_sec = time.time() - test_start_time
                best_fold_test_auc = test_auc
                best_fold_test_acc = test_acc
                best_fold_test_loss = test_loss
                self.logger.info(
                    f"Fold {fold + 1} Test ACC: {test_acc:.4f} | Test AUC: {test_auc:.4f} | "
                    f"Test Loss: {test_loss:.4f} | Test Time: {fold_test_time_sec:.2f}s\n"
                )
            else:
                self.logger.info("\n")
                
            fold_aucs.append(best_val_auc)
            fold_result_records.append({
                'fold': fold + 1,
                'best_epoch': best_epoch,
                'best_val_auc': best_val_auc,
                'best_val_acc': best_val_acc,
                'best_val_loss': best_val_loss,
                'test_auc': best_fold_test_auc,
                'test_acc': best_fold_test_acc,
                'test_loss': best_fold_test_loss,
                'fold_train_time_sec': fold_train_time_sec,
                'fold_val_time_sec': fold_val_time_sec,
                'fold_test_time_sec': fold_test_time_sec,
                'epochs_run': epochs_run,
            })
            
        # Save history to CSV and Plot
        history_csv_path = ""
        try:
            import pandas as pd
            df_history = pd.DataFrame(history_records)
            task_root_dir = os.path.join(base_out_dir, self.task_name)
            chart_data_dir = os.path.join(base_out_dir, self.task_name, "chart_data")
            chart_img_dir = os.path.join(base_out_dir, self.task_name, "chart")
            os.makedirs(chart_data_dir, exist_ok=True)
            os.makedirs(chart_img_dir, exist_ok=True)

            history_csv_path = os.path.join(chart_data_dir, f"{self.model_name}_{self.dataset_name}_history_{time_now}.csv")
            df_history.to_csv(history_csv_path, index=False)
            self.logger.info(f"Training history saved to {history_csv_path}")

            test_aucs = [record['test_auc'] for record in fold_result_records]
            test_accs = [record['test_acc'] for record in fold_result_records]
            test_losses = [record['test_loss'] for record in fold_result_records]
            total_train_time = sum(record['fold_train_time_sec'] for record in fold_result_records)
            total_val_time = sum(record['fold_val_time_sec'] for record in fold_result_records)
            total_test_time = sum(record['fold_test_time_sec'] for record in fold_result_records)
            total_wall_time = time.time() - cv_start_time

            avg_epoch_train_time = float(df_history['train_time_sec'].mean()) if not df_history.empty else 0.0
            avg_epoch_val_time = float(df_history['val_time_sec'].mean()) if not df_history.empty else 0.0
            avg_epoch_total_time = float(df_history['epoch_time_sec'].mean()) if not df_history.empty else 0.0
            # 说明：测试阶段通常每折仅执行一次，此处取“每折测试耗时均值”作为 Avg Epoch Test Time
            avg_epoch_test_time = float(total_test_time / len(fold_result_records)) if fold_result_records else 0.0

            mean_test_auc = float(np.mean(test_aucs)) if test_aucs else 0.0
            std_test_auc = float(np.std(test_aucs)) if test_aucs else 0.0
            mean_test_acc = float(np.mean(test_accs)) if test_accs else 0.0
            std_test_acc = float(np.std(test_accs)) if test_accs else 0.0
            mean_test_loss = float(np.mean(test_losses)) if test_losses else 0.0
            std_test_loss = float(np.std(test_losses)) if test_losses else 0.0

            max_auc_text = ""
            min_auc_text = ""
            if test_aucs:
                max_auc_idx = int(np.argmax(test_aucs))
                min_auc_idx = int(np.argmin(test_aucs))
                max_auc_text = f"{test_aucs[max_auc_idx]:.4f} (F{fold_result_records[max_auc_idx]['fold']})"
                min_auc_text = f"{test_aucs[min_auc_idx]:.4f} (F{fold_result_records[min_auc_idx]['fold']})"

            summary_path = os.path.join(task_root_dir, 'baseline_summary.csv')
            summary_row = {
                'Date': time_now,
                'Task Name': self.task_name,
                'Model': self.model_name,
                'Dataset': self.dataset_name,
                'K Fold': len(splits),
                'Run Name': run_name,
                'History CSV': history_csv_path,
                'Fold Best Epochs': self._format_fold_records(fold_result_records, 'best_epoch', 'd'),
                'Fold Best Val AUCs': self._format_fold_records(fold_result_records, 'best_val_auc', '.4f'),
                'Fold Best Val ACCs': self._format_fold_records(fold_result_records, 'best_val_acc', '.4f'),
                'Fold Best Val LOSSes': self._format_fold_records(fold_result_records, 'best_val_loss', '.4f'),
                'Fold Test AUCs': self._format_fold_records(fold_result_records, 'test_auc', '.4f'),
                'Fold Test ACCs': self._format_fold_records(fold_result_records, 'test_acc', '.4f'),
                'Fold Test LOSSes': self._format_fold_records(fold_result_records, 'test_loss', '.4f'),
                'Mean Test AUC': f"{mean_test_auc:.4f}",
                'Std Test AUC': f"{std_test_auc:.4f}",
                'Mean Test ACC': f"{mean_test_acc:.4f}",
                'Std Test ACC': f"{std_test_acc:.4f}",
                'Mean Test LOSS': f"{mean_test_loss:.4f}",
                'Std Test LOSS': f"{std_test_loss:.4f}",
                'Max AUC (Fold)': max_auc_text,
                'Min AUC (Fold)': min_auc_text,
                'Avg Epoch Train Time': f"{avg_epoch_train_time:.4f}",
                'Avg Epoch Val Time': f"{avg_epoch_val_time:.4f}",
                'Avg Epoch Total Time': f"{avg_epoch_total_time:.4f}",
                'Avg Epoch Test Time': f"{avg_epoch_test_time:.4f}",
                'Total Train Time': f"{total_train_time:.4f}",
                'Total Val Time': f"{total_val_time:.4f}",
                'Total Test Time': f"{total_test_time:.4f}",
                'Total Wall Time': f"{total_wall_time:.4f}",
            }
            self._append_baseline_summary(summary_path, summary_row)
            self.logger.info(f"Baseline summary appended to {summary_path}")

            # 输出 checkpoint 保存结果，便于后续评估脚本批量读取
            if self.save_model and self.save_fold_checkpoints and fold_ckpt_root:
                self.logger.info(f"Fold checkpoints saved under: {fold_ckpt_root}")
            if self.save_model and self.save_global_best and model_save_path and os.path.exists(model_save_path):
                if global_best_fold > 0 and global_best_val_auc > float('-inf'):
                    self.logger.info(
                        f"Global best checkpoint saved: {model_save_path} "
                        f"(Fold {global_best_fold}, Best Val AUC={global_best_val_auc:.4f})"
                    )
                else:
                    self.logger.info(f"Global best checkpoint saved: {model_save_path}")
            
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
        """绘制训练历史图：每折单独图 + 共享 epoch 聚合图（均值±标准差）"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import pandas as pd

            # 设置学术风的主题
            sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
            color_train = "#4DBBD5"  # 宁静蓝
            color_val = "#E64B35"    # 高光红

            if df is None or len(df) == 0:
                self.logger.warning("History dataframe is empty. Skip plotting training history.")
                return

            # 目录结构：folds/ 下存每折图；shared/ 下存聚合图
            folds_dir = os.path.join(save_dir, f"{self.model_name}_{self.dataset_name}_folds_{timestamp}")
            shared_dir = os.path.join(save_dir, "shared")
            os.makedirs(folds_dir, exist_ok=True)
            os.makedirs(shared_dir, exist_ok=True)

            def _plot_fold_panel(ax, fold_df: pd.DataFrame, metric_train: str, metric_val: str, ylabel: str, title: str):
                ax.plot(fold_df['epoch'], fold_df[metric_train], color=color_train, linewidth=2.0, label='Train')
                ax.plot(fold_df['epoch'], fold_df[metric_val], color=color_val, linewidth=2.0, label='Val')
                ax.set_title(title)
                ax.set_xlabel('Epoch')
                ax.set_ylabel(ylabel)
                ax.legend(fontsize=8)

            # 1) 输出每折单独图
            fold_ids = sorted(df['fold'].unique())
            for fold_id in fold_ids:
                fold_df = df[df['fold'] == fold_id].copy()
                if fold_df.empty:
                    continue

                fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                _plot_fold_panel(axes[0], fold_df, 'train_loss', 'val_loss', 'Loss', f'Fold {fold_id} - Loss')
                _plot_fold_panel(axes[1], fold_df, 'train_auc', 'val_auc', 'AUC', f'Fold {fold_id} - AUC')
                _plot_fold_panel(axes[2], fold_df, 'train_acc', 'val_acc', 'Accuracy', f'Fold {fold_id} - Accuracy')
                plt.tight_layout()

                fold_img_path = os.path.join(
                    folds_dir,
                    f"{self.model_name}_{self.dataset_name}_fold{int(fold_id)}_metrics_{timestamp}.png"
                )
                plt.savefig(fold_img_path, dpi=300, bbox_inches='tight')
                plt.close()

            # 2) 输出共享 epoch 聚合图（均值±标准差）
            fold_max_epochs = df.groupby('fold')['epoch'].max()
            shared_max_epoch = int(fold_max_epochs.min()) if len(fold_max_epochs) > 0 else 0
            if shared_max_epoch <= 0:
                self.logger.warning("No shared epochs found across folds. Skip shared plot.")
                return

            shared_df = df[df['epoch'] <= shared_max_epoch].copy()

            def _agg_mean_std(dataframe: pd.DataFrame, metric_key: str):
                grouped = dataframe.groupby('epoch')[metric_key].agg(['mean', 'std']).reset_index()
                grouped['std'] = grouped['std'].fillna(0.0)
                return grouped

            def _plot_shared_panel(ax, metric_train: str, metric_val: str, ylabel: str, title: str):
                g_train = _agg_mean_std(shared_df, metric_train)
                g_val = _agg_mean_std(shared_df, metric_val)

                ax.plot(g_train['epoch'], g_train['mean'], color=color_train, linewidth=2.2, label='Train Mean')
                ax.fill_between(
                    g_train['epoch'],
                    g_train['mean'] - g_train['std'],
                    g_train['mean'] + g_train['std'],
                    color=color_train,
                    alpha=0.18,
                    label='Train ±1 STD',
                )

                ax.plot(g_val['epoch'], g_val['mean'], color=color_val, linewidth=2.2, label='Val Mean')
                ax.fill_between(
                    g_val['epoch'],
                    g_val['mean'] - g_val['std'],
                    g_val['mean'] + g_val['std'],
                    color=color_val,
                    alpha=0.18,
                    label='Val ±1 STD',
                )

                ax.set_title(f"{title} (Shared Epochs: 1-{shared_max_epoch})")
                ax.set_xlabel('Epoch')
                ax.set_ylabel(ylabel)
                ax.legend(fontsize=8)

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            _plot_shared_panel(axes[0], 'train_loss', 'val_loss', 'Loss', 'Shared Loss')
            _plot_shared_panel(axes[1], 'train_auc', 'val_auc', 'AUC', 'Shared AUC')
            _plot_shared_panel(axes[2], 'train_acc', 'val_acc', 'Accuracy', 'Shared Accuracy')
            plt.tight_layout()

            shared_img_path = os.path.join(
                shared_dir,
                f"{self.model_name}_{self.dataset_name}_shared_metrics_{timestamp}.png"
            )
            plt.savefig(shared_img_path, dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info(f"Fold plots saved under {folds_dir}")
            self.logger.info(f"Shared plot saved to {shared_img_path}")
        except ModuleNotFoundError:
            self.logger.warning("Seaborn module not found. Please `pip install seaborn` for journal quality plots.")
        except Exception as e:
            self.logger.error(f"Failed to plot history: {e}")

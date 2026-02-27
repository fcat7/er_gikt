import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
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
        
        self.criterion = nn.BCEWithLogitsLoss()

    def _train_epoch(self, model, dataloader, optimizer):
        model.train()
        total_loss = 0
        total_samples = 0
        
        for batch in dataloader:
            features = {k: v.to(self.device) for k, v in batch.items()}
            question = features[SeqFeatureKey.Q].to(torch.long)
            response = features[SeqFeatureKey.R].to(torch.long)
            mask = features[SeqFeatureKey.MASK].to(torch.bool)
            eval_mask = features[SeqFeatureKey.EVAL_MASK].to(torch.bool)
            interval = features[SeqFeatureKey.T_INTERVAL].to(torch.float32)
            r_time = features[SeqFeatureKey.T_RESPONSE].to(torch.float32)
            
            interval = torch.nan_to_num(interval, nan=0.0)
            r_time = torch.nan_to_num(r_time, nan=0.0)

            optimizer.zero_grad()
            
            model_name = getattr(model, 'model_name', '').lower()
            
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
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * preds_filtered.size(0)
            total_samples += preds_filtered.size(0)
            
        return total_loss / total_samples if total_samples > 0 else 0

    def _evaluate(self, model, dataloader):
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                features = {k: v.to(self.device) for k, v in batch.items()}
                question = features[SeqFeatureKey.Q].to(torch.long)
                response = features[SeqFeatureKey.R].to(torch.long)
                mask = features[SeqFeatureKey.MASK].to(torch.bool)
                eval_mask = features[SeqFeatureKey.EVAL_MASK].to(torch.bool)
                interval = features[SeqFeatureKey.T_INTERVAL].to(torch.float32)
                r_time = features[SeqFeatureKey.T_RESPONSE].to(torch.float32)
                
                interval = torch.nan_to_num(interval, nan=0.0)
                r_time = torch.nan_to_num(r_time, nan=0.0)

                model_name = getattr(model, 'model_name', '').lower()
                
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
            print(f"--- Fold {fold + 1}/{len(splits)} ---")
            
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
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
            
            best_val_auc = 0.0
            patience_counter = 0
            
            for epoch in range(self.epochs):
                train_loss = self._train_epoch(model, train_loader, optimizer)
                val_auc, val_acc = self._evaluate(model, val_loader)
                
                print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f} | Val ACC: {val_acc:.4f}")
                
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
                    
            print(f"Fold {fold + 1} Best Val AUC: {best_val_auc:.4f}\n")
            fold_aucs.append(best_val_auc)
            
        return fold_aucs

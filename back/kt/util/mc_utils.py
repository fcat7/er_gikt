import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import sys
import time


# python evaluate_mc.py --model_path model/20260201_1304.pt --num_samples 5 --dataset assist09-sample_10%

def mc_evaluate(model, data_loader, num_samples=30, device=None):
    """
    基于蒙特卡洛Dropout的评估函数 (Monte Carlo Dropout Inference)
    
    该函数对给定的模型和数据进行多次前向传播（开启Dropout），
    以估计预测的概率分布，从而得到均值（作为最终预测）和标准差（作为不确定性度量）。
    
    Args:
        model: 训练好的GIKT模型
        data_loader: 用于测试的数据加载器
        num_samples (int): 蒙特卡洛采样次数 (T), 建议 20~50。次数越多越准，但计算越慢。
        device: 计算设备
    
    Returns:
        metrics (dict): 包含 'auc', 'acc' 等指标 (基于均值预测计算)
        results (dict): 包含详细数据:
            - 'y_true': 真实标签
            - 'y_mean': 预测概率均值 (最终预测结果)
            - 'y_std':  预测概率标准差 (不确定性度量)
    """
    
    # -----------------------------------------------------------
    # 1. 模式切换：开启Dropout，但冻结BatchNorm (及其他训练特有层)
    # -----------------------------------------------------------
    # 先将模型设为 eval 模式 (冻结 BN, Dropout)
    model.eval()
    
    # 然后只将 Dropout 层设回 train 模式 (开启 Dropout)
    # TF Alignment: 统一采用 Dropout 开启模式
    def set_dropout_train(m):
        if isinstance(m, torch.nn.Dropout):
            m.train()
    model.apply(set_dropout_train)

    all_targets = []
    all_means = []
    all_stds = []
    
    print(f"\n[MC-Dropout] Starting Inference with T={num_samples} samples...")
    start_time = time.time()  # 记录总开始时间
    
    # -----------------------------------------------------------
    # 2. 推理循环
    # -----------------------------------------------------------
    # 禁用梯度计算，节省显存
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            batch_start_time = time.time()  # 记录 Batch 开始时间
            # 数据准备
            if device:
                data_gpu = data.to(device)
            else:
                data_gpu = data
                
            x = data_gpu[:, :, 0].to(torch.long)
            y_target = data_gpu[:, :, 1].to(torch.long)
            mask = data_gpu[:, :, 2].to(torch.bool)
            
            # 处理时间特征 (包含 NaN 清洗逻辑)
            interval_time = data_gpu[:, :, 3].to(torch.float32)
            response_time = data_gpu[:, :, 4].to(torch.float32)
            if torch.isnan(interval_time).any(): interval_time = torch.nan_to_num(interval_time, nan=0.0)
            if torch.isnan(response_time).any(): response_time = torch.nan_to_num(response_time, nan=0.0)

            # 获取当前 Batch 的真实标签 (Masked)
            # 只需要取一次，因为标签在 T 次采样中是不变的
            target_masked = torch.masked_select(y_target, mask).cpu().numpy()
            
            # 如果该 batch 没有有效数据，跳过
            if len(target_masked) == 0:
                continue
            
            all_targets.extend(target_masked)

            # [关键] 重复采样 T 次
            batch_preds_t = [] # 用于存储 T 次的预测结果，形状将在之后转为 [num_samples, num_valid_interactions]
            
            for t in range(num_samples):
                # 前向传播 (此时 Dropout 是开启的，每次结果都会略有不同)
                y_hat = model(x, y_target, mask, interval_time, response_time)
                y_hat_masked = torch.masked_select(y_hat, mask)

                # TF Alignment: 统一采用 Logits -> Sigmoid 转换
                # Logits -> Sigmoid
                y_prob = torch.sigmoid(y_hat_masked)
                
                batch_preds_t.append(y_prob.cpu().numpy())
                
                # 详细进度反馈
                sys.stdout.write(f"\r  Batch {batch_idx+1}/{len(data_loader)}, Sample {t+1}/{num_samples}")
                sys.stdout.flush()

            # -----------------------------------------------------------
            # 3. 计算统计量
            # -----------------------------------------------------------
            # batch_preds_t 形状转化: [num_samples, num_interactions]
            batch_preds_t = np.array(batch_preds_t)
            
            # 均值：作为最终预测 (Bayesian Model Averaging)
            # axis=0 表示沿着采样次数 T 的维度求平均
            mean_pred = np.mean(batch_preds_t, axis=0) # Shape: [num_interactions]
            
            # 标准差：作为不确定性 (Uncertainty)
            std_pred = np.std(batch_preds_t, axis=0)   # Shape: [num_interactions]
            
            all_means.extend(mean_pred)
            all_stds.extend(std_pred)
            
            batch_elapsed = time.time() - batch_start_time
            sys.stdout.write(f"\rBatch {batch_idx+1}/{len(data_loader)} completed ({batch_elapsed:.1f}s)        \n")
            sys.stdout.flush()

    print("\n[MC-Dropout] Inference Completed.")
    
    # -----------------------------------------------------------
    # 4. 指标计算
    # -----------------------------------------------------------
    total_elapsed = time.time() - start_time
    total_hours = int(total_elapsed // 3600)
    total_minutes = int((total_elapsed % 3600) // 60)
    total_seconds = int(total_elapsed % 60)
    time_str = f"{total_hours}h {total_minutes}min {total_seconds}s"
    print(f"[MC-Dropout] Total elapsed time: {time_str}")
    print(f"             Average time per sample: {total_elapsed / (len(all_targets) if len(all_targets) > 0 else 1) * 1000:.2f}ms")
    all_targets = np.array(all_targets)
    all_means = np.array(all_means)
    all_stds = np.array(all_stds)
    
    try:
        auc = roc_auc_score(all_targets, all_means)
    except ValueError:
        auc = 0.0
        
    acc = accuracy_score(all_targets, all_means >= 0.5)
    
    return {
        'auc': auc, 
        'acc': acc
    }, {
        'y_true': all_targets,
        'y_mean': all_means,
        'y_std': all_stds
    }

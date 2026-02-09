import sys
import os
import torch
import numpy as np
from collections import defaultdict
from scipy import sparse
from tqdm import tqdm
import matplotlib.pyplot as plt

'''
(kt) H:\er_gikt>python back/kt/test/validate_diff.py
Loading dataset: assist09-sample_10%...
Computing item difficulties (Common Mode Baseline)...
100%|█| 421/421 [00:00<00:00, 7345.33it/
ack/kt/test/validate_diff.py:102: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torack/kt/test/validate_diff.py:102: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(DEVICE)   
100%|███| 50/50 [07:39<00:00,  9.18s/it] 

======================================== 
  DIFFERENTIAL ANALYSIS REPORT
======================================== 

[1] Correlation between Model Prediction & Item Baseline:
    Pearson Corr: 0.6552
    -> ℹ️ INFO: Balanced mix of difficultty and ability.

[2] Differential Score Distribution (Pred - Base):
count    3524.000000
mean       -0.027617
std         0.274651
min        -0.911940
25%        -0.217904
50%         0.000652
75%         0.169354
max         0.788565
Name: diff, dtype: float64

[3] Positive Surprises (Hard Question, High Ability):
    Found 10 instances (0.28%)
          base      pred      diff  label
1960  0.285714  0.714769  0.429055      0
2226  0.285714  0.703699  0.417985      0
2227  0.250000  0.832392  0.582392      0
2319  0.275862  0.869438  0.593576      1
2454  0.258065  0.950712  0.692648      1

[4] Negative Surprises (Easy Question, Low Ability):
    Found 98 instances (2.78%)
         base      pred      diff  label 
28   0.714286  0.163001 -0.551285      1 
67   0.725000  0.252941 -0.472059      0 
111  0.800000  0.071990 -0.728010      1 
114  0.723404  0.242446 -0.480958      1 
✅ Differential signals detected! Post-hoc strategy is viable.
   Recommendation Strategy:
✅ Differential signals detected! Post-hoc strategy is viable.
✅ Differential signals detected! Post-hoc strategy is viable.
   Recommendation Strategy:
   - Recommend items from Group [4] for remedial learning.
   - Recommend items from Group [3] for confidence boosting.
'''

# 添加父目录到路径以导入模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset import UserDataset
from config import Config, DEVICE
from params import HyperParameters
from util.utils import build_adj_list, gen_gikt_graph

def load_latest_model(model_dir):
    files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
    if not files:
        raise FileNotFoundError(f"No .pt model files found in {model_dir}")
    # Sort by modification time
    files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
    latest_model_path = os.path.join(model_dir, files[0])
    print(f"Loading latest model: {latest_model_path}")
    return torch.load(latest_model_path, map_location=DEVICE)

def compute_item_difficulty(dataset):
    """
    计算训练集中每道题的平均正确率（共模基准）
    """
    print("Computing item difficulties (Common Mode Baseline)...")
    q_correct = defaultdict(int)
    q_total = defaultdict(int)
    
    # 遍历数据集
    # dataset 是 UserDataset，getitem 返回 (seq_len, 5)
    # col 0: q_id, col 1: ans
    for i in tqdm(range(len(dataset))):
        data = dataset[i] # [seq_len, 5]
        q_ids = data[:, 0].long().numpy()
        answers = data[:, 1].long().numpy()
        mask = data[:, 2].bool().numpy()
        
        valid_q = q_ids[mask]
        valid_a = answers[mask]
        
        for q, a in zip(valid_q, valid_a):
            if q != 0: # 排除 padding 0
                q_total[q] += 1
                q_correct[q] += a
                
    difficulty_map = {}
    for q in q_total:
        difficulty_map[q] = q_correct[q] / q_total[q]
        
    print(f"Computed difficulties for {len(difficulty_map)} items.")
    return difficulty_map

def run_differential_analysis():
    # 1. 基础配置
    # Load default experiment config to get dataset_name and other params
    # Assuming running from project root H:\er_gikt
    exp_config_path = "back/kt/config/experiments/exp_full_default.toml"
    if not os.path.exists(exp_config_path):
        # Fallback if running from back/kt
        exp_config_path = "config/experiments/exp_full_default.toml"
    
    params = HyperParameters.load(exp_config_path)
    dataset_name = params.train.dataset_name
    config = Config(dataset_name=dataset_name)
    
    # 2. 加载数据
    print(f"Loading dataset: {dataset_name}...")
    dataset = UserDataset(config)
    
    # 3. 计算共模基准 (Item Difficulty)
    # 为了严谨，应该只用训练集算，但这里为了快速验证，全量算影响不大
    item_diff_map = compute_item_difficulty(dataset)
    
    # 4. 加载模型
    try:
        model = load_latest_model(config.path.MODEL_DIR)
        model.eval()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 5. 运行推理并收集"差分数据"
    print("Running inference to collect differential signals...")
    
    diff_records = [] # Store tuple: (q_id, pred_prob, base_diff, diff_val, ground_truth)
    
    # 只取测试集部分数据进行分析（例如前 50 个用户）或者随机采样
    num_users_to_analyze = 50
    indices = np.random.choice(len(dataset), num_users_to_analyze, replace=False)
    
    with torch.no_grad():
        for idx in tqdm(indices):
            data = dataset[idx]
            # 构造 Batch (batch_size=1)
            data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            x = data_tensor[:, :, 0].to(torch.long)
            y_target = data_tensor[:, :, 1].to(torch.long)
            mask = data_tensor[:, :, 2].to(torch.bool)
            interval = data_tensor[:, :, 3]
            response = data_tensor[:, :, 4]
            
            # Forward
            y_hat = model(x, y_target, mask, interval, response)
            
            # TF Alignment: Always convert Logits to Prob
            y_prob = torch.sigmoid(y_hat)
            
            # 提取有效数据
            eff_mask = mask.squeeze(0).cpu().numpy()
            q_seq = x.squeeze(0).cpu().numpy()[eff_mask]
            true_seq = y_target.squeeze(0).cpu().numpy()[eff_mask]
            pred_seq = y_prob.squeeze(0).cpu().numpy()[eff_mask]
            
            for q, p, t in zip(q_seq, pred_seq, true_seq):
                if q in item_diff_map:
                    base_diff = item_diff_map[q]
                    # Key Metric: Differential Score
                    # Diff > 0: Student is better than average
                    # Diff < 0: Student is worse than average
                    diff_val = p - base_diff
                    
                    diff_records.append({
                        'q_id': q,
                        'pred': p,
                        'base': base_diff,
                        'diff': diff_val,
                        'label': t
                    })

    # 6. 分析结果
    analyze_results(diff_records)

def analyze_results(records):
    import pandas as pd
    df = pd.DataFrame(records)
    
    print("\n" + "="*40)
    print("  DIFFERENTIAL ANALYSIS REPORT")
    print("="*40)
    
    # 1. 相关性分析
    corr = df['pred'].corr(df['base'])
    print(f"\n[1] Correlation between Model Prediction & Item Baseline:")
    print(f"    Pearson Corr: {corr:.4f}")
    if corr > 0.9:
        print("    -> ⚠️ WARNING: Model is very similar to a naive baseline.")
    elif corr < 0.6:
        print("    -> ✅ GOOD: Model captures significant personalized variance.")
    else:
        print("    -> ℹ️ INFO: Balanced mix of difficulty and ability.")

    # 2. 差分分布
    print(f"\n[2] Differential Score Distribution (Pred - Base):")
    print(df['diff'].describe())
    
    # 3. 寻找"正向差模惊喜" (Positive Surprise)
    # 题目很难 (Base < 0.3)，但学生掌握得很好 (Pred > 0.8)
    # 这是单端放大容易忽略的“微弱天才信号”
    pos_surprise = df[(df['base'] < 0.3) & (df['pred'] > 0.7)]
    print(f"\n[3] Positive Surprises (Hard Question, High Ability):")
    print(f"    Found {len(pos_surprise)} instances ({len(pos_surprise)/len(df)*100:.2f}%)")
    if not pos_surprise.empty:
        print(pos_surprise[['base', 'pred', 'diff', 'label']].head(5).to_string())

    # 4. 寻找"负向差模痛点" (Negative Surprise / Weakness)
    # 题目很简单 (Base > 0.7)，但学生掌握得很差 (Pred < 0.3)
    # 这是真正需要补习的地方
    neg_surprise = df[(df['base'] > 0.7) & (df['pred'] < 0.3)]
    print(f"\n[4] Negative Surprises (Easy Question, Low Ability):")
    print(f"    Found {len(neg_surprise)} instances ({len(neg_surprise)/len(df)*100:.2f}%)")
    if not neg_surprise.empty:
        print(neg_surprise[['base', 'pred', 'diff', 'label']].head(5).to_string())
        
    print("\n" + "="*40)
    print("CONCLUSION:")
    if len(pos_surprise) > 0 or len(neg_surprise) > 0:
        print("✅ Differential signals detected! Post-hoc strategy is viable.")
        print("   Recommendation Strategy:")
        print("   - Recommend items from Group [4] for remedial learning.")
        print("   - Recommend items from Group [3] for confidence boosting.")
    else:
        print("⚠️ Differential signals are weak. Model might need retraining with explicit differential structure.")

if __name__ == "__main__":
    run_differential_analysis()

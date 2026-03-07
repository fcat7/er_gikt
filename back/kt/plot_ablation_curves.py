import os
import pandas as pd
import matplotlib.pyplot as plt
import re
import glob

def parse_log_file(log_path):
    epochs = []
    train_losses = []
    val_aucs = []

    if not os.path.exists(log_path):
        print(f"找不到日志文件: {log_path}")
        return epochs, train_losses, val_aucs
    
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    current_epoch = -1
    for line in lines:
        # Match training line
        train_match = re.search(r'Epoch\s+(\d+)\s*\|\s*training:\s*loss:\s*([0-9\.]+).*?auc:\s*([0-9\.]+)', line)
        if train_match:
            current_epoch = int(train_match.group(1))
            train_losses.append(float(train_match.group(2)))
            epochs.append(current_epoch)
            
        # Match validation line
        val_match = re.search(r'\|\s*validate:\s*loss:\s*([0-9\.]+).*?auc:\s*([0-9\.]+)', line)
        if val_match and current_epoch != -1:
            # We assume it immediately follows the train line
            val_aucs.append(float(val_match.group(2)))
            current_epoch = -1
            
    # Safeguard if length mismatches
    min_len = min(len(epochs), len(train_losses), len(val_aucs))
    return epochs[:min_len], train_losses[:min_len], val_aucs[:min_len]

def main():
    search_path = "output/**/ablation_summary.csv"
    csv_files = glob.glob(search_path, recursive=True)
    
    if not csv_files:
        print("未找到 ablation_summary.csv")
        return
        
    latest_csv = max(csv_files, key=os.path.getmtime)
    print(f"读取汇总文件: {latest_csv}")
    df = pd.read_csv(latest_csv)
    
    # 获取最后5条记录（假设一次跑5个变体）
    # 或者根据同一次运行的日期筛选？这里简单取最后的 N 个唯一组
    # 按照实际你的消融实验组数
    N_GROUPS = 7
    recent_runs = df.tail(N_GROUPS)
    
    plt.figure(figsize=(14, 6))
    
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    
    plt.subplot(1, 2, 1)
    plt.title("Training Loss across Ablations", fontsize=14)
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    
    plt.subplot(1, 2, 2)
    plt.title("Validation AUC across Ablations", fontsize=14)
    plt.xlabel("Epoch")
    plt.ylabel("Val AUC")

    for idx, row in recent_runs.iterrows():
        group_name = row['Ablation Group']
        date_str = row['Date']
        
        # 构建日志文件路径
        log_dir = os.path.dirname(latest_csv)
        log_path = os.path.join(log_dir, f"{date_str}.log")
        
        epochs, train_losses, val_aucs = parse_log_file(log_path)
        
        if len(epochs) > 0:
            c = colors[idx % len(colors)]
            plt.subplot(1, 2, 1)
            plt.plot(epochs, train_losses, marker='o', markersize=3, label=group_name, color=c)
            
            plt.subplot(1, 2, 2)
            plt.plot(epochs, val_aucs, marker='s', markersize=3, label=group_name, color=c)

    plt.subplot(1, 2, 1)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.subplot(1, 2, 2)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(latest_csv), "ablation_curves_diagnosis.png")
    plt.savefig(save_path, dpi=300)
    print(f"✅ 诊断曲线已保存至: {save_path}")

if __name__ == "__main__":
    main()

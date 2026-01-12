import sys
import re
import numpy as np
import os

def recover_data(log_file_path):
    """
    从日志文件中恢复 _aver.txt 数据 (Test Loss, Test Acc, Test AUC)
    """
    if not os.path.exists(log_file_path):
        print(f"Error: File {log_file_path} not found.")
        return

    test_losses = []
    test_accs = []
    test_aucs = []
    epoch_indices = []

    # 匹配模式：
    # output_file.write(f"epoch: {epoch_total} | ")
    # output_file.write(f'training: ... | ')
    # output_file.write(f'testing: loss: {test_loss_aver:.4f}, acc: {test_acc_aver:.4f}, auc: {test_auc_aver: .4f}\n')
    # 日志文件中通常为一行:
    # epoch: 1 | training: loss: 0.6720, acc: 0.6559, auc: 0.6721 | testing: loss: 0.6277, acc: 0.6387, auc: 0.6504
    
    pattern = re.compile(r'epoch:\s*(\d+).*?testing:\s*loss:\s*([\d\.]+),\s*acc:\s*([\d\.]+),\s*auc:\s*([\d\.]+)')

    print(f"Reading log file: {log_file_path}")
    with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                acc = float(match.group(3))
                auc = float(match.group(4))
                
                epoch_indices.append(epoch)
                test_losses.append(loss)
                test_accs.append(acc)
                test_aucs.append(auc)

    if not test_losses:
        print("No valid epoch data found in the log file.")
        print("Expected format: 'epoch: 1 | ... testing: loss: 0.123, acc: 0.456, auc: 0.789'")
        return

    # 构建 numpy 数组 (3, N)
    # Row 0: Loss, Row 1: Acc, Row 2: AUC
    data_array = np.array([test_losses, test_accs, test_aucs])
    
    print(f"Found {len(test_losses)} epochs (Epochs: {epoch_indices}).")
    print("-" * 30)
    print("Recovered Data Preview:")
    print(f"Test Loss (First 5): {test_losses[:5]}")
    print(f"Test Acc  (First 5): {test_accs[:5]}")
    print(f"Test AUC  (First 5): {test_aucs[:5]}")
    print("-" * 30)

    # 生成输出文件路径
    # 假设 log 位于 kt/output/xxx.log，我们希望生成到 log 同级目录或 chart_data
    base_name = os.path.splitext(os.path.basename(log_file_path))[0]
    # 假设 log 文件在 kt/output/xxx.log，我们希望输出到 kt/chart_data/xxx_aver.txt
    
    dir_name = os.path.join(os.path.dirname(os.path.dirname(log_file_path)), "chart_data")
    
    
    output_path = os.path.join(dir_name, f"{base_name}_aver.txt")
    
    np.savetxt(output_path, data_array)
    print(f"Successfully saved recovered data to: {output_path}")
    print("You can verify the format matches your existing _aver.txt files.")

if __name__ == "__main__":
    recover_data("../output/log/20260108_1759.log")

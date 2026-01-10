import os
import numpy as np
import matplotlib.pyplot as plt

def plot_comparison(logs_dict, metric_type='auc', base_dir='chart_data'):
    """
    绘制多个日志文件的单一指标对比折线图，并输出原始数据表格。
    
    Args:
        logs_dict (dict): 日志配置字典，格式 {'filename.txt': 'Custom Label'}
        metric_type (str): 需要对比的指标类型，可选 'loss', 'acc', 'auc'。
        base_dir (str): 日志文件所在的文件夹路径，默认为 'chart_data'
    """
    # 指标名称到数据行索引的映射
    # 0: Loss, 1: Acc, 2: AUC
    metric_map = {
        'loss': 0, 
        'acc': 1, 
        'auc': 2
    }
    
    metric_key = metric_type.lower()
    if metric_key not in metric_map:
        print(f"Error: Metric '{metric_type}' is not supported. Supported metrics: {list(metric_map.keys())}")
        return

    row_idx = metric_map[metric_key]
    
    # 创建画布
    plt.figure(figsize=(10, 6))
    
    # 尝试设置中文字体，防止乱码
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial'] 
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass

    colors = plt.cm.tab10.colors # 使用默认色板

    # 设置图表标题和标签
    plt.title(f'{metric_key.upper()} Comparison across Logs')
    plt.xlabel('Epochs')
    plt.ylabel(metric_key.upper())
    plt.grid(True, linestyle='--', alpha=0.6)

    # 存储用于打印表格的数据
    # {label: [value_epoch1, value_epoch2, ...]}
    table_data = {}
    max_epochs = 0
    
    # 遍历每个日志文件进行绘制
    for i, (log_name, label) in enumerate(logs_dict.items()):
        file_path = os.path.join(base_dir, log_name)
        
        # 如果传入的是绝对路径，直接使用
        if os.path.isabs(log_name):
            file_path = log_name
        
        if not os.path.exists(file_path):
            # 尝试在当前目录下查找
            if os.path.exists(log_name):
                file_path = log_name
            else:
                print(f"Warning: File not found: {file_path}")
                continue
            
        try:
            # 加载数据: 3行 x Epochs列 (Loss, Acc, Auc)
            data = np.loadtxt(file_path)
            
            # 检查数据维度
            if data.ndim != 2 or data.shape[0] < 3:
                print(f"Warning: Invalid data format in {log_name}. Expected shape (3, epochs).")
                continue
                
            epochs = range(1, data.shape[1] + 1)
            
            # 获取指定指标的数据行
            y_values = data[row_idx, :]
            
            # @change: 自动提取文件名中的时间部分作为前缀
            # 假设文件名格式: YYYYMMDD_HHMM_aver.txt
            time_prefix = ""
            parts = os.path.basename(log_name).split('_')
            if len(parts) >= 2 and parts[1].isdigit():
                time_prefix = f"[{parts[1]}] "
            
            display_label = f"{time_prefix}{label}"
            
            # 存储数据用于后续打印表格
            table_data[display_label] = y_values
            if len(y_values) > max_epochs:
                max_epochs = len(y_values)

            color = colors[i % len(colors)]
            plt.plot(epochs, y_values, label=display_label, marker='.', color=color, linewidth=2)
            
        except Exception as e:
            print(f"Error reading {log_name}: {e}")

    plt.legend()
    plt.tight_layout()
    plt.show() # 显示图表

    # --- 打印数据表格 ---
    print(f"\n{'='*25} Data Table ({metric_key.upper()}) {'='*25}")
    
    # 表头: Epoch | Label1 | Label2 ...
    labels = list(table_data.keys())
    # 计算每列的宽度，确保对齐
    col_width = max([len(l) for l in labels] + [10]) + 2 
    
    # 打印表头
    header = f"{'Epoch':<8} | " + " | ".join([f"{l:<{col_width}}" for l in labels])
    print(header)
    print("-" * len(header))
    
    # 打印每一行数据
    for epoch in range(1, max_epochs + 1):
        row_str = f"{epoch:<8} | "
        for label in labels:
            values = table_data[label]
            if epoch <= len(values):
                val = values[epoch-1]
                row_str += f"{val:<{col_width}.4f} | "
            else:
                row_str += f"{'-':<{col_width}} | " # 如果该Epoch没有数据
        print(row_str.rstrip(" | "))
    print(f"{'='*65}\n")

if __name__ == "__main__":
    # 示例用法
    # 获取项目根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    chart_data_dir = os.path.join(current_dir, 'chart_data')
    
    # 只需要配置 logs_to_compare_dict 即可
    # 键为文件名，值为图例显示的 Label
    logs_to_compare_dict = {
        '20260108_0257_aver.txt': 'use_cognitive_model & no_BCELoss',
        '20260108_0308_aver.txt': 'no_cognitive_model & no_BCELoss',
        '20260108_1332_aver.txt': 'use_congnitive_model & use_BCELoss',
        # '20260108_1430_aver.txt': 'use_cognitive_model & no_BCELoss',
        '20260108_1537_aver.txt': 'no_cognitive_model & use_BCELoss'
    }

    # logs_to_compare_dict = { # 全量数据集 assist09
    #     '20260108_1209_aver.txt': 'no_cognitive_model',
    #     '20260108_1207_aver.txt': 'use_cognitive_model',
    # }
    
    # logs_to_compare_dict = { # 全量数据集 assist09
    #     '20260108_1759_aver.txt': 'use_pretrain',
    #     '20260108_1809_aver.txt': 'no_pretrain',
    # }
    
    # logs_to_compare_dict = { # 预训练的比较测试
    #     '20260108_1430_aver.txt': 'no_pretrain',
    #     '20260108_1657_aver.txt': 'use_pretrain(batch_size = 32, epoch = 100)',
    #     '20260108_1710_aver.txt': 'use_pretrain(batch_size = 256, epoch = 50)',
    #     '20260108_1723_aver.txt': 'use_pretrain(batch_size = 128, epoch = 50)',
    #     '20260108_1724_aver.txt': 'use_pretrain(batch_size = 64, epoch = 50)',
    #     '20260108_1726_aver.txt': 'use_pretrain(batch_size = 32, epoch = 50)'
    # }
    
    # logs_to_compare_dict = { # gat 和 gcn 的比较测试
    #     '20260109_0037_aver.txt': 'gat(new)',
    #     '20260109_0038_aver.txt': 'gcn',
    # }
    
    # logs_to_compare_dict = { # 全量数据集 assist09
    #     '20260109_0053_aver.txt': 'gcn',
    #     '20260109_0057_aver.txt': 'gat(new)',
    # }
    
    if logs_to_compare_dict:
        plot_comparison(logs_to_compare_dict, metric_type='auc', base_dir=chart_data_dir)
    else:
        print("请在 logs_to_compare_dict 中添加文件名以运行示例。")

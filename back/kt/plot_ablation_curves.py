import os
import argparse
import glob
import re

import pandas as pd
import matplotlib.pyplot as plt

JOURNAL_PALETTE = [
    '#4E79A7',  # blue
    '#F28E2B',  # orange
    '#59A14F',  # green
    '#E15759',  # red
    '#76B7B2',  # teal
    '#B07AA1',  # purple
    '#9C755F',  # brown
    '#BAB0AC',  # gray
    '#EDC948',  # yellow
    '#FF9DA7',  # pink
]

MARKERS = ['o', 's', '^', 'D', 'P', 'X', 'v', '<', '>', 'h']


def setup_journal_plot_style():
    """TKDE / C&E / TLT 友好的简洁绘图风格。"""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'SimHei', 'Microsoft YaHei'],
        'axes.unicode_minus': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10.5,
        'ytick.labelsize': 10.5,
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })


def parse_log_file(log_path):
    epochs = []
    train_losses = []
    train_aucs = []
    val_losses = []
    val_aucs = []

    if not os.path.exists(log_path):
        print(f"找不到日志文件: {log_path}")
        return epochs, train_losses, train_aucs, val_losses, val_aucs
    
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    current_epoch = -1
    for line in lines:
        # Match training line
        train_match = re.search(r'Epoch\s+(\d+)\s*\|\s*training:\s*loss:\s*([0-9\.]+).*?auc:\s*([0-9\.]+)', line)
        if train_match:
            current_epoch = int(train_match.group(1))
            train_losses.append(float(train_match.group(2)))
            train_aucs.append(float(train_match.group(3)))
            epochs.append(current_epoch)
            
        # Match validation line
        val_match = re.search(r'\|\s*validate:\s*loss:\s*([0-9\.]+).*?auc:\s*([0-9\.]+)', line)
        if val_match and current_epoch != -1:
            # We assume it immediately follows the train line
            val_losses.append(float(val_match.group(1)))
            val_aucs.append(float(val_match.group(2)))
            current_epoch = -1
            
    # Safeguard if length mismatches
    min_len = min(len(epochs), len(train_losses), len(train_aucs), len(val_losses), len(val_aucs))
    return epochs[:min_len], train_losses[:min_len], train_aucs[:min_len], val_losses[:min_len], val_aucs[:min_len]


def normalize_text(value):
    if pd.isna(value):
        return ''
    return str(value).strip()


def load_summary_csv(csv_path):
    """兼容历史日志中常见的 UTF-8 / GBK / Latin-1 编码。"""
    last_error = None
    for encoding in ('utf-8-sig', 'utf-8', 'gbk', 'gb18030', 'latin1'):
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            print(f'使用编码读取成功: {encoding}')
            return df
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
    raise last_error


def find_log_file(log_dir, date_str):
    """优先精确匹配，其次按日期前缀模糊匹配。"""
    exact_path = os.path.join(log_dir, f'{date_str}.log')
    if os.path.exists(exact_path):
        return exact_path

    pattern = os.path.join(log_dir, f'{date_str}*.log')
    candidates = glob.glob(pattern)
    if not candidates:
        return None

    candidates.sort(key=lambda p: (len(os.path.basename(p)), os.path.getmtime(p)))
    return candidates[0]


def select_runs(df, experiments_name=None, dataset_name=None, top_n=None):
    df = df.copy()
    if 'experiments_name' in df.columns:
        df['experiments_name'] = df['experiments_name'].map(normalize_text)
    else:
        df['experiments_name'] = ''

    if 'Dataset' in df.columns:
        df['Dataset'] = df['Dataset'].map(normalize_text)

    if experiments_name:
        target = normalize_text(experiments_name)
        exact_df = df[df['experiments_name'] == target]
        if not exact_df.empty:
            df = exact_df
        else:
            df = df[df['experiments_name'].str.contains(target, regex=False, na=False)]

    if dataset_name:
        df = df[df['Dataset'] == normalize_text(dataset_name)]

    if df.empty:
        return df

    df = df.sort_values('Date')

    if top_n is not None and top_n > 0:
        df = df.tail(top_n)

    return df


def infer_default_selection(df):
    """若用户未指定 experiments_name，则优先选择最新的非空实验组；否则退化为最新数据集。"""
    work_df = df.copy()
    work_df['experiments_name'] = work_df.get('experiments_name', '').map(normalize_text)
    work_df['Dataset'] = work_df.get('Dataset', '').map(normalize_text)
    work_df = work_df.sort_values('Date')

    non_empty = work_df[work_df['experiments_name'] != '']
    if not non_empty.empty:
        latest_row = non_empty.iloc[-1]
        latest_exp = latest_row['experiments_name']
        latest_dataset = latest_row['Dataset']
        selected = work_df[
            (work_df['experiments_name'] == latest_exp) &
            (work_df['Dataset'] == latest_dataset)
        ].sort_values('Date')
        return selected, latest_exp, latest_dataset

    latest_dataset = work_df.iloc[-1]['Dataset']
    selected = work_df[work_df['Dataset'] == latest_dataset].sort_values('Date')
    return selected, None, latest_dataset


def format_group_name(name):
    name = normalize_text(name)
    replacements = {
        'A_Baseline': 'FA-GIKT',
        'B_Remove_PID': 'w/o PID',
        'C_Remove_Cognitive': 'w/o Cognitive',
        'D_Remove_IRT_4PL': 'w/o IRT',
        'E_agg_method: gcn': 'GCN Aggregation',
        'E_agg_method-kk_gat': 'KK-GAT Aggregation',
        'F_old_gikt': 'Old GIKT',
        'G_remove_PID_cognitive': 'add_cognitive',
        'G_Remove_Cognitive_and_IRT': 'add_pid',
        'H_remove_PID_4pl': 'add_cognitive',
        'H_Remove_PID_and_Cognitive': 'add_irt',
        'I_remove_cognitive_4pl': 'add_pid',
        'I_Remove_PID_and_IRT': 'add_cognitive',
    }
    # Direct match first
    for old, new in replacements.items():
        if name.lower() == old.lower():
            return new
            
    if name in replacements:
        return replacements[name]
        
    # If not perfectly matched, try to replace common terms automatically
    processed = name
    processed = re.sub(r'(?i)add_cognitive', '+ Cognitive', processed)
    processed = re.sub(r'(?i)add_pid', '+ PID', processed)
    processed = re.sub(r'(?i)add_(?:irt|4pl)', '+ IRT', processed)
    processed = re.sub(r'(?i)^add\s+(.*)', r'+ \1', processed)
    
    processed = re.sub(r'(?i)(?:remove_|no_)pid', 'w/o PID', processed)
    processed = re.sub(r'(?i)(?:remove_|no_)cognitive', 'w/o Cognitive', processed)
    processed = re.sub(r'(?i)(?:remove_|no_)(?:irt|4pl)', 'w/o IRT', processed)
    
    if processed != name:
        return processed.replace('_', ' ').strip()
        
    return name.replace('_', ' ')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary_csv', type=str, default=None, help='ablation_summary.csv 路径；默认自动查找最新文件')
    parser.add_argument('--experiments_name', type=str, default=None, help='按 experiments_name 精确筛选')
    parser.add_argument('--dataset_name', type=str, default=None, help='按数据集筛选，例如 assist09 / assist12')
    parser.add_argument('--top_n', type=int, default=None, help='在筛选结果中仅保留最后 N 条')
    parser.add_argument('--annotate_peak', action='store_true', default=False, help='用此开关控制是否在验证集曲线上高亮最佳 AUC 点')
    args = parser.parse_args()

    # ==========================================
    # 1. 在此手动配置你要绘制的实验组组合 (图例名称必须和 format_group_name 返回值一致)
    #    字典的 key (如 "Group1_Ablation") 会附加到输出文件名上
    # ==========================================
    MANUAL_GROUPS = {
        "Group1_Ablation": [
            'FA-GIKT', 
            'w/o PID', 
            'w/o Cognitive', 
            'w/o IRT', 
            'Old GIKT'
        ],
        "Group2_Addition": [
            'Old GIKT', 
            '+ Cognitive', 
            '+ PID', 
            '+ IRT', 
            'FA-GIKT'
        ]
    }

    setup_journal_plot_style()

    search_path = "output/**/ablation_summary.csv"
    csv_files = glob.glob(search_path, recursive=True)
    
    if not csv_files:
        print("未找到 ablation_summary.csv")
        return
        
    latest_csv = args.summary_csv if args.summary_csv else max(csv_files, key=os.path.getmtime)
    print(f"读取汇总文件: {latest_csv}")
    df = load_summary_csv(latest_csv)

    if args.experiments_name or args.dataset_name or args.top_n:
        selected_runs = select_runs(
            df,
            experiments_name=args.experiments_name,
            dataset_name=args.dataset_name,
            top_n=args.top_n,
        )
    else:
        selected_runs, inferred_exp, inferred_dataset = infer_default_selection(df)
        print('未命中显式筛选条件，已自动切换为最近一组可用实验：')
        if inferred_exp:
            print(f'  - experiments_name = {inferred_exp}')
        if inferred_dataset:
            print(f'  - dataset_name = {inferred_dataset}')

    if selected_runs.empty:
        selected_runs, inferred_exp, inferred_dataset = infer_default_selection(df)
        print('显式筛选后无结果，已回退到最近一组可用实验：')
        if inferred_exp:
            print(f'  - experiments_name = {inferred_exp}')
        if inferred_dataset:
            print(f'  - dataset_name = {inferred_dataset}')

    if selected_runs.empty:
        print('没有可绘制的消融记录。')
        return

    log_dir = os.path.dirname(latest_csv)
    
    def get_run_by_label(df_runs, target_lab):
        """倒序查找最新匹配给定 label 的记录"""
        for i in range(len(df_runs)-1, -1, -1):
            row = df_runs.iloc[i]
            if format_group_name(row['Ablation Group']) == target_lab:
                return row
        return None

    # 遍历手动指定的组合进行绘图
    for group_key, target_labels in MANUAL_GROUPS.items():
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

        axes[0].set_title('Ablation Training Loss', pad=10)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Training Loss')

        axes[1].set_title('Ablation Training AUC', pad=10)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Training AUC')

        axes[2].set_title('Ablation Validation AUC', pad=10)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Validation AUC')

        plotted = 0
        for plot_idx, target_label in enumerate(target_labels):
            row = get_run_by_label(selected_runs, target_label)
            if row is None:
                print(f'[{group_key}] 警告：未找到图例为 "{target_label}" 的日志记录，跳过此线。')
                continue

            date_str = normalize_text(row['Date'])
            log_path = find_log_file(log_dir, date_str)

            if not log_path:
                print(f'未找到对应日志，已跳过: {date_str}')
                continue

            epochs, train_losses, train_aucs, val_losses, val_aucs = parse_log_file(log_path)
            if not epochs:
                print(f'日志中未解析到有效 epoch 曲线，已跳过: {os.path.basename(log_path)}')
                continue

            color = JOURNAL_PALETTE[plot_idx % len(JOURNAL_PALETTE)]
            marker = MARKERS[plot_idx % len(MARKERS)]
            label = target_label

            # 更高颜值的连线设置 (参考顶级期刊累积图质感：加宽，带白边粗锚点)
            plot_kwargs = dict(
                color=color,
                linewidth=2.5,
                marker=marker,
                markersize=8,
                markeredgecolor='white',
                markeredgewidth=1.2,
                markevery=max(1, len(epochs) // 10),
                alpha=0.9,
                label=label,
                zorder=10 - plot_idx
            )

            axes[0].plot(epochs, train_losses, **plot_kwargs)
            axes[1].plot(epochs, train_aucs, **plot_kwargs)
            axes[2].plot(epochs, val_aucs, **plot_kwargs)
            
            # 手动控制是否标注最高 AUC 点
            if args.annotate_peak and val_aucs:
                best_idx = val_aucs.index(max(val_aucs))
                best_epoch = epochs[best_idx]
                best_val = val_aucs[best_idx]
                
                axes[2].scatter([best_epoch], [best_val], color=color, marker='*', s=250, 
                                edgecolors='black', linewidths=0.5, zorder=20)
                axes[2].annotate(f'{best_val:.4f}', (best_epoch, best_val),
                                 textcoords="offset points", xytext=(0, 12), ha='center',
                                 fontsize=10, color='black', fontweight='bold',
                                 bbox=dict(boxstyle="round,pad=0.2", fc=color, ec="none", alpha=0.3))

            plotted += 1

        if plotted == 0:
            print(f'分组 {group_key} 都未成功解析出曲线，跳过绘图。')
            plt.close(fig)
            continue

        for ax in axes:
            ax.grid(True, linestyle='--', linewidth=1.0, alpha=0.3)
            ax.set_facecolor('white')

        # 图例：带轻微边框的半透明质感，确保不覆盖曲线底色
        axes[0].legend(loc='upper right', frameon=True, edgecolor='#dddddd', facecolor='white', framealpha=0.9, ncol=1)
        axes[1].legend(loc='lower right', frameon=True, edgecolor='#dddddd', facecolor='white', framealpha=0.9, ncol=1)
        axes[2].legend(loc='lower right', frameon=True, edgecolor='#dddddd', facecolor='white', framealpha=0.9, ncol=1)

        exp_tag = normalize_text(args.experiments_name) or normalize_text(selected_runs.iloc[-1].get('experiments_name', ''))
        dataset_tag = normalize_text(args.dataset_name) or normalize_text(selected_runs.iloc[-1].get('Dataset', ''))
        name_parts = ['ablation_curves']
        if dataset_tag:
            name_parts.append(dataset_tag)
        if exp_tag:
            safe_exp_tag = re.sub(r'[^0-9A-Za-z_\-\u4e00-\u9fa5]+', '_', exp_tag)
            name_parts.append(safe_exp_tag)
            
        # 把当前的自定义分组名加到文件名里
        name_parts.append(group_key)
            
        save_stem = '_'.join([part for part in name_parts if part])

        save_path_png = os.path.join(log_dir, f'{save_stem}.png')
        save_path_pdf = os.path.join(log_dir, f'{save_stem}.pdf')
        plt.tight_layout()
        plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(save_path_pdf, bbox_inches='tight')
        print(f'✅ 自定义分组 [{group_key}] 诊断曲线已保存：\n  - PNG: {save_path_png}\n  - PDF: {save_path_pdf}')
        plt.close(fig)

if __name__ == "__main__":
    main()

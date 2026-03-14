import os
import sys
import argparse
from datetime import datetime

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import Config

def setup_academic_plot_style():
    """顶刊绘图预设"""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.sans-serif': ['Times New Roman'],
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.dpi': 300
    })

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='assist09-min20')
    parser.add_argument('--checkpoint', type=str, default='checkpoint-min20/gikt_assist09_min20.pt')
    args = parser.parse_args()

    setup_academic_plot_style()
    config = Config(dataset_name=args.dataset_name)

    print("📦 1. 提取 Guessing (c) 与 Slipping (d) 参数...")
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if isinstance(ckpt, torch.nn.Module):
        state_dict = ckpt.state_dict()
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt
    
    try:
        c_weights = state_dict['guessing_bias.weight'].squeeze().numpy()
        d_weights = state_dict['slipping_bias.weight'].squeeze().numpy()
    except KeyError:
        raise ValueError("未找到 guessing_bias / slipping_bias，请确认模型是否包含 4PL-IRT！")
    
    # 转换为概率空间
    c_real = sigmoid(c_weights)
    d_real = sigmoid(d_weights)
    
    df_params = pd.DataFrame({
        'Question_ID': np.arange(len(c_real)),
        'Guessing Rate (c)': c_real,
        'Slipping Rate (d)': d_real
    })
    
    timestamp = datetime.now().strftime("%H%M%S")
    csv_path = os.path.join(config.path.OUTPUT_DIR, f"scheme4_guess_slip_params_{timestamp}.csv")
    df_params.to_csv(csv_path, index=False)
    print(f"💾 c, d 概率参数已保存至: {csv_path}")
    print(f"  > Guessing: min={c_real.min():.4f}, mean={c_real.mean():.4f}, max={c_real.max():.4f} (理论上限 0.2)")
    print(f"  > Slipping: min={d_real.min():.4f}, mean={d_real.mean():.4f}, max={d_real.max():.4f} (理论上限 0.05)")
    
    print("🎨 2. 绘制参数的 KDE 分布图...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 子图1：Guessing Rate
    sns.histplot(data=df_params, x='Guessing Rate (c)', kde=True, color="#4DBBD5", 
                 bins=40, ax=axes[0], edgecolor="white", alpha=0.6)
    axes[0].set_title('Distribution of True Guessing Rate ($c$)', pad=10)
    axes[0].set_xlabel('Guessing Probability')
    axes[0].set_ylabel('Density / Count')
    axes[0].axvline(df_params['Guessing Rate (c)'].mean(), color='r', linestyle='--', label=f"Mean: {df_params['Guessing Rate (c)'].mean():.3f}")
    axes[0].legend()
    
    # 子图2：Slipping Rate
    sns.histplot(data=df_params, x='Slipping Rate (d)', kde=True, color="#E64B35", 
                 bins=40, ax=axes[1], edgecolor="white", alpha=0.6)
    axes[1].set_title('Distribution of Careless Slipping Rate ($d$)', pad=10)
    axes[1].set_xlabel('Slipping Probability')
    axes[1].set_ylabel('Density / Count')
    axes[1].axvline(df_params['Slipping Rate (d)'].mean(), color='b', linestyle='--', label=f"Mean: {df_params['Slipping Rate (d)'].mean():.3f}")
    axes[1].legend()
    
    sns.despine(fig=fig)
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%H%M%S")
    out_path = os.path.join(config.path.OUTPUT_DIR, f"scheme4_guessing_slipping_dist_{timestamp}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图四 生成完毕：{out_path}")

if __name__ == '__main__':
    main()
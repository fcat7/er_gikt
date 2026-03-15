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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

def setup_academic_plot_style():
    """顶刊绘图预设"""
    plt.rcParams.update({
        'font.family': ['sans-serif'],
        'font.sans-serif': ['HarmonyOS Sans SC', 'SimHei', 'Arial', 'Times New Roman'],
        'axes.unicode_minus': False,
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

def softplus(x):
    return np.log(1 + np.exp(x))

def calc_prob(theta, a, b, c, d):
    # P(theta) = c + (1 - c - d) * sigmoid(a * (theta - b))
    return c + (1 - c - d) * sigmoid(a * (theta - b))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='assist09-min20')
    parser.add_argument('--checkpoint', type=str, default='../checkpoint/gikt_assist09_for_irt_猜测率设置0.05.pt')
    args = parser.parse_args()

    setup_academic_plot_style()
    config = Config(dataset_name=args.dataset_name)

    print("📦 1. 提取 4PL-IRT 模型参数 a, b, c, d...")
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if isinstance(ckpt, torch.nn.Module):
        state_dict = ckpt.state_dict()
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt
    
    try:
        b_weights = state_dict['difficulty_bias.weight'].squeeze().numpy()
        a_weights = state_dict['discrimination_bias.weight'].squeeze().numpy()
        c_weights = state_dict['guessing_bias.weight'].squeeze().numpy()
        d_weights = state_dict['slipping_bias.weight'].squeeze().numpy()
    except KeyError:
        raise ValueError("模型中未找到完整的 4PL IRT 权重，请检查！")
    
    # 根据模型定义进行反变换还原真实物理意义值(严格对齐 gikt.py 的 1283-1287 行)
    a_real = 1.0 + softplus(a_weights)
    b_real = b_weights
    c_real = 0.2 * sigmoid(c_weights)
    d_real = 0.05 * sigmoid(d_weights)
    
    print("🎯 2. 选择代表性题目绘制 ICC 曲线...")
    # 找几个极端的典型例子
    idx_high_a = np.argmax(a_real)                 # 区分度极高
    idx_high_b = np.argmax(b_real)                 # 极难
    idx_low_b = np.argmin(b_real)                  # 极简单
    idx_high_c = np.argmax(c_real)                 # 高猜测率
    
    selected_items = [
        {'id': idx_low_b, 'label': '高通过率试题 (低 $b$)', 'color': '#00A087', 'ls': '-'},
        {'id': idx_high_b, 'label': '低通过率试题 (高 $b$)', 'color': '#E64B35', 'ls': '-'},
        {'id': idx_high_a, 'label': '高区分度试题 (高 $a$)', 'color': '#3C5488', 'ls': '--'},
        {'id': idx_high_c, 'label': '高猜测率试题 (高 $c$)', 'color': '#F39B7F', 'ls': ':'}
    ]
    
    theta_range = np.linspace(-4, 4, 200)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for item in selected_items:
        q_id = item['id']
        a, b, c, d = a_real[q_id], b_real[q_id], c_real[q_id], d_real[q_id]
        p_val = calc_prob(theta_range, a, b, c, d)
        
        ax.plot(theta_range, p_val, label=f"{item['label']}\n$a$ = {a:.2f},$b$ = {b:.2f},$c$ = {c:.2f},$d$ = {d:.2f}",
                color=item['color'], linestyle=item['ls'], linewidth=2.5)

    # 绘制参考线
    ax.axhline(0.5, color='gray', linestyle='-.', alpha=0.5)
    ax.axvline(0.0, color='gray', linestyle='-.', alpha=0.5)

    # 样式修饰
    ax.set_title('典型表征试题的 ICC (项目特征曲线) 解析', pad=15)
    ax.set_xlabel('学生认知状态隐变量 $\\theta$ (标准化能力域)')
    ax.set_ylabel('响应正确概率 $P(\\theta)$')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-4.1, 4.1)
    
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True, framealpha=0.9, fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    sns.despine(ax=ax, trim=False)
    plt.tight_layout()

    # == 新增：保存参数到 CSV ==
    import copy
    csv_rows = []
    for item in selected_items:
        q_id = item['id']
        row = copy.deepcopy(item)
        row['a'] = float(a_real[q_id])
        row['b'] = float(b_real[q_id])
        row['c'] = float(c_real[q_id])
        row['d'] = float(d_real[q_id])
        csv_rows.append(row)
    
    df_icc = pd.DataFrame(csv_rows)
    out_dir = os.path.dirname(os.path.abspath(__file__))
    # timestamp = datetime.now().strftime("%H%M%S")
    timestamp = ''
    csv_path = os.path.join(out_dir, f"02_核心数据_ICC试题参数_{timestamp}.csv")
    df_icc[['id', 'label', 'a', 'b', 'c', 'd']].to_csv(csv_path, index=False)
    print(f"💾 ICC 参数已保存至: {csv_path}")

    out_path_png = os.path.join(out_dir, f"02_核心表征_试题ICC曲线_{timestamp}.png")
    out_path_pdf = os.path.join(out_dir, f"02_核心表征_试题ICC曲线_{timestamp}.pdf")
    plt.savefig(out_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_path_pdf, bbox_inches='tight')
    print(f"✅ 图二 ICC曲线 生成完毕：\n  - PNG: {out_path_png}\n  - PDF: {out_path_pdf}")

if __name__ == '__main__':
    main()

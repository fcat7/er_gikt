import os
import sys
import json
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def setup_academic_plot_style():
    """顶刊风格绘图预设。"""
    plt.rcParams.update({
        'font.family': ['sans-serif'],
        'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'Times New Roman'],
        'axes.unicode_minus': False,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.labelsize': 13,
        'axes.titlesize': 15,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.dpi': 300,
    })


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def load_state_dict(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, torch.nn.Module):
        return ckpt.state_dict()
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        return ckpt['state_dict']
    return ckpt


def build_type_param_df(checkpoint_path, map_path, raw_path):
    state_dict = load_state_dict(checkpoint_path)

    try:
        c_weights = state_dict['guessing_bias.weight'].squeeze().numpy()
        d_weights = state_dict['slipping_bias.weight'].squeeze().numpy()
    except KeyError as exc:
        raise ValueError('未找到 guessing_bias.weight 或 slipping_bias.weight，请确认模型包含 4PL-IRT。') from exc

    # 转换为概率空间 (适配全新的 Shifted Sigmoid 零均值架构)
    c_max = 0.50
    d_max = 0.20
    guessing_prob_init = 0.05
    slipping_prob_init = 0.02
    
    safe_guess_init = min(max(guessing_prob_init, 1e-5), c_max - 1e-5)
    safe_slip_init = min(max(slipping_prob_init, 1e-5), d_max - 1e-5)
    
    guess_offset = np.log(safe_guess_init / (c_max - safe_guess_init))
    slip_offset = np.log(safe_slip_init / (d_max - safe_slip_init))
    
    c_probs = c_max * sigmoid(c_weights + guess_offset)
    d_probs = d_max * sigmoid(d_weights + slip_offset)

    with open(map_path, 'r', encoding='utf-8') as f:
        question2idx = json.load(f)

    df_raw = pd.read_csv(
        raw_path,
        usecols=['problem_id', 'answer_type'],
        encoding='latin1'
    ).dropna()
    df_raw['problem_id'] = df_raw['problem_id'].astype(str)
    df_raw = df_raw.drop_duplicates(subset=['problem_id'])

    rows = []
    for problem_id, q_idx in question2idx.items():
        q_idx = int(q_idx)
        if 0 <= q_idx < len(c_probs):
            rows.append({
                'problem_id': str(problem_id),
                'q_idx': q_idx,
                'c': float(c_probs[q_idx]),
                'd': float(d_probs[q_idx]),
            })

    df_param = pd.DataFrame(rows)
    df_merged = pd.merge(df_param, df_raw, on='problem_id', how='left')
    df_merged = df_merged.dropna(subset=['answer_type']).copy()
    df_merged = df_merged[df_merged['answer_type'].isin(['algebra', 'choose_1'])].copy()

    if df_merged.empty:
        raise ValueError('未匹配到 algebra 或 choose_1 的题目。')

    return df_merged


def summarize_by_type(df):
    summary_rows = []
    for answer_type in ['algebra', 'choose_1']:
        sub = df[df['answer_type'] == answer_type]
        if sub.empty:
            continue
        for param in ['c', 'd']:
            values = sub[param].to_numpy()
            summary_rows.append({
                'answer_type': answer_type,
                'parameter': param,
                'count': int(len(values)),
                'mean': float(np.mean(values)),
                'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                'median': float(np.median(values)),
                'q1': float(np.quantile(values, 0.25)),
                'q3': float(np.quantile(values, 0.75)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
            })
    return pd.DataFrame(summary_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='assist09-min20')
    parser.add_argument('--checkpoint', type=str, default='../checkpoint/gikt_assist09_for_irt_猜测率设置0.05.pt')
    parser.add_argument('--map_path', type=str, default=r'H:\er_gikt\back\kt\data\assist09\question2idx.json')
    parser.add_argument('--raw_path', type=str, default=r'H:\dataset\assist09\assistments_2009_2010_non_skill_builder_data_new.csv')
    args = parser.parse_args()

    setup_academic_plot_style()
    out_dir = os.path.dirname(os.path.abspath(__file__))

    print('📦 1. 加载模型参数，并回溯题型映射...')
    df = build_type_param_df(args.checkpoint, args.map_path, args.raw_path)
    summary_df = summarize_by_type(df)

    print('📊 2. 输出 algebra / choose_1 的 c, d 统计量...')
    print(summary_df.to_string(index=False, float_format=lambda x: f'{x:.6f}'))

    timestamp = datetime.now().strftime('%H%M%S')
    detail_csv = os.path.join(out_dir, f'scheme4_type_cd_detail_{timestamp}.csv')
    summary_csv = os.path.join(out_dir, f'scheme4_type_cd_summary_{timestamp}.csv')
    df.to_csv(detail_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    print(f'💾 明细已保存: {detail_csv}')
    print(f'💾 统计已保存: {summary_csv}')

    print('🎨 3. 绘制 algebra / choose_1 的 c、d 分布图...')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    palette = {'algebra': '#4DBBD5', 'choose_1': '#E64B35'}

    sns.kdeplot(
        data=df, x='c', hue='answer_type', fill=True, common_norm=False,
        palette=palette, alpha=0.35, linewidth=2, ax=axes[0, 0]
    )
    axes[0, 0].set_title('Guessing Rate Distribution by Problem Type')
    axes[0, 0].set_xlabel('Learned Guessing Probability ($c$)')
    axes[0, 0].set_ylabel('Density')

    sns.boxplot(
        data=df, x='answer_type', y='c', hue='answer_type', palette=palette,
        dodge=False, width=0.5, ax=axes[0, 1]
    )
    legend_01 = axes[0, 1].get_legend()
    if legend_01 is not None:
        legend_01.remove()
    axes[0, 1].set_title('Guessing Rate Summary by Problem Type')
    axes[0, 1].set_xlabel('Problem Type')
    axes[0, 1].set_ylabel('Learned Guessing Probability ($c$)')

    sns.kdeplot(
        data=df, x='d', hue='answer_type', fill=True, common_norm=False,
        palette=palette, alpha=0.35, linewidth=2, ax=axes[1, 0]
    )
    axes[1, 0].set_title('Slipping Rate Distribution by Problem Type')
    axes[1, 0].set_xlabel('Learned Slipping Probability ($d$)')
    axes[1, 0].set_ylabel('Density')

    sns.boxplot(
        data=df, x='answer_type', y='d', hue='answer_type', palette=palette,
        dodge=False, width=0.5, ax=axes[1, 1]
    )
    legend_11 = axes[1, 1].get_legend()
    if legend_11 is not None:
        legend_11.remove()
    axes[1, 1].set_title('Slipping Rate Summary by Problem Type')
    axes[1, 1].set_xlabel('Problem Type')
    axes[1, 1].set_ylabel('Learned Slipping Probability ($d$)')

    for ax in axes.flat:
        ax.grid(alpha=0.15, linestyle='--')

    fig.suptitle('Type-wise 4PL Parameter Distributions: algebra vs choose_1', fontsize=17, y=0.98, fontweight='bold')
    sns.despine(fig=fig)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = os.path.join(out_dir, f'scheme4_type_cd_distribution_{timestamp}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f'✅ 图像已保存: {out_path}')


if __name__ == '__main__':
    main()

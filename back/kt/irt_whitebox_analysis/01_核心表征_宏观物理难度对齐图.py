import os
import sys
import argparse
import warnings
from datetime import datetime

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import scipy.stats
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from dataset import UnifiedParquetDataset, SeqFeatureKey


def setup_academic_plot_style():
    """顶刊风格绘图预设。"""
    plt.rcParams.update({
        'font.family': ['sans-serif'],
        'font.sans-serif': ['HarmonyOS Sans SC', 'SimHei', 'Arial', 'Times New Roman'],
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


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def zscore(arr):
    arr = np.asarray(arr, dtype=float)
    std = arr.std(ddof=0)
    if std < 1e-12:
        return np.zeros_like(arr)
    return (arr - arr.mean()) / std


def load_state_dict(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, torch.nn.Module):
        return ckpt.state_dict()
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        return ckpt['state_dict']
    return ckpt


def extract_learned_item_params(checkpoint_path):
    state_dict = load_state_dict(checkpoint_path)
    try:
        b_weights = state_dict['difficulty_bias.weight'].squeeze().numpy()
        a_bias = state_dict['discrimination_bias.weight'].squeeze().numpy()
    except KeyError as exc:
        raise ValueError('未找到 difficulty_bias.weight 或 discrimination_bias.weight。') from exc

    learned_df = pd.DataFrame({
        'q_id': np.arange(len(b_weights), dtype=int),
        'learned_b': b_weights.astype(float),
        'learned_a': (1.0 + softplus(a_bias)).astype(float),
    })
    return learned_df


def build_interaction_df(config, mode):
    dataset = UnifiedParquetDataset(config, augment=False, mode=mode)
    loader = DataLoader(dataset, batch_size=128, num_workers=0, shuffle=False)

    rows = []
    learner_offset = 0
    for batch in loader:
        q_batch = batch[SeqFeatureKey.Q].numpy()
        r_batch = batch[SeqFeatureKey.R].numpy()
        mask_batch = batch[SeqFeatureKey.MASK].numpy().astype(bool)

        batch_size = q_batch.shape[0]
        for local_idx in range(batch_size):
            learner_id = learner_offset + local_idx
            q_seq = q_batch[local_idx][mask_batch[local_idx]]
            r_seq = r_batch[local_idx][mask_batch[local_idx]]
            for q_id, is_correct in zip(q_seq, r_seq):
                rows.append({
                    'learner_id': learner_id,
                    'q_id': int(q_id),
                    'is_correct': int(is_correct),
                })
        learner_offset += batch_size

    if not rows:
        raise ValueError('未从数据集中提取到有效交互。')

    return pd.DataFrame(rows)


def compute_empirical_item_params(interaction_df, min_attempts):
    learner_ability = (
        interaction_df.groupby('learner_id')['is_correct']
        .mean()
        .rename('learner_ability')
        .reset_index()
    )
    df = interaction_df.merge(learner_ability, on='learner_id', how='left')

    item_acc = df.groupby('q_id')['is_correct'].agg(['count', 'mean']).reset_index()
    item_acc = item_acc.rename(columns={'count': 'attempts', 'mean': 'accuracy'})
    item_acc = item_acc[item_acc['attempts'] >= min_attempts].copy()

    eps = 1e-6
    item_acc['empirical_b'] = np.log((1.0 - item_acc['accuracy'] + eps) / (item_acc['accuracy'] + eps))

    discrim_rows = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for q_id, sub in df.groupby('q_id'):
            if len(sub) < min_attempts:
                continue
            if sub['is_correct'].nunique() < 2:
                continue
            if sub['learner_ability'].nunique() < 2:
                continue
            try:
                corr, _ = scipy.stats.pointbiserialr(sub['is_correct'], sub['learner_ability'])
            except Exception:
                corr = np.nan
            discrim_rows.append({
                'q_id': int(q_id),
                'empirical_a': float(corr) if pd.notna(corr) else np.nan,
            })

    discrim_df = pd.DataFrame(discrim_rows)
    empirical_df = item_acc.merge(discrim_df, on='q_id', how='left')
    empirical_df = empirical_df.dropna(subset=['empirical_a']).copy()
    return empirical_df


def correlation_stats(x, y):
    pearson_r, pearson_p = scipy.stats.pearsonr(x, y)
    spearman_rho, spearman_p = scipy.stats.spearmanr(x, y)
    return {
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p),
        'spearman_rho': float(spearman_rho),
        'spearman_p': float(spearman_p),
    }


def distribution_stats(x, y):
    ks_stat, ks_p = scipy.stats.ks_2samp(x, y)
    wasserstein = scipy.stats.wasserstein_distance(x, y)
    return {
        'ks_stat': float(ks_stat),
        'ks_p': float(ks_p),
        'wasserstein': float(wasserstein),
    }


def plot_dual_axis_alignment(ax, learned_z, empirical_z, title, color_left, color_right):
    ax_right = ax.twinx()

    sns.kdeplot(x=learned_z, ax=ax, color=color_left, linewidth=2.5, fill=False, label='模型学习难度 (z-score)')
    sns.kdeplot(x=empirical_z, ax=ax_right, color=color_right, linewidth=2.5, fill=False, linestyle='--', label='样本经验难度 (z-score)')

    ax.set_title(title)
    ax.set_xlabel('标准化数值 (z-score)')
    ax.set_ylabel('模型参数分布密度', color=color_left)
    ax_right.set_ylabel('经验统计分布密度', color=color_right)
    ax.tick_params(axis='y', colors=color_left)
    ax_right.tick_params(axis='y', colors=color_right)
    ax.grid(alpha=0.15, linestyle='--')

    handles_left, labels_left = ax.get_legend_handles_labels()
    handles_right, labels_right = ax_right.get_legend_handles_labels()
    ax.legend(handles_left + handles_right, labels_left + labels_right, loc='upper right', frameon=True)
    return ax_right


def plot_correlation_panel(ax, x, y, title, xlabel, ylabel, color):
    sns.regplot(
        x=x,
        y=y,
        ax=ax,
        scatter_kws={'alpha': 0.55, 's': 24, 'edgecolor': 'white', 'linewidth': 0.4},
        line_kws={'color': '#E64B35', 'linewidth': 2.2, 'linestyle': '--'},
        color=color,
        label='回归拟合线'
    )
    lim_min = min(np.min(x), np.min(y))
    lim_max = max(np.max(x), np.max(y))
    ax.plot([lim_min, lim_max], [lim_min, lim_max], color='gray', linestyle=':', linewidth=1.5, label='理想对角线 (y=x)')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.15, linestyle='--')
    ax.legend(loc='upper left', frameon=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='assist09')
    parser.add_argument('--checkpoint', type=str, default='../checkpoint/gikt_assist09_for_irt_猜测率设置0.05.pt')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'valid', 'test'])
    parser.add_argument('--min_attempts', type=int, default=10)
    args = parser.parse_args()

    setup_academic_plot_style()
    config = Config(dataset_name=args.dataset_name)

    print('📦 1. 提取模型学习到的 b / a 参数...')
    learned_df = extract_learned_item_params(args.checkpoint)
    print(f'学习参数题目数: {len(learned_df)}')

    print('🧮 2. 基于真实交互计算经验难度 / 区分度代理指标...')
    interaction_df = build_interaction_df(config, args.mode)
    empirical_df = compute_empirical_item_params(interaction_df, args.min_attempts)
    print(f'经验统计有效题目数: {len(empirical_df)} (min_attempts={args.min_attempts})')

    merged_df = pd.merge(learned_df, empirical_df, on='q_id', how='inner')
    if merged_df.empty:
        raise ValueError('学习参数与经验参数没有可对齐的题目。')

    merged_df['learned_b_z'] = zscore(merged_df['learned_b'])
    merged_df['empirical_b_z'] = zscore(merged_df['empirical_b'])
    merged_df['learned_a_z'] = zscore(merged_df['learned_a'])
    merged_df['empirical_a_z'] = zscore(merged_df['empirical_a'])

    b_corr = correlation_stats(merged_df['learned_b'], merged_df['empirical_b'])
    a_corr = correlation_stats(merged_df['learned_a'], merged_df['empirical_a'])
    b_dist = distribution_stats(merged_df['learned_b_z'], merged_df['empirical_b_z'])
    a_dist = distribution_stats(merged_df['learned_a_z'], merged_df['empirical_a_z'])

    stats_df = pd.DataFrame([
        {'parameter': 'b', **b_corr, **b_dist},
        {'parameter': 'a', **a_corr, **a_dist},
    ])

    # timestamp = datetime.now().strftime('%H%M%S')
    timestamp = ''
    out_dir = os.path.dirname(os.path.abspath(__file__))
    merged_csv = os.path.join(out_dir, f'01_核心数据_难度对齐明细_{timestamp}.csv')
    stats_csv = os.path.join(out_dir, f'01_核心数据_难度对齐统计_{timestamp}.csv')
    merged_df.to_csv(merged_csv, index=False)
    stats_df.to_csv(stats_csv, index=False)
    print(f'💾 对齐数据已保存: {merged_csv}')
    print(f'💾 对齐统计已保存: {stats_csv}')

    print('🎨 3. 绘制论文版 learned vs empirical 对齐图（去掉区分度 a）...')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    plot_dual_axis_alignment(
        axes[0],
        merged_df['learned_b_z'],
        merged_df['empirical_b_z'],
        '难度对齐与双轴核密度估计 (Learned $b$ vs Empirical $b^*$)',
        '#4DBBD5',
        '#E64B35',
    )
    axes[0].text(
        0.03, 0.95,
        f"皮尔逊相关系数 r = {b_corr['pearson_r']:.3f}\n斯皮尔曼等级相关 ρ = {b_corr['spearman_rho']:.3f}\nKS 统计量 = {b_dist['ks_stat']:.3f}\nWasserstein 距离 = {b_dist['wasserstein']:.3f}",
        transform=axes[0].transAxes,
        ha='left', va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray'),
    )

    plot_correlation_panel(
        axes[1],
        merged_df['learned_b_z'],
        merged_df['empirical_b_z'],
        '真实物理难度相关性分析 (Standardized)',
        '模型学习难度 $b$ (z-score)',
        '样本经验难度 $b^*$ (z-score)',
        '#4DBBD5',
    )

    fig.suptitle('FA-GIKT 难度参数与真实试题样本错误率对齐印证', fontsize=18, y=1.02, fontweight='bold')
    sns.despine(fig=fig)
    plt.tight_layout()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path_png = os.path.join(out_dir, f'01_核心表征_难度物理对齐图_{timestamp}.png')
    out_path_pdf = os.path.join(out_dir, f'01_核心表征_难度物理对齐图_{timestamp}.pdf')
    plt.savefig(out_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_path_pdf, bbox_inches='tight')
    print(f'✅ 图像已保存:\n  - PNG: {out_path_png}\n  - PDF: {out_path_pdf}')


if __name__ == '__main__':
    main()

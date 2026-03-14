import os
import re
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


def setup_academic_plot_style():
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


def parse_simple_toml(config_path):
    result = {}
    if not config_path or not os.path.exists(config_path):
        return result
    pattern = re.compile(r'^\s*([A-Za-z0-9_]+)\s*=\s*(.+?)\s*$')
    with open(config_path, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.split('#', 1)[0].strip()
            if not line or line.startswith('['):
                continue
            match = pattern.match(line)
            if not match:
                continue
            key, value = match.groups()
            value = value.strip().strip('"').strip("'")
            try:
                if value.lower() in {'true', 'false'}:
                    result[key] = value.lower() == 'true'
                elif any(ch in value for ch in ['.', 'e', 'E']):
                    result[key] = float(value)
                else:
                    result[key] = int(value)
            except ValueError:
                result[key] = value
    return result


def compute_stats(values):
    arr = np.asarray(values, dtype=float)
    return {
        'count': int(len(arr)),
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        'min': float(np.min(arr)),
        'q1': float(np.quantile(arr, 0.25)),
        'median': float(np.median(arr)),
        'q3': float(np.quantile(arr, 0.75)),
        'max': float(np.max(arr)),
    }


def inverse_sigmoid(y):
    y = np.clip(y, 1e-8, 1 - 1e-8)
    return np.log(y / (1.0 - y))


def required_logit_for_probability(target_prob, cap):
    if target_prob <= 0:
        return float('-inf')
    if target_prob >= cap:
        return float('inf')
    return float(inverse_sigmoid(target_prob / cap))


def build_type_map(map_path, raw_path):
    with open(map_path, 'r', encoding='utf-8') as f:
        question2idx = json.load(f)

    df_raw = pd.read_csv(raw_path, usecols=['problem_id', 'answer_type'], encoding='latin1').dropna()
    df_raw['problem_id'] = df_raw['problem_id'].astype(str)
    df_raw = df_raw.drop_duplicates(subset=['problem_id'])

    rows = []
    for problem_id, q_idx in question2idx.items():
        rows.append({'problem_id': str(problem_id), 'q_idx': int(q_idx)})
    df_map = pd.DataFrame(rows)
    return pd.merge(df_map, df_raw, on='problem_id', how='left')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='assist09')
    parser.add_argument('--checkpoint', type=str, default='../checkpoint/gikt_assist09_for_irt_猜测率设置0.05.pt')
    parser.add_argument('--config_path', type=str, default='H:/er_gikt/back/kt/config/experiments/exp_full_default.toml')
    parser.add_argument('--map_path', type=str, default=r'H:\er_gikt\back\kt\data\assist09\question2idx.json')
    parser.add_argument('--raw_path', type=str, default=r'H:\dataset\assist09\assistments_2009_2010_non_skill_builder_data_new.csv')
    parser.add_argument('--c_cap', type=float, default=None)
    parser.add_argument('--d_cap', type=float, default=None)
    parser.add_argument('--guessing_init', type=float, default=None)
    parser.add_argument('--slipping_init', type=float, default=None)
    parser.add_argument('--guessing_reg_boundary', type=float, default=-2.0)
    parser.add_argument('--slipping_reg_boundary', type=float, default=-3.0)
    parser.add_argument('--reg_4pl', type=float, default=None)
    args = parser.parse_args()

    setup_academic_plot_style()
    config = Config(dataset_name=args.dataset_name)
    exp_cfg = parse_simple_toml(args.config_path)

    c_cap = args.c_cap if args.c_cap is not None else 0.25
    d_cap = args.d_cap if args.d_cap is not None else 0.05
    guessing_init = args.guessing_init if args.guessing_init is not None else float(exp_cfg.get('guessing_prob_init', 0.05))
    slipping_init = args.slipping_init if args.slipping_init is not None else float(exp_cfg.get('slipping_prob_init', 0.02))
    reg_4pl = args.reg_4pl if args.reg_4pl is not None else float(exp_cfg.get('reg_4pl', 1e-5))

    print('📦 1. 读取 4PL 参数并检查塌缩风险...')
    state_dict = load_state_dict(args.checkpoint)
    c_logits = state_dict['guessing_bias.weight'].squeeze().numpy()
    d_logits = state_dict['slipping_bias.weight'].squeeze().numpy()
    c_probs = c_cap * sigmoid(c_logits)
    d_probs = d_cap * sigmoid(d_logits)

    # 与 gikt.py 初始化逻辑保持严格一致：
    # raw_logit = log(p / (1-p))，随后前向中再乘 cap
    init_c_logit = float(inverse_sigmoid(guessing_init))
    init_d_logit = float(inverse_sigmoid(slipping_init))
    effective_init_c = c_cap * guessing_init
    effective_init_d = d_cap * slipping_init
    free_c_ceiling = c_cap * sigmoid(args.guessing_reg_boundary)
    free_d_ceiling = d_cap * sigmoid(args.slipping_reg_boundary)

    c_stats = compute_stats(c_probs)
    d_stats = compute_stats(d_probs)
    c_logit_stats = compute_stats(c_logits)
    d_logit_stats = compute_stats(d_logits)

    pct_c_over_boundary = float(np.mean(c_logits > args.guessing_reg_boundary) * 100)
    pct_d_over_boundary = float(np.mean(d_logits > args.slipping_reg_boundary) * 100)
    pct_c_near_init = float(np.mean(np.abs(c_logits - init_c_logit) < 0.1) * 100)
    pct_d_near_init = float(np.mean(np.abs(d_logits - init_d_logit) < 0.1) * 100)

    diagnosis_rows = [
        {
            'parameter': 'c',
            'cap': c_cap,
            'config_init_prob': guessing_init,
            'effective_init_prob': effective_init_c,
            'init_logit': init_c_logit,
            'reg_boundary_logit': args.guessing_reg_boundary,
            'free_ceiling_prob': free_c_ceiling,
            'mean_prob': c_stats['mean'],
            'std_prob': c_stats['std'],
            'max_prob': c_stats['max'],
            'mean_logit': c_logit_stats['mean'],
            'std_logit': c_logit_stats['std'],
            'pct_over_reg_boundary': pct_c_over_boundary,
            'pct_near_init_logit_pm_0.1': pct_c_near_init,
        },
        {
            'parameter': 'd',
            'cap': d_cap,
            'config_init_prob': slipping_init,
            'effective_init_prob': effective_init_d,
            'init_logit': init_d_logit,
            'reg_boundary_logit': args.slipping_reg_boundary,
            'free_ceiling_prob': free_d_ceiling,
            'mean_prob': d_stats['mean'],
            'std_prob': d_stats['std'],
            'max_prob': d_stats['max'],
            'mean_logit': d_logit_stats['mean'],
            'std_logit': d_logit_stats['std'],
            'pct_over_reg_boundary': pct_d_over_boundary,
            'pct_near_init_logit_pm_0.1': pct_d_near_init,
        },
    ]
    diagnosis_df = pd.DataFrame(diagnosis_rows)
    print(diagnosis_df.to_string(index=False, float_format=lambda x: f'{x:.6f}'))

    print('🔍 2. 计算在当前 cap 下达到若干目标概率所需的 logit...')
    target_rows = []
    for target_c in [0.03, 0.05, 0.10, 0.15, 0.20, 0.24]:
        target_rows.append({
            'parameter': 'c',
            'target_probability': target_c,
            'required_logit': required_logit_for_probability(target_c, c_cap),
            'beyond_reg_boundary': required_logit_for_probability(target_c, c_cap) > args.guessing_reg_boundary,
        })
    for target_d in [0.002, 0.005, 0.01, 0.02, 0.03, 0.04]:
        target_rows.append({
            'parameter': 'd',
            'target_probability': target_d,
            'required_logit': required_logit_for_probability(target_d, d_cap),
            'beyond_reg_boundary': required_logit_for_probability(target_d, d_cap) > args.slipping_reg_boundary,
        })
    target_df = pd.DataFrame(target_rows)
    print(target_df.to_string(index=False, float_format=lambda x: f'{x:.6f}'))

    print('🧭 3. 做题型层面的分化检查...')
    type_df = build_type_map(args.map_path, args.raw_path)
    param_df = pd.DataFrame({
        'q_idx': np.arange(len(c_probs), dtype=int),
        'c_prob': c_probs,
        'd_prob': d_probs,
        'c_logit': c_logits,
        'd_logit': d_logits,
    })
    merged_type_df = pd.merge(param_df, type_df[['q_idx', 'answer_type']], on='q_idx', how='left')
    merged_type_df = merged_type_df[merged_type_df['answer_type'].isin(['algebra', 'choose_1'])].copy()
    type_summary = merged_type_df.groupby('answer_type').agg(
        c_mean=('c_prob', 'mean'),
        c_std=('c_prob', 'std'),
        d_mean=('d_prob', 'mean'),
        d_std=('d_prob', 'std'),
        c_logit_mean=('c_logit', 'mean'),
        d_logit_mean=('d_logit', 'mean'),
        count=('q_idx', 'count'),
    ).reset_index()
    print(type_summary.to_string(index=False, float_format=lambda x: f'{x:.6f}'))

    timestamp = datetime.now().strftime('%H%M%S')
    diag_csv = os.path.join(config.path.OUTPUT_DIR, f'irt_cd_collapse_diagnosis_{timestamp}.csv')
    target_csv = os.path.join(config.path.OUTPUT_DIR, f'irt_cd_target_logit_table_{timestamp}.csv')
    type_csv = os.path.join(config.path.OUTPUT_DIR, f'irt_cd_type_summary_{timestamp}.csv')
    diagnosis_df.to_csv(diag_csv, index=False)
    target_df.to_csv(target_csv, index=False)
    type_summary.to_csv(type_csv, index=False)

    report_lines = [
        'IRT c/d collapse diagnosis report',
        f'reg_4pl = {reg_4pl}',
        f'c cap = {c_cap}, d cap = {d_cap}',
        f'config guessing init = {guessing_init}, config slipping init = {slipping_init}',
        f'effective guessing init after cap = {effective_init_c:.6f}, effective slipping init after cap = {effective_init_d:.6f}',
        f'c regularization boundary logit = {args.guessing_reg_boundary}, free ceiling probability = {free_c_ceiling:.6f}',
        f'd regularization boundary logit = {args.slipping_reg_boundary}, free ceiling probability = {free_d_ceiling:.6f}',
        f'Observed c mean/std/max = {c_stats["mean"]:.6f}/{c_stats["std"]:.6f}/{c_stats["max"]:.6f}',
        f'Observed d mean/std/max = {d_stats["mean"]:.6f}/{d_stats["std"]:.6f}/{d_stats["max"]:.6f}',
        f'Pct c logits over reg boundary = {pct_c_over_boundary:.2f}%',
        f'Pct d logits over reg boundary = {pct_d_over_boundary:.2f}%',
        f'Pct c logits still near init = {pct_c_near_init:.2f}%',
        f'Pct d logits still near init = {pct_d_near_init:.2f}%',
    ]

    if c_stats['max'] <= free_c_ceiling + 1e-4:
        report_lines.append('Diagnosis: c is very likely trapped by the joint effect of low initialization + boundary regularization, not only by the cap itself.')
    else:
        report_lines.append('Diagnosis: c has crossed the regularization-free region; collapse is not explained by cap alone.')

    if d_stats['max'] <= free_d_ceiling + 1e-4:
        report_lines.append('Diagnosis: d is also trapped below the regularization-free ceiling and may be over-shrunk.')

    report_path = os.path.join(config.path.OUTPUT_DIR, f'irt_cd_collapse_report_{timestamp}.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    print('🎨 4. 绘制诊断图...')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    sns.histplot(c_logits, bins=50, kde=True, color='#4DBBD5', edgecolor='white', alpha=0.65, ax=axes[0, 0])
    axes[0, 0].axvline(init_c_logit, color='#E64B35', linestyle='--', linewidth=2, label=f'init logit = {init_c_logit:.3f}')
    axes[0, 0].axvline(args.guessing_reg_boundary, color='#3C5488', linestyle='-.', linewidth=2, label=f'reg boundary = {args.guessing_reg_boundary:.3f}')
    axes[0, 0].set_title('Guessing Logit Distribution')
    axes[0, 0].set_xlabel('guessing_bias raw logit')
    axes[0, 0].set_ylabel('Item Count')
    axes[0, 0].legend(frameon=True)

    sns.histplot(d_logits, bins=50, kde=True, color='#00A087', edgecolor='white', alpha=0.65, ax=axes[0, 1])
    axes[0, 1].axvline(init_d_logit, color='#E64B35', linestyle='--', linewidth=2, label=f'init logit = {init_d_logit:.3f}')
    axes[0, 1].axvline(args.slipping_reg_boundary, color='#3C5488', linestyle='-.', linewidth=2, label=f'reg boundary = {args.slipping_reg_boundary:.3f}')
    axes[0, 1].set_title('Slipping Logit Distribution')
    axes[0, 1].set_xlabel('slipping_bias raw logit')
    axes[0, 1].set_ylabel('Item Count')
    axes[0, 1].legend(frameon=True)

    sns.histplot(c_probs, bins=50, kde=True, color='#F39B7F', edgecolor='white', alpha=0.65, ax=axes[1, 0])
    axes[1, 0].axvline(effective_init_c, color='#E64B35', linestyle='--', linewidth=2, label=f'effective init = {effective_init_c:.3f}')
    axes[1, 0].axvline(free_c_ceiling, color='#3C5488', linestyle='-.', linewidth=2, label=f'free ceiling = {free_c_ceiling:.3f}')
    axes[1, 0].set_title('Guessing Probability Distribution')
    axes[1, 0].set_xlabel('Learned guessing probability $c$')
    axes[1, 0].set_ylabel('Item Count')
    axes[1, 0].legend(frameon=True)

    sns.boxplot(data=merged_type_df, x='answer_type', y='c_prob', hue='answer_type', dodge=False, palette={'algebra': '#4DBBD5', 'choose_1': '#E64B35'}, ax=axes[1, 1])
    legend = axes[1, 1].get_legend()
    if legend is not None:
        legend.remove()
    axes[1, 1].set_title('Type-wise Guessing Differentiation Check')
    axes[1, 1].set_xlabel('Problem Type')
    axes[1, 1].set_ylabel('Learned guessing probability $c$')

    for ax in axes.flat:
        ax.grid(alpha=0.15, linestyle='--')

    fig.suptitle('Collapse Diagnosis for Guessing / Slipping Parameters', fontsize=18, y=0.98, fontweight='bold')
    sns.despine(fig=fig)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    fig_path = os.path.join(config.path.OUTPUT_DIR, f'irt_cd_collapse_diagnosis_{timestamp}.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')

    print(f'💾 诊断表已保存: {diag_csv}')
    print(f'💾 目标 logit 表已保存: {target_csv}')
    print(f'💾 题型汇总已保存: {type_csv}')
    print(f'💾 文字报告已保存: {report_path}')
    print(f'✅ 诊断图已保存: {fig_path}')


if __name__ == '__main__':
    main()

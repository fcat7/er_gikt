import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import toml
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import random
import argparse
from datetime import datetime

"""
Step 1: 环境与系统初始化 (load_system)
做的事：锁定随机种子（为了可复现性），加载你训练好的 GIKT/深度学习模型，以及测试集数据。
Step 2: 典型标本筛选（苛刻的选角）
做的事：从测试集中捞出 300 个学生。根据极其死板的学术硬指标（不仅看正确率 acc，还看历史记录长度 len），强行挑出三个最具代表性的“群像”：
Student A (差生): 答题率 < 0.45，需要补基础。
Student B (中等生): 答题率在 0.5~0.6，处于瓶颈期。
Student C (尖子生): 答题率 > 0.75，需要拔高拓展。
意义：保证你的案例不是“特例”，而是统计学意义上的典型。
Step 3: 拦截并窃听粒子群算法 (patched_optimize & analyze_student)
做的事：我们使用了一个黑魔法（Monkey Patching），入侵并篡改了原生的 MOPSO 函数。在算法每次迭代寻找最优解时，强行把中间记录（Archive）抓取下来存进内存。
意义：这是本脚本的灵魂。没有这一步，你的引擎就是一个黑盒，只能吐出最终结果。有了这一步，我们就能画出 evolution 中那些粒子是如何一步步汇聚到帕累托前沿的。
Step 4: 数据绘制与沉淀 (plot_pareto & plot_zpd & export_cases)
Pareto 系列：把 Step 3 截获的进化轨迹画在 F1-F2 坐标系上。加上那张神来之笔的“九宫格策略矩阵”（平稳巩固区、极度激进区等），赋予枯燥的点阵以人类教育学的意义。
ZPD 对齐图：把最终选择的那 5 道题拆散，映射到每个学生各自的认知区间（
τ）里，并用气泡大小表示题目的知识密度。证明模型没有“一刀切”。
CSV 归档：把你刚才画图用的所有底层详细数据（题号、概率、知识点个数）全部打印成冰冷的表格。
意义：CSV 表是你的底座，防止导师说你图是伪造的；Pareto 代表“宏观策略均衡”，ZPD 代表“微观题目适配”。
"""

LOW_ACC_THRESHOLD = 0.30
MID_ACC_LOWER = 0.50
MID_ACC_UPPER = 0.60
HIGH_ACC_THRESHOLD = 0.85

def set_seed(seed):
    """固定所有随机种子，确保论文图表每次生成的结果百分百精确一致"""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

from matplotlib.font_manager import FontProperties

# Ensure fonts for Chinese
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# Set up paths
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
_KT_DIR = os.path.join(os.path.dirname(_THIS_DIR), 'kt')
if _KT_DIR not in sys.path:
    sys.path.insert(0, _KT_DIR)

from config import get_config, DEVICE
from run_er_eval import RecDataLoader, get_exp_config_path
from pso_recommend import RecommendationSystem, DiscreteMOPSO

logger = logging.getLogger('case_study')
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Global reference
original_optimize = DiscreteMOPSO.optimize
archive_history_global = []

def patched_optimize(self, candidates_probs: torch.Tensor) -> np.ndarray:
    global archive_history_global
    archive_history_global = []
    
    self.probs = {q.item(): p.item() for q, p in zip(self.candidates, candidates_probs)}
    particles = [np.random.choice(self.candidates, size=self.K, replace=False) for _ in range(self.n_particles)]
    
    pbest = particles[:]
    pbest_fitness = [
        self.evaluator.evaluate_combo(p, np.array([self.probs.get(q, 0.5) for q in p]), self.m_k, self.tau)
        for p in particles
    ]

    archive = self._build_archive(particles, pbest_fitness)
    
    # Needs cd computation to find max
    if archive:
        fitness_list = [af for ap, af in archive]
        cd = self._compute_crowding_distance(fitness_list)
        max_cd = max(cd) if len(cd) > 0 else 0
        if max_cd == 0: max_cd = 1e-8
        top_indices = [j for j, d in enumerate(cd) if d >= max_cd * 0.99]
        gbest = archive[np.random.choice(top_indices)][0]
    else:
        gbest = pbest[0]
    
    # iter 0 history
    archive_history_global.append([(ind[0], ind[1]) for ind in archive])

    for it in range(self.n_iterations):
        for i in range(self.n_particles):
            current = particles[i]
            c_probs = np.array([self.probs.get(q, 0.5) for q in current])
            f_curr = self.evaluator.evaluate_combo(current, c_probs, self.m_k, self.tau)
            
            new_p = self._update_particle(current, pbest[i], gbest)
            new_probs = np.array([self.probs.get(q, 0.5) for q in new_p])
            f_new = self.evaluator.evaluate_combo(new_p, new_probs, self.m_k, self.tau)

            if self._dominates(f_new, f_curr) or np.random.rand() < 0.3:
                particles[i] = new_p
                f_curr = f_new

            if self._dominates(f_curr, pbest_fitness[i]):
                pbest[i] = particles[i]
                pbest_fitness[i] = f_curr

        archive = self._build_archive([p for p in particles] + [a[0] for a in archive], [f for f in pbest_fitness] + [a[1] for a in archive])

        if archive:
            fitness_list = [af for ap, af in archive]
            cd = self._compute_crowding_distance(fitness_list)
            max_cd = max(cd) if len(cd) > 0 else 0
            if max_cd == 0: max_cd = 1e-8
            top_indices = [j for j, d in enumerate(cd) if d >= max_cd * 0.99]
            gbest = archive[np.random.choice(top_indices)][0]
            
        # save histories
        if (it + 1) in [2, 10, self.n_iterations]:
            archive_history_global.append([(ind[0], ind[1]) for ind in archive])

    return original_optimize(self, candidates_probs)

# Apply global patch
DiscreteMOPSO.optimize = patched_optimize

def load_system():
    conf_path = get_exp_config_path(False)
    with open(conf_path, 'r', encoding='utf-8') as f:
        config_dict = {k: v for sect in toml.load(f).values() for k, v in (sect.items() if isinstance(sect, dict) else {})}
    
    loader = RecDataLoader(config_dict)
    
    model_path = config_dict.get('model_path')
    if not model_path:
        raise ValueError("Model path missing in config")
        
    logger.info(f"Loading sys from {model_path}")
    recommender = RecommendationSystem(model_path, loader.metadata_path, config_dict, device=DEVICE)
    return recommender, loader, config_dict

def analyze_student(recommender, loader, student_idx, row):
    all_qs, all_rs = np.array(row['q_seq']), np.array(row['r_seq'])
    T = len(all_qs)
    K = 5
    hist_qs, gt_qs = all_qs[:T - K], all_qs[T - K:]
    hist_rs = all_rs[:T - K]
    
    h_q = torch.tensor(hist_qs.reshape(1, -1), dtype=torch.long, device=DEVICE)
    h_r = torch.tensor(hist_rs.reshape(1, -1), dtype=torch.long, device=DEVICE)
    h_mask = torch.ones_like(h_q)
    
    t_int, t_resp = None, None
    if 't_interval' in row.index and row['t_interval'] is not None:
        t_int = torch.tensor(np.array(row['t_interval'], dtype=np.float32)[:T-K].reshape(1, -1), device=DEVICE)
    if 't_response' in row.index and row['t_response'] is not None:
        t_resp = torch.tensor(np.array(row['t_response'], dtype=np.float32)[:T-K].reshape(1, -1), device=DEVICE)
    
    global archive_history_global
    archive_history_global = []
    
    rec_ids, info = recommender.recommend(h_q, h_r, h_mask, K=K, interval_time=t_int, response_time=t_resp)
    
    # Store locally
    archive_history = archive_history_global.copy()
    
    # Compile info
    c_state = info['cognitive_state']
    tau = c_state['tau']
    m_k = c_state['m_k']
    
    # Re-calculate WKC to match threshold=mean
    threshold = float(np.mean(m_k))
    weak_skills = set(np.where(m_k < threshold)[0])
    
    # Gather actual skills
    qs_table = loader.qs_table
    rec_skills = []
    for q in rec_ids:
        rec_skills.append(np.where(qs_table[q] > 0)[0].tolist())
    
    feat_path = os.path.join(os.path.dirname(loader.metadata_path), 'q_features.npy')
    try:
        q_feats = np.load(feat_path)
        # Assuming idx 0 is difficulty from your processing script
        rec_global_diffs = [float(q_feats[q, 0]) for q in rec_ids]
    except Exception as e:
        logger.warning(f"Could not load q_features.npy: {e}")
        rec_global_diffs = [0.0]*len(rec_ids)

    # Save student data
    stu_dict = {
        'student_idx': student_idx, # 学生索引
        'history_len': int(T - K), # 历史记录长度
        'tau': float(tau), # 掌握度阈值
        'weak_skills_count': len(weak_skills), # 薄弱知识点数量
        'rec_ids': rec_ids.tolist(), # 推荐题目ID列表
        'rec_probs': [float(p) for p in info.get('rec_probs', [])] if info.get('rec_probs') is not None else [], # 推荐题目概率列表
        'rec_skills': rec_skills, # 推荐题目对应的知识点列表
        'rec_global_diffs': rec_global_diffs, # 题目的全局静态难度
        'archive_history': archive_history, # 迭代历史记录（每次迭代的解和对应的 F1/F2）
        'm_k_mean': float(np.mean(m_k)) # 学生整体掌握度水平（m_k 的平均值）
    }
    logger.info(f"Analyzed Student {student_idx}: History Len={stu_dict['history_len']}, Tau={stu_dict['tau']:.3f}, Weak Skills={stu_dict['weak_skills_count']}, Rec IDs={stu_dict['rec_ids']}, Rec Probs={[f'{p:.3f}' for p in stu_dict['rec_probs']]}, Rec Skills={stu_dict['rec_skills']}, m_k Mean={stu_dict['m_k_mean']:.3f}")
    
    return stu_dict

def plot_pareto(stu_data_list, out_dir, suffix=""):
    """
    九宫格说明：F1/F2 均已相对于每个学生自身认知状态计算，
    因此统一网格是跨学生公平比较的唯一合法方式。
    """
    # ── Step 0: 动态计算全局坐标边界（解决越界问题） ──
    all_f1, all_f2 = [], []
    for stu in stu_data_list[:3]:
        for arch_snap in stu['archive_history']:
            for item in arch_snap:
                all_f1.append(item[1][0])
                all_f2.append(-item[1][1])
    f1_max = max(all_f1) * 1.15 if all_f1 else 0.008
    f2_min = min(all_f2) * 0.9 if all_f2 else 1.5
    f2_max = max(all_f2) * 1.10 if all_f2 else 6.5
    # 对齐到美观的整数刻度
    x_limit = (0, max(round(f1_max, 3) + 0.001, 0.005))
    y_limit = (max(0, round(f2_min - 0.5, 0)), round(f2_max + 0.5, 0))

    def add_strategy_grid(ax):
        """九宫格：将 F1/F2 空间均匀三等分"""
        x_step = x_limit[1] / 3
        y_step = (y_limit[1] - y_limit[0]) / 3
        x_bins = [(x_limit[0] + i*x_step, x_limit[0] + (i+1)*x_step) for i in range(3)]
        y_bins = [(y_limit[0] + i*y_step, y_limit[0] + (i+1)*y_step) for i in range(3)]
        grid_def = [
            [{"c": "#f5f5f5", "t": "极度保守\n(低覆盖/低误差)"}, {"c": "#eeeeee", "t": "低效低质"},            {"c": "#e0e0e0", "t": "绝对劣解\n(难度脱节/无覆盖)"}],
            [{"c": "#e3f2fd", "t": "平稳巩固"},                   {"c": "#fff8e1", "t": "均衡发展区\n(ZPD核心)"}, {"c": "#ffebee", "t": "认知过载"}],
            [{"c": "#c8e6c9", "t": "最优扩展\n(高覆盖/低误差)"}, {"c": "#fff3e0", "t": "进阶拓展"},            {"c": "#ffcdd2", "t": "极度激进\n(压迫式修补)"}],
        ]
        for i, (ylo, yhi) in enumerate(y_bins):
            for j, (xlo, xhi) in enumerate(x_bins):
                cell = grid_def[i][j]
                ax.fill_between([xlo, xhi], ylo, yhi, color=cell["c"], alpha=0.55, zorder=0, edgecolor='#bdbdbd', linewidth=0.8)
                ax.text((xlo+xhi)/2, (ylo+yhi)/2, cell["t"], color='#757575', alpha=0.85, fontsize=8.5, ha='center', va='center', weight='bold', zorder=1)

    # ── Part 1: 三学生演化子图 ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 7.5))
    titles = ["Student A (Low Mastery)", "Student B (Mid Mastery)", "Student C (High Mastery)"]

    iter_colors  = ['#9e9e9e', '#81d4fa', '#0288d1', '#0d47a1']
    iter_labels  = ['Iter 0 (随机探索)', 'Iter 2 (初期聚拢)', 'Iter 10 (深度寻优)', 'Final (前沿收敛)']
    iter_markers = ['o', 's', '^', 'D']
    snap_indices = [0, 1, 2, -1]

    for i, stu in enumerate(stu_data_list[:3]):
        ax = axes[i]
        hist = stu['archive_history']
        if not hist: continue

        add_strategy_grid(ax)

        for c_idx, (snap_i, lbl) in enumerate(zip(snap_indices, iter_labels)):
            if snap_i >= len(hist): continue
            arch_snap = hist[snap_i]
            f1s = [item[1][0] for item in arch_snap]
            f2s = [-item[1][1] for item in arch_snap]
            n_pts = len(f1s)

            ax.scatter(f1s, f2s,
                       label=(f"{lbl} (n={n_pts})" if i == 0 else ""),
                       color=iter_colors[c_idx], marker=iter_markers[c_idx],
                       alpha=0.8, s=(50 if c_idx < 3 else 80),
                       edgecolors='black', linewidth=0.5, zorder=3)

            if snap_i == snap_indices[-1]:
                sorted_pts = sorted(zip(f1s, f2s))
                ax.plot([p[0] for p in sorted_pts], [p[1] for p in sorted_pts],
                        color='#ff3d00', linestyle='--', linewidth=2.5,
                        label=('Pareto Front' if i == 0 else ""), zorder=2)

        ax.set_xlim(x_limit)
        ax.set_ylim(y_limit)
        ax.set_xlabel('F1: Difficulty Gap (ZPD Error) $\\downarrow$', weight='bold')
        ax.set_ylabel('F2: Weak Skill Coverage (WKC) $\\uparrow$', weight='bold')
        final_n = len(hist[-1]) if hist else 0
        ax.set_title(f"{titles[i]}\nAcc: {stu['m_k_mean']:.2f} | Final Archive: {final_n} pts",
                      fontsize=12, weight='bold')

        if i == 0:
            ax.legend(loc='lower left', fontsize='small', framealpha=0.9)

    fig.subplots_adjust(bottom=0.22)
    plt.savefig(os.path.join(out_dir, f'pareto_evolution_unified{suffix}.pdf'), dpi=300, bbox_inches='tight')

    # ── Part 2: Master Comparison ──
    fig2, ax_m = plt.subplots(figsize=(9, 7))
    add_strategy_grid(ax_m)

    colors_m  = ['#d32f2f', '#388e3c', '#1976d2']
    markers_m = ['D', 's', '^']
    labels_m  = ['Low Mastery (A)', 'Mid Mastery (B)', 'High Mastery (C)']

    for i, stu in enumerate(stu_data_list[:3]):
        arch = stu['archive_history'][-1]
        f1s = [item[1][0] for item in arch]
        f2s = [-item[1][1] for item in arch]

        ax_m.scatter(f1s, f2s, color=colors_m[i], marker=markers_m[i],
                     alpha=0.9, s=90, edgecolors='black', linewidth=0.8,
                     label=f"{labels_m[i]} (n={len(f1s)})", zorder=4)
        sorted_pts = sorted(zip(f1s, f2s))
        ax_m.plot([p[0] for p in sorted_pts], [p[1] for p in sorted_pts],
                  color=colors_m[i], linestyle='-', linewidth=2.5, zorder=3)

    ax_m.set_xlim(x_limit)
    ax_m.set_ylim(y_limit)
    ax_m.set_xlabel('F1: Difficulty Map Error $\\downarrow$', fontsize=12, weight='bold')
    ax_m.set_ylabel('F2: Weakness Coverage $\\uparrow$', fontsize=12, weight='bold')
    ax_m.set_title('Multi-Archetype Pareto Front Comparison\n(Strategy Matrix Projection)', fontsize=14, weight='bold')
    ax_m.legend(loc='upper right', fontsize=10, framealpha=0.9)
    plt.savefig(os.path.join(out_dir, f'pareto_master_comparison{suffix}.pdf'), dpi=300, bbox_inches='tight')

    logger.info(f"Saved UNIFIED evolution and MASTER comparison plots (suffix='{suffix}').")

def plot_zpd(stu_data_list, out_dir, suffix=""):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), sharey=True)
    
    global_diffs_all = []
    for stu in stu_data_list[:3]:
        global_diffs_all.extend(stu.get('rec_global_diffs', [0.0]*len(stu['rec_probs'])))
        
    d_min, d_max = min(global_diffs_all), max(global_diffs_all) if global_diffs_all else (0, 0)
    
    # 配色方案 (Nature级别高级配色: Science Journal风格)
    color_map = {
        1: ('#4DBBD5', '1 Skill (Basic)'),            # 明亮蓝
        2: ('#E64B35', '2-3 Skills (Compound)'),      # 赭红
        3: ('#3C5488', '>3 Skills (Complex)')         # 深藏青
    }
    
    zone_colors = ['#ffcdd2', '#c8e6c9', '#bbdefb']
    line_colors = ['#d32f2f', '#388e3c', '#1976d2']
    titles = ["Low Mastery (A)", "Mid Mastery (B)", "High Mastery (C)"]
    
    # 追踪当前出现了哪些知识点层级，以便按需显示图例
    used_skill_levels = set()

    for i, (ax, stu) in enumerate(zip(axes, stu_data_list[:3])):
        tau = stu['tau']
        
        # 1. 画ZPD背景带 (降低Alpha透明度，让主题更加突出)
        ax.axhspan(tau - 0.1, tau + 0.1, color=zone_colors[i], alpha=0.25, zorder=1)
        ax.axhline(tau, color=line_colors[i], linestyle='--', linewidth=2.5, zorder=2, label=f'Target $\\tau$ ({tau:.2f})')
        
        # 2. 画基准中心垂直参考线 x=0
        ax.axvline(0, color='#9e9e9e', linestyle=':', zorder=1, alpha=0.8)
        
        for p, sks, d in zip(stu['rec_probs'], stu['rec_skills'], stu['rec_global_diffs']):
            # 全局客观难度极值映射到 [-0.25, 0.25]
            offset = (d - d_min) / (d_max - d_min) * 0.5 - 0.25 if d_max > d_min else 0.0
            
            k = len(sks)
            # 指数级放大散点尺寸差距，视觉更震撼
            size = (k ** 1.3) * 100 
            
            if k == 1:
                c_key = 1
            elif k <= 3:
                c_key = 2
            else:
                c_key = 3
                
            used_skill_levels.add(c_key)
            c_val, l_val = color_map[c_key]
            
            # 使用高品质描边
            ax.scatter([offset], [p], color=c_val, s=size, edgecolors='white', linewidth=1.2, zorder=5, alpha=0.85)

        ax.set_xlim(-0.35, 0.35)
        # 精细化刻度与标签
        ax.set_xticks([-0.25, 0, 0.25])
        ax.set_xticklabels(['-0.25\n(Easier)', '0\n(Median)', '+0.25\n(Harder)'], fontsize=9, color='#616161')
        
        ax.set_title(titles[i], fontweight='bold', fontsize=13)
        ax.grid(True, axis='y', linestyle='-', alpha=0.3, color='#e0e0e0', zorder=0)

    # 精确设定Y轴
    axes[0].set_ylim(0.35, 0.70)
    axes[0].set_ylabel('Target Expected Mastery Probability ($\sim$ ZPD Alignment)', fontsize=12, weight='bold')
    
    fig.supxlabel('Relative Global Difficulty Offset (from q_features)', fontsize=13, weight='bold', y=0.0)
    fig.suptitle('Individual ZPD Constraints & Question Complexity Allocation', fontsize=16, weight='bold', y=1.02)
    
    # 3. 高级图例组合
    from matplotlib.lines import Line2D
    custom_lines = []
    
    # 动态添加气泡图例
    if 1 in used_skill_levels:
        custom_lines.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[1][0], markersize=10, label=color_map[1][1]))
    if 2 in used_skill_levels:
        custom_lines.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[2][0], markersize=14, label=color_map[2][1]))
    if 3 in used_skill_levels:
        custom_lines.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[3][0], markersize=18, label=color_map[3][1]))
        
    custom_lines.append(Line2D([0], [0], color='#424242', linestyle='--', linewidth=2, label='Target Optimal $\\tau$'))
    custom_lines.append(Line2D([0], [0], marker='s', color='w', markerfacecolor='#e0e0e0', markersize=15, label='ZPD Band ($\pm 0.1$)'))
    
    fig.legend(handles=custom_lines, loc='center left', bbox_to_anchor=(0.91, 0.5), fontsize=11, framealpha=0.9)
    plt.subplots_adjust(wspace=0.1, right=0.9) 
    
    plt.savefig(os.path.join(out_dir, f'zpd_alignment{suffix}.pdf'), dpi=300, bbox_inches='tight')
    logger.info(f"Saved zpd_alignment{suffix}.pdf")

def export_cases(stu_data_list, out_dir, loader, suffix=""):
    rows = []
    for stu in stu_data_list:
        skills_str = " | ".join([f"Q{q} ({','.join(map(str, sks))})" for q, sks in zip(stu['rec_ids'], stu['rec_skills'])])
        probs_str = ", ".join([f"{p:.3f}" for p in stu['rec_probs']])
        row = {
            'Student': f"Stu {stu['student_idx']}",
            'History Length': stu['history_len'],
            'ZPD Target (Tau)': f"{stu['tau']:.3f}",
            'Avg Mastery': f"{stu['m_k_mean']:.3f}",
            'Weak Skills Found': stu['weak_skills_count'],
            'Rec IDs': str(stu['rec_ids']),
            'Rec Probs (Targeting Tau)': probs_str,
            'Rec Skills (ID: Skills)': skills_str
        }
        rows.append(row)
        
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, f'case_study_exports{suffix}.csv'), index=False, encoding='utf-8-sig')
    logger.info(f"Saved case_study_exports{suffix}.csv")

def main():
    parser = argparse.ArgumentParser(description="Case Study: Pareto Front & ZPD Visualization")
    parser.add_argument("--seed", type=int, default=None,
                        help="固定随机种子 (如 --seed 42)。不指定时使用时间戳作为种子，可事后复现。")
    args = parser.parse_args()

    # ── 种子策略 ──
    if args.seed is not None:
        seed = args.seed
        suffix = f"_{seed}"
        logger.info(f"[Seed] 用户指定种子: {seed}")
    else:
        seed = int(datetime.now().strftime("%m%d%H%M%S"))  # 如 0310012345
        suffix = f"_{seed}"
        logger.info(f"[Seed] 自动生成时间戳种子: {seed} (可用 --seed {seed} 复现)")

    set_seed(seed)
    
    out_dir = os.path.join(_THIS_DIR, 'output', 'case_study')
    os.makedirs(out_dir, exist_ok=True)
    
    # Load system
    recommender, loader, config = load_system()
    # Expand pool to 300 to find trully representative archetypes
    df_samples = loader.get_eval_samples(300) 
    
    pool = []
    for idx, row in df_samples.iterrows():
        ans_rate = np.mean(row['r_seq'][:len(row['r_seq'])-5])
        hist_len = len(row['r_seq']) - 5
        pool.append({'idx': idx, 'acc': float(ans_rate), 'len': hist_len})
        
    df_pool = pd.DataFrame(pool)
    
    # Select archetypes strictly based on academic thresholds
    min_len = 40  # 提高历史长度门槛，确保评估的稳定性与真实性
    low_candidates = df_pool[(df_pool['acc'] < LOW_ACC_THRESHOLD) & (df_pool['len'] >= min_len)]
    idx_low = low_candidates.iloc[0]['idx'] if not low_candidates.empty else df_pool.nsmallest(1, 'acc').iloc[0]['idx']
    
    mid_candidates = df_pool[(df_pool['acc'] >= MID_ACC_LOWER) & (df_pool['acc'] <= MID_ACC_UPPER) & (df_pool['len'] >= min_len)]
    idx_mid = mid_candidates.iloc[0]['idx'] if not mid_candidates.empty else df_pool.iloc[(df_pool['acc']-0.55).abs().argsort()[:1]].iloc[0]['idx']
    
    high_candidates = df_pool[(df_pool['acc'] > HIGH_ACC_THRESHOLD) & (df_pool['len'] >= min_len)]
    idx_high = high_candidates.iloc[0]['idx'] if not high_candidates.empty else df_pool.nlargest(1, 'acc').iloc[0]['idx']
    
    target_indices = [int(idx_low), int(idx_mid), int(idx_high)]
    logger.info(f"Selected archetypes metrics:")
    for role, i in zip(['Low', 'Mid', 'High'], target_indices):
         logger.info(f"  {role} Mastery -> Idx: {i}, Acc: {df_pool.iloc[i]['acc']:.3f}, Len: {df_pool.iloc[i]['len']}")
    
    results = []
    for i, idx in enumerate(target_indices):
        logger.info(f"\nAnalyzing {['Low', 'Mid', 'High'][i]} student (Index {idx})...")
        res = analyze_student(recommender, loader, idx, df_samples.iloc[idx])
        results.append(res)
        
    plot_pareto(results, out_dir, suffix=suffix)
    plot_zpd(results, out_dir, suffix=suffix)
    export_cases(results, out_dir, loader, suffix=suffix)
    
if __name__ == "__main__":
    main()

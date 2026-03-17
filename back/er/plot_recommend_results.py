import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def set_style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)
    # 支持中文显示，同时指定英文使用 Times New Roman 会更美观
    plt.rcParams['font.family'] = ['Times New Roman', 'SimHei', 'Microsoft YaHei', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    try:
        plt.rcParams['text.usetex'] = False
    except:
        pass

def parse_data(csv_path):
    df = pd.read_csv(csv_path)
    return df

def length_alpha(base_alpha):
    return max(0.02, base_alpha * 0.4)

def plot_radar_chart(df, output_dir):
    """
    绘制多目标雷达图，非常适合展示多目标优化(MOPSO)能取得“帕累托占优”的综合优势。
    """
    metrics_map = {
        'dm': '难度适配度(DM)', 
        'wkc': '弱点知识覆盖率(WKC)', 
        'ild': '列表内多样性(ILD)',
        'novelty': '新颖性(Novelty)',
        'skill_hit_rate': '知识命中率(SHR)',
        'novelty': '新颖性(Novelty)'
    }
    avail_metrics = [m for m in metrics_map.keys() if m in df.columns]
    labels = [metrics_map[m] for m in avail_metrics]
    
    # 聚合均值数据
    mean_df = df.groupby('mode')[avail_metrics].mean().reset_index()
    mean_df = mean_df.fillna(0)
    
    std_df = df.groupby('mode')[avail_metrics].std().reset_index()
    std_df = std_df.fillna(0)
    
    print("\n--- 原始均值数据 ---")
    print(mean_df.to_string(index=False))

    # 【核心调整】数据 Min-Max 标准化展示外缘轮廓
    # 警告：此标准化仅为绘制雷达图形状使用，必须在论文题注中说明！
    plot_df = mean_df.copy()
    plot_std_df = std_df.copy()
    
    for col in avail_metrics:
        col_min = mean_df[col].min()
        col_max = mean_df[col].max()
        if col_max > col_min:
            scale = 0.9 / (col_max - col_min)
            plot_df[col] = 0.1 + (mean_df[col] - col_min) * scale
            # 方差进行同比例放缩，以保证置信区间的视觉比例正确
            plot_std_df[col] = std_df[col] * scale
        else:
            plot_df[col] = 1.0  
            plot_std_df[col] = 0.0
            
    print("\n--- 归一化后的数据 ---")
    print(plot_df.to_string(index=False))

    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # 增加加粗的全局标题
    plt.title("Comprehensive Trade-off (Radar Chart)", fontsize=18, fontweight='bold', pad=40)
    
    # 【恢复清新配色】参考之前经典的柔和配色，并新增消融实验的对应项
    color_dict = {
        'ours': '#e377c2',            # 粉粉嫩嫩的粉红色 (最上层)
        'full': '#e377c2',            # 消融实验中的 full (等同于 ours)
        'dkt_greedy': '#66c2a5',      # 护眼的青绿色/薄荷绿
        'no_pid': '#66c2a5',          # 消融：no_pid
        'dkvmn_greedy': '#fc8d62',    # 柔和的桃橘色
        'no_mopso': '#fc8d62',        # 消融：no_mopso
        'greedy': '#8da0cb',          # 优雅的灰蓝色/淡紫
        'no_f2': '#8da0cb',           # 消融：no_f2
        'popularity': '#a6d854',      # 清新的黄绿色
        'random': '#ffd92f'           # 明亮的黄色
    }
    
    # 根据之前的图设置不同 marker 形状以增加区分度
    marker_dict = {
        'ours': 'D',            # 菱形
        'full': 'D',
        'dkt_greedy': 'o',      # 圆形
        'no_pid': 'o',
        'dkvmn_greedy': 's',    # 正方形
        'no_mopso': 's',
        'greedy': '^',          # 正三角
        'no_f2': '^',
        'popularity': 'v',      # 倒三角
        'random': '*'           # 星形
    }
    
    # 针对遮挡问题：为模型配置不同的线型
    ls_dict = {
        'ours': '-',
        'full': '-',
        'dkt_greedy': '--',
        'no_pid': '--',
        'dkvmn_greedy': '-.',
        'no_mopso': '-.',
        'greedy': ':',
        'no_f2': ':',
        'popularity': '--',
        'random': '-.'
    }
    
    # 制定绘制层级和样式，确保 OURS/FULL 覆盖在最上层
    for idx, row in plot_df.iterrows():
        mode_id = row['mode'].lower()
        values = row[avail_metrics].values.tolist()
        values += values[:1]
        
        mode_name_disp = row['mode'].upper()
        
        color = color_dict.get(mode_id, '#999999')
        marker = marker_dict.get(mode_id, 'o')
        ls = ls_dict.get(mode_id, '-')
        
        if mode_id in ['ours', 'full']:
            lw = 2.5
            alpha = 0.15
            zorder = 10
            markersize = 8
            mfc = color  # 实心
        else:
            lw = 1.5
            alpha = 0.05
            zorder = 2
            markersize = 6
            mfc = 'none' # 空心，透视后面的线
        
        ax.plot(angles, values, color=color, linewidth=lw, label=mode_name_disp, 
                linestyle=ls, marker=marker, markersize=markersize, 
                markerfacecolor=mfc, markeredgewidth=1.2, zorder=zorder)
                
        # 添加带方差的置信带(Shaded Area) 或 误差棒 (Error Bars)
        std_row = plot_std_df[plot_std_df['mode'].str.lower() == mode_id].iloc[0]
        stds = std_row[avail_metrics].values.tolist()
        stds += stds[:1]
        
        # 方案B: 如果是核心模型，画一个明显一点的晕影（半透明带）
        if mode_id in ['ours', 'full']:
            upper_bound = np.clip(np.array(values) + np.array(stds), 0, 1)
            lower_bound = np.clip(np.array(values) - np.array(stds), 0, 1)
            ax.fill_between(angles, lower_bound, upper_bound, color=color, alpha=0.3, zorder=zorder-1)
            
            # 并在各个顶点画上 errorbar
            ax.errorbar(angles, values, yerr=stds, fmt='none', ecolor=color, elinewidth=2, capsize=4, zorder=zorder+1)
        
        ax.fill(angles, values, color=color, alpha=alpha, zorder=zorder)
        
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # 扩大 y_lim 给外围标签留出缓冲空间，防止由于数据直接触顶导致的挤压重叠
    ax.set_ylim(0, 1.15)
    
    ax.set_xticks(angles[:-1])
    # 增加 pad 参数，同时加上一定大小和家族字体设定，避免中文覆盖
    ax.tick_params(axis='x', pad=25) 
    ax.set_xticklabels(labels, fontsize=13, fontweight='bold', color='#333333', family='SimHei')
    
    # 用虚线重新画同心圆，显得更通透，同时提高 zorder 确保网格线在色块之上
    ax.yaxis.grid(True, linestyle='--', color='gray', alpha=0.5, zorder=15)
    ax.xaxis.grid(True, linestyle='-', color='gray', alpha=0.3, zorder=15)
    
    # 恢复带层次的同心圆内部网格标注 (0.2, 0.4, 0.6, 0.8, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    y_labels = ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], color="dimgrey", size=10, fontweight='bold')
    for label in y_labels:
        label.set_zorder(20)  # 【解决由于遮挡导致字体不可见的核心】提高字体层级
        
    # 将内部标签的角度旋转到斜右上方以免被正上方的线条挡住
    ax.set_rlabel_position(75) 
    
    # 图例设计，将其稍微向左调或者向上调避免由于扩大了y_lim导致的图例脱离
    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), frameon=True, fontsize=11, prop={'family': 'Times New Roman', 'size': 11})
    
    plt.savefig(os.path.join(output_dir, 'recommendation_radar.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Enhanced Radar chart saved to {os.path.join(output_dir, 'recommendation_radar.pdf')}")

if __name__ == "__main__":
    set_style()
    csv_path = r"output/recommendation_full/recommend_eval_full_all.bak-论文当前用图.csv"
    # csv_path = r"output/ablation_er/ablation_eval_full.bak-论文当前用图.csv"
    if not os.path.exists(csv_path):
        import glob
        files = glob.glob(r"output/recommendation_*/recommend_eval_*.csv")
        if files:
            csv_path = files[0]
        else:
            print("❌ 找不到 CSV 评估结果。请先执行评估")
            exit(1)
            
    out_dir = os.path.dirname(csv_path)
    df = parse_data(csv_path)
    
    # 保留雷达图绘制
    plot_radar_chart(df, out_dir)
    print("🎉 雷达图已生成！")

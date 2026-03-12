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
    try:
        plt.rcParams['text.usetex'] = False
    except:
        pass

def parse_data(csv_path):
    df = pd.read_csv(csv_path)
    return df

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
    
    print("\n--- 原始均值数据 ---")
    print(mean_df.to_string(index=False))

    # 【核心调整】数据 Min-Max 标准化展示外缘轮廓
    # 警告：此标准化仅为绘制雷达图形状使用，必须在论文题注中说明！
    plot_df = mean_df.copy()
    for col in avail_metrics:
        col_min = plot_df[col].min()
        col_max = plot_df[col].max()
        if col_max > col_min:
            plot_df[col] = 0.1 + 0.9 * (plot_df[col] - col_min) / (col_max - col_min)
        else:
            plot_df[col] = 1.0  
            
    print("\n--- 归一化后的数据 ---")
    print(plot_df.to_string(index=False))

    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # 增加加粗的全局标题
    plt.title("Comprehensive Trade-off (Radar Chart)", fontsize=18, fontweight='bold', pad=40)
    
    # 【恢复清新配色】参考之前经典的柔和配色
    color_dict = {
        'ours': '#e377c2',            # 粉粉嫩嫩的粉红色 (最上层)
        'dkt_greedy': '#66c2a5',      # 护眼的青绿色/薄荷绿
        'dkvmn_greedy': '#fc8d62',    # 柔和的桃橘色
        'greedy': '#8da0cb',          # 优雅的灰蓝色/淡紫
        'popularity': '#a6d854',      # 清新的黄绿色
        'random': '#ffd92f'           # 明亮的黄色
    }
    
    # 根据之前的图设置不同 marker 形状以增加区分度
    marker_dict = {
        'ours': 'D',            # 菱形
        'dkt_greedy': 'o',      # 圆形
        'dkvmn_greedy': 's',    # 正方形
        'greedy': '^',          # 正三角
        'popularity': 'v',      # 倒三角
        'random': '*'           # 星形
    }
    
    # 制定绘制层级和样式，确保 OURS 覆盖在最上层
    for idx, row in plot_df.iterrows():
        mode_id = row['mode'].lower()
        values = row[avail_metrics].values.tolist()
        values += values[:1]
        
        mode_name_disp = row['mode'].upper()
        
        color = color_dict.get(mode_id, '#999999')
        marker = marker_dict.get(mode_id, 'o')
        
        if mode_id == 'ours':
            lw = 2.5
            alpha = 0.15
            zorder = 10
            markersize = 8
        else:
            lw = 1.5
            alpha = 0.05
            zorder = 2
            markersize = 6
        
        # 统一使用实线，视觉较虚线更连贯、清新
        ls = '-'
        
        ax.plot(angles, values, color=color, linewidth=lw, label=mode_name_disp, 
                linestyle=ls, marker=marker, markersize=markersize, zorder=zorder)
        ax.fill(angles, values, color=color, alpha=alpha, zorder=zorder)
        
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # 扩大 y_lim 给外围标签留出缓冲空间，防止由于数据直接触顶导致的挤压重叠
    ax.set_ylim(0, 1.15)
    
    ax.set_xticks(angles[:-1])
    # 增加 pad 参数，同时加上一定大小和家族字体设定，避免中文覆盖
    ax.tick_params(axis='x', pad=25) 
    ax.set_xticklabels(labels, fontsize=13, fontweight='bold', color='#333333', family='SimHei')
    
    # 用虚线重新画同心圆，显得更通透
    ax.yaxis.grid(True, linestyle='--', color='gray', alpha=0.5)
    
    # 恢复带层次的同心圆内部网格标注 (0.2, 0.4, 0.6, 0.8, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], color="grey", size=10)
    # 将内部标签的角度旋转到斜右上方以免被正上方的线条挡住
    ax.set_rlabel_position(75) 
    
    # 图例设计，将其稍微向左调或者向上调避免由于扩大了y_lim导致的图例脱离
    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), frameon=True, fontsize=11, prop={'family': 'Times New Roman', 'size': 11})
    
    plt.savefig(os.path.join(output_dir, 'recommendation_radar.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Enhanced Radar chart saved to {os.path.join(output_dir, 'recommendation_radar.pdf')}")

if __name__ == "__main__":
    set_style()
    csv_path = r"output/recommendation_full/recommend_eval_full_all.csv"
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

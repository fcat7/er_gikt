import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ================= 配置区 =================
LANGUAGE = "zh"  # 可选: "zh" | "en" 
USE_TEX = False  # 建议 False。如果在英文版下确实想要 \textsc 效果再开启。
# ==========================================

def set_style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)
    
    # 彻底解决中文不矢量的问题：强制保存为 TrueType 字体 (Type 42)
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    if LANGUAGE == "zh":
        plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "Times New Roman", "sans-serif"]
    else:
        plt.rcParams["font.family"] = ["Times New Roman", "sans-serif"]
    
    plt.rcParams["axes.unicode_minus"] = False
    
    global USE_TEX
    
    if USE_TEX and LANGUAGE == "en":
        try:
            plt.rcParams["text.usetex"] = True
            plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath} \usepackage{amssymb}"
        except:
            pass
    elif USE_TEX and LANGUAGE == "zh":
        print("警告: 开启了 USE_TEX 但 LANGUAGE 为 zh。LaTeX 渲染中文常常因缺少系统宏包而失败。已自动为您回退为非 LaTeX 渲染以确保矢量中文输出。")
        USE_TEX = False

def plot_radar_chart(df, output_dir):
    # 根据中英文彻底分离文案
    if LANGUAGE == "zh":
        dataset_map = {
            "assist09": "Assist2009",
            "assist12": "Assist2012",
            "assist17": "Assist2017",
            "ednet_kt1": "EdNet-KT1",
            "nips2020_task34": "Nips2020",
        }
        model_display_map = {
            "A_Baseline": "完整模型 (Ours)",
            "G_remove_PID_Cognitive": "仅 +IRT",
            "H_remove_PID_IRT": "仅 +Cog.",
            "I_remove_Cognitive_IRT": "仅 +PID",
            "D_Remove_IRT": "移除 IRT",
            "C_Remove_Cognitive": "移除 Cog.",
            "B_Remove_PID": "移除 PID",
            "F_old_gikt": "基线 (Base-GIKT)",
        }
        fig_title = "各个变体在不同数据集上的消融实验表现"
        left_subtitle = "向上搭建 (Incremental: Base -> \u2191)"
        right_subtitle = "向下拆解 (Decremental: Ours -> \u2193)"
        note_text = "(注：为凸显微小的性能变化，坐标轴已按数据集进行 Min-Max 归一化缩放)"
    else:
        if USE_TEX:
            dataset_map = {
                "assist09": r"\textbf{\textsc{Assist2009}}",
                "assist12": r"\textbf{\textsc{Assist2012}}",
                "assist17": r"\textbf{\textsc{Assist2017}}",
                "ednet_kt1": r"\textbf{\textsc{EdNet-KT1}}",
                "nips2020_task34": r"\textbf{\textsc{Nips2020}}",
            }
        else:
            dataset_map = {
                "assist09": "Assist2009",
                "assist12": "Assist2012",
                "assist17": "Assist2017",
                "ednet_kt1": "EdNet-KT1",
                "nips2020_task34": "Nips2020",
            }
        model_display_map = {
            "A_Baseline": "FA-GIKT (Ours)",
            "G_remove_PID_Cognitive": "+ IRT",
            "H_remove_PID_IRT": "+ Cog.",
            "I_remove_Cognitive_IRT": "+ PID",
            "D_Remove_IRT": "w/o IRT",
            "C_Remove_Cognitive": "w/o Cog.",
            "B_Remove_PID": "w/o PID",
            "F_old_gikt": "Base-GIKT",
        }
        fig_title = "Ablation Study Trade-off (Radar Chart)"
        left_subtitle = "Incremental Building (Base -> \u2191)"
        right_subtitle = "Decremental Ablation (Ours -> \u2193)"
        note_text = "(Note: Axes are min-max normalized per dataset to highlight AUC differences)"

    # 过滤重命名
    df = df[df["group"].isin(model_display_map.keys()) & df["dataset"].isin(dataset_map.keys())].copy()
    df["model_name"] = df["group"].map(model_display_map)
    df["ds_name"] = df["dataset"].map(dataset_map)
    
    pivot_df = df.pivot_table(index="model_name", columns="ds_name", values="auc", aggfunc="mean")
    cols = [dataset_map[k] for k in dataset_map.keys() if dataset_map[k] in pivot_df.columns]
    pivot_df = pivot_df[cols]
    
    labels = cols
    num_vars = len(labels)
    
    # 归一化
    plot_df = pd.DataFrame(index=pivot_df.index)
    for col in cols:
        col_min = pivot_df[col].min() - 0.005
        col_max = pivot_df[col].max() + 0.005
        if col_max > col_min:
            plot_df[col] = 0.1 + (pivot_df[col] - col_min) / (col_max - col_min) * 0.9
        else:
            plot_df[col] = 1.0

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), subplot_kw=dict(polar=True))
    fig.suptitle(fig_title, fontsize=20, fontweight="bold", y=1.05)
    
    # 获取各个名称的实际文本
    name_ours = model_display_map["A_Baseline"]
    name_base = model_display_map["F_old_gikt"]
    
    name_add_irt = model_display_map["G_remove_PID_Cognitive"]
    name_add_cog = model_display_map["H_remove_PID_IRT"]
    name_add_pid = model_display_map["I_remove_Cognitive_IRT"]
    
    name_rm_irt = model_display_map["D_Remove_IRT"]
    name_rm_cog = model_display_map["C_Remove_Cognitive"]
    name_rm_pid = model_display_map["B_Remove_PID"]

    color_dict = {
        name_ours: "#e377c2",     # 粉色
        name_add_irt: "#a6d854",  # 黄绿色
        name_add_cog: "#fc8d62",  # 桃橘色
        name_add_pid: "#8da0cb",  # 灰蓝色
        name_rm_irt: "#66c2a5",   # 护眼青绿
        name_rm_cog: "#fc8d62",   # 桃橘色
        name_rm_pid: "#8da0cb",   # 灰蓝色
        name_base: "#999999",     # 灰色
    }
    
    marker_dict = {
        name_ours: "D", 
        name_add_irt: "o", name_add_cog: "s", name_add_pid: "^",
        name_rm_irt: "o", name_rm_cog: "s", name_rm_pid: "^", 
        name_base: "*"
    }

    # ================= 顶刊排版学审美：左图更侧重向上搭建 =================
    ax_left = axes[0]
    ax_left.set_title(left_subtitle, fontsize=16, pad=20, fontweight="bold")
    
    # 绘制顺序：Full最底(虚线不挡人) -> 中间增益层 -> Base基底
    draw_order_left = [name_ours, name_base, name_add_pid, name_add_cog, name_add_irt]
    for mode_name in draw_order_left:
        if mode_name not in plot_df.index: continue
        values = plot_df.loc[mode_name].tolist()
        values += values[:1]
        
        color, marker = color_dict.get(mode_name, "#999"), marker_dict.get(mode_name, "o")
        
        if mode_name == name_ours:
            # 目标轮廓线：粉色虚线外框，不填充，放在最底下防止遮挡
            lw, alpha, zorder, markersize, mfc, ls = 2.0, 0.0, 2, 8, "none", "--"
        elif mode_name == name_base:
            # 起点基座：坚固的灰色实心底盘填充
            lw, alpha, zorder, markersize, mfc, ls = 1.5, 0.35, 1, 6, color, "-"
        else:
            # 增益向上搭建过程：有颜色的半透明叠加带
            lw, alpha, zorder, markersize, mfc, ls = 1.8, 0.15, 5, 7, color, "-"
            
        ax_left.plot(angles, values, color=color, linewidth=lw, label=mode_name, linestyle=ls, marker=marker, markersize=markersize, markerfacecolor=mfc, markeredgewidth=1.2, zorder=zorder)
        if alpha > 0: ax_left.fill(angles, values, color=color, alpha=alpha, zorder=zorder)
            
    # ================= 顶刊排版学审美：右图更侧重向下拆解 =================
    ax_right = axes[1]
    ax_right.set_title(right_subtitle, fontsize=16, pad=20, fontweight="bold")
    
    # 绘制顺序：Base最底(虚线下限) -> 中间压降层 -> Full巨型基底
    draw_order_right = [name_base, name_ours, name_rm_pid, name_rm_cog, name_rm_irt]
    for mode_name in draw_order_right:
        if mode_name not in plot_df.index: continue
        values = plot_df.loc[mode_name].tolist()
        values += values[:1]
        
        color, marker = color_dict.get(mode_name, "#999"), marker_dict.get(mode_name, "o")
        
        if mode_name == name_ours:
            # 现有大盘基座：粉色实心大面积填充
            lw, alpha, zorder, markersize, mfc, ls = 2.0, 0.2, 1, 8, color, "-"
        elif mode_name == name_base:
            # 退化下限轮廓：灰色粗虚线框，不填充
            lw, alpha, zorder, markersize, mfc, ls = 1.5, 0.0, 2, 6, "none", "--"
        else:
            # 向下拆解凹陷：有颜色的实线，无填充以凸显凹陷区域
            lw, alpha, zorder, markersize, mfc, ls = 2.0, 0.0, 5, 7, "white", "-"
            
        ax_right.plot(angles, values, color=color, linewidth=lw, label=mode_name, linestyle=ls, marker=marker, markersize=markersize, markerfacecolor=mfc, markeredgewidth=1.2, zorder=zorder)
        if alpha > 0: ax_right.fill(angles, values, color=color, alpha=alpha, zorder=zorder)

    # 统一设置坐标系
    for ax in axes:
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 1.15)
        ax.set_xticks(angles[:-1])
        ax.tick_params(axis="x", pad=25) 
        
        if not USE_TEX:
            ax.set_xticklabels(labels, fontsize=14, fontweight="bold", color="#333333")
        else:
            ax.set_xticklabels(labels, fontsize=14, color="#333333")
            
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        y_labels = ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="dimgrey", size=10)
        for label in y_labels:
            label.set_zorder(20)
            if not USE_TEX: label.set_fontweight("bold")
        ax.set_rlabel_position(75) 
        
        ax.yaxis.grid(True, linestyle="--", color="gray", alpha=0.5, zorder=15)
        ax.xaxis.grid(True, linestyle="-", color="gray", alpha=0.3, zorder=15)
        
        # 图例放底下
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=True, fontsize=12)

    plt.figtext(0.5, -0.05, note_text, ha="center", fontsize=11, style="italic", color="dimgrey")
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, bottom=0.2)
    
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, "4_ablation_radar.pdf")
    png_path = os.path.join(output_dir, "4_ablation_radar.png")
    
    try:
        plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
        plt.savefig(png_path, dpi=300, bbox_inches="tight")
        print(f" 雷达图已保存至: {pdf_path} 和 {png_path}")
    except Exception as e:
        print(f" 绘图失败: {e}")
    finally:
        plt.close()

def main():
    csv_path = r"H:\er_gikt\back\kt\scripts\ablation_summary.csv"
    if not os.path.exists(csv_path):
        print(f" 找不到输入文件：{csv_path}")
        return
        
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except:
        df = pd.read_csv(csv_path, encoding="gbk")
        
    df = df.rename(columns={c: "group" for c in ["消融组", "Ablation Group", "group", "ablation_group"] if c in df.columns})
    df = df.rename(columns={c: "dataset" for c in ["数据集名称", "Dataset", "dataset", "dataset_name", "数据集"] if c in df.columns})
    df = df.rename(columns={c: "auc" for c in ["Test_AUC", "Mean Test AUC", "test_auc", "AUC", "auc"] if c in df.columns})
    
    if not all(col in df.columns for col in ["group", "dataset", "auc"]):
        print(" CSV 列不匹配:", df.columns)
        return
        
    df["auc"] = pd.to_numeric(df["auc"], errors="coerce")
    out_dir = os.path.dirname(csv_path)
    plot_radar_chart(df, out_dir)

if __name__ == "__main__":
    set_style()
    main()


import os
from collections import OrderedDict

import pandas as pd

# ================= 配置区 =================
INPUT_CSV = r"H:\er_gikt\back\kt\scripts\ablation_summary.csv"
OUTPUT_TEX = r"H:\er_gikt\back\kt\scripts\04_ablation_table.tex"
SHOW_ASSIST09_BUILDER = False
LANGUAGE = "zh"  # 可选: "zh" | "en" | "bilingual"

# 数据集简称字典：按字典顺序输出列顺序
DATASET_NAME_MAP = OrderedDict({
    "assist09": r"\textsc{Assist2009}",
    "assist09_builder": r"\textsc{Assist2009-Builder}",
    "assist12": r"\textsc{Assist2012}",
    "assist17": r"\textsc{Assist2017}",
    "ednet_kt1": r"\textsc{EdNet-Kt1}",
    "nips2020_task34": r"\textsc{Nips2020-Task3/4}",
})

# 消融组别分组字典：按组输出，组间加 \midrule
ABLATION_GROUPS = [
    {
        "group_desc": "基线",
        "items": OrderedDict({
            "F_old_gikt": "Base-GIKT",
        })
    },
    {
        "group_desc": "单模块增益 (对比 Base-GIKT 看提升)",
        "items": OrderedDict({
            "I_remove_Cognitive_IRT": "+ PID",
            "H_remove_PID_IRT": "+ Cog.",
            "G_remove_PID_Cognitive": "+ IRT",
        })
    },
    {
        "group_desc": "单模块消融 (对比完整版看压降)",
        "items": OrderedDict({
            "B_Remove_PID": "w/o PID",
            "C_Remove_Cognitive": "w/o Cog.",
            "D_Remove_IRT": "w/o IRT",
        })
    },
    {
        "group_desc": "完整模型",
        "items": OrderedDict({
            "A_Baseline": "FA-GIKT (Ours)",
        })
    }
]

# 扁平化映射用于兼容后续查找
ABLATION_NAME_MAP = OrderedDict()
for g in ABLATION_GROUPS:
    for k, v in g["items"].items():
        ABLATION_NAME_MAP[k] = v

TABLE_CAPTION_ZH = "消融实验在不同数据集上的性能比较"
TABLE_CAPTION_EN = "Performance comparison of ablation variants across different datasets"
TABLE_LABEL = "tab:ablation_results-kt"
# ==========================================

# 支持原始汇总CSV列名和提取后的列名
GROUP_COL_CANDIDATES = ["消融组", "Ablation Group", "group", "ablation_group"]
DATASET_COL_CANDIDATES = ["数据集名称", "Dataset", "dataset", "dataset_name", "数据集"]
AUC_COL_CANDIDATES = ["Test_AUC", "Mean Test AUC", "test_auc", "AUC", "auc"]
ACC_COL_CANDIDATES = ["Test_ACC", "Mean Test ACC", "test_acc", "ACC", "acc"]


def find_column(df, candidates):
    for name in candidates:
        if name in df.columns:
            return name
    return None


def latex_escape(text):
    replacements = {
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }
    text = str(text)
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def render_dataset_label(dataset_key):
    if dataset_key in DATASET_NAME_MAP:
        return DATASET_NAME_MAP[dataset_key]
    return latex_escape(dataset_key)


def build_caption():
    if LANGUAGE == "zh":
        return TABLE_CAPTION_ZH
    if LANGUAGE == "en":
        return TABLE_CAPTION_EN
    return TABLE_CAPTION_ZH + r" / " + TABLE_CAPTION_EN


def build_note_text():
    note_zh = r"注：粗体表示最优结果，下划线表示次优结果。AUC 和 ACC 分别对应测试集上的预测性能。Base-GIKT 为仅保留基础图网络传播的改进基线；+ 前缀表示在基线上单独增加该模块，w/o 前缀表示从 FA-GIKT 中移除对应模块（PID：个性化练习表现追踪；Cog.：认知状态RNN模块；IRT：题目特征与反馈融合）。"
    note_en = r"Note: Bold indicates the best result, and underlining indicates the second-best result. AUC and ACC denote the prediction performance on the test set. Base-GIKT is the strengthened baseline retaining only basic graph propagation; '+' indicates adding the module to the baseline separately, while 'w/o' denotes removing the corresponding module from FA-GIKT (PID: Personalized representation; Cog.: Cognitive state RNN; IRT: Item feature and feedback fusion)."
    if LANGUAGE == "zh":
        return r"{\footnotesize " + note_zh + r"}"
    if LANGUAGE == "en":
        return r"{\footnotesize " + note_en + r"}"
    return r"{\footnotesize " + note_zh + r"\\" + note_en + r"}"


def format_metric_ranked(value, all_vals):
    if pd.isna(value):
        return "-"
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return "-"

    formatted = f"{fv:.4f}"
    valid = sorted({float(v) for v in all_vals if not pd.isna(v)}, reverse=True)
    if len(valid) == 0:
        return formatted
    if abs(fv - valid[0]) < 1e-9:
        return r"\textbf{" + formatted + r"}"
    if len(valid) >= 2 and abs(fv - valid[1]) < 1e-9:
        return r"\underline{" + formatted + r"}"
    return formatted


def build_latex_table(df):
    group_col = find_column(df, GROUP_COL_CANDIDATES)
    dataset_col = find_column(df, DATASET_COL_CANDIDATES)
    auc_col = find_column(df, AUC_COL_CANDIDATES)
    acc_col = find_column(df, ACC_COL_CANDIDATES)

    missing = []
    if group_col is None:
        missing.append("消融组列")
    if dataset_col is None:
        missing.append("数据集列")
    if auc_col is None:
        missing.append("AUC列")
    if acc_col is None:
        missing.append("ACC列")
    if missing:
        raise ValueError(f"输入CSV缺少必要列：{', '.join(missing)}")

    work_df = df[[group_col, dataset_col, auc_col, acc_col]].copy()
    work_df.columns = ["group", "dataset", "auc", "acc"]
    work_df["auc"] = pd.to_numeric(work_df["auc"], errors="coerce")
    work_df["acc"] = pd.to_numeric(work_df["acc"], errors="coerce")

    dataset_set = set(work_df["dataset"].astype(str))
    dataset_keys = [key for key in DATASET_NAME_MAP.keys() if key in dataset_set]
    if not SHOW_ASSIST09_BUILDER:
        dataset_keys = [key for key in dataset_keys if key != "assist09_builder"]
    extra_datasets = [
        key for key in work_df["dataset"].astype(str).unique().tolist()
        if key not in DATASET_NAME_MAP
    ]
    if not SHOW_ASSIST09_BUILDER:
        extra_datasets = [key for key in extra_datasets if key != "assist09_builder"]
    dataset_keys.extend(sorted(extra_datasets))

    group_set = set(work_df["group"].astype(str))
    # 严格按照字典中存在的键进行输出
    group_keys = [key for key in ABLATION_NAME_MAP.keys() if key in group_set]

    auc_pivot = work_df.pivot_table(index="group", columns="dataset", values="auc", aggfunc="first")
    acc_pivot = work_df.pivot_table(index="group", columns="dataset", values="acc", aggfunc="first")

    col_all_auc = {}
    col_all_acc = {}
    for ds in dataset_keys:
        col_all_auc[ds] = [
            auc_pivot.loc[g, ds]
            for g in group_keys
            if g in auc_pivot.index and ds in auc_pivot.columns
        ]
        col_all_acc[ds] = [
            acc_pivot.loc[g, ds]
            for g in group_keys
            if g in acc_pivot.index and ds in acc_pivot.columns
        ]

    col_spec = "l" + "cc" * len(dataset_keys)

    lines = []
    lines.append(r"\begin{table*}[htbp]")
    lines.append(r"    \centering")
    lines.append(f"    \\caption{{{build_caption()}}}")
    lines.append(f"    \\label{{{TABLE_LABEL}}}")
    lines.append(r"    \resizebox{\textwidth}{!}{%")
    lines.append(f"    \\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"        \toprule")

    group_header_text = "Ablation Group" if LANGUAGE == "en" else "消融组"
    header_top = [rf"        \multirow{{2}}{{*}}{{{group_header_text}}}"]
    for dataset in dataset_keys:
        header_top.append(rf"\multicolumn{{2}}{{c}}{{{render_dataset_label(dataset)}}}")
    lines.append(" & ".join(header_top) + r" \\")

    cmidrules = []
    start_col = 2
    for _ in dataset_keys:
        cmidrules.append(rf"\cmidrule(lr){{{start_col}-{start_col + 1}}}")
        start_col += 2
    lines.append("        " + " ".join(cmidrules))

    header_bottom = ["        "]
    for _ in dataset_keys:
        header_bottom.extend(["AUC", "ACC"])
    lines.append(" & ".join(header_bottom) + r" \\")
    lines.append(r"        \midrule")

    for i, group_info in enumerate(ABLATION_GROUPS):
        group_items = group_info["items"]
        # 提取当前分组中确实出现在数据里的键
        valid_keys = [k for k in group_items.keys() if k in group_set]
        
        if not valid_keys:
            continue
            
        for group in valid_keys:
            group_display = ABLATION_NAME_MAP.get(group, group)
            row = [f"        {latex_escape(group_display)}"]
            for dataset in dataset_keys:
                auc_val = auc_pivot.loc[group, dataset] if group in auc_pivot.index and dataset in auc_pivot.columns else pd.NA
                acc_val = acc_pivot.loc[group, dataset] if group in acc_pivot.index and dataset in acc_pivot.columns else pd.NA
                row.append(format_metric_ranked(auc_val, col_all_auc.get(dataset, [])))
                row.append(format_metric_ranked(acc_val, col_all_acc.get(dataset, [])))
            lines.append(" & ".join(row) + r" \\")
            
        # 如果不是最后一个有效组，则插入一条细横线实现层次感
        if i < len(ABLATION_GROUPS) - 1:
            lines.append(r"        \midrule")

    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}%")
    lines.append(r"    }")
    lines.append(r"    \vspace{2pt}")
    lines.append(r"    \begin{flushleft}")
    lines.append("    " + build_note_text())
    lines.append(r"    \end{flushleft}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def main():
    if not os.path.exists(INPUT_CSV):
        print(f"❌ 找不到输入文件：{INPUT_CSV}")
        return

    df = None
    encodings = ["utf-8", "utf-8-sig", "gb2312", "gbk", "latin-1", "cp1252"]
    for encoding in encodings:
        try:
            df = pd.read_csv(INPUT_CSV, encoding=encoding)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception as exc:
            print(f"❌ 读取 CSV 失败（编码 {encoding}）：{exc}")
            return

    if df is None:
        print(f"❌ 无法读取 CSV：尝试了所有编码 {encodings}，均失败")
        return

    if df.empty:
        print(f"⚠️ 输入文件为空：{INPUT_CSV}")
        return

    try:
        latex_code = build_latex_table(df)
    except Exception as exc:
        print(f"❌ 生成 LaTeX 失败：{exc}")
        return

    with open(OUTPUT_TEX, "w", encoding="utf-8-sig") as f:
        f.write(latex_code)

    print("✅ 消融实验 LaTeX 表格已生成")
    print(f"输入文件：{INPUT_CSV}")
    print(f"输出文件：{OUTPUT_TEX}")
    print("\n===== 预览 =====\n")
    print(latex_code)


if __name__ == "__main__":
    main()

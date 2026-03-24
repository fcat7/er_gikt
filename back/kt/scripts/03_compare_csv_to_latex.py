import os
from collections import OrderedDict

import pandas as pd

# ================= 配置区 =================
INPUT_CSV = r"H:\er_gikt\back\kt\scripts\compare.csv"
OUTPUT_TEX = r"H:\er_gikt\back\kt\scripts\03_compare_table.tex"

# 数据集简称字典：按字典顺序输出列顺序
DATASET_NAME_MAP = OrderedDict({
    "assist09": r"\textsc{Assist2009}",
    # "assist09_builder": r"\textsc{Assist2009-Builder}",
    "assist12": r"\textsc{Assist2012}",
    "assist17": r"\textsc{Assist2017}",
    "ednet_kt1": r"\textsc{EdNet-Kt1}",
    "nips2020_task34": r"\textsc{Nips2020-Task3/4}",
})

# 模型简称字典：按字典顺序输出行顺序
MODEL_NAME_MAP = OrderedDict({
    "dkt": "DKT",
    "simplekt": "SimpleKT",
    "akt": "AKT",
    "deep_irt": "Deep-IRT",
    "dkvmn": "DKVMN",
    "gikt_old": "GIKT",
    "fa_gikt": "FA-GIKT",
})

TABLE_CAPTION = "不同模型在不同数据集上的预测性能比较"
TABLE_LABEL = "tab:kt_compare_results"
DELTA_SUMMARY_MODE = "mean"  # 顶刊主表风格：仅在表注中概述平均提升
# ==========================================


def find_column(df, candidates):
    """从候选列名中找到第一个存在的列。"""
    for name in candidates:
        if name in df.columns:
            return name
    return None


MODEL_COL_CANDIDATES = ["模型名称", "Model", "model", "model_name", "模型"]
DATASET_COL_CANDIDATES = ["数据集名称", "Dataset", "dataset", "dataset_name", "数据集"]
AUC_COL_CANDIDATES = ["Test_AUC", "test_auc", "AUC", "auc"]
ACC_COL_CANDIDATES = ["Test_ACC", "test_acc", "ACC", "acc"]


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
    """已在字典中的数据集简称允许直接写 LaTeX；未知名称再做转义。"""
    if dataset_key in DATASET_NAME_MAP:
        return DATASET_NAME_MAP[dataset_key]
    return latex_escape(dataset_key)


def format_metric(value):
    """不带排名的简单格式化。"""
    if pd.isna(value):
        return "-"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "-"


def format_metric_ranked(value, all_vals):
    """格式化指标值：最优 \\textbf，次优 \\underline。"""
    if pd.isna(value):
        return "-"
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return "-"
    formatted = f"{fv:.4f}"
    valid = sorted([float(v) for v in all_vals if not pd.isna(v)], reverse=True)
    if len(valid) == 0:
        return formatted
    if abs(fv - valid[0]) < 1e-9:
        return r'\textbf{' + formatted + r'}'
    if len(valid) >= 2 and abs(fv - valid[1]) < 1e-9:
        return r'\underline{' + formatted + r'}'
    return formatted


def compute_delta_summary(auc_pivot, acc_pivot, dataset_keys, fa_key="fa_gikt", base_key="gikt_old"):
    """返回 FA-GIKT 相对 GIKT 的平均绝对提升和平均相对提升。"""
    if fa_key not in auc_pivot.index or base_key not in auc_pivot.index:
        return None

    auc_diffs, auc_pcts = [], []
    acc_diffs, acc_pcts = [], []

    for dataset in dataset_keys:
        if dataset in auc_pivot.columns:
            fa_auc = auc_pivot.loc[fa_key, dataset]
            base_auc = auc_pivot.loc[base_key, dataset]
            if not pd.isna(fa_auc) and not pd.isna(base_auc):
                diff = float(fa_auc) - float(base_auc)
                auc_diffs.append(diff)
                auc_pcts.append(diff / abs(float(base_auc)) * 100 if float(base_auc) != 0 else 0.0)

        if dataset in acc_pivot.columns:
            fa_acc = acc_pivot.loc[fa_key, dataset]
            base_acc = acc_pivot.loc[base_key, dataset]
            if not pd.isna(fa_acc) and not pd.isna(base_acc):
                diff = float(fa_acc) - float(base_acc)
                acc_diffs.append(diff)
                acc_pcts.append(diff / abs(float(base_acc)) * 100 if float(base_acc) != 0 else 0.0)

    if not auc_diffs and not acc_diffs:
        return None

    return {
        "auc_diff": sum(auc_diffs) / len(auc_diffs) if auc_diffs else None,
        "auc_pct": sum(auc_pcts) / len(auc_pcts) if auc_pcts else None,
        "acc_diff": sum(acc_diffs) / len(acc_diffs) if acc_diffs else None,
        "acc_pct": sum(acc_pcts) / len(acc_pcts) if acc_pcts else None,
    }


def build_latex_table(df):
    model_col = find_column(df, MODEL_COL_CANDIDATES)
    dataset_col = find_column(df, DATASET_COL_CANDIDATES)
    auc_col = find_column(df, AUC_COL_CANDIDATES)
    acc_col = find_column(df, ACC_COL_CANDIDATES)

    missing = []
    if model_col is None:
        missing.append("模型列")
    if dataset_col is None:
        missing.append("数据集列")
    if auc_col is None:
        missing.append("AUC列")
    if acc_col is None:
        missing.append("ACC列")
    if missing:
        raise ValueError(f"compare.csv 缺少必要列：{', '.join(missing)}")

    work_df = df[[model_col, dataset_col, auc_col, acc_col]].copy()
    work_df.columns = ["model", "dataset", "auc", "acc"]
    work_df["auc"] = pd.to_numeric(work_df["auc"], errors="coerce")
    work_df["acc"] = pd.to_numeric(work_df["acc"], errors="coerce")

    dataset_keys = [key for key in DATASET_NAME_MAP.keys() if key in set(work_df["dataset"].astype(str))]
    extra_datasets = [
        key for key in work_df["dataset"].astype(str).unique().tolist()
        if key not in DATASET_NAME_MAP
    ]
    dataset_keys.extend(sorted(extra_datasets))

    model_keys = [key for key in MODEL_NAME_MAP.keys() if key in set(work_df["model"].astype(str))]
    extra_models = [
        key for key in work_df["model"].astype(str).unique().tolist()
        if key not in MODEL_NAME_MAP
    ]
    model_keys.extend(sorted(extra_models))

    auc_pivot = work_df.pivot_table(index="model", columns="dataset", values="auc", aggfunc="first")
    acc_pivot = work_df.pivot_table(index="model", columns="dataset", values="acc", aggfunc="first")

    # 预计算每列（dataset × metric）中所有模型的值，用于排名
    col_all_auc = {}
    col_all_acc = {}
    for ds in dataset_keys:
        col_all_auc[ds] = [
            auc_pivot.loc[m, ds]
            for m in model_keys
            if m in auc_pivot.index and ds in auc_pivot.columns
        ]
        col_all_acc[ds] = [
            acc_pivot.loc[m, ds]
            for m in model_keys
            if m in acc_pivot.index and ds in acc_pivot.columns
        ]

    col_spec = "l" + "cc" * len(dataset_keys)

    lines = []
    lines.append(r"\begin{table*}[htbp]")
    lines.append(r"    \centering")
    lines.append(f"    \\caption{{{TABLE_CAPTION}}}")
    lines.append(f"    \\label{{{TABLE_LABEL}}}")
    lines.append(r"    \resizebox{\textwidth}{!}{%")
    lines.append(f"    \\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"        \toprule")

    header_top = [r"        \multirow{2}{*}{模型}"]
    for dataset in dataset_keys:
        short_name = render_dataset_label(dataset)
        header_top.append(rf"\multicolumn{{2}}{{c}}{{{short_name}}}")
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

    for model in model_keys:
        model_display = MODEL_NAME_MAP.get(model, model)
        row = [f"        {latex_escape(model_display)}"]
        for dataset in dataset_keys:
            auc_val = auc_pivot.loc[model, dataset] if model in auc_pivot.index and dataset in auc_pivot.columns else pd.NA
            acc_val = acc_pivot.loc[model, dataset] if model in acc_pivot.index and dataset in acc_pivot.columns else pd.NA
            row.append(format_metric_ranked(auc_val, col_all_auc.get(dataset, [])))
            row.append(format_metric_ranked(acc_val, col_all_acc.get(dataset, [])))
        lines.append(" & ".join(row) + r" \\")

    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}%")
    lines.append(r"    }")
    lines.append(r"    \vspace{2pt}")
    lines.append(r"    \begin{flushleft}")
    note_text = r"{\footnotesize 注：粗体表示最优结果，下划线表示次优结果。"
    if DELTA_SUMMARY_MODE == "mean":
        delta_summary = compute_delta_summary(auc_pivot, acc_pivot, dataset_keys)
        if delta_summary is not None:
            auc_diff = delta_summary["auc_diff"]
            auc_pct = delta_summary["auc_pct"]
            acc_diff = delta_summary["acc_diff"]
            acc_pct = delta_summary["acc_pct"]
            auc_sign = "+" if auc_diff is not None and auc_diff >= 0 else ""
            acc_sign = "+" if acc_diff is not None and acc_diff >= 0 else ""
            note_text += (
                rf" 相较于 GIKT，FA-GIKT 在所有数据集上的平均提升为 "
                rf"AUC {auc_sign}{auc_diff:.4f} ({auc_sign}{auc_pct:.2f}\%)，"
                rf"ACC {acc_sign}{acc_diff:.4f} ({acc_sign}{acc_pct:.2f}\%)。"
            )
    note_text += "}"
    lines.append("    " + note_text)
    lines.append(r"    \end{flushleft}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def main():
    if not os.path.exists(INPUT_CSV):
        print(f"❌ 找不到输入文件：{INPUT_CSV}")
        return

    df = None
    encodings = ['utf-8', 'utf-8-sig', 'gb2312', 'gbk', 'latin-1', 'cp1252']
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
        print("请先准备 compare.csv，再运行本脚本。")
        return

    try:
        latex_code = build_latex_table(df)
    except Exception as exc:
        print(f"❌ 生成 LaTeX 失败：{exc}")
        return

    with open(OUTPUT_TEX, "w", encoding="utf-8-sig") as f:
        f.write(latex_code)

    print("✅ LaTeX 表格已生成")
    print(f"输入文件：{INPUT_CSV}")
    print(f"输出文件：{OUTPUT_TEX}")
    print("\n===== 预览 =====\n")
    print(latex_code)


if __name__ == "__main__":
    main()

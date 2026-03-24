import pandas as pd
import os

# ================= 配置区 =================
INPUT_CSV = r"C:\Users\fzq\Desktop\跑的结果\ablation_results_output\run_ablation\ablation_summary.csv"
OUTPUT_CSV = r"H:\er_gikt\back\kt\scripts\02_ablation_extracted.csv"

# 映射关系: {新列名: 原CSV列名}
COLUMN_MAPPING = {
    "消融组": "Ablation Group",
    "数据集名称": "Dataset",
    "Test_AUC": "Mean Test AUC",
    "Test_ACC": "Mean Test ACC",
    "Max_Validation_AUC": "Fold Best Val AUCs",
    "Max_Validation_ACC": "Fold Best Val ACCs"
}
# ==========================================

def extract_ablation_info():
    if not os.path.exists(INPUT_CSV):
        print(f"❌ 错误：找不到输入文件 -> {INPUT_CSV}")
        return

    try:
        # 读取原始CSV
        df = pd.read_csv(INPUT_CSV)
        
        # 检查必要的原始列是否存在
        missing_cols = [col for col in COLUMN_MAPPING.values() if col not in df.columns]
        if missing_cols:
            print(f"❌ 错误：原CSV中缺少以下列：{missing_cols}")
            return

        # 提取并重命名列
        extracted_df = df[list(COLUMN_MAPPING.values())].copy()
        
        # 这里的关键是：如果原列名相同（比如最后两个映射到了同一个原列名），
        # pandas 选择列时会产生重复列。我们需要确保重命名的对应关系。
        # 我们可以通过索引直接重命名，或者重新构建
        import re
        new_df = pd.DataFrame()
        for new_name, old_name in COLUMN_MAPPING.items():
            new_df[new_name] = df[old_name]

        # 对Max_Validation_AUC和Max_Validation_ACC去除F[数字]:前缀
        for col in ["Max_Validation_AUC", "Max_Validation_ACC"]:
            if col in new_df.columns:
                new_df[col] = new_df[col].astype(str).str.replace(r"F\d+:", "", regex=True)

        # 保存为新的CSV
        new_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
        print(f"✅ 提取并重命名成功！")
        print(f"输入路径：{INPUT_CSV}")
        print(f"输出路径：{OUTPUT_CSV}")
        print("\n数据预览：")
        print(new_df.head())

    except Exception as e:
        print(f"❌ 运行过程中发生错误: {e}")

if __name__ == "__main__":
    extract_ablation_info()
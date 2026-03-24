import os
import re
import csv
import glob

# ================= 配置区 =================
LOG_DIR = r"C:\Users\fzq\Desktop\final_output_and_checkpoints\output\kaggle_logs"
DATASET_NAMES = ["assist12", "assist09_builder", "assist09", "ednet_kt1", "assist17", "nips2020_task34"]
MODEL_NAMES = ['simplekt', 'akt', 'deep_irt', 'dkvmn', 'gikt_old', 'dkt']
OUTPUT_CSV = "01_training_results_extracted.csv"
# ==========================================

# 预编译正则表达式以加速匹配
REGEX_TEST = re.compile(r"Test ACC:\s*([\d.]+)\s*\|\s*Test AUC:\s*([\d.]+)")
REGEX_MAX_VAL = re.compile(r"Max validation AUC:\s*([\d.]+)")
REGEX_CHART_DATA = re.compile(r"Training history saved to .*/chart_data/([^ ]+\.csv)")

def extract_log_info():
    results = []
    success_combos = []
    failed_combos = []

    if not os.path.exists(LOG_DIR):
        print(f"警告：设定的日志目录不存在 -> {LOG_DIR}")
        print("请检查路径。但脚本仍将假设您后续会将其配置正确运行。")

    for model in MODEL_NAMES:
        for dataset in DATASET_NAMES:
            search_pattern = os.path.join(LOG_DIR, f"{model}_{dataset}*.log")
            matched_files = glob.glob(search_pattern)

            if not matched_files:
                failed_combos.append((model, dataset))
                continue

            matched_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            latest_log_file = matched_files[0]

            test_auc, test_acc, max_val_auc, chart_data = "N/A", "N/A", "N/A", "N/A"
            internal_error = False

            try:
                with open(latest_log_file, 'r', encoding='utf-8', errors='replace') as f:
                    for line in f:
                        if "CUDA out of memory" in line or "Traceback (most recent call last)" in line:
                            internal_error = True

                        m_test = REGEX_TEST.search(line)
                        if m_test:
                            test_acc, test_auc = m_test.groups()

                        m_max_val = REGEX_MAX_VAL.search(line)
                        if m_max_val:
                            max_val_auc = m_max_val.group(1)

                        m_chart = REGEX_CHART_DATA.search(line)
                        if m_chart:
                            chart_data = m_chart.group(1)
            except Exception as e:
                print(f"读取文件 {latest_log_file} 时发生异常: {e}")
                internal_error = True

            is_incomplete = (test_auc == "N/A" or test_acc == "N/A" or max_val_auc == "N/A")
            if is_incomplete:
                err_msg = " [存在错误/OOM崩溃]" if internal_error else " [日志未正常结束]"
                print(f"⚠️ 警告: 最新日志不完整 {os.path.basename(latest_log_file)}{err_msg}")
                failed_combos.append((model, dataset))
            else:
                success_combos.append((model, dataset))

            results.append({
                "模型名称": model,
                "数据集名称": dataset,
                "Test_AUC": test_auc,
                "Test_ACC": test_acc,
                "Max_Validation_AUC": max_val_auc,
                "chart_data名称": chart_data,
                "日志文件": os.path.basename(latest_log_file)
            })

    csv_columns = ["模型名称", "数据集名称", "Test_AUC", "Test_ACC", "Max_Validation_AUC", "chart_data名称", "日志文件"]

    try:
        with open(OUTPUT_CSV, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n✅ 提取完成！共提取 {len(results)} 条记录。结果已保存至 {OUTPUT_CSV}")
    except Exception as e:
        print(f"❌ 写入CSV时发生错误: {e}")

    if failed_combos:
        print("\n以下组合提取失败：")
        for m, d in failed_combos:
            print(f"  - 模型: {m} | 数据集: {d}")
        print("\n成功提取的组合：")
        for m, d in success_combos:
            print(f"  - 模型: {m} | 数据集: {d}")
    else:
        print("\n所有模型-数据集组合均提取成功！")

if __name__ == "__main__":
    extract_log_info()
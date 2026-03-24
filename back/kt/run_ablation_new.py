import os
import json
import subprocess
import argparse
import pandas as pd
from datetime import datetime

# 你的基础配置名 (对应 config/experiments/exp_sample_default.toml)
BASE_CONFIG_NAME = "default" 

# 用于保存进度状态的文件
CHECKPOINT_FILE = "ablation_checkpoint.json"

# 定义我们要跑的消融实验组
ABLATION_TARGETS = {
    # "assist09-full": [
    #     "train.dataset_name=assist09",
    #     "train.save_model=True"
    # ],
    # "assist12-full": [
    #     "train.dataset_name=assist12",
    #     "train.save_model=True"
    # ],
    # "assist17-full": [
    #     "train.dataset_name=assist17",
    #     "train.save_model=True"
    # ],
    # "ednet_kt1-full": [
    #     "train.dataset_name=ednet_kt1",
    #     "train.save_model=True"
    # ],
    # "nips2020-full": [
    #     "train.dataset_name=nips2020_task34",
    #     "train.save_model=True"
    # ],
    # 实验名 : 要覆写（Override）的参数列表
    "A_Baseline": [
    ],
    "B_Remove_PID": [
        "model.use_pid=False"
    ],
    
    "C_Remove_Cognitive": [
        "model.use_cognitive_model=False"
    ],
    
    "D_Remove_IRT": [
        "model.use_4pl_irt=False"
    ],
    
    "E_agg_method-kk_gat": [
        "model.agg_method=kk_gat"
    ],
    
    "F_old_gikt": [
        "model.use_pid=False",
        "model.use_cognitive_model=False",
        "model.use_4pl_irt=False"
    ],
    "G_remove_PID_Cognitive": [
        "model.use_pid=False",
        "model.use_cognitive_model=False"
    ],
    "H_remove_PID_IRT": [
        "model.use_pid=False",
        "model.use_4pl_irt=False"
    ],
    "I_remove_Cognitive_IRT": [
        "model.use_cognitive_model=False",
        "model.use_4pl_irt=False"
    ]
}

CSV_HEADERS = [
    'Ablation Group', 'Date', 'Dataset', 'K Fold',
    'Fold Best Epochs', 'Fold Best Val AUCs', 'Fold Best Val ACCs', 'Fold Best Val LOSSes', 'Fold Best Thresholds',
    'Fold Test AUCs', 'Fold Test ACCs', 'Fold Test LOSSes',
    'Mean Test AUC', 'Std Test AUC', 'Mean Test ACC', 'Std Test ACC', 'Mean Test LOSS', 'Std Test LOSS',
    'Max AUC (Fold)', 'Min AUC (Fold)',
    'Avg Epoch Train Time', 'Avg Epoch Val Time', 'Avg Epoch Total Time', 'Avg Epoch Test Time',
    'Total Train Time', 'Total Val Time', 'Total Test Time', 'Total Wall Time',
    'Time', 'experiments_name'
]

def ensure_csv_initialized():
    """防御性代码：确保 CSV 文件和表头存在，避免后续读取报错"""
    import csv
    csv_path = os.path.join("output", "run_ablation", "ablation_summary.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(CSV_HEADERS)
            print(f"🆕 已防御性初始化消融结果表: {csv_path}")
        except Exception as e:
            print(f"⚠️ 初始化 CSV 失败 (非致命): {e}")

def main():
    ensure_csv_initialized()
    parser = argparse.ArgumentParser(description="GIKT 消融实验运行脚本")
    parser.add_argument("--full", action="store_true", help="使用全量数据集 (覆盖 --full 标志)")
    parser.add_argument("--save_global_best", action="store_true", help="透传给 train_test.py：额外保存全局最优 checkpoint")
    parser.add_argument("--nolog", action="store_true", help="禁用详细日志 (train.verbose=false)")
    parser.add_argument("--dataset_name", type=str, help="设置数据集名称 (train.dataset_name)")
    parser.add_argument("--epochs", type=int, help="强制指定实验轮数 (覆盖配置)")
    parser.add_argument("--patience", type=int, help="强制指定早停轮数 (覆盖配置)")
    parser.add_argument("--resume", action="store_true", help="从断点恢复实验")
    parser.add_argument("--reset", action="store_true", help="清理进度并重新开始")
    args = parser.parse_args()

    completed_experiments = []
    if args.resume:
        if os.path.exists(CHECKPOINT_FILE):
            try:
                with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                    completed_experiments = json.load(f)
                print(f"🔄 检测到中断点，已发现下列完成的实验: {completed_experiments}")
            except Exception as e:
                print(f"⚠️ 读取断点文件异常: {e}")
    elif args.reset:
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            print("🗑️ 已清理历史断点，将重新运行所有指定的实验。")

    print(f"🚀 开始执行 GIKT 模块消融实验（共 {len(ABLATION_TARGETS)} 组）")
    
    # 尝试加载当前数据集的最优参数
    best_params_overrides = []
    if args.dataset_name:
        # 先以 "-" 分割去掉抽样后缀，再把 "_window" 替换为空
        base_dataset_name = args.dataset_name.split('-')[0].replace("_window", "")
        
        best_params_path = os.path.join("config", "best_params", "gikt_best_params.json")
        if os.path.exists(best_params_path):
            try:
                with open(best_params_path, "r", encoding="utf-8") as f:
                    params_dict = json.load(f)
                
                # 模糊匹配逻辑（支持 assist09_builder 匹配到 assist09 的参数）
                matched_key = None
                for k in params_dict.keys():
                    if k == "default": 
                        continue
                    # 统一转小写并兼容年份简写
                    k_norm = k.lower().replace("2009", "09").replace("2012", "12").replace("2015", "15").replace("2017", "17")
                    if k_norm == base_dataset_name or k_norm in base_dataset_name:
                        matched_key = k
                        break
                
                # 获取匹配到的参数，如果没有找到则回退到 default
                target_params = params_dict.get(matched_key, params_dict.get("default", {}))
                display_key = matched_key if matched_key else "default"
                
                if target_params:
                    print(f"✅ 成功从 {best_params_path} 加载 [{display_key}] 的最优参数！(基于输入 {args.dataset_name})")
                    for k, v in target_params.items():
                        # 注意：字典中学习率可能是 float，其它是模型参数
                        if k == "learning_rate":
                            best_params_overrides.append(f"train.learning_rate={v}")
                        else:
                            best_params_overrides.append(f"model.{k}={v}")
            except Exception as e:
                print(f"⚠️ 读取 {best_params_path} 解析失败，将使用默认参数: {e}")

    for exp_name, override_args in ABLATION_TARGETS.items():
        if exp_name in completed_experiments:
            print("\n" + "="*80)
            print(f"⏩ 跳过已完成的消融组别: 【{exp_name}】")
            print("="*80)
            continue

        print("\n" + "="*80)
        print(f"🎯 正在运行消融组别: 【{exp_name}】")
        print("="*80)
        
        # 组装基础命令
        cmd = ["python", "train_test.py", "--name", BASE_CONFIG_NAME]
        if args.full:
            cmd.append("--full")
        if args.save_global_best:
            cmd.append("--save_global_best")
            
        # 加入消融专用标识，让 train_test.py 帮我们写 CSV
        cmd.extend(["--ablation_name", exp_name])
        
        # 处理覆写参数，先加入消融模块特定参数，然后加入从 JSON 提取的最优超参
        final_overrides = list(override_args)
        final_overrides.extend(best_params_overrides)
        
        if args.nolog:
            final_overrides.append("train.verbose=False")
        if args.dataset_name:
            final_overrides.append(f"train.dataset_name={args.dataset_name}")
        
        # 强制配置诊断性训练轮数和早停
        if args.epochs:
            final_overrides.append(f"train.epochs={args.epochs}")
        if args.patience:
            final_overrides.append(f"train.patience={args.patience}")

        # 加入覆写参数指令
        if final_overrides:
            cmd.append("--override")
            cmd.extend(final_overrides)
            
        print(f"💻 执行命令: {' '.join(cmd)}\n")
        
        # 执行子进程，阻断当前代码直到跑完
        try:
            subprocess.run(cmd, check=True)
            completed_experiments.append(exp_name)
            with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
                json.dump(completed_experiments, f, ensure_ascii=False, indent=4)
        except subprocess.CalledProcessError as e:
            print(f"❌ 实验【{exp_name}】遭遇崩溃退出了，错误码: {e.returncode}")
            print("❗ 脚本中断。使用 --resume 重新运行以从当前位置恢复...")
            break
            
    # 收尾汇总
    print("\n" + "="*80)
    print("✅ 消融实验流程结束！汇总结果如下：")
    
    # 全部若跑完，清空断点文件
    if len(completed_experiments) == len(ABLATION_TARGETS):
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            print("🎉 所有组别运行完成，已自动清除断点记录！")
    
    csv_path = os.path.join("output", "run_ablation", "ablation_summary.csv")

    if not os.path.exists(csv_path):
        print("🤔 仍未找到 ablation_summary.csv (防御性检查失败)，请检查 training 过程是否有致命错误。")
    else:
        try:
            if os.path.getsize(csv_path) > 10: # 大于10字节(考虑到BOM头)才尝试读取
                df = pd.read_csv(csv_path)
                print(f"\n📊 实验结果汇总表 [累计 {len(df)} 条记录]: {csv_path}")
                
                # 防御性展示：防止 DataFrame 为空报错，或行数不足
                if not df.empty:
                    display_count = min(len(df), len(ABLATION_TARGETS))
                    print(df.tail(display_count).to_markdown(index=False))
                else:
                    print("⚠️ 表格仅包含表头，尚未写入任何数据行。")
            else:
                print("⚠️ CSV 文件存在但似乎为空。")
        except Exception as e:
            print(f"⚠️ 读取 CSV 展示结果时发生异常 (非致命): {str(e)}")
            print(f"💡 文件路径: {csv_path}")

# python run_ablation_new.py
# python run_ablation_new.py --full
# python run_ablation_new.py --full --nolog --dataset_name assist09
if __name__ == "__main__":
    main()

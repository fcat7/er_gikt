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
    # 实验名 : 要覆写（Override）的参数列表
    # "A_Baseline": [],
    
    # "B_Remove_PID": [
    #     "model.use_pid=False"
    # ],
    
    # "C_Remove_Cognitive": [
    #     "model.use_cognitive_model=False"
    # ],
    
    # "D_Remove_IRT_4PL": [
    #     "model.use_4pl_irt=False"
    # ],
    
    # "E_agg_method: gcn": [
    #     "model.agg_method=gcn"
    # ],
    
    "F_old_gikt": [
        "model.use_pid=False",
        "model.use_cognitive_model=False",
        "model.use_4pl_irt=False"
    ]
}

def main():
    parser = argparse.ArgumentParser(description="GIKT 消融实验运行脚本")
    parser.add_argument("--full", action="store_true", help="使用全量数据集 (覆盖 --full 标志)")
    parser.add_argument("--log", action="store_true", help="启用详细日志 (train.verbose=true)")
    parser.add_argument("--dataset_name", type=str, help="设置数据集名称 (train.dataset_name)")
    parser.add_argument("--epochs", type=int, default=30, help="强制指定实验轮数 (覆盖配置，默认30)")
    parser.add_argument("--patience", type=int, default=5, help="强制指定早停轮数 (覆盖配置，默认5)")
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
            
        # 加入消融专用标识，让 train_test.py 帮我们写 CSV
        cmd.extend(["--ablation_name", exp_name])
        
        # 处理覆写参数
        final_overrides = list(override_args)
        if args.log:
            final_overrides.append("train.verbose=True")
        if args.dataset_name:
            final_overrides.append(f"train.dataset_name={args.dataset_name}")
        
        # 强制配置诊断性训练轮数和早停
        final_overrides.append(f"train.epochs={args.epochs}")
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
    
    # 因为跑的是默认配置，数据集可能是 assist09，我们需要去读取 CSV
    # 在这个项目中 log_dir_name 根据数据集来定，但前面 train_test.py 里把 csv 和普通 log 存一起了。
    # 我们可以用简单的正则或者固定的基础目录来扫（由于不知道具体存哪，我们做一个模糊查找）
    
    # 我们知道 train_test.py 会把文件写到输出目录。如果你跑多次不同的数据集，可能会写到不同的位置。
    # 为了保险，我们搜索一下 back/kt 目录下最近被修改的 ablation_summary.csv
    import glob
    search_path = "output/**/ablation_summary.csv"
    csv_files = glob.glob(search_path, recursive=True)
    
    if not csv_files:
        print("🤔 没有找到 ablation_summary.csv，请检查上方日志是否有报错。")
    else:
        # 取最新修改的 CSV
        latest_csv = max(csv_files, key=os.path.getmtime)
        try:
            df = pd.read_csv(latest_csv)
            # 过滤出含有今天日期的记录，或者是最近运行跑的这几组
            print(f"\n📊 读取自: {latest_csv}")
            print(df.tail(len(ABLATION_TARGETS)).to_markdown(index=False))
        except Exception as e:
            print(f"读取 CSV 异常：{str(e)}")

# python run_ablation_new.py
# python run_ablation_new.py --full
# python run_ablation_new.py --log --dataset_name assist09-sample_20%
if __name__ == "__main__":
    main()

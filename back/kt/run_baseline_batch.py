import os
import subprocess
import time
from datetime import datetime

# ================= 🔧 配置区域 =================
# 配置你想跑的模型，支持：['dkt', 'dkvmn', 'akt', 'simplekt', 'qikt', 'lbkt', 'gikt_old']
MODELS = ['dkt', 'dkvmn', 'akt', 'simplekt', 'qikt', 'lbkt', 'gikt_old'] 

# 配置你想跑的数据集名称
DATASETS = [
    # 'assist09',
    'assist12'
    ]

# 通用训练超参数配置
EPOCHS = 200
K_FOLD = 1
PATIENCE = 10
# ===============================================

def run_batch():
    # 获取当前所在目录，并创建专门存放夜间挂机日志的文件夹
    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(current_dir, 'output', 'batch_logs')
    os.makedirs(log_dir, exist_ok=True)
    
    total_tasks = len(MODELS) * len(DATASETS)
    current_task = 0
    
    start_time = time.time()
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🌙 开始执行夜间串行批处理任务，共需要执行 {total_tasks} 个组合。")
    print(f"模型的训练细节日志将被归档至: {log_dir}\n")
    
    for dataset in DATASETS:
        for model in MODELS:
            current_task += 1
            task_name = f"{model}_{dataset}"
            task_start_fmt = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{task_start_fmt}] 🚀 进度: {current_task}/{total_tasks} | 正在训练: {task_name}")
            
            # 构建命令
            cmd = [
                "python", "train_baseline.py",
                "--model_name", model,
                "--dataset", dataset,
                "--epochs", str(EPOCHS),
                "--k_fold", str(K_FOLD),
                "--patience", str(PATIENCE)
            ]
            
            # 定义日志文件路径: model_dataset_20231015_083000.log
            log_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{task_name}.log"
            log_file = os.path.join(log_dir, log_filename)
            
            try:
                # 记录到日志文件中
                with open(log_file, "w", encoding="utf-8") as f:
                    f.write(f"=== 运行命令: {' '.join(cmd)} ===\n")
                    f.write(f"=== 开始时间: {task_start_fmt} ===\n\n")
                    f.flush()
                    
                    # 串行阻塞执行该任务，并将 stdout 和 stderr 合并输出至文件
                    # 注入 PYTHONIOENCODING=utf-8，防止 Windows 捕获含有 emoji 的输出时发生 GBK 编码报错
                    my_env = os.environ.copy()
                    my_env["PYTHONIOENCODING"] = "utf-8"
                    process = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True, encoding="utf-8", env=my_env)
                    
                    f.write(f"\n=== 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                    f.write(f"=== 退出码: {process.returncode} ===\n")
                    
                # 检查执行结果
                if process.returncode == 0:
                    print(f"  ✅ [成功] 任务 {task_name} 执行完毕！详细日志参见: {log_filename}")
                else:
                    print(f"  ❌ [失败] 任务 {task_name} 异常终结 (退出码:{process.returncode})！请明天检查日志: {log_filename}")
                    
            except Exception as e:
                print(f"  ⚠️ [系统异常] 启动任务 {task_name} 时发生系统级错误: {str(e)}")
                
    total_time = time.time() - start_time
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🎉 所有夜间批处理任务已执行完毕！总耗时: {total_time/3600:.2f} 小时。")

if __name__ == "__main__":
    run_batch()

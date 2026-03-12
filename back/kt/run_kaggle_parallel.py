import os
import subprocess
import time
import argparse
from datetime import datetime

def run_parallel_batch():
    parser = argparse.ArgumentParser(description="Kaggle Parallel KT Training")
    parser.add_argument("--models", nargs='+', default=['dkt', 'dkvmn', 'deep_irt', 'akt', 'gikt_old'], help="List of models to run")
    parser.add_argument("--datasets", nargs='+', default=['assist09'], help="List of datasets")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--k_fold", type=int, default=1, help="K-fold cross validation")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--max_jobs", type=int, default=3, help="Max concurrent jobs")
    
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(current_dir, 'output', 'kaggle_logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 构建所有任务
    tasks = []
    for dataset in args.datasets:
        for model in args.models:
            tasks.append({
                'model': model,
                'dataset': dataset,
                'cmd': [
                    "python", "train_baseline.py",
                    "--model_name", model,
                    "--dataset", dataset,
                    "--epochs", str(args.epochs),
                    "--k_fold", str(args.k_fold),
                    "--patience", str(args.patience)
                ]
            })

    total_tasks = len(tasks)
    active_processes = []
    completed_tasks = 0

    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 开始 KAGGLE 并行训练，并发数: {args.max_jobs}，共 {total_tasks} 个任务")
    print(f"配置: Models={args.models}, Datasets={args.datasets}, Epochs={args.epochs}")
    
    task_idx = 0
    while task_idx < total_tasks or len(active_processes) > 0:
        # 清理已完成的进程
        for p_info in active_processes[:]:
            p = p_info['process']
            if p.poll() is not None:  # 进程已结束
                completed_tasks += 1
                status = "✅ 成功" if p.returncode == 0 else f"❌ 失败(Exit {p.returncode})"
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {status}: {p_info['name']}")
                active_processes.remove(p_info)
        
        # 只要进程池没满，且还有任务，就继续塞任务
        while len(active_processes) < args.max_jobs and task_idx < total_tasks:
            t = tasks[task_idx]
            task_name = f"{t['model']}_{t['dataset']}"
            log_filename = os.path.join(log_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{task_name}.log")
            
            f = open(log_filename, "w", encoding="utf-8")
            f.write(f"=== CMD: {' '.join(t['cmd'])} ===\n\n")
            f.flush()
            
            # 启动异步子进程
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 🏃 启动任务: {task_name} -> 日志: {os.path.basename(log_filename)}")
            my_env = os.environ.copy()
            my_env["PYTHONIOENCODING"] = "utf-8"
            
            p = subprocess.Popen(t['cmd'], stdout=f, stderr=subprocess.STDOUT, text=True, encoding="utf-8", env=my_env)
            active_processes.append({'process': p, 'name': task_name, 'file': f})
            task_idx += 1
            
        time.sleep(2) # 避免死循环狂占CPU

    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🎉 所有任务已完毕！赶快去写论文！")

if __name__ == "__main__":
    run_parallel_batch()
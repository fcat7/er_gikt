import os
import subprocess
import time
from datetime import datetime
import traceback

# ================= 🔧 配置区域 =================
# 全局开关：是否为快速调试模式？
# True：所有模型强制使用较高的学习率（加快收敛步伐，用于快速摸底验证）
# False：严格遵守 config/best_params 中每个基线论文宣称的最佳超参（用于写论文的夜间最终跑批）
IS_DEBUG_MODE = False

# 配置你想跑的模型及其专属的 Batch Size（以此适配 6GB 显存，防止 OOM）
# 建议的快速试错学习率同样配置在下方字典中。若 IS_DEBUG_MODE=False，'debug_lr' 将被忽略。
MODELS = {
    'dkt': {'batch_size': 256, 'debug_lr': 0.01},           # RNN经典，极省内存。调试时可激进（time: 6.48s | VRAM: 2.52G (Res: 3.78G)；
    'dkvmn': {'batch_size': 128, 'debug_lr': 0.01},         # 记忆网络矩阵较小（128，time: 31.04s | VRAM: 4.15G (Res: 6.05G；64，time: 27.93s | VRAM: 2.11G (Res: 3.23G))
    'deep_irt': {'batch_size': 64, 'debug_lr': 0.005},     # 增加IR特性，适中（128，time: 44.85s |VRAM: 6.32G (Res: 8.99G)；64，time: 15.84s | VRAM: 3.18G (Res: 4.53G)）
    'gikt_old': {'batch_size': 128, 'debug_lr': 0.005},      # GNN节点聚合，需要一定显存（128，time: 49.19s | VRAM: 2.64G (Res: 6.64G)；64，time: 84.24s | VRAM: 1.36G (Res: 5.26G)）
    'simplekt': {'batch_size': 256, 'debug_lr': 0.001},      # 注意力模型(Transformer架构)，内存消耗大，防炸（256，time: 8.40s | VRAM: 3.27G (Res: 3.62G)；128，time: 8.54s | VRAM: 1.64G (Res: 1.67G)）
    'akt': {'batch_size': 64, 'debug_lr': 0.001},           # 同为重型注意力机制，占用大（64，time: 38.41s | VRAM: 4.06G (Res: 4.14G)）
}

# 配置你想跑的数据集名称
DATASETS = [
    # 'assist09',
    # 'assist09_builder',
    'assist12',
    # 'ednet_kt1',
    # 'nips2020_task34'
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
        for model_name, m_config in MODELS.items():
            current_task += 1
            batch_size = m_config['batch_size']
            task_name = f"{model_name}_{dataset}_bs{batch_size}"
            task_start_fmt = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{task_start_fmt}] 🚀 进度: {current_task}/{total_tasks} | 正在训练: {task_name}")
            
            # 构建命令
            cmd = [
                "python", "train_baseline.py",
                "--model_name", model_name,
                "--dataset", dataset,
                "--epochs", str(EPOCHS),
                "--batch_size", str(batch_size),
                "--k_fold", str(K_FOLD),
                "--patience", str(PATIENCE)
            ]
            
            # 如果是快速调试模式，则强制追加学习率来覆盖最优参数
            if IS_DEBUG_MODE and 'debug_lr' in m_config:
                cmd.extend(["--learning_rate", str(m_config['debug_lr'])])
            
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
                err_tb = traceback.format_exc()
                print(f"  ⚠️ [系统异常] 启动任务 {task_name} 时发生系统级错误: {str(e)}")
                print(err_tb)
                # 尽力把异常也落到对应的 batch log 里
                try:
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write("\n=== Batch Runner Exception ===\n")
                        f.write(err_tb)
                        f.write("\n")
                except Exception:
                    pass
                
    total_time = time.time() - start_time
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🎉 所有夜间批处理任务已执行完毕！总耗时: {total_time/3600:.2f} 小时。")

if __name__ == "__main__":
    run_batch()

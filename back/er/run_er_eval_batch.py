import os
import sys
import time
import argparse
import subprocess
import pandas as pd

# 
# python run_er_eval_batch.py --full --baseline all --workers 4 --total_samples 51584
# 单模型
# python run_er_eval_batch.py --full --baseline ours --workers 8 --total_samples 51584
def main():
    parser = argparse.ArgumentParser(description="Multi-Process Parallel Evaluator for ER-GIKT")
    parser.add_argument('--workers', type=int, default=4, help="启动的子进程/终端数量（找你电脑的上限）")
    parser.add_argument('--total_samples', type=int, default=51584, help="测试集的总有效长序列数量")
    parser.add_argument('--full', action='store_true', help="使用 full 版本的配置文件")
    parser.add_argument('--baseline', type=str, default='ours', help="需要测试的模式 (e.g., ours, all, greedy)")
    args = parser.parse_args()

    # 计算均分的 Chunk_size
    chunk_size = (args.total_samples + args.workers - 1) // args.workers
    processes = []

    print(f"\n🚀 [并行评估启动] 规划了 {args.workers} 个并发 Workers ...")
    print(f"📊 总样本: {args.total_samples} | 每块大致分配: {chunk_size} 样本")
    
    log_dir = "output/logs/batch_workers"
    os.makedirs(log_dir, exist_ok=True)

    start_time = time.time()

    # 第一部分：派发进子进程独立计算
    for i in range(args.workers):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, args.total_samples)

        if start_idx >= args.total_samples:
            break

        cmd = [
            sys.executable, "run_er_eval.py",
            "--start_idx", str(start_idx),
            "--end_idx", str(end_idx),
            "--baseline", args.baseline
        ]
        if args.full:
            cmd.append("--full")

        log_file_path = os.path.join(log_dir, f"worker_{i}_{start_idx}_to_{end_idx}.log")
        log_f = open(log_file_path, "w", encoding='utf-8')
        
        print(f"  [子进程 {i}] 处理区间: {start_idx:5d} -> {end_idx:5d} | 日志: {log_file_path}")
        p = subprocess.Popen(cmd, stdout=log_f, stderr=log_f)
        processes.append((p, start_idx, end_idx, log_file_path, log_f))

    print("\n⏳ 所有命令已离线派发。请等待它们全部运行完毕，或打开对应 log 文件以监视实时进度...")
    
    failed = False
    for p, s_idx, e_idx, log_path, log_f in processes:
        p.wait()
        log_f.close()
        if p.returncode != 0:
            print(f"❌ [进程报错] 区间 [{s_idx}:{e_idx}] 发生异常 (Return Code: {p.returncode}), 请检查日志: {log_path}")
            failed = True
        else:
            print(f"✅ 区间 [{s_idx}:{e_idx}] 运行完毕.")

    if failed:
        print("\n⚠️ 警报: 有一个或多个评测段中途崩溃，程序将停止，不执行全量 CSV 合并操作。")
        sys.exit(1)

    print(f"\n⏱️ 评测总耗时: {(time.time() - start_time) / 60:.2f} 分钟.")
    print("🔗 开始执行 CSV 碎片数据拼接归档...")

    # 第二部分：将分散落地的CSV寻找出来并拼合
    out_dir = "output/recommendation_full" if args.full else "output/recommendation_sample"
    combined_df = pd.DataFrame()
    
    all_found = True
    suffix = f"_{args.baseline}" if args.baseline else ""
    
    for p, s_idx, e_idx, log_path, log_f in processes:
        chunk_tag = f"_parts_{s_idx}_to_{e_idx}"
        file_name = f"recommend_eval_{'full' if args.full else 'sample'}{suffix}{chunk_tag}.csv"
        expected_file = os.path.join(out_dir, file_name)

        if os.path.exists(expected_file):
            df_chunk = pd.read_csv(expected_file)
            combined_df = pd.concat([combined_df, df_chunk], ignore_index=True)
            print(f"  + 追加读取了分段区块: {file_name} ({len(df_chunk)} 行)")
            
            # (可选) 合并完后删除碎片，取消注释以下这行可开启
            # os.remove(expected_file) 
        else:
            print(f"  ❌ 未找到预期输出的文件碎片: {expected_file}")
            all_found = False

    if all_found and not combined_df.empty:
        final_file = os.path.join(out_dir, f"recommend_eval_{'full' if args.full else 'sample'}_{args.baseline}_MERGED.csv")
        combined_df.to_csv(final_file, index=False)
        print(f"\n🎉 碎片合并大功告成！全量评测汇总数据已写入: \n 👉 {final_file} (合计总行数: {len(combined_df)})\n")
    else:
        print("\n⚠️ 警告: 因为未找齐所有文件碎片，最终合并存在残缺。")

if __name__ == '__main__':
    main()

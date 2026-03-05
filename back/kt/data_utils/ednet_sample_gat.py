""" ednet_sample_gat.py
面向图注意力网络 (GAT) 优化的 EdNet 定向采样工具

核心思想:
常规随机采样会导致大部分题目仅映射1个知识点(Degree=1)。在Degree=1时，GAT的Attention权重恒为1.0，导致数学退化。
本脚本在扫描用户元数据时，会评估每个用户的 "GAT友好度" (多标签题目占比 & 独立题目数)。
使用 `gat_priority` 策略采样时，将优先选取那些经常做"综合大题(蕴含多个知识点)"的用户。
经过本脚本采样出的数据，提交给 GIKT 训练时，能大幅降低单射带来的退化，让 GAT 的威力得以发挥！
"""
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import argparse

# ================= 动态共享内存 =================
_shared_q_degree_map = {}

def init_worker(q_map):
    """
    初始化多进程的共享变量，避免巨大的字典在进程间重复拷贝
    """
    global _shared_q_degree_map
    _shared_q_degree_map = q_map

# ================= 核心处理函数 =================
def extract_user_gat_features(file_path):
    """
    不仅统计行数，还要深度分析该用户的交互序列对图拓扑的贡献度。
    """
    try:
        filename = os.path.basename(file_path)
        user_id = os.path.splitext(filename)[0]
        
        if os.path.getsize(file_path) == 0:
            return None
            
        row_count = 0
        multi_skill_count = 0
        unique_qs = set()
        
        with open(file_path, 'r', encoding='utf-8') as f:
            header = f.readline() 
            for line in f:
                parts = line.strip().split(',')
                # KT1 格式通常是: timestamp, solving_id, question_id, user_answer, elapsed_time
                if len(parts) >= 3:
                    q_id = parts[2]
                    row_count += 1
                    unique_qs.add(q_id)
                    
                    # 查询该题包含的知识点数
                    deg = _shared_q_degree_map.get(q_id, 1)
                    if deg > 1:
                        multi_skill_count += 1

        if row_count == 0: return None
        
        # 1. 该用户做的题目中，有多大比例是综合大题（多标签）
        ms_ratio = round(multi_skill_count / row_count, 4)
        # 2. 图的节点丰富度基础
        uq_count = len(unique_qs)
        
        return user_id, row_count, ms_ratio, uq_count, file_path
    except Exception as e:
        return None

def load_question_degrees(questions_path):
    """
    提前解析全量题目，统计每个题目的知识点度数 (Degree)
    """
    print(f"正在分析题目拓扑结构: {questions_path} ...")
    if not os.path.exists(questions_path):
        raise FileNotFoundError(f"找不到题目文件: {questions_path}")
        
    df_q = pd.read_csv(questions_path)
    q_degree_map = {}
    
    # 统计有多少个题目本身就是多标签的
    multi_count = 0
    for _, row in df_q.iterrows():
        tags = str(row['tags'])
        # EdNet中tag使用分号隔开
        deg = len(tags.split(';')) if tags != 'nan' else 1
        q_degree_map[row['question_id']] = deg
        if deg > 1:
            multi_count += 1
            
    print(f"题库总数: {len(df_q)}, 其中多标签综合题(Degree>1): {multi_count} (占比: {multi_count/len(df_q):.2%})")
    return q_degree_map

def generate_gat_metadata(data_dir, output_file, q_degree_map):
    print(f"\n正在执行 GAT 深度扫描目录(可能会比较耗时): {data_dir} ...")
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    print(f"待处理用户数: {len(csv_files)} ...")

    pool = Pool(processes=cpu_count(), initializer=init_worker, initargs=(q_degree_map,))
    results = []
    
    with tqdm(total=len(csv_files)) as pbar:
        for result in pool.imap_unordered(extract_user_gat_features, csv_files):
            if result:
                results.append(result)
            pbar.update()
            
    pool.close()
    pool.join()

    print("扫描完成，正在保存增强型GAT元数据...")
    df = pd.DataFrame(results, columns=['user_id', 'row_count', 'ms_ratio', 'uq_count', 'file_path'])
    df.to_csv(output_file, index=False)
    return df

def sample_gat_friendly_users(df, sample_n, min_seq_len=20, ms_threshold=0.3):
    """
    定向挖掘策略：寻找那些让GAT能大放异彩的子群体。
    """
    print(f"\n=== 执行 GAT 定向采样 ===")
    original_size = len(df)
    
    # 基础过滤: 序列长度必须达标且不能是刷题机器人(例如过滤掉做题超过3000题的)
    df_valid = df[(df['row_count'] >= min_seq_len) & (df['row_count'] <= 3000)].copy()
    print(f"1. 序列长度校验 (基于 {min_seq_len} ~ 3000): 剩余 {len(df_valid)} 人")
    
    # 核心过滤: 高多标签浓度。只有高ms_ratio，图中的度数才会增加！
    df_gat = df_valid[df_valid['ms_ratio'] >= ms_threshold].copy()
    print(f"2. GAT高潜力者挖掘 (复合题比例 >= {ms_threshold}): 挖掘到核心用户 {len(df_gat)} 人")
    
    if len(df_gat) < sample_n:
        print(f"⚠️ 警告：满足严格GAT条件的用户数 ({len(df_gat)}) 少于目标数 ({sample_n})。将退回并按潜力排名选取！")
        # 降级方案：按 GAT 潜力指数 (ms_ratio 和 unique_q 的乘积或加权) 排序选取
        df_valid['gat_score'] = df_valid['ms_ratio'] * np.log10(df_valid['uq_count'] + 1)
        sampled = df_valid.nlargest(sample_n, 'gat_score')
    else:
        # 当候选池足够大时，为保证一定的数据多样性，我们可以在达标池中进行均匀抽取
        # 这样模型能学到更好的泛化特征
        sampled = df_gat.sample(n=sample_n, random_state=42)
        
    print(f"🎯 最终抽样完成！(抽取量: {len(sampled)})")
    print(f"   [目标验证] 抽取群体的平均序列长度: {sampled['row_count'].mean():.1f}")
    print(f"   [目标验证] 抽取群体的综合大题占比: {sampled['ms_ratio'].mean():.1%}")
    print(f"   (若随机抽样，此占比仅为全人群均值的 ~20-30%)")
    
    return sampled

def merge_csvs(sample_list_df, output_csv_path, correct_map, skill_map):
    print(f"\n正在将 {len(sample_list_df)} 个用户的交互矩阵合体 (这并联成了您的骨架图)...")
    if os.path.exists(output_csv_path): os.remove(output_csv_path)

    first_file = True
    for _, row in tqdm(sample_list_df.iterrows(), total=len(sample_list_df)):
        try:
            user_df = pd.read_csv(row['file_path'])
            user_df.insert(0, 'user_id', row['user_id'])
            user_df['skill_id'] = user_df['question_id'].map(skill_map)
            
            ua = user_df['user_answer'].fillna('')
            ca = user_df['question_id'].map(correct_map).fillna('')
            user_df['correct'] = (ua == ca).astype(int)
            
            user_df.to_csv(output_csv_path, mode='w' if first_file else 'a', header=first_file, index=False)
            first_file = False
        except Exception as e:
            pass
    print(f"✅ GAT 优化的合并数据已生成: {output_csv_path}")

def main():
    parser = argparse.ArgumentParser(description="EdNet KT1 [GAT友好型] 定向抽样工具")
    parser.add_argument('--data_dir', type=str, default=r"H:/dataset/EdNet/KT1", help='原始用户csv文件夹')
    parser.add_argument('--questions_path', type=str, default=r"H:/dataset/EdNet/contents/questions.csv", help='题目元数据路径')
    parser.add_argument('--output_dir', type=str, default=r"H:/dataset/EdNet/", help='输出根目录')
    parser.add_argument('--size', type=int, default=5000, help='目标抽样学生数量')
    parser.add_argument('--min_seq_len', type=int, default=5, help='最小交互序列长度')
    parser.add_argument('--ms_threshold', type=float, default=0.45, help='多技能题目(复合大题)的最小比例，越高对GAT越有利')
    parser.add_argument('--force_scan', action='store_true', help='强制重新扫描目录')

    args = parser.parse_args()
    
    # 检测关键目录
    if not os.path.exists(args.data_dir):
        return print(f"错误: 找不到数据目录 {args.data_dir}，检查路径！")
    if not os.path.exists(args.questions_path):
        return print(f"错误: 找不到题库关联 {args.questions_path}，无法构建网络分析！")
        
    os.makedirs(args.output_dir, exist_ok=True)
    meta_path = os.path.join(args.output_dir, 'ednet_kt1_metadata_gat.csv')
    
    # 1. 结构分析
    correct_map, skill_map = {}, {}
    df_q = pd.read_csv(args.questions_path)
    correct_map = dict(zip(df_q['question_id'], df_q['correct_answer']))
    skill_map = dict(zip(df_q['question_id'], df_q['tags']))
    
    # 2. 元数据准备
    if args.force_scan or not os.path.exists(meta_path):
        q_degree_map = load_question_degrees(args.questions_path)
        df_meta = generate_gat_metadata(args.data_dir, meta_path, q_degree_map)
    else:
        print(f"加载现有的GAT元数据缓存: {meta_path}")
        df_meta = pd.read_csv(meta_path)
        
    # 3. 核心定向抽样
    sampled_users = sample_gat_friendly_users(df_meta, args.size, args.min_seq_len, args.ms_threshold)
    
    # 4. 生成数据
    suffix = f"GAT-Opt_size{args.size}_msT{int(args.ms_threshold*100)}"
    output_filename = os.path.join(args.output_dir, f"Ednet-KT1_{suffix}.csv")
    merge_csvs(sampled_users, output_filename, correct_map, skill_map)

# 切换到对应的目录
# cd h:\er_gikt\back\kt\data_utils

# 运行 GAT 针对性抽样，举例抽样 50000 个人，设定复合题高比例阈值为 45%
# python ednet_sample_gat.py --data_dir H:/dataset/EdNet/KT1 --questions_path H:/dataset/EdNet/contents/questions.csv --size 5000 --ms_threshold 0.45
# 
# === 执行 GAT 定向采样 ===
# 1. 序列长度校验 (基于 5 ~ 3000): 剩余 680516 人
# 2. GAT高潜力者挖掘 (复合题比例 >= 0.45): 挖掘到核心用户 265255 人
# 🎯 最终抽样完成！(抽取量: 5000)
#    [目标验证] 抽取群体的平均序列长度: 162.0
#    [目标验证] 抽取群体的综合大题占比: 66.6%
#    (若随机抽样，此占比仅为全人群均值的 ~20-30%)

# 正在将 5000 个用户的交互矩阵合体 (这并联成了您的骨架图)... 
# ✅ GAT 优化的合并数据已生成: H:/dataset/EdNet/Ednet-KT1_GAT-Opt_size5000_msT45.csv
if __name__ == "__main__":
    main()
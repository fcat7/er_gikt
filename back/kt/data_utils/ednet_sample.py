""" ednet-kt1_sample-and-analysis.py
EdNet-KT1 数据集分层抽样与分析工具 (EdNet-KT1 Sampling & Analysis Tool)

核心功能:
1. 数据预处理: 扫描海量用户CSV文件，统计交互数，生成元数据(metadata.csv)。支持多进程加速。
2. 数据过滤: 根据交互序列长度清洗数据 (min_seq_len, max_seq_len)，剔除由于数据过少无法用于训练的用户。
3. 分布分析:
    - 生成描述性统计 (Descriptive Stats, Quantiles).
    - 绘制分布图: CDF累积分布图, 箱线图, 按Log10分桶的直方图.
    - 分析长尾分布特征。
4. 策略抽样:
    - Stratified (分层抽样): 保持原始数据的分布比例 (Default). 适合构建测试集，还原真实场景。
    - Balanced (均衡抽样): 各分桶抽取等量用户. 适合构建训练集，让模型充分学习长序列用户的模式，解决长尾问题。
5. 数据合并: 将抽样用户的单独CSV合并为单一的大文件，自动添加 user_id 列。

依赖配置:
- 原始数据路径 (DATA_DIR): /Users/fcatnoby/data/dataset/EdNet/KT1
- 输出结果路径 (OUTPUT_DIR): /Users/fcatnoby/data/dataset/EdNet/

使用示例:
1. 仅分析分布 (默认 min_seq_len=20):
    python ednet_sample.py

2. 不限制最小交互数，分析完整分布: (min_seq_len=0)
    python ednet_sample.py --min_seq_len 0

3. 抽样 5000 人，保持原始分布 (Stratified):
    python ednet_sample.py --size 5000

4. 抽样 5000 人，采用均衡抽样 (Balanced)，并保留长序列:
    python ednet_sample.py --size 5000 --strategy balanced

5. 强制重新扫描目录:
    python ednet_sample.py --force_scan

根据 32 万的有效用户基数：

- **5,000 人 (约 1.5%)** ：足够进行模型调试、超参搜索，速度快。
- **20,000 人 (约 6%)** ：适合发表论文用的最终对比实验，结果具有统计显著性。
"""
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import argparse

# ================= 配置区域 =================
# 原始数据路径
DATA_DIR = '/Users/fcatnoby/data/dataset/EdNet/KT1'
# 题目元数据路径 (用于判断题目是否正确)
QUESTIONS_PATH = '/Users/fcatnoby/data/dataset/EdNet/contents/questions.csv'
OUTPUT_DIR = '/Users/fcatnoby/data/dataset/EdNet/' 
# 缓存的元数据文件，避免重复扫描
METADATA_FILE = OUTPUT_DIR + 'ednet_kt1_metadata.csv'
# 抽样时的随机种子
RANDOM_SEED = 42
# 交互次数的分桶区间 (将被动态调整或用于默认)
# BINS = [0, 10, 100, 1000, 10000, float('inf')]
# LABELS = ['1-10', '11-100', '101-1000', '1001-10000', '10001+']

def get_row_count(file_path):
    """
    读取单个CSV文件的行数（交互数）。
    EdNet KT1 文件通常包含 header，所以数据行数 = 总行数 - 1。
    如果文件为空或读取失败，返回 None。
    """
    try:
        # 文件名如 u123.csv -> user_id = u123
        filename = os.path.basename(file_path)
        user_id = os.path.splitext(filename)[0]
        
        # 快速检查文件大小，如果为空直接跳过
        if os.path.getsize(file_path) == 0:
            return None
            
        # 仅读取 header 确认格式（如果确定所有文件格式一致，可以用更快的行数统计方法）
        # 这里为了稳健性，读取整个文件计算长度（对于大文件可能稍慢，但准确）
        # 使用 chunksize 分块读取防止内存溢出，但通常统计行数不需要加载整个 df
        # 优化方案：直接统计换行符，假设包含 header
        with open(file_path, 'rb') as f:
            count = 0
            for _ in f:
                count += 1
        
        # 减去 header 行（假设有 header）
        row_count = count - 1
        
        if row_count <= 0:
            return None
            
        return user_id, row_count, file_path
    except Exception as e:
        # 对于损坏的文件直接忽略
        return None

def generate_metadata(data_dir, output_file):
    """
    扫描目录，生成包含 user_id, row_count, file_path 的元数据 CSV。
    """
    print(f"正在扫描目录: {data_dir} ...")
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    print(f"找到 {len(csv_files)} 个 CSV 文件。开始统计行数（此步骤较耗时，请等待）...")

    # 使用多进程加速 I/O
    pool = Pool(processes=cpu_count())
    results = []
    
    # 使用 tqdm 显示进度条
    with tqdm(total=len(csv_files)) as pbar:
        for result in pool.imap_unordered(get_row_count, csv_files):
            if result:
                results.append(result)
            pbar.update()
            
    pool.close()
    pool.join()

    print("扫描完成，正在保存元数据...")
    df = pd.DataFrame(results, columns=['user_id', 'row_count', 'file_path'])
    df.to_csv(output_file, index=False)
    print(f"元数据已保存至: {output_file}")
    return df

def analyze_distribution(df, min_seq_len=0):
    """
    分析并绘制交互分布直方图，包含统计描述、CDF图和箱线图。
    """
    interactions = df['row_count']

    # --- Part 1: 详细描述性统计 ---
    print("\n=== 详细统计指标 ===")
    # 计算分位数
    quantiles = [0.25, 0.5, 0.75, 0.90, 0.95, 0.99]
    desc = interactions.describe(percentiles=quantiles)
    # 格式化输出避免科学计数法
    print(desc.apply(lambda x: format(x, '.2f')))
    
    # 计算极值贡献比例 (Top 1% 的用户贡献了多少比例的交互总数)
    total_interactions = interactions.sum()
    top_1_percent_interactions = interactions[interactions >= interactions.quantile(0.99)].sum()
    print(f"\n[偏度分析] Top 1% 活跃用户贡献了 {top_1_percent_interactions/total_interactions:.2%} 的总交互量")

    # --- Part 2: 累积分布函数 (CDF) ---
    plt.figure(figsize=(10, 6))
    sorted_data = np.sort(interactions)
    yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)
    
    plt.plot(sorted_data, yvals, label='CDF', linewidth=2)
    plt.xscale('log') # 使用对数坐标轴处理长尾
    plt.axvline(x=min_seq_len, color='r', linestyle='--', alpha=0.5, label=f'Min Seq ({min_seq_len})')
    # 标注中位数
    median = np.median(interactions)
    plt.axvline(x=median, color='g', linestyle=':', alpha=0.5, label=f'Median ({int(median)})')
    
    plt.title(f'Cumulative Distribution Function (Log Scale) \n(Filtered >= {min_seq_len})')
    plt.xlabel('Interaction Count (Log Scale)')
    plt.ylabel('Cumulative Proportion of Users')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'ednet_cdf.png'))
    print(f"CDF累积分布图已保存至: {os.path.join(OUTPUT_DIR, 'ednet_cdf.png')}")

    # --- Part 3: 箱线图 (Box Plot) ---
    plt.figure(figsize=(10, 4))
    # showfliers=False 隐藏异常值，否则长尾会让箱体压缩成一条线看不清
    plt.boxplot(interactions, vert=False, showfliers=False, patch_artist=True, 
                boxprops=dict(facecolor="lightblue"))
    plt.title(f'Box Plot (Outliers Hidden) - Filtered >= {min_seq_len}')
    plt.xlabel('Interaction Count')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ednet_boxplot.png'))
    print(f"箱线图(无离群点)已保存至: {os.path.join(OUTPUT_DIR, 'ednet_boxplot.png')}")

    # --- Part 4: 原有的分桶直方图 ---
    # 动态定义分桶，确保适应过滤后的数据
    # 基础分界点
    base_points = [100, 1000, 10000]
    # 过滤掉小于 min_seq_len 的点
    points = [p for p in base_points if p > min_seq_len]
    
    # 重新构建 BINS 和 LABELS
    bins = [0] + points + [float('inf')]
    
    # 构建 Labels
    labels = []
    # 修正：第一个桶从 min_seq_len 开始（如果数据已经被过滤了，那么最小值肯定是 min_seq_len）
    # 但为了 pd.cut 能够正确工作，左边界通常需要比最小值小一点，或者包含最小值
    # 这里我们假设 df 已经单纯包含了 >= min_seq_len 的数据
    # 为了显示好看，第一个 Label 应该是 f"{min_seq_len}-{points[0]}"
    
    # 实际用于 cut 的 bins 还是从 0 开始比较安全，能够捕获所有数据，或者从 min_seq_len-0.001 开始
    # 简单起见，我们手动指定 bins 适配当前场景
    
    # 场景: min_seq_len=20 -> bins: [0, 100, 1000, 10000, inf]. 
    # 数据本身>=20，所以第一组实际上是 [20, 100]。
    
    # 重新硬编码一套适应 KT 的逻辑
    if min_seq_len > 0:
        # 下限是 min_seq_len
        # BINS: [0, 100, 1000, 10000, inf] 依然适用，只是 Label 变了
        bins = [0, 100, 1000, 10000, float('inf')]
        if min_seq_len < 100:
            labels = [f'{min_seq_len}-100', '101-1000', '1001-10000', '10001+']
        else:
            # 极少情况，暂不细究
            labels = ['<100', '101-1000', '1001-10000', '10001+']
    else:
        bins = [0, 10, 100, 1000, 10000, float('inf')]
        labels = ['1-10', '11-100', '101-1000', '1001-10000', '10001+']

    # 分桶
    df['group'] = pd.cut(df['row_count'], bins=bins, labels=labels, right=True)
    
    # 统计各组数量
    group_counts = df['group'].value_counts().sort_index()
    # 加上比例
    group_ratios = group_counts / len(df)

    print("\n=== 数据分布统计 ===")
    # 输出各分桶的用户数和比例
    distribution_df = pd.DataFrame({
        'user_count': group_counts, 
        'user_ratio': group_ratios
    })
    print(distribution_df)
    
    # 计算比例
    total_users = len(df)
    group_ratios = group_counts / total_users
    
    # 绘图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(group_counts.index.astype(str), group_counts.values, color='skyblue', edgecolor='black')
    
    # 在柱状图上显示数值和比例
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 100,
                    f'{int(height)}\n({height/total_users:.1%})',
                    ha='center', va='bottom')

    plt.title(f'EdNet KT1 User Interaction Distribution (Total Users: {total_users})')
    plt.xlabel('Interaction Count Range')
    plt.ylabel('Number of Users')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ednet_distribution.png'))
    print(f"分布图已保存至: {os.path.join(OUTPUT_DIR, 'ednet_distribution.png')}")
    
    return group_ratios

def sample_users(df, sample_n=None, sample_ratio=None, strategy='stratified'):
    """
    分层抽样逻辑。
    参数:
    - strategy: 'stratified' (保持原比例分层抽样) 或 'balanced' (各分桶数量均等)
    """
    if sample_n is None and sample_ratio is None:
        raise ValueError("必须指定 sample_n (数量) 或 sample_ratio (比例)")

    sampled_dfs = []
    total_users = len(df)
    
    print(f"\n=== 开始抽样 (策略: {strategy}) ===")
    
    # 按照之前的分组进行遍历
    groups = df.groupby('group', observed=True)
    
    # 获取有效的分桶（有些分桶可能为空）
    valid_groups = [g for n, g in groups if len(g) > 0]
    num_bins = len(valid_groups)
    
    for group_name, group_df in groups:
        group_total = len(group_df)
        if group_total == 0:
            continue
            
        # 根据策略计算目标数量
        if strategy == 'balanced':
            # 均衡抽样
            if sample_n:
                target_count = int(np.ceil(sample_n / num_bins))
            else: # sample_ratio
                target_count = int(np.ceil((total_users * sample_ratio) / num_bins))
        else:
            # 默认：分层抽样 (Stratified) - 保持原比例
            group_prop = group_total / total_users
            if sample_n:
                target_count = int(np.round(sample_n * group_prop))
            else:
                target_count = int(np.round(group_total * sample_ratio))
        
        # 边界处理：如果不满则全取
        if target_count > group_total:
            print(f"分桶 [{group_name}]: 目标 {target_count} > 实际 {group_total}，全量保留。")
            actual_sample = group_df
        else:
            actual_sample = group_df.sample(n=target_count, random_state=RANDOM_SEED)
            
        sampled_dfs.append(actual_sample)
        print(f"分桶 [{group_name}]: 总数 {group_total} -> 抽取 {len(actual_sample)} (目标: {target_count})")
    
    final_sample_list = pd.concat(sampled_dfs)
    print(f"抽样完成。总抽取用户数: {len(final_sample_list)} (原始总数: {total_users})")
    
    return final_sample_list

def load_question_metadata(questions_path):
    """
    加载题目元数据。
    返回两个字典:
    1. correct_map: question_id -> correct_answer
    2. skill_map: question_id -> tags (将被重命名为 skill_id)
    """
    if not os.path.exists(questions_path):
        print(f"警告: 题目文件不存在 {questions_path}，将跳过正确率计算和技能ID合并。")
        return None, None
    
    try:
        print(f"Loading questions from {questions_path}...")
        df_q = pd.read_csv(questions_path)
        
        # 转换前检查列名
        required_cols = {'question_id', 'correct_answer', 'tags'}
        if not required_cols.issubset(df_q.columns):
             print(f"错误: 题目文件缺少必要的列 {required_cols - set(df_q.columns)}")
             return None, None
            
        # 1. 正确答案映射
        correct_map = df_q.set_index('question_id')['correct_answer'].to_dict()
        
        # 2. 技能(tags)映射
        # tags 可能是 "1;2;3" 格式，通常 GIKT 需要 skill_id
        # 此处我们只负责把 tags 这一列的内容 map 过去，列名改为 skill_id
        skill_map = df_q.set_index('question_id')['tags'].to_dict()
        
        return correct_map, skill_map
        
    except Exception as e:
        print(f"读取题目文件失败: {e}")
        return None, None

def merge_csvs(sample_list_df, output_csv_path, correct_map=None, skill_map=None):
    """
    读取抽样用户的 CSV 文件并合并为一个大文件。
    关键步骤：
    1. 添加 user_id 列。
    2. 如果提供了 correct_map，则根据 user_answer 和 correct_answer 计算 'correct' 列 (0/1)。
    3. 如果提供了 skill_map，则添加 'skill_id' 列 (即原始的 tags)。
    """
    print(f"\n正在合并 {len(sample_list_df)} 个用户的文件到 {output_csv_path} ...")
    
    # 检查输出文件是否存在，存在则删除，避免追加
    if os.path.exists(output_csv_path):
        os.remove(output_csv_path)

    first_file = True
    
    # 使用 tqdm 显示合并进度
    for _, row in tqdm(sample_list_df.iterrows(), total=len(sample_list_df)):
        file_path = row['file_path']
        user_id = row['user_id']
        
        try:
            # 读取单个用户数据
            user_df = pd.read_csv(file_path)
            
            # 添加 user_id 列 (放在第一列比较好看)
            user_df.insert(0, 'user_id', user_id)
            
            # 增补技能 ID (tags)
            if skill_map:
                # 映射 skill_id (即 tags)
                # 使用 map 由于比 merge 快很多
                user_df['skill_id'] = user_df['question_id'].map(skill_map)
                
                # 如果找不到对应的 skill_id (fillna)，根据需要处理，这里暂且保留 NaN 或填 0/-1
                # user_df['skill_id'] = user_df['skill_id'].fillna(-1)

            # 计算正确性 (如果提供了映射表)
            if correct_map:
                # 映射标准答案
                # 注意：map 如果找不到key会产生 NaN，需要处理
                user_df['correct_answer'] = user_df['question_id'].map(correct_map)
                
                # 比较用户答案和标准答案
                # 逻辑：
                # 1. 如果 user_answer == correct_answer -> 1
                # 2. 否则 -> 0
                # 注意处理 user_answer 可能为空的情况，或者 correct_answer 为空的情况
                
                # 确保都是字符串并小写 (假设 EdNet 格式标准，通常是 a,b,c,d)
                # 为了安全起见，转 string striped lower
                # 但出于性能考虑，如果数据比较规整，直接比较即可
                
                # 方法 A: 向量化比较
                # 填充 NaN 为不可匹配的字符
                ua = user_df['user_answer'].fillna('')
                ca = user_df['correct_answer'].fillna('')
                
                user_df['correct'] = (ua == ca).astype(int)
                
                # 也可以选择移除 correct_answer 列，保持文件整洁，只留 correct
                user_df.drop(columns=['correct_answer'], inplace=True)

            # 写入模式：如果是第一个文件，'w'并写header；否则'a'并不写header
            mode = 'w' if first_file else 'a'
            header = first_file
            
            user_df.to_csv(output_csv_path, mode=mode, header=header, index=False)
            
            first_file = False
        except Exception as e:
            print(f"读取或写入文件 {file_path} 失败: {e}")

    print(f"合并完成！文件保存至: {output_csv_path}")

def main():
    parser = argparse.ArgumentParser(description="EdNet KT1 分层抽样工具")
    parser.add_argument('--size', type=int, default=None, help='目标抽样学生数量 (例如 5000)')
    parser.add_argument('--ratio', type=float, default=None, help='目标抽样比例 (例如 0.01)')
    parser.add_argument('--min_seq_len', type=int, default=20, help='最小交互序列长度，小于此值的用户将被丢弃 (默认: 20)')
    parser.add_argument('--max_seq_len', type=int, default=None, help='最大交互序列长度，大于此值的用户将被丢弃 (默认: None，即保留长序列)')
    parser.add_argument('--strategy', type=str, choices=['stratified', 'balanced'], default='stratified', help='抽样策略：stratified(分层保持比例, 默认) 或 balanced(各桶均衡)')
    parser.add_argument('--force_scan', action='store_true', help='强制重新扫描目录生成元数据')
    # 不进行合并操作
    parser.add_argument('--nomerge', action='store_true', help='仅抽样用户列表，不进行 CSV 合并操作')

    args = parser.parse_args()
    
    # 0. 准备工作
    if not os.path.exists(DATA_DIR):
        print(f"错误: 数据目录不存在 {DATA_DIR}")
        return

    # 1. 获取/生成元数据
    if args.force_scan or not os.path.exists(METADATA_FILE):
        df_meta = generate_metadata(DATA_DIR, METADATA_FILE)
    else:
        print(f"读取现有元数据: {METADATA_FILE}")
        df_meta = pd.read_csv(METADATA_FILE)

    # 1.5 过滤短序列数据
    original_count = len(df_meta)
    
    # 构建过滤条件
    condition = df_meta['row_count'] >= args.min_seq_len
    filter_desc = f">= {args.min_seq_len}"
    
    if args.max_seq_len:
        condition = condition & (df_meta['row_count'] <= args.max_seq_len)
        filter_desc += f" & <= {args.max_seq_len}"
        
    df_meta = df_meta[condition].copy()
    filtered_count = len(df_meta)
    
    print(f"\n=== 数据过滤 ({filter_desc}) ===")
    print(f"原始用户数: {original_count}")
    print(f"保留用户数: {filtered_count} (丢弃 {original_count - filtered_count})")
    print(f"有效数据占比: {filtered_count/original_count:.2%}")
    
    if filtered_count == 0:
        print("错误: 过滤后没有剩余用户，请检查阈值设置。")
        return

    # 2. 分析分布
    analyze_distribution(df_meta, min_seq_len=args.min_seq_len)
    
    # 3. 执行抽样
    if args.size or args.ratio:
        sampled_users = sample_users(df_meta, sample_n=args.size, sample_ratio=args.ratio, strategy=args.strategy)
        
        suffix = f"size_{len(sampled_users)}_{args.strategy}" if args.size else f"ratio_{args.ratio}_{args.strategy}"
        
        # 4. 合并数据（可选，默认为 True）
        if not args.nomerge:
            output_filename = os.path.join(OUTPUT_DIR, f"Ednet-KT1_sample_{suffix}.csv")
            
            # 加载题目元数据 (答案 和 技能Tags)
            correct_map, skill_map = load_question_metadata(QUESTIONS_PATH)
            
            # 将 tags 作为 skill_id 传入合并函数
            merge_csvs(sampled_users, output_filename, correct_map=correct_map, skill_map=skill_map)
        
        # 5. 保存抽样用户的ID列表并计算每个用户交互式的 log10 值，方便后续查找与分析
        sampled_users['log10_row_count'] = np.log10(sampled_users['row_count']).round(2)
        sampled_users[['user_id', 'row_count', 'log10_row_count']].to_csv(os.path.join(OUTPUT_DIR, f"sampled_users_list_with_log10_{suffix}.csv"), index=False)
        print(f"抽样用户的ID列表已保存至 {os.path.join(OUTPUT_DIR, f'sampled_users_list_with_log10_{suffix}.csv')}")
        
        
        # 6. 保存 log10 列表至单独文件
        log10_list_data = sampled_users['log10_row_count'].tolist()
        row_count_list_data = sampled_users['row_count'].tolist()
        # 保存为 txt 文件，以','分隔，首行添加说明
        # 在线直方图生成器： https://www.bchrt.com/tools/histogram-generator/
        with open(os.path.join(OUTPUT_DIR, f"sampled_users_log10_list_{suffix}.txt"), 'w') as f:
            f.write("# 在线直方图生成器： https://www.bchrt.com/tools/histogram-generator/\n")
            f.write(','.join(map(str, log10_list_data)))
            f.write('\n\n')
            f.write(','.join(map(str, row_count_list_data)))
        
        print(f"抽样用户的 log10 列表已保存至: {os.path.join(OUTPUT_DIR, f'sampled_users_log10_list_{suffix}.txt')}")

    else:
        print("\n未指定 --size 或 --ratio，仅执行分布分析。由需抽样请添加参数运行。")
        print("示例: python ednet_sample.py --size 5000")

if __name__ == "__main__":
    main()





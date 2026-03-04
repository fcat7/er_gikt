#!/usr/bin/env python3
"""
从 CSV 文件中提取指定列的数据，并输出去重后的唯一值。

用法示例：
    python extract_unique.py data.csv --columns name
    python extract_unique.py data.csv --columns name age --delimiter ';' --output unique.txt
    python extract_unique.py data.csv --columns 0 2 --no-header   # 无标题行，使用列索引
"""

import csv
import argparse
import sys
from collections import OrderedDict

def main():
    parser = argparse.ArgumentParser(description="提取 CSV 指定列的唯一值")
    parser.add_argument("csv_file", help="输入的 CSV 文件路径")
    parser.add_argument("--columns", "-c", nargs="+", required=True,
                        help="要提取的列名（如果有标题行）或列索引（如果没有标题行）")
    parser.add_argument("--delimiter", "-d", default=",",
                        help="CSV 分隔符，默认为逗号")
    parser.add_argument("--no-header", action="store_true",
                        help="如果 CSV 没有标题行，请使用此选项，此时 --columns 应指定列索引（从 0 开始）")
    parser.add_argument("--output", "-o",
                        help="输出文件路径，不指定则打印到标准输出")
    parser.add_argument("--preserve-order", action="store_true", default=True,
                        help="保留首次出现的顺序（默认开启）")
    args = parser.parse_args()

    # 确定列标识是列名还是索引
    if args.no_header:
        # 尝试将列参数转换为整数索引
        try:
            col_indices = [int(c) for c in args.columns]
        except ValueError:
            sys.exit("错误：当使用 --no-header 时，--columns 必须指定整数索引（例如 0 1 2）")
    else:
        col_names = args.columns
        col_indices = None  # 稍后读取标题后确定

    unique_rows = OrderedDict() if args.preserve_order else {}

    try:
        with open(args.csv_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=args.delimiter)

            # 处理标题行
            if not args.no_header:
                try:
                    header = next(reader)
                except StopIteration:
                    sys.exit("错误：CSV 文件为空或无法读取标题行")
                # 找到每个列名对应的索引
                col_indices = []
                missing = []
                for name in col_names:
                    try:
                        col_indices.append(header.index(name))
                    except ValueError:
                        missing.append(name)
                if missing:
                    sys.exit(f"错误：找不到列名 {missing}，可用的列名为：{header}")

            # 遍历数据行，提取指定列
            for row_num, row in enumerate(reader, start=2 if not args.no_header else 1):
                try:
                    # 提取所需列的值，如果多列则组成元组
                    extracted = tuple(row[i] for i in col_indices)
                except IndexError:
                    sys.exit(f"错误：第 {row_num} 行缺少列，需要的索引 {col_indices}，实际只有 {len(row)} 列")

                if args.preserve_order:
                    unique_rows[extracted] = None
                else:
                    # 如果不保留顺序，直接用集合去重
                    # 但这里为了统一，使用 dict 并忽略值
                    unique_rows[extracted] = None

    except FileNotFoundError:
        sys.exit(f"错误：文件 '{args.csv_file}' 不存在")
    except Exception as e:
        sys.exit(f"读取文件时发生错误：{e}")

    # 准备输出
    output_lines = list(unique_rows.keys())

    # 输出到文件或 stdout
    out_fh = open(args.output, 'w', newline='', encoding='utf-8') if args.output else sys.stdout
    try:
        if len(col_indices) == 1:
            # 单列，直接输出每行一个值
            for val in output_lines:
                # val 是元组，取出第一个元素
                print(val[0], file=out_fh)
        else:
            # 多列，用 CSV 格式输出（使用原分隔符）
            writer = csv.writer(out_fh, delimiter=args.delimiter, lineterminator='\n')
            for row in output_lines:
                writer.writerow(row)
    finally:
        if args.output:
            out_fh.close()

# python extract_unique.py "H:\dataset\assist12\2012-2013-data-with-predictions-4-final.csv" --columns skill_id skill
if __name__ == "__main__":
    main()
import os
import sys


from config import Config
from util.plot_utils import plot_comparison


# 获取项目配置
# 注意：Config 初始化需要 dataset_name，这里给一个默认的或者占位符，因为我们主要使用 path 配置
# 或者如果 plot 不需要特定数据集配置，Config('assist09') 即可
chart_data_dir = Config().path.CHART_DIR
print(f"Reading charts from: {chart_data_dir}")

# 配置要对比的日志文件
# 键为文件名，值为图例显示的 Label
logs_to_compare_dict = {
    '20260108_0257_aver.txt': 'use_cognitive_model & no_BCELoss',
    '20260108_0308_aver.txt': 'no_cognitive_model & no_BCELoss',
    '20260108_1332_aver.txt': 'use_congnitive_model & use_BCELoss',
    # '20260108_1430_aver.txt': 'use_cognitive_model & no_BCELoss',
    '20260108_1537_aver.txt': 'no_cognitive_model & use_BCELoss'
}

# 示例：全量数据集 assist09
# logs_to_compare_dict = { 
#     '20260108_1209_aver.txt': 'no_cognitive_model',
#     '20260108_1207_aver.txt': 'use_cognitive_model',
# }

logs_to_compare_dict = { # 重构代码后的对比实验
    '20260112_1754_aver.txt': 'default',
    # '20260112_2154_aver.txt': 'use input attention',
    # '20260112_2218_aver.txt': 'use input attention-v2',
    '20260112_2233_aver.txt': 'use input attention-v3',
    # '20260112_2251_aver.txt': 'use input attention-v4',
    # '20260112_2320_aver.txt': 'use input attention-v4_fix_cognitive',
    # '20260113_0005_aver.txt': 'use input attention-v3.1',
    '20260113_0058_aver.txt': 'use input attention-v3.2_alignment',
    '20260113_0131_aver.txt': 'use input attention-v3.2_no_hsei',
}

logs_to_compare_dict = { # 重构代码后的对比实验
    '20260112_1754_aver.txt': 'default',
    '20260113_0058_aver.txt': 'use input attention-v3.2_alignment',
    '20260113_1626_aver.txt': 'use input attention-v3.2 + dev2_v1',
    '20260113_1928_aver.txt': 'use input attention-v3.2 + dev2_v2' # 补充 Target Projection (目标项投影)。
}

if logs_to_compare_dict:
    plot_comparison(logs_to_compare_dict, metric_type='auc', base_dir=chart_data_dir)
else:
    print("请配置 logs_to_compare_dict 以生成图表。")

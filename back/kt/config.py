
import os
import torch

class BaseConfig:
    """
    基础配置类，定义通用路径和参数
    """
    # 项目根目录 (kt_fzq)
    # 假设 config.py 在 kt_fzq/kt/config.py，所以向上两级
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    
    # 基础目录结构
    # DATASET_DIR = os.path.join(PROJECT_ROOT, 'dataset')          # 原始数据集目录
    # 临时指向旧位置，直到用户移动文件
    # DATASET_DIR = os.path.join(PROJECT_ROOT, 'dataset')  # 原始数据集目录
    DATASET_DIR = '/Users/fcatnoby/data/dataset' # 数据集原始文件目录，文件过大，自行下载放置
    PROCESSED_DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')      # 处理后数据根目录

    @property
    def PROCESSED_DATA_DIR(self):
        """返回当前数据集的处理后数据目录: kt/data/dataset_name"""
        if hasattr(self, 'DATASET_NAME'):
            return os.path.join(self.PROCESSED_DATA_ROOT, self.DATASET_NAME)
        return self.PROCESSED_DATA_ROOT

    MODULE_DIR = os.path.join(PROJECT_ROOT, 'model')            # 模型模块目录
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output/')             # 日志输出目录
    LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')                   # 日志目录
    CHART_DIR = os.path.join(OUTPUT_DIR, 'chart_data')            # 图表保存目录
    REPORT_DIR = os.path.join(OUTPUT_DIR, 'data_analysis_reports') # 报告目录

    # 预处理通用参数
    MIN_SEQ_LEN = 20   # 最小交互序列长度
    MAX_SEQ_LEN = 200  # 最大交互序列长度
    MIN_INTERACTIONS = 3 # 最小交互数 (用于初步筛选)
    
    # 划分参数
    TEST_RATIO = 0.2
    RANDOM_SEED = 42
    
    # 默认编码
    ENCODING = 'utf-8'

    # 计算设备: 优先使用gpu
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 日志颜色
    LOG_B = '\033[1;34m' # 蓝色
    LOG_Y = '\033[1;33m' # 黄色
    LOG_G = '\033[1;36m' # 深绿色
    LOG_END = '\033[m' # 结束标记

    # 数据超参数
    SIZE_Q_FEATURE = 8
    SIZE_EMBEDDING = 100 # 问题和技能的嵌入向量维度

    @property
    def raw_data_path(self):
        """返回原始数据文件的绝对路径"""
        raise NotImplementedError("Subclasses must implement raw_data_path")

class Assist09Config(BaseConfig):
    DATASET_NAME = 'assist09'
    RAW_FILENAME = 'assistments_2009_2010_non_skill_builder_data_new.csv'
    ENCODING = 'ISO-8859-1'
    
    # 列名映射 (Standard -> Raw)
    # Standard columns: user_id, problem_id, skill_id, correct, timestamp
    COLUMN_MAP = {
        'user_id': 'user_id',
        'problem_id': 'problem_id',
        'skill_id': 'skill_id',
        'correct': 'correct',
        'timestamp': 'order_id',
        'response_time': 'ms_first_response', # 可选
        # 'overlap_time': 'overlap_time',       # 可选
        'original': 'original'                # original： 1 = Main problem；0 = Scaffolding problem
    }

    @property
    def raw_data_path(self):
        return os.path.join(self.DATASET_DIR, self.RAW_FILENAME)


class Assist12Config(BaseConfig):
    DATASET_NAME = 'assist12'
    RAW_FILENAME = '2012-2013-data-with-predictions-4-final.csv'
    ENCODING = 'utf-8'
    
    # 列名映射 (Standard -> Raw)
    COLUMN_MAP = {
        'user_id': 'user_id',
        'problem_id': 'problem_id',
        'skill_id': 'skill_id',
        'correct': 'correct',
        'timestamp': 'start_time',
    }

    @property
    def raw_data_path(self):
        return os.path.join(self.DATASET_DIR, self.RAW_FILENAME)

class EdNetKT1Config(BaseConfig):
    DATASET_NAME = 'ednet_kt1'
    RAW_FILENAME = 'EdNet/Ednet-KT1_sample_size_5000_stratified.csv'
    ENCODING = 'utf-8'
    
    # 列名映射 (Standard -> Raw)
    COLUMN_MAP = {
        'user_id': 'user_id',
        'problem_id': 'question_id',
        'skill_id': 'skill_id',
        'correct': 'correct',
        'timestamp': 'timestamp',
        'response_time': 'elapsed_time', # 可选
    }

    @property
    def raw_data_path(self):
        return os.path.join(self.DATASET_DIR, self.RAW_FILENAME)

# 配置映射表
CONFIG_MAP = {
    'assist09': Assist09Config,
    'assist12': Assist12Config,
    'ednet_kt1': EdNetKT1Config
}

def get_config(dataset_name='assist09'):
    """
    工厂函数：根据数据集名称获取配置对象
    支持变体名称，例如 "assist09-sample-10%" 会加载 Assist09Config，
    并自动将 PROCESSED_DATA_DIR 指向 "kt/data/assist09-sample-10%"
    """
    # 1. 精确匹配
    if dataset_name in CONFIG_MAP:
        return CONFIG_MAP[dataset_name]()
    
    # 2. 前缀模糊匹配 (支持 assist09-sample-10% 这种变体)
    # 我们遍历以知的配置类型，看 requested_name 是否以它开头
    for key, config_cls in CONFIG_MAP.items():
        if dataset_name.startswith(key):
            # 实例化基础配置
            config_instance = config_cls()
            # @fix_fzq: 动态覆盖 DATASET_NAME
            # 这样 config.PROCESSED_DATA_DIR 就会指向传入的目录名 (如 assist09-sample-10%)
            # 而不是硬编码的 assist09
            config_instance.DATASET_NAME = dataset_name
            return config_instance

    raise ValueError(f"Unknown dataset: {dataset_name}. Base types available: {list(CONFIG_MAP.keys())}")

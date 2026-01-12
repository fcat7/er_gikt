import os
import torch
import toml

# 全局设备定义
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 日志颜色
COLOR_LOG_B = '\033[1;34m'
COLOR_LOG_Y = '\033[1;33m'
COLOR_LOG_G = '\033[1;36m'
COLOR_LOG_END = '\033[m'
RANDOM_SEED = 42
MIN_SEQ_LEN = 20
MAX_SEQ_LEN = 200

class PathConfig:
    def __init__(self, paths, project_root):
        def _get(key):
            val = paths.get(key)
            if not val: raise ValueError(f"Missing config: paths.{key}")
            if os.path.isabs(val): return val
            return os.path.join(project_root, val)

        self.PROCESSED_DATA_ROOT = _get('processed_data_root')
        self.MODEL_DIR = _get('model_dir')
        self.OUTPUT_DIR = _get('output_dir')
        self.LOG_DIR = _get('log_dir')
        self.CHART_DIR = _get('chart_dir')
        self.REPORT_DIR = _get('report_dir')
        
        # Ensure directories exist
        for d in [self.PROCESSED_DATA_ROOT, self.OUTPUT_DIR, self.LOG_DIR, self.CHART_DIR, self.REPORT_DIR]:
            os.makedirs(d, exist_ok=True)

class DatasetConfig:
    def __init__(self, data):
        self.DATA_NAME = data.get('name')
        self.FILE_PATH = data.get('file_path') 
        self.ENCODING = data.get('encoding', 'utf-8')
        self.COLUMN_MAP = data.get('column_map', {})

class Config:
    def __init__(self, dataset_name=None):
        self.DATASET_NAME = dataset_name
        self.PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
        self.CONFIG_ROOT = os.path.join(self.PROJECT_ROOT, 'config')

        # 1. Base project config
        self.config_data = {
            'paths': {
                'processed_data_root': 'data',
                'model_dir': 'model',
                'output_dir': 'output',
                'log_dir': 'output/logs',
                'chart_dir': 'output/chart_data',
                'report_dir': 'output/data_analysis_reports'
            }
        }
        
        try:
            project_config_path = os.path.join(self.CONFIG_ROOT, 'project.toml')
            loaded_config = self._load_toml(project_config_path)
            if 'paths' in loaded_config:
                self.config_data['paths'].update(loaded_config['paths'])
        except Exception as e:
            print(f"Warning: Failed to load project.toml: {e}")

        # Init PathConfig
        self.path = PathConfig(self.config_data['paths'], self.PROJECT_ROOT)
        
        # 2. Dataset config
        dataset_config_path = self._find_dataset_config(dataset_name)
        if dataset_config_path:
            dataset_data = self._load_toml(dataset_config_path)
            self.dataset = DatasetConfig(dataset_data)
        else:
            available_datasets = self._list_available_datasets()
            print(f"Dataset config not found for: '{dataset_name}'. Available: {available_datasets}")
            self.dataset = None

    def _list_available_datasets(self):
        dataset_dir = os.path.join(self.CONFIG_ROOT, 'datasets')
        if not os.path.exists(dataset_dir):
            return []
        return sorted([f[:-5] for f in os.listdir(dataset_dir) if f.endswith('.toml')])

    def _load_toml(self, path):
        if not os.path.exists(path):
            return {}
        try:
            with open(path, 'rb') as f:
                return toml.load(f)
        except TypeError:
            with open(path, 'r', encoding='utf-8') as f:
                return toml.load(f)

    def _find_dataset_config(self, dataset_name):
        if dataset_name is None: return None
        dataset_dir = os.path.join(self.CONFIG_ROOT, 'datasets')
        if not os.path.exists(dataset_dir): os.makedirs(dataset_dir, exist_ok=True)
            
        exact_match = os.path.join(dataset_dir, f"{dataset_name}.toml")
        if os.path.exists(exact_match): return exact_match
            
        for filename in os.listdir(dataset_dir):
            if not filename.endswith('.toml'): continue
            if dataset_name.startswith(filename[:-5]):
                return os.path.join(dataset_dir, filename)
        return None

    @property
    def PROCESSED_DATA_DIR(self):
        if hasattr(self, 'DATASET_NAME') and self.DATASET_NAME:
            return os.path.join(self.path.PROCESSED_DATA_ROOT, self.DATASET_NAME)
        return self.path.PROCESSED_DATA_ROOT

# 如何标注其返回值类型
def get_config(dataset_name) -> Config:
    return Config(dataset_name=dataset_name)


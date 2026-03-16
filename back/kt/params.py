from dataclasses import dataclass, field, asdict
from typing import Tuple
import os
import toml

@dataclass
class CommonParams:
    num_workers: int = 0  # Windows下建议设为0，避免多进程启动开销(spawn overhead)；Mac/Linux可设为2或4
    random_seed: int = 42

@dataclass
class ModelParams:
    emb_dim: int = 100
    size_q_neighbors: int = 4
    size_s_neighbors: int = 10
    agg_hops: int = 3
    # 拆分原 dropoput = [0.2, 0.4] 为具名参数
    dropout_linear: float = 0.2 # 针对 LSTM/Linear 层的 Dropout
    dropout_gnn: float = 0.4    # 针对 GNN 聚合后的 Dropout
    drop_edge_rate: float = 0.0 # [New] 图结构增强：随机丢边率 (建议 0.1~0.2)
    
    # @add_fzq: Feature Perturbation (特征扰动)
    feature_noise_scale: float = 0.0  # 噪声强度 σ ~ N(0, σ²)，0.0 表示关闭。推荐 [0.005, 0.01, 0.02]

    rank_k: int = 10
    use_cognitive_model: bool = True
    cognitive_mode: str = 'autonomous' # 认知模型模式: 'classic' 或 'autonomous' (推荐)
    pre_train: bool = False
    hard_recap: bool = True
    agg_method: str = 'gcn'
    recap_source: str = 'hssi'  # 回顾特征来源: 'hsei' 或 'hssi'
    use_pid: bool = False # 是否使用 PID-GIKT 控制器架构
    pid_mode: str = 'global' # PID 模式: 'global' (标量) 或 'domain' (向量)
    pid_ema_alpha: float = 0.1 # PID 积分项衰减率
    pid_lambda: float = 1.0 # PID 微分项缩放系数
    pid_init_i: float = 0.5 # PID 积分项权重初始化 (W_i)
    pid_init_d: float = 0.1 # PID 微分项权重初始化 (W_d)
    guessing_prob_init: float = 0.05 # 4PL 猜测率初始概率 (0.05 = 5%)
    slipping_prob_init: float = 0.02 # 4PL 失误率初始概率 (0.02 = 2%)
    use_4pl_irt: bool = True # 是否启用 4PL-IRT (Path 2 特性)


@dataclass
class TrainParams:
    dataset_name: str
    batch_size: int = 32
    learning_rate: float = 0.001
    lr_gamma: float = 0.95
    epochs: int = 100
    weight_decay: float = 0.0 # 权重衰减 (L2 正则化)
    enable_data_augmentation: bool = False # 是否启用数据增强 (总开关)
    aug_mask_prob: float = 0.1 # 随机Mask概率 (默认 10%)
    reg_4pl: float = 1e-5 # 4PL/IRT 正则化系数 (针对区分度/猜测率/失误率)
    patience: int = 0 # 早停机制 (0 表示禁用)
    enable_lr_scheduler: bool = True # 是否启用学习率调度器 (余弦退火)
    prefetch_factor: int = 4
    k_fold: int = 1
    verbose: bool = True
    save_model: bool = False # 是否保存模型
    amp_enabled: bool = False # 是否启用 AMP
    gradient_clip_norm: float = 1.0 # 梯度裁剪阈值
    label_smoothing: float = 0.0 # BCF 损失的标签平滑 (如 0.05-0.1)

@dataclass
class HyperParameters:
    """
    聚合所有参数的根类
    """
    common: CommonParams = field(default_factory=CommonParams)
    model: ModelParams = field(default_factory=ModelParams)
    train: TrainParams = field(default_factory=TrainParams)
    def __str__(self):
        def _fmt(name, obj):
            # 将配置对象转换为紧凑的 key=value 字符串
            params_str = ", ".join(f"{k}={v}" for k, v in asdict(obj).items())
            return f"  [{name:<6}] {params_str}"
            
        return "\n".join([
            "HyperParameters:",
            _fmt("Common", self.common),
            _fmt("Model", self.model),
            _fmt("Train", self.train)
        ])

    
    @staticmethod
    def from_dict(data: dict) -> 'HyperParameters':
        """
        从字典递归创建对象
        """
        common_data = data.get('common', {})
        model_data = data.get('model', {})
        train_data = data.get('train', {})

        return HyperParameters(
            common=CommonParams(**common_data),
            model=ModelParams(**model_data),
            train=TrainParams(**train_data)
        )
    
    @classmethod
    def load(cls, exp_config_path=None):
        """
        直接从配置文件加载超参数
        """
        PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
        CONFIG_ROOT = os.path.join(PROJECT_ROOT, 'config')
        
        # 1. 加载全局配置
        final_config = cls._load_toml(os.path.join(CONFIG_ROOT, 'hyper_parameters-global.toml'))

        # 2. 加载实验配置
        if exp_config_path and os.path.exists(exp_config_path):
            cls._merge_config(final_config, cls._load_toml(exp_config_path))
        else:
            raise FileNotFoundError(f"实验配置文件未找到: {exp_config_path}")
            
        return cls.from_dict(final_config)

    @staticmethod
    def _load_toml(path):
        if not os.path.exists(path):
            return {}
        try:
            with open(path, 'rb') as f:
                return toml.load(f)
        except TypeError:
            with open(path, 'r', encoding='utf-8') as f:
                return toml.load(f)

    @staticmethod
    def _merge_config(base, update):
        for k, v in update.items():
            if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                HyperParameters._merge_config(base[k], v)
            else:
                base[k] = v

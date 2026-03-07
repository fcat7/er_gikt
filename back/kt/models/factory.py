import torch
import os
from scipy import sparse

# 导入基准模型
from baselines import DKT, DKVMN, AKT, SimpleKT, QIKT, LBKT
from .gikt import GIKT

class ModelFactory:
    """
    模型工厂类：根据模型名称和参数实例化对应的模型
    """
    @staticmethod
    def get_model(model_name, num_question, num_skill, device, config=None, **kwargs):
        name_key = model_name.lower()
        model = None
        
        if name_key == 'gikt':
            # GIKT 特有的图聚合参数，通常从 kwargs 或外部逻辑传入
            q_neighbors = kwargs.get('q_neighbors')
            s_neighbors = kwargs.get('s_neighbors')
            qs_table = kwargs.get('qs_table')
            
            if q_neighbors is None or s_neighbors is None or qs_table is None:
                # 如果没有直接传，尝试从磁盘加载 (兼容自动化脚本)
                if config is None:
                    raise ValueError("GIKT requires neighbors and qs_table or a valid config to load them.")
                from util.utils import build_adj_list, gen_gikt_graph
                # 先加载邻居列表
                q_neighbors_list, s_neighbors_list = build_adj_list(config.PROCESSED_DATA_DIR)
                # 再采样邻居
                q_neighbors, s_neighbors = gen_gikt_graph(
                    q_neighbors_list,
                    s_neighbors_list,
                    kwargs.get('size_q_neighbors', 4),
                    kwargs.get('size_s_neighbors', 10)
                )
                # 加载 qs_table
                qs_table = torch.tensor(
                    sparse.load_npz(os.path.join(config.PROCESSED_DATA_DIR, 'qs_table.npz')).toarray(),
                    dtype=torch.int64
                ).to(device)

            model = GIKT(
                num_question=num_question,
                num_skill=num_skill,
                q_neighbors=q_neighbors,
                s_neighbors=s_neighbors,
                qs_table=qs_table,
                emb_dim=kwargs.get('emb_dim', 100),
                agg_hops=kwargs.get('agg_hops', 3),
                dropout_linear=kwargs.get('dropout_linear', 0.1),
                dropout_gnn=kwargs.get('dropout_gnn', 0.1),
                drop_edge_rate=kwargs.get('drop_edge_rate', 0.1),
                feature_noise_scale=kwargs.get('feature_noise_scale', 0.01),
                hard_recap=kwargs.get('hard_recap', True),
                use_cognitive_model=kwargs.get('use_cognitive_model', True),
                cognitive_mode=kwargs.get('cognitive_mode', 'classic'),
                pre_train=kwargs.get('pre_train', True),
                data_dir=config.PROCESSED_DATA_DIR if config else None,
                agg_method=kwargs.get('agg_method', 'gcn'),
                recap_source=kwargs.get('recap_source', 'hsei'),
                use_pid=kwargs.get('use_pid', True),
                pid_mode=kwargs.get('pid_mode', 'domain'),
                pid_ema_alpha=kwargs.get('pid_ema_alpha', 0.1),
                pid_lambda=kwargs.get('pid_lambda', 1.0),
                pid_init_i=kwargs.get('pid_init_i', 0.5),
                pid_init_d=kwargs.get('pid_init_d', 0.1),
                guessing_prob_init=kwargs.get('guessing_prob_init', 0.05),
                slipping_prob_init=kwargs.get('slipping_prob_init', 0.02),
                use_4pl_irt=kwargs.get('use_4pl_irt', True)
            ).to(device)

        elif name_key == 'dkt':
            emb_size = kwargs.get('emb_size', 64)
            dropout = kwargs.get('dropout', 0.1)
            model = DKT(num_question, num_skill, emb_dim=emb_size, dropout=dropout).to(device)
            
        elif name_key == 'dkvmn':
            emb_size = kwargs.get('emb_size', 64)
            dim_s = kwargs.get('dim_s', emb_size)
            size_m = kwargs.get('size_m', 20)
            dropout = kwargs.get('dropout', 0.1)
            model = DKVMN(num_question, dim_s=dim_s, size_m=size_m, dropout=dropout).to(device)
            
        elif name_key == 'akt':
            emb_size = kwargs.get('emb_size', 64)
            d_model = kwargs.get('d_model', emb_size)
            n_blocks = kwargs.get('n_blocks', 2)
            d_ff = kwargs.get('d_ff', 256)
            num_attn_heads = kwargs.get('num_attn_heads', 8)
            dropout = kwargs.get('dropout', 0.1)
            model = AKT(n_question=num_question, d_model=d_model, n_blocks=n_blocks, 
                    d_ff=d_ff, num_attn_heads=num_attn_heads, dropout=dropout).to(device)

        elif name_key == 'simplekt':
            emb_size = kwargs.get('emb_size', 64)
            d_model = kwargs.get('d_model', emb_size)
            dropout = kwargs.get('dropout', 0.1)
            model = SimpleKT(n_question=num_question, d_model=d_model, dropout=dropout).to(device)

        elif name_key == 'qikt':
            emb_size = kwargs.get('emb_size', 64)
            dropout = kwargs.get('dropout', 0.1)
            if config is None:
                raise ValueError("QIKT requires config to load qs_table")
            qs_table = torch.tensor(sparse.load_npz(os.path.join(config.PROCESSED_DATA_DIR, 'qs_table.npz')).toarray(), dtype=torch.float32).to(device)
            model = QIKT(num_question=num_question, num_concept=num_skill, qs_table=qs_table, dim_emb=emb_size, dropout=dropout).to(device)
        
        elif name_key == 'lbkt':
            emb_size = kwargs.get('emb_size', 64)
            dim_factor = kwargs.get('dim_factor', 1)
            if config is None:
                raise ValueError("LBKT requires config to load qs_table")
            qs_table = torch.tensor(sparse.load_npz(os.path.join(config.PROCESSED_DATA_DIR, 'qs_table.npz')).toarray(), dtype=torch.float32).to(device)
            model = LBKT(num_question=num_question, num_concept=num_skill, qs_table=qs_table, dim_h=emb_size, dim_factor=dim_factor).to(device)
            
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
        # 为模型实例添加 model_name 属性，方便 Trainer 识别逻辑
        if model is not None:
            model.model_name = name_key
        return model

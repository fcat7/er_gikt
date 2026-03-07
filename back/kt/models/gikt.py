import os
import json
import numpy as np
import torch
import math
import torch.nn.functional as F
from torch.nn import Module, Embedding, Linear, ModuleList, Dropout, LSTMCell, MultiheadAttention

from config import DEVICE

# @add_fzq 2025-12-25 -------------------------------------------
class AutonomousCognitiveCell(Module):
    """
    Scheme 2: 自治型认知单元 (Autonomous Cognitive Cell)
    
    设计哲学 (Design Philosophy):
    1. 接纳不完美 (Accept Imperfection): 承认模型会有预测误差。
    2. 转化矛盾 (Convert Contradiction): 将 "预测误差 (Surprise)" 转化为 "主要学习动力 (Plasticity Control)"。
        - 如果我预判我会做对，且真的做对了 -> 矛盾小 -> 巩固模式 (Consolidation Mode)。
        - 如果我预判我会做对，结果做错了 -> 矛盾大 -> 重组模式 (Adaptation Mode)。
    3. 追求自治 (Pursue Autonomy): 整个调节过程在 Cell 内部闭环完成，无需外部手动调整超参。
    """
    def __init__(self, input_size, hidden_size):
        super(AutonomousCognitiveCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 1. 内部预测器 (Internal Predictor)
        # 输入: [Hidden, Question_Embedding] -> 输出: [1] (Logits)
        # 注意: 这里的 input_size 应该是单纯问题嵌入的维度，但在 GIKT 调用中，
        # 我们可能会传入变换后的 input_trans (融合了 Q+A)。
        # 为了结构严谨性，我们需要在 forward 中分离 Q 和 A，或者传入 Q_emb.
        # 鉴于 GIKT 现有架构，input_size 实际上是 emb_dim。
        # 我们假设传入的 'question_input' 是问题的特征。
        self.predictor = Linear(hidden_size + input_size, 1)
        
        # 2. 惊奇度编码器 (Surprise Encoder)
        self.surprise_encoder = Linear(1, hidden_size)

        # 3. 基础门控 (Basic Gates) - 改进版 GRU
        # 输入扩展: [Hidden + Question + Response + Surprise]
        # Gate Input Size = Hidden + Input(Q+A) + Surprise
        # 注意: 标准 input_data 已经是 Q+A 的融合/拼接
        gate_input_dim = hidden_size + input_size + hidden_size 
        
        self.linear_z = Linear(gate_input_dim, hidden_size) # Update Gate
        self.linear_r = Linear(gate_input_dim, hidden_size) # Reset Gate
        self.linear_h = Linear(gate_input_dim, hidden_size) # Candidate
        
    def forward(self, input_data, h_prev, question_input, response_val):
        """
        input_data: [batch, input_size] (可能是 Q+A 的融合特征，用于常规状态更新)
        h_prev: [batch, hidden_size]
        question_input: [batch, input_size] (仅问题的特征，用于无偏预测)
        response_val: [batch, 1] (真实的 0/1 结果，用于计算 Surprise)
        """
        # 1. 自治预测 (Autonomous Prediction)
        # 使用旧状态 + 问题 -> 预测结果
        # 这一步体现"结构": 在看到答案之前，先建立期望
        combined_predict = torch.cat((h_prev, question_input), dim=1)
        pred_logits = self.predictor(combined_predict)
        pred_prob = torch.sigmoid(pred_logits)
        
        # 2. 感知矛盾 (Perceive Contradiction)
        # 计算 Surprise
        # response_val 必须是 float 类型
        # 这里保留梯度回传，让 predictor 与主任务共同优化。
        surprise = torch.abs(response_val - pred_prob)
        surprise_vec = F.relu(self.surprise_encoder(surprise))
        
        # 3. 动态调节 (Dynamic Regulation)
        combined_gate = torch.cat((h_prev, input_data, surprise_vec), dim=1)
        
        z_t = torch.sigmoid(self.linear_z(combined_gate))
        r_t = torch.sigmoid(self.linear_r(combined_gate))
        
        combined_candidate = torch.cat((r_t * h_prev, input_data, surprise_vec), dim=1)
        h_tilde = torch.tanh(self.linear_h(combined_candidate))
        
        h_new = (1 - z_t) * h_prev + z_t * h_tilde
        
        return h_new, pred_prob, surprise, z_t

# @add_fzq 2025-12-25 -------------------------------------------
class CognitiveRNNCell(Module):
    def __init__(self, input_size, hidden_size):
        super(CognitiveRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 1. 保留门 (Retention Gate) - 基于艾宾浩斯遗忘曲线思想
        # 输入: [Hidden, Input, Interval] -> 输出: [Hidden]
        # 这里的 Input 包含了题目信息，所以保留门是"内容感知"的，能实现不同知识点不同的保留率
        self.linear_forget = Linear(hidden_size + input_size + 1, hidden_size)
        
        # 2. 学习门 (Learning Gate / Cognitive Gate) - 基于认知风格
        # 输入: [Hidden, Input, ResponseTime] -> 输出: [Hidden]
        self.linear_learn = Linear(hidden_size + input_size + 1, hidden_size)
        
        # 3. 候选状态 (Candidate State / Learning Gain)
        # 输入: [Hidden, Input] -> 输出: [Hidden]
        self.linear_candidate = Linear(hidden_size + input_size, hidden_size)

    def forward(self, input_data, h_prev, interval_time, response_time):
        # input_data: [batch, input_size] (包含题目Embedding + 作答Embedding)
        # h_prev: [batch, hidden_size]
        # interval_time: [batch, 1] (距离上次做题的时间，需归一化)
        # response_time: [batch, 1] (本次作答耗时，需归一化)
        
        # 拼接输入
        combined_forget = torch.cat((h_prev, input_data, interval_time), dim=1)
        combined_learn = torch.cat((h_prev, input_data, response_time), dim=1)
        combined_candidate = torch.cat((h_prev, input_data), dim=1)
        
        # 计算门控
        # 保留门 (Retention Gate): 范围 [0, 1]，值越小忘得越多 (对应 LSTM 的 forget gate)
        # 艾宾浩斯思想: interval 越大 -> retention 越小
        r_t = torch.sigmoid(self.linear_forget(combined_forget))
        
        # 学习门: 范围 [0, 1]，值越大通过的新知识越多
        # 认知风格思想: 冲动型答错 -> learn 越小
        l_t = torch.sigmoid(self.linear_learn(combined_learn))
        
        # 候选状态 (新知识增量)
        c_t = torch.tanh(self.linear_candidate(combined_candidate))
        
        # 状态更新公式
        # h_new = (保留门 * 旧状态) + (学习门 * 新知识)
        # h_new = r_t * h_prev + l_t * c_t # 原始方案：存在数值不稳定的风险（如果 r_t 和 l_t 都很大，状态会无限累积）
        # 和上面效果一致，进行归一化后上面公式：输入受控 + 增量受控 + 遗忘机制 ≈ 系统自然稳定
        # 但建议使用下面的方案，更加稳健
        h_new = torch.tanh(r_t * h_prev + l_t * c_t) # 修改方案：加入 tanh 激活函数，将状态限制在 [-1, 1] 之间，避免数值不稳定
        
        return h_new, h_new # 返回两次是为了兼容 LSTM 接口
# @add_fzq 2025-12-25 -------------------------------------------

class GIKT(Module):

    def __init__(self, num_question, num_skill, q_neighbors, s_neighbors, qs_table, agg_hops=3, emb_dim=100, dropout_linear=0.2, dropout_gnn=0.4, drop_edge_rate=0.0, feature_noise_scale=0.0, hard_recap=True, rank_k=10, pre_train=False, use_cognitive_model=False, cognitive_mode='autonomous', data_dir=None, agg_method='gcn', recap_source='hssi', use_pid=False, pid_mode='global', pid_ema_alpha=0.1, pid_lambda=1.0, pid_init_i=0.5, pid_init_d=0.1, guessing_prob_init=0.05, slipping_prob_init=0.02, use_4pl_irt=True):
        '''
        概述：这是一个名为GIKT的模型的初始化函数，用于设置模型的各个参数和层结构，包括问题数量、技能数量、邻居数量、聚合层数、嵌入维度、dropout率等，并定义了用于聚合、查询、键和权重计算的线性层。

            参数：
            num_question: 问题数量
            num_skill: 技能数量
            q_neighbors (numpy.ndarray): 问题邻居矩阵。形状为(num_question, q_neighbor_size)的二维数组，包含每个问题选取的邻居ID。
            s_neighbors (numpy.ndarray): 技能邻居矩阵。形状为(num_skill, s_neighbor_size)的二维数组，包含每个技能选取的邻居ID。
            qs_table: 问题-技能表
            agg_hops: 聚合层数，默认为3
            emb_dim: 嵌入维度，默认为100
            dropout_linear: 线性层/RNN层的 dropout 率
            dropout_gnn: GNN 聚合后的 dropout 率
            drop_edge_rate: [New] 图结构的随机丢边率 (DropEdge)
            feature_noise_scale: [New] 特征扰动强度 (0.0 表示关闭)
            hard_recap: 是否使用硬回顾，默认为True
            rank_k: 排序的k值，默认为10
            pre_train: 是否预训练，默认为False
            use_cognitive_model: 是否使用认知模型(CognitiveRNNCell)，默认为False
            cognitive_mode: 认知模型模式 ('classic' or 'autonomous'), 默认为 'autonomous'
            data_dir: 数据目录，用于加载预训练向量，默认为None
            agg_method: 聚合方法，默认为'gcn'
            recap_source: 回顾特征来源，默认为'hssi'
            use_pid: 是否使用PID控制器架构
            pid_mode: PID 模式 ('global' 或 'domain') 
            use_4pl_irt: 是否使用 4PL IRT 差异化建模特性
        '''
        super(GIKT, self).__init__()  # 调用父类的初始化方法
        self.model_name = "gikt" #  模型名称设置
        self.num_question = num_question #  问题数量设置
        self.num_skill = num_skill #  技能数量设置
        self.agg_hops = agg_hops #  聚合跳跃层数设置
        self.qs_table = qs_table #  问题-技能关系表设置
        self.emb_dim = emb_dim #  嵌入维度设置
        
        self.dropout_linear_val = dropout_linear
        self.dropout_gnn_val = dropout_gnn
        self.drop_edge_rate = drop_edge_rate # 丢边率
        
        # @add_fzq: Feature Perturbation Parameters
        self.feature_noise_scale = feature_noise_scale

        self.hard_recap = hard_recap #  困难回顾设置
        self.rank_k = rank_k #  排名参数k设置
        
        # @add_fzq: Ablation Support for Cognitive Model
        # 如果 cognitive_mode == 'none', 强制关闭 use_cognitive_model
        # 否则尊重 use_cognitive_model 传入的值
        # 如果调用方已经通过 cognitive_mode='none' 期望关闭，就应该覆盖旧参数
        if cognitive_mode == 'none':
            self.use_cognitive_model = False
        else:
            self.use_cognitive_model = use_cognitive_model
            
        self.cognitive_mode = cognitive_mode # 认知模型模式
        self.use_4pl_irt = use_4pl_irt # 是否使用 4PL IRT 特性
        
        self.agg_method = agg_method # 聚合方法
        self.recap_source = recap_source # 硬回顾特征来源
        self.use_pid = use_pid # 是否使用PID控制器架构
        self.pid_mode = pid_mode # PID模式
        self.pre_train = pre_train # 是否预训练
        # pid_num_domains will be set dynamically if mode is 'domain'

        # 将邻居表预加载到计算设备上，以便重复使用/缓存
        self.register_buffer('q_neighbors_t', torch.as_tensor(q_neighbors, dtype=torch.long, device=DEVICE))
        self.register_buffer('s_neighbors_t', torch.as_tensor(s_neighbors, dtype=torch.long, device=DEVICE))
        self.q_neighbor_size = self.q_neighbors_t.shape[1]
        self.s_neighbor_size = self.s_neighbors_t.shape[1]

        # 简化 qk_gat 相关初始化
        if self.agg_method in ['qk_gat', 'kk_gat'] and data_dir is not None:
            metadata_path = os.path.join(data_dir, 'metadata.json')
            q_features_path = None
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    q_feat_file = metadata.get('features', {}).get('q_features', {}).get('file')
                    if q_feat_file:
                        q_features_path = os.path.join(data_dir, q_feat_file)
            if q_features_path and os.path.exists(q_features_path):
                import numpy as np
                q_feats = np.load(q_features_path).astype(np.float32)
                self.q_feature_embedding = Embedding.from_pretrained(torch.from_numpy(q_feats), freeze=True)
                self.q_feat_dim = q_feats.shape[1]
                
                # ONLY INITIALIZE THESE IF qk_gat
                if self.agg_method == 'qk_gat':
                    self.gat_attn_layers = ModuleList([
                        Linear(emb_dim * 2 + self.q_feat_dim, 1) for _ in range(agg_hops)
                    ])
                    for layer in self.gat_attn_layers:
                        torch.nn.init.xavier_uniform_(layer.weight)
            else:
                if self.agg_method == 'qk_gat':
                    print(f"Warning: QK-GAT selected but q_features not found. Fallback to GCN.")
                    self.agg_method = 'gcn'
                else:
                    print(f"Warning: KK-GAT selected but q_features not found. Attributes will be ignored.")
                    self.q_feat_dim = 0

        if self.agg_method == 'kk_gat':
            # K-K 图的自注意力计算层
            self.kk_gat_attention = MultiheadAttention(embed_dim=emb_dim, num_heads=2, dropout=dropout_gnn, batch_first=True)
            # 融合层通道：支持问题嵌入 + 汇聚技能 + (可选)题目属性
            q_feat_dim = getattr(self, 'q_feat_dim', 0)
            self.kk_q_fusion = Linear(emb_dim * 2 + q_feat_dim, emb_dim)


        if pre_train:
            # 使用预训练之后的向量
            # @fix_fzq: 优化目录获取逻辑，优先使用传入的 data_dir
            if data_dir is None:
                from config import get_config
                dataset_name = os.environ.get('DATASET', 'assist09')
                config = get_config(dataset_name)
                data_dir = config.PROCESSED_DATA_DIR
            
            print(f"Loading pre-trained embeddings from: {data_dir}")
            _weight_q = torch.load(f=os.path.join(data_dir, 'q_embedding.pt'), map_location=DEVICE) 
            _weight_s = torch.load(f=os.path.join(data_dir, 's_embedding.pt'), map_location=DEVICE) 
            self.emb_table_question = Embedding(num_question, emb_dim, _weight=_weight_q) # 问题嵌入表
            self.emb_table_skill = Embedding(num_skill, emb_dim, _weight=_weight_s)   #  技能嵌入表
        else:
            # 不使用预训练，随机加载向量
            self.emb_table_question = Embedding(num_question, emb_dim) #  初始化问题ID的嵌入表
            self.emb_table_skill = Embedding(num_skill, emb_dim) #  初始化技能ID的嵌入表
        self.emb_table_response = Embedding(2, emb_dim) # 回答结果的嵌入表（0和1两种可能）

        # self.gru1 = GRUCell(emb_dim * 2, emb_dim) # 使用GRU网络
        # self.gru2 = GRUCell(emb_dim, emb_dim)
        if self.use_cognitive_model:
            # TF Alignment: Uses projected input (emb_dim)
            if self.cognitive_mode == 'autonomous':
                # @change_fzq: Switch to AutonomousCognitiveCell (Scheme 2)
                self.cognitive_cell = AutonomousCognitiveCell(input_size=emb_dim, hidden_size=emb_dim)
            else:
                # Classic Cognitive Model
                self.cognitive_cell = CognitiveRNNCell(input_size=emb_dim, hidden_size=emb_dim)
        else:
            # TF Alignment: Uses projected input (emb_dim)
            self.lstm_cell = LSTMCell(input_size=emb_dim, hidden_size=emb_dim) # 使用LSTM网络

        # TF Alignment: Always initialize input_trans_layer for projected input
        # 将 2*emb_dim 的拼接特征映射到 emb_dim
        # 参考 TF 源码: input_trans_embedding = dense(concat([feature_trans, input_answers]), hidden_size)
        self.input_trans_layer = Linear(emb_dim * 2, emb_dim)

        # ----------------------------------------------------
        # @add_fzq v5 optimization: Feature Transform Layer
        # TF Alignment: feature_trans_embedding = Relu(Dense(aggregate_embedding))
        # 即使在 PyTorch 中我们不直接用这个结果拼接，但为了特征空间的丰富度，我们可以加上这个非线性变换
        # ----------------------------------------------------
        self.feature_transform_layer = Linear(emb_dim, emb_dim)
            
        self.mlps4agg = ModuleList(Linear(emb_dim, emb_dim) for _ in range(agg_hops)) #  创建多个线性层用于聚合操作
        
        self.MLP_AGG_last = Linear(emb_dim, emb_dim) #  最后一个聚合操作的线性层
        self.dropout_lstm = Dropout(self.dropout_linear_val) #  LSTM网络的dropout层
        # self.dropout_gru = Dropout(dropout[0]) #  GRU网络的dropout层
        self.dropout_gnn = Dropout(self.dropout_gnn_val) #  GNN网络的dropout层
        self.MLP_query = Linear(emb_dim, emb_dim) #  查询向量转换的线性层
        self.MLP_key = Linear(emb_dim, emb_dim) #  键向量转换的线性层
        # 公式10中的W
        self.MLP_W = Linear(2 * emb_dim, 1)

        # @add_fzq: Differential GIKT Architecture (Path 2)
        # 1. 静态难度基准 (Common Mode): 层次化难度建模
        if self.use_4pl_irt:
            # (1) 题目级难度偏置: 捕捉题目特有的细微难度差异
            self.difficulty_bias = Embedding(num_question, 1)
            
            # (2) 知识点级难度偏置 (Hierarchical Step 2): 
            # 用于抑制题目稀疏性带来的噪声，为关联同一知识点的题目提供稳健的共模难度基准
            self.skill_difficulty_bias = Embedding(num_skill, 1)
            
            # 2. 区分度 (Discrimination): a_q (Step 3: 4PL 升级为题目级参数)
            # 初始化为 0，后期配合 softplus(x) + 1.0 得到约为 1.69 的初始区分度
            self.discrimination_bias = Embedding(num_question, 1)
            
            # 3. 猜测率 (Guessing) c_q 与 失误率 (Slipping) d_q: 噪声吸收器
            # 使用较大的负数初始化，使得初始状态下 c_q 约为 0.05, d_q 约为 0.02
            self.guessing_bias = Embedding(num_question, 1)
            self.slipping_bias = Embedding(num_question, 1)

        # @add_fzq: PID-GIKT Controller Parameters
        if self.use_pid:
            # PID Setup
            if self.pid_mode == 'domain':
                # Determine data_dir to load skill_domain_map
                if data_dir is None:
                    from config import get_config
                    dataset_name = os.environ.get('DATASET', 'assist09')
                    config = get_config(dataset_name)
                    data_dir = config.PROCESSED_DATA_DIR
                
                # 尝试从 metadata.json 获取领域映射信息
                metadata_path = os.path.join(data_dir, 'metadata.json')
                skill_domain_map = None
                self.pid_num_domains = 1
                
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        
                        self.pid_num_domains = metadata.get('metrics', {}).get('n_domain', 1)
                        map_filename = metadata.get('mappings', {}).get('skill_domain_map', 'skill_domain_map.json')
                        json_path = os.path.join(data_dir, map_filename)
                        
                        if os.path.exists(json_path):
                            print(f"Loading Skill-Domain Map for PID from JSON: {json_path}")
                            with open(json_path, 'r', encoding='utf-8') as jf:
                                raw = json.load(jf)
                            try:
                                keys = sorted(raw.keys(), key=lambda x: int(x))
                            except Exception:
                                keys = list(raw.keys())
                            skill_domain_map = np.array([int(raw[k]) for k in keys], dtype=int)
                            print(f"Detected {self.pid_num_domains} PID domains from metadata.")
                        else:
                            print(f"Error: skill_domain_map file {json_path} not found. Fallback to 1 domain.")
                    except Exception as e:
                        print(f"Error reading metadata for PID domains: {e}. Fallback to 1 domain.")
                else:
                    print(f"Error: metadata.json not found in {data_dir}. Fallback to 1 domain.")
            else:
                self.pid_num_domains = 1

            # Purpose: Introduce "Momentum" (Integral) and "Acceleration" (Derivative) control for student ability
            # ema_alpha: Decay rate for Integral term (0.1 means long-term memory dominant)
            # pid_lambda: Scaling factor for Tanh Limiter on Derivative term
            self.ema_alpha = pid_ema_alpha
            self.pid_lambda = pid_lambda
            
            # PID Weights (Learnable): Initialize to near-zero to prevent initial distribution shock
            # w_pid_i: Weight for Integral term (Streak tracking)
            # w_pid_d: Weight for Derivative term (Trend tracking)
            # @mod_fzq: Increased initialization from 1e-2 to 0.5 to make PID effect visible
            # @opt_fzq: Support per-domain learnable weights
            if self.pid_mode == 'domain':
                # Vectorized weights: each domain has its own I/D sensitivity
                # Initialize to 0.5 and 0.1
                self.w_pid_i = torch.nn.Parameter(torch.full((self.pid_num_domains,), pid_init_i))
                self.w_pid_d = torch.nn.Parameter(torch.full((self.pid_num_domains,), pid_init_d))
            else:
                # Global weights: scalar
                self.w_pid_i = torch.nn.Parameter(torch.tensor(pid_init_i))
                self.w_pid_d = torch.nn.Parameter(torch.tensor(pid_init_d))

            # @add_fzq: Domain-Level PID Initialization (Pre-compute mask)
            if self.pid_mode == 'domain' and 'skill_domain_map' in locals() and skill_domain_map is not None:
                # 1. Create mapping tensor
                s2d_tensor = torch.tensor(skill_domain_map, device=DEVICE, dtype=torch.long) # [S]
                
                # 2. Map Q -> S -> D
                print(f"Building Question-Domain Mask for {num_question} questions...")
                
                # Convert Skill-to-Domain mapping to One-hot matrix [S, D]
                # s2d_tensor is [S] containing domain IDs
                s_domain_one_hot = F.one_hot(s2d_tensor.long(), num_classes=self.pid_num_domains).float()
                
                # Compute Q-D mask via matrix multiplication: [Q, S] @ [S, D] -> [Q, D]
                # In this version, self.qs_table is the binary [Q, S] matrix.
                q_domain_counts = torch.matmul(self.qs_table.float(), s_domain_one_hot)
                
                # Binarize: [Q, D] where 1 if question is associated with any skill in that domain
                self.q_domain_mask = (q_domain_counts > 0).float()
                # Register as buffer to save with model
                self.register_buffer('q_domain_mask_buffer', self.q_domain_mask)
                print(f"Question-Domain Mask Built. Shape: {self.q_domain_mask.shape}")
            elif self.pid_mode == 'domain':
                print(f"Error: skill_domain_map not found or loaded. Falling back to Global PID.")
                self.pid_num_domains = 1
                self.pid_mode = 'global'
                self.w_pid_i = torch.nn.Parameter(torch.tensor(pid_init_i))
                self.w_pid_d = torch.nn.Parameter(torch.tensor(pid_init_d))
        self.reset_parameters()
        
        # @fix_fzq: Re-apply 4PL initialization (must be done AFTER reset_parameters)
        # Convert Probability to Logits (Bias): x = ln(p / (1-p))
        guessing_bias = math.log(guessing_prob_init / (1 - guessing_prob_init)) if 0 < guessing_prob_init < 1 else -3.0
        slipping_bias = math.log(slipping_prob_init / (1 - slipping_prob_init)) if 0 < slipping_prob_init < 1 else -4.0

        if hasattr(self, 'difficulty_bias'):
            torch.nn.init.constant_(self.difficulty_bias.weight, 0.0)
        if hasattr(self, 'skill_difficulty_bias'):
            torch.nn.init.constant_(self.skill_difficulty_bias.weight, 0.0)
        if hasattr(self, 'discrimination_bias'):
            torch.nn.init.constant_(self.discrimination_bias.weight, 0.0)
        if hasattr(self, 'guessing_bias'):
            torch.nn.init.constant_(self.guessing_bias.weight, guessing_bias)
        if hasattr(self, 'slipping_bias'):
            torch.nn.init.constant_(self.slipping_bias.weight, slipping_bias)
        # # 可学习的融合参数 beta，初始化为 0.1
        # self.beta = torch.nn.Parameter(torch.tensor(0.1))

    def forward(self, question, response, mask, interval_time=None, response_time=None):
        # question: [batch_size, seq_len]
        # response: [batch_size, 1]
        # mask: [batch_size, seq_len] 和question一样的形状, 表示在question中哪些索引是真正的数据(1), 哪些是补零的数据(0)
        # interval_time: [batch_size, seq_len] (可选)
        # response_time: [batch_size, seq_len] (可选)
        # 每一个在forward中new出来的tensor都要.to(DEVICE)
        batch_size, seq_len = question.shape # batch_size表示多少个用户, seq_len表示每个用户最多回答了多少个问题
        q_neighbor_size, s_neighbor_size = self.q_neighbor_size, self.s_neighbor_size
        h1_pre = torch.nn.init.xavier_uniform_(torch.zeros(self.emb_dim, device=DEVICE).repeat(batch_size, 1)) #  使用Xavier初始化方法创建两个全零的嵌入向量，并重复batch_size次
        h2_pre = torch.nn.init.xavier_uniform_(torch.zeros(self.emb_dim, device=DEVICE).repeat(batch_size, 1))
        # state_history = torch.zeros(batch_size, seq_len, self.emb_dim, device=DEVICE) #  初始化状态历史记录和预测结果张量
        y_hat = torch.zeros(batch_size, seq_len, device=DEVICE)

        # @add_fzq: PID-GIKT State Initialization
        # pid_ema: Tracks the running average of correctness (Integral Term)
        # Initialized to 0.5 (Neutral state)
        if self.use_pid:
            if self.pid_mode == 'domain':
                # Vectorized State: [Batch, Num_Domains]
                pid_ema = torch.full((batch_size, self.pid_num_domains), 0.5, device=DEVICE)
                pid_diff = torch.zeros((batch_size, self.pid_num_domains), device=DEVICE)
            else:
                # Scalar State: [Batch]
                pid_ema = torch.full((batch_size,), 0.5, device=DEVICE)
                pid_diff = torch.zeros((batch_size,), device=DEVICE)
        else:
            pid_ema = None
            pid_diff = None

        # @add_fzq: PID-GIKT State Initialization
        # pid_ema: Tracks the running average of correctness (Integral Term)
        # Initialized to 0.5 (Neutral state)
        if self.use_pid:
            if self.pid_mode == 'domain':
                # Vectorized State: [Batch, Num_Domains]
                pid_ema = torch.full((batch_size, self.pid_num_domains), 0.5, device=DEVICE)
                pid_diff = torch.zeros((batch_size, self.pid_num_domains), device=DEVICE)
            else:
                # Scalar State: [Batch]
                pid_ema = torch.full((batch_size,), 0.5, device=DEVICE)
                pid_diff = torch.zeros((batch_size,), device=DEVICE)
        else:
            pid_ema = None
            pid_diff = None

        # [Level 2 优化] 批次级 GNN 预计算 (第一部分：准备工作)
        # 仅在需要时预计算唯一问题的聚合逻辑（主要用于下一个问题预测）
        # 但这里我们也需要当前问题的聚合用于 LSTM 输入。
        # 所以我们预计算本批次全量唯一问题。
        
        # 1. 识别唯一问题（如果可能排除 padding 0，但保持逻辑简单）
        # unique_questions = torch.unique(question)
        # 为避免 padding 0 引发索引错误（通常 0 在 embedding 中是有效索引）
        
        # 2. 分配本批次查找表
        # 我们不能轻易为所有问题建立密集表（太大）。
        # 但我们可以将结果 scatter 回 [Batch, Seq] 或使用临时稀疏映射。
        # 鉴于 Batch 较小，我们可以仅计算 unique_questions 并使用索引映射。
        
        # 构造仅针对 UNIQUE 问题的邻居缓存
        # Level 2 改动：使用 torch.unique 两次以获取 inverse_indices
        # 展平 question 以进行映射
        flat_questions = question.view(-1)
        unique_q, inverse_ind = torch.unique(flat_questions, return_inverse=True)

        # 构造 unique_q 的邻居缓存
        unique_neighbors_cache = [unique_q]
        cache_curr = unique_q
        for hop in range(self.agg_hops):
            # 使用缓存的邻居表
            if hop % 2 == 0:
                cache_next = self.q_neighbors_t[cache_curr]
            else:
                cache_next = self.s_neighbors_t[cache_curr]
            unique_neighbors_cache.append(cache_next)
            cache_curr = cache_next

        # 3. 针对唯一问题的批次聚合
        # 提取 unique 结构的嵌入
        # unique_neighbors_cache 结构: List of [Num_Unique_Q, Neighbors]
        emb_unique_neighbors = [] 
        for i, nodes in enumerate(unique_neighbors_cache):
            if i % 2 == 0: 
                # 问题节点 Embedding
                emb = self.emb_table_question(nodes)
                # @add_fzq: Feature Perturbation (特征扰动) for Question
                if self.training and self.feature_noise_scale > 1e-6:
                    emb = emb + torch.randn_like(emb) * self.feature_noise_scale
                emb_unique_neighbors.append(emb)
            else:
                # 技能节点 Embedding
                emb = self.emb_table_skill(nodes)
                # @add_fzq: Feature Perturbation (特征扰动) for Skill
                if self.training and self.feature_noise_scale > 1e-6:
                    emb = emb + torch.randn_like(emb) * self.feature_noise_scale
                emb_unique_neighbors.append(emb)
        
        if self.agg_method == 'kk_gat':
            # --- kk_gat decoupled approach ---
            all_s = torch.arange(self.num_skill, device=DEVICE)
            emb_s = self.emb_table_skill(all_s) # [Num_K, Emb]
            if self.training and self.feature_noise_scale > 1e-6:
                emb_s = emb_s + torch.randn_like(emb_s) * self.feature_noise_scale
                
            # K-K multihead attention (expects [batch=1, seq_len, emb_dim])
            emb_s_batch = emb_s.unsqueeze(0) 
            agg_s_batch, _ = self.kk_gat_attention(emb_s_batch, emb_s_batch, emb_s_batch)
            agg_s = agg_s_batch.squeeze(0) # [Num_K, Emb]
            
            # Average pooling for unique questions
            qs_multi_hot = self.qs_table[unique_q].float() # [Num_Unique_Q, Num_K]
            pooled_skills = torch.matmul(qs_multi_hot, agg_s)
            skill_counts = qs_multi_hot.sum(dim=-1, keepdim=True).clamp(min=1.0)
            pooled_skills = pooled_skills / skill_counts
            
            # Question embeddings for unique questions
            q_emb = self.emb_table_question(unique_q) # [Num_Unique_Q, Emb]
            if self.training and self.feature_noise_scale > 1e-6:
                q_emb = q_emb + torch.randn_like(q_emb) * self.feature_noise_scale
            
            # 动态融合题目属性（如果可用）
            if hasattr(self, 'q_feature_embedding'):
                q_attrs = self.q_feature_embedding(unique_q) # [Num_Unique_Q, Q_Feat_Dim]
                fusion_input = torch.cat([q_emb, pooled_skills, q_attrs], dim=-1)
            else:
                fusion_input = torch.cat([q_emb, pooled_skills], dim=-1)
                
            agg_unique_q = torch.tanh(self.kk_q_fusion(fusion_input))
            
            # Pack agg_list_unique to satisfy downstream needs
            agg_unique_s_neighbors = agg_s[unique_neighbors_cache[1]] 
            agg_list_unique = [agg_unique_q, agg_unique_s_neighbors]
        else:
            # 对所有唯一问题执行一次性聚合
            agg_unique_q, agg_list_unique = self.aggregate(emb_unique_neighbors, unique_neighbors_cache)
        
        # 4. 创建查找机制
        # [Level 2] 直接映射回 [Batch, Seq, Emb] 形状
        # output = u_agg_q[inverse_ind]
        precomputed_gnn_features = agg_unique_q[inverse_ind].view(batch_size, seq_len, self.emb_dim)
        
        # TF Alignment: Always precompute gnn_skills for target aggregation
        # u_agg_list[1] is [Num_Unique_Q, Neighbor_Size, Emb]
        # agg_list_unique[1] 对应第一层聚合后的 Skill 邻居特征
        precomputed_gnn_skills = agg_list_unique[1][inverse_ind].view(batch_size, seq_len, -1, self.emb_dim)

        # [Level 2 优化] 投影历史缓存 (KV Cache)
        # 使用 List 避免 autograd 期间的 In-place 操作错误
        projected_keys_list = []
        # [Level 2 Fix] 历史状态列表 (Value Cache)
        # 同样使用 List 避免 state_history[:, t] = ... 的 In-place 错误
        state_history_list = []

        for t in range(seq_len - 1): # 第t时刻
            question_t = question[:, t] #  取出所有学生在第 t 个时间步所做的题目ID
            response_t = response[:, t] #  取出所有学生在第 t 个时间步所做的题目ID的回答结果
            
            mask_t = torch.eq(mask[:, t], 1) #  创建一个布尔掩码
            emb_response_t = self.emb_table_response(response_t) 
            
            # [Level 2 变更] 检索预计算的 GNN 特征
            # 以前: 掩码选择 -> 聚合 -> 填充
            # 现在: 仅切片。
            # 注意：为了正确性，我们仍然尊崇 `mask_t`，但预计算处理了一切。
            # 我们只是取值。对于填充条目，值来自 aggregate(ID=0)，这是可以接受的/被忽略的。
            emb0_question_t = precomputed_gnn_features[:, t]
            
            # @fix_fzq: AMP FP16 不匹配修复 (保留)
            dtype = emb0_question_t.dtype
            
            # 我们直接使用 emb0_question_t，但原始代码对 ~mask_t 有降级处理
            # "emb_question_t[~mask_t] = self.emb_table_question(question_t[~mask_t])"
            # 由于我们的预计算对所有问题（包括 ID 0 的填充）运行聚合，
            # ID 0 的结果是 aggregate(0)。如果我们想要 ID 0 的原始嵌入，我们可以混合使用。
            # 然而，对于填充，这并不重要。
            # 为了与原始逻辑 "emb_node_neighbor" vs "emb_table_question" 的正确性保持一致，
            # GIKT 对有效问题应用 GNN。
            # 让我们信任预计算的值。
            emb_question_t = emb0_question_t

            
            # ----------------------------------------------------
            # @add_fzq: PID 状态更新 (积分 & 微分)
            # ----------------------------------------------------
            if self.use_pid:
                # 计算误差 (Response - EMA)
                # 注意: response_t 是 0/1. ema 是连续的 [0,1].
                
                if self.pid_mode == 'domain':
                    # 向量化领域更新
                    # 1. 获取当前问题的领域掩码 [Batch, Num_Domains]
                    #直接使用 question_t 索引查找
                    # self.q_domain_mask 是 [Num_Q, Num_Domains]
                    # 我们需要直接访问缓冲区
                    mask_domains = self.q_domain_mask_buffer[question_t] # [Batch, D]
                    
                    # 2. 仅更新相关领域的 EMA
                    # 当前响应广播到 [Batch, 1]
                    curr_resp_exp = response_t.float().unsqueeze(1) # [Batch, 1]
                    
                    # 从当前响应导出的目标 EMA，适用于所有领域（但我们只应用于被掩盖的领域）
                    # New_EMA = Alpha * Resp + (1-Alpha) * Old_EMA
                    # 但我们只想更改 mask_domains == 1 的位置
                    
                    pid_ema_new = self.ema_alpha * curr_resp_exp + (1.0 - self.ema_alpha) * pid_ema
                    
                    # 应用领域掩码：如果领域相关且批次项有效 (mask_t) 则更新
                    # mask_t: [Batch] -> [Batch, 1]
                    valid_update_mask = mask_domains * mask_t.unsqueeze(1).float() # [Batch, D] (1 if update, 0 if keep)
                    
                    # 软更新 (向量化形式):
                    # New = Mask * Calculated + (1-Mask) * Old
                    pid_ema_updated = valid_update_mask * pid_ema_new + (1.0 - valid_update_mask) * pid_ema
                    
                    # 3. 计算微分
                    pid_diff_new = pid_ema_updated - pid_ema
                    
                    # 更新状态
                    pid_ema = pid_ema_updated
                    pid_diff = pid_diff_new
                    
                else:
                    # 标量全局更新 (原始逻辑)
                    # 基于当前 response_t 更新 EMA。
                    # 掩码检查：仅更新有效交互。
                    current_response_float = response_t.float()
                    pid_ema_new = self.ema_alpha * current_response_float + (1.0 - self.ema_alpha) * pid_ema
                    
                    # 计算微分 (状态变化)
                    pid_diff_new = pid_ema_new - pid_ema
                    
                    # 使用掩码保护更新状态 (防止填充扭曲状态)
                    pid_ema = torch.where(mask_t, pid_ema_new, pid_ema)
                    pid_diff = torch.where(mask_t, pid_diff_new, torch.zeros_like(pid_diff))
            
            # ----------------------------------------------------
            # @add_fzq v5 optimization: Feature Transform
            # TF Alignment: Apply transformation to question embedding before concat
            # ----------------------------------------------------
            emb_question_trans = torch.relu(self.feature_transform_layer(emb_question_t))

            # LSTM/GRU更新知识状态
            # 使用变换后的 Question Embedding 进行拼接
            lstm_input = torch.cat((emb_question_trans, emb_response_t), dim=1) # [batch_size, emb_dim * 2]
            
            # --- Prepare Recap Feature (TF Alignment) ---
            # 使用 Input (经过线性变换) 作为 History State 或 LSTM Input
            # Align with TF: input_trans_embedding (Dense layer, Linear activation)
            # Removed torch.tanh to match TF default
            recap_feature = self.input_trans_layer(lstm_input) 
            # TF code snippet shows: input_trans_embedding = tf.reshape(tf.layers.dense(input_fa_embedding, hidden_size), ...) 
            # Default activation of dense is None (Linear). But RNN usually operates on Tanh/ReLU. 
            # Model GIKT aggregate uses Tanh. Let's assume Tanh for consistency with state.
            
            # Determine LSTM Input (TF Alignment)
            # LSTM 使用投影特征 (100 dim)
            # 注意: `recap_feature` 在上方计算为 input_trans_layer(lstm_input)
            lstm_cell_input = recap_feature

            if self.use_cognitive_model:
                # Mode 1: Autonomous Cognitive Cell (Scheme 2)
                if self.cognitive_mode == 'autonomous':
                    # We need to extract the raw answer correctness for the internal error signal
                    # response_t is [batch_size], we need [batch_size, 1] float
                    curr_response_val = response_t.float().unsqueeze(1)
                    
                    if 'emb_question_trans' not in locals():
                        # Just in case (though it should be defined above)
                        emb_question_trans = torch.relu(self.feature_transform_layer(emb_question_t))

                    # Invoke AutonomousCognitiveCell
                    # Arguments: (input_data_fused, h_prev, question_emb, response_val)
                    h_new, pred_prob, surprise, z_t = self.cognitive_cell(
                        lstm_cell_input, 
                        h2_pre, 
                        emb_question_trans, 
                        curr_response_val
                    )
                # Mode 2: Classic Cognitive Cell (Ebbinghaus + Style)
                else:
                    if interval_time is None or response_time is None:
                        raise ValueError("Classic CognitiveRNNCell requires interval_time and response_time")
                    
                    # 获取当前时刻的时间特征 [batch_size, 1]
                    curr_interval = interval_time[:, t].unsqueeze(1)
                    curr_response = response_time[:, t].unsqueeze(1)
                    
                    # h2_pre 作为上一个时刻的 hidden state
                    # Cognitive Model 使用包含非线性变换特征的 lstm_input (宽输入)
                    # Arguments: (input_data_fused, h_prev, interval, response_time)
                    h_new, _ = self.cognitive_cell(lstm_cell_input, h2_pre, curr_interval, curr_response)
                
                # Common Output handling for Cognitive Models
                lstm_output = self.dropout_lstm(h_new) if self.training else h_new
            else:
                lstm_cell_output = self.lstm_cell(lstm_cell_input)[0] # [batch_size, emb_dim]
                lstm_output = self.dropout_lstm(lstm_cell_output) if self.training else lstm_cell_output
                # @fix_fzq: Maintain dropout for history saving
                # If we use `lstm_cell_output` (raw) for history, it mismatches what the network sees in training.
                # If we use `lstm_output` (dropped) for history, then training history is sparse, but test history is dense.
                # However, Standard Dropout scales by 1/(1-p). So expected value is maintained.
                # The issue with HSSI is high variance.
                # Let's trust `lstm_output` as it aligns with typical RNN stacking.
                
            # 找t+1时刻的[习题]以及[其对应的知识点]
            q_next = question[:, t + 1] # [batch_size, ]
            skills_related = self.qs_table[q_next] # [batch_size, num_skill]
            
            # @opt_fzq 2026-02-04: 批量化技能嵌入获取 - 移除 Python for 循环
            skill_mask = skills_related.bool()  # [batch_size, num_skill]
            max_num_skill = skill_mask.sum(dim=1).max().item()  # 找到最大技能数

            # 将习题和对应知识点embedding拼接起来
            # --- 逻辑分支修复：仅在 hsei 模式下启用 Target 聚合，hssi 模式保持原样 ---
            use_legacy_concat = True # 使用“旧版连接”功能
            qs_concat = None

            if self.recap_source == 'hsei':
                # [Case: HSEI] 需要特征对齐，执行图聚合
                # [Level 2 优化] 批次级 GNN 预计算 (第二部分：使用)
                # 不再重新聚合，而是从预计算张量中获取
                # precomputed_gnn_features: [Batch, Seq, Emb] -> At t+1
                emb_q_next = precomputed_gnn_features[:, t+1]
                
                # @fix_fzq: TF Alignment - Target Transform
                # Apply same transform (ReLU) to target question as applied to input question
                # GNN Output is Tanh, TF expects ReLU for dot product with Hidden State
                emb_q_next = torch.relu(self.feature_transform_layer(emb_q_next))

                # TF Alignment: Always use precomputed skills for target aggregation
                # 提取 Agg List (使用预计算的 skills)
                emb_skills_next_batch = precomputed_gnn_skills[:, t+1]
                # @fix_fzq: 对齐 TF - 技能嵌入也应通过同样的变换层，确保 qs_concat 都在同一特征空间
                emb_skills_next_batch = torch.relu(self.feature_transform_layer(emb_skills_next_batch))
                qs_concat = torch.cat((emb_q_next.unsqueeze(1), emb_skills_next_batch), dim=1)
                use_legacy_concat = False
                
            if use_legacy_concat:
                # @fix_fzq: AMP Fix - qs_concat 必须与 emb_q_next (FP16/Full) 保持相同类型
                dtype = emb_q_next.dtype if 'emb_q_next' in locals() else torch.float32
                
                # Check emb_q_next existence first (case HSSI)
                if not 'emb_q_next' in locals():
                    emb_q_next = self.emb_table_question(q_next) 
                    dtype = emb_q_next.dtype
                
                # @opt_fzq 2026-02-04: 批量化 qs_concat 填充 - 优化循环结
                # 使用 q_next 的 device 避免 CPU/GPU 不一致
                qs_concat = torch.zeros(batch_size, max_num_skill + 1, self.emb_dim, device=q_next.device, dtype=dtype)
                # 第一列始终是问题嵌入 [batch_size, emb_dim]
                qs_concat[:, 0, :] = emb_q_next
                
                # 批量获取技能嵌入 (Level 1 优化: 全向量化)
                # 使用 Masked Scatter 替代 Python Loop
                active_indices = torch.nonzero(skill_mask, as_tuple=True) # (batch_rows, skill_cols)
                batch_rows, skill_cols = active_indices[0], active_indices[1]
                
                if len(batch_rows) > 0:
                    # rank: 0基索引（问题嵌入占用了索引0，所以从1开始，cumsum默认从1开始）
                    # 我们使用之前定义的 skill_mask
                    # ranks = 1, 2, ...
                    ranks = torch.cumsum(skill_mask.long(), dim=1)[batch_rows, skill_cols]
                    
                    # 过滤范围 (针对 max_num_skill 的健全性检查)
                    valid_mask = ranks <= max_num_skill
                    
                    if valid_mask.any():
                        valid_b = batch_rows[valid_mask]
                        valid_s = skill_cols[valid_mask]
                        valid_r = ranks[valid_mask]
                        
                        # 向量化赋值
                        qs_concat[valid_b, valid_r, :] = self.emb_table_skill(valid_s).to(dtype)

            ####################################上述代码解释#####################################
            # qs_concat 最终变为 [batch_size, max_num_skill + 1, emb_dim] 的张量
            # 示例如下:
            # # 两个问题的嵌入向量，每个都是4维
            # emb_q_next = [
            #     [0.1, 0.2, 0.3, 0.4],  # 问题1的嵌入
            #     [0.5, 0.6, 0.7, 0.8]   # 问题2的嵌入
            # ]

            # # 两个问题对应的技能嵌入列表
            # skills_related_list = [
            #     # 问题1关联2个技能
            #     torch.tensor([[0.11, 0.12, 0.13, 0.14],  # 技能1
            #                 [0.15, 0.16, 0.17, 0.18]]), # 技能2
            #     # 问题2关联1个技能, pid_data=(pid_ema, pid_diff)
            #     torch.tensor([[0.21, 0.22, 0.23, 0.24]])  # 技能1
            # ]
            # # 将问题嵌入和技能嵌入拼接
            # qs_concat[0] = 
            # [[0.1, 0.2, 0.3, 0.4],  # 问题嵌入
            # [0.11, 0.12, 0.13, 0.14],  # 技能1
            # [0.15, 0.16, 0.17, 0.18],  # 技能2
            # [0.0, 0.0, 0.0, 0.0]]  # 零填充
            # 目的：
            # 1. 将每个问题与其相关的技能嵌入组合在一起
            # 2. 保持所有序列的长度一致（通过零填充）
            # 3. 形成一个批处理友好的张量格式，便于后续的神经网络处理
            ####################################################################################

            # 第一个问题, 无需寻找历史问题, 直接预测
            if t == 0:
                # @fix_fzq: 由于现在网络统一返回 Logits，预测概率 0.5 对应的 Logit 是 0.0
                y_hat[:, 0] = 0.0 # 概率0.5对应的Logit为0
                pid_data = (pid_ema, pid_diff) if self.use_pid else None
                y_hat[:, 1] = self.predict(qs_concat, torch.unsqueeze(lstm_output, dim=1), q_target=q_next, pid_data=pid_data)
                
                # [Level 2 Fix] 列表对齐
                # 为 t=0 附加零状态，以保持与 state_history 原始逻辑一致（index 0 为空）
                # 注意：如果我们不想让模型注意 t=0 (因为它是空的)，Key 应该是某种不会触发注意力的值。
                # 使用 Zero State 生成 Key (Tanh(MLP(0))) 通常会产生较小的分数，这与 Value=0 配合较好。
                zero_state = torch.zeros(batch_size, self.emb_dim, device=q_next.device)
                projected_keys_list.append(torch.tanh(self.MLP_query(zero_state)))
                state_history_list.append(zero_state)
                
                continue
            # recap硬选择历史问题
            if self.hard_recap:
                # history_mask: [Batch, t]. True=Retrieved. 
                # (Level 1: Returns Boolean Mask from Vectorized Method)
                history_mask = self.recap_hard(q_next, question[:, 0:t]) 
                
                # Determine "neighbor" source (retrieved items)
                # neighbor_source 是通过 state_history 隐式使用的，而 state_history 是在前几步中更新的
                
                # Level 1 优化: 带 Top-K 裁剪的向量化 Gather
                # 目的: 创建 [Batch, 1+K, Dim] 张量以消除循环
                
                # 1. 排序以将所有有效值推向左侧（虚拟 "pad_sequence"）
                # 将布尔值转换为整数进行排序
                sorted_mask, sorted_indices = torch.sort(history_mask.int(), dim=1, descending=True)
                
                # 2. 确定有效宽度 K (本批次的最大检索计数)
                k_active = sorted_mask.sum(dim=1).max().item()
                
                # 3. 基础状态：始终包含 current_state
                # @fix_fzq: Reverted HSSI alignment - now handled by target transform
                # lstm_output: [Batch, Dim] -> [Batch, 1, Dim]
                curr_state_expanded = lstm_output.unsqueeze(1) 

                if k_active == 0:
                    current_history_state = curr_state_expanded
                    # @fix_fzq: 必须在此分支为缓存分配正确形状的张量，避免 Locals() 遗留导致形状错误
                    cached_keys_tensor = torch.tanh(self.MLP_query(curr_state_expanded))
                    neighbor_source = recap_feature
                else:
                    # 4. 聚集历史记录
                    # 裁剪到 k_active 以减少计算
                    gather_cols = sorted_indices[:, :k_active] # [B, k_active] 时间步索引
                    valid_mask = sorted_mask[:, :k_active].unsqueeze(-1) # [B, k_active, 1] 有效位
                    
                    # [Level 2 优化] 投影键值 Gather
                    # 以前：从 state_history 聚集 (Raw)
                    # 现在：从 projected_keys_list 聚集 (即时转换为张量)
                    
                    # 动态构建 History Tensor 避免 In-place 错误
                    # state_history_list 长度为 t (indices 0..t-1)
                    # gather_cols 索引范围 [0, t-1]
                    temp_history_tensor = torch.stack(state_history_list, dim=1) # [Batch, t, Dim]
                    
                    # 从 temp_history_tensor 聚集原始值
                    gather_indices = gather_cols.unsqueeze(-1).expand(-1, -1, self.emb_dim)
                    selected_history = torch.gather(temp_history_tensor, 1, gather_indices) * valid_mask
                    current_history_state = torch.cat((curr_state_expanded, selected_history), dim=1)
                    
                    # 从 projected_keys_list 聚集投影值
                    # 检查列表是否为空
                    if len(projected_keys_list) > 0:
                        # 堆叠为 [Batch, Len_List, Dim]
                        # 注意：projected_keys_list 包含形状为 [Batch, Dim] 的张量
                        # 堆叠维度 1 -> [Batch, Len_List, Dim]
                        temp_keys_buffer = torch.stack(projected_keys_list, dim=1)
                        
                        # 是否需要填充 temp_buffer 以匹 state_history 大小？
                        # 不需要，gather_indices 严格位于 [0, t-1] 范围内（概念上）
                        # state_history 的大小为 seq_len，但我们只填充到了 t。
                        # projected_keys_list 的长度为 t。索引应小于 t。
                        # 然而，`recap_hard` 返回的索引是序列中的全局索引？
                        # `recap_hard` 返回 `time_select`。
                        # 如果 time_select 引用索引 T，但我们在列表中只有 0..t-1。
                        # 等等，`recap_hard` 接受 `question[:, 0:t]`。所以索引在 [0, t-1] 中。
                        # 所以大小为 t 的 temp_keys_buffer 就足够了。
                        
                        selected_keys = torch.gather(temp_keys_buffer, 1, gather_indices) * valid_mask
                    else:
                        # 降级处理：如果 t=0 或历史记录为空（尽管 hard_recap 通常 t>0）
                        # 创建正确形状的虚拟张量 [Batch, K_active, Dim]
                        selected_keys = torch.zeros(batch_size, k_active, self.emb_dim, device=DEVICE)

                    curr_key = torch.tanh(self.MLP_query(curr_state_expanded))
                    cached_keys_tensor = torch.cat((curr_key, selected_keys), dim=1)
                    # 修复：硬回顾分支补充 neighbor_source 赋值，保持与软回顾一致
                    neighbor_source = recap_feature
                    
            else: # 软选择
                # 确定 "neighbor" 来源 (retrieved items)
                if self.recap_source == 'hsei':
                    neighbor_source = recap_feature
                    # @fix_fzq: Symmetric Current State for HSEI
                    # If history is Input Projection, Current should also be Input Projection?
                    # But TF uses LSTM output for current state in Softmax logic?
                    # Let's keep HSEI as is (it works well).
                    current_state = lstm_output.unsqueeze(dim=1)
                else:
                    # HSSI (Hidden State Selection)
                    neighbor_source = lstm_output 
                    current_state = lstm_output.unsqueeze(dim=1)

                # 软选择更新：准备原始值和键值
                curr_key = torch.tanh(self.MLP_query(current_state))
                
                if t <= self.rank_k:
                    # 原始值 (使用 List Stack)
                    hist_vals = torch.stack(state_history_list, dim=1)
                    current_history_state = torch.cat((current_state, hist_vals), dim=1)
                    # 键值
                    if t == 0:
                        cached_keys_tensor = curr_key
                    else:
                        # 堆叠列表
                        hist_keys = torch.stack(projected_keys_list, dim=1)
                        cached_keys_tensor = torch.cat((curr_key, hist_keys), dim=1)
                else: 
                    Q = self.emb_table_question(q_next).clone().detach().unsqueeze(dim=-1)
                    K = self.emb_table_question(question[:, 0:t]).clone().detach()
                    product_score = torch.bmm(K, Q).squeeze(dim=-1)
                    _, indices = torch.topk(product_score, k=self.rank_k, dim=1)
                    
                    # 原始值 (Level 2 优化: 向量化 Gather 替代 Loop + List Stack Fix)
                    temp_history_tensor = torch.stack(state_history_list, dim=1)
                    idx_expanded_raw = indices.unsqueeze(-1).expand(-1, -1, self.emb_dim)
                    select_history = torch.gather(temp_history_tensor, 1, idx_expanded_raw)
                    current_history_state = torch.cat((current_state, select_history), dim=1)
                    
                    # Keys
                    temp_keys_buffer = torch.stack(projected_keys_list, dim=1)
                    idx_expanded = indices.unsqueeze(-1).expand(-1, -1, self.emb_dim)
                    select_keys = torch.gather(temp_keys_buffer, 1, idx_expanded)
                    cached_keys_tensor = torch.cat((curr_key, select_keys), dim=1)
            
            pid_data = (pid_ema, pid_diff) if self.use_pid else None
            
            # Pass cached_keys to predict
            y_hat[:, t + 1] = self.predict(qs_concat, current_history_state, q_target=q_next, pid_data=pid_data, cached_keys=cached_keys_tensor)
            
            # --- Update History State ---
            if self.recap_source == 'hsei':
                # 如果配置为 hsei，历史状态存储的是 Input Embedding (Projected)
                # 使用刚才定义的 neighbor_source (recap_feature)
                new_state = neighbor_source
            else:
                # @fix_fzq: Reverted usage of alignment layer.
                # In HSSI, we use raw LSTM output.
                new_state = lstm_output
            
            # state_history[:, t] = new_state
            # [Level 2 Fix] Append to History List (Avoid Inplace)
            state_history_list.append(new_state)

            # [Level 2 Optimization] Update Projected Cache
            # Compute projection for next time step usage
            # new_state: [Batch, Dim]
            # MLP_query(new_state) -> Tanh -> Cache
            projected_val = torch.tanh(self.MLP_query(new_state))
            
            # Append to list (Safe for Autograd)
            projected_keys_list.append(projected_val)
            
            h2_pre = lstm_output
            # state_history[:, t] = lstm_output
        return y_hat

    def aggregate(self, emb_node_neighbor, node_neighbors=None):
        # 图扩散模型
        # 输入是节点（习题节点）的embedding，计算步骤是：将节点和邻居的embedding相加，再通过一个MLP输出（embedding维度不变），激活函数用的tanh
        # 假设聚合3跳，那么输入是[0,1,2,3]，分别表示输入节点，1跳节点，2跳节点，3跳节点，总共聚合3次
        # 第1次聚合（每次聚合使用相同的MLP），(0,1)聚合得到新的embedding，放到输入位置0上；然后(1,2)聚合得到新的embedding，放到输入位置1上；然后(2,3)聚合得到新的embedding，放到输入位置2上
        # 第2次聚合，(0',1')，聚合得到新的embedding，放到输入位置0上；然后(1',2')聚合得到新的embedding，放到输入位置1上
        # 第3次聚合，(0'',1'')，聚合得到新的embedding，放到输入位置0上
        # 最后0'''通过一个MLP得到最终的embedding
        # aggregate from outside to inside
        
        for i in range(self.agg_hops):
            for j in range(self.agg_hops - i):
                if self.agg_method == 'qk_gat':
                    if node_neighbors is None:
                        raise ValueError("QK-GAT requires node_neighbors indices")
                    nodes_self = node_neighbors[j] 
                    nodes_neighbor = node_neighbors[j+1]
                    emb_node_neighbor[j] = self.gat_aggregate(emb_node_neighbor[j], emb_node_neighbor[j + 1], nodes_self, nodes_neighbor, j)
                else:
                    emb_node_neighbor[j] = self.sum_aggregate(emb_node_neighbor[j], emb_node_neighbor[j + 1], j)
        
        # 返回: (最终聚合结果[经过MLP], 聚合过程中的列表[包含邻居聚合信息])
        final_res = torch.tanh(self.MLP_AGG_last(emb_node_neighbor[0]))
        return final_res, emb_node_neighbor

    def sum_aggregate(self, emb_self, emb_neighbor, hop):
        """
        GCN-style Sum Aggregation with DropEdge support.
        将邻居节点求和平均之后与自己相加，得到聚合后的特征。
        """
        # @add_fzq: DropEdge Implementation for GCN
        # 在训练阶段随机 Mask 掉一部分邻居
        if self.training and self.drop_edge_rate > 0.0:
            # emb_neighbor shape: [Batch, ..., K, Dim]
            # 生成邻居级别的 mask: [Batch, ..., K, 1]
            edge_keep_prob = 1.0 - self.drop_edge_rate
            
            # 创建 mask，形状与邻居维度匹配
            edge_mask = torch.bernoulli(
                torch.full((*emb_neighbor.shape[:-1], 1), edge_keep_prob, device=emb_neighbor.device)
            )
            
            # 应用 mask (被 drop 的邻居置为 0)
            emb_neighbor_masked = emb_neighbor * edge_mask
            
            # 重新归一化：只对保留的邻居求平均
            # edge_mask sum 得到每个节点保留的邻居数量
            kept_count = edge_mask.sum(dim=-2, keepdim=True).clamp(min=1.0)  # 至少保留 1，防止除零
            emb_sum_neighbor = emb_neighbor_masked.sum(dim=-2) / kept_count.squeeze(-1)
        else:
            # 标准聚合：对所有邻居求平均
            emb_sum_neighbor = torch.mean(emb_neighbor, dim=-2)
        
        emb_sum = emb_sum_neighbor + emb_self
        mlp_output = self.mlps4agg[hop](emb_sum)
        gnn_output = self.dropout_gnn(mlp_output) if self.training else mlp_output
        return torch.tanh(gnn_output)

    def gat_aggregate(self, emb_self, emb_neighbor, nodes_self, nodes_neighbor, hop):
        """
        Graph Attention Aggregation
        Weights calculated using [Current, Neighbor, EdgeAttr]
        EdgeAttr comes from Question Features.
        """
        # nodes_self: [Batch, ...] (Indices)
        # nodes_neighbor: [Batch, ..., K] (Indices)
        # emb_self: [Batch, ..., Dim]
        # emb_neighbor: [Batch, ..., K, Dim]
        
        # 1. 确定 Edge Attributes (题目特征)
        # 聚合方向: Neighbor(j+1) -> Self(j)
        # hop=j.
        # j=0: Q <- S. Target=Q. EdgeAttr=Q_Features(Target).
        # j=1: S <- Q. Target=S. EdgeAttr=Q_Features(Source).
        # j=2: Q <- S. Target=Q. EdgeAttr=Q_Features(Target).
        # 偶数跳: Self 是 Q. 特征来自 Self.
        # 奇数跳: Self 是 S. 特征来自 Neighbor.
        
        is_self_q = (hop % 2 == 0)
        
        if is_self_q:
            # 特征来自 Emb_Self 对应的节点
            # nodes_self need to be broadcasted to match neighbor structure for concat
            # nodes_self: [Batch, ...,] -> emb_self: [Batch, ..., Dim]
            # We need features: [Batch, ..., 3] -> expand -> [Batch, ..., K, 3]
            
            # 查找特征
            q_idx = nodes_self.long()
            edge_attr = self.q_feature_embedding(q_idx) # [Batch, ..., 3]
            
            # 扩展到邻居维度 K
            # emb_neighbor shape last dim is Dim. second last is K.
            # edge_attr shape needs to insert K at second last dim.
            edge_attr = edge_attr.unsqueeze(-2) # [Batch, ..., 1, 3]
            edge_attr = edge_attr.expand(*emb_neighbor.shape[:-1], self.q_feat_dim) # [Batch, ..., K, 3]
            
        else:
            # 特征来自 Emb_Neighbor 对应的节点 (Source is Q)
            q_idx = nodes_neighbor.long()
            edge_attr = self.q_feature_embedding(q_idx) # [Batch, ..., K, 3]
            
        # 2. 准备注意力输入 [h_i, h_j, f_edge]
        # h_i = emb_self expanded
        emb_self_exp = emb_self.unsqueeze(-2).expand_as(emb_neighbor) # [Batch, ..., K, Dim]
        
        # Concat
        cat_input = torch.cat([emb_self_exp, emb_neighbor, edge_attr], dim=-1) # [Batch, ..., K, 2*Dim + 3]
        
        # 3. 计算分数与权重
        # scores = self.gat_attn_layers[hop](cat_input) # [Batch, ..., K, 1]
        # scores = torch.nn.functional.leaky_relu(scores, negative_slope=0.2)
        scores = torch.nn.functional.leaky_relu(self.gat_attn_layers[hop](cat_input), negative_slope=0.2)

        # @add_fzq: DropEdge Implementation (Masking Attention Scores)
        if self.training and self.drop_edge_rate > 0.0:
            # DropEdge: set scores to -inf for dropped edges before softmax
            edge_keep_prob = 1.0 - self.drop_edge_rate
            edge_mask = torch.bernoulli(torch.full_like(scores, edge_keep_prob))
            # 模型启用了 AMP（自动混合精度），将 DropEdge 屏蔽值改为 -1e4，确保 float16 不会溢出
            scores = scores.masked_fill(edge_mask == 0, -1e4)
        
        # Softmax over neighbors
        alpha = torch.softmax(scores, dim=-2) # [Batch, ..., K, 1]
        
        # 4. 加权聚合
        emb_weighted = emb_neighbor * alpha # [Batch, ..., K, Dim]
        emb_agg = torch.sum(emb_weighted, dim=-2) # [Batch, ..., Dim]
        
        # 5. 更新状态 (Residual / Skip Connection)
        # 与 sum_aggregate 保持一致结构: Output = MLP( Agg + Self )
        # GAT原版通常直接对 Layer 2 再次 Attention. 
        # 但这里为了适配 GIKT 架构 (ResNet style)，我们做 (Agg + Self)
        emb_sum = emb_agg + emb_self
        mlp_output = self.mlps4agg[hop](emb_sum)
        gnn_output = self.dropout_gnn(mlp_output) if self.training else mlp_output
        return torch.tanh(gnn_output)

    def recap_hard(self, q_next, q_history):
        # 硬选择, 直接在q_history中选出与q_next有相同技能的问题
        # q_next: [batch_size, 1], q_history: [batch_size, t-1]
        batch_size = q_next.shape[0]
        q_next = q_next.reshape(-1)

        # 1. 获取与当前问题相关的技能和对应的候选问题集合
        skill_related = self.q_neighbors_t[q_next]  # [batch, q_neighbor_size]
        q_related = self.s_neighbors_t[skill_related].reshape(batch_size, -1)  # [batch, q_neighbor_size * s_neighbor_size]

        # 2.1 扩展历史问题维度以支持广播
        history_expanded = q_history.unsqueeze(-1)  # [batch, history_len, 1]
        # 2.2 扩展候选问题维度
        related_expanded = q_related.unsqueeze(1)   # [batch, 1, num_related]

        # 2.3 广播比较：生成匹配矩阵
        matches = history_expanded.eq(related_expanded)
        # 2.4 检查有效历史：排除填充值
        valid_history = q_history.ne(0)  # 避免填充位被误选
        match_any = matches.any(dim=-1) & valid_history

        # 3. 提取结果 - 返回布尔掩码而非列表
        # Change Level 1: Return mask [Batch, t-1] for vectorization
        return match_any 

    def predict(self, qs_concat, current_history_state, q_target=None, pid_data=None, cached_keys=None):
        # qs_concat: [batch_size, num_qs, dim_emb]
        # current_history_state: [batch_size, num_state, dim_emb] (Values)
        # cached_keys: [batch_size, num_state, dim_emb] (可选: 预计算键值)
        # q_target: [batch_size] (可选: 用于 Diff-GIKT)
        # pid_data: tuple(pid_ema, pid_diff) (可选: 用于 PID-GIKT)
        
        # 1. 计算原始相关性 (使用 Values)
        output_g = torch.bmm(qs_concat, torch.transpose(current_history_state, 1, 2)) 
        num_qs, num_state = qs_concat.shape[1], current_history_state.shape[1]
        
        # 2. 计算注意力权重
        # If cached_keys provided, use directly. Else compute.
        if cached_keys is not None:
            K_base = cached_keys
        else:
            K_base = torch.tanh(self.MLP_query(current_history_state))
            
        Q_base = torch.tanh(self.MLP_key(qs_concat))  # [batch_size, num_qs, dim_emb]
        
        w1 = self.MLP_W.weight[:, :self.emb_dim] # [1, Dim]
        w2 = self.MLP_W.weight[:, self.emb_dim:] # [1, Dim]
        b = self.MLP_W.bias

        score_q = F.linear(Q_base, w1) 
        score_k = F.linear(K_base, w2) 
        
        tmp = score_q.unsqueeze(2) + score_k.unsqueeze(1) + b 
        tmp = torch.squeeze(tmp, dim=-1)  
        alpha = torch.softmax(tmp, dim=2) 
        
        # 3. 加权聚合与预测
        p = torch.sum(alpha * output_g, dim=(1, 2)) 
        
        # @fix_fzq: HSSI Stability - Scale Dot Product Attention
        # 即使只在 predict 计算 score 是不够的，最终的聚合能力值 p 也是点积的结果
        # HSSI (Hidden State): 方差较大，导致 p 值分布过宽，sigmoid 后梯度消失/爆炸
        # HSEI (Input Emb): 方差较小，p 值稳定
        # 添加缩放因子以稳定训练动态 (参考 Transformer)
        p = p / (self.emb_dim ** 0.5)

        # @add_fzq: PID-GIKT Bias Injection
        # Modify Student Ability (p) with PID Control Signal
        if pid_data is not None:
            pid_ema, pid_diff = pid_data

            # Prepare PID Signals for current target question
            if self.pid_mode == 'domain' and q_target is not None:
                # Retrieve Domain Mask for target question [Batch, D]
                target_mask = self.q_domain_mask_buffer[q_target] # [Batch, D]
                
                # Check for questions with NO domains (all zeros) for safety
                # Add epsilon to denominator
                domain_count = target_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
                
                # @mod_fzq: Apply per-domain weights before aggregation
                # term_i for all domains: [Batch, D]
                all_term_i = (pid_ema - 0.5) * self.w_pid_i.unsqueeze(0)
                all_term_d = torch.tanh(self.pid_lambda * pid_diff) * self.w_pid_d.unsqueeze(0)
                
                # Aggregate weighted bias: [Batch]
                pid_bias = (all_term_i * target_mask + all_term_d * target_mask).sum(dim=1) / domain_count.squeeze(-1)
                
            else:
                # Global Mode: pid_ema is [Batch]
                pid_i_val = pid_ema
                pid_d_val = pid_diff
                
                # Integral Term: Deviation from neutral (0.5), range approx [-0.5, 0.5]
                # w_pid_i * (S_t - 0.5)
                term_i = pid_i_val - 0.5
                
                # Derivative Term: Trend direction
                # Tanh Limiter prevents noise amplification
                term_d = torch.tanh(self.pid_lambda * pid_d_val)
                
                # Apply Bias to Student Ability (p)
                # p represents "Theta" in IRT contexts
                pid_bias = self.w_pid_i * term_i + self.w_pid_d * term_d
            
            p = p + pid_bias
            
            # # @debug_fzq: Print PID stats usually once to verify it works
            # if not self.training and torch.rand(1).item() < 0.001:  # Low probability print during testing
            #     print(f"[Diff-GIKT Debug] Mode: {self.pid_mode}, PID Bias Mean: {pid_bias.mean().item():.4f}, Max Bias: {pid_bias.max().item():.4f}")

        if q_target is not None and self.use_4pl_irt:
            # [Backward Compatibility Check]
            if hasattr(self, 'difficulty_bias'):
                # 1. 获取题目级基础难度 (Question-specific Difficulty)
                q_diff = torch.squeeze(self.difficulty_bias(q_target), dim=-1) 
                
                # 2. 获取知识点级共性难度 (Hierarchical Step 2)
                if hasattr(self, 'skill_difficulty_bias'):
                    related_mask = self.qs_table[q_target].to(torch.float32)
                    s_diff_sum = torch.matmul(related_mask, self.skill_difficulty_bias.weight).squeeze(-1)
                    s_count = torch.sum(related_mask, dim=1)
                    s_diff_avg = s_diff_sum / (s_count + 1e-8)
                    difficulty = q_diff + s_diff_avg
                else:
                    difficulty = q_diff
                
                # 3. 引入判别度、猜测率与失误率 (Step 3: 4PL Model)
                if self.use_4pl_irt:
                    # a_q (Discrimination): 区分度，必须为正。使用 softplus 确保平滑且 > 0
                    a_q = 1.0 + F.softplus(torch.squeeze(self.discrimination_bias(q_target), dim=-1))
                    
                    # c_q (Guessing): 猜测率，收紧到 [0, 0.2]
                    c_q = 0.2 * torch.sigmoid(torch.squeeze(self.guessing_bias(q_target), dim=-1))
                    
                    # d_q (Slipping): 失误率，收紧到 [0, 0.05]
                    d_q = 0.05 * torch.sigmoid(torch.squeeze(self.slipping_bias(q_target), dim=-1))
                    
                    # 4PL 核心预测公式: P = c + (1 - c - d) * sigmoid(a * (theta - b))
                    # 此处 p 作为注意力聚合后的能力值 theta (Student Ability)
                    logits = a_q * (p - difficulty)
                    p_base = torch.sigmoid(logits)
                    p_final = c_q + (1.0 - c_q - d_q) * p_base
                    
                    # TF Alignment: 为了兼容 BCEWithLogitsLoss，将概率反向映射回 logit：L = log(p/(1-p))
                    p_final = torch.clamp(p_final, 1e-7, 1.0 - 1e-7)
                    p = torch.log(p_final / (1.0 - p_final))
        # @add_fzq: TF Alignment - Always return Logits for BCEWithLogitsLoss
        # 因为主训练循环使用了 BCEWithLogitsLoss，所以这里必须严格返回 Logits
        result = p 
        
        return result

    def reset_parameters(self):
        """
        Initialize parameters with Xavier Uniform to match TensorFlow implementation
        """
        for name, param in self.named_parameters():
            # Skip initialization for pre-trained embeddings
            if self.pre_train and 'emb_table' in name:
                continue
                
            if 'weight' in name:
                if len(param.shape) >= 2:
                    torch.nn.init.xavier_uniform_(param)
                else:
                    torch.nn.init.uniform_(param, -0.1, 0.1) # Fallback for 1D weights
            elif 'bias' in name:
                torch.nn.init.zeros_(param)

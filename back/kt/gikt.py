import os
import torch
from torch.nn import Module, Embedding, Linear, ModuleList, Dropout, LSTMCell

from config import DEVICE

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

    def __init__(self, num_question, num_skill, q_neighbors, s_neighbors, qs_table, agg_hops=3, emb_dim=100, dropout=(0.2, 0.4), hard_recap=True, rank_k=10, pre_train=False, use_cognitive_model=False, data_dir=None, agg_method='gcn', recap_source='hssi', enable_tf_alignment=False):
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
            dropout: dropout率，默认为(0.2, 0.4)
            hard_recap: 是否使用硬回顾，默认为True
            rank_k: 排序的k值，默认为10
            pre_train: 是否预训练，默认为False
            use_cognitive_model: 是否使用认知模型(CognitiveRNNCell)，默认为False
            data_dir: 数据目录，用于加载预训练向量，默认为None
            agg_method: 聚合方法，默认为'gcn'
            recap_source: 回顾特征来源，默认为'hssi'
            enable_tf_alignment: 是否启用 TF 对齐 (Logits 输出, Xavier Init)，默认为False
        '''
        super(GIKT, self).__init__()  # 调用父类的初始化方法
        self.model_name = "gikt" #  模型名称设置
        self.num_question = num_question #  问题数量设置
        self.num_skill = num_skill #  技能数量设置
        self.q_neighbors = q_neighbors #  问题邻居数量设置
        self.s_neighbors = s_neighbors #  技能邻居数量设置
        self.agg_hops = agg_hops #  聚合跳跃层数设置
        self.qs_table = qs_table #  问题-技能关系表设置
        self.emb_dim = emb_dim #  嵌入维度设置
        self.hard_recap = hard_recap #  困难回顾设置
        self.rank_k = rank_k #  排名参数k设置
        self.use_cognitive_model = use_cognitive_model # 是否使用认知模型
        self.agg_method = agg_method # 聚合方法
        self.recap_source = recap_source # 硬回顾特征来源
        self.enable_tf_alignment = enable_tf_alignment # 是否启用TF对齐

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
            # Step 3 Init: TF Alignment uses projected input (emb_dim), Original uses concatenated (emb_dim*2)
            _input_size = emb_dim if (hasattr(self, 'enable_tf_alignment') and self.enable_tf_alignment) else emb_dim * 2
            self.cognitive_cell = CognitiveRNNCell(input_size=_input_size, hidden_size=emb_dim)
        else:
            _input_size = emb_dim if (hasattr(self, 'enable_tf_alignment') and self.enable_tf_alignment) else emb_dim * 2
            self.lstm_cell = LSTMCell(input_size=_input_size, hidden_size=emb_dim) # 使用LSTM网络

        if self.recap_source == 'hsei':
            # 如果使用 Input Embedding 作为回顾特征，需要将其从 2*emb_dim 映射到 emb_dim (假设)
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
        self.dropout_lstm = Dropout(dropout[0]) #  LSTM网络的dropout层
        # self.dropout_gru = Dropout(dropout[0]) #  GRU网络的dropout层
        self.dropout_gnn = Dropout(dropout[1]) #  GNN网络的dropout层
        self.MLP_query = Linear(emb_dim, emb_dim) #  查询向量转换的线性层
        self.MLP_key = Linear(emb_dim, emb_dim) #  键向量转换的线性层
        # 公式10中的W
        self.MLP_W = Linear(2 * emb_dim, 1)

        # @add_fzq: TF Alignment - Initialize weights if enabled
        if hasattr(self, 'enable_tf_alignment') and self.enable_tf_alignment:
            self.reset_parameters()
        
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
        q_neighbor_size, s_neighbor_size = self.q_neighbors.shape[1], self.s_neighbors.shape[1]
        h1_pre = torch.nn.init.xavier_uniform_(torch.zeros(self.emb_dim, device=DEVICE).repeat(batch_size, 1)) #  使用Xavier初始化方法创建两个全零的嵌入向量，并重复batch_size次
        h2_pre = torch.nn.init.xavier_uniform_(torch.zeros(self.emb_dim, device=DEVICE).repeat(batch_size, 1))
        state_history = torch.zeros(batch_size, seq_len, self.emb_dim, device=DEVICE) #  初始化状态历史记录和预测结果张量
        y_hat = torch.zeros(batch_size, seq_len, device=DEVICE)
        for t in range(seq_len - 1): # 第t时刻
            question_t = question[:, t] #  取出所有学生在第 t 个时间步所做的题目ID
            response_t = response[:, t] #  取出所有学生在第 t 个时间步所做的题目ID的回答结果
            # 在实际数据中，每个学生的答题序列长度不同。为了能批量处理，会用0填充（padding）到统一长度 seq_len。
            # mask 张量就是用来标记哪些位置是真实数据(1)，哪些是填充(0)。
            # 如果mask[:, t]是[1, 0, 1]，比较结果就是[True, False, True]            
            mask_t = torch.eq(mask[:, t], torch.tensor(1)) #  创建一个布尔掩码，用于标识第t个时间步中哪些位置是有效的（值为1）
            emb_response_t = self.emb_table_response(response_t) # [batch_size, emb_dim]
            # GNN获得习题的embedding
            # question_t[mask_t]：这是 PyTorch 中的布尔索引操作，它会返回 question_t 中对应 mask_t 为 True 的位置的所有元素。
            node_neighbors = [question_t[mask_t]] # 当前节点的邻居节点列表,[自己, 第一跳节点, 第二跳节点...]
            _batch_size = len(node_neighbors[0]) # 当前的批量, 不一定是设定好的批量。得到当前步的真实有效样本数。
            for i in range(self.agg_hops):
                nodes_current = node_neighbors[-1].reshape(-1) # 当前正在遍历的node(上次新添加的邻居)
                # nodes_current = nodes_current.reshape(-1)
                neighbor_shape = [_batch_size] + [(q_neighbor_size if j % 2 == 0 else s_neighbor_size) for j in range(i + 1)]
                # [t时刻问题数量, q_neighbor_size, s_neighbor_size, q_neighbor_size, ...]
                if i % 2 == 0: # 找知识点节点
                    node_neighbors.append(self.q_neighbors[nodes_current].reshape(neighbor_shape))
                else: # 找习题节点
                    node_neighbors.append(self.s_neighbors[nodes_current].reshape(neighbor_shape))
            emb_node_neighbor = [] # 每层邻居(问题或者知识点)的嵌入向量,形状为node_neighbor.shape + [emb_dim]
            for i, nodes in enumerate(node_neighbors):
                if i % 2 == 0: # 问题索引->问题向量
                    emb_node_neighbor.append(self.emb_table_question(nodes))
                else: # 技能索引->技能向量
                    emb_node_neighbor.append(self.emb_table_skill(nodes)) #  将邻居节点的嵌入表示添加到列表中
            emb0_question_t, _ = self.aggregate(emb_node_neighbor) # [batch_size, emb_dim] 该时刻聚合更新过的问题向量
            emb_question_t = torch.zeros(batch_size, self.emb_dim, device=DEVICE) #  初始化一个全零的张量，形状为(batch_size, self.emb_dim)，并指定设备为DEVICE
            emb_question_t[mask_t] = emb0_question_t #  将emb0_question_t的值赋给emb_question_t中mask_t为True的位置
            emb_question_t[~mask_t] = self.emb_table_question(question_t[~mask_t]) #  对于emb_question_t中mask_t为False的位置，使用emb_table_question查找question_t中对应位置的嵌入向量
            
            # ----------------------------------------------------
            # @add_fzq v5 optimization: Feature Transform
            # TF Alignment: Apply transformation to question embedding before concat
            # ----------------------------------------------------
            emb_question_trans = torch.relu(self.feature_transform_layer(emb_question_t))

            # LSTM/GRU更新知识状态
            # 使用变换后的 Question Embedding 进行拼接
            lstm_input = torch.cat((emb_question_trans, emb_response_t), dim=1) # [batch_size, emb_dim * 2]
            
            # --- Prepare Recap Feature ---
            if self.recap_source == 'hsei':
                # 使用 Input (经过线性变换) 作为 History State
                # Align with TF: input_trans_embedding (Dense layer, Linear activation)
                # Removed torch.tanh to match TF default
                recap_feature = self.input_trans_layer(lstm_input) 
                # TF code snippet shows: input_trans_embedding = tf.reshape(tf.layers.dense(input_fa_embedding, hidden_size), ...) 
                # Default activation of dense is None (Linear). But RNN usually operates on Tanh/ReLU. 
                # Model GIKT aggregate uses Tanh. Let's assume Tanh for consistency with state.
            
            # Determine LSTM Input
            if hasattr(self, 'enable_tf_alignment') and self.enable_tf_alignment:
                # [Case: TF Alignment] LSTM uses Projected Features (100 dim)
                # Note: `recap_feature` is calculated above as input_trans_layer(lstm_input)
                lstm_cell_input = recap_feature
            else:
                # [Case: Original] LSTM uses Concatenated Features (200 dim)
                lstm_cell_input = lstm_input

            if self.use_cognitive_model:
                if interval_time is None or response_time is None:
                    raise ValueError("CognitiveRNNCell requires interval_time and response_time")
                
                # 获取当前时刻的时间特征 [batch_size, 1]
                curr_interval = interval_time[:, t].unsqueeze(1)
                curr_response = response_time[:, t].unsqueeze(1)
                
                # h2_pre 作为上一个时刻的 hidden state
                # Cognitive Model 使用包含非线性变换特征的 lstm_input (宽输入)
                # @change_fzq: Pass correct input size based on alignment
                h_new, _ = self.cognitive_cell(lstm_cell_input, h2_pre, curr_interval, curr_response)
                lstm_output = self.dropout_lstm(h_new)
            else:
                lstm_output = self.dropout_lstm(self.lstm_cell(lstm_cell_input)[0]) # [batch_size, emb_dim]

            # 找t+1时刻的[习题]以及[其对应的知识点]
            q_next = question[:, t + 1] # [batch_size, ]
            skills_related = self.qs_table[q_next] # [batch_size, num_skill]
            skills_related_list = [] # [[num_skill1, emb_dim], [num_skill2, emb_dim], ...]
            max_num_skill = 1 # 求一个问题的最多关联的技能的数量
            for i in range(batch_size):
                skills_index = torch.nonzero(skills_related[i]).squeeze()
                if len(skills_index.shape) == 0: # 只有一个技能
                    skills_related_list.append(torch.unsqueeze(self.emb_table_skill(skills_index), dim=0)) # [1, emb_dim]
                else: # 不止一个技能
                    skills_related_list.append(self.emb_table_skill(skills_index)) # [num_skill, emb_dim]
                    if skills_index.shape[0] > max_num_skill:
                        max_num_skill = skills_index.shape[0]

            # 将习题和对应知识点embedding拼接起来
            # --- 逻辑分支修复：仅在 hsei 模式下启用 Target 聚合，hssi 模式保持原样 ---
            use_legacy_concat = True # 使用“旧版连接”功能
            qs_concat = None

            if self.recap_source == 'hsei':
                # [Case: HSEI] 需要特征对齐，执行图聚合
                # 1. 构建 q_next 的多跳邻居图
                node_neighbors_next = [q_next] # 初始节点
                _batch_size_next = len(node_neighbors_next[0])
                
                for i in range(self.agg_hops):
                    nodes_current = node_neighbors_next[-1].reshape(-1)
                    neighbor_shape = [_batch_size_next] + [(q_neighbor_size if j % 2 == 0 else s_neighbor_size) for j in range(i + 1)]
                    
                    if i % 2 == 0: 
                        node_neighbors_next.append(self.q_neighbors[nodes_current].reshape(neighbor_shape))
                    else: 
                        node_neighbors_next.append(self.s_neighbors[nodes_current].reshape(neighbor_shape))
                
                # 2. 转换为 Embedding 并聚合
                emb_node_neighbor_next = []
                for i, nodes in enumerate(node_neighbors_next):
                    if i % 2 == 0: 
                        emb_node_neighbor_next.append(self.emb_table_question(nodes))
                    else: 
                        emb_node_neighbor_next.append(self.emb_table_skill(nodes))
                
                # 3. 聚合
                # Return: (final_res, full_layers_list)
                emb_q_next, agg_list_next = self.aggregate(emb_node_neighbor_next)
                
                if hasattr(self, 'enable_tf_alignment') and self.enable_tf_alignment:
                    # TF Logic: qs_concat = concat(emb_q_next, agg_results_next[1])
                    # agg_results_next[1] 是第一层聚合后的 Skill 邻居特征 [Batch, Q_Neighbor_Size, Emb]
                    emb_skills_next_batch = agg_list_next[1] # [Batch, Q_Neighbor_Size, Emb]
                    qs_concat = torch.cat((emb_q_next.unsqueeze(1), emb_skills_next_batch), dim=1)
                    use_legacy_concat = False
                
            else:
                # [Case: Defaults/HSSI] 保持原代码逻辑，使用原始 Embedding
                # (对应原代码注释：经验效果不好，不使用聚合)
                emb_q_next = self.emb_table_question(q_next) 
            # -------------------------------------------------------------------------

            if use_legacy_concat:
                qs_concat = torch.zeros(batch_size, max_num_skill + 1, self.emb_dim).to(DEVICE) #  创建一个零张量，用于存储问题与技能的拼接嵌入 形状为: batch_size × (max_num_skill + 1) × emb_dim 并将其移动到指定设备(GPU/CPU)上
                for i, emb_skills in enumerate(skills_related_list): # emb_skills: [num_skill, emb_dim]
                    num_qs = 1 + emb_skills.shape[0] # 总长度为1(问题嵌入长度) + num_skill(技能嵌入长度) #  计算问题与技能拼接后的总长度，包含问题嵌入和所有技能嵌入
                    emb_next = torch.unsqueeze(emb_q_next[i], dim=0) # [1, emb_dim] #  对下一个问题的嵌入向量进行维度扩展，从[emb_dim]变为[1, emb_dim]
                    qs_concat[i, 0 : num_qs] = torch.cat((emb_next, emb_skills), dim=0) #  将下一个问题的嵌入和技能嵌入拼接后存入qs矩阵的相应位置 拼接后的向量长度为num_qs，从qs矩阵的第0个位置开始存储

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
            #     # 问题2关联1个技能
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
                y_hat[:, 0] = 0.5 # 第一个问题默认0.5的正确率
                y_hat[:, 1] = self.predict(qs_concat, torch.unsqueeze(lstm_output, dim=1))
                continue
            # recap硬选择历史问题
            if self.hard_recap:
                history_time = self.recap_hard(q_next, question[:, 0:t]) # 选取哪些时刻的问题
                selected_states = [] # 不同时刻t选择的历史状态
                max_num_states = 1 # 求最大的历史状态数量
                
                # Determine "neighbor" source (retrieved items)
                if self.recap_source == 'hsei':
                    neighbor_source = recap_feature
                else:
                    neighbor_source = lstm_output

                for row, selected_time in enumerate(history_time):
                    # Align with TF: Element 0 (Current State) is ALWAYS lstm_output
                    # regardless of what we retrieve from history (neighbor_source)
                    current_state = torch.unsqueeze(lstm_output[row], dim=0) # [1, emb_dim]
                    
                    if len(selected_time) == 0: # 没有历史状态,直接取当前状态
                        selected_states.append(current_state)
                    else: # 有历史状态,将历史状态和当前状态连接起来
                        # Retrieve neighbors from the correct source
                        selected_state = state_history[row, torch.tensor(selected_time, dtype=torch.int64)]
                        selected_states.append(torch.cat((current_state, selected_state), dim=0))
                        if (selected_state.shape[0] + 1) > max_num_states:
                            max_num_states = selected_state.shape[0] + 1
                current_history_state = torch.zeros(batch_size, max_num_states, self.emb_dim).to(DEVICE)
                # 当前状态
                for b, c_h_state in enumerate(selected_states):
                    num_states = c_h_state.shape[0]
                    current_history_state[b, 0 : num_states] = c_h_state
            else: # 软选择
                current_state = lstm_output.unsqueeze(dim=1)
                if t <= self.rank_k:
                    current_history_state = torch.cat((current_state, state_history[:, 0:t]), dim=1)
                else: # 基于注意力机制，从历史状态中选择最相关的信息来辅助当前决策
                    Q = self.emb_table_question(q_next).clone().detach().unsqueeze(dim=-1)
                    K = self.emb_table_question(question[:, 0:t]).clone().detach()
                    product_score = torch.bmm(K, Q).squeeze(dim=-1)
                    _, indices = torch.topk(product_score, k=self.rank_k, dim=1)
                    select_history = torch.cat(tuple(state_history[i][indices[i]].unsqueeze(dim=0) for i in range(batch_size)), dim=0)
                    current_history_state = torch.cat((current_state, select_history), dim=1)
            y_hat[:, t + 1] = self.predict(qs_concat, current_history_state)
            
            # --- Update History State ---
            if self.recap_source == 'hsei':
                # 如果配置为 hsei，历史状态存储的是 Input Embedding (Projected)
                # 使用刚才定义的 neighbor_source (recap_feature)
                state_history[:, t] = neighbor_source
            else:
                # 默认 (hssi): 历史状态存储的是 LSTM Output (Hidden State)
                state_history[:, t] = lstm_output
            
            h2_pre = lstm_output
            # state_history[:, t] = lstm_output
        return y_hat

    def aggregate(self, emb_node_neighbor):
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
                emb_node_neighbor[j] = self.sum_aggregate(emb_node_neighbor[j], emb_node_neighbor[j + 1], j)
        
        # 返回: (最终聚合结果[经过MLP], 聚合过程中的列表[包含邻居聚合信息])
        final_res = torch.tanh(self.MLP_AGG_last(emb_node_neighbor[0]))
        return final_res, emb_node_neighbor

    def sum_aggregate(self, emb_self, emb_neighbor, hop):
        # 求和式聚合, 将邻居节点求和平均之后与自己相加, 得到聚合后的特征
        emb_sum_neighbor = torch.mean(emb_neighbor, dim=-2)
        emb_sum = emb_sum_neighbor + emb_self
        return torch.tanh(self.dropout_gnn(self.mlps4agg[hop](emb_sum)))

    def recap_hard(self, q_next, q_history):
        # 硬选择, 直接在q_history中选出与q_next有相同技能的问题
        # q_next: [batch_size, 1], q_history: [batch_size, t-1]
        batch_size = q_next.shape[0]
        q_neighbor_size, s_neighbor_size = self.q_neighbors.shape[1], self.s_neighbors.shape[1] #  获取Q邻居和S邻居的大小，即每个问题/技能的邻居数量
        q_next = q_next.reshape(-1) #  将q_next重塑为一维数组，便于后续索引操作
        # 获取与当前问题相关的技能列表
        skill_related = self.q_neighbors[q_next].reshape((batch_size, q_neighbor_size)).reshape(-1) 
        # 获取与当前问题相关的技能对应的问题列表
        q_related = self.s_neighbors[skill_related].reshape((batch_size, q_neighbor_size * s_neighbor_size)).tolist() #  获取与当前问题相关的问题列表，并转换为Python列表
        time_select = [[] for _ in range(batch_size)] #  初始化一个列表，用于存储每个用户的相关时间点
        for row in range(batch_size): #  遍历批次中的每个用户
            key = q_history[row].tolist() # 该用户的回答过的问题列表
            query = q_related[row] # 与该用户的当前问题相关的问题列表
            for t, k in enumerate(key): #  遍历用户的历史回答记录
                if k in query: #  检查历史问题是否在相关问题列表中
                    time_select[row].append(t) #  如果是，则记录该时间点
        return time_select #  返回每个用户的相关时间点列表

    def recap_soft(self, rank_k=10):
        # 软选择
        pass

    def predict(self, qs_concat, current_history_state):
        # qs_concat: [batch_size, num_qs, dim_emb]
        # current_history_state: [batch_size, num_state, dim_emb]
        # 1. 计算原始相关性
        output_g = torch.bmm(qs_concat, torch.transpose(current_history_state, 1, 2)) # 计算待预测题目（及关联知识点）与每个历史状态的原始点积分数
        num_qs, num_state = qs_concat.shape[1], current_history_state.shape[1]
        states = torch.unsqueeze(current_history_state, dim=1)  # [batch_size, 1, num_state, dim_emb]
        states = states.repeat(1, num_qs, 1, 1)  # [batch_size, num_qs, num_state, dim_emb]
        qs_concat2 = torch.unsqueeze(qs_concat, dim=2)  # [batch_size, num_qs, 1, dim_emb]
        qs_concat2 = qs_concat2.repeat(1, 1, num_state, 1)  # [batch_size, num_qs, num_state, dim_emb]
        # 2. 计算注意力权重
        K = torch.tanh(self.MLP_query(states))  # [batch_size, num_qs, num_state, dim_emb]
        Q = torch.tanh(self.MLP_key(qs_concat2))  # [batch_size, num_qs, num_state, dim_emb]
        tmp = self.MLP_W(torch.cat((Q, K), dim=-1))  # [batch_size, num_qs, num_state, 1]
        tmp = torch.squeeze(tmp, dim=-1)  # [batch_size, num_qs, num_state]
        alpha = torch.softmax(tmp, dim=2)  # [batch_size, num_qs, num_state]
        # 3. 加权聚合与预测
        # 用注意力权重 alpha 对原始相关性分数 output_g 进行加权求和，得到一个综合得分
        p = torch.sum(torch.sum(alpha * output_g, dim=1), dim=1)  # [batch_size, 1]
        
        # @add_fzq: TF Alignment
        if self.enable_tf_alignment:
            # Case 1: Return Logits (No Sigmoid) to use with BCEWithLogitsLoss
            result = torch.squeeze(p, dim=-1)
        else:
            # Case 2: Return Probabilities (Sigmoid) (Original behavior)
            result = torch.sigmoid(torch.squeeze(p, dim=-1)) 
        
        return result

    def reset_parameters(self):
        """
        Initialize parameters with Xavier Uniform to match TensorFlow implementation
        """
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    torch.nn.init.xavier_uniform_(param)
                else:
                    torch.nn.init.uniform_(param, -0.1, 0.1) # Fallback for 1D weights
            elif 'bias' in name:
                torch.nn.init.zeros_(param)
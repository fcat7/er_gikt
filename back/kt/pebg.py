"""
PEBG问题向量预训练模型
仅仅用于提取问题和技能之间的关系,不涉及具体用户
PEBG（Pre-training Embeddings on Bipartite Graphs）就像是在教学生之前，先让您通过“目录”和“标签”去预习题目之间的关系。

它利用了三种现成的关系表（这些关系是固定的，不需要看学生做题就知道）：

Q-S (题目-知识点): 题目A 考察 加法，题目B 考察 加法。 -> 推导: A和B有关联。
Q-Q (题目-题目): 如果两个题目考察同一个知识点，它们就是“邻居”。
S-S (知识点-知识点): 两个知识点经常在同一道题里出现，它俩就有关联。
"""
import torch
import torch.nn as nn
from config import BaseConfig

DEVICE = BaseConfig.DEVICE

class PEBG(nn.Module):

    def __init__(self, qs_table, qq_table, ss_table, emb_dim=100):
        super().__init__()
        num_q, num_s = qs_table.shape[0], qs_table.shape[1]
        
        # 定义已知(对比)变量
        self.qs_target = torch.as_tensor(qs_table, dtype=torch.float, device=DEVICE) # [num_q, num_s] 问题-技能表
        self.qq_target = torch.as_tensor(qq_table, dtype=torch.float, device=DEVICE) # [num_q, num_q] 问题-问题表
        self.ss_target = torch.as_tensor(ss_table, dtype=torch.float, device=DEVICE) # [num_s, num_s] 技能-技能表
        
        # 移除原代码中对 q_feature.npy 的依赖 (因其在 forward 中未被使用且 builder.py 中未生成)
        
        # 需要训练的参数
        self.q_embedding = nn.Parameter(torch.randn(size=[num_q, emb_dim])).to(DEVICE)  # 问题顶点特征
        self.s_embedding = nn.Parameter(torch.randn(size=[num_s, emb_dim])).to(DEVICE)  # 技能顶点特征
        
        # 定义网络层
        self.fc_q = nn.Linear(self.q_embedding.shape[1], emb_dim) # [num_q, emb_dim]
        self.fc_s = nn.Linear(self.s_embedding.shape[1], emb_dim) # [num_s, emb_dim]
        self.relu_q = nn.ReLU()
        self.relu_s = nn.ReLU()
        self.sigmoid = [nn.Sigmoid().to(DEVICE) for _ in range(3)]
        print('PEBG model built')

    def forward(self, q_embedding, s_embedding):
        # 接收已经确定好批次的输入向量
        q_embedding_fc = self.relu_q(self.fc_q(q_embedding)) # [batch_size_q, emb_dim]
        s_embedding_fc = self.relu_s(self.fc_s(s_embedding)) # [num_s, emb_dim]
        
        qs_logit = self.sigmoid[0](torch.matmul(q_embedding_fc, s_embedding_fc.T))  # [batch_size_q, num_s]
        qq_logit = self.sigmoid[1](torch.matmul(q_embedding_fc, q_embedding_fc.T))  # [batch_size_q, batch_size_q]
        ss_logit = self.sigmoid[2](torch.matmul(s_embedding_fc, s_embedding_fc.T))  # [num_s, num_s]
        
        return qs_logit, qq_logit, ss_logit

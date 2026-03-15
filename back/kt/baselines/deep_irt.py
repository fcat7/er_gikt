import torch
from torch.nn import Module, Parameter, Embedding, Linear, Dropout
from torch.nn.init import kaiming_normal_

class DeepIRT(Module):
    def __init__(self, num_q, num_c, dim_s=64, size_m=50, dropout=0.2):
        super().__init__()
        self.model_name = "deep_irt"
        self.num_q = num_q
        self.num_c = num_c
        self.dim_s = dim_s
        self.size_m = size_m

        # 注意：这里我们同时支持知识点和题目的映射，为了兼容原版，核心以题目/知识点作为键。
        # 这里进行双重嵌入（Joint Embedding Trick）
        
        # 互动 embedding。一半由于Q，一半用于C
        self.v_q_emb_layer = Embedding(self.num_q * 2 + 10, self.dim_s // 2)
        self.v_c_emb_layer = Embedding(self.num_c * 2 + 10, self.dim_s // 2)
        
        self.k_q_emb_layer = Embedding(self.num_q + 10, self.dim_s // 2)
        self.k_c_emb_layer = Embedding(self.num_c + 10, self.dim_s // 2)

        self.Mk = Parameter(torch.Tensor(self.size_m, self.dim_s))
        self.Mv0 = Parameter(torch.Tensor(self.size_m, self.dim_s))

        kaiming_normal_(self.Mk)
        kaiming_normal_(self.Mv0)

        self.f_layer = Linear(self.dim_s * 2, self.dim_s)
        self.dropout_layer = Dropout(dropout)
        self.p_layer = Linear(self.dim_s, 1)

        # DeepIRT 核心认知诊断层
        self.diff_layer = nn.Sequential(Linear(self.dim_s, 1), nn.Tanh())
        self.ability_layer = nn.Sequential(Linear(self.dim_s, 1), nn.Tanh())

        self.e_layer = Linear(self.dim_s, self.dim_s)
        self.a_layer = Linear(self.dim_s, self.dim_s)

    def forward(self, question, response, mask=None, interval_time=None, r_time=None, skill=None, **kwargs):
        # question: [bs, seq_len], response: [bs, seq_len]
        batch_size = question.shape[0]
        
        skill = skill if skill is not None else torch.zeros_like(question)
        
        # Interaction index
        xq = question + self.num_q * response
        xc = skill + self.num_c * response
        
        k_q = self.k_q_emb_layer(question)
        k_c = self.k_c_emb_layer(skill)
        k = torch.cat([k_q, k_c], dim=-1) # question embedding [bs, seq_len, dim_s]
        
        v_q = self.v_q_emb_layer(xq)
        v_c = self.v_c_emb_layer(xc)
        v = torch.cat([v_q, v_c], dim=-1) # interaction embedding [bs, seq_len, dim_s]
        
        # 初始记忆矩阵
        Mvt = self.Mv0.unsqueeze(0).repeat(batch_size, 1, 1) # [bs, size_m, dim_s]

        Mv = [Mvt]

        # 注意力权重矩阵计算 [bs, seq_len, size_m]
        w = torch.softmax(torch.matmul(k, self.Mk.T), dim=-1)

        # Erase & Add vectors
        e = torch.sigmoid(self.e_layer(v)) # [bs, seq_len, dim_s]
        a = torch.tanh(self.a_layer(v))    # [bs, seq_len, dim_s]

        # Memory Update Process (RNN-like loop)
        for et, at, wt in zip(
            e.permute(1, 0, 2), a.permute(1, 0, 2), w.permute(1, 0, 2)
        ):
            Mvt = Mvt * (1 - (wt.unsqueeze(-1) * et.unsqueeze(1))) + \
                (wt.unsqueeze(-1) * at.unsqueeze(1))
            
            # 如果有 mask，理论上在 pad 位应保持记忆不变，这里为了简单基线，依赖外部 loss mask 过滤
            Mv.append(Mvt)

        # Mv shape: [bs, seq_len+1, size_m, dim_s]
        Mv = torch.stack(Mv, dim=1)

        # Read Process (We use Mv[:, :-1] to predict next state equivalent to t)
        # wt is [bs, seq_len, size_m]
        f = torch.tanh(
            self.f_layer(
                torch.cat(
                    [
                        (w[:, 1:].unsqueeze(-1) * Mv[:, 1:-1]).sum(-2), # [bs, seq_len, dim_s]
                        k[:, 1:]
                    ],
                    dim=-1
                )
            )
        )
        
        # 可解释的 1PL-IRT 层
        stu_ability = self.ability_layer(self.dropout_layer(f)) # 学生能力
        que_diff = self.diff_layer(self.dropout_layer(k[:, 1:]))       # 题目难度

        # 输出 Logits，为了适配我们 Trainer 里的 BCEWithLogitsLoss
        # 原始公式为 p = torch.sigmoid(3.0 * stu_ability - que_diff) 
        # 所以 logit = 3.0 * stu_ability - que_diff
        logits = 3.0 * stu_ability - que_diff
        logits = logits.squeeze(-1)
        
        return logits
import torch.nn as nn

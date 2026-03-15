import torch
import torch.nn as nn
from torch.nn import Module, Parameter, Embedding, Linear, Dropout
from torch.nn.init import kaiming_normal_

class DKVMN(Module):
    def __init__(self, num_question, num_skill=None, dim_s=50, size_m=20, dropout=0.2):
        super(DKVMN, self).__init__()
        self.model_name = "dkvmn"
        self.num_question = num_question
        self.num_skill = num_skill if num_skill is not None else num_question
        self.dim_s = dim_s
        self.size_m = size_m

        # 降维：仅使用知识点(KC)级别进行嵌入
        self.k_c_emb = Embedding(self.num_skill, self.dim_s)
        
        self.Mk = Parameter(torch.Tensor(self.size_m, self.dim_s))
        self.Mv0 = Parameter(torch.Tensor(self.size_m, self.dim_s))

        kaiming_normal_(self.Mk)
        kaiming_normal_(self.Mv0)

        self.v_c_emb = Embedding(self.num_skill * 2, self.dim_s)

        self.f_layer = Linear(self.dim_s * 2, self.dim_s)
        self.dropout_layer = Dropout(dropout)
        self.p_layer = Linear(self.dim_s, 1)

        self.e_layer = Linear(self.dim_s, self.dim_s)
        self.a_layer = Linear(self.dim_s, self.dim_s)

    def forward(self, question, response, mask=None, interval_time=None, response_time=None, skill=None):
        skill = skill if skill is not None else torch.zeros_like(question)
            
        batch_size = question.shape[0]

        xc = skill + self.num_skill * response
        
        k = self.k_c_emb(skill)
        v = self.v_c_emb(xc)

        Mvt = self.Mv0.unsqueeze(0).repeat(batch_size, 1, 1)
        Mv = [Mvt]

        w = torch.softmax(torch.matmul(k, self.Mk.T), dim=-1)

        e = torch.sigmoid(self.e_layer(v))
        a = torch.tanh(self.a_layer(v))

        if mask is None:
            mask = torch.ones((batch_size, question.shape[1]), device=question.device)
            
        for t in range(question.shape[1]):
            et = e[:, t]
            at = a[:, t]
            wt = w[:, t]
            mt = mask[:, t].unsqueeze(-1).unsqueeze(-1)

            Mvt_next = Mvt * (1 - (wt.unsqueeze(-1) * et.unsqueeze(1))) + \
                (wt.unsqueeze(-1) * at.unsqueeze(1))

            Mvt = torch.where(mt == 1, Mvt_next, Mvt)
            Mv.append(Mvt)

        Mv = torch.stack(Mv, dim=1)

        f = torch.tanh(
            self.f_layer(
                torch.cat(
                    [
                        (w[:, 1:].unsqueeze(-1) * Mv[:, 1:-1]).sum(-2),
                        k[:, 1:]
                    ],
                    dim=-1
                )
            )
        )

        p = self.p_layer(self.dropout_layer(f))
        p = p.squeeze(-1)

        return p

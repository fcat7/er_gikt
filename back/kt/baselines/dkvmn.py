import torch
import torch.nn as nn
from torch.nn import Module, Parameter, Embedding, Linear, Dropout
from torch.nn.init import kaiming_normal_

class DKVMN(Module):
    def __init__(self, num_question, dim_s=50, size_m=20, dropout=0.2):
        super(DKVMN, self).__init__()
        self.model_name = "dkvmn"
        self.num_question = num_question
        self.dim_s = dim_s
        self.size_m = size_m

        self.k_emb_layer = Embedding(self.num_question, self.dim_s)
        self.Mk = Parameter(torch.Tensor(self.size_m, self.dim_s))
        self.Mv0 = Parameter(torch.Tensor(self.size_m, self.dim_s))

        kaiming_normal_(self.Mk)
        kaiming_normal_(self.Mv0)

        self.v_emb_layer = Embedding(self.num_question * 2, self.dim_s)

        self.f_layer = Linear(self.dim_s * 2, self.dim_s)
        self.dropout_layer = Dropout(dropout)
        self.p_layer = Linear(self.dim_s, 1)

        self.e_layer = Linear(self.dim_s, self.dim_s)
        self.a_layer = Linear(self.dim_s, self.dim_s)

    def forward(self, question, response, mask=None, interval_time=None, response_time=None):
        """
        question: [batch_size, seq_len]
        response: [batch_size, seq_len]
        """
        batch_size = question.shape[0]
        
        # Interaction embedding
        x = question + self.num_question * response
        k = self.k_emb_layer(question)
        v = self.v_emb_layer(x)
        
        Mvt = self.Mv0.unsqueeze(0).repeat(batch_size, 1, 1)
        Mv = [Mvt]

        w = torch.softmax(torch.matmul(k, self.Mk.T), dim=-1)

        # Write Process
        e = torch.sigmoid(self.e_layer(v))
        a = torch.tanh(self.a_layer(v))

        # Padding Mask for sequential update
        # mask shape: [batch_size, seq_len]
        if mask is None:
            mask = torch.ones((batch_size, question.shape[1]), device=question.device)

        for t in range(question.shape[1]):
            et = e[:, t]
            at = a[:, t]
            wt = w[:, t]
            mt = mask[:, t].unsqueeze(-1).unsqueeze(-1) # [batch_size, 1, 1]
            
            # 只有当 mt 为 1 时才更新内存
            Mvt_next = Mvt * (1 - (wt.unsqueeze(-1) * et.unsqueeze(1))) + \
                (wt.unsqueeze(-1) * at.unsqueeze(1))
            
            Mvt = torch.where(mt == 1, Mvt_next, Mvt)
            Mv.append(Mvt)

        Mv = torch.stack(Mv, dim=1)

        # Read Process
        # Mv[:, :-1] is the memory state before the update at time t
        # We want to predict the response at time t+1 using the memory state after the update at time t
        # So we use Mv[:, 1:] which is the memory state after the update at time t
        # and k[:, 1:] which is the question embedding at time t+1
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
        
        # Output logits (to be compatible with BCEWithLogitsLoss)
        p = self.p_layer(self.dropout_layer(f))
        p = p.squeeze(-1)
        
        return p

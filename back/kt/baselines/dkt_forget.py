import torch
import torch.nn as nn
from torch.nn import Module, Embedding, LSTM, Linear, Dropout
import math

class DKTForget(Module):
    def __init__(self, num_q, num_c, emb_size=64, dropout=0.1):
        super().__init__()
        self.model_name = "dkt_forget"
        self.num_q = num_q
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.outputs_logits = True

        self.time_project = nn.Sequential(
            nn.Linear(1, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size)
        )

        self.interaction_emb = Embedding(self.num_q * 2, self.emb_size)

        self.lstm_layer = LSTM(self.emb_size * 2, self.hidden_size, batch_first=True)
        self.dropout_layer = Dropout(dropout)

        # 改回和标准 DKT 完全一致的多重逻辑输出头，而不是受限参数的压缩线性映射
        self.out_layer = Linear(self.hidden_size, self.num_q)

    def forward(self, question, response, mask=None, interval_time=None, r_time=None, **kwargs):
        if interval_time is None:
            interval_time = torch.zeros_like(question, dtype=torch.float32)

        bs, seq_len = question.shape

        t_feat = interval_time.unsqueeze(-1)
        t_emb = self.time_project(t_feat)

        x = question + self.num_q * response
        x_emb = self.interaction_emb(x)

        theta_in = torch.cat([x_emb, t_emb], dim=-1)

        h, _ = self.lstm_layer(theta_in)
        h = self.dropout_layer(h)

        y_logits = self.out_layer(h) # [B, L, Q]

        # 用 h_t 去预测第 t+1 题
        pred_logits_full = y_logits[:, :-1, :] # [B, L-1, Q]
        next_questions = question[:, 1:]       # [B, L-1]

        gathered_logits = torch.gather(pred_logits_full, 2, next_questions.unsqueeze(2)).squeeze(2)

        final_logits = torch.zeros(bs, seq_len, device=question.device)
        final_logits[:, :-1] = gathered_logits

        # 在 Trainer 中同样使用 preds = y_hat[:, :-1]
        return final_logits

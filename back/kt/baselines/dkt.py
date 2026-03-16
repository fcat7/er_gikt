import torch
from torch import nn
from torch.nn import Module, Embedding, LSTM, Linear, Dropout

class DKT(Module):
    def __init__(self, num_question, num_skill=None, emb_dim=100, dropout=0.6):
        super(DKT, self).__init__()
        self.model_name = "dkt"
        self.num_question = num_question
        self.num_skill = num_skill if num_skill is not None else num_question
        self.emb_dim = emb_dim
        self.hidden_size = emb_dim
        
        # 降维：仅使用知识点(KC)级别进行嵌入和预测，防止在小数据集上因题目参数过多引起严重过拟合
        # +10 防御性编程：防止 1-based index 加上偏移量后，取到最大的 2*num_skill 导致越界崩溃
        self.c_emb = Embedding(self.num_skill * 2 + 10, self.emb_dim)

        self.lstm_layer = LSTM(self.emb_dim, self.hidden_size, batch_first=True)
        self.dropout_layer = Dropout(dropout)

        self.out_layer = Linear(self.hidden_size, self.num_skill)

    def forward(self, question, response, mask=None, interval_time=None, response_time=None, skill=None):
        skill = skill if skill is not None else torch.zeros_like(question)

        xc = skill + self.num_skill * response

        input_emb = self.c_emb(xc)

        h, _ = self.lstm_layer(input_emb)
        h = self.dropout_layer(h)

        y_logits = self.out_layer(h) # [B, L, num_skill]

        pred_logits_full = y_logits[:, :-1, :] # [B, L-1, num_skill]
        next_items = skill[:, 1:]       # [B, L-1] 使用 skill 预测下一个 skill

        gathered_logits = torch.gather(pred_logits_full, 2, next_items.unsqueeze(2)).squeeze(2)
        
        batch_size, seq_len = question.shape
        final_logits = torch.zeros(batch_size, seq_len, device=question.device)
        final_logits[:, :-1] = gathered_logits

        return final_logits

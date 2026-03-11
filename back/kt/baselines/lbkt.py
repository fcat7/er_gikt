import torch
import torch.nn as nn

class Layer(nn.Module):
    def __init__(self, dim_h, d, k, b):
        super(Layer, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(2 * dim_h, dim_h))
        self.bias = nn.Parameter(torch.zeros(1, dim_h))

        nn.init.xavier_normal_(self.weight)
        nn.init.xavier_normal_(self.bias)

        self.d = d
        self.k = k
        self.b = b

    def forward(self, factor, interact_emb, h):
        gate = self.k + (1 - self.k) / (1 + torch.exp(-self.d * (factor - self.b)))
        w = torch.cat([h, interact_emb], -1).matmul(self.weight) + self.bias
        w = nn.Sigmoid()(w * gate)

        return w

class LBKTcell(nn.Module):
    def __init__(self, dim_question, dim_h, dim_factor, num_concept, r=3, dropout=0.1, d=1.0, k=0.5, b=0.5):
        super(LBKTcell, self).__init__()
        
        self.dim_h = dim_h
        self.num_concept = num_concept
        self.dim_factor = dim_factor

        self.time_gain = Layer(dim_h, d, b, k)
        self.attempt_gain = Layer(dim_h, d, b, k)
        self.hint_gain = Layer(dim_h, d, b, k)

        self.time_weight = nn.Parameter(torch.Tensor(r, dim_h + 1, dim_h))
        nn.init.xavier_normal_(self.time_weight)

        self.attempt_weight = nn.Parameter(torch.Tensor(r, dim_h + 1, dim_h))
        nn.init.xavier_normal_(self.attempt_weight)

        self.hint_weight = nn.Parameter(torch.Tensor(r, dim_h + 1, dim_h))
        nn.init.xavier_normal_(self.hint_weight)

        self.Wf = nn.Parameter(torch.Tensor(1, r))
        nn.init.xavier_normal_(self.Wf)

        self.bias = nn.Parameter(torch.Tensor(1, dim_h))
        nn.init.xavier_normal_(self.bias)

        self.gate3 = nn.Linear(2 * dim_h + 3 * dim_factor, dim_h)
        torch.nn.init.xavier_normal_(self.gate3.weight)

        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(dim_question + dim_h, dim_h)
        torch.nn.init.xavier_normal_(self.output_layer.weight)
        self.sig = nn.Sigmoid()
        
        # To output logits instead of probabilities
        self.out_logit = nn.Linear(dim_h, 1)

    def forward(self, interact_emb, correlation_weight, topic_emb, time_factor, attempt_factor, hint_factor, h_pre):
        # bs *1 * memory_size , bs * memory_size * d_k
        h_pre_tilde = torch.squeeze(torch.bmm(correlation_weight.unsqueeze(1), h_pre), 1)
        
        # predict performance (logits)
        # h_pre_tilde: [bs, dim_h], topic_emb: [bs, dim_h]
        predict_logit = self.out_logit(h_pre_tilde).squeeze(-1)

        # characterize each behavior's effect
        time_gain = self.time_gain(time_factor, interact_emb, h_pre_tilde)
        attempt_gain = self.attempt_gain(attempt_factor, interact_emb, h_pre_tilde)
        hint_gain = self.hint_gain(hint_factor, interact_emb, h_pre_tilde)

        # capture the dependency among different behaviors
        pad = torch.ones_like(time_factor[:, :1])  # bs * 1
        time_gain1 = torch.cat([time_gain, pad], -1)  # bs * num_units + 1
        attempt_gain1 = torch.cat([attempt_gain, pad], -1)
        hint_gain1 = torch.cat([hint_gain, pad], -1)
        
        # bs * r  *num_units: bs * num_units + 1 ,r * num_units + 1 *num_units
        fusion_time = torch.matmul(time_gain1, self.time_weight)
        fusion_attempt = torch.matmul(attempt_gain1, self.attempt_weight)
        fusion_hint = torch.matmul(hint_gain1, self.hint_weight)
        fusion_all = fusion_time * fusion_attempt * fusion_hint
        
        # 1 * r, bs * r * num_units -> bs * 1 * num_units -> bs * num_units
        fusion_all = torch.matmul(self.Wf, fusion_all.permute(1, 0, 2)).squeeze(1) + self.bias
        learning_gain = torch.relu(fusion_all)

        LG = torch.matmul(correlation_weight.unsqueeze(-1), learning_gain.unsqueeze(1))

        # forget effect
        forget_gate = self.gate3(torch.cat([h_pre, interact_emb.unsqueeze(1).repeat(1, self.num_concept, 1),
                                            time_factor.unsqueeze(1).repeat(1, self.num_concept, self.dim_factor),
                                            attempt_factor.unsqueeze(1).repeat(1, self.num_concept, self.dim_factor),
                                            hint_factor.unsqueeze(1).repeat(1, self.num_concept, self.dim_factor)], -1))
        LG = self.dropout(LG)
        h = h_pre * self.sig(forget_gate) + LG

        return predict_logit, h

class LBKT(nn.Module):
    def __init__(self, num_question, num_concept, qs_table, dim_h=100, dim_factor=1):
        super(LBKT, self).__init__()
        self.model_name = "lbkt"
        self.num_question = num_question
        self.num_concept = num_concept
        self.dim_h = dim_h
        
        # Normalize qs_table to be correlation_weight
        # qs_table: [num_question, num_concept]
        qs_sum = qs_table.sum(dim=1, keepdim=True)
        qs_sum = torch.where(qs_sum == 0, torch.ones_like(qs_sum), qs_sum)
        self.q_matrix = nn.Parameter(qs_table / qs_sum, requires_grad=False)

        self.question_emb = nn.Embedding(num_question, dim_h)
        self.correctness_emb = nn.Embedding(2, dim_h)
        
        self.input_layer = nn.Linear(dim_h * 2, dim_h)
        torch.nn.init.xavier_normal_(self.input_layer.weight)
        
        self.lbkt_cell = LBKTcell(dim_question=dim_h, dim_h=dim_h, dim_factor=dim_factor, num_concept=num_concept)
        
        self.init_h = nn.Parameter(torch.Tensor(num_concept, dim_h))
        nn.init.xavier_normal_(self.init_h)

    def forward(self, question, response, mask=None, interval_time=None, response_time=None):
        batch_size, seq_len = question.size(0), question.size(1)
        
        question_emb = self.question_emb(question)
        correctness_emb = self.correctness_emb(response)

        correlation_weight = self.q_matrix[question]
        acts_emb = torch.relu(self.input_layer(torch.cat([question_emb, correctness_emb], -1)))

        # Use interval_time as time_factor, response_time as attempt_factor, and zeros for hint_factor
        if interval_time is None:
            interval_time = torch.zeros_like(question, dtype=torch.float)
        if response_time is None:
            response_time = torch.zeros_like(question, dtype=torch.float)
            
        time_factor_seq = interval_time.unsqueeze(-1).float()
        attempt_factor_seq = response_time.unsqueeze(-1).float()
        hint_factor_seq = torch.zeros_like(time_factor_seq).float()

        h_init = self.init_h.unsqueeze(0).repeat(batch_size, 1, 1)
        h_pre = h_init
        
        predict_logits = []
        
        for t in range(seq_len):
            pred, h = self.lbkt_cell(acts_emb[:, t], correlation_weight[:, t], question_emb[:, t],
                                     time_factor_seq[:, t], attempt_factor_seq[:, t], hint_factor_seq[:, t], h_pre)
            
            # 如果提供了 mask，则在 padding 处保持状态不变
            if mask is not None:
                mt = mask[:, t].unsqueeze(-1).unsqueeze(-1) # [batch_size, 1, 1]
                h = torch.where(mt == 1, h, h_pre)
                
            h_pre = h
            predict_logits.append(pred)

        predict_logits = torch.stack(predict_logits, dim=1)
        return predict_logits

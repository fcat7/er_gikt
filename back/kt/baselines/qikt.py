import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, num_layer, dim_in, dim_out, dropout):
        super().__init__()

        self.linear_list = nn.ModuleList([
            nn.Linear(dim_in, dim_in)
            for _ in range(num_layer)
        ])
        self.dropout = nn.Dropout(p=dropout)
        self.out = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        for lin in self.linear_list:
            x = torch.relu(lin(x))
        return self.out(self.dropout(x))

class QIKT(nn.Module):
    def __init__(self, num_question, num_concept, qs_table, dim_emb=100, num_rnn_layer=1, num_mlp_layer=1, dropout=0.1):
        super().__init__()
        self.model_name = "qikt"
        self.num_question = num_question
        self.num_concept = num_concept
        self.dim_emb = dim_emb
        
        # Convert qs_table to q2c_transfer_table and q2c_mask_table
        # qs_table is [num_question, num_concept]
        max_concepts = qs_table.sum(dim=1).max().item()
        if max_concepts == 0:
            max_concepts = 1
            
        self.q2c_transfer_table = torch.zeros((num_question, int(max_concepts)), dtype=torch.long)
        self.q2c_mask_table = torch.zeros((num_question, int(max_concepts)), dtype=torch.float)
        
        for q in range(num_question):
            concepts = torch.nonzero(qs_table[q]).squeeze(-1)
            if len(concepts) > 0:
                self.q2c_transfer_table[q, :len(concepts)] = concepts
                self.q2c_mask_table[q, :len(concepts)] = 1.0
            else:
                self.q2c_transfer_table[q, 0] = 0
                self.q2c_mask_table[q, 0] = 0.0 # Masked out
                
        self.q2c_transfer_table = nn.Parameter(self.q2c_transfer_table, requires_grad=False)
        self.q2c_mask_table = nn.Parameter(self.q2c_mask_table, requires_grad=False)

        self.question_emb = nn.Embedding(num_question, dim_emb)
        self.concept_emb = nn.Embedding(num_concept, dim_emb)

        self.rnn_layer4question = nn.LSTM(dim_emb * 4, dim_emb, batch_first=True, num_layers=num_rnn_layer)
        self.rnn_layer4concept = nn.LSTM(dim_emb * 2, dim_emb, batch_first=True, num_layers=num_rnn_layer)
        
        self.dropout_layer = nn.Dropout(dropout)
        self.predict_layer4q_next = MLP(num_mlp_layer, dim_emb * 3, 1, dropout)
        self.predict_layer4q_all = MLP(num_mlp_layer, dim_emb, num_question, dropout)
        self.predict_layer4c_next = MLP(num_mlp_layer, dim_emb * 3, num_concept, dropout)
        self.predict_layer4c_all = MLP(num_mlp_layer, dim_emb, num_concept, dropout)
        self.que_discrimination_layer = MLP(num_mlp_layer, dim_emb * 2, 1, dropout)

    def get_concept_emb_fused(self, question_seq):
        # question_seq: [bs, seq_len]
        q2c_seq = self.q2c_transfer_table[question_seq] # [bs, seq_len, max_concepts]
        q2c_mask = self.q2c_mask_table[question_seq] # [bs, seq_len, max_concepts]
        
        c_emb = self.concept_emb(q2c_seq) # [bs, seq_len, max_concepts, dim_emb]
        c_emb = c_emb * q2c_mask.unsqueeze(-1)
        
        sum_mask = q2c_mask.sum(dim=-1, keepdim=True)
        sum_mask = torch.where(sum_mask == 0, torch.ones_like(sum_mask), sum_mask)
        
        c_emb_fused = c_emb.sum(dim=-2) / sum_mask # [bs, seq_len, dim_emb]
        return c_emb_fused

    def forward(self, question, response, mask=None, interval_time=None, response_time=None):
        batch_size, seq_len = question.shape
        
        concept_emb = self.get_concept_emb_fused(question)
        question_emb = self.question_emb(question)
        qc_emb = torch.cat((question_emb, concept_emb), dim=-1)
        
        qca_emb = torch.cat([
            qc_emb.mul((1 - response).unsqueeze(-1).repeat(1, 1, self.dim_emb * 2)),
            qc_emb.mul(response.unsqueeze(-1).repeat(1, 1, self.dim_emb * 2))
        ], dim=-1)
        
        ca_emb = torch.cat([
            concept_emb.mul((1 - response).unsqueeze(-1).repeat(1, 1, self.dim_emb)),
            concept_emb.mul(response.unsqueeze(-1).repeat(1, 1, self.dim_emb))
        ], dim=-1)

        self.rnn_layer4question.flatten_parameters()
        self.rnn_layer4concept.flatten_parameters()
        
        # We need to predict for the next step, so we use up to t-1 to predict t
        # But to keep output shape [bs, seq_len], we pad the first step
        
        latent_question, _ = self.rnn_layer4question(qca_emb[:, :-1, :])
        latent_concept, _ = self.rnn_layer4concept(ca_emb[:, :-1, :])
        
        latent_question = self.dropout_layer(latent_question)
        latent_concept = self.dropout_layer(latent_concept)

        # Predict next question
        predict_score_q_next = self.predict_layer4q_next(
            torch.cat((qc_emb[:, 1:, :], latent_question), dim=-1)
        ).squeeze(-1)
        
        # Return logits for the next question [batch_size, seq_len - 1]
        return predict_score_q_next

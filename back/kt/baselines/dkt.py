import torch
from torch import nn
from torch.nn import Module, Embedding, LSTM, Linear, Dropout

class DKT(Module):
    def __init__(self, num_question, num_skill=None, emb_dim=100, dropout=0.1):
        """
        Deep Knowledge Tracing (DKT) implementation.
        Reference: https://arxiv.org/abs/1506.05908
        """
        super(DKT, self).__init__()
        self.model_name = "dkt"
        self.num_question = num_question
        self.num_skill = num_skill
        self.emb_dim = emb_dim
        self.hidden_size = emb_dim
        
        # In standard DKT on datasets with skills, we predict Skill/Concept instead of Question to avoid massive sparsity and overfitting.
        # [紧急修改]: 根据要求，强制使用题目级别(Question-level)。如需恢复技能级别(Skill-level)，将 force_question_level 改为 False
        force_question_level = False
        self.use_skill = (not force_question_level) and (self.num_skill is not None and self.num_skill > 0 and self.num_skill < self.num_question)
        self.num_concepts = self.num_skill if self.use_skill else self.num_question

        # DKT Input: Interaction Embedding (Concept_ID + Correctness)
        self.interaction_emb = Embedding(self.num_concepts * 2, self.emb_dim)

        self.lstm_layer = LSTM(self.emb_dim, self.hidden_size, batch_first=True)
        self.dropout_layer = Dropout(dropout)

        # Output layer predicts the probability of answering correctly for EACH concept
        self.out_layer = Linear(self.hidden_size, self.num_concepts)

    def forward(self, question, response, mask=None, interval_time=None, response_time=None, skill=None):
        """
        Args:
            question: [batch_size, seq_len]
            response: [batch_size, seq_len]
            skill: [batch_size, seq_len] Optional skill sequences
            mask: [batch_size, seq_len] (Padding mask)
        """
        inp = skill if (self.use_skill and skill is not None) else question

        # x = concept + num_concepts * response
        x = inp + self.num_concepts * response 

        input_emb = self.interaction_emb(x)

        h, _ = self.lstm_layer(input_emb)
        h = self.dropout_layer(h)

        y_logits = self.out_layer(h) # [B, L, C]
        
        # Prediction logic (Shifted for Next-Item Prediction)
        # y_logits[:, t] predicts all items at t+1. 
        # We gather only the logit for the actual next concept inp[:, t+1].

        # Next-item prediction
        pred_logits_full = y_logits[:, :-1, :] # [B, L-1, C]
        next_items = inp[:, 1:]       # [B, L-1]

        # Gather specific concept logits
        # gathered_logits: [B, L-1]
        gathered_logits = torch.gather(pred_logits_full, 2, next_items.unsqueeze(2)).squeeze(2)
        
        batch_size, seq_len = question.shape
        final_logits = torch.zeros(batch_size, seq_len, device=question.device)
        final_logits[:, :-1] = gathered_logits

        return final_logits
        final_logits[:, :-1] = gathered_logits
        
        return final_logits

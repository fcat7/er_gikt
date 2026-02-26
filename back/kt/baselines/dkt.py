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
        
        # DKT Input: Interaction Embedding (Question_ID + Correctness)
        self.interaction_emb = Embedding(self.num_question * 2, self.emb_dim)

        self.lstm_layer = LSTM(self.emb_dim, self.hidden_size, batch_first=True)
        self.dropout_layer = Dropout(dropout)
        
        # Output layer predicts the probability of answering correctly for EACH question
        self.out_layer = Linear(self.hidden_size, self.num_question)
        
    def forward(self, question, response, mask=None, interval_time=None, response_time=None):
        """
        Args:
            question: [batch_size, seq_len]
            response: [batch_size, seq_len]
            mask: [batch_size, seq_len] (Padding mask)
        """
        # x = question + num_question * response
        x = question + self.num_question * response 
        
        input_emb = self.interaction_emb(x)
        
        h, _ = self.lstm_layer(input_emb)
        h = self.dropout_layer(h)
        
        y_logits = self.out_layer(h) # [B, L, Q]
        
        # Prediction logic (Shifted for Next-Item Prediction)
        # y_logits[:, t] predicts all items at t+1. 
        # We gather only the logit for the actual next question q[:, t+1].
        
        # Next-item prediction
        pred_logits_full = y_logits[:, :-1, :] # [B, L-1, Q]
        next_questions = question[:, 1:]       # [B, L-1]
        
        # Gather specific question logits
        # gathered_logits: [B, L-1]
        gathered_logits = torch.gather(pred_logits_full, 2, next_questions.unsqueeze(2)).squeeze(2)
        
        # Compatible with BCEWithLogitsLoss: return logits
        # We need to return [B, L-1] or [B, L]. To keep consistent with Trainer:
        # We pad the last position with 0 (which will be masked out by the trainer anyway)
        batch_size, seq_len = question.shape
        final_logits = torch.zeros(batch_size, seq_len, device=question.device)
        final_logits[:, :-1] = gathered_logits
        
        return final_logits

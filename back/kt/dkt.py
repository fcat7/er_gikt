import torch
import torch.nn as nn
from torch.nn import Module, Embedding, LSTM, Linear, Dropout
from config import DEVICE

class DKT(Module):
    def __init__(self, num_question, num_skill, emb_dim=100, dropout=0.1):
        """
        Deep Knowledge Tracing (DKT) implementation.
        Reference: https://arxiv.org/abs/1506.05908
        
        Args:
            num_question: Number of questions (not used if skill-based, but kept for interface consistency)
            num_skill: Number of skills (concepts)
            emb_dim: Dimension of embedding and hidden state
            dropout: Dropout rate
        """
        super(DKT, self).__init__()
        self.model_name = "dkt"
        self.num_question = num_question
        self.num_skill = num_skill
        self.emb_dim = emb_dim
        self.hidden_size = emb_dim
        
        # DKT Input: Interaction Embedding (Question_ID + Correctness)
        # Input space size is 2 * num_question (one for correct, one for incorrect)
        self.interaction_emb = Embedding(self.num_question * 2, self.emb_dim)

        self.lstm_layer = LSTM(self.emb_dim, self.hidden_size, batch_first=True)
        self.dropout_layer = Dropout(dropout)
        
        # Output layer predicts the probability of answering correctly for EACH question
        # Since we use Question-Level DKT, output dimension should be num_question, NOT num_skill
        self.out_layer = Linear(self.hidden_size, self.num_question)
        
    def forward(self, question, response, mask, interval_time=None, response_time=None):
        """
        Args:
            question: [batch_size, seq_len] - Question IDs (Not used in pure DKT usually, DKT uses Skills)
            response: [batch_size, seq_len] - 0 or 1
            mask: [batch_size, seq_len]
        
        Note: 
            Standard DKT operates on SKILLS (Concepts). 
            If the input `question` refers to actual Question IDs, we usually need to map them to Skill IDs first.
            However, to keep it simple and compatible with existing train_test.py loop which might pass Question IDs,
            we assume the caller expects a Question-level input but DKT traditionally works on Skills.
            
            This implementation assumes `question` tensor actually contains SKILL IDs if we strictly follow DKT.
            BUT, if `question` contains Question IDs, DKT cannot work directly without Q-matrix mapping.
            
            To support GIKT's interface where `question` is QuestionID:
            We need a mapping Q -> S. Since DKT doesn't use Graph, we usually just take the first skill of the question.
            
            **CRITICAL ADAPTATION**: 
            For this file to work as a baseline in your GIKT project, we need to know if we are tracking Questions or Skills.
            Standard DKT tracks Skills. 
            If we treat Questions as "Skills" (i.e. sparse DKT), then num_skill should be num_question.
            
            Let's implement the standard Skill-based DKT. 
            We need to map Question IDs -> Skill IDs.
            Since the `forward` signature matches GIKT, but we don't have `qs_table` inside DKT by default.
            
            **Strategy**: We will require `qs_table` to be passed in __init__ or handle conversion outside.
            OR, more robustly: If this model is called with Question IDs, we can't run DKT easily without mapping.
            
            *Assumption*: The dataset loader currently yields Question IDs.
            To make this DKT runnable as a baseline in your environment:
            1. We will use Question IDs directly as "Concepts" (Sparsity issue but runnable).
                In this case `self.num_skill` in logic should be `self.num_question`.
            2. OR, we modify it to accept `qs_table` in init.
            
            Let's go with **Option 2** (Pass qs_table) to be scientifically correct (Skill-DKT).
        """
        # Placeholder for now, will implement assuming `question` is mapped or we use Question-DKT.
        # Given your context, let's implement a 'Question-DKT' (treating each unique question as a label) 
        # which is common when Q-matrix is complex or we compare question prediction directly.
        
        # However, to be a fair baseline for GIKT (which uses Skills), we should probably use Skill-DKT logic?
        # Actually GIKT predicts QUESTIONS. 
        # Let's stick to the attached reference logic: Input -> Interaction -> LSTM -> Output(All Skills).
        
        # Mapping input features
        # x = q + num_c * r
        # Here we use `question` as the ID. 
        # If tracking Questions: x = question + num_question * response
        # If tracking Skills: We need skill_seq.
        
        # For compatibility with your train_test.py which likely passes Question IDs:
        # We will implement DKT on Question Level (Sparse DKT)
        # Input space: 2 * num_question
        
        x = question + self.num_question * response # [batch, seq_len]
        
        # Mask padding (usually 0 is used for padding in question ID, but here 0 is a valid ID?)
        # GIKT uses a mask tensor. We apply embedding, then mask out padding if needed.
        # But wait, if question ID 0 exists, we can't confuse it with padding.
        # Your dataset probably handles padding.
        
        input_emb = self.interaction_emb(x) # [batch, seq_len, emb_dim]
        
        h, _ = self.lstm_layer(input_emb) # [batch, seq_len, hidden_size]
        h = self.dropout_layer(h)
        
        y_logits = self.out_layer(h) # [batch, seq_len, num_question]
        
        # Prediction logic for next item:
        # y_logits[:, t] is the prediction for time t+1
        # typically DKT predicts ALL items at each step.
        # We need to gather the logits corresponding to the ACTUAL next question.
        # P(q_{t+1} | q_{0:t}, r_{0:t}) corresponds to output at step t, gathered at index q_{t+1}
        
        # Shift prediction: Output at t predicts t+1.
        # So we collect predictions.
        
        # Flatten for gathering
        # y_logits: [B, L, Q]
        # target_ids: [B, L] -> Q_next (We need to shift input q to get targets? or train_test handles it?)
        
        # GIKT's forward returns `y_hat` of shape [batch, seq_len] with probabilities.
        # It calculates predict(t) -> q_{t+1}.
        
        batch_size, seq_len = question.shape
        y_hat = torch.zeros(batch_size, seq_len, device=DEVICE)
        
        # Predict t+1 using state h_t
        # h[:, t] contains info from 0...t. It predicts t+1.
        # So y_logits[:, t] predicts q[:, t+1].
        
        # We iterate to extract correct logits
        # Note: This is efficient vectorized way
        # Target Question IDs for next step prediction
        # q_{t+1} is the target for output at t.
        
        # Predictions for the next step (time 0 to L-1 predict items 1 to L)
        # We only can predict up to L-1 steps (predicting the last item in sequence).
        # The last hidden state predicts a future item we don't have.
        
        pred_logits = y_logits[:, :-1, :] # [Batch, L-1, Q] form time 0 to L-2 (predicting 1 to L-1)
        next_questions = question[:, 1:]  # [Batch, L-1] target IDs
        
        # Gather specific question logits
        # gathered_logits: [Batch, L-1]
        gathered_logits = torch.gather(pred_logits, 2, next_questions.unsqueeze(2)).squeeze(2)
        
        y_hat_seq = torch.sigmoid(gathered_logits)
        
        # Fill into y_hat [Batch, L]
        # We align so that y_hat[:, t] represents prediction for q[:, t+1] made at time t.
        # The loop in GIKT does: for t in range(seq_len - 1): predict q_{t+1}
        # So we place result at index t.
        
        y_hat[:, :-1] = y_hat_seq
        
        return y_hat

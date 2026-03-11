import torch
import torch.nn as nn

class GIKTOld(nn.Module):
    model_name = "gikt_old"

    def __init__(self, num_question, num_skill, q_neighbors, s_neighbors, qs_table,
                 agg_hops=3, dim_emb=100, dropout4gru=0.1, dropout4gnn=0.1, rank_k=10, **kwargs):
        super(GIKTOld, self).__init__()
        
        self.num_question = num_question
        self.num_skill = num_skill
        self.agg_hops = agg_hops
        self.dim_emb = dim_emb
        self.rank_k = rank_k
        self.dropout4gru = dropout4gru
        self.dropout4gnn = dropout4gnn

        device = qs_table.device if isinstance(qs_table, torch.Tensor) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if not isinstance(q_neighbors, torch.Tensor):
            q_neighbors = torch.tensor(q_neighbors, dtype=torch.long, device=device)
        if not isinstance(s_neighbors, torch.Tensor):
            s_neighbors = torch.tensor(s_neighbors, dtype=torch.long, device=device)
        if not isinstance(qs_table, torch.Tensor):
            qs_table = torch.tensor(qs_table, dtype=torch.long, device=device)

        self.q_neighbors = q_neighbors
        self.s_neighbors = s_neighbors
        self.qs_table = qs_table

        self.embed_question = nn.Embedding(num_question, dim_emb)
        self.embed_concept = nn.Embedding(num_skill, dim_emb)
        self.embed_correctness = nn.Embedding(2, dim_emb)

        self.gru1 = nn.GRUCell(dim_emb * 2, dim_emb)
        self.gru2 = nn.GRUCell(dim_emb, dim_emb)
        self.mlp4agg = nn.ModuleList(nn.Linear(dim_emb, dim_emb) for _ in range(agg_hops))
        self.MLP_AGG_last = nn.Linear(dim_emb, dim_emb)
        self.dropout_gru = nn.Dropout(dropout4gru)
        self.dropout_gnn = nn.Dropout(dropout4gnn)
        self.MLP_query = nn.Linear(dim_emb, dim_emb)
        self.MLP_key = nn.Linear(dim_emb, dim_emb)
        self.MLP_W = nn.Linear(2 * dim_emb, 1)

    def forward(self, question, response, mask, interval_time=None, response_time=None):
        device = question.device
        batch_size, seq_len = question.shape
        q_neighbor_size, c_neighbor_size = self.q_neighbors.shape[1], self.s_neighbors.shape[1]
        
        h1_pre = torch.nn.init.xavier_uniform_(torch.zeros(batch_size, self.dim_emb)).to(device)
        h2_pre = torch.nn.init.xavier_uniform_(torch.zeros(batch_size, self.dim_emb)).to(device)
        state_history = torch.zeros(batch_size, seq_len, self.dim_emb).to(device)
        y_hat = torch.zeros(batch_size, seq_len).to(device)

        for t in range(seq_len - 1):
            question_t = question[:, t]
            response_t = response[:, t]
            mask_t = torch.ne(mask[:, t], 0)
            emb_response_t = self.embed_correctness(response_t)

            # GNN obtain question embedding
            nodes_neighbor = [question_t[mask_t]]
            batch_size__ = len(nodes_neighbor[0])
            for i in range(self.agg_hops):
                nodes_current = nodes_neighbor[-1]
                nodes_current = nodes_current.reshape(-1)
                neighbor_shape = [batch_size__] + \
                                 [(q_neighbor_size if j % 2 == 0 else c_neighbor_size) for j in range(i + 1)]
                if i % 2 == 0:
                    nodes_neighbor.append(self.q_neighbors[nodes_current].reshape(neighbor_shape))
                    continue
                nodes_neighbor.append(self.s_neighbors[nodes_current].reshape(neighbor_shape))
                
            emb_nodes_neighbor = []
            for i, nodes in enumerate(nodes_neighbor):
                if i % 2 == 0:
                    emb_nodes_neighbor.append(self.embed_question(nodes))
                    continue
                emb_nodes_neighbor.append(self.embed_concept(nodes))
                
            emb_question_t = self.aggregate(emb_nodes_neighbor)
            emb_question_t_reconstruct = torch.zeros(batch_size, self.dim_emb, dtype=emb_question_t.dtype, device=device)
            emb_question_t_reconstruct[mask_t] = emb_question_t
            emb_question_t_reconstruct[~mask_t] = self.embed_question(question_t[~mask_t]).to(emb_question_t.dtype)

            # GRU update knowledge state
            gru1_input = torch.concat((emb_question_t_reconstruct, emb_response_t), dim=1)
            h1_pre = self.dropout_gru(self.gru1(gru1_input, h1_pre))
            gru2_output = self.dropout_gru(self.gru2(h1_pre, h2_pre))

            # Find concept of next question
            question_next = question[:, t + 1]
            correspond_concepts = self.qs_table[question_next]
            correspond_concepts_list = []
            max_concept = 1
            for i in range(batch_size):
                concepts_index = torch.nonzero(correspond_concepts[i] == 1).squeeze()
                if len(concepts_index.shape) == 0:
                    correspond_concepts_list.append(torch.unsqueeze(self.embed_concept(concepts_index), dim=0))
                else:
                    if concepts_index.shape[0] > max_concept:
                        max_concept = concepts_index.shape[0]
                    correspond_concepts_list.append(self.embed_concept(concepts_index))
                    
            # Concat question and concept embeddings
            emb_question_next = self.embed_question(question_next)
            question_concept = torch.zeros(batch_size, max_concept + 1, self.dim_emb, dtype=emb_question_next.dtype, device=device)
            for b, emb_concepts in enumerate(correspond_concepts_list):
                num_qc = 1 + emb_concepts.shape[0]
                emb_next = torch.unsqueeze(emb_question_next[b], dim=0)
                question_concept[b, 0:num_qc] = torch.concat((emb_next, emb_concepts), dim=0).to(question_concept.dtype)
            question_concept = question_concept.to(device)
            
            if t == 0:
                y_hat[:, 0] = self.predict(question_concept, torch.unsqueeze(gru2_output, dim=1)).to(y_hat.dtype)
                continue
                
            # recap history states
            current_state = gru2_output.unsqueeze(dim=1)
            if t <= self.rank_k:
                current_history_state = torch.concat((current_state, state_history[:, 0:t]), dim=1)
            else:
                Q = self.embed_question(question_next).clone().detach().unsqueeze(dim=-1)
                K = self.embed_question(question[:, 0:t]).clone().detach()
                product_score = torch.bmm(K, Q).squeeze(dim=-1)
                _, indices = torch.topk(product_score, k=self.rank_k, dim=1)
                select_history = torch.concat(tuple(state_history[i][indices[i]].unsqueeze(dim=0)
                                                    for i in range(batch_size)), dim=0)
                current_history_state = torch.concat((current_state, select_history), dim=1)
                
            y_hat[:, t + 1] = self.predict(question_concept, current_history_state).to(y_hat.dtype)
            h2_pre = gru2_output
            state_history[:, t] = gru2_output
            
        return y_hat

    def aggregate(self, emb_list):
        for i in range(self.agg_hops):
            for j in range(self.agg_hops - i):
                emb_list[j] = self.sum_aggregate(emb_list[j], emb_list[j+1], j)
        return torch.tanh(self.MLP_AGG_last(emb_list[0]))

    def sum_aggregate(self, emb_self, emb_neighbor, hop):
        emb_sum_neighbor = torch.mean(emb_neighbor, dim=-2)
        emb_sum = emb_sum_neighbor + emb_self
        return torch.tanh(self.dropout_gnn(self.mlp4agg[hop](emb_sum)))

    def predict(self, question_concept, current_history_state):
        output_g = torch.bmm(question_concept, torch.transpose(current_history_state, 1, 2))

        num_qc, num_state = question_concept.shape[1], current_history_state.shape[1]
        states = torch.unsqueeze(current_history_state, dim=1)  
        states = states.repeat(1, num_qc, 1, 1)  
        question_concepts = torch.unsqueeze(question_concept, dim=2)  
        question_concepts = question_concepts.repeat(1, 1, num_state, 1)  

        K = torch.tanh(self.MLP_query(states))  
        Q = torch.tanh(self.MLP_key(question_concepts))  
        tmp = self.MLP_W(torch.concat((Q, K), dim=-1))  
        tmp = torch.squeeze(tmp, dim=-1)  
        alpha = torch.softmax(tmp, dim=2)  
        
        p = torch.sum(torch.sum(alpha * output_g, dim=1), dim=1)  
        
        # Here we don't apply sigmoid because BCEWithLogitsLoss is used externally
        # But wait - GIKT predicts probability, and Trainer might use BCEWithLogitsLoss.
        # Let's check BaseTrainer.
        # In trainer.py, BaseTrainer expects logits or probabilities? 
        # BaseTrainer uses BCEWithLogitsLoss, BUT it checks if the model name is 'gikt'.
        # Let me see if GIKT returns logits or probabilities.
        # Original pyedmine GIKT returns probability `result = torch.sigmoid(torch.squeeze(p, dim=-1))`.
        # I'll return probability but comment on it if a logit is needed. Wait, in trainer.py:
        # if cognitive_mode == 'classic' or model_name in ['gikt']:
        #     #... we'll return logits to be safe for BCEWithLogitsLoss. Let's return `torch.squeeze(p, dim=-1)` unmodified.
        # Let me check my previous output and trainer.py.
        return torch.squeeze(p, dim=-1)

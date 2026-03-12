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

        # Optimization: Pre-process qs_table mappings into padded dense tensors to prevent expensive torch.nonzero loops
        max_concepts = int(self.qs_table.sum(dim=1).max().item())
        self.max_concepts = max(1, max_concepts)
        self.q_concept_idx = torch.zeros(self.num_question, self.max_concepts, dtype=torch.long, device=device)
        self.q_concept_mask = torch.zeros(self.num_question, self.max_concepts, dtype=torch.bool, device=device)
        qs_sum = (self.qs_table > 0).sum(dim=1)
        for q in range(self.num_question):
            n = int(qs_sum[q].item())
            if n > 0:
                idx = torch.nonzero(self.qs_table[q] > 0).squeeze(-1)
                self.q_concept_idx[q, :n] = idx
                self.q_concept_mask[q, :n] = True

    def forward(self, question, response, mask, interval_time=None, response_time=None):
        device = question.device
        batch_size, seq_len = question.shape
        q_neighbor_size, c_neighbor_size = self.q_neighbors.shape[1], self.s_neighbors.shape[1]
        
        # Optimization 1: Pre-calculate GNN embeddings for the WHOLE valid sequence simultaneously
        q_flat = question.view(-1)
        m_flat = mask.view(-1).bool()
        
        emb_reconstruct_flat = self.embed_question(q_flat)
        valid_q = q_flat[m_flat]
        
        if valid_q.numel() > 0:
            nodes_neighbor = [valid_q]
            batch_size__ = len(nodes_neighbor[0])
            for i in range(self.agg_hops):
                nodes_current = nodes_neighbor[-1].reshape(-1)
                neighbor_shape = [batch_size__] + \
                                 [(q_neighbor_size if j % 2 == 0 else c_neighbor_size) for j in range(i + 1)]
                if i % 2 == 0:
                    nodes_neighbor.append(self.q_neighbors[nodes_current].reshape(neighbor_shape))
                else:
                    nodes_neighbor.append(self.s_neighbors[nodes_current].reshape(neighbor_shape))
                    
            emb_nodes_neighbor = []
            for i, nodes in enumerate(nodes_neighbor):
                if i % 2 == 0:
                    emb_nodes_neighbor.append(self.embed_question(nodes))
                else:
                    emb_nodes_neighbor.append(self.embed_concept(nodes))
                    
            emb_valid = self.aggregate(emb_nodes_neighbor)
            emb_reconstruct_flat = emb_reconstruct_flat.clone()
            emb_reconstruct_flat[m_flat] = emb_valid.to(emb_reconstruct_flat.dtype)
            
        emb_question_reconstruct = emb_reconstruct_flat.view(batch_size, seq_len, self.dim_emb)
        all_emb_question = self.embed_question(question)
        
        h1_pre = torch.nn.init.xavier_uniform_(torch.zeros(batch_size, self.dim_emb)).to(device)
        h2_pre = torch.nn.init.xavier_uniform_(torch.zeros(batch_size, self.dim_emb)).to(device)
        state_history = torch.zeros(batch_size, seq_len, self.dim_emb).to(device)
        y_hat = torch.zeros(batch_size, seq_len).to(device)

        for t in range(seq_len - 1):
            response_t = response[:, t]
            emb_response_t = self.embed_correctness(response_t)

            # GRU update knowledge state (No GNN loop needed anymore)
            emb_question_t_reconstruct = emb_question_reconstruct[:, t]
            gru1_input = torch.concat((emb_question_t_reconstruct, emb_response_t), dim=1)
            h1_pre = self.dropout_gru(self.gru1(gru1_input, h1_pre))
            gru2_output = self.dropout_gru(self.gru2(h1_pre, h2_pre))

            # Optimization 2: Find concept of next question using O(1) vectorized mapping
            question_next = question[:, t + 1]
            c_idx = self.q_concept_idx[question_next]
            c_mask = self.q_concept_mask[question_next]
            max_c_batch = max(1, int(c_mask.sum(1).max().item()))
            
            c_idx = c_idx[:, :max_c_batch]
            c_mask = c_mask[:, :max_c_batch]
            
            emb_concepts = self.embed_concept(c_idx)
            emb_concepts = emb_concepts * c_mask.unsqueeze(-1).to(emb_concepts.dtype)
            
            # Concat question and concept embeddings
            emb_next = all_emb_question[:, t + 1].unsqueeze(1)
            question_concept = torch.concat((emb_next, emb_concepts), dim=1)
            
            if t == 0:
                y_hat[:, 0] = self.predict(question_concept, torch.unsqueeze(gru2_output, dim=1)).to(y_hat.dtype)
                continue
                
            # recap history states
            current_state = gru2_output.unsqueeze(dim=1)
            if t <= self.rank_k:
                current_history_state = torch.concat((current_state, state_history[:, 0:t]), dim=1)
            else:
                Q = all_emb_question[:, t + 1].clone().detach().unsqueeze(dim=-1)
                K = all_emb_question[:, 0:t].clone().detach()       
                product_score = torch.bmm(K, Q).squeeze(dim=-1)
                _, indices = torch.topk(product_score, k=self.rank_k, dim=1)
                
                # Fast Vectorized Select
                batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, self.rank_k)
                select_history = state_history[batch_indices, indices]
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
        
        # Optimization 4: Avoid explicit large tensor repeats in memory
        K = torch.tanh(self.MLP_query(current_history_state))
        Q = torch.tanh(self.MLP_key(question_concept))
        
        K_exp = K.unsqueeze(1).expand(-1, num_qc, -1, -1)
        Q_exp = Q.unsqueeze(2).expand(-1, -1, num_state, -1)
        
        tmp = self.MLP_W(torch.concat((Q_exp, K_exp), dim=-1))
        tmp = torch.squeeze(tmp, dim=-1)
        alpha = torch.softmax(tmp, dim=2)
        
        p = torch.sum(torch.sum(alpha * output_g, dim=1), dim=1)
        return torch.squeeze(p, dim=-1)

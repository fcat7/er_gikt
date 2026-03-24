"""KT baseline recommenders: DKT+Greedy, DKVMN+Greedy.

接口契约（与 RecommendationSystem.recommend() 一致）:
    recommend(history_q, history_r, history_mask, K=5, interval_time=None, response_time=None)
    -> (recommendation: np.ndarray[K], info: Dict)
"""

import json
import os
import sys
import importlib.util
from typing import Dict, Tuple, Optional

import numpy as np
import torch

_BASELINES_DIR = os.path.dirname(os.path.abspath(__file__))   # back/er/baselines
_ER_DIR = os.path.dirname(_BASELINES_DIR)                     # back/er
_KT_DIR = os.path.join(os.path.dirname(_ER_DIR), 'kt')        # back/kt
for _p in [_ER_DIR, _KT_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_kt_model_classes():
    """通过绝对路径加载 back/kt/baselines/{dkt,dkvmn}.py，避免同名包冲突。"""
    dkt_file = os.path.join(_KT_DIR, 'baselines', 'dkt.py')
    dkvmn_file = os.path.join(_KT_DIR, 'baselines', 'dkvmn.py')

    dkt_spec = importlib.util.spec_from_file_location('kt_dkt_module', dkt_file)
    dkt_module = importlib.util.module_from_spec(dkt_spec)
    dkt_spec.loader.exec_module(dkt_module)

    dkvmn_spec = importlib.util.spec_from_file_location('kt_dkvmn_module', dkvmn_file)
    dkvmn_module = importlib.util.module_from_spec(dkvmn_spec)
    dkvmn_spec.loader.exec_module(dkvmn_module)

    return dkt_module.DKT, dkvmn_module.DKVMN


DKT, DKVMN = _load_kt_model_classes()


class _KTGreedyBase:
    """传统 KT 模型 + 固定 ZPD(τ=0.55) + Hellinger 贪心。"""

    def __init__(self, kt_model_path: str, metadata_path: str, config: Dict, model_type: str, device='cuda'):
        self.device = device
        self.config = config
        self.model_type = model_type.lower()
        self.fixed_tau = 0.55

        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        self.n_question = int(metadata.get('metrics', {}).get('n_question', config.get('n_question', 0)))
        self.n_skill = int(metadata.get('metrics', {}).get('n_skill', config.get('n_skill', 0)))
        
        if self.n_question <= 1:
            raise ValueError('Invalid n_question in metadata/config')

        # Load qs_table for mapping question -> first skill
        data_dir = os.path.dirname(metadata_path)
        qs_table_path = os.path.join(data_dir, 'qs_table.npz')
        import scipy.sparse as sp
        self.qs_table = sp.load_npz(qs_table_path).toarray()

        checkpoint = torch.load(kt_model_path, map_location=device)
        self.model = self._build_model(self.n_question, self.n_skill, checkpoint)
        self._safe_load_checkpoint(checkpoint)
        self.model.to(device)
        self.model.eval()

    def _build_model(self, n_question: int, n_skill: int, checkpoint: dict):
        # 动态推断隐藏层参数，防止 state_dict 尺寸不匹配
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        if hasattr(checkpoint, 'state_dict'):
            state_dict = checkpoint.state_dict()
            
        if self.model_type == 'dkt':
            # DKT: c_emb.weight -> [num_interaction, emb_dim] 或者 interaction_emb.weight
            emb_dim = 100
            for k, v in state_dict.items():
                if 'c_emb.weight' in k or 'interaction_emb.weight' in k:
                    emb_dim = v.shape[1]
                    break
            return DKT(num_question=n_question, num_skill=n_skill, emb_dim=emb_dim)
            
        if self.model_type == 'dkvmn':
            # DKVMN: Mk -> [size_m, dim_s]
            dim_s, size_m = 50, 20
            self.dkvmn_is_q_level = False
            ckp_num_skill = n_skill
            for k, v in state_dict.items():
                if 'Mk' in k:
                    size_m, dim_s = v.shape[0], v.shape[1]
                if 'k_c_emb.weight' in k:
                    # 如果 checkpoint 的 emb size 接近 n_question，说明模型是在题级别训练的
                    if v.shape[0] > n_skill + 50:
                        ckp_num_skill = v.shape[0] - 10
                        self.dkvmn_is_q_level = True
            return DKVMN(num_question=n_question, num_skill=ckp_num_skill, dim_s=dim_s, size_m=size_m)
            
        raise ValueError(f'Unsupported model type: {self.model_type}')

    def _safe_load_checkpoint(self, checkpoint):
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            return
        if isinstance(checkpoint, dict):
            self.model.load_state_dict(checkpoint, strict=False)
            return
        if hasattr(checkpoint, 'state_dict'):
            self.model.load_state_dict(checkpoint.state_dict(), strict=False)
            return
        raise ValueError('Unsupported checkpoint format for KT baseline model')

    def _map_q_to_s(self, q_tensor: torch.Tensor) -> torch.Tensor:
        """Map question tensor to their primary skill using qs_table"""
        q_np = q_tensor.cpu().numpy()
        # qs_table shape: [n_question, n_skill]
        # argmax returns the first index where value is max.
        # Since qs_table is binary, it returns the first valid skill index,
        # or 0 if it's padding / no skill.
        s_np = np.argmax(self.qs_table[q_np], axis=-1)
        return torch.tensor(s_np, dtype=torch.long, device=q_tensor.device)

    def _predict_all_probs(self, history_q: torch.Tensor, history_r: torch.Tensor, history_mask: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            
            history_s = self._map_q_to_s(history_q)
            
            if self.model_type == 'dkt':
                x = history_s + self.model.num_skill * history_r
                # 适配修改过的 DKT: DKT 使用的是 c_emb 而不是 interaction_emb
                input_emb = self.model.c_emb(x)
                h, _ = self.model.lstm_layer(input_emb)
                h = self.model.dropout_layer(h)
                y_logits = self.model.out_layer(h)       # [B, L, num_skill]
                last_logits = y_logits[:, -1, :]         # [B, num_skill]
                probs_skill = torch.sigmoid(last_logits).squeeze(0) # [num_skill]
                
                # Broaden skill probs to question probs
                all_q_idx = torch.arange(self.n_question, device=history_q.device)
                all_s_idx = self._map_q_to_s(all_q_idx)
                probs = probs_skill[all_s_idx]
                
            else:  # dkvmn
                if getattr(self, 'dkvmn_is_q_level', False):
                    eval_s = history_q
                else:
                    eval_s = history_s

                batch_size = history_q.shape[0]
                x = eval_s + self.model.num_skill * history_r
                # 适配修改过的 DKVMN: 使用 k_c_emb 和 v_c_emb
                k = self.model.k_c_emb(eval_s)
                v = self.model.v_c_emb(x)

                Mvt = self.model.Mv0.unsqueeze(0).repeat(batch_size, 1, 1)
                e = torch.sigmoid(self.model.e_layer(v))
                a = torch.tanh(self.model.a_layer(v))
                w = torch.softmax(torch.matmul(k, self.model.Mk.T), dim=-1)

                for t in range(history_q.shape[1]):
                    et = e[:, t]
                    at = a[:, t]
                    wt = w[:, t]
                    mt = history_mask[:, t].unsqueeze(-1).unsqueeze(-1)
                    Mvt_next = Mvt * (1 - (wt.unsqueeze(-1) * et.unsqueeze(1))) + (wt.unsqueeze(-1) * at.unsqueeze(1))
                    Mvt = torch.where(mt == 1, Mvt_next, Mvt)

                all_q_idx = torch.arange(self.n_question, device=history_q.device)
                all_s_idx = self._map_q_to_s(all_q_idx)
                k_all = self.model.k_c_emb(all_s_idx)
                w_all = torch.softmax(torch.matmul(k_all, self.model.Mk.T), dim=-1)
                read_content = (w_all.unsqueeze(-1) * Mvt.squeeze(0)).sum(-2)
                f = torch.tanh(self.model.f_layer(torch.cat([read_content, k_all], dim=-1)))
                logits = self.model.p_layer(self.model.dropout_layer(f)).squeeze(-1)
                probs = torch.sigmoid(logits)

        probs_np = probs.detach().cpu().numpy().astype(np.float32)
        probs_np[0] = 0.5  # padding index
        return probs_np

    def recommend(
        self,
        history_q: torch.Tensor,
        history_r: torch.Tensor,
        history_mask: torch.Tensor,
        K: int = 5,
        interval_time: Optional[torch.Tensor] = None,
        response_time: Optional[torch.Tensor] = None,
    ) -> Tuple[np.ndarray, Dict]:
        done_qs = history_q[history_mask.bool()].unique().cpu().numpy()
        all_qs = np.arange(1, self.n_question, dtype=np.int64)
        avail_qs = np.setdiff1d(all_qs, done_qs)

        if len(avail_qs) == 0:
            avail_qs = all_qs

        all_probs = self._predict_all_probs(history_q, history_r, history_mask)
        valid_probs = all_probs[avail_qs]

        if len(avail_qs) <= K:
            rec = avail_qs
            rec_probs = valid_probs
        else:
            tau = self.fixed_tau
            hellinger = (
                (np.sqrt(valid_probs) - np.sqrt(tau)) ** 2 +
                (np.sqrt(np.clip(1 - valid_probs, 0, 1)) - np.sqrt(np.clip(1 - tau, 0, 1))) ** 2
            )
            top_idx = np.argsort(hellinger)[:K]
            rec = avail_qs[top_idx]
            rec_probs = valid_probs[top_idx]

        order = np.argsort(rec_probs)
        rec = rec[order]
        rec_probs = rec_probs[order]

        return rec.astype(np.int64), {
            'rec_probs': rec_probs,
            'cognitive_state': {
                'tau': self.fixed_tau,
                'm_k': None,
            },
        }


class DKTGreedyRecommender(_KTGreedyBase):
    def __init__(self, kt_model_path: str, metadata_path: str, config: Dict, device='cuda'):
        super().__init__(kt_model_path, metadata_path, config, model_type='dkt', device=device)


class DKVMNGreedyRecommender(_KTGreedyBase):
    def __init__(self, kt_model_path: str, metadata_path: str, config: Dict, device='cuda'):
        super().__init__(kt_model_path, metadata_path, config, model_type='dkvmn', device=device)

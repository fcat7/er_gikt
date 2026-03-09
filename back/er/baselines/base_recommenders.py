"""CCS-MOPSO-ER 对比基线推荐器实现

接口契约（与 RecommendationSystem.recommend() 完全一致）:
    recommend(history_q, history_r, history_mask, K=5,
              interval_time=None, response_time=None)
    → (recommendation: np.ndarray[K], info: Dict)

其中 info 包含:
    'cognitive_state': {'tau': float, 'm_k': np.ndarray or None, ...}
    'rec_probs':       np.ndarray[K] or None
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from typing import Dict, Tuple, Optional

# ---- 路径设置（确保能 import back/kt 的模块）----
_BASELINES_DIR = os.path.dirname(os.path.abspath(__file__))  # back/er/baselines
_ER_DIR = os.path.dirname(_BASELINES_DIR)                    # back/er
_KT_DIR = os.path.join(os.path.dirname(_ER_DIR), 'kt')       # back/kt
for _p in [_ER_DIR, _KT_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ============================================================
# 公共辅助：复用 pso_recommend 的前三阶段（状态提取 + 候选集）
# ============================================================
class _BaseRecommender:
    """
    公共基类：模型加载 + 认知状态提取 + 候选集构建。
    只有第四阶段（PSO 优化）被各子类替换。
    """

    def __init__(self, model_path: str, metadata_path: str, config: Dict, device='cuda'):
        from pso_recommend import CognitiveStateExtractor, CandidateBuilder
        from models.gikt import GIKT

        self.device = device
        self.config = config

        # 加载模型（同 RecommendationSystem）
        checkpoint = torch.load(model_path, map_location=device)
        self.model = checkpoint if isinstance(checkpoint, GIKT) else None
        if self.model is None:
            raise ValueError(f"Failed to load GIKT model from {model_path}")

        self.state_extractor = CognitiveStateExtractor(self.model, metadata_path, device)
        self.candidate_builder = CandidateBuilder(self.model, metadata_path, config)

    def _extract_state_and_candidates(
        self,
        history_q: torch.Tensor,
        history_r: torch.Tensor,
        history_mask: torch.Tensor,
        K: int,
        interval_time: Optional[torch.Tensor],
        response_time: Optional[torch.Tensor],
    ) -> Tuple[np.ndarray, np.ndarray, Dict, float]:
        """
        共享前三阶段（与 RecommendationSystem.recommend() 完全相同）:
            0. 冷启动检测
            1. 认知状态提取
            2. 候选集构建
        Returns:
            (candidates, probs_np, cognitive_state, tau)
        """
        from pso_recommend import CandidateBuilder

        T = history_mask.sum().item()
        n_q = self.candidate_builder.n_question

        # 冷启动回退
        if T < 10:
            done_qs = history_q[history_mask.bool()].unique().cpu().numpy()
            all_qs = np.arange(1, n_q)
            avail = np.setdiff1d(all_qs, done_qs)
            rec = np.random.choice(avail if len(avail) >= K else all_qs, K, replace=False)
            cog = {'tau': 0.55, 'm_k': None, 'pid_ema': 0.5, 'pid_diff': 0.0}
            return rec, np.full(K, 0.5, dtype=np.float32), cog, 0.55

        # 阶段 1: 认知状态
        cognitive_state = self.state_extractor.extract(
            history_q, history_r, history_mask,
            interval_time=interval_time,
            response_time=response_time,
        )

        # 阶段 2: 候选集（保留完整 G1-G4 门控，对所有基线公平）
        candidates, probs_tensor = self.candidate_builder.build(
            self.model, history_q, history_r, history_mask, cognitive_state
        )
        cognitive_state['tau'] = self.candidate_builder._compute_zpd(cognitive_state)

        probs_np = probs_tensor.cpu().numpy()
        tau = cognitive_state['tau']

        return candidates, probs_np, cognitive_state, tau


# ============================================================
# P0-1: GreedyRecommender（单目标贪心，消融 A4 实现基础）
# ============================================================
class GreedyRecommender(_BaseRecommender):
    """
    P0-1 Greedy 基线：
        对候选集中所有题，按 Hellinger-ZPD 距离（F1）升序排列，
        直接取 Top-K，不做多目标优化、不考虑多样性。

    论文作用（消融 A4）：
        Full Model(MOPSO) vs Greedy 的 ILD 差距 → 证明多目标优化的多样性贡献。
        Full Model vs Greedy 的 DM 差距很小 → MOPSO 在 F1 上同样高效。
    """

    def recommend(
        self,
        history_q: torch.Tensor,
        history_r: torch.Tensor,
        history_mask: torch.Tensor,
        K: int = 5,
        interval_time: Optional[torch.Tensor] = None,
        response_time: Optional[torch.Tensor] = None,
    ) -> Tuple[np.ndarray, Dict]:

        candidates, probs_np, cognitive_state, tau = self._extract_state_and_candidates(
            history_q, history_r, history_mask, K, interval_time, response_time
        )

        # 冷启动已在基类处理（candidates 就是最终推荐）
        if len(candidates) <= K:
            rec = candidates[:K] if len(candidates) >= K else candidates
            rec_probs = probs_np[:len(rec)]
        else:
            # Hellinger-ZPD F1 = (√p - √τ)² + (√(1-p) - √(1-τ)²)，越小越好
            hellinger = (
                (np.sqrt(probs_np) - np.sqrt(tau)) ** 2 +
                (np.sqrt(np.clip(1 - probs_np, 0, 1)) - np.sqrt(np.clip(1 - tau, 0, 1))) ** 2
            )
            sorted_idx = np.argsort(hellinger)          # F1 升序 → 最优先
            top_idx = sorted_idx[:K]
            rec = candidates[top_idx]
            rec_probs = probs_np[top_idx]

        # 按难度升序排列（易→难，对齐 MOPSO 后处理）
        order = np.argsort(rec_probs)
        rec = rec[order]
        rec_probs = rec_probs[order]

        return rec, {
            'cognitive_state': cognitive_state,
            'rec_probs': rec_probs,
        }


# ============================================================
# P0-2: RandomRecommender
# ============================================================
class RandomRecommender(_BaseRecommender):
    """
    P0-2 Random 基线：
        从未做过的题（G4 门控，不使用 G1-G3）中随机选 K 题。
        完全不使用认知状态，是最弱的下界基线。
    """

    def recommend(
        self,
        history_q: torch.Tensor,
        history_r: torch.Tensor,
        history_mask: torch.Tensor,
        K: int = 5,
        interval_time: Optional[torch.Tensor] = None,
        response_time: Optional[torch.Tensor] = None,
    ) -> Tuple[np.ndarray, Dict]:

        n_q = self.candidate_builder.n_question
        done_qs = history_q[history_mask.bool()].unique().cpu().numpy()

        # 只做 G4 门控：排除已做过题，从全量可用题中随机选
        all_qs = np.arange(1, n_q)   # 0 是 padding，跳过
        avail_qs = np.setdiff1d(all_qs, done_qs)

        if len(avail_qs) < K:
            rec = np.random.choice(all_qs, K, replace=True)
        else:
            rec = np.random.choice(avail_qs, K, replace=False)

        return rec, {
            'cognitive_state': {'tau': 0.55, 'm_k': None},
            'rec_probs': None,
        }


# ============================================================
# P0-3: PopularityRecommender
# ============================================================
class PopularityRecommender(_BaseRecommender):
    """
    P0-3 Popularity 基线：
        选全训练集交互频率最高的 Top-K 题，排除学生已做过的。
        global_popular_list 在构造时从 train.parquet 预计算。
    """

    def __init__(self, model_path: str, metadata_path: str, config: Dict, device='cuda', train_parquet_path: str = None):
        super().__init__(model_path, metadata_path, config, device)

        n_q = self.candidate_builder.n_question

        # 预计算全局题目热度排行
        if train_parquet_path is None:
            from config import get_config
            dataset_name = config.get('dataset_name', 'assist09')
            kt_config = get_config(dataset_name)
            train_parquet_path = os.path.join(kt_config.PROCESSED_DATA_DIR, 'train.parquet')

        train_data = pd.read_parquet(train_parquet_path)
        global_freq = np.zeros(n_q, dtype=np.int64)
        for q_seq in train_data['q_seq']:
            arr = np.array(q_seq, dtype=np.int64)
            arr = arr[(arr > 0) & (arr < n_q)]
            np.add.at(global_freq, arr, 1)

        # 降序排列，0（padding）已被排除
        self.global_popular_list = np.argsort(-global_freq)
        self.global_popular_list = self.global_popular_list[global_freq[self.global_popular_list] > 0]

    def recommend(
        self,
        history_q: torch.Tensor,
        history_r: torch.Tensor,
        history_mask: torch.Tensor,
        K: int = 5,
        interval_time: Optional[torch.Tensor] = None,
        response_time: Optional[torch.Tensor] = None,
    ) -> Tuple[np.ndarray, Dict]:

        done_qs = set(history_q[history_mask.bool()].unique().cpu().numpy().tolist())

        rec = []
        for q in self.global_popular_list:
            if int(q) not in done_qs:
                rec.append(int(q))
            if len(rec) == K:
                break

        # 如果热门题不够（极端情况），随机补充
        if len(rec) < K:
            n_q = self.candidate_builder.n_question
            remain = np.setdiff1d(np.arange(1, n_q), list(done_qs) + rec)
            extra = np.random.choice(remain, K - len(rec), replace=len(remain) < K - len(rec))
            rec.extend(extra.tolist())

        return np.array(rec[:K], dtype=np.int64), {
            'cognitive_state': {'tau': 0.55, 'm_k': None},
            'rec_probs': None,
        }

# ============================================================
# CCS-MOPSO-ER: Cognitive Coordinate System guided MOPSO
# for Exercise Recommendation (FA-GIKT backend)
# ============================================================

import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from itertools import combinations
from typing import Dict, Tuple, List, Optional
from tqdm import tqdm

try:
    from icecream import ic
except ImportError:
    ic = print

class CognitiveStateExtractor:
    """阶段 0：从 FA-GIKT 模型提取认知状态与 Probing 掌握度"""
    def __init__(self, model, metadata_path: str, device='cuda'):
        """
        Args:
            model: 训练好的 FA-GIKT 模型
            metadata_path: metadata.json 文件路径（包含 skill_domain_map 等）
            device: torch device
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # 加载元数据
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        data_dir = os.path.dirname(metadata_path)
        # skill_domain_map 可能内联也可能是外部文件引用
        sdm_raw = metadata.get('mappings', metadata).get('skill_domain_map', {})
        if isinstance(sdm_raw, str):
            sdm_path = os.path.join(data_dir, sdm_raw)
            with open(sdm_path, 'r', encoding='utf-8') as f:
                sdm_raw = json.load(f)
        self.skill_domain_map = {int(k): v for k, v in sdm_raw.items()}
        self.n_skill = metadata.get('metrics', {}).get('n_skill', len(self.skill_domain_map))
        self.n_domain = metadata.get('metrics', {}).get('n_domain', len(set(self.skill_domain_map.values())))
        
        # 使用模型 IRT 难度参数选择锚题（框架 V∞ C5: b_q 最接近 0 → P ≈ 0.5）
        self.qs_table = self.model.qs_table.cpu().numpy()  # [n_question, n_skill]
        n_question = self.qs_table.shape[0]
        with torch.no_grad():
            all_q = torch.arange(n_question, device=device)
            if hasattr(self.model, 'difficulty_bias'):
                self.difficulty_params = self.model.difficulty_bias(all_q).squeeze(-1).cpu().numpy()
            else:
                # fallback: 预计算特征
                q_features_path = os.path.join(os.path.dirname(metadata_path), 'q_features.npy')
                self.difficulty_params = np.load(q_features_path)[:, 0]

        # 找每个技能的锚题（IRT b_q 最接近 0 → 预测概率最接近 0.5）
        self.anchor_questions = {}
        for s in range(self.n_skill):
            q_mask = self.qs_table[:, s] > 0
            if q_mask.sum() == 0:
                self.anchor_questions[s] = 0
            else:
                q_ids = np.where(q_mask)[0]
                difficulties = self.difficulty_params[q_ids]
                closest_idx = np.argmin(np.abs(difficulties))
                self.anchor_questions[s] = q_ids[closest_idx]

    def extract(self, history_q: torch.Tensor, history_r: torch.Tensor,
                history_mask: torch.Tensor,
                interval_time: torch.Tensor = None,
                response_time: torch.Tensor = None) -> Dict:
        """
        运行 Probing 获取学生各技能掌握度向量和认知坐标
        Args:
            history_q: [1, T] 历史问题序列
            history_r: [1, T] 历史回答序列
            history_mask: [1, T] 掩码
            interval_time: [1, T] 时间间隔（可选，CognitiveRNNCell 需要）
            response_time: [1, T] 作答耗时（可选，CognitiveRNNCell 需要）
        Returns:
            dict with keys:
                - 'm_k': [n_skill] 掌握度向量
                - 'pid_ema': [n_domain] or scalar 领域级 EMA
                - 'pid_diff': [n_domain] or scalar 领域级微分
                - 'h_T': [1, emb_dim] 最后时刻隐藏状态
                - 'state_dict': 包含所有捕获的底层张量字典
        """
        # 捕获模型末端的认知状态
        state_dict = self.model.get_state_for_recommendation(
            history_q.to(self.device),
            history_r.to(self.device),
            history_mask.to(self.device),
            interval_time=interval_time.to(self.device) if interval_time is not None else None,
            response_time=response_time.to(self.device) if response_time is not None else None,
        )
        
# 构造批量 probing (Shape: [n_skill, 2, emb_dim])
        BATCH_SIZE = min(512, self.n_skill)
        m_k = np.zeros(self.n_skill)

        with torch.no_grad():
            for i in range(0, self.n_skill, BATCH_SIZE):
                end_i = min(i + BATCH_SIZE, self.n_skill)
                b_size = end_i - i
                
                # 获取此批次技能的锚题
                anchor_qs = [self.anchor_questions[s] for s in range(i, end_i)]
                q_targets = torch.tensor(anchor_qs, dtype=torch.long, device=self.device)
                
                qs_concat = torch.zeros(b_size, 2, self.model.emb_dim, device=self.device)
                qs_concat[:, 0] = self.model.emb_table_question(q_targets).detach()
                qs_concat[:, 1] = self.model.emb_table_skill(
                    torch.arange(i, end_i, dtype=torch.long, device=self.device)
                ).detach()
                
                curr_state_expanded = state_dict['current_history_state'].expand(b_size, -1, -1).to(self.device)
                
                cached_keys_expanded = None
                if state_dict['cached_keys'] is not None:
                    cached_keys_expanded = state_dict['cached_keys'].expand(b_size, -1, -1).to(self.device)
                    
                pid_data_expanded = None
                if state_dict['pid_data'] is not None:
                    pid_e, pid_d = state_dict['pid_data']
                    pid_e_exp = pid_e.expand(b_size, -1) if len(pid_e.shape)==2 else pid_e.expand(b_size)
                    pid_d_exp = pid_d.expand(b_size, -1) if pid_d is not None and len(pid_d.shape)==2 else (pid_d.expand(b_size) if pid_d is not None else None)
                    pid_data_expanded = (pid_e_exp, pid_d_exp)

                logits = self.model.predict(
                    qs_concat,
                    curr_state_expanded,
                    q_target=q_targets,
                    pid_data=pid_data_expanded,
                    cached_keys=cached_keys_expanded
                )
                probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
                m_k[i:end_i] = probs
        
        # 提取 PID 状态
        pid_ema, pid_diff = state_dict['pid_data'] if state_dict['pid_data'] is not None else (None, None)
        if pid_ema is not None:
            if len(pid_ema.shape) == 2:
                # Domain mode: [1, n_domain]
                pid_ema = pid_ema[0].detach().cpu().numpy()
                pid_diff = pid_diff[0].detach().cpu().numpy() if pid_diff is not None else np.zeros(self.n_domain)
            else:
                # Scalar mode
                pid_ema = pid_ema.item()
                pid_diff = pid_diff.item() if pid_diff is not None else 0.0
                
        return {
            'm_k': m_k,
            'pid_ema': pid_ema,
            'pid_diff': pid_diff,
            'h_T': state_dict['current_history_state'],
            'state_dict': state_dict
        }


class CandidateBuilder:
    """阶段 1：候选集构建（五重门控 + 高斯软排序）"""
    def __init__(self, model, metadata_path: str, config: Dict):
        """
        Args:
            model: FA-GIKT 模型
            metadata_path: metadata.json 路径
            config: 包含参数如 gate_params, gaussian_sigma 等的配置字典
        """
        self.model = model
        self.device = next(model.parameters()).device
        
        # 加载元数据
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        self.n_question = metadata['metrics']['n_question']
        data_dir = os.path.dirname(metadata_path)
        sdm_raw = metadata.get('mappings', metadata).get('skill_domain_map', {})
        if isinstance(sdm_raw, str):
            sdm_path = os.path.join(data_dir, sdm_raw)
            with open(sdm_path, 'r', encoding='utf-8') as f:
                sdm_raw = json.load(f)
        self.skill_domain_map = {int(k): v for k, v in sdm_raw.items()}
        
        # G1 区分度使用模型 4PL IRT 参数（框架 V∞: a_q = 1 + softplus(bias)）
        with torch.no_grad():
            all_q = torch.arange(self.n_question, device=self.device)
            if hasattr(model, 'discrimination_bias'):
                self.discrimination = 1.0 + F.softplus(model.discrimination_bias(all_q).squeeze(-1))
            else:
                # fallback: 预计算特征
                q_features_path = os.path.join(os.path.dirname(metadata_path), 'q_features.npy')
                q_features = np.load(q_features_path)
                self.discrimination = torch.tensor(q_features[:, 1], dtype=torch.float32, device=self.device)
        
        # 题目-技能矩阵
        self.qs_table = self.model.qs_table.to(self.device)
        
        # 配置参数
        self.gate_p_min = config.get('gate_p_min', 0.15)
        self.gate_p_max = config.get('gate_p_max', 0.88)
        self.gaussian_sigma = config.get('gaussian_sigma', 0.15)
        self.confidence_discount = config.get('confidence_discount', 0.7)
        self.top_k = config.get('top_k_candidates', 200)

    def build(self, model, history_q, history_r, history_mask,
              cognitive_state: Dict) -> Tuple[np.ndarray, torch.Tensor]:
        """
        构建高质量候选集
        Args:
            model: FA-GIKT 模型
            history_q, history_r, history_mask: 历史序列
            cognitive_state: CognitiveStateExtractor.extract() 的输出
        Returns:
            (候选题ID数组, 预测概率张量)
        """
        m_k = cognitive_state['m_k']  # [n_skill]
        state_dict = cognitive_state['state_dict']
        
        # 计算所有题目的预测概率 (可批量化提升效率)
        n_q = self.n_question
        probs = np.zeros(n_q)
        
        # Vectorized batch prediction (to speed up)
        BATCH_SIZE = 512
        for start_idx in range(1, n_q, BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, n_q)
            batch_q_ids = torch.arange(start_idx, end_idx, device=self.device)
            B = end_idx - start_idx
            
            # max skills for this batch
            batch_qs_table = self.qs_table[batch_q_ids] # [B, n_skill]
            skill_mask = batch_qs_table.bool()
            max_num_skill = skill_mask.sum(dim=1).max().item() if B>0 else 0
            max_num_skill = max(max_num_skill, 1) # avoid size 0
            
            qs_concat = torch.zeros(B, max_num_skill + 1, self.model.emb_dim, device=self.device)
            qs_concat[:, 0, :] = self.model.emb_table_question(batch_q_ids).detach()
            
            active_indices = torch.nonzero(skill_mask, as_tuple=True)
            batch_rows, skill_cols = active_indices[0], active_indices[1]
            if len(batch_rows) > 0:
                ranks = torch.cumsum(skill_mask.long(), dim=1)[batch_rows, skill_cols]
                valid_mask = ranks <= max_num_skill
                if valid_mask.any():
                    valid_b = batch_rows[valid_mask]
                    valid_s = skill_cols[valid_mask]
                    valid_r = ranks[valid_mask]
                    qs_concat[valid_b, valid_r, :] = self.model.emb_table_skill(valid_s).detach()
            
            # Expand current_history_state to match B
            curr_state_expanded = state_dict['current_history_state'].expand(B, -1, -1).to(self.device)
            
            cached_keys_expanded = None
            if state_dict['cached_keys'] is not None:
                cached_keys_expanded = state_dict['cached_keys'].expand(B, -1, -1).to(self.device)
                
            pid_data_expanded = None
            if state_dict['pid_data'] is not None:
                pid_e, pid_d = state_dict['pid_data']
                pid_e_exp = pid_e.expand(B, -1) if len(pid_e.shape)==2 else pid_e.expand(B)
                pid_d_exp = pid_d.expand(B, -1) if pid_d is not None and len(pid_d.shape)==2 else (pid_d.expand(B) if pid_d is not None else None)
                pid_data_expanded = (pid_e_exp, pid_d_exp)

            with torch.no_grad():
                logits = model.predict(
                    qs_concat,
                    curr_state_expanded,
                    q_target=batch_q_ids,
                    pid_data=pid_data_expanded,
                    cached_keys=cached_keys_expanded
                )
                batch_probs = torch.sigmoid(logits).cpu().numpy()
                probs[start_idx:end_idx] = batch_probs

        # G1: 区分度预筛
        disc_median = np.median(self.discrimination.cpu().numpy()[1:])
        g1_mask = self.discrimination.cpu().numpy() > disc_median
        g1_mask[0] = False # Padding
        
        # G2: 难度区间
        g2_mask = (probs >= self.gate_p_min) & (probs <= self.gate_p_max)
        
        # G3: 薄弱技能关联
        qs_table_np = self.qs_table.cpu().numpy()
        threshold = float(np.mean(m_k))
        weak_skills_mask = m_k < threshold
        
        g3_mask = np.zeros(n_q, dtype=bool)
        if weak_skills_mask.any() and n_q > 1:
            g3_mask[1:] = (qs_table_np[1:] * weak_skills_mask).sum(axis=1) > 0
                
        # G4: 未做过
        done_qs = history_q[history_mask.bool()].unique().cpu().numpy()
        g4_mask = np.ones(n_q, dtype=bool)
        g4_mask[done_qs] = False
        
        # 融合四重门控
        candidate_mask = g1_mask & g2_mask & g3_mask & g4_mask
        candidates = np.where(candidate_mask)[0]
        
        # 如果候选题不足或者完全没有，放宽限制
        if len(candidates) < self.top_k // 2:
            # 放弃 G3
            candidate_mask = g1_mask & g2_mask & g4_mask
            candidates = np.where(candidate_mask)[0]
        if len(candidates) == 0:
            # 备降：返回全局热门题
            candidates = np.argsort(-np.bincount(history_q[history_mask.bool()].cpu().numpy(), minlength=n_q))[:200]
            candidates = candidates[candidates != 0]

        # G5（软）：置信折扣 + 高斯软排序
        beta_q = np.ones(len(candidates)) * self.confidence_discount
        seen_skills_idx = self.qs_table[done_qs].nonzero(as_tuple=True)[1].cpu().numpy()
        seen_skills_mask = np.zeros(qs_table_np.shape[1], dtype=bool)
        seen_skills_mask[seen_skills_idx] = True
        
        if len(candidates) > 0:
            candidates_qs = qs_table_np[candidates]
            has_seen_skill = (candidates_qs * seen_skills_mask).sum(axis=1) > 0
            beta_q[has_seen_skill] = 1.0 # 只要涉及过相关技能，则不进行折扣
            
        # 计算动态 ZPD
        tau = self._compute_zpd(cognitive_state)
        
        # 高斯软排序
        weights = beta_q * np.exp(-((probs[candidates] - tau) ** 2) / (2 * self.gaussian_sigma ** 2))
        sorted_idx = np.argsort(-weights)
        
        # 取 Top-K
        final_candidates = candidates[sorted_idx[:min(self.top_k, len(candidates))]]
        final_probs = torch.tensor(probs[final_candidates], dtype=torch.float32, device=self.device)
        
        return final_candidates, final_probs

    def _compute_zpd(self, cognitive_state: Dict) -> float:
        """计算动态 ZPD 锚点"""
        pid_ema = cognitive_state['pid_ema']
        pid_diff = cognitive_state['pid_diff']
        
        if isinstance(pid_ema, np.ndarray):
            # Domain mode
            mu_bar = np.mean(pid_ema)
        else:
            # Scalar mode
            mu_bar = float(pid_ema) if pid_ema is not None else 0.5
            
        if isinstance(pid_diff, np.ndarray):
            delta_mu = np.mean(pid_diff)
        else:
            delta_mu = float(pid_diff) if pid_diff is not None else 0.0
            
        # tau = 0.55 + 0.1*(mu_bar - 0.5) + 0.05*tanh(5*delta_mu)
        tau = 0.55 + 0.1 * (mu_bar - 0.5) + 0.05 * np.tanh(5 * delta_mu)
        return np.clip(tau, 0.2, 0.85)


class FitnessEvaluator:
    """多目标适应度评估：F1(Hellinger-ZPD) + F2(薄弱覆盖率)"""
    def __init__(self, qs_table: torch.Tensor, config: Dict):
        """
        Args:
            qs_table: [n_question, n_skill] 题目-技能矩阵
            config: 包含参数配置
        """
        self.qs_table = qs_table
        self.qs_table_np = qs_table.cpu().numpy()
        self.K = config.get('K', 5)

    def evaluate_combo(self, combo: np.ndarray, probs: np.ndarray,
                       m_k: np.ndarray, tau: float) -> Tuple[float, float]:
        """
        计算一个推荐组合的两个目标函数值
        Args:
            combo: [K] 推荐题ID
            probs: [K] 对应的预测概率
            m_k: [n_skill] 掌握度向量
            tau: float ZPD 锚点
        Returns:
            (F1, F2) Hellinger 距离与薄弱覆盖率（取负）
        """
        # F1: Hellinger distance to ZPD
        hellinger = np.mean(
            (np.sqrt(probs) - np.sqrt(tau)) ** 2 +
            (np.sqrt(1 - probs) - np.sqrt(1 - tau)) ** 2
        )
        
                # F2: Weak skill coverage (negative for minimization)
        threshold = float(np.mean(m_k))
        weak_skills_mask = m_k < threshold
        
        qs_combo = self.qs_table_np[combo] # [K, n_skill]
        covered_mask = qs_combo.sum(axis=0) > 0 # [n_skill]
        covered_weak_mask = covered_mask & weak_skills_mask # [n_skill]

        if not weak_skills_mask.any():
            f2 = 0.0
        else:
            # 加权覆盖（只覆盖没有就算）
            f2 = -float(np.sum(1 - m_k[covered_weak_mask]))
            
        return float(hellinger), float(f2)


class DiscreteMOPSO:
    """离散多目标粒子群优化"""
    def __init__(self, candidates: np.ndarray, evaluator: FitnessEvaluator,
                 cognitive_state: Dict, config: Dict):
        """
        Args:
            candidates: [n_cand] 候选题ID
            evaluator: FitnessEvaluator 实例
            cognitive_state: 认知状态字典
            config: PSO 配置（粒子数、迭代数等）
        """
        self.candidates = candidates
        self.evaluator = evaluator
        self.m_k = cognitive_state['m_k']
        self.tau = cognitive_state.get('tau', 0.55)
        
        self.n_particles = config.get('n_particles', 30)
        self.n_iterations = config.get('n_iterations', 80)
        self.K = config.get('K', 5)
        self.archive_size = config.get('archive_size', 50)
        
        self.probs = {q: np.random.rand() for q in candidates}  # To be correctly populated if needed
        self.w = config.get('inertia_weight', 0.4)
        self.c1 = config.get('c1', 0.5)
        self.c2 = config.get('c2', 0.5)

    def _compute_crowding_distance(self, fitness_values: List[Tuple[float, float]]) -> List[float]:
        """计算拥挤距离 (Crowding Distance) 以维持 Pareto 前沿多样性"""
        n = len(fitness_values)
        distances = [0.0] * n
        if n <= 2:
            return [float('inf')] * n
            
        for m in range(2): # 此处是双目标
            sorted_idx = np.argsort([f[m] for f in fitness_values])
            distances[sorted_idx[0]] = float('inf')
            distances[sorted_idx[-1]] = float('inf')
            
            f_max = fitness_values[sorted_idx[-1]][m]
            f_min = fitness_values[sorted_idx[0]][m]
            
            if f_max == f_min:
                continue
                
            for i in range(1, n - 1):
                distances[sorted_idx[i]] += (fitness_values[sorted_idx[i+1]][m] - fitness_values[sorted_idx[i-1]][m]) / (f_max - f_min)
                
        return distances

    def optimize(self, candidates_probs: torch.Tensor) -> np.ndarray:
        """运行 PSO 并返回最优推荐组合"""
        # Store exact prob dict
        self.probs = {q.item(): p.item() for q, p in zip(self.candidates, candidates_probs)}
        
        # 初始化粒子（随机选 K 道题）
        particles = [
            np.random.choice(self.candidates, size=self.K, replace=False)
            for _ in range(self.n_particles)
        ]
        
        # 初始化个体最优
        pbest = particles[:]
        pbest_fitness = [
            self.evaluator.evaluate_combo(
                p, np.array([self.probs.get(q, 0.5) for q in p]),
                self.m_k, self.tau
            )
            for p in particles
        ]
        
        # 初始化帕累托前沿
        archive = self._build_archive(particles, pbest_fitness)
        
        # 迭代
        for iteration in range(self.n_iterations):
            for i in range(self.n_particles):
                # 冯·诺依曼修正：全局最优选取拥挤距离最大的 Pareto 解，避免多样性坍塌
                # 修复：当多个解拥挤距离相同（如 archive≤2 时均为 inf），随机选取而非总选 idx-0
                if len(archive) > 0:
                    fitness_list = [af for ap, af in archive]
                    cd = self._compute_crowding_distance(fitness_list)
                    max_cd = max(cd)
                    # 所有拥挤距离 ≥ max_cd*0.99 的解均作为候选 gbest（随机选取）
                    top_indices = [j for j, d in enumerate(cd) if d >= max_cd * 0.99]
                    gbest = archive[np.random.choice(top_indices)][0]
                else:
                    gbest = particles[i]
                    
                # 更新粒子（集合替换操作）
                particles[i] = self._update_particle(particles[i], pbest[i], gbest)
                
                # 评估新粒子
                fit = self.evaluator.evaluate_combo(
                    particles[i], np.array([self.probs.get(q, 0.5) for q in particles[i]]),
                    self.m_k, self.tau
                )
                
                # 更新个体最优
                if self._dominates(fit, pbest_fitness[i]):
                    pbest[i] = particles[i]
                    pbest_fitness[i] = fit
                    
            # 更新 Archive
            archive = self._build_archive(particles + [pbest[i] for i in range(self.n_particles)], 
                                          [self.evaluator.evaluate_combo(p, np.array([self.probs.get(q, 0.5) for q in p]), self.m_k, self.tau) for p in particles] + pbest_fitness)
            
        # 冯·诺依曼修正：返回拥挤距离最大的帕累托解作为最终推荐方案
        # 修复：同样使用随机加权选取，避免 archive 退化时总返回第一个解
        if len(archive) > 0:
            fitness_list = [af for ap, af in archive]
            cd = self._compute_crowding_distance(fitness_list)
            max_cd = max(cd)
            top_indices = [j for j, d in enumerate(cd) if d >= max_cd * 0.99]
            return archive[np.random.choice(top_indices)][0]
        return particles[0]

    def _update_particle(self, current, pbest, gbest) -> np.ndarray:
        """粒子更新：惯性探索 + 认知/社会替换（框架 V∞ §2.5）"""
        current = current.copy()
        # 惯性分量: 以概率 w 随机替换一个位置（保持探索多样性）
        if np.random.rand() < self.w:
            unused = np.setdiff1d(self.candidates, current)
            if len(unused) > 0:
                replace_idx = np.random.randint(self.K)
                current[replace_idx] = np.random.choice(unused)
        # 认知分量: 以概率 c1 向 pbest 靠近
        if np.random.rand() < self.c1 and len(pbest) > 0:
            pbest_unique = list(set(pbest) - set(current))
            if pbest_unique:
                to_add = np.random.choice(pbest_unique, size=1)[0]
                current[np.random.randint(self.K)] = to_add
                
        # 社会分量: 以概率 c2 向 gbest 靠近
        if np.random.rand() < self.c2 and len(gbest) > 0:
            gbest_unique = list(set(gbest) - set(current))
            if gbest_unique:
                to_add = np.random.choice(gbest_unique, size=1)[0]
                current[np.random.randint(self.K)] = to_add
                
        return current

    def _build_archive(self, particles, fitness_values) -> List:
        """构建帕累托前沿并根据拥挤距离裁剪"""
        archive = []
        for p, f in zip(particles, fitness_values):
            dominated = False
            for ap, af in archive:
                if self._dominates(af, f):
                    dominated = True
                    break
            if not dominated:
                archive = [(ap, af) for ap, af in archive if not self._dominates(f, af)] + [(p, f)]
                
        # Trim archive size via crowding distance
        if len(archive) > self.archive_size:
            fitness_list = [af for ap, af in archive]
            cd = self._compute_crowding_distance(fitness_list)
            sorted_indices = np.argsort(cd)[::-1] # 降序，距离越大越好多样性越好
            archive = [archive[i] for i in sorted_indices[:self.archive_size]]
            
        return archive

    def _dominates(self, f1: Tuple, f2: Tuple) -> bool:
        """f1 支配 f2？(Minimization)"""
        return f1[0] <= f2[0] and f1[1] <= f2[1] and (f1[0] < f2[0] or f1[1] < f2[1])


class RecommendationSystem:
    """主入口：串联所有模块"""
    def __init__(self, model_path: str, metadata_path: str, config: Dict, device='cuda'):
        """
        Args:
            model_path: 训练好的 gikt.pt 路径
            metadata_path: metadata.json 路径
            config: 总体配置字典
            device: torch 设备
        """
        # 加载模型
        import sys
        _KT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'kt')
        if _KT_DIR not in sys.path:
            sys.path.insert(0, _KT_DIR)
        from models.gikt import GIKT
        
        checkpoint = torch.load(model_path, map_location=device)
        # 如果保存的是整个对象
        if isinstance(checkpoint, GIKT):
            self.model = checkpoint
        elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # 实例化一个然后加载state_dict
            pass # 简化起见，我们目前假定保存的是整个模型或有其它加载方式
        
        self.model = checkpoint if isinstance(checkpoint, GIKT) else None
        
        if self.model is None:
            raise ValueError(f"Failed to load model from {model_path}")
            
        self.device = device
        self.config = config
        self.state_extractor = CognitiveStateExtractor(self.model, metadata_path, device)
        self.candidate_builder = CandidateBuilder(self.model, metadata_path, config)

    def recommend(self, history_q: torch.Tensor, history_r: torch.Tensor,
                  history_mask: torch.Tensor, K: int = 5,
                  interval_time: torch.Tensor = None,
                  response_time: torch.Tensor = None) -> Tuple[np.ndarray, Dict]:
        """
        为一个学生生成推荐
        Returns:
            (推荐题ID数组, 诊断信息字典)
        """
        # 冷启动判断
        T = history_mask.sum().item()
        if T < 10:
            # Cold start 回退：随机热门题
            n_q = self.candidate_builder.n_question
            done_qs = history_q[history_mask.bool()].unique().cpu().numpy()
            all_qs = np.arange(1, n_q)
            avaliable_qs = np.setdiff1d(all_qs, done_qs)
            if len(avaliable_qs) < K:
                rec = np.random.choice(all_qs, K, replace=True)
            else:
                rec = np.random.choice(avaliable_qs, K, replace=False)
            return rec, {'cognitive_state': {'tau': 0.55, 'm_k': None}, 'rec_probs': None}
            
        # 阶段 0: 提取认知状态
        cognitive_state = self.state_extractor.extract(
            history_q, history_r, history_mask,
            interval_time=interval_time,
            response_time=response_time,
        )
        
        # 阶段 1: 构建候选集
        candidates, probs = self.candidate_builder.build(
            self.model, history_q, history_r, history_mask, cognitive_state
        )
        
        # 计算 Tau
        cognitive_state['tau'] = self.candidate_builder._compute_zpd(cognitive_state)
        
        # 阶段 2: MOPSO 优化
        evaluator = FitnessEvaluator(self.candidate_builder.qs_table, self.config)
        pso = DiscreteMOPSO(candidates, evaluator, cognitive_state, self.config)
        recommendation = pso.optimize(probs)
        
        # 阶段3: 后处理 (a) 多样性增强——贪心替换相似题; (b) 难度梯度排序: 按 P_q 升序，易→难
        recommendation = self._enforce_diversity(
            recommendation, candidates,
            self.candidate_builder.qs_table.cpu().numpy(),
            ild_threshold=0.70,
        )
        rec_probs = np.array([probs[np.where(candidates == q)[0][0]].item() if q in candidates else 0.5 for q in recommendation])
        sorted_idx = np.argsort(rec_probs)
        recommendation = recommendation[sorted_idx]
        rec_probs = rec_probs[sorted_idx]

        return recommendation, {
            'cognitive_state': cognitive_state,
            'rec_probs': rec_probs,
        }

    # ------------------------------------------------------------------
    def _enforce_diversity(self,
                           recommendation: np.ndarray,
                           candidates: np.ndarray,
                           qs_table: np.ndarray,
                           ild_threshold: float = 0.70) -> np.ndarray:
        """
        如果推荐列表的 ILD 低于阈値，贪心地用候选池中更多样的题目替换相似题。
        本质上是一个幗次贪心 DPP-like 方法：循环 max_iter 次。
        尝试替换相似度最高对中指数较大的题目，使 ILD 上升至阈値上方。
        """
        K = len(recommendation)
        if K < 2:
            return recommendation

        def _ild(rec: np.ndarray) -> float:
            vecs = qs_table[rec].astype(np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms < 1e-8, 1.0, norms)
            vecs_n = vecs / norms
            s = 0.0
            cnt = 0
            for i, j in combinations(range(K), 2):
                s += 1.0 - float(np.dot(vecs_n[i], vecs_n[j]))
                cnt += 1
            return s / cnt if cnt > 0 else 0.0

        rec = recommendation.copy()
        cand_set_full = set(candidates.tolist())

        for _attempt in range(K):            # 最多替换 K 次
            if _ild(rec) >= ild_threshold:
                break

            # 找相似度最高的对 (i, j)，替换其中指数较大的頹 j
            vecs = qs_table[rec].astype(np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms < 1e-8, 1.0, norms)
            vecs_n = vecs / norms

            min_dissim, worst_j = float('inf'), -1
            for i, j in combinations(range(K), 2):
                d = 1.0 - float(np.dot(vecs_n[i], vecs_n[j]))
                if d < min_dissim:
                    min_dissim, worst_j = d, j  # j 总是较大索引

            # 在候选池中找最能增大 ILD 的替换题
            rec_set = set(rec.tolist())
            best_gain, best_replacement = -float('inf'), -1
            others = [k for k in range(K) if k != worst_j]
            others_vn = vecs_n[others]

            for cand in cand_set_full:
                if cand in rec_set:
                    continue
                c_vec = qs_table[cand].astype(np.float32)
                c_norm = np.linalg.norm(c_vec)
                if c_norm < 1e-8:
                    continue
                c_vn = c_vec / c_norm
                avg_dissim = float(np.mean(1.0 - others_vn @ c_vn))
                if avg_dissim > best_gain:
                    best_gain, best_replacement = avg_dissim, cand

            # 只有替换确实能提升多样性时才操作
            if best_replacement == -1 or best_gain <= min_dissim:
                break
            rec[worst_j] = best_replacement

        return rec


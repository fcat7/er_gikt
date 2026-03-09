"""CCS-MOPSO-ER 推荐系统离线评估脚本
==================================================
位置：back/er/recommend_eval.py
功能：加载训练好的 FA-GIKT 模型，调用推荐系统，离线评估七大指标。

使用方式（从 back/er/ 目录运行）:
    # 快速测试（sample 数据集, 少量样本）:
    python recommend_eval.py --model "path/to/model.pt"

    # 完整评估（全量数据集, 全量评估）:
    python recommend_eval.py --full --model "path/to/model.pt"

    # 如果 TOML 中已配置 model_path，则无需 --model:
    python recommend_eval.py --full
"""
import os
import sys
import json
import time
import logging
import argparse
import numpy as np
import pandas as pd
from itertools import combinations
from datetime import datetime, timezone, timedelta
from tqdm import tqdm
import torch

# ---- 路径设置（使用 abspath 避免相对路径解析失败）----
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # back/er
_KT_DIR = os.path.join(os.path.dirname(_THIS_DIR), 'kt')  # back/kt
if _KT_DIR not in sys.path:
    sys.path.insert(0, _KT_DIR)

from config import get_config, DEVICE  # back/kt/config.py
from pso_recommend import RecommendationSystem  # back/er/pso_recommend.py

# back/er/baselines 用 importlib 加载，绕开 back/kt/baselines 的命名冲突
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    'er_baselines',
    os.path.join(_THIS_DIR, 'baselines', 'base_recommenders.py')
)
_ER_BASELINES = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_ER_BASELINES)

_spec_kt = _ilu.spec_from_file_location(
    'er_kt_baselines',
    os.path.join(_THIS_DIR, 'baselines', 'kt_recommenders.py')
)
_ER_KT_BASELINES = _ilu.module_from_spec(_spec_kt)
_spec_kt.loader.exec_module(_ER_KT_BASELINES)

# 模块级 logger（在 main() 中统一初始化文件 + 控制台双 Handler）
logger = logging.getLogger('recommend_eval')


def get_exp_config_path(is_full: bool = False) -> str:
    """构建实验配置文件路径，风格对齐 train_test.py 的 --full 模式"""
    config_type = 'full' if is_full else 'sample'
    return os.path.join(
        _THIS_DIR, 'config', 'experiments',
        f'exp_recommend_{config_type}.toml'
    )


class RecommendationEvaluator:
    """离线评估推荐系统（七指标体系，对齐框架 V∞ §2.7）"""

    def __init__(self, model_path: str, config_dict: dict):
        self.model_path = model_path
        self.config = config_dict
        self.dataset_name = config_dict.get('dataset_name', 'assist09')

        kt_config = get_config(self.dataset_name)
        self.data_dir = kt_config.PROCESSED_DATA_DIR
        self.metadata_path = os.path.join(self.data_dir, 'metadata.json')

        logger.info(f"📂 Loading metadata: {self.metadata_path}")
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        self.rec_system = RecommendationSystem(
            model_path, self.metadata_path, config_dict, device=DEVICE
        )

        # qs_table 用于 ILD / WKC 计算
        self.qs_table = self.rec_system.candidate_builder.qs_table.cpu().numpy()
        n_q = self.qs_table.shape[0]

        # 预计算全局热门题（基于训练集全量交互频率，用于 Novelty 修正）
        # 修正：排除 padding(idx=0)，扩展至 Top-200 保证统计有效性
        train_file = os.path.join(self.data_dir, 'train.parquet')
        logger.info(f"📊 Loading train data for global popularity: {train_file}")
        train_data = pd.read_parquet(train_file)
        global_freq = np.zeros(n_q, dtype=np.int64)
        for q_seq in train_data['q_seq']:
            arr = np.array(q_seq, dtype=np.int64)
            arr = arr[(arr > 0) & (arr < n_q)]  # 排除 padding(0) 与越界值
            np.add.at(global_freq, arr, 1)
        # Top-200 热门题（freq>0 排除从未出现的问题），更宽窗口保证 Novelty 统计意义
        sorted_by_freq = np.argsort(-global_freq)
        self.global_popular_qs = set(
            int(x) for x in sorted_by_freq[:200] if global_freq[int(x)] > 0
        )
        logger.info(f"  ↳ Global popular pool: {len(self.global_popular_qs)} qs "
                    f"(top freq={global_freq[sorted_by_freq[0]]} ~ {global_freq[sorted_by_freq[min(199, len(sorted_by_freq)-1)]]})")

        test_file = os.path.join(self.data_dir, 'test.parquet')
        logger.info(f"📂 Loading test data: {test_file}")
        self.test_data = pd.read_parquet(test_file)

        # 保留路径供基线推荐器使用
        self.train_parquet_path = train_file

    def evaluate_offline(self, n_samples: int = None,
                          recommender=None,
                          mode_label: str = 'ours') -> pd.DataFrame:
        """离线评估（留一法）
        Args:
            n_samples: 限制评估学生数
            recommender: 如果为 None 使用 self.rec_system，否则用传入的基线推荐器
            mode_label: 日志前缀，区分不同运行模式
        """""
        results = []
        min_seq_len = self.config.get('min_seq_len', 10)
        K = self.config.get('K', 5)

        valid_samples = []
        for _, row in self.test_data.iterrows():
            q_seq = row['q_seq']
            if len(q_seq) >= min_seq_len + K:
                valid_samples.append(row)

        valid_df = pd.DataFrame(valid_samples).reset_index(drop=True)

        if n_samples is not None:
            valid_df = valid_df.iloc[:n_samples]

        _rec = recommender if recommender is not None else self.rec_system
        logger.info(f"🚀 [{mode_label}] Evaluating {len(valid_df)} test samples (K={K}, min_seq_len={min_seq_len})")
        eval_start = time.time()

        for idx, row in tqdm(valid_df.iterrows(), total=len(valid_df), desc=f"{mode_label}"):
            try:
                sample_start = time.time()
                all_qs = np.array(row['q_seq'])
                all_rs = np.array(row['r_seq'])

                T = len(all_qs)
                history_qs = all_qs[:T - K]
                history_rs = all_rs[:T - K]
                gt_qs = all_qs[T - K:]

                history_q = torch.tensor(np.array(history_qs, dtype=np.int64).reshape(1, -1), dtype=torch.long).to(DEVICE)
                history_r = torch.tensor(np.array(history_rs, dtype=np.int64).reshape(1, -1), dtype=torch.long).to(DEVICE)
                history_mask = torch.ones_like(history_q).to(DEVICE)

                # 时间特征（CognitiveRNNCell 需要）
                interval_time = None
                response_time = None
                if 't_interval' in row.index:
                    t_int = np.array(row['t_interval'], dtype=np.float32)[:T - K]
                    interval_time = torch.tensor(t_int.reshape(1, -1), dtype=torch.float32).to(DEVICE)
                if 't_response' in row.index:
                    t_resp = np.array(row['t_response'], dtype=np.float32)[:T - K]
                    response_time = torch.tensor(t_resp.reshape(1, -1), dtype=torch.float32).to(DEVICE)

                recommendation, info = _rec.recommend(
                    history_q, history_r, history_mask, K=K,
                    interval_time=interval_time,
                    response_time=response_time,
                )

                rec_probs = info.get('rec_probs', None)
                cog_state = info.get('cognitive_state', {})
                tau = cog_state.get('tau', 0.55)
                m_k = cog_state.get('m_k', None)

                metrics = self._compute_metrics(
                    recommendation, gt_qs, history_qs,
                    rec_probs=rec_probs, tau=tau, m_k=m_k
                )
                metrics['student_idx'] = idx
                metrics['mode'] = mode_label
                metrics['elapsed_s'] = round(time.time() - sample_start, 2)
                results.append(metrics)
                logger.debug(f"  Sample {idx:4d} | DM={metrics['dm']:.4f} "
                             f"WKC={metrics['wkc']:.4f} ILD={metrics['ild']:.4f} "
                             f"Novelty={metrics['novelty']:.2f} "
                             f"SkillHit={metrics['skill_hit_rate']:.4f} "
                             f"[{metrics['elapsed_s']:.1f}s]")
                # 写入日志
                logger.info(f"  Sample {idx:4d} | DM={metrics['dm']:.4f} "
                             f"WKC={metrics['wkc']:.4f} ILD={metrics['ild']:.4f} "
                             f"Novelty={metrics['novelty']:.2f} "
                             f"SkillHit={metrics['skill_hit_rate']:.4f} "
                             f"[{metrics['elapsed_s']:.1f}s]")

            except Exception as e:
                logger.warning(f"⚠️  Sample {idx} failed: {e}")
                continue

        total_elapsed = time.time() - eval_start
        logger.info(f"✅ Evaluation done: {len(results)}/{len(valid_df)} samples "
                    f"in {total_elapsed:.1f}s (avg {total_elapsed/max(len(results),1):.1f}s/sample)")
        return pd.DataFrame(results)

    def _compute_metrics(self, recommendation: np.ndarray,
                        gt_qs: np.ndarray, history_qs: np.ndarray,
                        rec_probs: np.ndarray = None,
                        tau: float = 0.55,
                        m_k: np.ndarray = None) -> dict:
        """
        八指标评价体系（框架 V∞ §2.7 + 修正版）:
            核心教育指标: SkillHit@K（技能级命中率，替代题目级Precision）
            多样: ILD (Jaccard 平均相异度), Novelty（全局训练集频率修正）
            教育: DM (难度匹配度), WKC (弱点覆盖率)
            参考: precision, recall, f1_score（保留在CSV中，预期为0，符合干预式推荐特性）
        """
        K = len(recommendation)
        n_q = self.qs_table.shape[0]

        # --- SkillHit@K（技能级命中率）: |Skills(R) ∩ Skills(GT)| / |Skills(GT)| ---
        # 度量推荐题目的知识点覆盖与学生即将触及的知识点的交集，适用于干预式推荐场景
        rec_skills = set()
        for q in recommendation:
            rec_skills.update(np.where(self.qs_table[q] > 0)[0])
        gt_skills = set()
        for q in gt_qs:
            if q < n_q:
                gt_skills.update(np.where(self.qs_table[q] > 0)[0])
        skill_hit_rate = len(rec_skills & gt_skills) / max(len(gt_skills), 1) if gt_skills else 0.0

        # --- 保留题目级 Precision/Recall/F1（预期为0，体现干预式推荐与预测式推荐的范式差异）---
        hits = len(set(recommendation) & set(gt_qs))
        precision = hits / K if K > 0 else 0.0
        recall = hits / len(set(gt_qs)) if len(set(gt_qs)) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        # --- ILD (多样性): 1/C(K,2) * Σ_{i<j} (1 - Jaccard(Skills_i, Skills_j)) ---
        if K >= 2:
            ild_sum = 0.0
            count = 0
            for i, j in combinations(range(K), 2):
                skills_i = set(np.where(self.qs_table[recommendation[i]] > 0)[0])
                skills_j = set(np.where(self.qs_table[recommendation[j]] > 0)[0])
                union = skills_i | skills_j
                inter = skills_i & skills_j
                jaccard = len(inter) / len(union) if len(union) > 0 else 0.0
                ild_sum += (1.0 - jaccard)
                count += 1
            ild = ild_sum / count if count > 0 else 0.0
        else:
            ild = 0.0

        # --- Novelty（全局修正版）: 1 - |R ∩ GlobalPopular_100| / K ---
        # 使用训练集全量交互频率统计热门题（在__init__中预计算为 self.global_popular_qs）
        # 避免使用个人历史导致的虚假 Novelty=1.0 问题
        novelty = 1.0 - len(set(recommendation) & self.global_popular_qs) / K if K > 0 else 0.0

        # --- DM (难度匹配): 1 - (1/K) Σ |P_ri - τ_s| ---
        if rec_probs is not None:
            dm = 1.0 - np.mean(np.abs(rec_probs - tau))
        else:
            dm = float('nan')

        # --- WKC (弱点覆盖率): Σ_covered(1-m_k) / Σ_weak(1-m_k) ---
        if m_k is not None:
            weak_skills = np.where(m_k < 0.7)[0]
            if len(weak_skills) > 0:
                total_weak_weight = np.sum(1.0 - m_k[weak_skills])
                covered_weak = set()
                for q in recommendation:
                    related = np.where(self.qs_table[q] > 0)[0]
                    covered_weak.update(np.intersect1d(related, weak_skills))
                covered_weight = np.sum(1.0 - m_k[list(covered_weak)]) if covered_weak else 0.0
                wkc = covered_weight / total_weak_weight if total_weak_weight > 0 else 0.0
                n_weak = len(weak_skills)  # 记录薄弱技能数量，供论文报告
            else:
                wkc = 1.0  # 无弱技能 → 覆盖率视为满分
                n_weak = 0
        else:
            wkc = float('nan')
            n_weak = -1

        return {
            'skill_hit_rate': skill_hit_rate,   # 核心：技能级命中率（非零，替代题目级Precision）
            'ild': ild,
            'novelty': novelty,                  # 已修正：基于全局训练集热门题
            'dm': dm,
            'wkc': wkc,
            'n_weak_skills': n_weak,             # 辅助字段：薄弱技能数量，用于解释WKC量级
            'precision': precision,              # 保留（预期为0，干预式推荐特性）
            'recall': recall,
            'f1_score': f1,
        }

    def print_results(self, df: pd.DataFrame, mode_label: str = 'ours'):
        """打印评估结果汇总（主报告 + 参考说明）"""
        logger.info("=" * 60)
        logger.info(f"  OFFLINE EVALUATION RESULTS  [{mode_label.upper()}]")
        logger.info("=" * 60)
        # 主报告：有效教育推荐指标
        main_metrics = [
            ('skill_hit_rate', 'SkillHit@K '),
            ('ild',            'ILD        '),
            ('novelty',        'Novelty    '),
            ('dm',             'DM         '),
            ('wkc',            'WKC        '),
        ]
        for col, label in main_metrics:
            if col in df.columns:
                mean_val = df[col].mean()
                std_val = df[col].std()
                logger.info(f"  {label}: {mean_val:.4f} ± {std_val:.4f}")
        if 'n_weak_skills' in df.columns:
            logger.info(f"  [avg weak skills per student: {df['n_weak_skills'].mean():.1f}]")
        logger.info("-" * 60)
        # 参考信息：干预式推荐中预期为0的传统指标
        logger.info("  [REF] Item-level metrics (expected 0.0 in interventional rec):")
        for col in ['precision', 'recall', 'f1_score']:
            if col in df.columns:
                logger.info(f"    {col:12s}: {df[col].mean():.4f}")
        logger.info("=" * 60)


def _print_comparison_table(all_results: dict):
    """打印多方案对比表格（--baseline all 时使用）"""
    metrics = [
        ('dm',             'DM ↑        '),
        ('wkc',            'WKC ↑       '),
        ('ild',            'ILD ↑       '),
        ('novelty',        'Novelty ↑   '),
        ('skill_hit_rate', 'SkillHit ↑  '),
    ]
    modes = list(all_results.keys())
    col_w = 16

    logger.info("")
    logger.info("=" * (14 + col_w * len(modes)))
    logger.info("  COMPARISON TABLE")
    logger.info("=" * (14 + col_w * len(modes)))
    header = f"  {'Metric':<12}" + "".join(f"{m.upper():>{col_w}}" for m in modes)
    logger.info(header)
    logger.info("-" * (14 + col_w * len(modes)))
    for col, label in metrics:
        row = f"  {label:<12}"
        best_val = -1.0
        vals = []
        for m in modes:
            df = all_results[m]
            v = df[col].mean() if col in df.columns else float('nan')
            vals.append(v)
            if not np.isnan(v) and v > best_val:
                best_val = v
        for v in vals:
            tag = " *" if (not np.isnan(v) and abs(v - best_val) < 1e-6) else "  "
            cell = f"{v:.4f}{tag}" if not np.isnan(v) else "  n/a  "
            row += f"{cell:>{col_w}}"
        logger.info(row)
    logger.info("=" * (14 + col_w * len(modes)))
    logger.info("  * = best in row")
    logger.info("")


def main():
    # ---- 1. CLI 解析（风格对齐 train_test.py）----
    parser = argparse.ArgumentParser(description="CCS-MOPSO-ER Offline Evaluation")
    parser.add_argument('--full', action='store_true',
                        help='Use full dataset config (default: sample)')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to .pt model (optional if specified in TOML)')
    parser.add_argument('--n_eval_samples', type=int, default=None,
                        help='Limit number of evaluated students')
    parser.add_argument('--debug', action='store_true',
                        help='Enable DEBUG level logging (per-sample details)')
    parser.add_argument('--baseline', type=str, default=None,
                        choices=['greedy', 'random', 'popularity', 'dkt_greedy', 'dkvmn_greedy', 'all'],
                        help='Run baseline instead of (or alongside) full model. '
                            'all = run ours + available baselines and print comparison table')
    parser.add_argument('--dkt_model', type=str, default=None,
                        help='Path to DKT checkpoint (.pt), used by dkt_greedy baseline')
    parser.add_argument('--dkvmn_model', type=str, default=None,
                        help='Path to DKVMN checkpoint (.pt), used by dkvmn_greedy baseline')
    args = parser.parse_args()

    # ---- 2. 时间戳（UTC+8 北京时间，对齐 train_test.py）----
    beijing_time = datetime.now(timezone(timedelta(hours=8)))
    time_now = beijing_time.strftime('%Y%m%d_%H%M%S')

    # ---- 3. 日志系统：文件 + 控制台双 Handler ----
    log_dir = os.path.join(_THIS_DIR, 'output', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'recommend_eval_{time_now}.log')

    log_level = logging.DEBUG if args.debug else logging.INFO
    # Windows GBK 终端防御：为 StreamHandler 强制 utf-8 输出
    _stream_handler = logging.StreamHandler(sys.stdout)
    _stream_handler.setFormatter(
        logging.Formatter('[%(asctime)s] [%(levelname)-7s] %(message)s',
                          datefmt='%Y-%m-%d %H:%M:%S')
    )
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass
    logging.basicConfig(
        level=log_level,
        format='[%(asctime)s] [%(levelname)-7s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            _stream_handler,
        ]
    )
    logger.info(f"{'='*60}")
    logger.info(f"  CCS-MOPSO-ER  Offline Evaluation")
    logger.info(f"  Run ID  : {time_now}")
    logger.info(f"  Log File: {log_file}")
    logger.info(f"  Device  : {DEVICE}")
    logger.info(f"{'='*60}")

    # ---- 4. 加载 TOML 配置 ----
    import toml
    config_file = get_exp_config_path(is_full=args.full)
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config not found: {config_file}")

    logger.info(f"📋 Config : {config_file}")
    with open(config_file, 'r', encoding='utf-8') as f:
        toml_conf = toml.load(f)

    # 展平 TOML sections → flat dict
    config_dict = {}
    for sect in toml_conf.values():
        if isinstance(sect, dict):
            config_dict.update(sect)

    # ---- 5. 模型路径：命令行 > TOML > 报错 ----
    model_path = args.model or config_dict.get('model_path', '')
    if not model_path or not os.path.exists(model_path):
        raise ValueError(
            f"Valid model path required.\n"
            f"  Via CLI:  python recommend_eval.py --model path/to/model.pt\n"
            f"  Via TOML: set model_path in {config_file}"
        )
    logger.info(f"🤖 Model  : {model_path}")

    # 安全检查：数据集名与模型路径是否匹配
    dataset_name = config_dict.get('dataset_name', 'assist09')
    base_dataset = dataset_name.split('-')[0]  # assist09-sample_20% → assist09
    if base_dataset not in model_path:
        logger.warning(f"⚠️  Dataset='{dataset_name}' vs model='{os.path.basename(model_path)}' — 请确认匹配！")

    # ---- 6. 覆盖评估样本数 ----
    if args.n_eval_samples is not None:
        config_dict['n_eval_samples'] = args.n_eval_samples
    logger.info(f"📌 Samples: {config_dict.get('n_eval_samples', 'all')} | K={config_dict.get('K', 5)}")
    logger.info(f"{'='*60}")

    # ---- 7. 运行评估 ----
    run_start = time.time()
    evaluator = RecommendationEvaluator(model_path, config_dict)
    n_eval = config_dict.get('n_eval_samples', None)

    all_results = {}  # mode_label -> DataFrame

    if args.baseline in (None, 'all'):
        # 运行完整模型
        res_df = evaluator.evaluate_offline(n_samples=n_eval, mode_label='ours')
        evaluator.print_results(res_df, mode_label='ours')
        all_results['ours'] = res_df

    if args.baseline in ('greedy', 'all'):
        greedy = _ER_BASELINES.GreedyRecommender(model_path, evaluator.metadata_path, config_dict, device=DEVICE)
        res_g = evaluator.evaluate_offline(n_samples=n_eval, recommender=greedy, mode_label='greedy')
        evaluator.print_results(res_g, mode_label='greedy')
        all_results['greedy'] = res_g

    if args.baseline in ('random', 'all'):
        random_rec = _ER_BASELINES.RandomRecommender(model_path, evaluator.metadata_path, config_dict, device=DEVICE)
        res_r = evaluator.evaluate_offline(n_samples=n_eval, recommender=random_rec, mode_label='random')
        evaluator.print_results(res_r, mode_label='random')
        all_results['random'] = res_r

    if args.baseline in ('popularity', 'all'):
        pop_rec = _ER_BASELINES.PopularityRecommender(
            model_path, evaluator.metadata_path, config_dict,
            device=DEVICE, train_parquet_path=evaluator.train_parquet_path
        )
        res_p = evaluator.evaluate_offline(n_samples=n_eval, recommender=pop_rec, mode_label='popularity')
        evaluator.print_results(res_p, mode_label='popularity')
        all_results['popularity'] = res_p

    if args.baseline in ('dkt_greedy', 'all'):
        dkt_model_path = args.dkt_model or config_dict.get('dkt_model_path', None)
        if dkt_model_path and os.path.exists(dkt_model_path):
            dkt_rec = _ER_KT_BASELINES.DKTGreedyRecommender(
                dkt_model_path, evaluator.metadata_path, config_dict, device=DEVICE
            )
            res_dkt = evaluator.evaluate_offline(n_samples=n_eval, recommender=dkt_rec, mode_label='dkt_greedy')
            evaluator.print_results(res_dkt, mode_label='dkt_greedy')
            all_results['dkt_greedy'] = res_dkt
        else:
            logger.warning('⚠️ Skip dkt_greedy: missing --dkt_model (or dkt_model_path in TOML)')

    if args.baseline in ('dkvmn_greedy', 'all'):
        dkvmn_model_path = args.dkvmn_model or config_dict.get('dkvmn_model_path', None)
        if dkvmn_model_path and os.path.exists(dkvmn_model_path):
            dkvmn_rec = _ER_KT_BASELINES.DKVMNGreedyRecommender(
                dkvmn_model_path, evaluator.metadata_path, config_dict, device=DEVICE
            )
            res_dkvmn = evaluator.evaluate_offline(n_samples=n_eval, recommender=dkvmn_rec, mode_label='dkvmn_greedy')
            evaluator.print_results(res_dkvmn, mode_label='dkvmn_greedy')
            all_results['dkvmn_greedy'] = res_dkvmn
        else:
            logger.warning('⚠️ Skip dkvmn_greedy: missing --dkvmn_model (or dkvmn_model_path in TOML)')

    # ---- 对比表格（--baseline all 时打印）----
    if args.baseline == 'all' and len(all_results) > 1:
        _print_comparison_table(all_results)

    # ---- 合并所有结果保存 ----
    if not all_results:
        raise RuntimeError('No evaluation results produced. Check baseline arguments and model paths.')
    combined_df = pd.concat(all_results.values(), ignore_index=True)

    # ---- 8. 保存结果 ----
    output_dir = config_dict.get('save_dir', 'output/recommendation_sample' if not args.full else 'output/recommendation_full')
    os.makedirs(output_dir, exist_ok=True)
    config_type = 'full' if args.full else 'sample'
    # 文件名包含 baseline 标签，避免覆盖
    baseline_tag = f'_{args.baseline}' if args.baseline else ''
    out_file = os.path.join(output_dir, f"recommend_eval_{config_type}{baseline_tag}.csv")
    combined_df.to_csv(out_file, index=False)

    total_run = time.time() - run_start
    logger.info(f"💾 Results saved → {out_file}")
    logger.info(f"⏱️  Total wall time: {total_run:.1f}s")
    logger.info(f"{'='*60}")


# 使用示例（从 back/er/ 目录运行）:
#   python recommend_eval.py --model "H:\er_gikt\back\kt\output\model\xxx.pt"
#   python recommend_eval.py --full --model "H:\er_gikt\back\kt\output\model\xxx.pt"
#   python recommend_eval.py --full   # (需在 TOML 中配置 model_path)
if __name__ == '__main__':
    main()

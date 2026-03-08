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
import argparse
import numpy as np
import pandas as pd
from itertools import combinations
from tqdm import tqdm
import torch

try:
    from icecream import ic
except ImportError:
    ic = print

# ---- 路径设置（使用 abspath 避免相对路径解析失败）----
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # back/er
_KT_DIR = os.path.join(os.path.dirname(_THIS_DIR), 'kt')  # back/kt
if _KT_DIR not in sys.path:
    sys.path.insert(0, _KT_DIR)

from config import get_config, DEVICE  # back/kt/config.py
from pso_recommend import RecommendationSystem  # back/er/pso_recommend.py


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

        ic(f"Loading metadata from {self.metadata_path}")
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        self.rec_system = RecommendationSystem(
            model_path, self.metadata_path, config_dict, device=DEVICE
        )

        # qs_table 用于 ILD / WKC 计算
        self.qs_table = self.rec_system.candidate_builder.qs_table.cpu().numpy()

        test_file = os.path.join(self.data_dir, 'test.parquet')
        ic(f"Loading test data from {test_file}")
        self.test_data = pd.read_parquet(test_file)

    def evaluate_offline(self, n_samples: int = None) -> pd.DataFrame:
        """离线评估（留一法：取每条序列最后 K 步作为 GT）"""
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

        ic(f"Evaluating {len(valid_df)} test samples (K={K})")

        for idx, row in tqdm(valid_df.iterrows(), total=len(valid_df), desc="Evaluating"):
            try:
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

                recommendation, info = self.rec_system.recommend(
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
                results.append(metrics)

            except Exception as e:
                ic(f"Error processing sample {idx}: {e}")
                continue

        return pd.DataFrame(results)

    def _compute_metrics(self, recommendation: np.ndarray,
                        gt_qs: np.ndarray, history_qs: np.ndarray,
                        rec_probs: np.ndarray = None,
                        tau: float = 0.55,
                        m_k: np.ndarray = None) -> dict:
        """
        七指标评价体系（框架 V∞ §2.7）:
            传统: Precision@K, Recall@K, F1@K
            多样: ILD (Jaccard 平均相异度), Novelty
            教育: DM (难度匹配度), WKC (弱点覆盖率)
        """
        K = len(recommendation)

        # --- Precision / Recall / F1 ---
        hits = len(set(recommendation) & set(gt_qs))
        precision = hits / K if K > 0 else 0.0
        recall = hits / len(set(gt_qs)) if len(set(gt_qs)) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        # --- ILD (多样性): 1/(K(K-1)) * Σ_{i≠j} (1 - Jaccard(Skills_i, Skills_j)) ---
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

        # --- Novelty: 1 - |R ∩ Popular_100| / K ---
        n_q = self.qs_table.shape[0]
        popular_qs = set(
            np.bincount(history_qs, minlength=n_q).argsort()[-100:]
        )
        novelty = 1.0 - len(set(recommendation) & popular_qs) / K if K > 0 else 0.0

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
            else:
                wkc = 1.0  # 无弱技能 → 覆盖率视为满分
        else:
            wkc = float('nan')

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'ild': ild,
            'novelty': novelty,
            'dm': dm,
            'wkc': wkc,
        }

    def print_results(self, df: pd.DataFrame):
        """打印评估结果汇总"""
        print("=" * 60)
        print("  OFFLINE RECOMMENDATION EVALUATION RESULTS")
        print("=" * 60)
        metrics = ['precision', 'recall', 'f1_score', 'ild', 'novelty', 'dm', 'wkc']
        for metric in metrics:
            if metric in df.columns:
                mean_val = df[metric].mean()
                std_val = df[metric].std()
                print(f"  {metric:12s}: {mean_val:.4f} ± {std_val:.4f}")
        print("=" * 60)


def main():
    # CLI 风格对齐 train_test.py: 默认 sample, --full 切全量
    parser = argparse.ArgumentParser(description="CCS-MOPSO-ER Offline Evaluation")
    parser.add_argument('--full', action='store_true',
                        help='Use full dataset config (default: sample)')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to .pt model (optional if specified in TOML)')
    parser.add_argument('--n_eval_samples', type=int, default=None,
                        help='Limit number of evaluated students')
    args = parser.parse_args()

    # 加载 TOML 配置
    import toml
    config_file = get_exp_config_path(is_full=args.full)
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config not found: {config_file}")

    ic(f"Loading config: {config_file}")
    with open(config_file, 'r', encoding='utf-8') as f:
        toml_conf = toml.load(f)

    # 展平 TOML sections → flat dict
    config_dict = {}
    for sect in toml_conf.values():
        if isinstance(sect, dict):
            config_dict.update(sect)

    # 模型路径：命令行 > TOML > 报错
    model_path = args.model or config_dict.get('model_path', '')
    if not model_path or not os.path.exists(model_path):
        raise ValueError(
            f"Valid model path required.\n"
            f"  Via CLI:  python recommend_eval.py --model path/to/model.pt\n"
            f"  Via TOML: set model_path in {config_file}"
        )

    # 安全检查：数据集名与模型路径是否匹配
    dataset_name = config_dict.get('dataset_name', 'assist09')
    base_dataset = dataset_name.split('-')[0]  # assist09-sample_20% → assist09
    if base_dataset not in model_path:
        print(f"⚠️ WARNING: dataset='{dataset_name}' vs model='{os.path.basename(model_path)}'")
        print("  请确认模型与数据集是否匹配！")

    # 覆盖评估样本数
    if args.n_eval_samples is not None:
        config_dict['n_eval_samples'] = args.n_eval_samples

    # 运行评估
    evaluator = RecommendationEvaluator(model_path, config_dict)
    n_eval = config_dict.get('n_eval_samples', None)
    res_df = evaluator.evaluate_offline(n_samples=n_eval)
    evaluator.print_results(res_df)

    # 保存结果
    output_dir = config_dict.get('save_dir', 'output/recommendation')
    os.makedirs(output_dir, exist_ok=True)
    config_type = 'full' if args.full else 'sample'
    out_file = os.path.join(output_dir, f"recommend_eval_{config_type}.csv")
    res_df.to_csv(out_file, index=False)
    ic(f"Results saved to {out_file}")


# 使用示例（从 back/er/ 目录运行）:
#   python recommend_eval.py --model "H:\er_gikt\back\kt\output\model\xxx.pt"
#   python recommend_eval.py --full --model "H:\er_gikt\back\kt\output\model\xxx.pt"
#   python recommend_eval.py --full   # (需在 TOML 中配置 model_path)
if __name__ == '__main__':
    main()

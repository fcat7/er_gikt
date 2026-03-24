import os
import sys
import json
import time
import logging
import argparse
import toml
import numpy as np
import pandas as pd
from itertools import combinations
from datetime import datetime, timezone, timedelta
from tqdm import tqdm
import torch
import scipy.sparse as sp

# ---- 1. 路径与环境配置 ----
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # back/er
_KT_DIR = os.path.join(os.path.dirname(_THIS_DIR), 'kt')  # back/kt
if _KT_DIR not in sys.path:
    sys.path.insert(0, _KT_DIR)

from config import get_config, DEVICE  # noqa
logger = logging.getLogger('recommend_eval')

def get_exp_config_path(is_full: bool = False) -> str:
    config_type = 'full' if is_full else 'sample'
    return os.path.join(_THIS_DIR, 'config', 'experiments', f'exp_recommend_{config_type}.toml')

# ---- 2. 数据提供层 ----
class RecDataLoader:
    """负责加载和预处理元数据、题库特征、流行度与测试序列"""
    def __init__(self, config_dict: dict):
        self.config = config_dict
        dataset_name = config_dict.get('dataset_name', 'assist09')
        kt_config = get_config(dataset_name)
        self.data_dir = kt_config.PROCESSED_DATA_DIR
        
        self.metadata_path = os.path.join(self.data_dir, 'metadata.json')
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata not found: {self.metadata_path}")
            
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
            
        self.train_parquet_path = os.path.join(self.data_dir, 'train.parquet')
        self.test_parquet_path = os.path.join(self.data_dir, 'test.parquet')
        self.qs_table_path = os.path.join(self.data_dir, 'qs_table.npz')
        
        # Lazy Loading
        self._qs_table = None
        self._global_popular_qs = None
        self._test_samples = None

    @property
    def qs_table(self) -> np.ndarray:
        if self._qs_table is None:
            self._qs_table = sp.load_npz(self.qs_table_path).toarray()
        return self._qs_table
        
    @property
    def global_popular_qs(self) -> set:
        if self._global_popular_qs is None:
            logger.info(f"📊 计算全局热题(基于 {self.train_parquet_path})...")
            train_data = pd.read_parquet(self.train_parquet_path)
            n_q = self.metadata['metrics']['n_question']
            global_freq = np.zeros(n_q, dtype=np.int64)
            for q_seq in train_data['q_seq']:
                arr = np.array(q_seq, dtype=np.int64)
                arr = arr[(arr > 0) & (arr < n_q)]
                np.add.at(global_freq, arr, 1)
            sorted_by_freq = np.argsort(-global_freq)
            # 宽窗口保证 Novelty 统计意义 (Top 200)
            self._global_popular_qs = set(int(x) for x in sorted_by_freq[:200] if global_freq[int(x)] > 0)
        return self._global_popular_qs

    def get_eval_samples(self, start_idx: int = 0, end_idx: int = None) -> pd.DataFrame:
        if self._test_samples is None:
            logger.info(f"📂 加载测试集: {self.test_parquet_path}")
            min_seq_len = self.config.get('min_seq_len', 10)
            K = self.config.get('K', 5)
            
            test_data = pd.read_parquet(self.test_parquet_path)
            valid_samples = [row for _, row in test_data.iterrows() if len(row['q_seq']) >= min_seq_len + K]
            self._test_samples = pd.DataFrame(valid_samples).reset_index(drop=True)
            logger.info(f"  ↳ 有效长序列样本: {len(self._test_samples)}")
            
        df = self._test_samples
        _end_idx = end_idx if end_idx is not None else len(df)
        return df.iloc[start_idx:_end_idx]

# ---- 3. 推荐器静态工厂 ----
class RecommenderFactory:
    """按需实例化对应策略的推荐器，避免冗余资源占用"""
    @staticmethod
    def create(mode: str, data_loader: RecDataLoader, config_dict: dict, args):
        import importlib.util as _ilu
        def _load_module(name, path):
            spec = _ilu.spec_from_file_location(name, path)
            mod = _ilu.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod

        _base = _load_module('er_base', os.path.join(_THIS_DIR, 'baselines', 'base_recommenders.py'))
        _kt = _load_module('er_kt', os.path.join(_THIS_DIR, 'baselines', 'kt_recommenders.py'))

        metadata_path = data_loader.metadata_path
        
        if mode == 'ours':
            from pso_recommend import RecommendationSystem
            model_path = args.model or config_dict.get('model_path', '')
            if not model_path: raise ValueError("Ours mode requires model path.")
            logger.info(f"🔌 Loading GIKT from {model_path}...")
            return RecommendationSystem(model_path, metadata_path, config_dict, device=DEVICE)

        elif mode == 'greedy':
            path = args.model or config_dict.get('model_path', '')
            return _base.GreedyRecommender(path, metadata_path, config_dict, device=DEVICE)

        elif mode == 'random':
            path = args.model or config_dict.get('model_path', '')
            return _base.RandomRecommender(path, metadata_path, config_dict, device=DEVICE)

        elif mode == 'popularity':
            path = args.model or config_dict.get('model_path', '')
            return _base.PopularityRecommender(path, metadata_path, config_dict, device=DEVICE, train_parquet_path=data_loader.train_parquet_path)

        elif mode == 'dkt_greedy':
            path = args.dkt_model or config_dict.get('dkt_model_path', '')
            if not os.path.exists(path): raise FileNotFoundError(f"DKT model '{path}' not found.")
            return _kt.DKTGreedyRecommender(path, metadata_path, config_dict, device=DEVICE)

        elif mode == 'dkvmn_greedy':
            path = args.dkvmn_model or config_dict.get('dkvmn_model_path', '')
            if not os.path.exists(path): raise FileNotFoundError(f"DKVMN model '{path}' not found.")
            return _kt.DKVMNGreedyRecommender(path, metadata_path, config_dict, device=DEVICE)

        else:
            raise ValueError(f"Unknown recommender strategy: {mode}")

# ---- 4. 无状态指标计算引擎 ----
class MetricsCalculator:
    """仅接受原始数组，纯函数计算评价指标"""
    def __init__(self, qs_table: np.ndarray, global_popular_qs: set):
        self.qs_table = qs_table
        self.global_popular_qs = global_popular_qs
        self.n_q = self.qs_table.shape[0]

    def compute(self, recommendation: np.ndarray, gt_qs: np.ndarray, 
                rec_probs: np.ndarray = None, tau: float = 0.55, m_k: np.ndarray = None) -> dict:
        K = len(recommendation)
        
        # SkillHit@K
        rec_skills = set()
        for q in recommendation:
            rec_skills.update(np.where(self.qs_table[q] > 0)[0])
        gt_skills = set()
        for q in gt_qs:
            if q < self.n_q: gt_skills.update(np.where(self.qs_table[q] > 0)[0])
        skill_hit_rate = len(rec_skills & gt_skills) / max(len(gt_skills), 1) if gt_skills else 0.0

        # ILD
        ild = 0.0
        if K >= 2:
            ild_sum, count = 0.0, 0
            for i, j in combinations(range(K), 2):
                skills_i = set(np.where(self.qs_table[recommendation[i]] > 0)[0])
                skills_j = set(np.where(self.qs_table[recommendation[j]] > 0)[0])
                union = skills_i | skills_j
                inter = skills_i & skills_j
                if len(union) > 0: ild_sum += (1.0 - len(inter) / len(union))
                count += 1
            ild = ild_sum / count if count > 0 else 0.0

        # Novelty (基于全局热门的惩罚)
        novelty = 1.0 - len(set(recommendation) & self.global_popular_qs) / K if K > 0 else 0.0

        # DM
        dm = 1.0 - np.mean(np.abs(rec_probs - tau)) if rec_probs is not None else float('nan')

        # WKC
        wkc, n_weak = float('nan'), -1
        if m_k is not None:
            # 采用相对阈值：个体平均掌握度及一定下限，避免所有技能被判定为弱项
            weak_threshold = float(np.mean(m_k))
            weak_skills = np.where(m_k < weak_threshold)[0]
            if len(weak_skills) > 0:
                total_weak_weight = np.sum(1.0 - m_k[weak_skills])
                covered_weak = set()
                for q in recommendation:
                    related = np.where(self.qs_table[q] > 0)[0]
                    covered_weak.update(np.intersect1d(related, weak_skills))
                cov_val = np.sum(1.0 - m_k[list(covered_weak)]) if covered_weak else 0.0
                wkc = cov_val / total_weak_weight if total_weak_weight > 0 else 0.0
                n_weak = len(weak_skills)
            else:
                wkc, n_weak = 1.0, 0

        # Reference bounds
        hits = len(set(recommendation) & set(gt_qs))
        precision = hits / K if K > 0 else 0.0
        recall = hits / len(set(gt_qs)) if len(set(gt_qs)) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        return {
            'skill_hit_rate': skill_hit_rate, 'ild': ild, 'novelty': novelty,
            'dm': dm, 'wkc': wkc, 'n_weak_skills': n_weak,
            'precision': precision, 'recall': recall, 'f1_score': f1,
        }

# ---- 5. 核心测试控制流 ----
class RecommendationEngine:
    def __init__(self, data_loader: RecDataLoader, config_dict: dict):
        self.config = config_dict
        self.metrics_engine = MetricsCalculator(data_loader.qs_table, data_loader.global_popular_qs)

    def run_eval(self, recommender, df_samples: pd.DataFrame, mode_label: str) -> pd.DataFrame:
        K = self.config.get('K', 5)
        results = []
        
        logger.info(f"🚀 [{mode_label.upper()}] Starting evaluate loop on {len(df_samples)} samples...")
        start_time = time.time()

        for idx, row in tqdm(df_samples.iterrows(), total=len(df_samples), desc=f"{mode_label}"):
            try:
                t0 = time.time()
                all_qs, all_rs = np.array(row['q_seq']), np.array(row['r_seq'])
                T = len(all_qs)
                history_qs, gt_qs = all_qs[:T - K], all_qs[T - K:]
                history_rs = all_rs[:T - K]

                h_q = torch.tensor(history_qs.reshape(1, -1), dtype=torch.long, device=DEVICE)
                h_r = torch.tensor(history_rs.reshape(1, -1), dtype=torch.long, device=DEVICE)
                h_mask = torch.ones_like(h_q)

                t_int, t_resp = None, None
                if 't_interval' in row.index:
                    t_int = torch.tensor(np.array(row['t_interval'], dtype=np.float32)[:T-K].reshape(1, -1), device=DEVICE)
                if 't_response' in row.index:
                    t_resp = torch.tensor(np.array(row['t_response'], dtype=np.float32)[:T-K].reshape(1, -1), device=DEVICE)

                # Recommender Call
                rec_ids, info = recommender.recommend(h_q, h_r, h_mask, K=K, interval_time=t_int, response_time=t_resp)
                
                # Metrics Call
                c_state = info.get('cognitive_state', {})
                metrics = self.metrics_engine.compute(
                    rec_ids, gt_qs,
                    rec_probs=info.get('rec_probs'), 
                    tau=c_state.get('tau', 0.55), 
                    m_k=c_state.get('m_k')
                )
                metrics.update({'student_idx': idx, 'mode': mode_label, 'elapsed_s': round(time.time() - t0, 2)})
                results.append(metrics)

                logger.debug(f"  Sample {idx:4d} | DM={metrics['dm']:.4f} WKC={metrics['wkc']:.4f} ILD={metrics['ild']:.4f}")
            except Exception as e:
                logger.warning(f"⚠️ Sample {idx} failed: {e}")

        total_time = time.time() - start_time
        logger.info(f"✅ [{mode_label.upper()}] completed {len(results)} samples in {total_time:.1f}s")
        return pd.DataFrame(results)

# ---- 6. 报告与持久化 ----
class Reporter:
    @staticmethod
    def print_summary(df: pd.DataFrame, mode: str):
        logger.info("-" * 60)
        cols = [('skill_hit_rate', 'SkillHit@K '), ('ild', 'ILD        '), ('novelty', 'Novelty    '), ('dm', 'DM         '), ('wkc', 'WKC        ')]
        for c, l in cols:
            if c in df.columns: logger.info(f"  {l}: {df[c].mean():.4f} ± {df[c].std():.4f}")
        if 'n_weak_skills' in df.columns: logger.info(f"  [avg weak skills: {df['n_weak_skills'].mean():.1f}]")
        logger.info("-" * 60)

    @staticmethod
    def print_comparison_table(all_res: dict):
        mets = [('dm', 'DM ↑        '), ('wkc', 'WKC ↑       '), ('ild', 'ILD ↑       '), ('novelty', 'Novelty ↑   '), ('skill_hit_rate', 'SkillHit ↑  ')]
        modes, col_w = list(all_res.keys()), 16
        logger.info("\n" + "=" * (14 + col_w * len(modes)) + "\n  COMPARISON TABLE\n" + "=" * (14 + col_w * len(modes)))
        logger.info(f"  {'Metric':<12}" + "".join(f"{m.upper():>{col_w}}" for m in modes))
        logger.info("-" * (14 + col_w * len(modes)))
        
        for col, label in mets:
            row_str, max_val, vals = f"  {label:<12}", -1.0, []
            for m in modes:
                v = all_res[m][col].mean() if col in all_res[m].columns else float('nan')
                vals.append(v)
                if not np.isnan(v) and v > max_val: max_val = v
            for v in vals:
                tag = " *" if (not np.isnan(v) and abs(v - max_val) < 1e-6) else "  "
                cell = f"{v:.4f}{tag}" if not np.isnan(v) else "  n/a  "
                row_str += f"{cell:>{col_w}}"
            logger.info(row_str)
        logger.info("=" * (14 + col_w * len(modes)) + "\n")


def setup_logger(is_debug: bool):
    import sys
    log_dir = os.path.join(_THIS_DIR, 'output', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    t_now = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    f_path = os.path.join(log_dir, f'recommend_eval_{t_now}.log')
    
    sh = logging.StreamHandler(sys.stdout)
    if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
    
    logging.basicConfig(level=logging.DEBUG if is_debug else logging.INFO,
                        format='[%(asctime)s] [%(levelname)-7s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[logging.FileHandler(f_path, encoding='utf-8'), sh])
    logger.info("=" * 60 + f"\n  MOPSO Offline Evaluation (Log: {f_path})\n" + "=" * 60)
    return t_now

def set_seed(seed: int):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---- 7. 主控启动 ----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--n_eval_samples', type=int, default=None)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=None)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--baseline', type=str, default=None, choices=['greedy', 'random', 'popularity', 'dkt_greedy', 'dkvmn_greedy', 'all'])
    parser.add_argument('--dkt_model', type=str, default=None)
    parser.add_argument('--dkvmn_model', type=str, default=None)
    args = parser.parse_args()

    setup_logger(args.debug)
    
    # 解析配置
    conf_path = get_exp_config_path(args.full)
    with open(conf_path, 'r', encoding='utf-8') as f:
        config_dict = {k: v for sect in toml.load(f).values() for k, v in (sect.items() if isinstance(sect, dict) else {})}
    if args.n_eval_samples: config_dict['n_eval_samples'] = args.n_eval_samples

    # 设定全局随机种子，保证实验可复现
    seed = config_dict.get('seed', 42)
    set_seed(seed)
    logger.info(f"🌱 Global Random Seed Set to: {seed}")

    # 初始化重构的各类
    data_loader = RecDataLoader(config_dict)
    
    # 兼容老配置，如果传入了评估总数但没传end_idx
    if args.n_eval_samples and args.end_idx is None:
        args.end_idx = args.start_idx + args.n_eval_samples
        
    samples_df = data_loader.get_eval_samples(args.start_idx, args.end_idx)
    engine = RecommendationEngine(data_loader, config_dict)
    
    task_queue = ['ours'] if not args.baseline else (
        ['ours', 'greedy', 'random', 'popularity', 'dkt_greedy', 'dkvmn_greedy'] if args.baseline == 'all' else [args.baseline]
    )

    all_results = {}
    for task in task_queue:
        try:
            recommender = RecommenderFactory.create(task, data_loader, config_dict, args)
            df = engine.run_eval(recommender, samples_df, mode_label=task)
            Reporter.print_summary(df, task)
            all_results[task] = df
        except Exception as e:
            logger.error(f"❌ Failed to run {task}: {e}", exc_info=True)

    if args.baseline == 'all' and len(all_results) > 1:
        Reporter.print_comparison_table(all_results)

    # 汇总保存
    if all_results:
        combined_df = pd.concat(all_results.values(), ignore_index=True)
        out_dir = config_dict.get('save_dir', 'output/recommendation_full' if args.full else 'output/recommendation_sample')
        os.makedirs(out_dir, exist_ok=True)
        
        chunk_tag = f"_parts_{args.start_idx}_to_{args.end_idx}" if args.end_idx else ""
        out_file = os.path.join(out_dir, f"recommend_eval_{'full' if args.full else 'sample'}{'_'+args.baseline if args.baseline else ''}{chunk_tag}.csv")
        
        combined_df.to_csv(out_file, index=False)
        logger.info(f"💾 Results saved to {out_file}")

if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
import os
import sys
import argparse
import time
import pandas as pd
import numpy as np
import torch
import copy
import logging

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from run_er_eval import RecDataLoader, RecommendationEngine, Reporter, get_exp_config_path, set_seed, DEVICE
from pso_recommend import RecommendationSystem, FitnessEvaluator, DiscreteMOPSO
import toml
from datetime import datetime, timezone, timedelta

def setup_ablation_logger():
    log_dir = os.path.join(_THIS_DIR, 'output', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    t_now = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    f_path = os.path.join(log_dir, f'ablation_er_{t_now}.log')
    
    sh = logging.StreamHandler(sys.stdout)
    if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
    
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] [%(levelname)-7s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[logging.FileHandler(f_path, encoding='utf-8'), sh])
    
    logger = logging.getLogger('ablation_real')
    logger.info("=" * 60 + f"\n  Ablation Real Evaluation (Log: {f_path})\n" + "=" * 60)
    return logger

class AblationRecommendationSystem(RecommendationSystem):
    def __init__(self, mode, model_path, metadata_path, config, device='cuda'):
        super().__init__(model_path, metadata_path, config, device)
        self.ablation_mode = mode

    def recommend(self, history_q, history_r, history_mask, K=5, interval_time=None, response_time=None):
        T = history_mask.sum().item()
        if T < 10:
            n_q = self.candidate_builder.n_question
            done_qs = history_q[history_mask.bool()].unique().cpu().numpy()
            all_qs = np.arange(1, n_q)
            avaliable_qs = np.setdiff1d(all_qs, done_qs)
            if len(avaliable_qs) < K:
                rec = np.random.choice(all_qs, K, replace=True)
            else:
                rec = np.random.choice(avaliable_qs, K, replace=False)
            return rec, {'cognitive_state': {'tau': 0.55, 'm_k': None}, 'rec_probs': None}

        cognitive_state = self.state_extractor.extract(
            history_q, history_r, history_mask,
            interval_time=interval_time,
            response_time=response_time,
        )

        candidates, probs = self.candidate_builder.build(
            self.model, history_q, history_r, history_mask, cognitive_state
        )

        if self.ablation_mode == 'no_pid':
            cognitive_state['tau'] = 0.55
        else:
            cognitive_state['tau'] = self.candidate_builder._compute_zpd(cognitive_state)
            
        probs_np = probs.cpu().numpy()

        if self.ablation_mode == 'no_mopso':
            tau = cognitive_state['tau']
            dist = (np.sqrt(probs_np) - np.sqrt(tau))**2 + (np.sqrt(1 - probs_np) - np.sqrt(1 - tau))**2
            sorted_idx = np.argsort(dist)
            recommendation = candidates[sorted_idx[:K]]
            
        else:
            evaluator = FitnessEvaluator(self.candidate_builder.qs_table, self.config)
            
            if self.ablation_mode == 'no_f2':
                original_eval = evaluator.evaluate_combo
                def ablated_eval(*args, **kwargs):
                    f1, _ = original_eval(*args, **kwargs)
                    return float(f1), 0.0
                evaluator.evaluate_combo = ablated_eval

            pso = DiscreteMOPSO(candidates, evaluator, cognitive_state, self.config)
            recommendation = pso.optimize(probs)

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

def main():
    parser = argparse.ArgumentParser(description="Real Model Ablation Study")
    parser.add_argument('--n_eval_samples', type=int, default=None, help="Num samples to evaluate. Defaults to all.")
    parser.add_argument('--is_full', action='store_true', help="Use the full config if set.")
    parser.add_argument('--start_idx', type=int, default=0, help="起始样本索引")
    parser.add_argument('--end_idx', type=int, default=None, help="结束样本索引（开区间）")
    parser.add_argument('--mode', type=str, default='all', choices=['all', 'full', 'no_pid', 'no_mopso', 'no_f2'], help="指定要跑的消融模式")
    args = parser.parse_args()

    logger = setup_ablation_logger()

    config_path = get_exp_config_path(args.is_full)
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = toml.load(f)
    flat_config = {
        k: v
        for sect in config_dict.values()
        for k, v in (sect.items() if isinstance(sect, dict) else [])
    }

    c_common = config_dict.get('common', {})
    c_eval = config_dict.get('evaluation', {})
    c_rec = config_dict.get('recommendation', {})

    seed = flat_config.get('seed', 42)
    set_seed(seed)
    logger.info(f"🌱 Global Random Seed Set to: {seed}")
    
    data_loader = RecDataLoader(flat_config)
    
    if args.n_eval_samples is None and not c_eval.get('use_full_test_set', False):
        args.n_eval_samples = c_eval.get('n_eval_samples', None)
    if args.n_eval_samples is not None and args.end_idx is None:
        args.end_idx = args.start_idx + args.n_eval_samples

    df_eval = data_loader.get_eval_samples(args.start_idx, args.end_idx)
    print(f"\n[*] Extracted test sequences: {len(df_eval)} (Requires length >= {flat_config.get('min_seq_len', 10) + flat_config.get('K', 5)})")

    model_path = c_common.get('model_path', '')
    metadata_path = data_loader.metadata_path

    engine = RecommendationEngine(data_loader, flat_config)
    
    modes = ["full", "no_pid", "no_mopso", "no_f2"] if args.mode == 'all' else [args.mode]
    out_results = {}

    print("\n================ Real PyTorch Ablation Starting ================\n")
    
    actual_device = DEVICE

    for mode in modes:
        recommender = AblationRecommendationSystem(
            mode=mode, 
            model_path=model_path, 
            metadata_path=metadata_path, 
            config=flat_config, 
            device=actual_device
        )
        
        df_res = engine.run_eval(recommender, df_eval, mode_label=mode)
        out_results[mode] = df_res

    Reporter.print_comparison_table(out_results)
    
    # Save results
    out_dir = os.path.join(_THIS_DIR, 'output', 'ablation_er')
    os.makedirs(out_dir, exist_ok=True)
    combined_df = pd.concat(out_results.values(), ignore_index=True)
    
    suffix = '' if args.mode == 'all' else f'_{args.mode}'
    chunk_tag = f"_parts_{args.start_idx}_to_{args.end_idx}" if args.end_idx is not None else ''
    out_file = os.path.join(out_dir, f"ablation_eval_{'full' if args.is_full else 'sample'}{suffix}{chunk_tag}.csv")
    combined_df.to_csv(out_file, index=False)
    logger.info(f"💾 Results saved to {out_file}")
    
    print("\n[+] Ablation metrics have been faithfully calculated using real FA-GIKT predictions!")

if __name__ == '__main__':
    main()

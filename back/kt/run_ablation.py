import argparse
import yaml
import optuna
import torch
import os
import json
import numpy as np

from config import get_config
from dataset import UnifiedParquetDataset
from models.factory import ModelFactory
from core.trainer import BaseTrainer
from optuna.samplers import GridSampler

def load_yaml_config(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_metadata(processed_data_dir):
    meta_path = os.path.join(processed_data_dir, 'metadata.json')
    config_path = os.path.join(processed_data_dir, 'config.json')
    
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            data_config = json.load(f)
    elif os.path.exists(config_path):
        with open(config_path, 'r') as f:
            data_config = json.load(f)
    else:
        raise FileNotFoundError(f"Neither metadata.json nor config.json found in {processed_data_dir}")
        
    num_question = data_config.get('num_questions') or data_config.get('n_question')
    num_skill = data_config.get('num_skills') or data_config.get('n_skill')
        
    return num_question, num_skill

def objective(trial, ablation_config, dataset, num_question, num_skill, device, data_config):
    # 1. 设置消融实验参数（直接使用 ablation_config 中的固定值和 grid 搜索值）
    params = {}
    for param_name, param_config in ablation_config['search_space'].items():
        if param_config.get('type') == 'categorical':
            params[param_name] = trial.suggest_categorical(param_name, param_config['choices'])
        else:
            raise ValueError("Ablation study should only use categorical types (True/False or specific options) for grid search.")
    
    # 获取基础的已经调优过的最佳超参数
    base_params = ablation_config.get('base_params', {})
    
    # 2. 合并基础配置、基础最优超参数和消融实验正在采样的参数
    kwargs = {
        'epochs': ablation_config.get('epochs', 20),
        'patience': ablation_config.get('patience', 5),
        'k_fold': ablation_config.get('k_fold', 5),
        **base_params,
        **params # 覆盖基准参数
    }
    
    print(f"\n========== Ablation Trial {trial.number} ==========")
    print(f"Ablation Modules: {params}")
    
    # 3. 定义模型工厂函数
    model_name = ablation_config['model_name']
    
    # 特殊处理组合逻辑
    # 如果把 cognitive 去掉了，那么 cognitive_mode 参数应该被忽略或保持默认
    if 'use_cognitive_model' in params and not params['use_cognitive_model']:
        # 可以通过传递特定参数或在模型中让 use_cognitive_model = False 主导
        pass
        
    def factory_func(**f_kwargs):
        return ModelFactory.get_model(model_name, **f_kwargs)
        
    # 4. 初始化 Trainer
    trainer = BaseTrainer(
        model_factory_func=factory_func,
        num_question=num_question,
        num_skill=num_skill,
        device=device,
        config=data_config,
        kwargs=kwargs
    )
    
    # 5. 执行交叉验证
    try:
        fold_aucs = trainer.cross_validate(dataset)
    except RuntimeError as e:
        msg = str(e)
        if 'out of memory' in msg or 'cudaErrorMemoryAllocation' in msg:
            print(f"Trial {trial.number} 显存溢出，自动跳过。")
            raise optuna.exceptions.TrialPruned()
        else:
            raise
            
    mean_auc = np.mean(fold_aucs)
    print(f"Ablation Trial {trial.number} Finished | Mean CV AUC: {mean_auc:.4f}")
    
    # 对于消融实验，记录下当前组合是个不错的习惯，方便后期分析
    trial.set_user_attr("fold_aucs", [float(v) for v in fold_aucs])
    
    return mean_auc

def main():
    parser = argparse.ArgumentParser(description='Optuna Ablation Study')
    parser.add_argument('--config', type=str, required=True, help='Path to ablation config yaml')
    args = parser.parse_args()
    
    ablation_config = load_yaml_config(args.config)
    dataset_name = ablation_config['dataset_name']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_config = get_config(dataset_name)
    num_question, num_skill = get_metadata(data_config.PROCESSED_DATA_DIR)
    
    print(f"Loading dataset {dataset_name}...")
    dataset = UnifiedParquetDataset(data_config, mode='train')
    
    model_name = ablation_config['model_name']
    epochs = ablation_config.get('epochs', 20)
    
    # 命名 Study：明确这是 ablation 实验
    study_name = f"ablation_{model_name}_{dataset_name}_e{epochs}"
    storage = ablation_config.get('storage', "sqlite:///ablation_experiments.db")

    print(f"Starting Ablation Study: {study_name}")
    print(f"Storage: {storage}")

    # 对于消融实验，强烈建议使用 GridSampler 进行穷举，因为我们要查所有组合 (2^N)
    grid_dict = {}
    grid_size = 1
    for k, v in ablation_config['search_space'].items():
        if v.get('type') != 'categorical':
            raise ValueError(f"Grid sampler requires all search_space entries to be categorical.")
        choices = v.get('choices')
        grid_dict[k] = choices
        grid_size *= len(choices)

    print(f"Total ablation combinations = {grid_size}")
    
    sampler = GridSampler(search_space=grid_dict)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        direction="maximize",
        load_if_exists=True
    )

    try:
        study.optimize(
            lambda trial: objective(trial, ablation_config, dataset, num_question, num_skill, device, data_config),
            n_trials=grid_size
        )
    except Exception as e:
        print(f"Study Error: {e}")
        return
    
    print("\n================= Ablation Study Finished =================")
    
    # 漂亮的输出：列出所有完成的组合及其分数
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed_trials.sort(key=lambda t: t.value, reverse=True)
    
    for i, t in enumerate(completed_trials):
        print(f"Rank {i+1} | AUC: {t.value:.4f} | Params: {t.params}")

# python run_ablation.py --config config/tune/ablation_gikt.yaml
if __name__ == '__main__':
    main()

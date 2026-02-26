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
    
    if num_question is None and 'metrics' in data_config:
        num_question = data_config['metrics'].get('n_question')
    if num_skill is None and 'metrics' in data_config:
        num_skill = data_config['metrics'].get('n_skill')
        
    return num_question, num_skill

def sample_hyperparameters(trial, search_space):
    """
    根据 YAML 中定义的 search_space 动态采样超参数
    """
    params = {}
    for param_name, param_config in search_space.items():
        param_type = param_config.get('type')
        
        if param_type == 'categorical':
            params[param_name] = trial.suggest_categorical(param_name, param_config['choices'])
        elif param_type == 'uniform':
            params[param_name] = trial.suggest_float(param_name, float(param_config['low']), float(param_config['high']))
        elif param_type == 'loguniform':
            params[param_name] = trial.suggest_float(param_name, float(param_config['low']), float(param_config['high']), log=True)
        elif param_type == 'int':
            params[param_name] = trial.suggest_int(param_name, int(param_config['low']), int(param_config['high']))
        else:
            raise ValueError(f"Unsupported parameter type: {param_type}")
            
    return params

def objective(trial, tune_config, dataset, num_question, num_skill, device, data_config):
    # 1. 采样超参数
    sampled_params = sample_hyperparameters(trial, tune_config['search_space'])
    
    # 2. 合并基础配置和采样参数
    kwargs = {
        'epochs': tune_config.get('epochs', 50),
        'patience': tune_config.get('patience', 5),
        'k_fold': tune_config.get('k_fold', 5),
        **sampled_params
    }
    
    print(f"\n========== Trial {trial.number} ==========")
    print(f"Params: {sampled_params}")
    
    # 3. 定义模型工厂函数 (使用闭包绑定 model_name)
    model_name = tune_config['model_name']
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
    fold_aucs = trainer.cross_validate(dataset)
    
    # 6. 计算平均 AUC 作为优化目标
    mean_auc = np.mean(fold_aucs)
    print(f"Trial {trial.number} Finished | Mean CV AUC: {mean_auc:.4f}")
    
    return mean_auc

def main():
    parser = argparse.ArgumentParser(description='Optuna Hyperparameter Tuning')
    parser.add_argument('--config', type=str, required=True, help='Path to tune config yaml')
    args = parser.parse_args()
    
    # 加载调参配置
    tune_config = load_yaml_config(args.config)
    dataset_name = tune_config['dataset_name']
    
    # 初始化环境和数据
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_config = get_config(dataset_name)
    num_question, num_skill = get_metadata(data_config.PROCESSED_DATA_DIR)
    
    print(f"Loading dataset {dataset_name}...")
    dataset = UnifiedParquetDataset(data_config, mode='train')
    
    # 配置 Optuna Study（支持 GridSampler）
    # 自动生成更具描述性的 study_name (如: dkt_assist09_e5_k2)
    model_name = tune_config['model_name']
    epochs = tune_config.get('epochs', 50)
    k_fold = tune_config.get('k_fold', 5)

    study_name = f"{model_name}_{dataset_name}_e{epochs}_k{k_fold}"

    # 统一使用一个大数据库文件，方便 Dashboard 集中管理
    default_db = "kt_tuning.db"
    # 如果配置文件没指定 storage，则默认使用 kt_tuning.db
    storage = tune_config.get('storage', f"sqlite:///{default_db}")

    print(f"Stats: Model: {model_name} | Dataset: {dataset_name} | Study Name: {study_name}")
    print(f"Stats: Storage: {storage}")

    sampler_type = tune_config.get('sampler', 'tpe')
    if sampler_type == 'grid':
        # 构建 GridSampler 所需的搜索字典（所有参数必须为 categorical）
        grid_dict = {}
        grid_size = 1
        for k, v in tune_config['search_space'].items():
            if v.get('type') != 'categorical':
                raise ValueError(f"Grid sampler requires all search_space entries to be categorical. Param {k} is {v.get('type')}")
            choices = v.get('choices')
            # ensure proper numeric types
            converted = []
            for c in choices:
                # leave ints/floats as-is
                converted.append(c)
            grid_dict[k] = converted
            grid_size *= len(converted)

        print(f"Using GridSampler with grid size = {grid_size} trials")
        sampler = GridSampler(search_space=grid_dict)
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            sampler=sampler,
            direction="maximize",
            load_if_exists=True
        )
        n_trials = grid_size
    else:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize",
            load_if_exists=True # 关键：允许断点续传和多窗口并发
        )
        n_trials = tune_config.get('n_trials', 50)

    # 开始优化
    study.optimize(
        lambda trial: objective(trial, tune_config, dataset, num_question, num_skill, device, data_config),
        n_trials=n_trials
    )
    
    print("\n==================================================")
    print("Tuning Finished!")
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (Mean AUC): {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # 4. 自动保存最佳参数到 JSON，方便后续 train_test.py 直接读取
    # 格式: {model_name}_best_params.json
    best_params_file = f"{tune_config['model_name']}_best_params_{dataset_name}.json"
    best_params_file = os.path.join("config", "best_params", best_params_file)
    os.makedirs(os.path.dirname(best_params_file), exist_ok=True)
    
    with open(best_params_file, 'w') as f:
        json.dump(trial.params, f, indent=4)
    print(f"Saved best params to: {best_params_file}")

if __name__ == '__main__':
    main()

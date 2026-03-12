import argparse
import yaml
import optuna
import torch
import os
import json
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from config import get_config
from dataset import UnifiedParquetDataset
from models.factory import ModelFactory
from core.trainer import BaseTrainer
from optuna.samplers import GridSampler

def _sanitize_hparams(params):
    safe = {}
    for k, v in params.items():
        if isinstance(v, np.generic):
            v = v.item()
        if isinstance(v, (int, float, str, bool)):
            safe[k] = v
        else:
            safe[k] = str(v)
    return safe

def load_yaml_config(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_metadata(processed_data_dir):
    meta_path = os.path.join(processed_data_dir, 'metadata.json')
    
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            data_config = json.load(f)
    else:
        raise FileNotFoundError(f"metadata.json not found in {processed_data_dir}")

    num_question = data_config['metrics'].get('n_question')
    num_skill = data_config['metrics'].get('n_skill')
        
    if num_question is None or num_skill is None:
        raise ValueError(f"Could not parse num_question or num_skill from metadata.json in {processed_data_dir}")

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

def objective(trial, tune_config, dataset, num_question, num_skill, device, data_config, tb_root=None):
    # 1. 采样超参数
    sampled_params = sample_hyperparameters(trial, tune_config['search_space'])
    model_name = tune_config['model_name']
    dataset_name = tune_config['dataset_name']
    tb_writer = None
    if tb_root:
        tb_writer = SummaryWriter(log_dir=os.path.join(tb_root, f"trial_{trial.number}"))
        tb_writer.add_text("trial/params", json.dumps(sampled_params, ensure_ascii=False), 0)
    
    # 新增：检查是否已有相同参数组合，如果有则跳过
    for past_trial in trial.study.trials:
        if past_trial.params == sampled_params and past_trial.state == optuna.trial.TrialState.COMPLETE:
            raise optuna.exceptions.TrialPruned()
    
    # 2. 合并基础配置和采样参数
    kwargs = {
        'task_name': 'tune',
        'model_name': model_name,
        'dataset_name': dataset_name,
        'epochs': tune_config.get('epochs', 50),
        'patience': tune_config.get('patience', 5),
        'k_fold': tune_config.get('k_fold', 5),
        'trial_number': trial.number,
        **sampled_params
    }
    
    print(f"\n========== Trial {trial.number} ==========")
    print(f"Params: {sampled_params}")
    
    # 3. 定义模型工厂函数 (使用闭包绑定 model_name)
    def factory_func(**f_kwargs):
        # 去除 Trainer 侧管理字段，避免与 get_model(model_name, ...) 参数冲突
        f_kwargs.pop('model_name', None)
        f_kwargs.pop('dataset_name', None)
        f_kwargs.pop('task_name', None)
        f_kwargs.pop('trial_number', None)
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
    
    # 5. 执行交叉验证（显存溢出自动跳过）
    try:
        fold_aucs = trainer.cross_validate(dataset)
    except RuntimeError as e:
        msg = str(e)
        if 'out of memory' in msg or 'cudaErrorMemoryAllocation' in msg:
            if tb_writer:
                tb_writer.add_text("trial/status", "pruned: out of memory", 0)
                tb_writer.close()
            print(f"Trial {trial.number} 显存溢出，自动跳过该参数组合。")
            raise optuna.exceptions.TrialPruned()
        else:
            if tb_writer:
                tb_writer.add_text("trial/status", f"failed: {msg}", 0)
                tb_writer.close()
            raise
    # 6. 计算平均 AUC 作为优化目标
    mean_auc = np.mean(fold_aucs)

    if tb_writer:
        for idx, auc in enumerate(fold_aucs, start=1):
            tb_writer.add_scalar('trial/fold_auc', auc, idx)
        tb_writer.add_scalar('trial/mean_auc', mean_auc, 0)
        tb_writer.close()

    # HParams: 写到 tune 根目录，显式 run_name，避免出现不可读的时间戳子目录
    if tb_root:
        hp_writer = SummaryWriter(log_dir=tb_root)
        hp_writer.add_hparams(
            hparam_dict=_sanitize_hparams(sampled_params),
            metric_dict={'hparam/mean_auc': float(mean_auc)},
            run_name=f"trial_{trial.number}"
        )
        hp_writer.close()

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
    tune_name = tune_config.get('tune_name', '')
    if tune_name:
        study_name = f"{model_name}_{dataset_name}_e{epochs}_k{k_fold}_{tune_name}"
    else:
        study_name = f"{model_name}_{dataset_name}_e{epochs}_k{k_fold}"

    # 统一使用一个大数据库文件，方便 Dashboard 集中管理
    default_db = "kt_tuning.db"
    # 如果配置文件没指定 storage，则默认使用 kt_tuning.db
    storage = tune_config.get('storage', f"sqlite:///{default_db}")

    print(f"Stats: Model: {model_name} | Dataset: {dataset_name} | Study Name: {study_name}")
    print(f"Stats: Storage: {storage}")

    # 可选 TensorBoard（最小侵入）：用于查看 trial 级别指标，不替代 optuna .db
    try:
        tb_base_out = data_config.path.OUTPUT_DIR # type: ignore
    except AttributeError:
        tb_base_out = './output'
    tune_tb_root = os.path.join(tb_base_out, 'tune', 'runs', study_name)
    print(f"Stats: TensorBoard: {tune_tb_root}")

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

    # 开始优化（捕获参数空间变动相关错误并友好提示）
    try:
        study.optimize(
            lambda trial: objective(trial, tune_config, dataset, num_question, num_skill, device, data_config, tune_tb_root),
            n_trials=n_trials
        )
    except ValueError as e:
        msg = str(e)
        if 'CategoricalDistribution does not support dynamic value space' in msg:
            print(f"\n[Optuna 错误] 参数空间冲突！")
            print(f"检测到您修改了 search_space 中的选项（choices），但当前的 Study 记录 '{study_name}' 仍在使用旧的参数定义。")
            print(f"解决方法：")
            print(f"1. 在 YAML 配置文件中修改 'tune_name'，例如：{tune_name}-v2")
            print(f"2. 或者删除数据库文件（{storage}）重启实验。")
        elif 'is already registered with different' in msg:
            print(f"\n[Optuna 错误] 您可能修改了参数的类型（例如从 categorical 改为 int）。")
            print(f"请更换 'tune_name' 以启动新的实验。")
        else:
            print(f"Optuna参数空间错误：{e}")
        return
    
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

# python tune.py --config config/tune/dkt.yaml
# optuna-dashboard sqlite:///kt_tuning.db
# tensorboard --logdir=runs --port=6006
if __name__ == '__main__':
    main()

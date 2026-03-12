"""
Generic baseline training script for standard KT models (DKT, DKVMN, etc.)
Outputs models to the checkpoint directory for downstream evaluation.
"""
import os
import argparse
import sys
import torch
from datetime import datetime

# Ensure back package is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from config import get_config
from core.trainer import BaseTrainer
from models.factory import ModelFactory
from dataset import UnifiedParquetDataset

def get_metadata(processed_data_dir):
    import json
    metadata_path = os.path.join(processed_data_dir, 'metadata.json')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}. Please execute data_process.py first.")
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    num_question = meta['metrics'].get('n_question', 0)
    num_skill = meta['metrics'].get('n_skill', 0)
    return num_question, num_skill

def main():
    parser = argparse.ArgumentParser(description="Train Baseline KT Models")
    parser.add_argument('--model_name', type=str, required=True, choices=['dkt', 'dkvmn', 'akt', 'simplekt', 'qikt', 'lbkt', 'deep_irt', 'dkt_forget', 'gikt_old'], help="Baseline model name (e.g., dkt, dkvmn)")
    parser.add_argument('--dataset', type=str, required=True, help="Dataset name, e.g. assist09, ednet")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--k_fold', type=int, default=5, help="K-Fold cross validation splits")
    parser.add_argument('--patience', type=int, default=3, help="Early stopping patience")
    parser.add_argument('--no_save_model', action='store_true', help="启用则不保存模型")
    parser.add_argument('--batch_size', type=int, default=128, help="批大小")
    
    args = parser.parse_args()

    # Load optimal hyperparameters dynamically from json
    best_params_path = os.path.join(current_dir, 'config', 'best_params', f'{args.model_name}_best_params.json')
    model_kwargs = {}
    
    if os.path.exists(best_params_path):
        import json
        with open(best_params_path, 'r', encoding='utf-8') as f:
            best_params_dict = json.load(f)
            
        ds_name = args.dataset.lower()
        base_ds_name = ds_name.split('-')[0]  # 提取基础数据集名，例如 assist09, ednet_kt1
        
        matched_key = None
        for k in best_params_dict.keys():
            k_norm = k.lower().replace("2009", "09").replace("2012", "12").replace("2015", "15").replace("2017", "17")
            if k_norm == base_ds_name or k_norm in base_ds_name:
                matched_key = k
                break
                
        if matched_key:
            print(f"[{args.model_name.upper()}] 🌟 Auto-loading optimal parameters for '{matched_key}':")
            override_params = best_params_dict[matched_key]
        elif 'default' in best_params_dict:
            print(f"[{args.model_name.upper()}] ⚠️ No optimal params found for dataset '{args.dataset}', using 'default' parameters:")
            override_params = best_params_dict['default']
        else:
            print(f"[{args.model_name.upper()}] ⚠️ No optimal params found for dataset '{args.dataset}' in JSON, and no 'default' provided.")
            override_params = {}
            
        for param_k, param_v in override_params.items():
            model_kwargs[param_k] = param_v
            print(f"  |-- {param_k} = {param_v}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_config = get_config(args.dataset)
    num_question, num_skill = get_metadata(data_config.PROCESSED_DATA_DIR)
    
    print(f"[{args.model_name.upper()}] Loading dataset {args.dataset}...")
    dataset = UnifiedParquetDataset(data_config, mode='train')
    test_dataset = UnifiedParquetDataset(data_config, mode='test')

    checkpoint_dir = 'checkpoint'
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_path = os.path.join(checkpoint_dir, f"{args.model_name}_{args.dataset}_best_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")

    # Define model factory wrapper
    def factory_func(**f_kwargs):
        # Merge JSON params into factory kwargs
        f_kwargs.update(model_kwargs)
        f_kwargs.pop('model_name', None) # 
        f_kwargs.pop('dataset_name', None)
        return ModelFactory.get_model(args.model_name, **f_kwargs)
    # Supply kwargs to BaseTrainer
    kwargs = {
        'task_name': 'train_baseline',
        'model_name': args.model_name,
        'dataset_name': args.dataset,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': model_kwargs.get('learning_rate', 1e-3),
        'k_fold': args.k_fold,
        'patience': args.patience,
        'save_model': not args.no_save_model,
        'model_save_path': save_path,
        'amp_enabled': False
    }

    trainer = BaseTrainer(
        model_factory_func=factory_func,
        num_question=num_question,
        num_skill=num_skill,
        device=device,
        config=data_config,
        kwargs=kwargs
    )
    
    print(f"Start training {args.model_name.upper()}... Best model will be saved to => {save_path}")
    fold_aucs = trainer.cross_validate(dataset, test_dataset=test_dataset)
    
    best_auc = max(fold_aucs) if fold_aucs else 0.0
    print(f"[{args.model_name.upper()}] Training finished. Max validation AUC: {best_auc:.4f}")
    if os.path.exists(save_path):
        print(f"✅ Successfully saved to {save_path}!")

# python train_baseline.py --model_name dkt --dataset assist09 --epochs 10 --k_fold 1 --patience 1 --no_save_model --batch_size 256
if __name__ == '__main__':
    main()
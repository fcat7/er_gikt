"""
Generic baseline training script for standard KT models (DKT, DKVMN, etc.)
Outputs models to the checkpoint directory for downstream evaluation.
"""
import os
import argparse
import sys
import torch

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
    parser.add_argument('--model_name', type=str, required=True, choices=['dkt', 'dkvmn', 'akt', 'simplekt', 'qikt', 'lbkt', 'gikt_old'], help="Baseline model name (e.g., dkt, dkvmn)")
    parser.add_argument('--dataset', type=str, required=True, help="Dataset name, e.g. assist09, ednet")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--k_fold', type=int, default=5, help="K-Fold cross validation splits")
    parser.add_argument('--patience', type=int, default=3, help="Early stopping patience")
    
    args = parser.parse_args()

    # Load optimal hyperparameters dynamically from json
    best_params_path = os.path.join(current_dir, 'config', 'best_params', f'{args.model_name}_best_params.json')
    model_kwargs = {}
    
    if os.path.exists(best_params_path):
        import json
        with open(best_params_path, 'r', encoding='utf-8') as f:
            best_params_dict = json.load(f)
            
        ds_name = args.dataset.lower()
        matched_key = None
        for k in best_params_dict.keys():
            if k == ds_name or k.replace("2009", "09").replace("2012", "12").replace("2015", "15").replace("2017", "17") in ds_name or k in ds_name:
                matched_key = k
                break
                
        if matched_key:
            print(f"[{args.model_name.upper()}] 🌟 Auto-loading optimal parameters for '{matched_key}':")
            override_params = best_params_dict[matched_key]
            for param_k, param_v in override_params.items():
                model_kwargs[param_k] = param_v
                print(f"  |-- {param_k} = {param_v}")
        else:
            print(f"[{args.model_name.upper()}] ⚠️ No optimal params found for dataset '{args.dataset}' in JSON.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_config = get_config(args.dataset)
    num_question, num_skill = get_metadata(data_config.PROCESSED_DATA_DIR)
    
    print(f"[{args.model_name.upper()}] Loading dataset {args.dataset}...")
    dataset = UnifiedParquetDataset(data_config, mode='train')

    checkpoint_dir = 'checkpoint'
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_path = os.path.join(checkpoint_dir, f"{args.model_name}_{args.dataset}_best.pt")

    # Define model factory wrapper
    def factory_func(**f_kwargs):
        # Merge JSON params into factory kwargs
        f_kwargs.update(model_kwargs)
        return ModelFactory.get_model(args.model_name, **f_kwargs)

    # Supply kwargs to BaseTrainer
    kwargs = {
        'epochs': args.epochs,
        'batch_size': model_kwargs.get('batch_size', 64),
        'learning_rate': model_kwargs.get('learning_rate', 1e-3),
        'k_fold': args.k_fold,
        'patience': args.patience,
        'save_model': True,
        'model_save_path': save_path,
        'amp_enabled': True
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
    fold_aucs = trainer.cross_validate(dataset)
    
    best_auc = max(fold_aucs) if fold_aucs else 0.0
    print(f"[{args.model_name.upper()}] Training finished. Max validation AUC: {best_auc:.4f}")
    if os.path.exists(save_path):
        print(f"✅ Successfully saved to {save_path}!")

# python train_baseline.py --model_name dkt,dkvmn --dataset assist09 --epochs 200 --k_fold 1 --patience 10
if __name__ == '__main__':
    main()
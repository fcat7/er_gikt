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
    parser = argparse.ArgumentParser(description="Train and Test GIKT_OLD Model")
    parser.add_argument('--dataset', type=str, required=True, help="Dataset name, e.g. assist09-sample_20%")
    
    # 训练超参数相关
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--k_fold', type=int, default=1, help="K-Fold cross validation splits")
    parser.add_argument('--patience', type=int, default=3, help="Early stopping patience")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size")
    
    # GIKT_OLD 模型特有参数
    parser.add_argument('--dim_emb', type=int, default=100, help="Embedding dimension")
    parser.add_argument('--agg_hops', type=int, default=3, help="GNN aggregation hops")
    parser.add_argument('--dropout4gru', type=float, default=0.1, help="Dropout rate for GRU")
    parser.add_argument('--dropout4gnn', type=float, default=0.1, help="Dropout rate for GNN")
    
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_config = get_config(args.dataset)
    num_question, num_skill = get_metadata(data_config.PROCESSED_DATA_DIR)
    
    print(f"[TEST GIKT_OLD] Loading dataset {args.dataset}...")
    dataset = UnifiedParquetDataset(data_config, mode='train')

    checkpoint_dir = os.path.join(current_dir, 'checkpoint')
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_path = os.path.join(checkpoint_dir, f"gikt_old_{args.dataset}_test.pt")

    # 指定 gikt_old 模型参数
    model_kwargs = {
        'dim_emb': args.dim_emb,
        'agg_hops': args.agg_hops,
        'dropout4gru': args.dropout4gru,
        'dropout4gnn': args.dropout4gnn
    }

    # Define model factory wrapper
    def factory_func(**f_kwargs):
        # 覆写或增加额外的模型参数
        f_kwargs.update(model_kwargs)
        return ModelFactory.get_model('gikt_old', **f_kwargs)

    # Supply kwargs to BaseTrainer
    params_dict = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
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
        kwargs=params_dict
    )
    
    print(f"\n=========================================")
    print(f"🚀 Starting Test for gikt_old")
    print(f"Dataset: {args.dataset}")
    print(f"Params: {params_dict}")
    print(f"Model Params: {model_kwargs}")
    print(f"=========================================\n")

    trainer.cross_validate(dataset)

if __name__ == '__main__':
    main()

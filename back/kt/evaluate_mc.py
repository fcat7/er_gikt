import argparse
import os
import torch
import numpy as np
from sklearn.model_selection import ShuffleSplit
from torch.utils.data import DataLoader, Subset

from config import Config, DEVICE, COLOR_LOG_G, COLOR_LOG_END
from dataset import UserDataset
# 必须导入 GIKT 类，否则 torch.load(model) 会报错找不到类定义
from gikt import GIKT  
from util.mc_utils import mc_evaluate

def get_parser():
    parser = argparse.ArgumentParser(description="Evaluate GIKT Model with Monte Carlo Dropout")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved .pt model file')
    parser.add_argument('--dataset', type=str, default='assist09', help='Dataset name (default: assist09)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for inference')
    parser.add_argument('--num_samples', type=int, default=30, help='Number of MC samples (T)')
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    # 1. 加载模型
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return

    print(f"Loading model from {args.model_path}...")
    try:
        model = torch.load(args.model_path, map_location=DEVICE)
    except Exception as e:
        print(f"Failed to load model. Ensure it was saved using torch.save(model, ...). Error: {e}")
        return
        
    model.to(DEVICE)
    model.eval()
    
    # TF Alignment is now always enabled
    # (Model info for compatibility checking)
    model_num_q = getattr(model, 'num_question', None)
    model_num_s = getattr(model, 'num_skill', None)
    print(f"Model loaded. TF Alignment: Enabled (Default)")
    print(f"Model capacity: num_question={model_num_q}, num_skill={model_num_s}")

    # 2. 准备数据
    print(f"Loading dataset: {args.dataset}...")
    config = Config(dataset_name=args.dataset)
    dataset = UserDataset(config)
    
    # 安全检查：验证数据集与模型的兼容性
    if model_num_q is not None:
        # 检查数据集中的最大 Question ID
        from scipy import sparse
        qs_table = sparse.load_npz(os.path.join(config.PROCESSED_DATA_DIR, 'qs_table.npz'))
        dataset_num_q, dataset_num_s = qs_table.shape
        print(f"Dataset capacity: num_question={dataset_num_q}, num_skill={dataset_num_s}")
        
        if dataset_num_q != model_num_q.item() or dataset_num_s != model_num_s.item():
            print(f"\n⚠️  WARNING: Dataset mismatch detected!")
            print(f"   Model was trained on: {model_num_q.item()} questions, {model_num_s.item()} skills")
            print(f"   Current dataset has:  {dataset_num_q} questions, {dataset_num_s} skills")
            print(f"\n   This WILL cause 'index out of bounds' errors!")
            print(f"   Please specify the correct dataset with --dataset flag.\n")
            return
    
    # 复现数据集划分 (保持 Random State = 42, 与训练一致)
    # 这里默认取 20% 测试集
    print("Splitting dataset (80/20)...")
    splitter = ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, test_indices = next(splitter.split(dataset))
    
    test_set = Subset(dataset, test_indices)
    test_loader = DataLoader(
        test_set, 
        batch_size=args.batch_size, 
        shuffle=False,
        pin_memory=(DEVICE.type != 'cpu')
    )
    
    print(f"Test set size: {len(test_set)} users")

    # 3. 运行 MC 推断
    print(f"\nRunning Monte Carlo Inference (T={args.num_samples})...")
    mc_metrics, mc_results = mc_evaluate(
        model, test_loader, 
        num_samples=args.num_samples, 
        device=DEVICE
    )
    
    # 4. 生成报告
    msg_header = f"\n[MC-Dropout Summary (T={args.num_samples})]\n"
    msg_metric = f"AUC (Mean): {mc_metrics['auc']:.4f}, ACC (Mean): {mc_metrics['acc']:.4f}\n"
    
    std_all = mc_results['y_std']
    y_true = mc_results['y_true']
    y_mean = mc_results['y_mean']
    
    # 转换预测类别
    y_pred_cls = (y_mean >= 0.5).astype(int)
    correct_mask = (y_pred_cls == y_true)
    
    # 避免空数组警告
    std_correct = np.mean(std_all[correct_mask]) if np.any(correct_mask) else 0.0
    std_wrong = np.mean(std_all[~correct_mask]) if np.any(~correct_mask) else 0.0
    
    msg_kpi = f"Average Uncertainty (Std): {np.mean(std_all):.4f}\n"
    msg_kpi += f"  - On Correct Predictions: {std_correct:.4f}\n"
    msg_kpi += f"  - On Wrong Predictions:   {std_wrong:.4f}\n"
    
    diff = std_wrong - std_correct
    if diff > 0.05:
        verdict = "EXCELLENT. The model knows when it is confused."
    elif diff > 0:
        verdict = "GOOD. Uncertainty is correlated with error."
    else:
        verdict = "POOR. The model is confidently wrong."
        
    msg_kpi += f"\nDiagnosis Verdict: {verdict}\n"
    
    print(COLOR_LOG_G + msg_header + msg_metric + msg_kpi + COLOR_LOG_END)
    
    # 可选保存
    # save_path = f"mc_analysis_{args.dataset}.npz"
    # np.savez(save_path, **mc_results)
    # print(f"Detailed results saved to {save_path}")


# 使用示例：
# python evaluate_mc.py --model_path model/20260203_2247.pt --num_samples 5 --dataset assist09
# 注意：--dataset 必须与训练时使用的数据集一致！
if __name__ == '__main__':
    # 简单的种子设置
    torch.manual_seed(42)
    np.random.seed(42)
    main()

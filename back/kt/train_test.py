"""
训练并测试模型
使用五折交叉验证法
"""
import os
import time
from datetime import datetime
import numpy as np
from scipy import sparse
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, ShuffleSplit
from torch.utils.data import DataLoader, Subset
from dataset import UserDataset
from config import Config, DEVICE, COLOR_LOG_B, COLOR_LOG_Y, COLOR_LOG_G, COLOR_LOG_END
from params import HyperParameters
from util.utils import gen_gikt_graph, build_adj_list
import argparse
from gikt import GIKT

# try :
#     from icecream import ic
#     print = ic
# except ImportError:
#     print("warning: icecream not installed, using standard print.")


# @add_fzq 2025-12-25 10:42:10 -------------------------------------------
# 固定随机种子，保证实验可复现
import random
import torch
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
# @add_fzq 2025-12-25 10:42:10 -------------------------------------------

def get_parser():
    parser = argparse.ArgumentParser(description="Train and Test GIKT Model")
    parser.add_argument('--full', action='store_true', help='Use full dataset (overrides --mode)')

    parser.add_argument('--name', type=str, default='default', help='Name of the experiment')
    return parser

def get_exp_config_path(isFull=False, name='default'):
    # 默认路径为： config/experiments/exp_gcn_sample_default.toml
    return f"config/experiments/exp_{'full' if isFull else 'sample'}_{name}.toml"

# 使用方法
# Windows PowerShell 禁用 GPU
# $env:CUDA_VISIBLE_DEVICES = "-1"  
# python train_test.py 
# python train_test.py --full
# python train_test.py --agg_method gat
# python train_test.py --name my_exp
# python train_test.py --name mc_optim_a
if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()

    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # ⚠️ 仅用于调试，会大幅降低性能！正常训练时应注释掉

    # @add_fzq 2025-12-24 17:28:09 -------------------------------------------
    # 1. 解决时区问题：强制使用 UTC+8 (北京时间)
    from datetime import timedelta, timezone
    beijing_time = datetime.now(timezone(timedelta(hours=8)))
    time_now = beijing_time.strftime('%Y%m%d_%H%M')
    # @add_fzq 2025-12-24 17:28:09 -------------------------------------------

    # 加载超参数
    exp_config_path = get_exp_config_path(isFull=args.full, name=args.name)
    params = HyperParameters.load(exp_config_path=exp_config_path)
    # 加载配置
    dataset_name = params.train.dataset_name
    config = Config(dataset_name=dataset_name)

    output_path = f'{config.path.LOG_DIR}/{time_now}.log'
    output_dir = os.path.dirname(output_path)  # 获取目录路径    
    os.makedirs(output_dir, exist_ok=True) # 创建目录（如果不存在）
    output_file = open(output_path, 'a', buffering=1) # 解决日志丢失问题，使用 'a' (append) 模式，并设置 buffering=1 (行缓冲)

    print(f"Using dataset: {dataset_name}, Data dir: {config.PROCESSED_DATA_DIR}\n")
    print(f"Using experiment config: {exp_config_path}\n")
    
    # 打印当前使用的设备的名称 get_device_name
    print(f"Using device: {torch.cuda.is_available() and torch.cuda.get_device_name(0) or 'cpu'}\n")
    

    # 打印并写超参数
    output_file.write(str(params) + '\n')
    print(params)
    
    batch_size = params.train.batch_size
    
    # 构建模型需要的数据结构, 全部转化为正确类型tensor再输入模型中
    qs_table = torch.tensor(sparse.load_npz(os.path.join(config.PROCESSED_DATA_DIR, 'qs_table.npz')).toarray(), dtype=torch.int64, device=DEVICE)  # [num_q, num_c]
    num_question = torch.tensor(qs_table.shape[0], device=DEVICE)
    num_skill = torch.tensor(qs_table.shape[1], device=DEVICE)
    q_neighbors_list, s_neighbors_list = build_adj_list(config.PROCESSED_DATA_DIR)
    
    # Config 使用结构化访问
    q_neighbors, s_neighbors = gen_gikt_graph(q_neighbors_list, s_neighbors_list, params.model.size_q_neighbors, params.model.size_s_neighbors)
    q_neighbors = torch.tensor(q_neighbors, dtype=torch.int64, device=DEVICE)
    s_neighbors = torch.tensor(s_neighbors, dtype=torch.int64, device=DEVICE)

    # 处理 dropout (list -> tuple)
    dropout_val = params.model.dropout
    if isinstance(dropout_val, list):
        dropout_val = tuple(dropout_val)

    # 初始化模型
    model = GIKT(
        num_question, num_skill, q_neighbors, s_neighbors, qs_table,
        agg_hops=params.model.agg_hops,
        emb_dim=params.model.emb_dim,
        dropout=dropout_val,
        hard_recap=params.model.hard_recap,
        use_cognitive_model=params.model.use_cognitive_model,
        pre_train=params.model.pre_train,
        data_dir=config.PROCESSED_DATA_DIR,
        agg_method=params.model.agg_method,
        recap_source='hsei' if params.model.use_input_attention else 'hssi', # 通过 toml 配置控制
        enable_tf_alignment=params.model.enable_tf_alignment
    ).to(DEVICE)

    # @change_fzq 2026-01-08: 修改损失函数为 BCELoss
    # 原因：模型输出已经是 Sigmoid 概率，BCEWithLogitsLoss 会再次 Sigmoid，导致梯度消失
    # 原始损失函数备份：
    loss_fun = torch.nn.BCEWithLogitsLoss().to(DEVICE) # 损失函数
    if params.train.use_bce_loss:
        loss_fun = torch.nn.BCELoss().to(DEVICE)

    # @add_fzq: TF Alignment Override
    if params.model.enable_tf_alignment:
        # If alignment enabled, Model outputs Logits -> Must use BCEWithLogitsLoss
        loss_fun = torch.nn.BCEWithLogitsLoss().to(DEVICE)
    
    dataset = UserDataset(config)  # 数据集
    data_len = len(dataset)  # 数据总长度

    # 写当前数据量
    output_file.write(f'Total number of users in dataset: {data_len}\n')
    print('model has been built')

    # @add_fzq 2025-12-24 17:28:09 -------------------------------------------
    # 记录总开始时间
    total_start_time = time.time()
    # @add_fzq 2025-12-24 17:28:09 -------------------------------------------

    # 优化器
    epoch_total = 0
    optimizer = torch.optim.Adam(params=model.parameters(), lr=params.train.lr)
    torch.optim.lr_scheduler.ExponentialLR(optimizer, params.train.lr_gamma)
    
    # 在matplotlib中绘制的y轴数据，三行分别表示loss, acc, auc
    y_label_aver = np.zeros([3, params.train.epochs]) # 平均精度值
    y_label_all = np.zeros([3, params.train.epochs * params.train.k_fold]) # 全部精度值

    # KFold的shuffle操作是在用户级别进行的，而不是在答题记录级别
    if params.train.k_fold == 1:
        # 如果 k_fold 为 1，使用 ShuffleSplit 进行单次划分 (80% 训练, 20% 测试)
        k_fold = ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    else:
        # 否则使用 KFold 进行交叉验证
        k_fold = KFold(n_splits=params.train.k_fold, shuffle=True, random_state=42)

    for epoch in range(params.train.epochs):
        train_loss_aver = train_acc_aver = train_auc_aver = 0
        test_loss_aver = test_acc_aver = test_auc_aver = 0
        # 五折的平均值
        for fold, (train_indices, test_indices) in enumerate(k_fold.split(dataset)):
            # 使用五折交叉验证，每次的训练集和测试集都不相同
            train_set = Subset(dataset, train_indices)  # 训练集
            test_set = Subset(dataset, test_indices)  # 测试集
            if DEVICE.type == 'cpu':  # Cpu(本机)
                train_loader = DataLoader(train_set, batch_size=batch_size)  # 训练数据加载器
                test_loader = DataLoader(test_set, batch_size=batch_size)  # 测试数据加载器
            else:  # Gpu(服务器)
                # @fix_fzq: prefetch_factor only works with num_workers > 0
                loader_kwargs = {
                    'batch_size': batch_size,
                    'num_workers': params.common.num_workers,
                    'pin_memory': True
                }
                if params.common.num_workers > 0:
                    loader_kwargs['prefetch_factor'] = params.train.prefetch_factor
                
                train_loader = DataLoader(train_set, **loader_kwargs)
                test_loader = DataLoader(test_set, **loader_kwargs)
            train_data_len, test_data_len = len(train_set), len(test_set)
            # @delete_fzq 2025-12-23 22:10:41
            #  print('===================' + COLOR_LOG_Y + f'epoch: {epoch_total + 1}'+ COLOR_LOG_END + '====================')
            # @add_fzq 2025-12-23 22:10:41
            print('===================' + COLOR_LOG_Y + f'fold: {fold + 1}'+ COLOR_LOG_END + '====================')

            # 训练阶段，既有前向传播，也有反向传播
            print('-------------------training------------------')
            torch.set_grad_enabled(True) # @add_fzq: Enable grad for training
            model.train() # @add_fzq: Switch to train mode
            time0 = time.time()
            train_step = train_loss = train_total = train_right = train_auc = 0
            # 每轮训练第几个批量, 总损失, 训练的真实样本个数, 其中正确的个数, 总体训练的auc
            for data in train_loader:
                # 梯度清零
                optimizer.zero_grad()

                # -- delete_fzq 2025-12-25 17:15:13-----------------
                # x, y_target, mask = data[:, :, 0].to(DEVICE), data[:, :, 1].to(DEVICE), data[:, :, 2].to(torch.bool).to(DEVICE)
                # y_hat = model(x, y_target, mask) # 原始代码
                # --------------------------------------------
                # 批量转移到 GPU（单次 PCIe）
                data_gpu = data.to(DEVICE)
                # 在 GPU 上切片和类型转换（GPU 内操作，无开销）
                x = data_gpu[:, :, 0].to(torch.long)
                y_target = data_gpu[:, :, 1].to(torch.long)
                mask = data_gpu[:, :, 2].to(torch.bool)
                interval_time = data_gpu[:, :, 3].to(torch.float32)
                response_time = data_gpu[:, :, 4].to(torch.float32)

                # @add_fzq 2026-01-08: 输入数据 NaN 运行时检查与清洗
                # 必须步骤：防止 data/assist09/*.npy 中存在的 NaN 导致训练崩溃
                if torch.isnan(interval_time).any():
                    interval_time = torch.nan_to_num(interval_time, nan=0.0)
                if torch.isnan(response_time).any():
                    response_time = torch.nan_to_num(response_time, nan=0.0)

                y_hat = model(x, y_target, mask, interval_time, response_time)
                # --------------------------------------------
    
                y_hat = torch.masked_select(y_hat, mask)
                y_target = torch.masked_select(y_target, mask)

                # @add_fzq: Logic Branch for TF Alignment (Logits vs Probs)
                if params.model.enable_tf_alignment:
                    # y_hat is logits. No clamping needed for BCEWithLogitsLoss.
                    loss = loss_fun(y_hat, y_target.to(torch.float32))
                    
                    # Metrics: Convert to Probabilities
                    y_prob = torch.sigmoid(y_hat) 
                else: 
                    # Original Behavior: y_hat is probabilities (Sigmoid applied in model)
                    # @add_fzq 2026-01-08: Clamping for BCELoss numerical stability
                    y_hat = torch.clamp(y_hat, min=1e-6, max=1.0 - 1e-6)
                    loss = loss_fun(y_hat, y_target.to(torch.float32))
                    
                    y_prob = y_hat # Already probabilities
                
                # @add_fzq: Regularization Constraint (Path 2 & 3)
                # 防止区分度参数爆炸。对于 Step 2 (scalar)，约束其趋近 0(即gain=1)；
                # 对于 Step 3 (Embedding)，约束整个表。
                reg_loss = 0.0
                if hasattr(model, 'discrimination_gain'):
                    reg_loss += 0.01 * (model.discrimination_gain ** 2)
                if hasattr(model, 'discrimination_bias'):
                    # 区分度正则：鼓励其靠近 1.0 (即偏差靠近 0)
                    reg_loss += 1e-5 * torch.sum(model.discrimination_bias.weight ** 2)
                
                if hasattr(model, 'guessing_bias') and hasattr(model, 'slipping_bias'):
                    # 猜测和失误率正则：防止它们过大
                    # 因为 sigmoid(-3) 约等于 0.05，我们不希望这些参数漂移回 0 (0.5) 或更高
                    # 这里限制其权重的 L2，但更重要的是限制其不要变得太大
                    reg_loss += 1e-5 * torch.sum(torch.relu(model.guessing_bias.weight + 2.0)**2) 
                    reg_loss += 1e-5 * torch.sum(torch.relu(model.slipping_bias.weight + 3.0)**2)
                
                loss += reg_loss

                train_loss += loss.item()
                
                # 计算acc
                y_pred = torch.ge(y_prob, torch.tensor(0.5))
                acc = torch.sum(torch.eq(y_target, y_pred)) / torch.sum(mask)
                train_right += torch.sum(torch.eq(y_target, y_pred))
                train_total += torch.sum(mask)
                # 计算auc
                # @optimize: Use probabilities (y_prob) instead of labels (y_pred) for better precision
                auc = roc_auc_score(y_target.cpu().detach(), y_prob.cpu().detach())
                train_auc += auc * len(x) / train_data_len
                loss.backward()
                optimizer.step()
                train_step += 1
                if params.train.verbose:
                    print(f'step: {train_step}, loss: {loss.item():.4f}, acc: {acc.item():.4f}, auc: {auc:.4f}')
            train_loss, train_acc = train_loss / train_step, train_right / train_total
            train_loss_aver += train_loss
            train_acc_aver += train_acc
            train_auc_aver += train_auc

            # 测试阶段，只有前向传递，没有反向传播阶段
            print('-------------------testing------------------')
            model.eval() # @add_fzq: Switch to eval mode
            test_step = test_loss = test_total = test_right = test_auc = 0
            
            # @add_fzq: Global AUC Support
            all_y_targets = []
            all_y_probs = []

            # 每轮训练第几个批量, 总损失, 训练的真实样本个数, 其中正确的个数, 总体的auc
            torch.set_grad_enabled(False) # @add_fzq: Disable grad for testing (Save VRAM)
            for data in test_loader:
                    # -- delete_fzq 2025-12-25 17:15:13-----------------
                    # x, y_target, mask = data[:, :, 0].to(DEVICE), data[:, :, 1].to(DEVICE), data[:, :, 2].to(torch.bool).to(DEVICE)
                    # y_hat = model(x, y_target, mask) # 原始代码
                    # --------------------------------------------

                    # ---------------- 新增时间特征 ----------------
                    # -- add_fzq 2025-12-25 17:15:13-----------------
                    # 批量转移到 GPU（单次 PCIe）
                    data_gpu = data.to(DEVICE)

                    # 在 GPU 上切片和类型转换（GPU 内操作，无开销）
                    x = data_gpu[:, :, 0].to(torch.long)
                    y_target = data_gpu[:, :, 1].to(torch.long)
                    mask = data_gpu[:, :, 2].to(torch.bool)
                    interval_time = data_gpu[:, :, 3].to(torch.float32)
                    response_time = data_gpu[:, :, 4].to(torch.float32)
                
                    # @add_fzq 2026-01-08: 测试集同样需要 NaN 检查
                    if torch.isnan(interval_time).any():
                        interval_time = torch.nan_to_num(interval_time, nan=0.0)
                    if torch.isnan(response_time).any():
                        response_time = torch.nan_to_num(response_time, nan=0.0)

                    y_hat = model(x, y_target, mask, interval_time, response_time)
                    # -- add_fzq 2025-12-25 17:15:13-----------------
                    # --------------------------------------------
                
                    y_hat = torch.masked_select(y_hat, mask.to(torch.bool))
                    y_target = torch.masked_select(y_target, mask.to(torch.bool))
                
                    # @add_fzq: Logic Branch for TF Alignment (Testing Phase)
                    if params.model.enable_tf_alignment:
                        loss = loss_fun(y_hat, y_target.to(torch.float32))
                        y_prob = torch.sigmoid(y_hat)
                    else:
                        # @add_fzq 2026-01-08: 截断
                        y_hat = torch.clamp(y_hat, min=1e-6, max=1.0 - 1e-6)
                        loss = loss_fun(y_hat, y_target.to(torch.float32))
                        y_prob = y_hat

                    test_loss += loss.item()
                    
                    # 计算acc
                    y_pred = torch.ge(y_prob, torch.tensor(0.5))
                    acc = torch.sum(torch.eq(y_target, y_pred)) / torch.sum(mask)
                    test_right += torch.sum(torch.eq(y_target, y_pred))
                    test_total += torch.sum(mask)
                    # 计算auc
                    if params.train.use_global_auc:
                        all_y_targets.extend(y_target.cpu().detach().numpy())
                        all_y_probs.extend(y_prob.cpu().detach().numpy())
                        test_step += 1
                        if params.train.verbose:
                            try:
                                batch_auc = roc_auc_score(y_target.cpu().detach(), y_prob.cpu().detach())
                                print(f'step: {test_step}, loss: {loss.item():.4f}, acc: {acc.item():.4f}, auc: {batch_auc:.4f} (Batch)')
                            except ValueError:
                                pass
                    else:
                        try:
                            auc = roc_auc_score(y_target.cpu().detach(), y_prob.cpu().detach())
                            test_auc += auc * len(x) / test_data_len
                            test_step += 1
                            if params.train.verbose:
                                print(f'step: {test_step}, loss: {loss.item():.4f}, acc: {acc.item():.4f}, auc: {auc:.4f}')
                        except ValueError:
                            test_step += 1
            
            if params.train.use_global_auc and len(all_y_targets) > 0:
                test_auc = roc_auc_score(all_y_targets, all_y_probs)

            test_loss, test_acc = test_loss / test_step, test_right / test_total
            test_loss_aver += test_loss
            test_acc_aver += test_acc
            test_auc_aver += test_auc

            # fold总结阶段
        
            time1 = time.time()
            run_time = time1 - time0
            print(COLOR_LOG_B + f'training: loss: {train_loss:.4f}, acc: {train_acc:.4f}, auc: {train_auc: .4f}' + COLOR_LOG_END)
            print(COLOR_LOG_B + f'testing: loss: {test_loss:.4f}, acc: {test_acc:.4f}, auc: {test_auc: .4f}' + COLOR_LOG_END)
            print(COLOR_LOG_B + f'time: {run_time:.2f}s, average batch time: {(run_time / (test_step + train_step)):.2f}s' + COLOR_LOG_END)
            # 保存输出至本地文件
            output_file.write(f'  fold {fold+1} | ')
            output_file.write(f'training: loss: {train_loss:.4f}, acc: {train_acc:.4f}, auc: {train_auc: .4f}\n         | ')
            output_file.write(f'testing: loss: {test_loss:.4f}, acc: {test_acc:.4f}, auc: {test_auc: .4f} | ')
            output_file.write(f'time: {run_time:.2f}s, average batch time: {(run_time / (test_step + train_step)):.2f}s\n')
            # 保存至数组，之后用matplotlib画图
            y_label_all[0][fold], y_label_all[1][fold], y_label_all[2][fold] = test_loss, test_acc, test_auc

        # epoch总结阶段
        epoch_total += 1
        train_loss_aver /= params.train.k_fold
        train_acc_aver /= params.train.k_fold
        train_auc_aver /= params.train.k_fold
        test_loss_aver /= params.train.k_fold
        test_acc_aver /= params.train.k_fold
        test_auc_aver /= params.train.k_fold
        print('>>>>>>>>>>>>>>>>>>' + COLOR_LOG_Y + f"epoch: {epoch_total}"+ COLOR_LOG_END + '<<<<<<<<<<<<<<<<<<')
        print(COLOR_LOG_G + f'training: loss: {train_loss_aver:.4f}, acc: {train_acc_aver:.4f}, auc: {train_auc_aver: .4f}' + COLOR_LOG_END)
        print(COLOR_LOG_G + f'testing: loss: {test_loss_aver:.4f}, acc: {test_acc_aver:.4f}, auc: {test_auc_aver: .4f}' + COLOR_LOG_END)
        output_file.write(f"epoch: {epoch_total} | ")
        output_file.write(f'training: loss: {train_loss_aver:.4f}, acc: {train_acc_aver:.4f}, auc: {train_auc_aver: .4f}\n         | ')
        output_file.write(f'testing: loss: {test_loss_aver:.4f}, acc: {test_acc_aver:.4f}, auc: {test_auc_aver: .4f}\n')
        y_label_aver[0][epoch], y_label_aver[1][epoch], y_label_aver[2][epoch] = test_loss_aver, test_acc_aver, test_auc_aver

        # @add_fzq: 实时保存 aver 数据，防止训练中断丢失数据
        # 每次 epoch 结束都覆盖保存一次，确保即使中断也能保留已完成的 epoch 数据
        # np.savetxt(f'{LOGCHART_DIR}/{time_now}_aver.txt', y_label_aver)

    # @add_fzq 2025-12-24 17:28:09 -------------------------------------------
    # 计算总耗时 
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    total_hours = int(total_duration // 3600)
    total_minutes = int((total_duration % 3600) // 60)
    total_seconds = int(total_duration % 60)
    time_str = f'{total_hours}h {total_minutes}min {total_seconds}s'
    print(f'Total training time: {time_str}')
    output_file.write(f'Total training time: {time_str}\n')
    # @add_fzq 2025-12-24 17:28:09 -------------------------------------------

    output_file.close()

    torch.save(model, f=f'{config.path.MODEL_DIR}/{time_now}.pt')
    np.savetxt(f'{config.path.CHART_DIR}/{time_now}_all.txt', y_label_all)
    np.savetxt(f'{config.path.CHART_DIR}/{time_now}_aver.txt', y_label_aver)
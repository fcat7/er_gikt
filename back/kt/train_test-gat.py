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
from gikt_gat import GIKT
from config import get_config
from utils import gen_gikt_graph, build_adj_list

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

# 设置设备
def get_device():
    try:
        # 尝试使用CUDA
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            # 测试设备是否可用
            torch.tensor([1.0]).to(device)
            return device
    except RuntimeError as e:
        print(f"CUDA device error: {e}")
        print("Falling back to CPU")
    
    # 如果CUDA不可用，使用CPU
    return torch.device('cpu')

# 在代码中使用
DEVICE = get_device()
print(f"Using device: {DEVICE}")

if __name__ == '__main__':

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # @add_fzq 2025-12-24 17:28:09 -------------------------------------------
    # 1. 解决时区问题：强制使用 UTC+8 (北京时间)
    from datetime import timedelta, timezone
    beijing_time = datetime.now(timezone(timedelta(hours=8)))
    time_now = beijing_time.strftime('%Y%m%d_%H%M')
    # @add_fzq 2025-12-24 17:28:09 -------------------------------------------

    output_path = os.path.join('output', f'{time_now}.log')
    output_dir = os.path.dirname(output_path)  # 获取目录路径
    # 创建目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # @add_fzq 2025-12-24 17:28:09 -------------------------------------------
    # 2. 解决日志丢失问题：使用 'a' (append) 模式，并设置 buffering=1 (行缓冲)
    # 这样每写入一行就会自动 flush 到磁盘，防止断连丢失
    output_file = open(output_path, 'a', buffering=1)
    # @add_fzq 2025-12-24 17:28:09 -------------------------------------------


    # 使用数据集
    #  Available: ['assist09', 'assist12', 'ednet_kt1'] 
    dataset_name = 'assist09'
    config = get_config(dataset_name)
    print(f"Using dataset: {dataset_name}, Data dir: {config.PROCESSED_DATA_DIR}\n")
    output_file.write(f"Using dataset: {dataset_name}\n")

    # 训练时的超参数 - 旧版本备份
    # params = {
    #     'max_seq_len': config.MAX_SEQ_LEN,
    #     'min_seq_len': config.MIN_SEQ_LEN,
    #     'epochs': 20, # 每折训练的轮数
    #     'lr': 0.01,
    #     'lr_gamma': 0.95,
    #     'batch_size': 16,
    #     'size_q_neighbors': 4,
    #     'size_s_neighbors': 10,
    #     'num_workers': 2, # 设置为8核，最大化利用CPU
    #     'prefetch_factor': 4,
    #     'agg_hops': 3,
    #     # 'emb_dim': 100, # @change_fzq: 使用 config 中的统一配置，确保预训练维度一致
    #     'emb_dim': config.SIZE_EMBEDDING,
    #     'hard_recap': True,
    #     'dropout': (0.2, 0.4),
    #     'rank_k': 10,
    #     'k_fold': 1,  # 几折交叉验证
    #     'use_cognitive_model': True, # 是否使用认知模型 (CognitiveRNNCell)
    #     'pre_train': False, # 是否使用预训练的向量
    #     'use_BCELoss': False, # 是否使用 BCELoss 代替 BCEWithLogitsLoss
    #     'verbose': True, # 控制是否打印详细的 step 日志 (默认关闭以提高速度)
    #     'agg_method': 'gat' # 图聚合方法: 'gcn' 或 'gat'
    # }

    # 训练时的超参数 - 全量数据集版本
    params = {
        'max_seq_len': config.MAX_SEQ_LEN,
        'min_seq_len': config.MIN_SEQ_LEN,
        'epochs': 50, # 每折训练的轮数
        'lr': 0.002,
        'lr_gamma': 0.95,
        'batch_size': 128,
        # 'batch_size': 16,
        'size_q_neighbors': 4,
        'size_s_neighbors': 10,
        'num_workers': 8, # macOS 上建议设为0，防止 DataLoader 多进程卡死
        'prefetch_factor': 2,
        'agg_hops': 3,
        'emb_dim': 100,
        'hard_recap': True,
        'dropout': (0.2, 0.4),
        'rank_k': 10,
        'k_fold': 1,  # 启用 5 折交叉验证，获取更可靠的实验指标
        'use_cognitive_model': False, # 是否使用认知模型 (CognitiveRNNCell)
        'pre_train': False, # 是否使用预训练的向量
        'use_BCELoss': False, # 是否使用 BCELoss 代替 BCEWithLogitsLoss
        'verbose': True, # 控制是否打印详细的 step 日志 (默认关闭以提高速度)
        'agg_method': 'gat' # 图聚合方法: 'gcn' 或 'gat'
    }

    # 打印并写超参数
    output_file.write(str(params) + '\n')
    print(params)
    batch_size = params['batch_size']
    # 构建模型需要的数据结构, 全部转化为正确类型tensor再输入模型中
    qs_table = torch.tensor(sparse.load_npz(os.path.join(config.PROCESSED_DATA_DIR, 'qs_table.npz')).toarray(), dtype=torch.int64, device=DEVICE)  # [num_q, num_c]
    num_question = torch.tensor(qs_table.shape[0], device=DEVICE)
    num_skill = torch.tensor(qs_table.shape[1], device=DEVICE)
    q_neighbors_list, s_neighbors_list = build_adj_list(config.PROCESSED_DATA_DIR)
    q_neighbors, s_neighbors = gen_gikt_graph(q_neighbors_list, s_neighbors_list, params['size_q_neighbors'], params['size_s_neighbors'])
    q_neighbors = torch.tensor(q_neighbors, dtype=torch.int64, device=DEVICE)
    s_neighbors = torch.tensor(s_neighbors, dtype=torch.int64, device=DEVICE)

    # 初始化模型
    model = GIKT(
        num_question, num_skill, q_neighbors, s_neighbors, qs_table,
        agg_hops=params['agg_hops'],
        emb_dim=params['emb_dim'],
        dropout=params['dropout'],
        hard_recap=params['hard_recap'],
        use_cognitive_model=params['use_cognitive_model'],
        pre_train=params['pre_train'],
        data_dir=config.PROCESSED_DATA_DIR,
        agg_method=params['agg_method']    
    ).to(DEVICE)

    # @change_fzq 2026-01-08: 修改损失函数为 BCELoss
    # 原因：模型输出已经是 Sigmoid 概率，BCEWithLogitsLoss 会再次 Sigmoid，导致梯度消失
    # 原始损失函数备份：
    loss_fun = torch.nn.BCEWithLogitsLoss().to(DEVICE) # 损失函数
    if params['use_BCELoss']:
        loss_fun = torch.nn.BCELoss().to(DEVICE)
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
    optimizer = torch.optim.Adam(params=model.parameters(), lr=params['lr'])
    torch.optim.lr_scheduler.ExponentialLR(optimizer, params['lr_gamma'])
    # 在matplotlib中绘制的y轴数据，三行分别表示loss, acc, auc
    y_label_aver = np.zeros([3, params['epochs']]) # 平均精度值
    y_label_all = np.zeros([3, params['epochs'] * params['k_fold']]) # 全部精度值

    # KFold的shuffle操作是在用户级别进行的，而不是在答题记录级别
    if params['k_fold'] == 1:
        # 如果 k_fold 为 1，使用 ShuffleSplit 进行单次划分 (80% 训练, 20% 测试)
        k_fold = ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    else:
        # 否则使用 KFold 进行交叉验证
        k_fold = KFold(n_splits=params['k_fold'], shuffle=True, random_state=42)

    for epoch in range(params['epochs']):
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
                train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=params['num_workers'],
                                            pin_memory=True, prefetch_factor=params['prefetch_factor'])
                test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=params['num_workers'],
                                            pin_memory=True, prefetch_factor=params['prefetch_factor'])
            train_data_len, test_data_len = len(train_set), len(test_set)
            # @delete_fzq 2025-12-23 22:10:41
            #  print('===================' + LOG_Y + f'epoch: {epoch_total + 1}'+ LOG_END + '====================')
            # @add_fzq 2025-12-23 22:10:41
            print('===================' + config.LOG_Y + f'fold: {fold + 1}'+ config.LOG_END + '====================')

            # 训练阶段，既有前向传播，也有反向传播
            print('-------------------training------------------')
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

                # -- add_fzq 2025-12-25 17:15:13-----------------
                # 解包数据：[batch, seq, 5] -> x, y, mask, interval, response
                x = data[:, :, 0].to(torch.long).to(DEVICE)
                y_target = data[:, :, 1].to(torch.long).to(DEVICE)
                mask = data[:, :, 2].to(torch.bool).to(DEVICE)
                interval_time = data[:, :, 3].to(torch.float32).to(DEVICE)
                response_time = data[:, :, 4].to(torch.float32).to(DEVICE)

                # @add_fzq 2026-01-08: 输入数据 NaN 运行时检查与清洗
                # 必须步骤：防止 data/assist09/*.npy 中存在的 NaN 导致训练崩溃
                if torch.isnan(interval_time).any():
                    interval_time = torch.nan_to_num(interval_time, nan=0.0)
                if torch.isnan(response_time).any():
                    response_time = torch.nan_to_num(response_time, nan=0.0)

                y_hat = model(x, y_target, mask, interval_time, response_time)
                # --------------------------------------------
        
                y_hat = torch.masked_select(y_hat, mask)
                y_pred = torch.ge(y_hat, torch.tensor(0.5))
                y_target = torch.masked_select(y_target, mask)

                # @add_fzq 2026-01-08: 数值截断，防止 BCELoss 计算 Log(0) 溢出
                y_hat = torch.clamp(y_hat, min=1e-6, max=1.0 - 1e-6)

                loss = loss_fun(y_hat, y_target.to(torch.float32))
                train_loss += loss.item()
                # 计算acc
                acc = torch.sum(torch.eq(y_target, y_pred)) / torch.sum(mask)
                train_right += torch.sum(torch.eq(y_target, y_pred))
                train_total += torch.sum(mask)
                # 计算auc
                auc = roc_auc_score(y_target.cpu(), y_pred.cpu())
                train_auc += auc * len(x) / train_data_len
                loss.backward()
                optimizer.step()
                train_step += 1
                if params.get('verbose', False):
                    print(f'step: {train_step}, loss: {loss.item():.4f}, acc: {acc.item():.4f}, auc: {auc:.4f}')
            train_loss, train_acc = train_loss / train_step, train_right / train_total
            train_loss_aver += train_loss
            train_acc_aver += train_acc
            train_auc_aver += train_auc

            # 测试阶段，只有前向传递，没有反向传播阶段
            print('-------------------testing------------------')
            test_step = test_loss = test_total = test_right = test_auc = 0
            # 每轮训练第几个批量, 总损失, 训练的真实样本个数, 其中正确的个数, 总体的auc
            for data in test_loader:

                # -- delete_fzq 2025-12-25 17:15:13-----------------
                # x, y_target, mask = data[:, :, 0].to(DEVICE), data[:, :, 1].to(DEVICE), data[:, :, 2].to(torch.bool).to(DEVICE)
                # y_hat = model(x, y_target, mask) # 原始代码
                # --------------------------------------------

                # ---------------- 新增时间特征 ----------------
                # -- add_fzq 2025-12-25 17:15:13-----------------
                # 解包数据：[batch, seq, 5] -> x, y, mask, interval, response
                x = data[:, :, 0].to(torch.long).to(DEVICE)
                y_target = data[:, :, 1].to(torch.long).to(DEVICE)
                mask = data[:, :, 2].to(torch.bool).to(DEVICE)
                interval_time = data[:, :, 3].to(torch.float32).to(DEVICE)
                response_time = data[:, :, 4].to(torch.float32).to(DEVICE)
                
                # @add_fzq 2026-01-08: 测试集同样需要 NaN 检查
                if torch.isnan(interval_time).any():
                    interval_time = torch.nan_to_num(interval_time, nan=0.0)
                if torch.isnan(response_time).any():
                    response_time = torch.nan_to_num(response_time, nan=0.0)

                y_hat = model(x, y_target, mask, interval_time, response_time)
                # -- add_fzq 2025-12-25 17:15:13-----------------
                # --------------------------------------------
                
                y_hat = torch.masked_select(y_hat, mask.to(torch.bool))
                y_pred = torch.ge(y_hat, torch.tensor(0.5))
                y_target = torch.masked_select(y_target, mask.to(torch.bool))
                
                # @add_fzq 2026-01-08: 截断
                y_hat = torch.clamp(y_hat, min=1e-6, max=1.0 - 1e-6)
                
                loss = loss_fun(y_hat, y_target.to(torch.float32))
                test_loss += loss.item()
                # 计算acc
                acc = torch.sum(torch.eq(y_target, y_pred)) / torch.sum(mask)
                test_right += torch.sum(torch.eq(y_target, y_pred))
                test_total += torch.sum(mask)
                # 计算auc
                auc = roc_auc_score(y_target.cpu(), y_pred.cpu())
                test_auc += auc * len(x) / test_data_len
                test_step += 1
                if params.get('verbose', False):
                    print(f'step: {test_step}, loss: {loss.item():.4f}, acc: {acc.item():.4f}, auc: {auc:.4f}')
            test_loss, test_acc = test_loss / test_step, test_right / test_total
            test_loss_aver += test_loss
            test_acc_aver += test_acc
            test_auc_aver += test_auc

            # fold总结阶段
            
            time1 = time.time()
            run_time = time1 - time0
            print(config.LOG_B + f'training: loss: {train_loss:.4f}, acc: {train_acc:.4f}, auc: {train_auc: .4f}' + config.LOG_END)
            print(config.LOG_B + f'testing: loss: {test_loss:.4f}, acc: {test_acc:.4f}, auc: {test_auc: .4f}' + config.LOG_END)
            print(config.LOG_B + f'time: {run_time:.2f}s, average batch time: {(run_time / test_step):.2f}s' + config.LOG_END)
            # 保存输出至本地文件
            output_file.write(f'  fold {fold+1} | ')
            output_file.write(f'training: loss: {train_loss:.4f}, acc: {train_acc:.4f}, auc: {train_auc: .4f} | ')
            output_file.write(f'testing: loss: {test_loss:.4f}, acc: {test_acc:.4f}, auc: {test_auc: .4f} | ')
            output_file.write(f'time: {run_time:.2f}s, average batch time: {(run_time / test_step):.2f}s\n')
            # 保存至数组，之后用matplotlib画图
            y_label_all[0][fold], y_label_all[1][fold], y_label_all[2][fold] = test_loss, test_acc, test_auc

        # epoch总结阶段
        epoch_total += 1
        train_loss_aver /= params['k_fold']
        train_acc_aver /= params['k_fold']
        train_auc_aver /= params['k_fold']
        test_loss_aver /= params['k_fold']
        test_acc_aver /= params['k_fold']
        test_auc_aver /= params['k_fold']
        print('>>>>>>>>>>>>>>>>>>' + config.LOG_Y + f"epoch: {epoch_total}"+ config.LOG_END + '<<<<<<<<<<<<<<<<<<')
        print(config.LOG_G + f'training: loss: {train_loss_aver:.4f}, acc: {train_acc_aver:.4f}, auc: {train_auc_aver: .4f}' + config.LOG_END)
        print(config.LOG_G + f'testing: loss: {test_loss_aver:.4f}, acc: {test_acc_aver:.4f}, auc: {test_auc_aver: .4f}' + config.LOG_END)
        output_file.write(f"epoch: {epoch_total} | ")
        output_file.write(f'training: loss: {train_loss_aver:.4f}, acc: {train_acc_aver:.4f}, auc: {train_auc_aver: .4f} | ')
        output_file.write(f'testing: loss: {test_loss_aver:.4f}, acc: {test_acc_aver:.4f}, auc: {test_auc_aver: .4f}\n')
        y_label_aver[0][epoch], y_label_aver[1][epoch], y_label_aver[2][epoch] = test_loss_aver, test_acc_aver, test_auc_aver

        # @add_fzq: 实时保存 aver 数据，防止训练中断丢失数据
        # 每次 epoch 结束都覆盖保存一次，确保即使中断也能保留已完成的 epoch 数据
        # np.savetxt(f'{config.CHART_DIR}/{time_now}_aver.txt', y_label_aver)

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




    torch.save(model, f=f'{config.MODULE_DIR}/{time_now}.pt')
    np.savetxt(f'{config.CHART_DIR}/{time_now}_all.txt', y_label_all)
    np.savetxt(f'{config.CHART_DIR}/{time_now}_aver.txt', y_label_aver)
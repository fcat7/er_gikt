"""
预训练问题和技能向量
"""
import math
import os
import torch
import torch.nn as nn
from scipy import sparse
from torch.utils.tensorboard import SummaryWriter

from config import Config, DEVICE
from pebg import PEBG

# 获取配置
dataset_name = os.environ.get('DATASET', 'assist09-sample_10%')
# Config 仅负责路径
config = Config(dataset_name)

batch_size = 32
epoch_num = 100
emb_dim = 100
print(f"Pre-training on dataset: {dataset_name}, Data dir: {config.PROCESSED_DATA_DIR}")

# 路径设置
data_dir = config.PROCESSED_DATA_DIR
qs_path = os.path.join(data_dir, 'qs_table.npz')
qq_path = os.path.join(data_dir, 'qq_table.npz')
ss_path = os.path.join(data_dir, 'ss_table.npz')

# 加载数据
qs_table = torch.tensor(sparse.load_npz(qs_path).toarray(), dtype=torch.int64, device=DEVICE) # [num_q, num_c]
qq_table = torch.tensor(sparse.load_npz(qq_path).toarray(), dtype=torch.int64, device=DEVICE) # [num_q, num_c]
ss_table = torch.tensor(sparse.load_npz(ss_path).toarray(), dtype=torch.int64, device=DEVICE) # [num_q, num_c]

num_q = qs_table.shape[0]
num_s = qs_table.shape[1]
print(f"qs_table: {qs_table.shape}, qq_table: {qq_table.shape}, ss_table: {ss_table.shape}")

# @change_fzq: 针对小样本数据集，适当降低 batch_size 以增加迭代频次，避免收敛过快陷入局部最优
# 原值: 256 -> 建议: 32 (平衡图结构视野和随机性)
num_batch = math.ceil(num_q / batch_size)

# 初始化模型
model = PEBG(qs_table, qq_table, ss_table, emb_dim=emb_dim).to(DEVICE)
print('Start training PEBG model...')

optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
mse_loss = [nn.MSELoss().to(DEVICE) for _ in range(3)]

# 日志目录
log_dir = os.path.join(config.path.OUTPUT_DIR, 'logs_pretrain', dataset_name)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir=log_dir)

# 训练循环
# @change_fzq: 对于小数据集，增加 Epoch 以确保充分收敛 (20 -> 50)
for epoch in range(epoch_num):
    train_loss = 0  # 总损失
    for idx_batch in range(num_batch):
        optimizer.zero_grad()  # 梯度清零
        # 计算索引
        idx_start = idx_batch * batch_size
        idx_end = min((idx_batch + 1) * batch_size, num_q)  # 结束索引
        
        # 通过索引计算相应的批量
        q_embedding = model.q_embedding[idx_start: idx_end]  # [batch_size, emb_dim]
        s_embedding = model.s_embedding  # [num_s, emb_dim] (全部)
        
        qs_target = model.qs_target[idx_start: idx_end]  # [batch_size, num_s]
        qq_target = model.qq_target[idx_start: idx_end, idx_start: idx_end]  # [batch_size, batch_size]
        ss_target = model.ss_target  # [num_s, num_s] (全部)
        
        # 计算logit
        qs_logit, qq_logit, ss_logit = model(q_embedding, s_embedding)
        
        # 计算损失
        # @fix_fzq: 添加数值稳定性保护 (epsilon)，防止 sqrt(0) 导致梯度计算 NaN
        epsilon = 1e-8
        loss_qs = torch.sqrt(mse_loss[0](qs_logit, qs_target) + epsilon).sum()  # L1
        loss_qq = torch.sqrt(mse_loss[1](qq_logit, qq_target) + epsilon).sum()  # L2
        loss_ss = torch.sqrt(mse_loss[2](ss_logit, ss_target) + epsilon).sum()  # L3
        loss = loss_qs + loss_qq + loss_ss  # 总损失
        
        train_loss += loss.item()
        loss.backward()  # 反向传播
        
        # @add_fzq: 添加梯度裁剪，防止梯度爆炸导致 loss 为 nan
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()  # 参数优化
        
    avg_loss = train_loss / num_batch
    print(f'----------epoch: {epoch + 1}, total_loss: {train_loss:.4f}, avg_loss: {avg_loss:.4f}')
    writer.add_scalar(tag='pebg_loss', scalar_value=avg_loss, global_step=epoch)

# 保存训练好的 Embedding
save_q_path = os.path.join(data_dir, 'q_embedding.pt')
save_s_path = os.path.join(data_dir, 's_embedding.pt')

torch.save(model.q_embedding, f=save_q_path)
torch.save(model.s_embedding, f=save_s_path)

print(f"Embeddings saved to:\n  {save_q_path}\n  {save_s_path}")
writer.close()
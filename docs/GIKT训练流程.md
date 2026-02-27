GIKT 训练流程（基于 V2 数据流）

```text
前置：完成数据处理流程 → 生成 train.parquet / test.parquet / metadata.json
        （见：数据处理逻辑.md）
```

### 一、整体训练架构概览

```text
config/experiments/exp_xxx.toml
          ↓ HyperParameters.load()
train_test.py
  ├─ 解析实验配置（数据集名、batch_size、k_fold 等）
  ├─ 加载 metadata.json + 图结构 (qs_table, 邻接表)
  ├─ 构建 UnifiedParquetDataset（train/test）
  ├─ 根据 groups 决定 KFold / GroupKFold / ShuffleSplit
  ├─ 外层循环：Fold
  │    ├─ 初始化 GIKT 模型 + 优化器 + 调度器
  │    ├─ 内层循环：Epoch
  │    │    ├─ Training：使用带增强的数据集（dataset_train_augment）
  │    │    └─ Validation：使用干净数据集（dataset_train_clean）
  │    └─ Early Stopping & 记录每折最优结果
  └─ 可选：使用独立 test.parquet 做 Holdout 测试评估
```

### 二、数据集层：`UnifiedParquetDataset`

**文件**：`back/kt/dataset.py`  
**类**：`UnifiedParquetDataset`

- **输入依赖**：
  - `train.parquet` / `test.parquet`：由 `data_process.py` 生成的变长序列数据
  - `metadata.json`：用于读取 `max_seq_len` 等配置信息

- **初始化逻辑**：
  - 根据 `mode in {"train", "test"}` 选择对应的 parquet 文件
  - 加载 `metadata.json`，读取 `config_at_processing.max_seq_len` 作为统一的序列上限
  - 若存在 `group_id` 列，则暴露为 `self.groups`，供 `GroupKFold` 使用（避免窗口化数据泄露）

- **`__getitem__` 输出格式**：单个样本被整理为形状为 `[max_seq_len, 6]` 的 `Tensor`，六个通道依次为：

  ```text
  [:, 0] → q_seq         题目 ID 序列（已通过 question2idx 映射）
  [:, 1] → r_seq         答题结果（0/1）
  [:, 2] → mask          有效位置标记（True 表示该时间步存在真实交互）
  [:, 3] → t_interval    时间间隔特征（log → tanh 后的归一化结果）
  [:, 4] → t_response    作答时长特征（相对中位数 RT → tanh）
  [:, 5] → eval_mask     评估掩码（区分“历史预热”与“评估位置”）
  ```

- **核心能力**：
  - **动态 Padding**：内部 `_pad_sequence` 根据 `max_seq_len` 对变长序列进行截断或补零，保证模型侧恒定形状。
  - **数据增强（可选）**：
    - 通过 `prob_mask` 在 `mask == True` 的位置随机丢弃一部分交互，模拟缺失数据或子采样。
    - 保证至少保留一个有效位置（防止整个序列被 Mask 掉）。
    - 同步更新 `mask` 与 `eval_mask`，保持训练/评估语义一致。

> 直观理解：`UnifiedParquetDataset` 是 V2 数据流的“入口适配层”，把 Parquet 里的变长 List 列转换成模型可消费的 `[max_seq_len, 6]` 张量格式。

### 三、训练脚本层：`train_test.py`

**文件**：`back/kt/train_test.py`

#### 1. 配置与环境准备

- 固定随机种子 `set_seed(42)`，确保实验可复现。
- 通过命令行参数决定实验配置：

  ```bash
  python train_test.py --name default           # 使用 sample 配置
  python train_test.py --full --name assist09   # 使用 full 数据集配置
  ```

- 超参数加载：

  ```python
  exp_config_path = get_exp_config_path(isFull=args.full, name=args.name)
  params = HyperParameters.load(exp_config_path=exp_config_path)
  ```

- 数据集配置加载：

  ```python
  dataset_name = params.train.dataset_name
  config = Config(dataset_name=dataset_name)
  ```

- 自动选择 AMP 与设备（GPU / CPU），初始化 `GradScaler` 与日志文件。

#### 2. 图结构与邻接信息

- 从 `config.PROCESSED_DATA_DIR` 加载：
  - `qs_table.npz`：题目-技能稀疏邻接矩阵，用于确定 `num_question`、`num_skill`。
  - 邻接表：通过 `build_adj_list` + `gen_gikt_graph` 生成：

    ```python
    q_neighbors, s_neighbors = gen_gikt_graph(
        q_neighbors_list, s_neighbors_list,
        params.model.size_q_neighbors,
        params.model.size_s_neighbors
    )
    ```

- 这些张量在模型构造时传入 `GIKT`，作为图结构的先验信息。

#### 3. 数据加载与 K-Fold 划分

- **构建数据集**：

  ```python
  dataset_train_augment = UnifiedParquetDataset(
      config,
      augment=params.train.enable_data_augmentation,
      prob_mask=params.train.aug_mask_prob,
      mode='train'
  )
  dataset_train_clean = UnifiedParquetDataset(config, augment=False, mode='train')
  dataset_test = UnifiedParquetDataset(config, augment=False, mode='test')  # 可选
  ```

- **K-Fold / GroupKFold 策略**：
  - 若 `dataset_train_clean.groups` 不为 `None`：
    - 使用 `GroupKFold` 或 `GroupShuffleSplit`（当 `k_fold == 1`）进行基于用户的划分，严格避免同一用户的窗口落在不同 Fold。
  - 否则：
    - 使用标准 `KFold` / `ShuffleSplit`，随机划分索引。

- 每个 Fold 内：
  - 使用 `Subset(dataset_train_augment, train_indices)` 作为训练集。
  - 使用 `Subset(dataset_train_clean, val_indices)` 作为验证集。
  - 通过 `DataLoader` 构建批次，支持多进程加载和 `prefetch_factor`。

#### 4. 模型构建与优化器

- **模型实例化**（每个 Fold 重置一次）：

  ```python
  model = GIKT(
      num_question, num_skill, q_neighbors, s_neighbors, qs_table,
      agg_hops=params.model.agg_hops,
      emb_dim=params.model.emb_dim,
      dropout_linear=params.model.dropout_linear,
      dropout_gnn=params.model.dropout_gnn,
      drop_edge_rate=params.model.drop_edge_rate,
      feature_noise_scale=params.model.feature_noise_scale,
      hard_recap=params.model.hard_recap,
      use_cognitive_model=params.model.use_cognitive_model,
      cognitive_mode=params.model.cognitive_mode,
      pre_train=params.model.pre_train,
      data_dir=config.PROCESSED_DATA_DIR,
      agg_method=params.model.agg_method,
      recap_source=params.model.recap_source,
      use_pid=params.model.use_pid,
      pid_mode=params.model.pid_mode,
      pid_ema_alpha=params.model.pid_ema_alpha,
      pid_lambda=params.model.pid_lambda,
      pid_init_i=params.model.pid_init_i,
      pid_init_d=params.model.pid_init_d,
      guessing_prob_init=params.model.guessing_prob_init,
      slipping_prob_init=params.model.slipping_prob_init
  ).to(DEVICE)
  ```

- **损失函数与优化器**：
  - 统一采用 `BCEWithLogitsLoss`（与 TF 对齐模式保持一致）。
  - 优化器：`Adam`，可选权重衰减。
  - 学习率调度器：`ExponentialLR`，在每个 Epoch 结束时根据 step_count 更新。
  - 同时加入对 4PL 相关参数的正则化项（guessing/slipping/discrimination 等）。

#### 5. 单 Epoch 训练流程（Training Phase）

对于每个 Epoch，训练部分的关键逻辑：

- 从 `train_loader` 取出批次 `data`，形状 `[batch_size, max_seq_len, 6]`。
- 拆分各个通道：

  ```python
  x = data_gpu[:, :, 0].long()      # 题目 ID 序列
  y_target = data_gpu[:, :, 1].long()
  mask = data_gpu[:, :, 2].bool()
  interval_time = data_gpu[:, :, 3].float()
  response_time = data_gpu[:, :, 4].float()
  eval_mask = data_gpu[:, :, 5].bool()  # 若不存在则退化为 mask
  ```

- 前向过程：
  - 调用 `model(x, y_target, mask, interval_time, response_time)`，得到 logits `y_hat`。
  - 对齐“错位预测”语义：跳过序列首位，只预测 `t>=2` 的位置：

    ```python
    y_hat = y_hat[:, 1:]
    y_target_shift = y_target[:, 1:].float()
    mask_valid = mask[:, 1:]
    eval_mask_valid = eval_mask[:, 1:]
    final_mask = mask_valid & eval_mask_valid
    ```

  - 使用 `final_mask` 选出真实参与损失与指标计算的时间步：

    ```python
    y_hat_flat = torch.masked_select(y_hat, final_mask)
    y_target_flat = torch.masked_select(y_target_shift, final_mask)
    loss = loss_fun(y_hat_flat, y_target_flat) + reg_loss
    ```

- 反向与优化：
  - 使用 AMP (`autocast + GradScaler`) 进行混合精度训练。
  - 标准的 `backward()` → `step()` → `update()` 流程。

- 指标计算：
  - 对 `y_hat_flat` 做 sigmoid 得到 `y_prob`，计算：
    - Accuracy（基于 0.5 阈值）
    - Global AUC（整轮 epoch 的全量样本）
  - 额外收集“未应用 eval_mask 过滤”的 AUC，用于诊断 eval_mask 对样本量和性能的影响。

#### 6. 验证与 Early Stopping

- 验证阶段与训练阶段在逻辑上高度一致：
  - 同样使用 `eval_mask` 与 `mask` 组合成 `final_mask`。
  - 计算验证集的 Loss / Acc / Global AUC。
  - 记录 `val_auc`，并驱动 Early Stopping：

    ```python
    if val_auc > best_fold_val_auc:
        best_fold_val_auc = val_auc
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= params.train.patience:
            break  # 触发早停
    ```

- 同时输出 Eval Mask 的诊断信息，监控被过滤的样本比例与 AUC 差异。

#### 7. 独立测试集评估（Holdout Test）

- 若存在 `test.parquet`，则在每个 Fold 结束后，用当前 Fold 的最优模型在测试集上评估一次：
  - 完全沿用 `eval_mask` 语义，对 `[1:]` 时间步进行过滤与评估。
  - 记录每折的 Test AUC，最终输出：
    - 各 Fold AUC
    - 均值 ± 标准差
    - 最佳/最差 Fold

#### 8. 结果持久化

- 训练结束后：
  - 将跨 Fold × Epoch 的验证指标保存到 `chart_dir` 目录下的 txt 文件：

    ```text
    {time_now}_all.txt   # 展开后的 (Metric × Fold×Epoch)
    {time_now}_aver.txt  # 按 Fold 求平均后的 (Metric × Epoch)
    ```

  - 记录总训练时长、每 Epoch 运行耗时等信息。
  - （可选）保存模型权重到 `MODEL_DIR`。

---

### 四、从“数据处理”到“模型训练”的完整闭环

- **数据处理阶段**（见 `数据处理逻辑.md`）负责：
  - 将原始 CSV → 统一标准列 DataFrame → Parquet 序列 + 图结构 + 题目特征 + metadata。
- **数据加载阶段**（`UnifiedParquetDataset`）负责：
  - 将 Parquet 中的变长序列变成 `[max_seq_len, 6]` 的训练张量。
- **训练脚本阶段**（`train_test.py`）负责：
  - 组织 K-Fold / GroupKFold、数据增强、AMP 训练、Eval Mask 策略、Early Stopping 与最终评估。

三者合在一起，构成了 GIKT 在 V2 数据流下的完整训练流水线。


# GIKT PyTorch Implementation Update Log

## Version dev_win v0.1.0 约定版本控制规范 和 GIT 提交规范

*`[0.1.0] - 2026-01-23 🎯 约定版本控制 和 GIT 提交规范`*

1. Added
   - 约定版本控制规范 和 GIT 提交规范.
2. Changed
   - 训练/测试阶段一次性把 batch 整体搬到 GPU，再在 GPU 上切片出 x/y/mask/时间特征，避免多次 PCIe 传输。
3. Fixed
   - 修复 average batch time 计算公式。

> ###### 备注
>
> **Date:** 2026-01-23
> **Status:** ✅ Verified 
> **Code Branch:** `dev_win`


## Version  3.2 Alignment (Current Best)

**Date:** 2026-01-13
**Status:** ✅ Verified (AUC ~0.755 on Sample Dataset)
**Code Branch:** `hsei_fix_v3.2_alignment`

### 1. 核心数学对齐 (Math Alignment with TF)

为了解决 PyTorch 版本与 TensorFlow 原版在数学计算上的微小但关键的差异，进行了以下修正。这些修改即便是关闭 HSEI 模式也能带来显著提升 (AUC ~0.74+)。

- **输出层 (Logits Output)**:
  - `back/kt/gikt.py`: 修改 `predict` 方法。当启用 `enable_tf_alignment` 时，移除最后一层的 `torch.sigmoid`，直接输出 Logits。
  - **原因**: 避免“双重 Sigmoid”问题 (模型输出 Sigmoid -> Loss 再次 Sigmoid)。
- **权重初始化 (Xavier Init)**:
  - `back/kt/gikt.py`: 新增 `reset_parameters` 方法，对所有 Linear 层应用 `xavier_uniform_` 初始化。
  - **原因**: 匹配 TensorFlow 的默认初始化策略，加速收敛并跳出局部最优。
- **损失函数 (BCEWithLogitsLoss)**:
  - `back/kt/train_test.py`: 强制使用 `nn.BCEWithLogitsLoss` 配合 Logits 输出。
  - **原因**: 提供比 `BCELoss` 更高的数值稳定性 (LogSumExp trick)。
- **AUC 计算精度**:
  - `back/kt/train_test.py`: 将 `roc_auc_score` 的输入从 0/1 标签 (`y_pred`) 改为连续概率值 (`y_prob`)。
  - **原因**: 提升评估指标的精度。

### 2. 模型结构优化 (Model Architecture - HSEI Mode)

在数学对齐的基础上，进一步优化了 `hsei` (History State = Input Projection) 模式的实现。

- **特征变换 (Feature Transform)**:
  - `back/kt/gikt.py`: 引入 `Linear(emb_dim, emb_dim) + ReLU` 层。
  - **位置**: 在 Question Embedding 进入 LSTM 之前应用。
  - **原因**: 增加非线性特征表达，严格对齐 TF 的 `dense(relu)` 操作。
- **宽输入保持 (Wide Inputs)**:
  - `back/kt/gikt.py`: 保持 LSTM 输入维度为 `2 * emb_dim`。
  - **原因**: 修复了 v4 版本因降维导致的 CognitiveRNNCell 性能下降问题。
- **Target 聚合 (Target Aggregation)**:
  - `back/kt/gikt.py`: 在 `hsei` 模式下，对 Target Question (`q_next`) 进行 GNN 图聚合。
  - **原因**: 确保预测时的特征构建与历史状态中的特征构建方式一致。

### 3. 配置管理 (Configuration)

为了支持 A/B 测试，引入了可配置开关。

- **参数**: `enable_tf_alignment` (bool)
- **文件**: `back/kt/params.py`
- **用法**:
  - `True`: 启用 Logits 输出、Xavier 初始化、BCEWithLogitsLoss。
  - `False`: 保持旧版行为 (Sigmoid 输出 + BCELoss)。

### 4. 实验记录 (Experiment Results)

| Experiment ID    | Config Name        | Key Features                                                  | AUC (Sample)     | Note                                                                |
| :--------------- | :----------------- | :------------------------------------------------------------ | :--------------- | :------------------------------------------------------------------ |
| **[1626]** | `dev_v2`         | HSEI + TF Alignment +**Target Aggregation (Neighbors)** | **0.7535** | **Best Stable Version**. PyTorch features now match TF logic. |
| **[1928]** | `dev_v3 (无效)`  | `dev_v2` + **Target Transform (ReLU)**                | 0.7423           | Over-alignment caused regression. Reverted.                         |
| **[2233]** | `v3`             | Hybrid Mode (Wide Input)                                      | ~0.66            | Baseline                                                            |
| **[0131]** | `v3.2_no_hsei`   | HSSI Mode +**TF Alignment**                             | ~0.748           | Math fix is crucial                                                 |
| **[0058]** | `v3.2_alignment` | **HSEI Mode** + **TF Alignment**                  | **~0.755** | Similar to dev_v2                                                   |

### Dev Version Technical Details

#### dev_v2 (1626) - Target Context Fix 🚀

此版本修复了 PyTorch 实现与 TF 原版在构建“目标问题上下文”（Target Context - 即待预测问题）时的核心差异。

- **Original PyTorch (Old)**: 仅使用静态查表得到的 Skill Embedding。
  - `qs_concat = cat(emb_q_next, emb_skill_table[skill_id])`
- **dev_v2 Mode (New)**: 使用 **图聚合后的 Skill 邻居特征**。
  - `qs_concat = cat(emb_q_next, agg_list[1])`
  - 对应 TF 逻辑：Next Question 作为一个查询向量，应当包含其在知识图谱中的邻域信息（1-hop neighbors），而不仅仅是自身的 ID Embedding。

#### dev_v3 (1928) - Target Transform Experiment ⚠️ (已废弃)

尝试将“目标问题”的处理逻辑与“历史输入”的处理逻辑进行严格对称对齐。

- **Change**: 对 `emb_q_next` 施加了 `Linear + ReLU` 特征变换。
- **Result**: **无效 (Regression)**. AUC 从 ~0.75 下降至 ~0.74。
- **Analysis**: 实验表明，虽然历史状态需要投影（用于压缩信息），但 Target Query 保持在原始 Embedding 空间（Untransformed）能更好地检索相关历史。

---

## Historical Versions

### Version 3 (Hybrid / Baseline)

- **特点**: 恢复了宽输入 (Wide Inputs)，解决了 CognitiveCell 的兼容性问题。
- **问题**: 缺少 TF 的特征变换层和数学对齐，AUC 卡在 0.66。

### Version 4 (Strict Alignment)

- **特点**: 强制所有输入降维到 `emb_dim`。
- **问题**: 虽然对齐了 TF 结构，但破坏了 CognitiveRNNCell 的输入假设，导致 AUC 暴跌至 0.60~0.64。

### Version 1 & 2

- **特点**: 初步尝试实现 HSEI。
- **问题**: 存在 `UnboundLocalError` 和特征空间不匹配问题。

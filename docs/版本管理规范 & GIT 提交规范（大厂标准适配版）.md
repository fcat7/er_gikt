[toc]

本文基于 **语义化版本（SemVer 2.0）** + **约定式提交（Conventional Commits 1.0）** 核心规范，适配阿里/字节等大厂工程标准，同时强化研究项目的 **实验追踪、结果溯源、论文关联** 特性，补充标准语法、预发布规则、回滚规范、分支配套、自动化适配等核心内容，实现工程规范性和科研可追溯性的深度融合，达成「提交 → 版本 → CHANGELOG → 论文实验」的全链路溯源闭环。

# GIT 约定式提交

基于 Conventional Commits 1.0 官方规范（阿里/字节大厂核心遵循），补充标准语法、完整提交结构、回滚类型、标准化标记，强化研究项目的实验/论文溯源特性，确保提交记录可直接作为研究日志，且所有提交需通过自动化工具校验。

## 核心标准语法

Git 提交信息的 **唯一标准语法**，**所有提交必须遵循**，禁止自定义格式，语法结构分为 **必选部分** 和 **可选部分**，其中 **标题行（Header）是强制的**，其余部分根据变更重要性补充：

*==GIT 提交语法==*

```
<type>[(!)](<scope>): <short description>   <-- Header: 机器可读，极简 (≤50字符，禁止视觉标记)
// 空行（必须）
[结果标记] <body>                            <-- Body: 视觉标记 + 详细描述 (每行≤72字符，标记必放此处)
// 空行(必须)
<footer>                                    <-- Footer: 结构化元数据 (实验ID、Closes等)
```

_^tab^_

> **核心强制规则**
>
> 1. **Header 极简原则**：仅保留 `类型+范围+简短描述`，**禁止在 Header 中添加任何视觉标记**（[🔴/🟡/🟢]），字符数 ≤50，适配 GitHub/GitLab 列表无截断展示；
> 2. **视觉标记专属 Body**：[🔴 BREAKING]/[🟡 CHANGE]/[🟢 SAFE] 必须放在 Body 第一行，仅无 Body 的极简提交可在 Header 末尾加极简标记（如 `[🟢]`）；
> 3. **空行强制分隔**：Header 与 Body、Body 与 Footer 之间必须有且仅有 1 个空行，适配 Git 日志解析；
> 4. **字符限制**：Body 每行 ≤72 字符（适配 `git log` 展示），Footer 结构化键值对每行 ≤72 字符。
>
>> ###### 视觉标记（或结果影响标记）
>>
>> 将研究项目的结果影响标记 **与提交信息深度融合**，**标记位置强制在 Body 第一行**，大厂视觉统一标识，优先级从高到低：`[🔴 BREAKING]` > `[🟡 CHANGE]` > `[🟢 SAFE]`，**禁止自定义标记**（如 `[重要]`/`[无影响]`），且标记需与 CHANGELOG 中的标记完全一致：
>>
>> - `[🔴 BREAKING]`：修改会改变实验结果（如 Loss 计算修复、数据泄露修复），需重跑实验，**强制关联实验 ID/论文表格**
>> - `[🟡 CHANGE]`：修改运行行为，预期结果不变，建议验证核心实验，可关联实验 ID；
>> - `[🟢 SAFE]`：无实验影响，纯工程/文档变更，无需关联实验 ID。
>>

*==语法详细说明==*

| 部分                    | 说明                                                                    |
| ----------------------- | ----------------------------------------------------------------------- |
| `<type>`              | 提交类型（小写，不可自定义），对应版本号递增规则                        |
| `[(!)]`               | 破坏性变更标记，加在类型后（如 feat!），强制递增主版本号                |
| `(<scope>)`           | 变更范围（小写，最小粒度），无明确模块时可省略                          |
| `<short description>` | 简短描述（≤50 字符），首字母小写、动宾结构、结尾不加句号               |
| `<body>`              | 详细正文，说明变更动机、实现方式、影响；结果标记必须放在首行            |
| `<footer>`            | 结构化脚注（键值对），标注实验 ID、论文阶段、回滚记录等，禁止无规则文本 |

> **提交类型（type）规范**
>
> 在原类型基础上，补充大厂必备的 **revert（回滚）、merge（合并）** 类型，删除冗余类型，明确 **各类型与版本号的联动规则**，**禁止自定义类型**（如 `opt`/`modify`），所有类型均适配自动化工具识别。
>
> *==提交类型（阿里标准适配，新增 revert/merge）==*
>
> | 类型         | 核心说明                                                                  | 版本号联动               | 标准示例（语法）                             |
> | :----------- | :------------------------------------------------------------------------ | ------------------------ | :------------------------------------------- |
> | `feat`     | 新增功能/实验模块/算法逻辑，向后兼容                                      | 次版本 MINOR 递增        | `feat(model): 实现CognitiveRNN模块`        |
> | `feat!`    | 新增功能但包含**不兼容 API 变更**                                   | 主版本 MAJOR 递增        | `feat!(data): 重构数据加载核心接口`        |
> | `fix`      | 修复代码 bug/逻辑错误，向后兼容                                           | 修订号 PATCH 递增        | `fix(data): 修正NaN填充导致的数据泄露`     |
> | `fix!`     | 修复 bug 但需**修改现有 API**，存在不兼容变更                       | 主版本 MAJOR 递增        | `fix!(eval): 重构AUC计算核心逻辑`          |
> | `docs`     | 仅文档/注释变更，如更新实验步骤、补充参数说明、修改 README                | 不递增                   | `docs(config): 更新超参数调优说明`         |
> | `perf`     | 性能优化，如速度/显存/算力优化，无功能逻辑变更                            | 修订号 PATCH 递增        | `perf(train): 启用混合精度训练`            |
> | `refactor` | 代码重构，非功能新增、非 bug 修复、非性能优化，仅调整代码结构/命名        | 不递增                   | `refactor(eval): 简化指标计算逻辑`         |
> | `test`     | 增加/修改/删除测试代码，如单元测试、实验验证测试、形状检查测试            | 不递增                   | `test(model): 添加邻居图缓存形状验证`      |
> | `chore`    | 构建/工具/依赖变动，如升级框架、修改配置文件、调整 CI 脚本                | 不递增（特殊情况 PATCH） | `chore(deps): 升级PyTorch至2.1.0`          |
> | `style`    | 代码格式调整，无逻辑变更，如缩进、换行、命名规范修正（阿里新增）          | 不递增                   | `style(train): 统一代码缩进为4个空格`      |
> | `revert`   | 回滚之前的提交，**大厂强制类型**，不可用 fix 替代                   | 回滚对应版本号           | `revert: fix(data): 修正NaN填充逻辑`       |
> | `merge`    | 分支合并，仅用于主分支/开发分支的正规合并，禁止在特性分支使用（阿里标准） | 不递增                   | `merge: feat-model-cognitive into develop` |

> **标准范围（Scope）规范**
>
> 范围用于 **精准标注变更的模块/领域**，遵循 **“最小粒度”** 原则（如修改模型的注意力模块，用 `model/attn` 而非 `model`），原范围基础上 **拓展并标准化**，同时支持 **自定义范围**（研究项目适配，如实验 ID、论文模块），自定义范围需遵循 `模块/子模块` 格式，小写英文，禁止无意义的范围标注（如 `all`/`code`）。
>
> *==基础标准范围==*
>
> | Scope          | 说明                         | 子模块示例          | 完整示例                                         |
> | :------------- | :--------------------------- | :------------------ | :----------------------------------------------- |
> | `model`      | 模型结构、前向/反向逻辑      | attn/rnn/emb        | `feat(model/attn): 新增多头注意力`             |
> | `data`       | 数据加载、预处理、格式转换   | load/process/aug    | `fix(data/process): 修复数据增强逻辑`          |
> | `train`      | 训练循环、优化器、lr 调度    | opt/lr/amp          | `perf(train/amp): 优化混合精度训练`            |
> | `eval`       | 评估、指标、验证逻辑         | metric/score/val    | `fix!(eval/metric): 重构AUC计算`               |
> | `infra`      | GPU/硬件、显存、并行训练     | gpu/mem/dp          | `perf(infra/mem): 减少显存占用20%`             |
> | `config`     | 超参数、配置文件、实验配置   | param/exp/yaml      | `docs(config/exp): 更新EXP-20配置说明`         |
> | `experiment` | 实验脚本、对比框架、结果记录 | script/result/bench | `test(experiment/bench): 添加基线模型对比测试` |
> | `docs`       | 文档相关，无细分模块         | -                   | `docs: 更新实验复现步骤`                       |
> | `deps`       | 依赖管理，无细分模块         | -                   | `chore(deps): 升级numpy至1.26`                 |
>
>> ###### 自定义范围规则（研究项目适配）
>>
>> 可根据论文/实验需求自定义范围，格式为 `{论文模块}/{实验ID}` 或 `{实验ID}`，禁止过长的范围标注（≤20 字符），示例：
>>
>> - `paper/table2`：修改论文 Table2 对应的实验代码
>> - `exp20`：修改 EXP-20 对应的实验配置
>> - `model/cognitivernn`：修改 CognitiveRNN 模型的核心逻辑
>>

> **脚注规范**
>
> 脚注为 **结构化键值对**，**禁止无规则文本**，阿里标准要求脚注标注 `BREAKING CHANGE`/`Closes`，研究项目新增 `EXP`/`PAPER`/`VALIDATE`，键值对格式为 `KEY: VALUE`，多个值用逗号分隔，**禁止自定义键**，核心脚注键如下：
>
> 1. **BREAKING CHANGE**：标注不兼容变更的细节和适配方案，与 `!` 标记联动，必须填写；
> 2. **EXP**：关联实验 ID，如 `EXP: #15, #18, #20`，带 [🔴 BREAKING]/[🟡 CHANGE] 标记的提交必须填写；
> 3. **PAPER**：关联论文阶段/模块，如 `PAPER: CAMERA-READY, TABLE2`，论文相关提交必须填写；
> 4. **VALIDATE**：标注验证环境/结果，如 `VALIDATE: RTX4050验证通过，速度提升15%`，性能优化/修复类提交建议填写；
> 5. **Closes**：关闭的 issue/待办，如 `Closes: #3, #5`，关联 issue 的提交必须填写；
> 6. **Reverts**：回滚的提交哈希，仅 `revert` 类型使用，如 `Reverts: a1b2c3d`，必须填写。

> **研究日志化**
>
> 结合标准语法，让提交记录成为 **可直接追溯的研究日志**，单人/团队开发均适用，核心要求：
>
> 1. **标题行严格遵循标准语法**：字符 ≤50，动宾结构，无冗余信息，Scope 精准到子模块/实验 ID；
> 2. **正文必写“动机-实现-影响”**：研究项目必须说明变更的科研动机、核心实现方式、对实验/论文的影响，禁止仅写“修复 bug”/“新增功能”；
> 3. **标记与实验强绑定**：带 `[🔴 BREAKING]` 的提交，必须在正文/脚注标注 **需重跑的实验 ID/论文表格**，缺一不可；
> 4. **Perf 细分优化层级**：按 `算法级（model）→ 数据流级（data）→ 硬件级（infra）` 细分 Scope，便于后续性能溯源；
> 5. **Fix 明确问题根源**：修复类提交必须说明 **问题现象、根源、修复方案**，避免模糊描述；
> 6. **无意义提交禁止**：禁止提交“临时保存”/“代码更新”等无意义的提交信息，开发过程中的临时暂存需标注 [WIP]。

## 特殊提交规范

_^tab^_

> **小改动/快速修复**
>
> 无复杂逻辑的小改动，可省略正文/脚注，仅保留标题行，**但必须遵循标准语法**，仅无 Body 时可在 Header 末尾加极简视觉标记：
>
> - `docs: 修正README中的实验命令`
> - `style(model): 修正变量命名错误 [🟢]`
> - `fix(eval): 修复指标计算的笔误 [🟢]`

> **暂存**
>
> 开发过程中的临时暂存提交，**需在标题行末尾加 [WIP]**（Work in Progress），表示未完成，**禁止将 [WIP] 提交推送到远程开发/主分支**，后续需 squash 合并后删除 [WIP] 标记：
>
> - `feat(model): 开发CognitiveRNN模块 [WIP]`
> - `perf(data): 优化数据加载流程 [WIP]`

> **混合提交**
>
> 大厂核心要求为 **“一个提交对应一个核心变更”**（原子性），混合提交（同时包含 feat+fix/perf+fix 等）会导致版本号递增混乱、回滚风险升高，**优先拆分提交**：
>
> 推荐方案：拆分原子提交（大厂最佳实践）
>
> - 先提交 `fix` 类型的修复（仅包含 bug 修复内容）；
> - 再提交 `feat/perf` 类型的新增/优化（仅包含核心变更内容）。

> **回滚提交（revert）**
>
> 回滚之前的提交时，**必须使用 revert 类型**，不可用 fix 替代，标准语法，**禁止省略 Reverts 脚注**：
>
> *==回滚提交规范==*
>
> ```
> revert: <原提交类型>(<原提交范围>): <原提交简短描述>
> // 空行
> 本次回滚的原因，如：该变更导致实验结果异常，暂回滚待排查数据兼容性问题。
> // 空行
> Reverts: <被回滚的提交哈希>
> EXP: #15（若关联实验，补充实验ID，可选）
> PAPER: TABLE2（若关联论文，补充模块，可选）
> ```

> **版本发布提交（大厂标准）**
>
> 发布版本时的专属提交，必须使用 `chore(release)` 类型，**禁止自定义类型**，需在脚注中标注 TAG 和论文阶段：
>
> *==版本发布提交规范==*
>
> ```
> chore(release): 发布v0.1.0 [PAPER-CAMERA-READY]
> // 空行
> 该版本为AAAI-2026投稿最终版（Camera Ready），包含核心功能与性能优化，所有基准实验已验证通过；
> 合入所有feat/fix/perf变更，删除所有[WIP]草稿提交，提交记录已整理。
> // 空行
> TAG: v0.1.0
> PAPER: CAMERA-READY
> VALIDATE: 所有实验已在RTX4050/V100复现，结果已更新至论文
> ```

## 标准提交示例

### 基础提交示例

_^tab^_

> **新增**
>
> *==示例：新增实验功能（无破坏性）==*
>
> ```
> feat(model/attn): 实现动态稀疏注意力机制
> // 空行
> [🟡 CHANGE] 新增动态稀疏注意力模块，支持按注意力权重裁剪低贡献头；
> - 动机：降低模型计算量，适配小显存GPU训练
> - 实现：基于torch.nn.MultiheadAttention扩展，添加稀疏掩码逻辑
> - 影响：单Epoch训练耗时预计减少15%，核心指标无预期变化
> // 空行
> EXP: #25, #26
> VALIDATE: RTX3090验证通过，前向计算结果与基线一致
> ```

> **破坏性新增**
>
> *==示例：新增功能（含破坏性变更）==*
>
> ```
> feat!(eval/metric): 重构F1分数计算逻辑
> // 空行
> [🔴 BREAKING] 重构F1分数计算逻辑，修复宏平均/微平均混淆问题；
> - 动机：原逻辑错误导致多标签分类场景F1值偏高
> - 实现：拆分macro_f1/micro_f1为独立函数，统一阈值处理逻辑
> - 影响：所有多标签实验F1值预计修正0.03~0.05
> // 空行
> BREAKING CHANGE: eval.metric.f1_score() 移除average参数，需显式调用macro_f1/micro_f1
> EXP: #10~20
> PAPER: TABLE3, TABLE5
> Closes: #45
> ```

> **修复**
>
> *==示例：普通 bug 修复（无破坏性）==*
>
> ```
> fix(data/aug): 修复随机裁剪导致的标签偏移
> // 空行
> [🟡 CHANGE] 修复随机裁剪时标签坐标未同步平移的bug；
> - 问题现象：裁剪后目标检测框坐标超出图像范围
> - 根源：仅裁剪图像张量，未更新标签的x/y坐标
> - 修复：同步调整标签坐标，限制坐标在0~1范围内
> // 空行
> EXP: #18
> VALIDATE: V100验证通过，裁剪后标签坐标与图像匹配
> ```

> **破坏性修复**
>
> *==示例：修复（含破坏性变更）==*
>
> ```
> fix!(train/opt): 修复学习率调度器未生效的bug
> // 空行
> [🔴 BREAKING] 修复学习率调度器未绑定优化器的核心bug；
> - 问题现象：训练全程使用初始学习率，无衰减
> - 根源：scheduler.step()调用时机错误，未传入优化器实例
> - 修复：重构train_loop，将scheduler与optimizer强绑定
> // 空行
> BREAKING CHANGE: train.run() 需传入scheduler参数，不再自动创建
> EXP: #5~9
> PAPER: TABLE1
> ```

> **文档变更**
>
> *==示例：纯文档变更==*
>
> ```
> docs(experiment/script): 补充EXP-30的复现步骤
> // 空行
> [🟢 SAFE] 补充EXP-30（跨域适配实验）的完整复现步骤；
> - 新增环境依赖版本明细（torch=2.0.1, transformers=4.30.2）
> - 补充超参数调整说明（lr=1e-4, batch_size=16）
> - 修正原步骤中数据路径错误
> // 空行
> PAPER: APPENDIX-A
> ```

> **性能优化**
>
> *==示例：性能优化==*
>
> ```
> perf(infra/mem): 优化模型显存占用
> // 空行
> [🟡 CHANGE] 优化模型显存占用，降低峰值显存30%；
> - 优化点1：启用梯度检查点（gradient checkpointing）
> - 优化点2：将中间特征张量移至CPU，仅前向时加载
> - 优化点3：使用半精度存储嵌入层权重
> // 空行
> EXP: #22~24
> VALIDATE: RTX4050验证通过，可训练更大batch_size（8→16）
> ```

> **代码重构**
>
> *==示例：代码重构==*
>
> ```
> refactor(model/rnn): 简化RNN模块的初始化逻辑
> // 空行
> [🟢 SAFE] 重构RNN模块初始化逻辑，无功能变更；
> - 合并重复的参数校验代码
> - 将硬编码的默认值抽离为类常量
> - 简化forward函数的分支判断逻辑
> // 空行
> EXP: #15
> ```

> **测试变更**
>
> *==示例：测试代码变更==*
>
> ```
> test(experiment/bench): 添加模型推理速度基准测试
> // 空行
> [🟢 SAFE] 新增模型推理速度基准测试脚本；
> - 测试覆盖：model/attn、model/rnn、model/cnn三类模块
> - 测试指标：单样本推理耗时、吞吐量（samples/s）
> - 输出格式：CSV文件，便于论文图表生成
> // 空行
> EXP: #30
> PAPER: FIGURE4
> ```

> **构建/依赖变更**
>
> *==示例：构建/依赖变更==*
>
> ```
> chore(deps): 升级transformers至4.36.2
> // 空行
> [🟢 SAFE] 升级transformers库至4.36.2版本；
> - 动机：修复低版本中BERTTokenizer的编码bug
> - 验证：所有现有脚本运行正常，无API兼容问题
> // 空行
> VALIDATE: 本地环境+CI均验证通过
> ```

> **格式调整**
>
> *==示例：代码格式调整==*
>
> ```
> style(eval/metric): 统一指标计算代码的命名规范
> // 空行
> [🟢 SAFE] 仅调整代码格式，无逻辑变更；
> - 将驼峰命名（f1Score）改为下划线命名（f1_score）
> - 统一缩进为4个空格，删除多余空行
> - 修正注释格式，符合阿里Java编码规范（Python适配）
> // 空行
> EXP: #12
> ```

### 研究场景示例

_^tab^_

> **实验脚本临时暂存（WIP）**
>
> *==示例：实验脚本临时暂存（WIP）==*
>
> ```
> feat(experiment/exp35): 开发跨模态对比实验脚本 [WIP]
> // 空行
> [🟡 CHANGE] 初步完成跨模态对比实验脚本框架；
> - 已实现数据加载和模型初始化
> - 待完成：指标计算、结果保存逻辑
> - 待验证：多模态特征融合的维度匹配
> // 空行
> EXP: #35
> ```

> **论文表格关联变更**
>
> *==示例：论文表格关联变更==*
>
> ```
> fix(paper/table4): 修正TABLE4对应的实验数据计算
> // 空行
> [🔴 BREAKING] 修正TABLE4中模型准确率的计算错误；
> - 原错误：未排除无效样本（标签为-1的样本）
> - 修复：在指标计算前过滤无效样本
> - 影响：TABLE4中3个模型的准确率修正0.01~0.02
> // 空行
> EXP: #19, #21
> PAPER: TABLE4
> VALIDATE: 重新计算后数据已同步至论文文档
> ```

> **极简提交（无 Body/Footer）**
>
> *==示例：极简提交（无 Body/Footer）==*
>
> ```
> docs: 修正README中的实验环境说明 [🟢]
> ```

> **强绑定混合提交（feat+fix）**
>
> *==示例：强绑定混合提交（feat+fix）==*
>
> ```
> feat(data/process): 实现缺失值插补功能，修复空值判断bug
> // 空行
> [🟡 CHANGE] 核心变更：
> 1. 新增缺失值插补功能（feat）：支持均值/中位数/模式插补，适配时序数据
> 2. 修复空值判断bug（fix）：原逻辑将0值误判为空值，已调整判断条件
> // 空行
> EXP: #28
> VALIDATE: RTX4050验证通过，插补结果符合预期，空值判断准确
> ```

### 特殊类型示例

_^tab^_

> **回滚提交**
>
> *==示例：回滚提交==*
>
> ```
> revert: fix(data/aug): 修复随机裁剪导致的标签偏移
> // 空行
> 本次回滚原因：该修复导致小尺寸图像裁剪后标签丢失，需重新设计坐标调整逻辑；
> 暂回滚至修复前版本，待优化后重新提交。
> // 空行
> Reverts: a8b9c7d
> EXP: #18
> PAPER: TABLE3
> ```

> **分支合并**
>
> *==示例：分支合并==*
>
> ```
> merge: feat-exp35-cross-modal into develop
> // 空行
> [🟡 CHANGE] 合并跨模态对比实验分支至开发分支；
> - 合入内容：EXP-35的完整实验脚本、数据处理逻辑
> - 验证状态：分支内所有测试用例已通过，可正常运行
> // 空行
> EXP: #35
> VALIDATE: 合并后develop分支可正常训练/评估
> ```

> **混合提交拆分**
>
> 同时包含 feat+fix/perf+fix 的混合提交强烈建议拆分！
>
> *==混合提交拆分示例-1==*
>
> ```
> # 第一步：仅修复bug
> fix(model/cognitivernn): 修复初始化参数错误
> // 空行
> [🟢 SAFE] 修复CognitiveRNN模块初始化时hidden_dim参数默认值错误，避免维度不匹配报错；
> 该修复仅针对未发布的模块，无实验结果影响。
> // 空行
> EXP: #20
> ```
>
> *==混合提交拆分示例-2==*
>
> ```
> # 第二步：仅新增功能
> feat(model/cognitivernn): 实现认知记忆模块
> // 空行
> [🟡 CHANGE] 新增CognitiveRNN认知记忆模块，支持艾宾浩斯遗忘曲线逻辑；
> 已基于参数修复后的版本开发，功能正常。
> // 空行
> EXP: #20
> VALIDATE: RTX4050验证通过，模块前向计算结果符合预期
> ```

> **混合提交（强绑定）**
>
> 仅当“新增功能与修复该功能的即时小 bug 强绑定”（无其他影响）时，可混合提交，**以核心变更定类型，正文明确区分变更类型**：
>
> **示例（feat 为主，附带修复）**：
>
> *==混合提交（强绑定变更）规范==*
>
> ```
> feat(model/cognitivernn): 实现认知记忆模块，修复初始化参数错误
> // 空行
> [🟡 CHANGE] 核心变更：
> 1. 新增CognitiveRNN认知记忆模块，支持艾宾浩斯遗忘曲线逻辑（feat）；
> 2. 修复模块初始化时hidden_dim参数默认值错误的bug（fix），避免维度不匹配报错。
> // 空行
> EXP: #20
> VALIDATE: RTX4050验证通过，模块功能正常，参数错误已修复
> ```
>
> **示例（fix 为主，附带新增）**：
>
> *==混合提交（强绑定变更）示例==*
>
> ```
> fix(data/load): 修复NaN填充逻辑，新增缺失值统计功能
> // 空行
> [🔴 BREAKING] 核心变更：
> 1. 修复forward-fill填充导致的数据泄露bug（fix），替换为zero-fill逻辑；
> 2. 新增缺失值统计子功能（feat），输出各字段缺失值占比，便于实验分析。
> // 空行
> BREAKING CHANGE: data.load() 移除forward_fill参数，替换为fill_type；
> EXP: #5~14
> PAPER: TABLE2, TABLE4
> ```

# 版本号控制

## 语义化版本控制（SemVer）

遵循 `主版本号.次版本号.修订号`（`MAJOR.MINOR.PATCH`）标准，版本号的递增与 Git 提交类型 **硬绑定**，为自动化工具落地提供基础：

- **主版本号 `MAJOR`**：**不兼容的 API 变更** 或 **研究项目基准重置**（如论文终版迭代）。
  - 工程含义：用户代码需修改才能运行（如 `v1.x` -> `v2.0`）；
  - 提交联动：包含至少 1 个带 `!` 标记的破坏性提交（feat!/fix! 等）。
- **次版本号 `MINOR`**：**向下兼容的新功能/新实验模块/新算法逻辑**。
  - 工程含义：新增功能但旧代码仍可运行（如 `v1.1` -> `v1.2`）；
  - 提交联动：包含至少 1 个 feat 类型提交（无 `!` 标记），且无 MAJOR 级变更。
- **修订号 `PATCH`**：**向下兼容的问题修复/性能优化**。
  - 工程含义：修复 Bug/优化性能，不改变 API（如 `v1.1.0` -> `v1.1.1`）；
  - 提交联动：包含至少 1 个 fix/perf 类型提交（无 `!` 标记），且无 MAJOR/MINOR 级变更。

## 预发布版本规范（大厂必备）

针对研究项目的 **实验测试、论文初稿、版本候选** 阶段，补充 **预发布版本号** 规则（遵循 SemVer 大厂适配标准），用于正式发布前的版本迭代，格式为：
`MAJOR.MINOR.PATCH-[预发布标识].[迭代号]`

- **预发布标识**：按阶段递进，不可跨级使用
  - `alpha`：内部实验版，仅开发/研究团队使用，功能未定型；
  - `beta`：公开测试版，功能基本完成，可用于合作方复现实验；
  - `rc`（Release Candidate）：发布候选版，与正式版一致，仅修复致命 bug，可作为论文最终实验版本。
- **迭代号**：从 1 开始的正整数，每次更新递增（如 `0.1.0-alpha.1` → `0.1.0-alpha.2`）；
- **版本联动**：预发布版本迭代仅递增迭代号，不改变 `MAJOR.MINOR.PATCH`，转正为正式版本时按语义化规则递增主/次/修订号；
- **研究项目适配**：预发布版本可关联论文阶段（如 `0.1.0-rc.1[PAPER-FINAL]`）。

## 版本递增规则（与 Git 提交强联动）

版本号的递增由该版本包含的 **所有 Git 提交类型** 共同决定，遵循大厂自动化工具（standard-version/release-it）核心逻辑，**禁止人工随意递增版本号**：

1. 无版本递增场景：仅包含 docs/refactor/style/test/chore 类型提交，版本号不变（可添加内部迭代号，如 `v0.1.0+1`，仅用于内部追溯）；
2. 预发布版本转正：基于预发布版本的最终提交，按上述 MAJOR/MINOR/PATCH 规则递增正式版本号，移除预发布标识；
3. 破坏性变更强制规则：只要包含 1 个带 `!` 标记的提交，无论是否有其他提交类型，均强制递增主版本号；
4. 科研基准重置：论文终版/实验基准重构等场景，可主动重置主版本号（如 `v0.9.0` → `v1.0.0`），需在版本备注中明确说明。

## 版本打标与发布流程（阿里标准）

版本号需与 Git Tag **强绑定**，无 Tag 不视为正式版本，打标与发布遵循以下规范，确保溯源可查：

1. **Tag 命名规则**：统一为 `v{版本号}`，小写 v+完整版本号（如 `v0.1.0`、`v0.1.0-rc.1`），禁止自定义命名（如 `v0.1实验版`）；
2. **打标时机**：完成版本开发、实验验证、代码合入主分支后，**通过自动化工具** 打标并推送到远程仓库，禁止人工打标；
3. **发布备注**：打 Tag 时必须添加备注，备注格式与 `CHANGELOG.md` 版本标题一致（如 `git tag -a v0.1.0 -m "[0.1.0] - 2026-01-23 🎯 PAPER-CAMERA-READY"`）；
4. **版本归档**：正式版本（无预发布标识）打标后，在仓库 Releases 中归档，附带 `CHANGELOG.md` 对应版本内容、实验环境、核心结果、关键提交哈希；
5. **预发布版本归档**：rc 版本需在 Releases 中归档，alpha/beta 版本可仅打 Tag，无需归档。

# 维护变更日志

对于研究项目，`CHANGELOG.md` 与版本号 **同等重要**，遵循 **阿里 CHANGELOG 维护标准**，**按版本倒序排列**（最新版本在最上方），且 **每个版本标题需添加 Git Tag 链接**。CHANGELOG 内容需与 Git 提交记录 **1:1 对应**，禁止无提交记录的“空变更”，同时实现科研实验的全链路溯源。

## 核心格式要求

1. **术语统一**：分类名称与 Conventional Commits 提交类型一一对应，禁止自定义分类；
2. **溯源要求**：每个变更项必须标注对应 Git 提交短哈希，格式为 `内容描述 (#提交短哈希)`；
3. **标记一致**：结果影响标记（[🔴 BREAKING]/[🟡 CHANGE]/[🟢 SAFE]）与对应 Git 提交的 Body 标记完全一致；
4. **科研绑定**：实验 ID/论文模块在变更项和脚注中保持一致，与 Git 提交脚注的 EXP/PAPER 键值完全匹配；
5. **格式规范**：每个版本独立成段，分类下的变更项用列表展示，每行 ≤80 字符，便于阅读和自动化生成。

> ###### 版本标题语法
>
> ```
> [MAJOR.MINOR.PATCH[-预发布标识.迭代号]] - YYYY-MM-DD  → [Git Tag 链接]
> ```
>
>> Git Tag 链接格式：`[v{版本号}](仓库地址/tags/v{版本号})`，必须指向仓库实际 Tag 地址；
>>

> ###### 模块
>
> - **Features**：新增的功能/实验模块/算法逻辑，对应 `feat` 类型提交；
> - **Fixes**：修复的代码 bug/逻辑错误，对应 `fix` 类型提交；
> - **Performance Improvements**：性能优化（速度/显存/算力），对应 `perf` 类型提交；
> - **Changes**：代码重构/运行行为调整，对应 `refactor` 类型提交；
> - **Style Changes**：代码格式调整，无逻辑变更，对应 `style` 类型提交；
> - **Documentation**：仅文档/注释变更，对应 `docs` 类型提交；
> - **Breaking Changes**：不兼容的变更（若有，必须单独列项并标注 `!`，说明适配方案，引用对应提交哈希）；
> - **Experiments**：该版本关联的实验 ID、验证结果、需重跑的论文表格；
> - **Validation**：实验验证环境（硬件、框架版本、关键依赖），确保实验可复现。

> ###### 结果影响标记
>
> 在变更项后紧跟标准化标记，便于快速识别代码更改对实验结果的影响，**与 Git 提交 Body 中的标记完全一致**：
>
> - `[🔴 BREAKING]`：修改会改变基准实验结果（如 Loss 计算修复、数据泄露修复）。需重跑基准实验，**强制关联实验 ID/论文表格**；
> - `[🟡 CHANGE]`：修改了运行行为（如 GPU 显存优化、前向速度提升），预期结果不变但需验证，建议复现核心实验；
> - `[🟢 SAFE]`：纯工程重构、文档、注释、代码格式，不影响数学逻辑/实验结果，无需复现实验。

## CHANGELOG 完整示例

_^tab^_

> **示例 1**
>
> *==示例 1：预发布版本（alpha/beta/rc）==*
>
> ```
> **[0.1.0] - 2026-01-23 🎯 PAPER-CAMERA-READY** → [v0.1.0](https://github.com/xxx/xxx/tags/v0.1.0)
>
> 1. Features
>    - 新增`CognitiveRNN`认知记忆模块，支持艾宾浩斯遗忘曲线逻辑 [EXP-20] (#a1b2c3d) [🟡 CHANGE]
>    - 支持混合精度训练 (AMP)，兼容 PyTorch 2.1+ (#b3c4d5e) [🟡 CHANGE]
> 2. Performance Improvements
>    - 优化数据加载流程，单次PCIe传输替代多次传输，Epoch耗时减少20% (#c4d5e6f) [🟡 CHANGE]
>    - 缓存多跳邻居图+矢量化硬回顾逻辑，前向计算加速15% [EXP-15] (#d5e6f7g) [🟡 CHANGE]
> 3. Fixes
>    - 修复NaN填充逻辑（从forward-fill改为zero-fill），消除数据泄露，需重跑Table 2/4 [EXP-5~14] (#e6f7g8h) [🔴 BREAKING]
>    - 修复类型转换警告，优化代码注释 (#f7g8h9i) [🟢 SAFE]
> 4. Breaking Changes
>    - `data.load()` 方法移除`forward_fill`参数，替换为`fill_type="zero"` (#e6f7g8h)
>    - 适配方案：修改所有调用`data.load()`的实验脚本，将`forward_fill=True/False`替换为`fill_type="zero"/"forward"`
> 5. Experiments
>    - EXP #15–18：性能优化验证通过，结果符合预期（#c4d5e6f、#d5e6f7g）
>    - EXP #5–14：数据修复后已重跑，新结果已更新至论文（#e6f7g8h）
> 6. Validation
>    - 硬件：RTX 4050 / V100；框架：PyTorch 2.1.0、CUDA 12.1
>    - 依赖：numpy 1.26.0、pandas 2.1.4
>    - Table 2/4 基准实验需基于此版本重跑 (因数据修复)
> ```

> **示例 2**
>
> *==示例 2：仅修复的 PATCH 版本==*
>
> ```
> **[0.1.1] - 2026-01-25 🎯 PAPER-CAMERA-READY** → [v0.1.1](https://github.com/xxx/xxx/tags/v0.1.1)
>
> 1. Fixes
>    - 修复CognitiveRNN模块hidden_dim参数默认值错误 [EXP-20] (#g8h9i0j) [🟢 SAFE]
>    - 修复评估脚本中AUC计算精度丢失问题 [EXP-18] (#h9i0j1k) [🔴 BREAKING]
> 2. Breaking Changes
>    - `eval/auc.py` 中`calc_auc()`函数返回值从float32改为float64 (#h9i0j1k)
>    - 适配方案：实验脚本中无需修改调用方式，仅需重新计算AUC结果
> 3. Experiments
>    - EXP #20：参数修复后模块运行正常，结果无变化（对应提交#g8h9i0j）
>    - EXP #18：AUC精度修复后，Table 3结果更新（对应提交#h9i0j1k）
> 4. Validation
>    - 硬件：RTX 4050；框架：PyTorch 2.1.0、CUDA 12.1
>    - 依赖：numpy 1.26.0、scikit-learn 1.3.2
>    - 备注：Table 3 需重跑，其余表格无需调整
> ```

> **示例 3**
>
> *==示例 3：纯文档/格式变更的版本（无版本号递增）==*
>
> ```
> **[0.1.1+1] - 2026-01-26 🎯 PAPER-CAMERA-READY** → [v0.1.1+1](https://github.com/xxx/xxx/tags/v0.1.1+1)
>
> 1. Documentation
>    - 更新README中EXP-20的超参数调优说明 (#i0j1k2l) [🟢 SAFE]
>    - 补充CognitiveRNN模块的API注释 (#j1k2l3m) [🟢 SAFE]
> 2. Style Changes
>    - 统一训练脚本缩进为4个空格 (#k2l3m4n) [🟢 SAFE]
> 3. Experiments
>    - 无实验变更，所有实验结果保持不变
> 4. Validation
>    - 硬件/框架/依赖与v0.1.1一致，无需重新验证
> ```

> **示例 4**
>
> *==示例 4：包含 Breaking Changes 的 MAJOR 版本==*
>
> ```
> **[2.0.0] - 2026-02-01 🎯 PAPER-FINAL** → [v2.0.0](https://github.com/xxx/xxx/tags/v2.0.0)
>
> 1. Features
>    - 重构数据加载核心接口，支持多模态数据输入 [EXP-25] (#l3m4n5o) [🔴 BREAKING]
>    - 新增多任务训练框架，兼容分类/回归任务 [EXP-26] (#m4n5o6p) [🟡 CHANGE]
> 2. Performance Improvements
>    - 优化多模态数据拼接逻辑，显存占用减少30% [EXP-25] (#n5o6p7q) [🟡 CHANGE]
> 3. Breaking Changes
>    - `data/loader.py` 中`DataLoader`类初始化参数从`data_path`改为`dataset_config` (#l3m4n5o)
>    - 适配方案：将`DataLoader(data_path="./data")`改为`DataLoader(dataset_config="./config/dataset.yaml")`
>    - `train/trainer.py` 中`train()`方法移除`lr`参数，改为从配置文件读取 (#o6p7q8r)
>    - 适配方案：在训练配置文件中添加`learning_rate`字段，删除脚本中`lr`传参
> 4. Experiments
>    - EXP #25–26：多模态模块实验验证通过，Table 5/6结果已更新（对应提交#l3m4n5o、#m4n5o6p）
>    - EXP #1–24：基准实验已基于新接口重跑，结果无显著变化（对应提交#p7q8r9s）
> 5. Validation
>    - 硬件：A100 / V100；框架：PyTorch 2.2.0、CUDA 12.2
>    - 依赖：numpy 1.26.2、pandas 2.2.0、transformers 4.36.2
>    - 备注：所有论文表格已基于v2.0.0重跑，为最终提交版本
> ```

# 配套分支管理规范（阿里标准，版本/提交的基础）

版本和提交规范需与 **分支管理** 深度联动，阿里等大厂均有固定的分支模型，本规范适配研究项目的 **实验开发、论文迭代、版本发布**，核心分支分为 **永久分支** 和 **临时分支**，**禁止自定义永久分支**，临时分支命名遵循标准化规则，与提交类型/实验 ID 强绑定。

## 永久分支（仓库核心分支，不可删除）

永久分支为仓库的核心分支，仅存放可验证、可发布的代码，**禁止在永久分支上直接开发**，所有变更需通过 PR/MR 合入：

| 分支名      | 核心用途                                                                 | 提交规范要求                               | 版本要求                       |
| :---------- | :----------------------------------------------------------------------- | :----------------------------------------- | ------------------------------ |
| main/master | 正式版本分支，仅存放已打 Tag 的正式版本代码，可直接用于论文实验/成果展示 | 仅允许 merge/revert/chore(release)类型提交 | 仅存放正式版本（无预发布标识） |
| develop     | 开发分支，存放最新的开发/实验代码，团队协作的核心分支                    | 遵循所有标准提交规范，禁止 [WIP] 提交      | 可存放预发布版本/开发版本      |

## 临时分支（开发完成后删除，标准化命名）

临时分支用于功能开发、bug 修复、实验验证等，开发完成后需及时合入对应永久分支并 **删除**，命名格式为：`{类型}-{模块/实验ID}-{简短描述}`，全部小写，连接符用 `-`，**禁止过长的分支名**（≤30 字符），类型与 Git 提交类型强绑定：

| 分支类型    | 对应 Git 提交类型 | 命名示例                | 核心用途                     | 合入目标分支 |
| :---------- | :---------------- | :---------------------- | :--------------------------- | :----------- |
| feat-xxx    | feat/feat!        | feat-model-cognitivernn | 新增功能/实验模块/算法逻辑   | develop      |
| fix-xxx     | fix/fix!          | fix-data-nanfill        | 修复 bug/逻辑错误            | develop/main |
| perf-xxx    | perf              | perf-train-amp          | 性能优化（速度/显存/算力）   | develop      |
| exp-xxx     | 所有类型          | exp-20-amp-test         | 实验验证，研究项目专属       | develop      |
| release-xxx | chore(release)    | release-v0.1.0          | 版本发布，整理提交记录       | develop/main |
| hotfix-xxx  | fix!/fix          | hotfix-eval-auc         | 修复 main 分支的正式版本 bug | main/develop |
| style-xxx   | style             | style-model-naming      | 代码格式统一                 | develop      |
| docs-xxx    | docs              | docs-readme-experiment  | 文档/注释更新                | develop      |

## 分支-提交-版本 联动规则（大厂核心）

实现 **分支类型 → 提交类型 → 版本号** 的全链路联动，禁止跨类型使用，确保自动化工具可识别、可落地：

1. **feat-xxx 分支**：仅允许 feat/feat! 类型提交，合入 develop 后标记 **待 MINOR/MAJOR 版本递增**，禁止在该分支提交其他类型记录；
2. **fix-xxx 分支**：仅允许 fix/fix! 类型提交，合入 develop 后标记 **待 PATCH/MAJOR 版本递增**；hotfix-xxx 分支合入 main 后，立即打 PATCH/MAJOR 版本 Tag，并同步合入 develop；
3. **perf-xxx 分支**：仅允许 perf 类型提交，合入 develop 后标记 **待 PATCH 版本递增**；
4. **exp-xxx 分支**：允许所有类型提交，但标题需标注 [EXP-XX]，合入 develop 前需 squash 为标准提交，删除 [WIP] 标记，根据提交类型标记对应版本递增；
5. **release-xxx 分支**：仅允许 chore(release)/revert 提交，禁止新增 feat/fix/perf 等功能型提交，合入 main 后立即打正式版本 Tag，并同步更新 CHANGELOG；
6. **版本递增触发**：当 develop 分支中积累的标记“待递增”的提交达到发布条件时，创建 release-xxx 分支，执行版本发布流程，自动递增版本号、生成 CHANGELOG、打 Tag。

# 自动化工具适配（大厂落地方案）

阿里/字节等大厂均通过 **自动化工具** 实现规范落地，避免人工错误，本规范提供大厂标配的工具配置方案，实现 **提交校验 → 版本递增 →CHANGELOG 生成 →Tag 打标** 的全流程自动化，**禁止人工执行上述操作**。

## 核心工具（大厂标配）

所有工具均为开源工具，适配 Git 仓库，支持本地/CI/CD 校验，核心工具如下：

- **commitlint**：强制校验提交信息是否符合 Conventional Commits 标准语法，拒绝不符合规范的提交；
- **husky**：Git 钩子工具，配合 commitlint 实现 **提交前自动校验**，确保本地提交符合规范；
- **standard-version**：自动生成 CHANGELOG、递增版本号、打 Git Tag，完全遵循 SemVer 和 Conventional Commits 规范；
- **release-it**：进阶版本发布工具，支持预发布版本（alpha/beta/rc）、自定义标记，更适配研究项目的论文/实验关联需求。

## 基础配置（快速落地，package.json 示例）

适用于 Node.js 环境，非 Node.js 环境可通过 Docker/Shell 脚本实现，以下为 **大厂标准配置**，可直接复制使用：

```json
{
  "name": "your-project-name",
  "version": "0.1.0",
  "scripts": {
    "prepare": "husky install", // 初始化husky钩子
    "commitlint": "commitlint --edit", // 校验提交信息
    "release": "standard-version", // 发布正式版本，自动递增版本号、生成CHANGELOG
    "release:alpha": "standard-version --prerelease alpha", // 发布alpha预发布版本
    "release:beta": "standard-version --prerelease beta", // 发布beta预发布版本
    "release:rc": "standard-version --prerelease rc" // 发布rc预发布版本
  },
  "devDependencies": {
    "@commitlint/cli": "^18.6.0",
    "@commitlint/config-conventional": "^18.6.0",
    "husky": "^8.0.0",
    "standard-version": "^9.5.0"
  },
  "commitlint": {
    "extends": ["@commitlint/config-conventional"] // 遵循Conventional Commits官方规范
  }
}
```

## 钩子配置（husky，提交前自动校验）

通过 husky 添加 **commit-msg** 钩子，实现提交信息的自动校验，拒绝不符合规范的提交：

```bash
# 初始化husky
npm run prepare
# 添加commit-msg钩子
npx husky add .husky/commit-msg 'npx --no -- commitlint --edit "$1"'
```

## CI/CD 校验（团队协作必备）

在 GitLab CI/GitHub Actions 中加入提交信息校验和版本发布校验，**拒绝不符合规范的 PR/MR**，示例（GitHub Actions，.github/workflows/commitlint.yml）：

```yaml
name: CommitLint
on: [pull_request, push]
jobs:
  commitlint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
      - run: npm install
      - run: npx commitlint --from=HEAD~1 --to=HEAD --verbose
```

## 科研项目扩展配置

针对研究项目的实验/论文关联需求，可扩展 commitlint 规则，强制校验 EXP/PAPER 等脚注键，示例（commitlint.config.js）：

```javascript
module.exports = {
  extends: ['@commitlint/config-conventional'],
  rules: {
    // 强制带[🔴 BREAKING]标记的提交必须包含EXP脚注
    'footer-match': [2, 'always', [
      {
        pattern: /\[🔴 BREAKING\]/,
        footerPattern: /EXP: #\d+(-#\d+)?/
      }
    ]]
  }
};
```

# 规范落地与团队协作要求

1. **全员培训**：团队所有成员必须熟悉本规范，掌握提交语法、分支命名、版本递增规则，避免人工错误；
2. **本地校验**：所有开发人员必须在本地配置 husky+commitlint，实现提交前自动校验，禁止绕过校验提交；
3. **CI/CD 强制**：在 CI/CD 中加入提交校验、分支命名校验，拒绝不符合规范的 PR/MR，确保远程仓库的所有记录符合标准；
4. **自动化优先**：版本发布、CHANGELOG 生成、Tag 打标必须通过自动化工具执行，禁止人工操作；
5. **科研溯源**：所有与实验/论文相关的提交，必须关联实验 ID/论文模块，确保“提交 → 版本 → 实验 → 论文”的全链路可追溯；
6. **规范迭代**：本规范可根据项目需求迭代，但需团队全员共识，迭代后需更新文档并重新培训，禁止个人随意修改规范。

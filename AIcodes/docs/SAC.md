# SAC.py 分层讲解

## 快速理解
- critic 使用 SAC 的熵正则 Bellman target。
- actor 不用普通在线 SAC 更新，而是 AWAC 风格地提高数据动作概率。
- 适合离线场景，因为 actor 被数据动作约束住。

一句话记法：**SAC 离线 AWAC 风格版本** 的重点是：SAC 保留随机策略和熵温度；AWAC 更新让策略更像高优势的数据动作。这个组合比纯在线 SAC 更适合只拿离线数据训练。

## 先看整体流程
1. 读取完整转移并划分训练/验证集。
2. 建立 GaussianActor、两个 critic、两个 target critic。
3. critic target 包含 `-alpha * log_prob` 熵项。
4. 计算数据动作 Q 和策略动作 Q 的 advantage。
5. 把 advantage 转成权重。
6. actor 最大化加权后的数据动作 log probability。
7. 可选自动学习 alpha。
8. 保存 history、曲线和最佳 actor。

## 核心原理
SAC 保留随机策略和熵温度；AWAC 更新让策略更像高优势的数据动作。这个组合比纯在线 SAC 更适合只拿离线数据训练。

核心公式可以先记成：

```text
target_q = r + gamma * (min(Q1,Q2) - alpha * log_pi)
actor_loss = -mean(weight * log_pi(a_data | s))
```

读代码时不用一开始就追每个张量的形状，先抓住三件事：数据从哪里来、loss 怎么算、什么时候保存模型。

## 关键代码精讲
### 关键函数/类导航
| 类型 | 名称 | 源码行 | 作用 | 输入/输出 | 所在流程 |
| --- | --- | --- | --- | --- | --- |
| 函数 | `ensure_dirs` | 21 | 创建输出目录。 | 输入：save_dir；输出：中间结果或无返回值。 | 配置准备 |
| 函数 | `set_seed` | 26 | 固定随机种子，提升实验可复现性。 | 输入：seed；输出：中间结果或无返回值。 | 配置准备 |
| 函数 | `soft_update` | 34 | 软更新 target 网络。 | 输入：source_net, target_net, tau；输出：中间结果或无返回值。 | 训练更新 |
| 函数 | `gaussian_log_prob` | 39 | 辅助函数，服务于当前算法流程。 | 输入：noise, log_std；输出：中间结果或无返回值。 | 辅助逻辑 |
| 函数 | `parse_hidden_sizes` | 43 | 把命令行里的隐藏层字符串转成整数列表。 | 输入：value；输出：隐藏层整数列表。 | 配置准备 |
| 函数 | `is_finite_tensor` | 56 | 辅助函数，服务于当前算法流程。 | 输入：x；输出：中间结果或无返回值。 | 辅助逻辑 |
| 函数 | `sanitize_tensor` | 60 | 辅助函数，服务于当前算法流程。 | 输入：x, clamp_min, clamp_max；输出：中间结果或无返回值。 | 辅助逻辑 |
| 类 | `Config` | 70 | 集中保存训练参数和输出文件名。 | 输入：构造参数；输出：可复用对象。 | 配置准备 |
| 函数 | `build_config_from_args` | 106 | 解析命令行参数，创建配置对象，并保存 train_config.json。 | 输入：无显式输入；输出：配置对象 cfg。 | 配置准备 |
| 类 | `MLP` | 208 | 通用多层感知机，供 actor 或 critic 复用。 | 输入：构造参数；输出：可复用对象。 | 网络定义 |
| 方法 | `MLP.__init__` | 209 | 初始化网络层、优化器或训练状态。 | 输入：input_dim, output_dim, hidden_sizes；输出：中间结果或无返回值。 | 网络定义 |
| 方法 | `MLP.forward` | 220 | 定义模型前向计算。 | 输入：x；输出：网络输出张量。 | 网络定义 |
| 类 | `GaussianActor` | 224 | SAC 随机策略网络，输出动作分布并采样动作。 | 输入：构造参数；输出：可复用对象。 | 网络定义 |
| 方法 | `GaussianActor.__init__` | 225 | 初始化网络层、优化器或训练状态。 | 输入：state_dim, action_dim, hidden_sizes, log_std_min, log_std_max；输出：中间结果或无返回值。 | 网络定义 |
| 方法 | `GaussianActor.forward` | 243 | 定义模型前向计算。 | 输入：state；输出：网络输出张量。 | 网络定义 |
| 方法 | `GaussianActor.sample` | 250 | 从 Gaussian actor 中采样动作并计算 log probability。 | 输入：state；输出：网络输出张量。 | 网络定义 |
| 方法 | `GaussianActor.log_prob_of_action` | 267 | 计算给定数据动作在当前策略下的 log probability。 | 输入：state, action；输出：网络输出张量。 | 网络定义 |
| 类 | `Critic` | 284 | Q 网络，输入状态动作对输出 Q 值。 | 输入：构造参数；输出：可复用对象。 | 网络定义 |
| 方法 | `Critic.__init__` | 285 | 初始化网络层、优化器或训练状态。 | 输入：state_dim, action_dim, hidden_sizes；输出：中间结果或无返回值。 | 网络定义 |
| 方法 | `Critic.forward` | 289 | 定义模型前向计算。 | 输入：state, action；输出：网络输出张量。 | 网络定义 |
| 函数 | `load_hdf5_dataset` | 297 | 读取 HDF5 完整转移数据。 | 输入：path；输出：数据集、DataLoader、维度或模型对象。 | 数据处理 |
| 函数 | `split_dataset` | 313 | 按比例划分训练集和验证集。 | 输入：dataset, val_ratio, seed；输出：中间结果或无返回值。 | 数据处理 |
| 函数 | `compute_obs_stats` | 326 | 计算状态标准化用的均值和标准差。 | 输入：train_dataset；输出：obs_mean 和 obs_std。 | 数据处理 |
| 类 | `OfflineAWACSACAgent` | 340 | 封装 SAC+AWAC 的网络、优化器和训练步骤。 | 输入：构造参数；输出：可复用对象。 | 辅助逻辑 |
| 方法 | `OfflineAWACSACAgent.__init__` | 341 | 初始化网络层、优化器或训练状态。 | 输入：cfg, state_dim, action_dim；输出：中间结果或无返回值。 | 辅助逻辑 |
| 方法 | `OfflineAWACSACAgent.alpha` | 395 | 返回当前熵温度 alpha。 | 输入：无显式输入；输出：中间结果或无返回值。 | 训练更新 |
| 方法 | `OfflineAWACSACAgent.set_schedulers` | 398 | 保存学习率调度器引用。 | 输入：actor_scheduler, critic1_scheduler, critic2_scheduler；输出：中间结果或无返回值。 | 训练更新 |
| 方法 | `OfflineAWACSACAgent._compute_awac_weights` | 403 | 根据 advantage 计算 AWAC 权重。 | 输入：data_q, v_pi；输出：中间结果或无返回值。 | 训练更新 |
| 方法 | `OfflineAWACSACAgent.train_step` | 415 | 执行一个 batch 的训练更新。 | 输入：batch, obs_mean, obs_std, reward_mean, reward_std；输出：训练指标、日志或最佳模型文件。 | 训练更新 |
| 方法 | `OfflineAWACSACAgent.eval_step` | 523 | 计算一个 batch 的验证指标。 | 输入：batch, obs_mean, obs_std, reward_mean, reward_std；输出：评估指标或 loss。 | 验证评估 |
| 方法 | `OfflineAWACSACAgent.save_best` | 582 | 保存当前最佳模型。 | 输入：obs_mean, obs_std, reward_mean, reward_std, epoch, best_val_loss；输出：写入文件，通常无主要返回值。 | 保存/推理 |
| 函数 | `train_sac` | 608 | SAC 类算法主训练流程。 | 输入：agent, train_loader, val_loader, obs_mean, obs_std, reward_mean, reward_std, cfg；输出：训练指标、日志或最佳模型文件。 | 训练更新 |
| 函数 | `save_history_csv` | 742 | 保存 SAC 训练历史和曲线图。 | 输入：history, cfg；输出：写入文件，通常无主要返回值。 | 保存/推理 |
| 函数 | `load_best_actor` | 776 | 加载最佳 actor 用于推理。 | 输入：cfg；输出：数据集、DataLoader、维度或模型对象。 | 保存/推理 |
| 函数 | `main` | 801 | 脚本入口，串起配置、数据、训练和加载。 | 输入：无显式输入；输出：无返回值，负责启动流程。 | 脚本入口 |

### 关键训练逻辑怎么读
- **数据入口**：先找 `load_dataset`、`load_hdf5_dataset` 或 `OfflineDataset`。这些地方决定 batch 里到底有哪些张量。
- **网络定义**：再看 `BCPolicy`、`Actor`、`GaussianActor`、`Critic` 或 `MLP`。这些类决定状态如何变成动作，状态动作如何变成 Q 值。
- **损失计算**：重点看 `train_one_epoch`、`train_step`、`train_cql_improved` 或 `train_sac`。这里会出现 actor loss、critic loss、BC loss、CQL penalty 等核心逻辑。
- **保存与评估**：最后看 `evaluate_*`、`save_best*`、`save_*csv`。这些函数不改变算法本质，但决定你在网页或文件里看到哪些结果。

### 建议抓住的代码片段
```text
数据加载 -> 状态标准化 -> 网络前向 -> 计算 loss -> backward/step -> 验证 -> 保存 best_model.pt
```

如果你读到某一行不确定含义，不要在主文档里硬翻大表格，直接跳到逐行附录按行号查。

## 参数怎么调
### 数据与输出
| 参数 | 命令行/来源 | 源码默认值 | 当前训练记录 | 怎么理解 |
| --- | --- | --- | --- | --- |
| `best_model_name` | `--best-model-name` | `'best_model.pt'` | `best_model.pt` | 最佳模型 checkpoint 文件名。 |
| `data_path` | `--data-path` | `'halfcheetah_medium-v2.hdf5'` | `./halfcheetah_medium-v2.hdf5` | 数据文件路径，决定读取哪个 HDF5 离线数据集。 |
| `env_name` | `--env-name` | `'halfcheetah-medium-v2'` | `halfcheetah-medium-v2` | 环境名称标签，主要用于记录实验。 |
| `history_csv_name` | `--history-csv-name` | `'training_history.csv'` | `training_history.csv` | 训练历史 CSV 文件名。 |
| `save_dir` | `--save-dir` | `'checkpoints/SAC'` | `checkpoints/SAC` | 输出目录，用来保存 checkpoint、日志和图片。 |

### 训练通用
| 参数 | 命令行/来源 | 源码默认值 | 当前训练记录 | 怎么理解 |
| --- | --- | --- | --- | --- |
| `batch_size` | `--batch-size` | `256` | `256` | 每个 batch 的样本数量；大一些更稳定但更占显存。 |
| `device` | `Config` 字段 | `'cuda' if torch.cuda.is_available() else 'cpu'` | `cuda` | 训练设备，CPU 或 CUDA。 |
| `grad_clip` | `--grad-clip` | `5.0` | `5.0` | 梯度裁剪阈值，防止训练不稳定。 |
| `lr_actor` | `--lr-actor` | `0.0003` | `0.0003` | actor 学习率，控制策略网络更新速度。 |
| `lr_alpha` | `--lr-alpha` | `0.0003` | `0.0003` | SAC 熵温度 alpha 的学习率。 |
| `lr_critic` | `--lr-critic` | `0.0003` | `0.0003` | critic 学习率，控制 Q 网络更新速度。 |
| `num_epochs` | `--num-epochs` | `150` | `150` | 训练轮数；决定完整遍历数据集的次数。 |
| `num_workers` | `--num-workers` | `0` | `0` | DataLoader 子进程数。 |
| `seed` | `--seed` | `42` | `42` | 随机种子，用于复现实验。 |
| `val_ratio` | `--val-ratio` | `0.1` | `0.1` | 验证集比例。 |
| `weight_decay` | `--weight-decay` | `0.0001` | `0.0001` | Adam 的 L2 正则项。 |

### 网络结构
| 参数 | 命令行/来源 | 源码默认值 | 当前训练记录 | 怎么理解 |
| --- | --- | --- | --- | --- |
| `hidden_sizes` | `--hidden-sizes` | `field(default_factory=lambda: [256, 256])` | `[256, 256]` | 隐藏层结构，控制 MLP 容量。 |
| `log_std_max` | `--log-std-max` | `2.0` | `2.0` | Gaussian actor 的 log_std 上限。 |
| `log_std_min` | `--log-std-min` | `-5.0` | `-5.0` | Gaussian actor 的 log_std 下限。 |

### 算法核心
| 参数 | 命令行/来源 | 源码默认值 | 当前训练记录 | 怎么理解 |
| --- | --- | --- | --- | --- |
| `alpha_init` | `--alpha-init` | `0.2` | `0.2` | SAC 熵温度初始值。 |
| `gamma` | `--gamma` | `0.99` | `0.99` | 折扣因子，越接近 1 越重视长期回报。 |
| `learn_alpha` | `--learn-alpha` | `True` | `True` | 是否自动学习 SAC 熵温度。 |
| `reward_scale` | `--reward-scale` | `True` | `True` | 是否标准化 reward，影响 TD target 的数值尺度。 |
| `tau` | `--tau` | `0.005` | `0.005` | target 网络软更新系数，越小更新越平滑。 |
| `weight_clip` | `--weight-clip` | `20.0` | `20.0` | advantage 权重上限。 |

### 其它参数
| 参数 | 命令行/来源 | 源码默认值 | 当前训练记录 | 怎么理解 |
| --- | --- | --- | --- | --- |
| `awac_lambda` | `--awac-lambda` | `1.0` | `1.0` | 补充参数，具体作用可结合逐行附录查看。 |
| `cpu` | `--cpu` | `` | `` | 补充参数，具体作用可结合逐行附录查看。 |
| `gpu` | `--gpu` | `` | `` | 补充参数，具体作用可结合逐行附录查看。 |
| `no_learn_alpha` | `--no-learn-alpha` | `` | `` | 补充参数，具体作用可结合逐行附录查看。 |
| `no_reward_scale` | `--no-reward-scale` | `` | `` | 补充参数，具体作用可结合逐行附录查看。 |

## 读代码顺序建议
1. 先看 `Config`。
2. 先看 `build_config_from_args`。
3. 先看 `load_hdf5_dataset`。
4. 先看 `split_dataset`。
5. 先看 `compute_obs_stats`。
6. 先看 `GaussianActor`。
7. 先看 `Critic`。
8. 先看 `MLP`。
9. 先看 `OfflineAWACSACAgent.train_step`。
10. 先看 `train_sac`。
11. 先看 `main`。
12. 需要抠每一行时，再打开逐行附录：[`SAC_line_by_line.md`](line_by_line/SAC_line_by_line.md)。

## 常见困惑
- `alpha` 是熵温度，不是学习率。
- `log_prob_data` 表示数据动作在当前策略下的概率，不是采样动作概率。
- `SAC.py` 有较多 NaN/Inf 保护逻辑，读起来会比其它脚本稍长。

## 逐行附录链接
逐行解释已经移到单独文件，避免主文档太长：[`line_by_line/SAC_line_by_line.md`](line_by_line/SAC_line_by_line.md)。

## 注意事项
- 这是离线 SAC 变体，actor 更新不是普通在线 SAC，而是 AWAC 风格。
- 代码里有 NaN/Inf 检查和张量裁剪，说明作者考虑了 SAC 训练中常见的数值不稳定。
- `awac_lambda` 和 `weight_clip` 对 actor 行为影响很大。
- `learn_alpha=True` 时 alpha 会自动变化，曲线 `alpha.png` 可用于观察熵温度是否稳定。

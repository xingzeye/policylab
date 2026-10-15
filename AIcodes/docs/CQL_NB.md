# CQL_NB.py 分层讲解

## 快速理解
- 它是在 CQL 上加入 TD3 风格 target policy smoothing。
- 下一状态动作会加裁剪噪声，让 Q 目标更平滑。
- CQL penalty 也比较更多类型的动作。

一句话记法：**CQL_NB 带平滑目标的 CQL** 的重点是：这个版本可以看成 CQL 和 TD3 平滑技巧的结合：既保守估计 Q，又让 target action 附近的小扰动也被考虑进去。

## 先看整体流程
1. 读取完整离线转移。
2. 建立 actor、双 critic、target critic。
3. 计算下一状态动作时加入 `policy_noise` 并用 `noise_clip` 裁剪。
4. critic loss = Bellman MSE + CQL penalty。
5. CQL penalty 同时看随机动作、当前策略动作、下一状态噪声策略动作。
6. actor 仍使用 Q 项 + BC 项。
7. 保存验证 critic loss 最低的模型。

## 核心原理
这个版本可以看成 CQL 和 TD3 平滑技巧的结合：既保守估计 Q，又让 target action 附近的小扰动也被考虑进去。

核心公式可以先记成：

```text
next_a = clamp(actor(s_next) + clipped_noise, -1, 1)
penalty = logsumexp(Q(random), Q(pi(s)), Q(noisy_pi(s_next))) - Q(data)
```

读代码时不用一开始就追每个张量的形状，先抓住三件事：数据从哪里来、loss 怎么算、什么时候保存模型。

## 关键代码精讲
### 关键函数/类导航
| 类型 | 名称 | 源码行 | 作用 | 输入/输出 | 所在流程 |
| --- | --- | --- | --- | --- | --- |
| 函数 | `ensure_dirs` | 21 | 创建输出目录。 | 输入：output_dir；输出：中间结果或无返回值。 | 配置准备 |
| 函数 | `set_seed` | 26 | 固定随机种子，提升实验可复现性。 | 输入：seed；输出：中间结果或无返回值。 | 配置准备 |
| 函数 | `soft_update` | 34 | 软更新 target 网络。 | 输入：source_net, target_net, tau；输出：中间结果或无返回值。 | 训练更新 |
| 函数 | `parse_hidden_layers` | 39 | 把命令行里的隐藏层字符串转成整数列表。 | 输入：value；输出：隐藏层整数列表。 | 配置准备 |
| 类 | `Config` | 56 | 集中保存训练参数和输出文件名。 | 输入：构造参数；输出：可复用对象。 | 配置准备 |
| 函数 | `build_config_from_args` | 96 | 解析命令行参数，创建配置对象，并保存 train_config.json。 | 输入：无显式输入；输出：配置对象 cfg。 | 配置准备 |
| 类 | `MLP` | 194 | 通用多层感知机，供 actor 或 critic 复用。 | 输入：构造参数；输出：可复用对象。 | 网络定义 |
| 方法 | `MLP.__init__` | 195 | 初始化网络层、优化器或训练状态。 | 输入：input_dim, output_dim, hidden_sizes, use_tanh；输出：中间结果或无返回值。 | 网络定义 |
| 方法 | `MLP.forward` | 207 | 定义模型前向计算。 | 输入：x；输出：网络输出张量。 | 网络定义 |
| 函数 | `load_hdf5_dataset` | 218 | 读取 HDF5 完整转移数据。 | 输入：path, batch_size, val_ratio, seed, num_workers, pin_memory；输出：数据集、DataLoader、维度或模型对象。 | 数据处理 |
| 函数 | `compute_obs_stats` | 276 | 计算状态标准化用的均值和标准差。 | 输入：train_dataset；输出：obs_mean 和 obs_std。 | 数据处理 |
| 函数 | `compute_cql_penalty` | 290 | 计算 CQL 保守惩罚项。 | 输入：critic, actor, s, s_next, a_data, num_random_actions, policy_noise, noise_clip；输出：CQL penalty 标量。 | 训练更新 |
| 函数 | `evaluate_cql` | 321 | 在验证集上评估 CQL 指标。 | 输入：actor, critic1, critic2, target_critic1, target_critic2, loader, obs_mean, obs_std, reward_mean, reward_std, args；输出：评估指标或 loss。 | 验证评估 |
| 函数 | `train_cql_improved` | 420 | CQL 主训练流程。 | 输入：train_loader, val_loader, state_dim, action_dim, train_dataset, reward_mean, reward_std, args, save_dir；输出：训练指标、日志或最佳模型文件。 | 训练更新 |
| 函数 | `load_best_actor` | 657 | 加载最佳 actor 用于推理。 | 输入：args, save_dir；输出：数据集、DataLoader、维度或模型对象。 | 保存/推理 |

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
| `actor_curve_name` | `--actor-curve-name` | `'actor_loss_curve.png'` | `actor_loss_curve.png` | actor loss 曲线文件名。 |
| `best_model_name` | `--best-model-name` | `'best_model.pt'` | `best_model.pt` | 最佳模型 checkpoint 文件名。 |
| `cql_curve_name` | `--cql-curve-name` | `'cql_penalty_curve.png'` | `cql_penalty_curve.png` | CQL penalty 曲线文件名。 |
| `critic_curve_name` | `--critic-curve-name` | `'critic_loss_curve.png'` | `critic_loss_curve.png` | critic loss 曲线文件名。 |
| `dataset_path` | `--dataset-path` | `'halfcheetah_medium-v2.hdf5'` | `halfcheetah_medium-v2.hdf5` | 数据文件路径，决定读取哪个 HDF5 离线数据集。 |
| `log_csv_name` | `--log-csv-name` | `'training_log.csv'` | `training_log.csv` | 训练日志 CSV 文件名。 |
| `output_dir` | `--output-dir` | `os.path.join('checkpoints', 'CQL_NB')` | `checkpoints\CQL_NB` | 输出目录，用来保存 checkpoint、日志和图片。 |

### 训练通用
| 参数 | 命令行/来源 | 源码默认值 | 当前训练记录 | 怎么理解 |
| --- | --- | --- | --- | --- |
| `batch_size` | `--batch-size` | `256` | `256` | 每个 batch 的样本数量；大一些更稳定但更占显存。 |
| `device` | `Config` 字段 | `'cuda' if torch.cuda.is_available() else 'cpu'` | `cuda` | 训练设备，CPU 或 CUDA。 |
| `grad_clip` | `--grad-clip` | `5.0` | `5.0` | 梯度裁剪阈值，防止训练不稳定。 |
| `lr_actor` | `--lr-actor` | `0.0003` | `0.0003` | actor 学习率，控制策略网络更新速度。 |
| `lr_critic` | `--lr-critic` | `0.0003` | `0.0003` | critic 学习率，控制 Q 网络更新速度。 |
| `num_epochs` | `--num-epochs` | `150` | `150` | 训练轮数；决定完整遍历数据集的次数。 |
| `num_workers` | `--num-workers` | `0` | `0` | DataLoader 子进程数。 |
| `seed` | `--seed` | `42` | `42` | 随机种子，用于复现实验。 |
| `val_ratio` | `--val-ratio` | `0.1` | `0.1` | 验证集比例。 |
| `weight_decay` | `--weight-decay` | `0.0001` | `0.0001` | Adam 的 L2 正则项。 |

### 网络结构
| 参数 | 命令行/来源 | 源码默认值 | 当前训练记录 | 怎么理解 |
| --- | --- | --- | --- | --- |
| `hidden_layers` | `--hidden-layers` | `field(default_factory=lambda: [256, 256])` | `[256, 256]` | 隐藏层结构，控制 MLP 容量。 |

### 算法核心
| 参数 | 命令行/来源 | 源码默认值 | 当前训练记录 | 怎么理解 |
| --- | --- | --- | --- | --- |
| `alpha_bc` | `--alpha-bc` | `2.5` | `2.5` | Q 项自适应缩放系数。 |
| `bc_coef` | `--bc-coef` | `1.0` | `1.0` | 行为克隆正则权重。 |
| `gamma` | `--gamma` | `0.99` | `0.99` | 折扣因子，越接近 1 越重视长期回报。 |
| `lambda_cql` | `--lambda-cql` | `5.0` | `5.0` | CQL 保守惩罚权重。 |
| `noise_clip` | `--noise-clip` | `0.5` | `0.5` | target action 噪声裁剪范围。 |
| `num_random_actions` | `--num-random-actions` | `10` | `10` | 每个状态采样多少随机动作来估计 CQL penalty。 |
| `policy_noise` | `--policy-noise` | `0.2` | `0.2` | target action 噪声标准差。 |
| `reward_scale` | `--reward-scale` | `True` | `True` | 是否标准化 reward，影响 TD target 的数值尺度。 |
| `tau` | `--tau` | `0.005` | `0.005` | target 网络软更新系数，越小更新越平滑。 |

### 其它参数
| 参数 | 命令行/来源 | 源码默认值 | 当前训练记录 | 怎么理解 |
| --- | --- | --- | --- | --- |
| `cpu` | `--cpu` | `` | `` | 补充参数，具体作用可结合逐行附录查看。 |
| `gpu` | `--gpu` | `` | `` | 补充参数，具体作用可结合逐行附录查看。 |
| `no_reward_scale` | `--no-reward-scale` | `` | `` | 补充参数，具体作用可结合逐行附录查看。 |

## 读代码顺序建议
1. 先看 `Config`。
2. 先看 `build_config_from_args`。
3. 先看 `load_hdf5_dataset`。
4. 先看 `compute_obs_stats`。
5. 先看 `MLP`。
6. 先看 `train_cql_improved`。
7. 需要抠每一行时，再打开逐行附录：[`CQL_NB_line_by_line.md`](line_by_line/CQL_NB_line_by_line.md)。

## 常见困惑
- `policy_noise` 不是探索环境用的，而是训练 target 时的平滑噪声。
- 它比 `CQL.py` 更像“CQL + TD3 技巧”。
- `noise_clip` 太大可能让目标动作偏离数据分布。

## 逐行附录链接
逐行解释已经移到单独文件，避免主文档太长：[`line_by_line/CQL_NB_line_by_line.md`](line_by_line/CQL_NB_line_by_line.md)。

## 注意事项
- 该变体更接近“CQL + TD3 平滑目标”的混合实现。
- `policy_noise` 和 `noise_clip` 过大可能让 target action 偏离数据分布太多。
- `lambda_cql` 与 `bc_coef` 需要平衡：一个控制 critic 保守程度，一个控制 actor 离线约束。
- 和 `CQL.py` 一样，代码主要依赖 `terminals`，没有显式合并 `timeouts`。

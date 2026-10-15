# DDPG_NB.py 分层讲解

## 快速理解
- 它用双 Q 和 target policy smoothing 训练 critic。
- actor 的目标是 Q 项 + BC 项。
- 可以理解成 TD3+BC 思路的 DDPG 写法。

一句话记法：**DDPG_NB / TD3-lite + BC** 的重点是：离线 actor 不能只追求高 Q，因为 critic 可能对没见过的动作过度乐观。BC 项把 actor 拉回数据动作附近，Q 项则负责提升性能。

## 先看整体流程
1. 读取完整离线转移。
2. 建立 actor、target actor、双 critic、target critic。
3. critic 用 TD target 学 Q。
4. actor 每隔 `policy_delay` 更新一次。
5. actor loss 由 `-Q` 项和 `MSE(actor动作, 数据动作)` 组成。
6. 用 `alpha_bc / mean(abs(Q))` 自动缩放 Q 项。
7. 保存训练日志和最佳模型。

## 核心原理
离线 actor 不能只追求高 Q，因为 critic 可能对没见过的动作过度乐观。BC 项把 actor 拉回数据动作附近，Q 项则负责提升性能。

核心公式可以先记成：

```text
q_loss = -lam * mean(Q(s, pi(s)))
bc_loss = bc_coef * MSE(pi(s), a_data)
actor_loss = q_loss + bc_loss
```

读代码时不用一开始就追每个张量的形状，先抓住三件事：数据从哪里来、loss 怎么算、什么时候保存模型。

## 关键代码精讲
### 关键函数/类导航
| 类型 | 名称 | 源码行 | 作用 | 输入/输出 | 所在流程 |
| --- | --- | --- | --- | --- | --- |
| 函数 | `set_seed` | 20 | 固定随机种子，提升实验可复现性。 | 输入：seed；输出：中间结果或无返回值。 | 配置准备 |
| 函数 | `parse_hidden_layers` | 28 | 把命令行里的隐藏层字符串转成整数列表。 | 输入：value；输出：隐藏层整数列表。 | 配置准备 |
| 函数 | `ensure_dirs` | 41 | 创建输出目录。 | 输入：output_dir；输出：中间结果或无返回值。 | 配置准备 |
| 类 | `Config` | 50 | 集中保存训练参数和输出文件名。 | 输入：构造参数；输出：可复用对象。 | 配置准备 |
| 函数 | `build_config_from_args` | 85 | 解析命令行参数，创建配置对象，并保存 train_config.json。 | 输入：无显式输入；输出：配置对象 cfg。 | 配置准备 |
| 类 | `Actor` | 177 | 确定性策略网络，输入状态输出动作。 | 输入：构造参数；输出：可复用对象。 | 网络定义 |
| 方法 | `Actor.__init__` | 178 | 初始化网络层、优化器或训练状态。 | 输入：state_dim, action_dim, hidden；输出：中间结果或无返回值。 | 网络定义 |
| 方法 | `Actor.forward` | 190 | 定义模型前向计算。 | 输入：state；输出：网络输出张量。 | 网络定义 |
| 类 | `Critic` | 194 | Q 网络，输入状态动作对输出 Q 值。 | 输入：构造参数；输出：可复用对象。 | 网络定义 |
| 方法 | `Critic.__init__` | 195 | 初始化网络层、优化器或训练状态。 | 输入：state_dim, action_dim, hidden；输出：中间结果或无返回值。 | 网络定义 |
| 方法 | `Critic.forward` | 206 | 定义模型前向计算。 | 输入：state, action；输出：网络输出张量。 | 网络定义 |
| 类 | `OfflineDataset` | 214 | 把 HDF5 转移数据包装成 PyTorch Dataset。 | 输入：构造参数；输出：可复用对象。 | 数据处理 |
| 方法 | `OfflineDataset.__init__` | 215 | 初始化网络层、优化器或训练状态。 | 输入：hdf5_file；输出：中间结果或无返回值。 | 数据处理 |
| 方法 | `OfflineDataset.__len__` | 227 | 返回数据集长度。 | 输入：无显式输入；输出：中间结果或无返回值。 | 数据处理 |
| 方法 | `OfflineDataset.__getitem__` | 230 | 按索引取出一个训练样本。 | 输入：idx；输出：中间结果或无返回值。 | 数据处理 |
| 函数 | `compute_obs_stats` | 243 | 计算状态标准化用的均值和标准差。 | 输入：subset；输出：obs_mean 和 obs_std。 | 数据处理 |
| 类 | `OfflineDDPGBCAgent` | 257 | 封装 DDPG/TD3+BC 的网络、优化器和训练步骤。 | 输入：构造参数；输出：可复用对象。 | 辅助逻辑 |
| 方法 | `OfflineDDPGBCAgent.__init__` | 258 | 初始化网络层、优化器或训练状态。 | 输入：state_dim, action_dim, lr_actor, lr_critic, gamma, tau, hidden, device, weight_decay, grad_clip, alpha_bc, bc_coef, reward_scale, policy_noise, noise_clip, policy_delay；输出：中间结果或无返回值。 | 辅助逻辑 |
| 方法 | `OfflineDDPGBCAgent.set_schedulers` | 317 | 保存学习率调度器引用。 | 输入：actor_scheduler, critic_scheduler；输出：中间结果或无返回值。 | 训练更新 |
| 方法 | `OfflineDDPGBCAgent.soft_update` | 321 | 软更新 target 网络。 | 输入：net, net_target；输出：中间结果或无返回值。 | 训练更新 |
| 方法 | `OfflineDDPGBCAgent.select_action` | 325 | 用 actor 对单个状态输出动作。 | 输入：state, obs_mean, obs_std；输出：中间结果或无返回值。 | 保存/推理 |
| 方法 | `OfflineDDPGBCAgent.train_step` | 333 | 执行一个 batch 的训练更新。 | 输入：batch, obs_mean, obs_std, reward_mean, reward_std；输出：训练指标、日志或最佳模型文件。 | 训练更新 |
| 方法 | `OfflineDDPGBCAgent.eval_step` | 419 | 计算一个 batch 的验证指标。 | 输入：batch, obs_mean, obs_std, reward_mean, reward_std；输出：评估指标或 loss。 | 验证评估 |
| 方法 | `OfflineDDPGBCAgent.save_best` | 471 | 保存当前最佳模型。 | 输入：output_dir, best_model_name, obs_mean, obs_std, reward_mean, reward_std, epoch, best_val_loss, state_dim, action_dim, hidden；输出：写入文件，通常无主要返回值。 | 保存/推理 |
| 函数 | `train_ddpg_improved` | 510 | DDPG 类算法主训练流程。 | 输入：agent, train_loader, val_loader, num_epochs, output_dir, best_model_name, records_csv_name, actor_curve_name, critic_curve_name, bc_curve_name, obs_mean, obs_std, reward_mean, reward_std, state_dim, action_dim, hidden；输出：训练指标、日志或最佳模型文件。 | 训练更新 |
| 函数 | `load_best_actor` | 705 | 加载最佳 actor 用于推理。 | 输入：device, output_dir, best_model_name；输出：数据集、DataLoader、维度或模型对象。 | 保存/推理 |
| 函数 | `main` | 728 | 脚本入口，串起配置、数据、训练和加载。 | 输入：无显式输入；输出：无返回值，负责启动流程。 | 脚本入口 |

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
| `bc_curve_name` | `--bc-curve-name` | `'bc_loss_curve.png'` | `bc_loss_curve.png` | BC loss 曲线文件名。 |
| `best_model_name` | `--best-model-name` | `'best_model.pt'` | `best_model.pt` | 最佳模型 checkpoint 文件名。 |
| `critic_curve_name` | `--critic-curve-name` | `'critic_loss_curve.png'` | `critic_loss_curve.png` | critic loss 曲线文件名。 |
| `hdf5_file` | `--hdf5-file` | `'halfcheetah_medium-v2.hdf5'` | `halfcheetah_medium-v2.hdf5` | 数据文件路径，决定读取哪个 HDF5 离线数据集。 |
| `output_dir` | `--output-dir` | `'checkpoints/DDPG_NB'` | `checkpoints/DDPG_NB` | 输出目录，用来保存 checkpoint、日志和图片。 |
| `records_csv_name` | `--records-csv-name` | `'training_records.csv'` | `training_records.csv` | 训练记录 CSV 文件名。 |

### 训练通用
| 参数 | 命令行/来源 | 源码默认值 | 当前训练记录 | 怎么理解 |
| --- | --- | --- | --- | --- |
| `batch_size` | `--batch-size` | `256` | `256` | 每个 batch 的样本数量；大一些更稳定但更占显存。 |
| `device` | `Config` 字段 | `'cuda' if torch.cuda.is_available() else 'cpu'` | `cuda` | 训练设备，CPU 或 CUDA。 |
| `grad_clip` | `--grad-clip` | `5.0` | `5.0` | 梯度裁剪阈值，防止训练不稳定。 |
| `lr_actor` | `--lr-actor` | `0.0003` | `0.0003` | actor 学习率，控制策略网络更新速度。 |
| `lr_critic` | `--lr-critic` | `0.0003` | `0.0003` | critic 学习率，控制 Q 网络更新速度。 |
| `num_epochs` | `--num-epochs` | `120` | `150` | 训练轮数；决定完整遍历数据集的次数。 |
| `num_workers` | `--num-workers` | `0` | `0` | DataLoader 子进程数。 |
| `seed` | `--seed` | `42` | `42` | 随机种子，用于复现实验。 |
| `val_ratio` | `--val-ratio` | `0.1` | `0.1` | 验证集比例。 |
| `weight_decay` | `--weight-decay` | `0.0001` | `0.0001` | Adam 的 L2 正则项。 |

### 网络结构
| 参数 | 命令行/来源 | 源码默认值 | 当前训练记录 | 怎么理解 |
| --- | --- | --- | --- | --- |
| `hidden` | `--hidden` | `field(default_factory=lambda: [256, 256])` | `[256, 256]` | 隐藏层结构，控制 MLP 容量。 |

### 算法核心
| 参数 | 命令行/来源 | 源码默认值 | 当前训练记录 | 怎么理解 |
| --- | --- | --- | --- | --- |
| `alpha_bc` | `--alpha-bc` | `2.5` | `2.5` | Q 项自适应缩放系数。 |
| `bc_coef` | `--bc-coef` | `1.0` | `1.0` | 行为克隆正则权重。 |
| `gamma` | `--gamma` | `0.99` | `0.99` | 折扣因子，越接近 1 越重视长期回报。 |
| `noise_clip` | `--noise-clip` | `0.5` | `0.5` | target action 噪声裁剪范围。 |
| `policy_delay` | `--policy-delay` | `2` | `2` | actor 延迟更新间隔。 |
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
3. 先看 `compute_obs_stats`。
4. 先看 `Actor`。
5. 先看 `Critic`。
6. 先看 `OfflineDDPGBCAgent.train_step`。
7. 先看 `train_ddpg_improved`。
8. 先看 `main`。
9. 需要抠每一行时，再打开逐行附录：[`DDPG_NB_line_by_line.md`](line_by_line/DDPG_NB_line_by_line.md)。

## 常见困惑
- `alpha_bc` 控制 Q 项缩放，`bc_coef` 控制贴近数据的力度。
- 它和 `DDPG.py` 的差异主要在 actor loss。
- reward scale 会改变日志里的 reward 和 Q 数值尺度。

## 逐行附录链接
逐行解释已经移到单独文件，避免主文档太长：[`line_by_line/DDPG_NB_line_by_line.md`](line_by_line/DDPG_NB_line_by_line.md)。

## 注意事项
- 这个脚本的核心是 TD3+BC 风格 actor loss，而不是 `DDPG.py` 的 advantage-weighted BC。
- `alpha_bc` 和 `bc_coef` 决定“追求高 Q”和“贴近数据”的平衡。
- 只用 `critic1` 计算 actor 的 Q 项，这是 TD3/TD3+BC 常见做法。
- reward 标准化会改变日志中 reward 和 target Q 的数值尺度。

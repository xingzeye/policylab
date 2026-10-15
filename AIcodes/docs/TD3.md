# TD3.py 分层讲解

## 快速理解
- 它是 DDPG 的稳定版：双 Q、延迟 actor 更新、target policy smoothing。
- 代码直接用离线数据训练，不和环境交互。
- 没有 BC 正则，所以是理解 TD3 机制的基础版。

一句话记法：**TD3 离线训练** 的重点是：TD3 的三个技巧都服务于稳定 Q 学习：双 Q 防过估计，噪声 target 让 Q 更平滑，延迟 actor 让 critic 先学稳。

## 先看整体流程
1. 读取完整转移数据。
2. 建立 actor、target actor、双 Q critic、target critic。
3. target actor 的下一动作加入裁剪噪声。
4. critic 用两个 Q 的较小值构造 TD target。
5. 每步更新 critic。
6. 每隔 `policy_delay` 更新 actor。
7. 保存 critic loss 最低的模型和曲线。

## 核心原理
TD3 的三个技巧都服务于稳定 Q 学习：双 Q 防过估计，噪声 target 让 Q 更平滑，延迟 actor 让 critic 先学稳。

核心公式可以先记成：

```text
target_Q = r + gamma * min(Q1_target, Q2_target)
actor_loss = -mean(Q1(s, actor(s)))
```

读代码时不用一开始就追每个张量的形状，先抓住三件事：数据从哪里来、loss 怎么算、什么时候保存模型。

## 关键代码精讲
### 关键函数/类导航
| 类型 | 名称 | 源码行 | 作用 | 输入/输出 | 所在流程 |
| --- | --- | --- | --- | --- | --- |
| 函数 | `set_seed` | 20 | 固定随机种子，提升实验可复现性。 | 输入：seed；输出：中间结果或无返回值。 | 配置准备 |
| 函数 | `ensure_dirs` | 28 | 创建输出目录。 | 输入：save_dir；输出：中间结果或无返回值。 | 配置准备 |
| 函数 | `parse_hidden` | 33 | 把命令行里的隐藏层字符串转成整数列表。 | 输入：value；输出：隐藏层整数列表。 | 配置准备 |
| 函数 | `build_config_from_args` | 46 | 解析命令行参数，创建配置对象，并保存 train_config.json。 | 输入：无显式输入；输出：配置对象 cfg。 | 配置准备 |
| 类 | `Config` | 133 | 集中保存训练参数和输出文件名。 | 输入：构造参数；输出：可复用对象。 | 配置准备 |
| 类 | `Actor` | 166 | 确定性策略网络，输入状态输出动作。 | 输入：构造参数；输出：可复用对象。 | 网络定义 |
| 方法 | `Actor.__init__` | 167 | 初始化网络层、优化器或训练状态。 | 输入：state_dim, action_dim, max_action, hidden；输出：中间结果或无返回值。 | 网络定义 |
| 方法 | `Actor.forward` | 179 | 定义模型前向计算。 | 输入：state；输出：网络输出张量。 | 网络定义 |
| 类 | `Critic` | 183 | Q 网络，输入状态动作对输出 Q 值。 | 输入：构造参数；输出：可复用对象。 | 网络定义 |
| 方法 | `Critic.__init__` | 184 | 初始化网络层、优化器或训练状态。 | 输入：state_dim, action_dim, hidden；输出：中间结果或无返回值。 | 网络定义 |
| 方法 | `Critic.forward` | 203 | 定义模型前向计算。 | 输入：state, action；输出：网络输出张量。 | 网络定义 |
| 方法 | `Critic.q1_forward` | 207 | 只计算第一个 Q 网络的输出，常用于 actor loss。 | 输入：state, action；输出：网络输出张量。 | 网络定义 |
| 函数 | `compute_obs_stats` | 215 | 计算状态标准化用的均值和标准差。 | 输入：train_dataset；输出：obs_mean 和 obs_std。 | 数据处理 |
| 函数 | `soft_update` | 229 | 软更新 target 网络。 | 输入：source_net, target_net, tau；输出：中间结果或无返回值。 | 训练更新 |
| 类 | `TD3Trainer` | 237 | 封装 TD3 训练步骤。 | 输入：构造参数；输出：可复用对象。 | 辅助逻辑 |
| 方法 | `TD3Trainer.__init__` | 238 | 初始化网络层、优化器或训练状态。 | 输入：state_dim, action_dim, max_action, args；输出：中间结果或无返回值。 | 辅助逻辑 |
| 方法 | `TD3Trainer.set_schedulers` | 271 | 保存学习率调度器引用。 | 输入：actor_scheduler, critic_scheduler；输出：中间结果或无返回值。 | 训练更新 |
| 方法 | `TD3Trainer.train_step` | 275 | 执行一个 batch 的训练更新。 | 输入：batch, obs_mean, obs_std；输出：训练指标、日志或最佳模型文件。 | 训练更新 |
| 方法 | `TD3Trainer.eval_step` | 331 | 计算一个 batch 的验证指标。 | 输入：batch, obs_mean, obs_std；输出：评估指标或 loss。 | 验证评估 |
| 方法 | `TD3Trainer.save_best` | 364 | 保存当前最佳模型。 | 输入：save_path, obs_mean, obs_std, epoch, best_val_loss, args；输出：写入文件，通常无主要返回值。 | 保存/推理 |
| 函数 | `save_plot_csv` | 384 | 保存 CSV 日志和曲线图。 | 输入：df, cfg；输出：写入文件，通常无主要返回值。 | 保存/推理 |
| 函数 | `main` | 430 | 脚本入口，串起配置、数据、训练和加载。 | 输入：无显式输入；输出：无返回值，负责启动流程。 | 脚本入口 |

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
| `actor_curve_name` | `--actor-curve-name` | `'actor_loss.png'` | `actor_loss.png` | actor loss 曲线文件名。 |
| `best_model_name` | `--best-model-name` | `'best_model.pt'` | `best_model.pt` | 最佳模型 checkpoint 文件名。 |
| `critic_curve_name` | `--critic-curve-name` | `'critic_loss.png'` | `critic_loss.png` | critic loss 曲线文件名。 |
| `data_path` | `--data-path` | `'halfcheetah_medium-v2.hdf5'` | `halfcheetah_medium-v2.hdf5` | 数据文件路径，决定读取哪个 HDF5 离线数据集。 |
| `log_csv_name` | `--log-csv-name` | `'training_log.csv'` | `training_log.csv` | 训练日志 CSV 文件名。 |
| `q_curve_name` | `--q-curve-name` | `'q_value.png'` | `q_value.png` | Q 值曲线文件名。 |
| `save_dir` | `--save-dir` | `'checkpoints/TD3'` | `checkpoints/TD3` | 输出目录，用来保存 checkpoint、日志和图片。 |

### 训练通用
| 参数 | 命令行/来源 | 源码默认值 | 当前训练记录 | 怎么理解 |
| --- | --- | --- | --- | --- |
| `batch_size` | `--batch-size` | `256` | `256` | 每个 batch 的样本数量；大一些更稳定但更占显存。 |
| `device` | `Config` 字段 | `'cuda' if torch.cuda.is_available() else 'cpu'` | `cuda` | 训练设备，CPU 或 CUDA。 |
| `grad_clip` | `--grad-clip` | `5.0` | `5.0` | 梯度裁剪阈值，防止训练不稳定。 |
| `lr_actor` | `--lr-actor` | `0.001` | `0.001` | actor 学习率，控制策略网络更新速度。 |
| `lr_critic` | `--lr-critic` | `0.001` | `0.001` | critic 学习率，控制 Q 网络更新速度。 |
| `num_epochs` | `--num-epochs` | `100` | `150` | 训练轮数；决定完整遍历数据集的次数。 |
| `num_workers` | `--num-workers` | `0` | `0` | DataLoader 子进程数。 |
| `seed` | `--seed` | `42` | `42` | 随机种子，用于复现实验。 |
| `val_ratio` | `--val-ratio` | `0.1` | `0.1` | 验证集比例。 |
| `weight_decay` | `--weight-decay` | `0.0001` | `0.0001` | Adam 的 L2 正则项。 |

### 网络结构
| 参数 | 命令行/来源 | 源码默认值 | 当前训练记录 | 怎么理解 |
| --- | --- | --- | --- | --- |
| `hidden` | `--hidden` | `field(default_factory=lambda: [256, 256])` | `[256, 256]` | 隐藏层结构，控制 MLP 容量。 |
| `max_action` | `--max-action` | `1.0` | `1.0` | 动作最大幅度。 |

### 算法核心
| 参数 | 命令行/来源 | 源码默认值 | 当前训练记录 | 怎么理解 |
| --- | --- | --- | --- | --- |
| `gamma` | `--gamma` | `0.99` | `0.99` | 折扣因子，越接近 1 越重视长期回报。 |
| `noise_clip` | `--noise-clip` | `0.5` | `0.5` | target action 噪声裁剪范围。 |
| `policy_delay` | `--policy-delay` | `2` | `2` | actor 延迟更新间隔。 |
| `policy_noise` | `--policy-noise` | `0.2` | `0.2` | target action 噪声标准差。 |
| `tau` | `--tau` | `0.005` | `0.005` | target 网络软更新系数，越小更新越平滑。 |

### 其它参数
| 参数 | 命令行/来源 | 源码默认值 | 当前训练记录 | 怎么理解 |
| --- | --- | --- | --- | --- |
| `cpu` | `--cpu` | `` | `` | 补充参数，具体作用可结合逐行附录查看。 |
| `gpu` | `--gpu` | `` | `` | 补充参数，具体作用可结合逐行附录查看。 |

## 读代码顺序建议
1. 先看 `Config`。
2. 先看 `build_config_from_args`。
3. 先看 `compute_obs_stats`。
4. 先看 `Actor`。
5. 先看 `Critic`。
6. 先看 `TD3Trainer.train_step`。
7. 先看 `main`。
8. 需要抠每一行时，再打开逐行附录：[`TD3_line_by_line.md`](line_by_line/TD3_line_by_line.md)。

## 常见困惑
- `TD3.py` 没有 BC 约束，离线场景可能更容易出分布外动作。
- `actor_proxy` 是验证阶段的 actor 目标近似，不会更新参数。
- 这个文件没有 reward_scale。

## 逐行附录链接
逐行解释已经移到单独文件，避免主文档太长：[`line_by_line/TD3_line_by_line.md`](line_by_line/TD3_line_by_line.md)。

## 注意事项
- 这个文件没有 BC 正则，所以离线场景下 actor 有偏离数据分布的风险。
- `TD3_NB.py` 是带 BC 正则和 reward scale 的变体，更适合离线约束。
- 本脚本没有 reward 标准化参数，直接使用原始 reward。
- actor 只有延迟步才更新，因此部分 batch 的 `actor_loss` 可能为空。

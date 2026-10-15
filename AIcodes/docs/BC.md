# BC.py 分层讲解

## 快速理解
- 它把强化学习问题暂时当成监督学习：输入状态，直接预测数据集里的动作。
- 不使用 reward、不估计 Q 值、不做 Bellman 更新，所以最容易读。
- 适合作为理解其它离线强化学习脚本的第一站。

一句话记法：**BC 行为克隆** 的重点是：BC 的核心是模仿：如果数据动作足够好，就让 `pi(s)` 尽量接近数据里的 `a`。它不判断这个动作未来收益高不高，只关心“像不像”。

## 先看整体流程
1. 读取 `observations` 和 `actions`。
2. 按 `val_ratio` 划分训练集和验证集。
3. 只用训练集状态计算 `obs_mean` 和 `obs_std`。
4. 建立 `BCPolicy`，输入状态，输出动作。
5. 每个 batch 先标准化状态，再预测动作。
6. 用 SmoothL1 或 MSE 比较预测动作和数据动作。
7. 验证 loss 变低时保存 `best_model.pt`。
8. 训练结束保存 CSV 和曲线图。

## 核心原理
BC 的核心是模仿：如果数据动作足够好，就让 `pi(s)` 尽量接近数据里的 `a`。它不判断这个动作未来收益高不高，只关心“像不像”。

核心公式可以先记成：

```text
loss = SmoothL1Loss(pi(s), a_data)
# 或使用 MSELoss(pi(s), a_data)
```

读代码时不用一开始就追每个张量的形状，先抓住三件事：数据从哪里来、loss 怎么算、什么时候保存模型。

## 关键代码精讲
### 关键函数/类导航
| 类型 | 名称 | 源码行 | 作用 | 输入/输出 | 所在流程 |
| --- | --- | --- | --- | --- | --- |
| 类 | `Config` | 22 | 集中保存训练参数和输出文件名。 | 输入：构造参数；输出：可复用对象。 | 配置准备 |
| 函数 | `parse_hidden_layers` | 46 | 把命令行里的隐藏层字符串转成整数列表。 | 输入：value；输出：隐藏层整数列表。 | 配置准备 |
| 函数 | `build_config_from_args` | 64 | 解析命令行参数，创建配置对象，并保存 train_config.json。 | 输入：无显式输入；输出：配置对象 cfg。 | 配置准备 |
| 函数 | `set_seed` | 146 | 固定随机种子，提升实验可复现性。 | 输入：seed；输出：中间结果或无返回值。 | 配置准备 |
| 函数 | `load_dataset` | 158 | 读取 HDF5 数据并创建训练/验证 DataLoader。 | 输入：config；输出：数据集、DataLoader、维度或模型对象。 | 数据处理 |
| 函数 | `compute_obs_stats` | 200 | 计算状态标准化用的均值和标准差。 | 输入：train_dataset；输出：obs_mean 和 obs_std。 | 数据处理 |
| 类 | `BCPolicy` | 215 | 行为克隆策略网络。 | 输入：构造参数；输出：可复用对象。 | 网络定义 |
| 方法 | `BCPolicy.__init__` | 216 | 初始化网络层、优化器或训练状态。 | 输入：input_dim, output_dim, hidden_layers, use_layernorm；输出：中间结果或无返回值。 | 网络定义 |
| 方法 | `BCPolicy.forward` | 231 | 定义模型前向计算。 | 输入：x；输出：网络输出张量。 | 网络定义 |
| 函数 | `train_one_epoch` | 242 | 执行一轮行为克隆训练。 | 输入：model, loader, optimizer, loss_fn, obs_mean, obs_std, config；输出：训练指标、日志或最佳模型文件。 | 训练更新 |
| 函数 | `evaluate_loss` | 273 | 在验证集上计算 loss。 | 输入：model, loader, loss_fn, obs_mean, obs_std, config；输出：评估指标或 loss。 | 验证评估 |
| 函数 | `save_best_model` | 300 | 保存验证表现最好的模型。 | 输入：model, obs_mean, obs_std, state_dim, action_dim, config, epoch, best_val_loss；输出：写入文件，通常无主要返回值。 | 保存/推理 |
| 函数 | `load_best_model` | 320 | 加载保存好的 BC 模型。 | 输入：config；输出：数据集、DataLoader、维度或模型对象。 | 保存/推理 |
| 函数 | `evaluate_subset` | 348 | 对指定子集计算 MSE/MAE 等动作误差。 | 输入：model, subset, obs_mean, obs_std, config；输出：评估指标或 loss。 | 验证评估 |
| 函数 | `save_plots` | 389 | 保存训练曲线。 | 输入：df, config；输出：写入文件，通常无主要返回值。 | 保存/推理 |
| 函数 | `train_bc` | 418 | 行为克隆主训练流程。 | 输入：config；输出：训练指标、日志或最佳模型文件。 | 训练更新 |
| 函数 | `main` | 514 | 脚本入口，串起配置、数据、训练和加载。 | 输入：无显式输入；输出：无返回值，负责启动流程。 | 脚本入口 |

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
| `data_path` | `--data-path` | `'halfcheetah_medium-v2.hdf5'` | `halfcheetah_medium-v2.hdf5` | 数据文件路径，决定读取哪个 HDF5 离线数据集。 |
| `output_dir` | `--output-dir` | `'checkpoints/BC'` | `checkpoints/BC` | 输出目录，用来保存 checkpoint、日志和图片。 |

### 训练通用
| 参数 | 命令行/来源 | 源码默认值 | 当前训练记录 | 怎么理解 |
| --- | --- | --- | --- | --- |
| `batch_size` | `--batch-size` | `256` | `256` | 每个 batch 的样本数量；大一些更稳定但更占显存。 |
| `device` | `Config` 字段 | `torch.device('cuda' if torch.cuda.is_available() else 'cpu')` | `cuda` | 训练设备，CPU 或 CUDA。 |
| `grad_clip` | `--grad-clip` | `5.0` | `5.0` | 梯度裁剪阈值，防止训练不稳定。 |
| `lr` | `--lr` | `0.0003` | `0.0003` | 行为克隆策略网络学习率。 |
| `num_epochs` | `--num-epochs` | `150` | `150` | 训练轮数；决定完整遍历数据集的次数。 |
| `num_workers` | `--num-workers` | `0` | `0` | DataLoader 子进程数。 |
| `seed` | `--seed` | `42` | `42` | 随机种子，用于复现实验。 |
| `val_ratio` | `--val-ratio` | `0.1` | `0.1` | 验证集比例。 |
| `weight_decay` | `--weight-decay` | `0.0001` | `0.0001` | Adam 的 L2 正则项。 |

### 网络结构
| 参数 | 命令行/来源 | 源码默认值 | 当前训练记录 | 怎么理解 |
| --- | --- | --- | --- | --- |
| `hidden_layers` | `--hidden-layers` | `field(default_factory=lambda: [256, 256])` | `[256, 256]` | 隐藏层结构，控制 MLP 容量。 |
| `use_layernorm` | `--use-layernorm` | `` | `False` | 是否在隐藏层加入 LayerNorm。 |

### 算法核心
| 参数 | 命令行/来源 | 源码默认值 | 当前训练记录 | 怎么理解 |
| --- | --- | --- | --- | --- |
| `action_clip_eps` | `--action-clip-eps` | `1e-06` | `1e-06` | 动作裁剪边界余量。 |
| `use_smooth_l1` | `--use-smooth-l1` | `True` | `True` | 是否使用 SmoothL1Loss。 |

### 其它参数
| 参数 | 命令行/来源 | 源码默认值 | 当前训练记录 | 怎么理解 |
| --- | --- | --- | --- | --- |
| `cpu` | `--cpu` | `` | `` | 补充参数，具体作用可结合逐行附录查看。 |
| `gpu` | `--gpu` | `` | `` | 补充参数，具体作用可结合逐行附录查看。 |
| `use_mse` | `--use-mse` | `` | `` | 补充参数，具体作用可结合逐行附录查看。 |

## 读代码顺序建议
1. 先看 `Config`。
2. 先看 `build_config_from_args`。
3. 先看 `load_dataset`。
4. 先看 `compute_obs_stats`。
5. 先看 `BCPolicy`。
6. 先看 `train_one_epoch`。
7. 先看 `train_bc`。
8. 先看 `main`。
9. 需要抠每一行时，再打开逐行附录：[`BC_line_by_line.md`](line_by_line/BC_line_by_line.md)。

## 常见困惑
- `BC.py` 不读取 reward，所以它不会主动区分好轨迹和差轨迹。
- `tanh` 只是把动作压到 `[-1, 1]`，不是强化学习算法本身。
- 源码里 `Config` 和 CLI 参数有少量字段不完全一致，文档会按当前源码如实说明。

## 逐行附录链接
逐行解释已经移到单独文件，避免主文档太长：[`line_by_line/BC_line_by_line.md`](line_by_line/BC_line_by_line.md)。

## 注意事项
- 这是纯模仿学习算法，不使用奖励，也不会主动寻找比数据集更优的动作。
- 对离线数据质量非常敏感；如果数据中包含很多差动作，BC 会照样学习。
- 当前源码中 `Config` 与 `build_config_from_args()` 对 `use_layernorm`、`action_clip_eps` 的字段声明不完全一致，运行前需要留意。
- 训练输出目录已经存在历史结果；重新运行会覆盖同名日志、曲线和最佳模型。

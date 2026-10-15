# CQL.py 逐行代码解释附录

这是 [`CQL.md`](../CQL.md) 的细节附录，保留对应源码每一行非空代码的解释。主文档负责理解流程；这里适合查某一行具体在做什么。

## 逐行代码解释
下面按源码分块展示代码：每个小节先给出干净源码，再用简洁中文解释这段代码的作用、关键变量含义和训练逻辑。

### 模块级代码、导入和入口
**代码：**

```python
import os
import json
import random
import h5py
import copy
import argparse
from dataclasses import dataclass, asdict, field
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# ========================
# Utils
# ========================
# ========================
# Config
# ========================
# ========================
# Network
# ========================
# ========================
# Load Dataset + Split
# ========================
# ========================
# Compute Observation Stats
# ========================
# ========================
# CQL penalty: logsumexp version
# ========================
# ========================
# Evaluate
# ========================
# ========================
# Train CQL
# ========================
# ========================
# Load Best Model
# ========================
# ========================
# Main
# ========================
if __name__ == "__main__":
    cfg = build_config_from_args()
    save_dir = ensure_dirs(cfg.output_dir)
    set_seed(cfg.seed)
    print("Using config:")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")
    (
        train_loader,
        val_loader,
        state_dim,
        action_dim,
        train_dataset,
        val_dataset,
        reward_mean,
        reward_std
    ) = load_hdf5_dataset(
        cfg.dataset_path,
        cfg.batch_size,
        cfg.val_ratio,
        cfg.seed,
        cfg.num_workers,
        pin_memory=(cfg.device == "cuda")
    )
    actor, critic1, critic2 = train_cql_improved(
        train_loader, val_loader, state_dim, action_dim, train_dataset, reward_mean, reward_std, cfg, save_dir
    )
    actor, obs_mean, obs_std = load_best_actor(cfg, save_dir)
```

**解释：**

- `import os`：导入依赖库，供后续数据读取、模型训练、优化、绘图或参数解析使用。
- `import json`：导入依赖库，供后续数据读取、模型训练、优化、绘图或参数解析使用。
- `import random`：导入依赖库，供后续数据读取、模型训练、优化、绘图或参数解析使用。
- `import h5py`：导入依赖库，供后续数据读取、模型训练、优化、绘图或参数解析使用。
- `import copy`：导入依赖库，供后续数据读取、模型训练、优化、绘图或参数解析使用。
- `import argparse`：导入依赖库，供后续数据读取、模型训练、优化、绘图或参数解析使用。
- `from dataclasses import dataclass, asdict, field`：导入依赖库，供后续数据读取、模型训练、优化、绘图或参数解析使用。
- `import torch`：导入依赖库，供后续数据读取、模型训练、优化、绘图或参数解析使用。
- `import torch.nn as nn`：导入依赖库，供后续数据读取、模型训练、优化、绘图或参数解析使用。
- `import torch.optim as optim`：导入依赖库，供后续数据读取、模型训练、优化、绘图或参数解析使用。
- `from torch.utils.data import TensorDataset, DataLoader, random_split`：导入依赖库，供后续数据读取、模型训练、优化、绘图或参数解析使用。
- `import numpy as np`：导入依赖库，供后续数据读取、模型训练、优化、绘图或参数解析使用。
- `import pandas as pd`：导入依赖库，供后续数据读取、模型训练、优化、绘图或参数解析使用。
- `import matplotlib.pyplot as plt`：导入依赖库，供后续数据读取、模型训练、优化、绘图或参数解析使用。
- `if __name__ == "__main__":`：判断当前文件是否被直接运行；只有直接运行时才启动 main。
- `cfg = build_config_from_args()`：读取命令行参数并生成配置对象 `cfg`，后面的训练、保存和评估都使用这份配置。
- `save_dir = ensure_dirs(cfg.output_dir)`：save_dir 的作用：模型、日志和图片的输出目录。
- `set_seed(cfg.seed)`：设置随机种子，让数据划分、初始化和训练过程尽量可复现。
- `print("Using config:")`：把当前训练或评估信息输出到终端。
- `for k, v in asdict(cfg).items():`：遍历配置对象中的每个参数，把本次实验配置打印出来。
- `print(f"  {k}: {v}")`：把当前训练或评估信息输出到终端。
- `load_hdf5_dataset(`：加载离线数据集，得到训练/验证 DataLoader、状态维度、动作维度以及原始状态动作数据。

### 函数 `ensure_dirs`
**代码：**

```python
def ensure_dirs(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    return output_dir
```

**解释：**

- `def ensure_dirs(output_dir):`：函数作用：封装一段可复用逻辑，供主训练流程调用。
- `return output_dir`：返回训练结果、数据和标准化参数，供主流程继续评估或加载模型。

### 函数 `set_seed`
**代码：**

```python
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

**解释：**

- `def set_seed(seed=42):`：函数作用：设置随机种子，让实验尽量可复现。

### 函数 `soft_update`
**代码：**

```python
def soft_update(source_net, target_net, tau):
    for source_param, target_param in zip(source_net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)
```

**解释：**

- `def soft_update(source_net, target_net, tau):`：函数作用：软更新目标网络，让目标网络缓慢跟随当前网络，稳定 TD 目标。

### 函数 `parse_hidden_layers`
**代码：**

```python
def parse_hidden_layers(value):
    if isinstance(value, list):
        result = []
        for v in value:
            result.extend([int(x) for x in str(v).split(",") if x.strip()])
        return result
    value = str(value).strip()
    if not value:
        return [256, 256]
    return [int(x.strip()) for x in value.split(",") if x.strip()]
```

**解释：**

- `def parse_hidden_layers(value):`：函数作用：把命令行里的隐藏层配置转换成整数列表，用来搭建神经网络。
- `return result`：返回训练结果、数据和标准化参数，供主流程继续评估或加载模型。
- `return [256, 256]`：返回训练结果、数据和标准化参数，供主流程继续评估或加载模型。
- `return [int(x.strip()) for x in value.split(",") if x.strip()]`：返回训练结果、数据和标准化参数，供主流程继续评估或加载模型。

### 类 `Config`
**代码：**

```python
@dataclass
class Config:
    dataset_path: str = "halfcheetah_medium-v2.hdf5"
    output_dir: str = os.path.join("checkpoints", "CQL")
    batch_size: int = 256
    num_epochs: int = 150
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    lambda_cql: float = 5.0
    hidden_layers: list = field(default_factory=lambda: [256, 256])
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    val_ratio: float = 0.1
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    num_workers: int = 0
    best_model_name: str = "best_model.pt"
    log_csv_name: str = "training_log.csv"
    critic_curve_name: str = "critic_loss_curve.png"
    actor_curve_name: str = "actor_loss_curve.png"
    cql_curve_name: str = "cql_penalty_curve.png"

    reward_scale: bool = True
    num_random_actions: int = 10
    alpha_bc: float = 2.5
    bc_coef: float = 1.0
```

**解释：**

- `@dataclass`：让配置类自动生成初始化方法，写超参数更方便。
- `class Config:`：配置类，集中保存数据路径、学习率、batch_size、网络结构、设备和输出目录等超参数。
- `output_dir: str = os.path.join("checkpoints", "CQL")`：output_dir 的作用：模型、日志和图片的输出目录。
- `batch_size: int = 256`：batch_size 的作用：每个 batch 的样本数。
- `num_epochs: int = 150`：num_epochs 的作用：训练轮数。
- `gamma: float = 0.99`：gamma 的作用：折扣因子，控制未来奖励在 TD 目标中的权重。
- `tau: float = 0.005`：tau 的作用：目标网络软更新系数。
- `hidden_layers: list = field(default_factory=lambda: [256, 256])`：hidden_layers 的作用：隐藏层宽度列表，用来决定 MLP 结构。
- `device: str = "cuda" if torch.cuda.is_available() else "cpu"`：device 的作用：计算设备，优先使用 GPU，否则使用 CPU。
- `seed: int = 42`：seed 的作用：随机种子，用于复现实验。
- `val_ratio: float = 0.1`：val_ratio 的作用：验证集比例。
- `weight_decay: float = 1e-4`：weight_decay 的作用：L2 正则化系数，用于减轻过拟合。
- `grad_clip: float = 5.0`：grad_clip 的作用：梯度裁剪阈值，防止梯度爆炸。
- `best_model_name: str = "best_model.pt"`：best_model_name 的作用：最佳模型 checkpoint 文件名。
- `reward_scale: bool = True`：reward_scale 的作用：是否对奖励做标准化或缩放。
- `bc_coef: float = 1.0`：bc_coef 的作用：BC 正则项权重，控制策略贴近离线动作的程度。

### 函数 `build_config_from_args`
**代码：**

```python
def build_config_from_args():
    parser = argparse.ArgumentParser(description="Improved CQL Training")
    parser.add_argument("--dataset-path", type=str, default="halfcheetah_medium-v2.hdf5")
    parser.add_argument("--output-dir", type=str, default=os.path.join("checkpoints", "CQL"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-epochs", type=int, default=120)
    parser.add_argument("--lr-actor", type=float, default=3e-4)
    parser.add_argument("--lr-critic", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--lambda-cql", type=float, default=5.0)
    parser.add_argument("--hidden-layers", type=str, default="256,256")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--best-model-name", type=str, default="best_model.pt")
    parser.add_argument("--log-csv-name", type=str, default="training_log.csv")
    parser.add_argument("--critic-curve-name", type=str, default="critic_loss_curve.png")
    parser.add_argument("--actor-curve-name", type=str, default="actor_loss_curve.png")
    parser.add_argument("--cql-curve-name", type=str, default="cql_penalty_curve.png")
    parser.add_argument("--num-random-actions", type=int, default=10)
    parser.add_argument("--alpha-bc", type=float, default=2.5)
    parser.add_argument("--bc-coef", type=float, default=1.0)
    parser.add_argument("--reward-scale", dest="reward_scale", action="store_true")
    parser.add_argument("--no-reward-scale", dest="reward_scale", action="store_false")
    parser.set_defaults(reward_scale=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--gpu", action="store_true", help="强制使用 GPU")
    group.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    args = parser.parse_args()
    if args.cpu:
        device = "cpu"
    elif args.gpu:
        if not torch.cuda.is_available():
            raise RuntimeError("你指定了 --gpu，但当前环境没有可用的 CUDA/GPU。")
        device = "cuda"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Config(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        gamma=args.gamma,
        tau=args.tau,
        lambda_cql=args.lambda_cql,
        hidden_layers=parse_hidden_layers(args.hidden_layers),
        device=device,
        seed=args.seed,
        val_ratio=args.val_ratio,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        num_workers=args.num_workers,
        best_model_name=args.best_model_name,
        log_csv_name=args.log_csv_name,
        critic_curve_name=args.critic_curve_name,
        actor_curve_name=args.actor_curve_name,
        cql_curve_name=args.cql_curve_name,
        reward_scale=args.reward_scale,
        num_random_actions=args.num_random_actions,
        alpha_bc=args.alpha_bc,
        bc_coef=args.bc_coef,
    )
    ensure_dirs(cfg.output_dir)
    save_cfg = asdict(cfg)
    with open(os.path.join(cfg.output_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(save_cfg, f, ensure_ascii=False, indent=2)
    return cfg
```

**解释：**

- `def build_config_from_args():`：函数作用：解析命令行参数，生成统一配置对象。
- `device = "cpu"`：device 的作用：计算设备，优先使用 GPU，否则使用 CPU。
- `device = "cuda"`：device 的作用：计算设备，优先使用 GPU，否则使用 CPU。
- `device = "cuda" if torch.cuda.is_available() else "cpu"`：device 的作用：计算设备，优先使用 GPU，否则使用 CPU。
- `output_dir=args.output_dir,`：output_dir 的作用：模型、日志和图片的输出目录。
- `batch_size=args.batch_size,`：batch_size 的作用：每个 batch 的样本数。
- `num_epochs=args.num_epochs,`：num_epochs 的作用：训练轮数。
- `lr_actor=args.lr_actor,`：创建策略网络。它输入状态，输出动作，是后续训练要优化的核心模型。
- `gamma=args.gamma,`：gamma 的作用：折扣因子，控制未来奖励在 TD 目标中的权重。
- `tau=args.tau,`：tau 的作用：目标网络软更新系数。
- `hidden_layers=parse_hidden_layers(args.hidden_layers),`：hidden_layers 的作用：隐藏层宽度列表，用来决定 MLP 结构。
- `device=device,`：device 的作用：计算设备，优先使用 GPU，否则使用 CPU。
- `seed=args.seed,`：seed 的作用：随机种子，用于复现实验。
- `val_ratio=args.val_ratio,`：val_ratio 的作用：验证集比例。
- `grad_clip=args.grad_clip,`：grad_clip 的作用：梯度裁剪阈值，防止梯度爆炸。
- `best_model_name=args.best_model_name,`：best_model_name 的作用：最佳模型 checkpoint 文件名。
- `reward_scale=args.reward_scale,`：reward_scale 的作用：是否对奖励做标准化或缩放。
- `bc_coef=args.bc_coef,`：bc_coef 的作用：BC 正则项权重，控制策略贴近离线动作的程度。
- `return cfg`：返回训练结果、数据和标准化参数，供主流程继续评估或加载模型。

### 类 `MLP`
**代码：**

```python
class MLP(nn.Module):
```

**解释：**

- `class MLP(nn.Module):`：通用多层感知机模块，用 Linear 和 ReLU 组成 actor 或 critic 的基础网络。

### 方法 `MLP.__init__`
**代码：**

```python
    def __init__(self, input_dim, output_dim, hidden_sizes=(256, 256), use_tanh=False):
        super().__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(last_dim, output_dim)
        self.use_tanh = use_tanh
```

**解释：**

- `def __init__(self, input_dim, output_dim, hidden_sizes=(256, 256), use_tanh=False):`：函数作用：初始化当前类，创建网络层、优化器或训练需要的成员变量。

### 方法 `MLP.forward`
**代码：**

```python
    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        if self.use_tanh:
            x = torch.tanh(x)
        return x
```

**解释：**

- `def forward(self, x):`：函数作用：定义网络前向传播，说明输入张量如何变成输出。
- `return x`：返回训练结果、数据和标准化参数，供主流程继续评估或加载模型。

### 函数 `load_hdf5_dataset`
**代码：**

```python
def load_hdf5_dataset(path, batch_size=256, val_ratio=0.1, seed=42, num_workers=0, pin_memory=False):
    with h5py.File(path, "r") as f:
        obs = torch.tensor(f["observations"][:], dtype=torch.float32)
        actions = torch.tensor(f["actions"][:], dtype=torch.float32)
        rewards = torch.tensor(f["rewards"][:], dtype=torch.float32)
        next_obs = torch.tensor(f["next_observations"][:], dtype=torch.float32)
        terminals = torch.tensor(f["terminals"][:], dtype=torch.float32)
    dataset = TensorDataset(obs, actions, rewards, next_obs, terminals)
    total_size = len(dataset)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    state_dim = obs.shape[1]
    action_dim = actions.shape[1]
    reward_mean = rewards.mean()
    reward_std = rewards.std().clamp(min=1e-6)
    return (
        train_loader,
        val_loader,
        state_dim,
        action_dim,
        train_dataset,
        val_dataset,
        reward_mean,
        reward_std
    )
```

**解释：**

- `def load_hdf5_dataset(path, batch_size=256, val_ratio=0.1, seed=42, num_workers=0, pin_memory=False):`：函数作用：读取离线 HDF5 数据集，并构造训练集、验证集或 DataLoader。
- `batch_size=batch_size,`：batch_size 的作用：每个 batch 的样本数。
- `state_dim = obs.shape[1]`：state_dim 的作用：状态维度，本项目通常是 17。
- `action_dim = actions.shape[1]`：action_dim 的作用：动作维度，本项目通常是 6。
- `return (`：返回训练结果、数据和标准化参数，供主流程继续评估或加载模型。

### 函数 `compute_obs_stats`
**代码：**

```python
def compute_obs_stats(train_dataset):
    obs_list = []
    for item in train_dataset:
        obs_list.append(item[0].unsqueeze(0))
    obs_tensor = torch.cat(obs_list, dim=0)
    obs_mean = obs_tensor.mean(dim=0)
    obs_std = obs_tensor.std(dim=0).clamp(min=1e-6)
    return obs_mean, obs_std
```

**解释：**

- `def compute_obs_stats(train_dataset):`：函数作用：统计训练集状态均值和标准差，后续用于状态标准化。
- `obs_mean = obs_tensor.mean(dim=0)`：obs_mean 的作用：训练集 17 个状态维度的均值，用于状态标准化。
- `obs_std = obs_tensor.std(dim=0).clamp(min=1e-6)`：obs_std 的作用：训练集 17 个状态维度的标准差，用于状态标准化。
- `return obs_mean, obs_std`：返回训练结果、数据和标准化参数，供主流程继续评估或加载模型。

### 函数 `compute_cql_penalty`
**代码：**

```python
def compute_cql_penalty(critic, actor, s, a_data, num_random_actions=10):
    batch_size, action_dim = a_data.shape
    random_actions = torch.empty(
        batch_size, num_random_actions, action_dim, device=s.device
    ).uniform_(-1.0, 1.0)
    s_repeat = s.unsqueeze(1).repeat(1, num_random_actions, 1)
    q_rand = critic(
        torch.cat([s_repeat, random_actions], dim=-1).view(batch_size * num_random_actions, -1)
    ).view(batch_size, num_random_actions)
    pi_action = actor(s)
    q_pi = critic(torch.cat([s, pi_action], dim=1))
    q_data = critic(torch.cat([s, a_data], dim=1))
    cat_q = torch.cat([q_rand, q_pi], dim=1)
    penalty = torch.logsumexp(cat_q, dim=1, keepdim=True).mean() - q_data.mean()
    return penalty
```

**解释：**

- `def compute_cql_penalty(critic, actor, s, a_data, num_random_actions=10):`：函数作用：计算 CQL 保守惩罚项，压低数据外动作的过高 Q 值。
- `q_pi = critic(torch.cat([s, pi_action], dim=1))`：用 critic 评价当前策略动作的 Q 值；actor 更新会希望这个值更大。
- `return penalty`：返回训练结果、数据和标准化参数，供主流程继续评估或加载模型。

### 函数 `evaluate_cql`
**代码：**

```python
def evaluate_cql(
    actor,
    critic1,
    critic2,
    target_critic1,
    target_critic2,
    loader,
    obs_mean,
    obs_std,
    reward_mean,
    reward_std,
    args
):
    device = args.device
    actor.eval()
    critic1.eval()
    critic2.eval()
    target_critic1.eval()
    target_critic2.eval()
    total_reward = 0.0
    total_actor_loss = 0.0
    total_critic_loss = 0.0
    total_cql_penalty = 0.0
    total_count = 0
    mse_q1_total = 0.0
    mse_q2_total = 0.0
    bc_total = 0.0
    with torch.no_grad():
        for batch in loader:
            s, a, r, s_next, done = [x.to(device, non_blocking=True) for x in batch]
            s = (s - obs_mean) / obs_std
            s_next = (s_next - obs_mean) / obs_std
            if args.reward_scale:
                r = (r - reward_mean) / reward_std
            next_a = actor(s_next)
            q_target = r.unsqueeze(1) + args.gamma * (1.0 - done.unsqueeze(1)) * torch.min(
                target_critic1(torch.cat([s_next, next_a], dim=1)),
                target_critic2(torch.cat([s_next, next_a], dim=1))
            )
            q1_pred = critic1(torch.cat([s, a], dim=1))
            q2_pred = critic2(torch.cat([s, a], dim=1))
            mse_q1 = nn.functional.mse_loss(q1_pred, q_target)
            mse_q2 = nn.functional.mse_loss(q2_pred, q_target)
            critic_mse = mse_q1 + mse_q2
            cql1 = compute_cql_penalty(critic1, actor, s, a, args.num_random_actions)
            cql2 = compute_cql_penalty(critic2, actor, s, a, args.num_random_actions)
            cql_penalty = args.lambda_cql * (cql1 + cql2)
            total_critic = critic_mse + cql_penalty
            pred_a = actor(s)
            q_pi = critic1(torch.cat([s, pred_a], dim=1))
            lam = args.alpha_bc / (q_pi.abs().mean().detach() + 1e-6)
            bc_loss = nn.functional.mse_loss(pred_a, a)
            actor_loss = -lam * q_pi.mean() + args.bc_coef * bc_loss
            batch_size = s.size(0)
            total_reward += r.mean().item() * batch_size
            total_actor_loss += actor_loss.item() * batch_size
            total_critic_loss += total_critic.item() * batch_size
            total_cql_penalty += cql_penalty.item() * batch_size
            mse_q1_total += mse_q1.item() * batch_size
            mse_q2_total += mse_q2.item() * batch_size
            bc_total += bc_loss.item() * batch_size
            total_count += batch_size
    return {
        "reward": total_reward / total_count,
        "actor_loss": total_actor_loss / total_count,
        "critic_loss": total_critic_loss / total_count,
        "cql_penalty": total_cql_penalty / total_count,
        "q1_mse": mse_q1_total / total_count,
        "q2_mse": mse_q2_total / total_count,
        "bc_loss": bc_total / total_count,
    }
```

**解释：**

- `def evaluate_cql(`：函数作用：封装一段可复用逻辑，供主训练流程调用。
- `device = args.device`：device 的作用：计算设备，优先使用 GPU，否则使用 CPU。
- `total_critic_loss = 0.0`：计算 critic 的 TD 回归损失，让当前 Q 值逼近 TD 目标。
- `with torch.no_grad():`：进入不记录梯度的区域，通常用于计算 TD 目标或验证指标。
- `s, a, r, s_next, done = [x.to(device, non_blocking=True) for x in batch]`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `if args.reward_scale:`：如果启用奖励标准化/缩放，就对 reward 做归一化处理。
- `q_pi = critic1(torch.cat([s, pred_a], dim=1))`：用 critic 评价当前策略动作的 Q 值；actor 更新会希望这个值更大。
- `lam = args.alpha_bc / (q_pi.abs().mean().detach() + 1e-6)`：计算 Q 项的缩放系数，用来平衡 Q 最大化和 BC 正则。
- `bc_loss = nn.functional.mse_loss(pred_a, a)`：BC 正则项，惩罚策略动作偏离离线数据动作，降低离线策略外推风险。
- `actor_loss = -lam * q_pi.mean() + args.bc_coef * bc_loss`：BC 正则项，惩罚策略动作偏离离线数据动作，降低离线策略外推风险。
- `batch_size = s.size(0)`：batch_size 的作用：每个 batch 的样本数。
- `bc_total += bc_loss.item() * batch_size`：BC 正则项，惩罚策略动作偏离离线数据动作，降低离线策略外推风险。
- `return {`：返回本次更新或评估的指标字典，供训练日志记录。

### 函数 `train_cql_improved`
**代码：**

```python
def train_cql_improved(train_loader, val_loader, state_dim, action_dim, train_dataset, reward_mean, reward_std, args, save_dir):
    device = args.device
    obs_mean, obs_std = compute_obs_stats(train_dataset)
    obs_mean = obs_mean.to(device)
    obs_std = obs_std.to(device)
    reward_mean = reward_mean.to(device)
    reward_std = reward_std.to(device)
    actor = MLP(state_dim, action_dim, args.hidden_layers, use_tanh=True).to(device)
    critic1 = MLP(state_dim + action_dim, 1, args.hidden_layers).to(device)
    critic2 = MLP(state_dim + action_dim, 1, args.hidden_layers).to(device)
    target_critic1 = copy.deepcopy(critic1).to(device)
    target_critic2 = copy.deepcopy(critic2).to(device)
    actor_optimizer = optim.Adam(
        actor.parameters(),
        lr=args.lr_actor,
        weight_decay=args.weight_decay
    )
    critic_optimizer = optim.Adam(
        list(critic1.parameters()) + list(critic2.parameters()),
        lr=args.lr_critic,
        weight_decay=args.weight_decay
    )
    actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(actor_optimizer, T_max=args.num_epochs)
    critic_scheduler = optim.lr_scheduler.CosineAnnealingLR(critic_optimizer, T_max=args.num_epochs)
    logs = []
    best_val_critic_loss = float("inf")
    best_model_path = os.path.join(save_dir, args.best_model_name)
    for epoch in range(args.num_epochs):
        actor.train()
        critic1.train()
        critic2.train()
        epoch_reward = []
        epoch_actor_loss = []
        epoch_critic_loss = []
        epoch_cql_penalty = []
        epoch_q1_mse = []
        epoch_q2_mse = []
        epoch_bc_loss = []
        for batch in train_loader:
            s, a, r, s_next, done = [x.to(device, non_blocking=True) for x in batch]
            s = (s - obs_mean) / obs_std
            s_next = (s_next - obs_mean) / obs_std
            if args.reward_scale:
                r = (r - reward_mean) / reward_std
            with torch.no_grad():
                next_a = actor(s_next)
                q_target = r.unsqueeze(1) + args.gamma * (1.0 - done.unsqueeze(1)) * torch.min(
                    target_critic1(torch.cat([s_next, next_a], dim=1)),
                    target_critic2(torch.cat([s_next, next_a], dim=1))
                )
            q1_pred = critic1(torch.cat([s, a], dim=1))
            q2_pred = critic2(torch.cat([s, a], dim=1))
            mse_q1 = nn.functional.mse_loss(q1_pred, q_target)
            mse_q2 = nn.functional.mse_loss(q2_pred, q_target)
            critic_mse = mse_q1 + mse_q2
            cql1 = compute_cql_penalty(critic1, actor, s, a, args.num_random_actions)
            cql2 = compute_cql_penalty(critic2, actor, s, a, args.num_random_actions)
            cql_penalty = args.lambda_cql * (cql1 + cql2)
            total_critic_loss = critic_mse + cql_penalty
            critic_optimizer.zero_grad()
            total_critic_loss.backward()
            nn.utils.clip_grad_norm_(list(critic1.parameters()) + list(critic2.parameters()), args.grad_clip)
            critic_optimizer.step()
            pred_a = actor(s)
            q_pi = critic1(torch.cat([s, pred_a], dim=1))
            lam = args.alpha_bc / (q_pi.abs().mean().detach() + 1e-6)
            bc_loss = nn.functional.mse_loss(pred_a, a)
            actor_loss = -lam * q_pi.mean() + args.bc_coef * bc_loss
            actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), args.grad_clip)
            actor_optimizer.step()
            soft_update(critic1, target_critic1, args.tau)
            soft_update(critic2, target_critic2, args.tau)
            epoch_reward.append(r.mean().item())
            epoch_actor_loss.append(actor_loss.item())
            epoch_critic_loss.append(total_critic_loss.item())
            epoch_cql_penalty.append(cql_penalty.item())
            epoch_q1_mse.append(mse_q1.item())
            epoch_q2_mse.append(mse_q2.item())
            epoch_bc_loss.append(bc_loss.item())
        train_metrics = {
            "reward": np.mean(epoch_reward),
            "actor_loss": np.mean(epoch_actor_loss),
            "critic_loss": np.mean(epoch_critic_loss),
            "cql_penalty": np.mean(epoch_cql_penalty),
            "q1_mse": np.mean(epoch_q1_mse),
            "q2_mse": np.mean(epoch_q2_mse),
            "bc_loss": np.mean(epoch_bc_loss),
        }
        val_metrics = evaluate_cql(
            actor, critic1, critic2, target_critic1, target_critic2,
            val_loader, obs_mean, obs_std, reward_mean, reward_std, args
        )
        current_actor_lr = actor_optimizer.param_groups[0]["lr"]
        current_critic_lr = critic_optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1}/{args.num_epochs} | "
            f"Train Critic Loss: {train_metrics['critic_loss']:.6f} | "
            f"Val Critic Loss: {val_metrics['critic_loss']:.6f} | "
            f"Train Actor Loss: {train_metrics['actor_loss']:.6f} | "
            f"Val Actor Loss: {val_metrics['actor_loss']:.6f}"
        )
        logs.append({
            "epoch": epoch + 1,
            "train_reward": train_metrics["reward"],
            "val_reward": val_metrics["reward"],
            "train_actor_loss": train_metrics["actor_loss"],
            "val_actor_loss": val_metrics["actor_loss"],
            "train_critic_loss": train_metrics["critic_loss"],
            "val_critic_loss": val_metrics["critic_loss"],
            "train_cql_penalty": train_metrics["cql_penalty"],
            "val_cql_penalty": val_metrics["cql_penalty"],
            "train_q1_mse": train_metrics["q1_mse"],
            "train_q2_mse": train_metrics["q2_mse"],
            "val_q1_mse": val_metrics["q1_mse"],
            "val_q2_mse": val_metrics["q2_mse"],
            "train_bc_loss": train_metrics["bc_loss"],
            "val_bc_loss": val_metrics["bc_loss"],
            "actor_lr": current_actor_lr,
            "critic_lr": current_critic_lr
        })
        if val_metrics["critic_loss"] < best_val_critic_loss:
            best_val_critic_loss = val_metrics["critic_loss"]
            torch.save({
                "actor_state_dict": actor.state_dict(),
                "critic1_state_dict": critic1.state_dict(),
                "critic2_state_dict": critic2.state_dict(),
                "obs_mean": obs_mean.cpu(),
                "obs_std": obs_std.cpu(),
                "reward_mean": reward_mean.cpu(),
                "reward_std": reward_std.cpu(),
                "state_dim": state_dim,
                "action_dim": action_dim,
                "hidden_layers": list(args.hidden_layers),
                "epoch": epoch + 1,
                "best_val_critic_loss": best_val_critic_loss,
                "lambda_cql": args.lambda_cql,
                "alpha_bc": args.alpha_bc,
                "bc_coef": args.bc_coef,
                "reward_scale": args.reward_scale
            }, best_model_path)
            print(f"Best model saved at epoch {epoch+1}, val_critic_loss={best_val_critic_loss:.6f}")
        actor_scheduler.step()
        critic_scheduler.step()
    df = pd.DataFrame(logs)
    df.to_csv(os.path.join(save_dir, args.log_csv_name), index=False)
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_critic_loss"], label="Train Critic Loss")
    plt.plot(df["epoch"], df["val_critic_loss"], label="Val Critic Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CQL Critic Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, args.critic_curve_name))
    plt.close()
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_actor_loss"], label="Train Actor Loss")
    plt.plot(df["epoch"], df["val_actor_loss"], label="Val Actor Loss")
    plt.plot(df["epoch"], df["train_bc_loss"], label="Train BC Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CQL Actor Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, args.actor_curve_name))
    plt.close()
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_cql_penalty"], label="Train CQL Penalty")
    plt.plot(df["epoch"], df["val_cql_penalty"], label="Val CQL Penalty")
    plt.xlabel("Epoch")
    plt.ylabel("Penalty")
    plt.title("CQL Penalty")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, args.cql_curve_name))
    plt.close()
    return actor, critic1, critic2
```

**解释：**

- `def train_cql_improved(train_loader, val_loader, state_dim, action_dim, train_dataset, reward_mean, reward_std, args,...`：函数作用：封装一段可复用逻辑，供主训练流程调用。
- `device = args.device`：device 的作用：计算设备，优先使用 GPU，否则使用 CPU。
- `obs_mean, obs_std = compute_obs_stats(train_dataset)`：只用训练集计算状态均值和标准差，后面用它们对状态做标准化。
- `obs_mean = obs_mean.to(device)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `obs_std = obs_std.to(device)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `reward_mean = reward_mean.to(device)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `reward_std = reward_std.to(device)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `actor = MLP(state_dim, action_dim, args.hidden_layers, use_tanh=True).to(device)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `critic1 = MLP(state_dim + action_dim, 1, args.hidden_layers).to(device)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `critic2 = MLP(state_dim + action_dim, 1, args.hidden_layers).to(device)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `target_critic1 = copy.deepcopy(critic1).to(device)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `target_critic2 = copy.deepcopy(critic2).to(device)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `actor_optimizer = optim.Adam(`：创建 Adam 优化器，后面根据梯度更新网络参数。
- `critic_optimizer = optim.Adam(`：创建 Adam 优化器，后面根据梯度更新网络参数。
- `actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(actor_optimizer, T_max=args.num_epochs)`：创建学习率调度器，让学习率随训练轮数变化，后期训练更稳定。
- `critic_scheduler = optim.lr_scheduler.CosineAnnealingLR(critic_optimizer, T_max=args.num_epochs)`：创建学习率调度器，让学习率随训练轮数变化，后期训练更稳定。
- `best_val_critic_loss = float("inf")`：计算 critic 的 TD 回归损失，让当前 Q 值逼近 TD 目标。
- `best_model_path = os.path.join(save_dir, args.best_model_name)`：初始化最佳模型保存路径，真正保存模型后会更新它。
- `for epoch in range(args.num_epochs):`：开始按 epoch 训练模型，每轮都会训练一次、验证一次，并记录指标。
- `epoch_critic_loss = []`：计算 critic 的 TD 回归损失，让当前 Q 值逼近 TD 目标。
- `epoch_bc_loss = []`：BC 正则项，惩罚策略动作偏离离线数据动作，降低离线策略外推风险。
- `s, a, r, s_next, done = [x.to(device, non_blocking=True) for x in batch]`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `if args.reward_scale:`：如果启用奖励标准化/缩放，就对 reward 做归一化处理。
- `with torch.no_grad():`：进入不记录梯度的区域，通常用于计算 TD 目标或验证指标。
- `total_critic_loss = critic_mse + cql_penalty`：计算 critic 的 TD 回归损失，让当前 Q 值逼近 TD 目标。
- `critic_optimizer.zero_grad()`：清空上一轮梯度，避免梯度累积影响当前 batch。
- `total_critic_loss.backward()`：对当前损失反向传播，计算网络参数梯度。
- `nn.utils.clip_grad_norm_(list(critic1.parameters()) + list(critic2.parameters()), args.grad_clip)`：对梯度做裁剪，防止梯度过大导致训练不稳定。
- `critic_optimizer.step()`：优化器根据梯度更新参数。
- `q_pi = critic1(torch.cat([s, pred_a], dim=1))`：用 critic 评价当前策略动作的 Q 值；actor 更新会希望这个值更大。
- `lam = args.alpha_bc / (q_pi.abs().mean().detach() + 1e-6)`：计算 Q 项的缩放系数，用来平衡 Q 最大化和 BC 正则。
- `bc_loss = nn.functional.mse_loss(pred_a, a)`：BC 正则项，惩罚策略动作偏离离线数据动作，降低离线策略外推风险。
- `actor_loss = -lam * q_pi.mean() + args.bc_coef * bc_loss`：BC 正则项，惩罚策略动作偏离离线数据动作，降低离线策略外推风险。
- `actor_optimizer.zero_grad()`：清空上一轮梯度，避免梯度累积影响当前 batch。
- `actor_loss.backward()`：对当前损失反向传播，计算网络参数梯度。
- `nn.utils.clip_grad_norm_(actor.parameters(), args.grad_clip)`：对梯度做裁剪，防止梯度过大导致训练不稳定。
- `actor_optimizer.step()`：优化器根据梯度更新参数。
- `soft_update(critic1, target_critic1, args.tau)`：软更新目标 critic，让目标 Q 网络缓慢跟随当前 critic。
- `soft_update(critic2, target_critic2, args.tau)`：软更新目标 critic，让目标 Q 网络缓慢跟随当前 critic。
- `epoch_bc_loss.append(bc_loss.item())`：BC 正则项，惩罚策略动作偏离离线数据动作，降低离线策略外推风险。
- `print(`：把当前训练或评估信息输出到终端。
- `if val_metrics["critic_loss"] < best_val_critic_loss:`：如果当前验证结果优于历史最好结果，就保存当前模型。
- `best_val_critic_loss = val_metrics["critic_loss"]`：计算 critic 的 TD 回归损失，让当前 Q 值逼近 TD 目标。
- `print(f"Best model saved at epoch {epoch+1}, val_critic_loss={best_val_critic_loss:.6f}")`：计算 critic 的 TD 回归损失，让当前 Q 值逼近 TD 目标。
- `actor_scheduler.step()`：优化器根据梯度更新参数。
- `critic_scheduler.step()`：优化器根据梯度更新参数。
- `df = pd.DataFrame(logs)`：把训练记录转换成 pandas 表格，便于保存 CSV 和画曲线。
- `df.to_csv(os.path.join(save_dir, args.log_csv_name), index=False)`：把表格结果保存成 CSV 文件，方便论文记录、画图或复查。
- `plt.plot(df["epoch"], df["train_bc_loss"], label="Train BC Loss")`：BC 正则项，惩罚策略动作偏离离线数据动作，降低离线策略外推风险。
- `return actor, critic1, critic2`：返回训练结果、数据和标准化参数，供主流程继续评估或加载模型。

### 函数 `load_best_actor`
**代码：**

```python
def load_best_actor(args, save_dir):
    ckpt = torch.load(os.path.join(save_dir, args.best_model_name), map_location=args.device)
    actor = MLP(
        ckpt["state_dim"],
        ckpt["action_dim"],
        ckpt["hidden_layers"],
        use_tanh=True
    ).to(args.device)
    actor.load_state_dict(ckpt["actor_state_dict"])
    obs_mean = ckpt["obs_mean"].to(args.device)
    obs_std = ckpt["obs_std"].to(args.device)
    print(
        f"Loaded best actor from epoch {ckpt['epoch']}, "
        f"best_val_critic_loss={ckpt['best_val_critic_loss']:.6f}"
    )
    return actor, obs_mean, obs_std
```

**解释：**

- `def load_best_actor(args, save_dir):`：函数作用：加载训练过程中保存的最佳策略模型，用于评估或推理。
- `actor = MLP(`：创建策略网络。它输入状态，输出动作，是后续训练要优化的核心模型。
- `).to(args.device)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `obs_mean = ckpt["obs_mean"].to(args.device)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `obs_std = ckpt["obs_std"].to(args.device)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `print(`：把当前训练或评估信息输出到终端。
- `return actor, obs_mean, obs_std`：返回训练结果、数据和标准化参数，供主流程继续评估或加载模型。


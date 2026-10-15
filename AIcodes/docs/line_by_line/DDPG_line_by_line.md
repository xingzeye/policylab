# DDPG.py 逐行代码解释附录

这是 [`DDPG.md`](../DDPG.md) 的细节附录，保留对应源码每一行非空代码的解释。主文档负责理解流程；这里适合查某一行具体在做什么。

## 逐行代码解释
下面按源码分块展示代码：每个小节先给出干净源码，再用简洁中文解释这段代码的作用、关键变量含义和训练逻辑。

### 模块级代码、导入和入口
**代码：**

```python
import os
import json
import random
import argparse
from dataclasses import dataclass, asdict, field
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# =====================
# 0. 工具函数
# =====================
# =====================
# 1. 参数配置
# =====================
# =====================
# 2. Actor & Critic 网络
# =====================
# =====================
# 3. Offline Dataset
# =====================
# =====================
# 4. 计算训练集状态统计量
# =====================
# =====================
# 5. Advantage-Weighted DDPG_NB Agent
# =====================
# =====================
# 6. 训练与验证
# =====================
# =====================
# 7. 加载最优模型
# =====================
# =====================
# 8. Main
# =====================
if __name__ == "__main__":
    main()
```

**解释：**

- `import os`：导入依赖库，供后续数据读取、模型训练、优化、绘图或参数解析使用。
- `import json`：导入依赖库，供后续数据读取、模型训练、优化、绘图或参数解析使用。
- `import random`：导入依赖库，供后续数据读取、模型训练、优化、绘图或参数解析使用。
- `import argparse`：导入依赖库，供后续数据读取、模型训练、优化、绘图或参数解析使用。
- `from dataclasses import dataclass, asdict, field`：导入依赖库，供后续数据读取、模型训练、优化、绘图或参数解析使用。
- `import h5py`：导入依赖库，供后续数据读取、模型训练、优化、绘图或参数解析使用。
- `import torch`：导入依赖库，供后续数据读取、模型训练、优化、绘图或参数解析使用。
- `import torch.nn as nn`：导入依赖库，供后续数据读取、模型训练、优化、绘图或参数解析使用。
- `import torch.optim as optim`：导入依赖库，供后续数据读取、模型训练、优化、绘图或参数解析使用。
- `from torch.utils.data import DataLoader, random_split`：导入依赖库，供后续数据读取、模型训练、优化、绘图或参数解析使用。
- `import numpy as np`：导入依赖库，供后续数据读取、模型训练、优化、绘图或参数解析使用。
- `import pandas as pd`：导入依赖库，供后续数据读取、模型训练、优化、绘图或参数解析使用。
- `import matplotlib.pyplot as plt`：导入依赖库，供后续数据读取、模型训练、优化、绘图或参数解析使用。
- `if __name__ == "__main__":`：判断当前文件是否被直接运行；只有直接运行时才启动 main。
- `main()`：调用 main 函数，正式启动整个脚本。

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

### 类 `Config`
**代码：**

```python
@dataclass
class Config:
    hdf5_file: str = "halfcheetah_medium-v2.hdf5"
    output_dir: str = "checkpoints/DDPG"
    batch_size: int = 256
    num_epochs: int = 120
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    hidden: list = field(default_factory=lambda: [256, 256])
    seed: int = 42
    val_ratio: float = 0.1
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    reward_scale: bool = True
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    aw_lambda: float = 1.0
    weight_clip: float = 20.0
    best_model_name: str = "best_model.pt"
    records_csv_name: str = "training_records.csv"
    actor_curve_name: str = "actor_loss_curve.png"
    critic_curve_name: str = "critic_loss_curve.png"
    weight_curve_name: str = "weight_curve.png"
```

**解释：**

- `@dataclass`：让配置类自动生成初始化方法，写超参数更方便。
- `class Config:`：配置类，集中保存数据路径、学习率、batch_size、网络结构、设备和输出目录等超参数。
- `output_dir: str = "checkpoints/DDPG"`：output_dir 的作用：模型、日志和图片的输出目录。
- `batch_size: int = 256`：batch_size 的作用：每个 batch 的样本数。
- `num_epochs: int = 120`：num_epochs 的作用：训练轮数。
- `gamma: float = 0.99`：gamma 的作用：折扣因子，控制未来奖励在 TD 目标中的权重。
- `tau: float = 0.005`：tau 的作用：目标网络软更新系数。
- `seed: int = 42`：seed 的作用：随机种子，用于复现实验。
- `val_ratio: float = 0.1`：val_ratio 的作用：验证集比例。
- `weight_decay: float = 1e-4`：weight_decay 的作用：L2 正则化系数，用于减轻过拟合。
- `grad_clip: float = 5.0`：grad_clip 的作用：梯度裁剪阈值，防止梯度爆炸。
- `device: str = "cuda" if torch.cuda.is_available() else "cpu"`：device 的作用：计算设备，优先使用 GPU，否则使用 CPU。
- `reward_scale: bool = True`：reward_scale 的作用：是否对奖励做标准化或缩放。
- `policy_noise: float = 0.2`：policy_noise 的作用：TD3 目标动作平滑噪声强度。
- `noise_clip: float = 0.5`：noise_clip 的作用：目标动作噪声裁剪范围。
- `policy_delay: int = 2`：policy_delay 的作用：TD3 中 actor 延迟更新间隔。
- `weight_clip: float = 20.0`：weight_clip 的作用：优势权重裁剪上限，避免少数样本权重过大。
- `best_model_name: str = "best_model.pt"`：best_model_name 的作用：最佳模型 checkpoint 文件名。

### 函数 `build_config_from_args`
**代码：**

```python
def build_config_from_args():
    parser = argparse.ArgumentParser(description="Offline AW-DDPG_NB Training")
    parser.add_argument("--hdf5-file", type=str, default="halfcheetah_medium-v2.hdf5")
    parser.add_argument("--output-dir", type=str, default="checkpoints/DDPG")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-epochs", type=int, default=150)
    parser.add_argument("--lr-actor", type=float, default=3e-4)
    parser.add_argument("--lr-critic", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--hidden", type=str, default="256,256")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--policy-noise", type=float, default=0.2)
    parser.add_argument("--noise-clip", type=float, default=0.5)
    parser.add_argument("--policy-delay", type=int, default=2)
    parser.add_argument("--aw-lambda", type=float, default=1.0)
    parser.add_argument("--weight-clip", type=float, default=20.0)
    parser.add_argument("--best-model-name", type=str, default="best_model.pt")
    parser.add_argument("--records-csv-name", type=str, default="training_records.csv")
    parser.add_argument("--actor-curve-name", type=str, default="actor_loss_curve.png")
    parser.add_argument("--critic-curve-name", type=str, default="critic_loss_curve.png")
    parser.add_argument("--weight-curve-name", type=str, default="weight_curve.png")
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
        hdf5_file=args.hdf5_file,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        gamma=args.gamma,
        tau=args.tau,
        hidden=parse_hidden_layers(args.hidden),
        seed=args.seed,
        val_ratio=args.val_ratio,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        num_workers=args.num_workers,
        device=device,
        reward_scale=args.reward_scale,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        policy_delay=args.policy_delay,
        aw_lambda=args.aw_lambda,
        weight_clip=args.weight_clip,
        best_model_name=args.best_model_name,
        records_csv_name=args.records_csv_name,
        actor_curve_name=args.actor_curve_name,
        critic_curve_name=args.critic_curve_name,
        weight_curve_name=args.weight_curve_name,
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
- `seed=args.seed,`：seed 的作用：随机种子，用于复现实验。
- `val_ratio=args.val_ratio,`：val_ratio 的作用：验证集比例。
- `grad_clip=args.grad_clip,`：grad_clip 的作用：梯度裁剪阈值，防止梯度爆炸。
- `device=device,`：device 的作用：计算设备，优先使用 GPU，否则使用 CPU。
- `reward_scale=args.reward_scale,`：reward_scale 的作用：是否对奖励做标准化或缩放。
- `policy_noise=args.policy_noise,`：policy_noise 的作用：TD3 目标动作平滑噪声强度。
- `noise_clip=args.noise_clip,`：noise_clip 的作用：目标动作噪声裁剪范围。
- `policy_delay=args.policy_delay,`：policy_delay 的作用：TD3 中 actor 延迟更新间隔。
- `weight_clip=args.weight_clip,`：weight_clip 的作用：优势权重裁剪上限，避免少数样本权重过大。
- `best_model_name=args.best_model_name,`：best_model_name 的作用：最佳模型 checkpoint 文件名。
- `return cfg`：返回训练结果、数据和标准化参数，供主流程继续评估或加载模型。

### 类 `Actor`
**代码：**

```python
class Actor(nn.Module):
```

**解释：**

- `class Actor(nn.Module):`：Actor 是策略网络，输入状态，输出连续动作。

### 方法 `Actor.__init__`
**代码：**

```python
    def __init__(self, state_dim, action_dim, hidden=(256, 256)):
        super().__init__()
        layers = []
        input_dim = state_dim
        for h in hidden:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, action_dim))
        layers.append(nn.Tanh())
        self.model = nn.Sequential(*layers)
```

**解释：**

- `def __init__(self, state_dim, action_dim, hidden=(256, 256)):`：函数作用：初始化当前类，创建网络层、优化器或训练需要的成员变量。

### 方法 `Actor.forward`
**代码：**

```python
    def forward(self, state):
        return self.model(state)
```

**解释：**

- `def forward(self, state):`：函数作用：定义网络前向传播，说明输入张量如何变成输出。
- `return self.model(state)`：返回训练结果、数据和标准化参数，供主流程继续评估或加载模型。

### 类 `Critic`
**代码：**

```python
class Critic(nn.Module):
```

**解释：**

- `class Critic(nn.Module):`：Critic 是价值网络，输入状态和动作，输出 Q(s,a)。

### 方法 `Critic.__init__`
**代码：**

```python
    def __init__(self, state_dim, action_dim, hidden=(256, 256)):
        super().__init__()
        layers = []
        input_dim = state_dim + action_dim
        for h in hidden:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        self.model = nn.Sequential(*layers)
```

**解释：**

- `def __init__(self, state_dim, action_dim, hidden=(256, 256)):`：函数作用：初始化当前类，创建网络层、优化器或训练需要的成员变量。

### 方法 `Critic.forward`
**代码：**

```python
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.model(x)
```

**解释：**

- `def forward(self, state, action):`：函数作用：定义网络前向传播，说明输入张量如何变成输出。
- `return self.model(x)`：返回训练结果、数据和标准化参数，供主流程继续评估或加载模型。

### 类 `OfflineDataset`
**代码：**

```python
class OfflineDataset(torch.utils.data.Dataset):
```

**解释：**

- `class OfflineDataset(torch.utils.data.Dataset):`：数据集类负责把离线数据封装成 PyTorch 可以按 batch 读取的样本。

### 方法 `OfflineDataset.__init__`
**代码：**

```python
    def __init__(self, hdf5_file):
        with h5py.File(hdf5_file, "r") as f:
            self.obs = torch.tensor(f["observations"][:], dtype=torch.float32)
            self.actions = torch.tensor(f["actions"][:], dtype=torch.float32)
            self.rewards = torch.tensor(f["rewards"][:], dtype=torch.float32).unsqueeze(1)
            self.next_obs = torch.tensor(f["next_observations"][:], dtype=torch.float32)
            self.terminals = torch.tensor(f["terminals"][:], dtype=torch.float32).unsqueeze(1)
        self.length = self.obs.shape[0]
        self.reward_mean = self.rewards.mean()
        self.reward_std = self.rewards.std().clamp(min=1e-6)
```

**解释：**

- `def __init__(self, hdf5_file):`：函数作用：初始化当前类，创建网络层、优化器或训练需要的成员变量。

### 方法 `OfflineDataset.__len__`
**代码：**

```python
    def __len__(self):
        return self.length
```

**解释：**

- `def __len__(self):`：函数作用：封装一段可复用逻辑，供主训练流程调用。
- `return self.length`：返回训练结果、数据和标准化参数，供主流程继续评估或加载模型。

### 方法 `OfflineDataset.__getitem__`
**代码：**

```python
    def __getitem__(self, idx):
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.terminals[idx]
        )
```

**解释：**

- `def __getitem__(self, idx):`：函数作用：封装一段可复用逻辑，供主训练流程调用。
- `return (`：返回训练结果、数据和标准化参数，供主流程继续评估或加载模型。

### 函数 `compute_obs_stats`
**代码：**

```python
def compute_obs_stats(subset):
    obs_list = []
    for item in subset:
        obs_list.append(item[0].unsqueeze(0))
    obs_tensor = torch.cat(obs_list, dim=0)
    obs_mean = obs_tensor.mean(dim=0)
    obs_std = torch.clamp(obs_tensor.std(dim=0), min=1e-6)
    return obs_mean, obs_std
```

**解释：**

- `def compute_obs_stats(subset):`：函数作用：统计训练集状态均值和标准差，后续用于状态标准化。
- `obs_mean = obs_tensor.mean(dim=0)`：obs_mean 的作用：训练集 17 个状态维度的均值，用于状态标准化。
- `obs_std = torch.clamp(obs_tensor.std(dim=0), min=1e-6)`：做数值裁剪，防止动作、标准差或权重超出安全范围。
- `return obs_mean, obs_std`：返回训练结果、数据和标准化参数，供主流程继续评估或加载模型。

### 类 `OfflineAWDDPGAgent`
**代码：**

```python
class OfflineAWDDPGAgent:
```

**解释：**

- `class OfflineAWDDPGAgent:`：训练器/智能体类，封装 actor、critic、优化器以及 train_step/eval_step 等核心流程。

### 方法 `OfflineAWDDPGAgent.__init__`
**代码：**

```python
    def __init__(
        self,
        state_dim,
        action_dim,
        lr_actor=3e-4,
        lr_critic=3e-4,
        gamma=0.99,
        tau=0.005,
        hidden=(256, 256),
        device="cpu",
        weight_decay=1e-4,
        grad_clip=5.0,
        reward_scale=True,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2,
        aw_lambda=1.0,
        weight_clip=20.0
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.grad_clip = grad_clip
        self.reward_scale = reward_scale
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.aw_lambda = aw_lambda
        self.weight_clip = weight_clip
        self.total_it = 0
        self.actor = Actor(state_dim, action_dim, hidden).to(device)
        self.actor_target = Actor(state_dim, action_dim, hidden).to(device)
        self.critic1 = Critic(state_dim, action_dim, hidden).to(device)
        self.critic2 = Critic(state_dim, action_dim, hidden).to(device)
        self.critic1_target = Critic(state_dim, action_dim, hidden).to(device)
        self.critic2_target = Critic(state_dim, action_dim, hidden).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=lr_actor,
            weight_decay=weight_decay
        )
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=lr_critic,
            weight_decay=weight_decay
        )
        self.actor_scheduler = None
        self.critic_scheduler = None
```

**解释：**

- `def __init__(`：函数作用：初始化当前类，创建网络层、优化器或训练需要的成员变量。
- `lr_actor=3e-4,`：创建策略网络。它输入状态，输出动作，是后续训练要优化的核心模型。
- `gamma=0.99,`：gamma 的作用：折扣因子，控制未来奖励在 TD 目标中的权重。
- `tau=0.005,`：tau 的作用：目标网络软更新系数。
- `device="cpu",`：device 的作用：计算设备，优先使用 GPU，否则使用 CPU。
- `grad_clip=5.0,`：grad_clip 的作用：梯度裁剪阈值，防止梯度爆炸。
- `reward_scale=True,`：reward_scale 的作用：是否对奖励做标准化或缩放。
- `policy_noise=0.2,`：policy_noise 的作用：TD3 目标动作平滑噪声强度。
- `noise_clip=0.5,`：noise_clip 的作用：目标动作噪声裁剪范围。
- `policy_delay=2,`：policy_delay 的作用：TD3 中 actor 延迟更新间隔。
- `weight_clip=20.0`：weight_clip 的作用：优势权重裁剪上限，避免少数样本权重过大。
- `self.actor = Actor(state_dim, action_dim, hidden).to(device)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `self.actor_target = Actor(state_dim, action_dim, hidden).to(device)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `self.critic1 = Critic(state_dim, action_dim, hidden).to(device)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `self.critic2 = Critic(state_dim, action_dim, hidden).to(device)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `self.critic1_target = Critic(state_dim, action_dim, hidden).to(device)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `self.critic2_target = Critic(state_dim, action_dim, hidden).to(device)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `self.actor_optimizer = optim.Adam(`：创建 Adam 优化器，后面根据梯度更新网络参数。
- `self.critic_optimizer = optim.Adam(`：创建 Adam 优化器，后面根据梯度更新网络参数。
- `self.actor_scheduler = None`：创建学习率调度器，让学习率随训练轮数变化，后期训练更稳定。
- `self.critic_scheduler = None`：创建学习率调度器，让学习率随训练轮数变化，后期训练更稳定。

### 方法 `OfflineAWDDPGAgent.set_schedulers`
**代码：**

```python
    def set_schedulers(self, actor_scheduler, critic_scheduler):
        self.actor_scheduler = actor_scheduler
        self.critic_scheduler = critic_scheduler
```

**解释：**

- `def set_schedulers(self, actor_scheduler, critic_scheduler):`：函数作用：封装一段可复用逻辑，供主训练流程调用。
- `self.actor_scheduler = actor_scheduler`：创建学习率调度器，让学习率随训练轮数变化，后期训练更稳定。
- `self.critic_scheduler = critic_scheduler`：创建学习率调度器，让学习率随训练轮数变化，后期训练更稳定。

### 方法 `OfflineAWDDPGAgent.soft_update`
**代码：**

```python
    def soft_update(self, net, net_target):
        for param, target_param in zip(net.parameters(), net_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
```

**解释：**

- `def soft_update(self, net, net_target):`：函数作用：软更新目标网络，让目标网络缓慢跟随当前网络，稳定 TD 目标。

### 方法 `OfflineAWDDPGAgent.select_action`
**代码：**

```python
    def select_action(self, state, obs_mean=None, obs_std=None):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        if obs_mean is not None and obs_std is not None:
            state = (state - obs_mean) / obs_std
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]
        return np.clip(action, -1.0, 1.0)
```

**解释：**

- `def select_action(self, state, obs_mean=None, obs_std=None):`：函数作用：封装一段可复用逻辑，供主训练流程调用。
- `state = (state - obs_mean) / obs_std`：对当前状态做标准化：减去训练集均值，再除以训练集标准差。
- `with torch.no_grad():`：进入不记录梯度的区域，通常用于计算 TD 目标或验证指标。
- `return np.clip(action, -1.0, 1.0)`：返回训练结果、数据和标准化参数，供主流程继续评估或加载模型。

### 方法 `OfflineAWDDPGAgent.train_step`
**代码：**

```python
    def train_step(self, batch, obs_mean, obs_std, reward_mean, reward_std):
        self.total_it += 1
        obs, actions, rewards, next_obs, terminals = batch
        obs = obs.to(self.device, non_blocking=True)
        actions = actions.to(self.device, non_blocking=True)
        rewards = rewards.to(self.device, non_blocking=True)
        next_obs = next_obs.to(self.device, non_blocking=True)
        terminals = terminals.to(self.device, non_blocking=True)
        obs = (obs - obs_mean) / obs_std
        next_obs = (next_obs - obs_mean) / obs_std
        if self.reward_scale:
            rewards = (rewards - reward_mean) / reward_std
        with torch.no_grad():
            noise = torch.randn_like(actions) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_obs) + noise).clamp(-1.0, 1.0)
            q1_target_next = self.critic1_target(next_obs, next_actions)
            q2_target_next = self.critic2_target(next_obs, next_actions)
            q_target_next = torch.min(q1_target_next, q2_target_next)
            td_target = rewards + self.gamma * (1 - terminals) * q_target_next
        q1_val = self.critic1(obs, actions)
        q2_val = self.critic2(obs, actions)
        critic1_loss = nn.functional.mse_loss(q1_val, td_target)
        critic2_loss = nn.functional.mse_loss(q2_val, td_target)
        critic_loss = critic1_loss + critic2_loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            self.grad_clip
        )
        self.critic_optimizer.step()
        actor_loss_value = None
        bc_loss_value = None
        advantage_value = None
        weight_value = None
        if self.total_it % self.policy_delay == 0:
            with torch.no_grad():
                q_data = torch.min(
                    self.critic1(obs, actions),
                    self.critic2(obs, actions)
                )
                pi_actions = self.actor(obs)
                q_pi = torch.min(
                    self.critic1(obs, pi_actions),
                    self.critic2(obs, pi_actions)
                )
                advantage = q_data - q_pi
                weights = torch.exp(advantage / self.aw_lambda)
                weights = torch.clamp(weights, max=self.weight_clip)
            pred_actions = self.actor(obs)
            per_sample_bc = ((pred_actions - actions) ** 2).mean(dim=1, keepdim=True)
            actor_loss = (weights * per_sample_bc).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
            self.actor_optimizer.step()
            self.soft_update(self.actor, self.actor_target)
            self.soft_update(self.critic1, self.critic1_target)
            self.soft_update(self.critic2, self.critic2_target)
            actor_loss_value = actor_loss.item()
            bc_loss_value = per_sample_bc.mean().item()
            advantage_value = advantage.mean().item()
            weight_value = weights.mean().item()
        else:
            self.soft_update(self.critic1, self.critic1_target)
            self.soft_update(self.critic2, self.critic2_target)
        return {
            "actor_loss": actor_loss_value,
            "critic_loss": critic_loss.item(),
            "critic1_loss": critic1_loss.item(),
            "critic2_loss": critic2_loss.item(),
            "bc_loss": bc_loss_value,
            "advantage": advantage_value,
            "weight": weight_value,
            "reward": rewards.mean().item(),
            "Q1": q1_val.mean().item(),
            "Q2": q2_val.mean().item(),
            "actor_updated": 1 if actor_loss_value is not None else 0
        }
```

**解释：**

- `def train_step(self, batch, obs_mean, obs_std, reward_mean, reward_std):`：函数作用：执行一次 DDPG 更新：用 TD 目标训练 critic，再用 critic 的 Q 值指导 actor 更新。
- `obs, actions, rewards, next_obs, terminals = batch`：把一个 batch 拆成状态、动作、奖励、下一状态和终止标志。
- `obs = obs.to(self.device, non_blocking=True)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `actions = actions.to(self.device, non_blocking=True)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `rewards = rewards.to(self.device, non_blocking=True)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `next_obs = next_obs.to(self.device, non_blocking=True)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `terminals = terminals.to(self.device, non_blocking=True)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `obs = (obs - obs_mean) / obs_std`：对当前状态做标准化：减去训练集均值，再除以训练集标准差。
- `next_obs = (next_obs - obs_mean) / obs_std`：对下一状态做同样标准化，保证 TD 目标计算使用相同输入尺度。
- `if self.reward_scale:`：如果启用奖励标准化/缩放，就对 reward 做归一化处理。
- `rewards = (rewards - reward_mean) / reward_std`：对奖励做标准化，减少奖励尺度过大或过小带来的训练不稳定。
- `with torch.no_grad():`：进入不记录梯度的区域，通常用于计算 TD 目标或验证指标。
- `noise = torch.randn_like(actions) * self.policy_noise`：生成并裁剪目标动作噪声，这是 TD3 的 target policy smoothing。
- `noise = noise.clamp(-self.noise_clip, self.noise_clip)`：生成并裁剪目标动作噪声，这是 TD3 的 target policy smoothing。
- `next_actions = (self.actor_target(next_obs) + noise).clamp(-1.0, 1.0)`：用目标 actor 计算下一状态动作，并加入噪声后裁剪到动作范围内。
- `q1_target_next = self.critic1_target(next_obs, next_actions)`：用目标 critic 估计下一状态动作的 Q 值。
- `q2_target_next = self.critic2_target(next_obs, next_actions)`：用目标 critic 估计下一状态动作的 Q 值。
- `q_target_next = torch.min(q1_target_next, q2_target_next)`：双 Q 取较小值，降低 Q 值过估计风险。
- `td_target = rewards + self.gamma * (1 - terminals) * q_target_next`：构造 TD 目标：即时奖励加上折扣后的下一状态价值；done 为 1 时不再加未来价值。
- `q1_val = self.critic1(obs, actions)`：用当前 critic 估计数据集中真实状态动作的 Q 值。
- `q2_val = self.critic2(obs, actions)`：用当前 critic 估计数据集中真实状态动作的 Q 值。
- `critic1_loss = nn.functional.mse_loss(q1_val, td_target)`：用当前 critic 估计数据集中真实状态动作的 Q 值。
- `critic2_loss = nn.functional.mse_loss(q2_val, td_target)`：用当前 critic 估计数据集中真实状态动作的 Q 值。
- `critic_loss = critic1_loss + critic2_loss`：计算 critic 的 TD 回归损失，让当前 Q 值逼近 TD 目标。
- `self.critic_optimizer.zero_grad()`：清空上一轮梯度，避免梯度累积影响当前 batch。
- `critic_loss.backward()`：对当前损失反向传播，计算网络参数梯度。
- `nn.utils.clip_grad_norm_(`：对梯度做裁剪，防止梯度过大导致训练不稳定。
- `self.critic_optimizer.step()`：优化器根据梯度更新参数。
- `bc_loss_value = None`：BC 正则项，惩罚策略动作偏离离线数据动作，降低离线策略外推风险。
- `q_pi = torch.min(`：用 critic 评价当前策略动作的 Q 值；actor 更新会希望这个值更大。
- `weights = torch.clamp(weights, max=self.weight_clip)`：做数值裁剪，防止动作、标准差或权重超出安全范围。
- `actor_loss = (weights * per_sample_bc).mean()`：合成 actor 损失，通常由 Q 项、熵项或 BC 正则项组成。
- `self.actor_optimizer.zero_grad()`：清空上一轮梯度，避免梯度累积影响当前 batch。
- `actor_loss.backward()`：对当前损失反向传播，计算网络参数梯度。
- `nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)`：对梯度做裁剪，防止梯度过大导致训练不稳定。
- `self.actor_optimizer.step()`：优化器根据梯度更新参数。
- `self.soft_update(self.actor, self.actor_target)`：软更新目标 actor，让目标策略网络缓慢跟随当前 actor。
- `self.soft_update(self.critic1, self.critic1_target)`：软更新目标 critic，让目标 Q 网络缓慢跟随当前 critic。
- `self.soft_update(self.critic2, self.critic2_target)`：软更新目标 critic，让目标 Q 网络缓慢跟随当前 critic。
- `bc_loss_value = per_sample_bc.mean().item()`：BC 正则项，惩罚策略动作偏离离线数据动作，降低离线策略外推风险。
- `return {`：返回本次更新或评估的指标字典，供训练日志记录。

### 方法 `OfflineAWDDPGAgent.eval_step`
**代码：**

```python
    @torch.no_grad()
    def eval_step(self, batch, obs_mean, obs_std, reward_mean, reward_std):
        obs, actions, rewards, next_obs, terminals = batch
        obs = obs.to(self.device, non_blocking=True)
        actions = actions.to(self.device, non_blocking=True)
        rewards = rewards.to(self.device, non_blocking=True)
        next_obs = next_obs.to(self.device, non_blocking=True)
        terminals = terminals.to(self.device, non_blocking=True)
        obs = (obs - obs_mean) / obs_std
        next_obs = (next_obs - obs_mean) / obs_std
        if self.reward_scale:
            rewards = (rewards - reward_mean) / reward_std
        noise = torch.randn_like(actions) * self.policy_noise
        noise = noise.clamp(-self.noise_clip, self.noise_clip)
        next_actions = (self.actor_target(next_obs) + noise).clamp(-1.0, 1.0)
        q1_target_next = self.critic1_target(next_obs, next_actions)
        q2_target_next = self.critic2_target(next_obs, next_actions)
        q_target_next = torch.min(q1_target_next, q2_target_next)
        td_target = rewards + self.gamma * (1 - terminals) * q_target_next
        q1_val = self.critic1(obs, actions)
        q2_val = self.critic2(obs, actions)
        critic1_loss = nn.functional.mse_loss(q1_val, td_target)
        critic2_loss = nn.functional.mse_loss(q2_val, td_target)
        critic_loss = critic1_loss + critic2_loss
        q_data = torch.min(self.critic1(obs, actions), self.critic2(obs, actions))
        pi_actions = self.actor(obs)
        q_pi = torch.min(self.critic1(obs, pi_actions), self.critic2(obs, pi_actions))
        advantage = q_data - q_pi
        weights = torch.exp(advantage / self.aw_lambda)
        weights = torch.clamp(weights, max=self.weight_clip)
        per_sample_bc = ((pi_actions - actions) ** 2).mean(dim=1, keepdim=True)
        actor_loss = (weights * per_sample_bc).mean()
        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "critic1_loss": critic1_loss.item(),
            "critic2_loss": critic2_loss.item(),
            "bc_loss": per_sample_bc.mean().item(),
            "advantage": advantage.mean().item(),
            "weight": weights.mean().item(),
            "reward": rewards.mean().item(),
            "Q1": q1_val.mean().item(),
            "Q2": q2_val.mean().item(),
            "actor_updated": 1
        }
```

**解释：**

- `@torch.no_grad()`：装饰器作用：该函数执行时不记录梯度，适合验证、评估或构造目标值。
- `def eval_step(self, batch, obs_mean, obs_std, reward_mean, reward_std):`：函数作用：在指定数据集或 batch 上评估模型，不更新参数。
- `obs, actions, rewards, next_obs, terminals = batch`：把一个 batch 拆成状态、动作、奖励、下一状态和终止标志。
- `obs = obs.to(self.device, non_blocking=True)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `actions = actions.to(self.device, non_blocking=True)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `rewards = rewards.to(self.device, non_blocking=True)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `next_obs = next_obs.to(self.device, non_blocking=True)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `terminals = terminals.to(self.device, non_blocking=True)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `obs = (obs - obs_mean) / obs_std`：对当前状态做标准化：减去训练集均值，再除以训练集标准差。
- `next_obs = (next_obs - obs_mean) / obs_std`：对下一状态做同样标准化，保证 TD 目标计算使用相同输入尺度。
- `if self.reward_scale:`：如果启用奖励标准化/缩放，就对 reward 做归一化处理。
- `rewards = (rewards - reward_mean) / reward_std`：对奖励做标准化，减少奖励尺度过大或过小带来的训练不稳定。
- `noise = torch.randn_like(actions) * self.policy_noise`：生成并裁剪目标动作噪声，这是 TD3 的 target policy smoothing。
- `noise = noise.clamp(-self.noise_clip, self.noise_clip)`：生成并裁剪目标动作噪声，这是 TD3 的 target policy smoothing。
- `next_actions = (self.actor_target(next_obs) + noise).clamp(-1.0, 1.0)`：用目标 actor 计算下一状态动作，并加入噪声后裁剪到动作范围内。
- `q1_target_next = self.critic1_target(next_obs, next_actions)`：用目标 critic 估计下一状态动作的 Q 值。
- `q2_target_next = self.critic2_target(next_obs, next_actions)`：用目标 critic 估计下一状态动作的 Q 值。
- `q_target_next = torch.min(q1_target_next, q2_target_next)`：双 Q 取较小值，降低 Q 值过估计风险。
- `td_target = rewards + self.gamma * (1 - terminals) * q_target_next`：构造 TD 目标：即时奖励加上折扣后的下一状态价值；done 为 1 时不再加未来价值。
- `q1_val = self.critic1(obs, actions)`：用当前 critic 估计数据集中真实状态动作的 Q 值。
- `q2_val = self.critic2(obs, actions)`：用当前 critic 估计数据集中真实状态动作的 Q 值。
- `critic1_loss = nn.functional.mse_loss(q1_val, td_target)`：用当前 critic 估计数据集中真实状态动作的 Q 值。
- `critic2_loss = nn.functional.mse_loss(q2_val, td_target)`：用当前 critic 估计数据集中真实状态动作的 Q 值。
- `critic_loss = critic1_loss + critic2_loss`：计算 critic 的 TD 回归损失，让当前 Q 值逼近 TD 目标。
- `q_pi = torch.min(self.critic1(obs, pi_actions), self.critic2(obs, pi_actions))`：用 critic 评价当前策略动作的 Q 值；actor 更新会希望这个值更大。
- `weights = torch.clamp(weights, max=self.weight_clip)`：做数值裁剪，防止动作、标准差或权重超出安全范围。
- `actor_loss = (weights * per_sample_bc).mean()`：合成 actor 损失，通常由 Q 项、熵项或 BC 正则项组成。
- `return {`：返回本次更新或评估的指标字典，供训练日志记录。

### 方法 `OfflineAWDDPGAgent.save_best`
**代码：**

```python
    def save_best(
        self,
        output_dir,
        best_model_name,
        obs_mean,
        obs_std,
        reward_mean,
        reward_std,
        epoch,
        best_val_loss,
        state_dim,
        action_dim,
        hidden
    ):
        torch.save({
            "actor_state_dict": self.actor.state_dict(),
            "critic1_state_dict": self.critic1.state_dict(),
            "critic2_state_dict": self.critic2.state_dict(),
            "obs_mean": obs_mean.cpu(),
            "obs_std": obs_std.cpu(),
            "reward_mean": reward_mean.cpu(),
            "reward_std": reward_std.cpu(),
            "state_dim": state_dim,
            "action_dim": action_dim,
            "hidden": list(hidden),
            "epoch": epoch,
            "best_val_critic_loss": best_val_loss,
            "reward_scale": self.reward_scale,
            "policy_noise": self.policy_noise,
            "noise_clip": self.noise_clip,
            "policy_delay": self.policy_delay,
            "aw_lambda": self.aw_lambda,
            "weight_clip": self.weight_clip
        }, os.path.join(output_dir, best_model_name))
```

**解释：**

- `def save_best(`：函数作用：保存当前最佳模型，同时保存标准化参数和网络结构信息，便于之后恢复。

### 函数 `train_ddpg_improved`
**代码：**

```python
def train_ddpg_improved(
    agent,
    train_loader,
    val_loader,
    num_epochs,
    output_dir,
    best_model_name,
    records_csv_name,
    actor_curve_name,
    critic_curve_name,
    weight_curve_name,
    obs_mean,
    obs_std,
    reward_mean,
    reward_std,
    state_dim,
    action_dim,
    hidden
):
    os.makedirs(output_dir, exist_ok=True)
    records = {
        "epoch": [],
        "train_actor_loss": [],
        "train_critic_loss": [],
        "train_bc_loss": [],
        "train_advantage": [],
        "train_weight": [],
        "train_reward": [],
        "train_Q1": [],
        "train_Q2": [],
        "val_actor_loss": [],
        "val_critic_loss": [],
        "val_bc_loss": [],
        "val_advantage": [],
        "val_weight": [],
        "val_reward": [],
        "val_Q1": [],
        "val_Q2": [],
        "actor_lr": [],
        "critic_lr": []
    }
    best_val_critic_loss = float("inf")
    for epoch in range(1, num_epochs + 1):
        train_sum = {
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "bc_loss": 0.0,
            "advantage": 0.0,
            "weight": 0.0,
            "reward": 0.0,
            "Q1": 0.0,
            "Q2": 0.0
        }
        actor_count = 0
        for batch in train_loader:
            out = agent.train_step(batch, obs_mean, obs_std, reward_mean, reward_std)
            train_sum["critic_loss"] += out["critic_loss"]
            train_sum["reward"] += out["reward"]
            train_sum["Q1"] += out["Q1"]
            train_sum["Q2"] += out["Q2"]
            if out["actor_loss"] is not None:
                train_sum["actor_loss"] += out["actor_loss"]
                train_sum["bc_loss"] += out["bc_loss"]
                train_sum["advantage"] += out["advantage"]
                train_sum["weight"] += out["weight"]
                actor_count += 1
        n_train = len(train_loader)
        train_metrics = {
            "actor_loss": train_sum["actor_loss"] / max(actor_count, 1),
            "critic_loss": train_sum["critic_loss"] / n_train,
            "bc_loss": train_sum["bc_loss"] / max(actor_count, 1),
            "advantage": train_sum["advantage"] / max(actor_count, 1),
            "weight": train_sum["weight"] / max(actor_count, 1),
            "reward": train_sum["reward"] / n_train,
            "Q1": train_sum["Q1"] / n_train,
            "Q2": train_sum["Q2"] / n_train
        }
        val_sum = {
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "bc_loss": 0.0,
            "advantage": 0.0,
            "weight": 0.0,
            "reward": 0.0,
            "Q1": 0.0,
            "Q2": 0.0
        }
        for batch in val_loader:
            out = agent.eval_step(batch, obs_mean, obs_std, reward_mean, reward_std)
            for k in val_sum:
                val_sum[k] += out[k]
        n_val = len(val_loader)
        val_metrics = {k: v / n_val for k, v in val_sum.items()}
        actor_lr = agent.actor_optimizer.param_groups[0]["lr"]
        critic_lr = agent.critic_optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch}: "
            f"Train Actor={train_metrics['actor_loss']:.6f}, "
            f"Train Critic={train_metrics['critic_loss']:.6f}, "
            f"Train Adv={train_metrics['advantage']:.6f}, "
            f"Val Actor={val_metrics['actor_loss']:.6f}, "
            f"Val Critic={val_metrics['critic_loss']:.6f}, "
            f"Val Weight={val_metrics['weight']:.6f}"
        )
        records["epoch"].append(epoch)
        records["train_actor_loss"].append(train_metrics["actor_loss"])
        records["train_critic_loss"].append(train_metrics["critic_loss"])
        records["train_bc_loss"].append(train_metrics["bc_loss"])
        records["train_advantage"].append(train_metrics["advantage"])
        records["train_weight"].append(train_metrics["weight"])
        records["train_reward"].append(train_metrics["reward"])
        records["train_Q1"].append(train_metrics["Q1"])
        records["train_Q2"].append(train_metrics["Q2"])
        records["val_actor_loss"].append(val_metrics["actor_loss"])
        records["val_critic_loss"].append(val_metrics["critic_loss"])
        records["val_bc_loss"].append(val_metrics["bc_loss"])
        records["val_advantage"].append(val_metrics["advantage"])
        records["val_weight"].append(val_metrics["weight"])
        records["val_reward"].append(val_metrics["reward"])
        records["val_Q1"].append(val_metrics["Q1"])
        records["val_Q2"].append(val_metrics["Q2"])
        records["actor_lr"].append(actor_lr)
        records["critic_lr"].append(critic_lr)
        if val_metrics["critic_loss"] < best_val_critic_loss:
            best_val_critic_loss = val_metrics["critic_loss"]
            agent.save_best(
                output_dir=output_dir,
                best_model_name=best_model_name,
                obs_mean=obs_mean,
                obs_std=obs_std,
                reward_mean=reward_mean,
                reward_std=reward_std,
                epoch=epoch,
                best_val_loss=best_val_critic_loss,
                state_dim=state_dim,
                action_dim=action_dim,
                hidden=hidden
            )
            print(f"Best model saved at epoch {epoch}, val_critic_loss={best_val_critic_loss:.6f}")
        if agent.actor_scheduler is not None:
            agent.actor_scheduler.step()
        if agent.critic_scheduler is not None:
            agent.critic_scheduler.step()
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(output_dir, records_csv_name), index=False)
    plt.figure(figsize=(8, 5))
    plt.plot(records["epoch"], records["train_actor_loss"], label="Train Actor Loss")
    plt.plot(records["epoch"], records["val_actor_loss"], label="Val Actor Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("AW-DDPG_NB Actor Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, actor_curve_name))
    plt.close()
    plt.figure(figsize=(8, 5))
    plt.plot(records["epoch"], records["train_critic_loss"], label="Train Critic Loss")
    plt.plot(records["epoch"], records["val_critic_loss"], label="Val Critic Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("AW-DDPG_NB Critic Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, critic_curve_name))
    plt.close()
    plt.figure(figsize=(8, 5))
    plt.plot(records["epoch"], records["train_weight"], label="Train Weight")
    plt.plot(records["epoch"], records["val_weight"], label="Val Weight")
    plt.xlabel("Epoch")
    plt.ylabel("Weight")
    plt.legend()
    plt.title("AW-DDPG_NB Advantage Weight")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, weight_curve_name))
    plt.close()
```

**解释：**

- `def train_ddpg_improved(`：函数作用：封装一段可复用逻辑，供主训练流程调用。
- `best_val_critic_loss = float("inf")`：计算 critic 的 TD 回归损失，让当前 Q 值逼近 TD 目标。
- `for epoch in range(1, num_epochs + 1):`：开始按 epoch 训练模型，每轮都会训练一次、验证一次，并记录指标。
- `train_sum["bc_loss"] += out["bc_loss"]`：BC 正则项，惩罚策略动作偏离离线数据动作，降低离线策略外推风险。
- `print(`：把当前训练或评估信息输出到终端。
- `records["train_bc_loss"].append(train_metrics["bc_loss"])`：BC 正则项，惩罚策略动作偏离离线数据动作，降低离线策略外推风险。
- `records["val_bc_loss"].append(val_metrics["bc_loss"])`：BC 正则项，惩罚策略动作偏离离线数据动作，降低离线策略外推风险。
- `if val_metrics["critic_loss"] < best_val_critic_loss:`：如果当前验证结果优于历史最好结果，就保存当前模型。
- `best_val_critic_loss = val_metrics["critic_loss"]`：计算 critic 的 TD 回归损失，让当前 Q 值逼近 TD 目标。
- `agent.save_best(`：保存当前最佳模型，同时保存标准化参数和网络结构。
- `output_dir=output_dir,`：output_dir 的作用：模型、日志和图片的输出目录。
- `best_model_name=best_model_name,`：best_model_name 的作用：最佳模型 checkpoint 文件名。
- `best_val_loss=best_val_critic_loss,`：初始化最佳验证指标为无穷大，后面只要验证损失更小就保存模型。
- `print(f"Best model saved at epoch {epoch}, val_critic_loss={best_val_critic_loss:.6f}")`：计算 critic 的 TD 回归损失，让当前 Q 值逼近 TD 目标。
- `agent.actor_scheduler.step()`：优化器根据梯度更新参数。
- `agent.critic_scheduler.step()`：优化器根据梯度更新参数。
- `df = pd.DataFrame(records)`：把训练记录转换成 pandas 表格，便于保存 CSV 和画曲线。
- `df.to_csv(os.path.join(output_dir, records_csv_name), index=False)`：把表格结果保存成 CSV 文件，方便论文记录、画图或复查。

### 函数 `load_best_actor`
**代码：**

```python
def load_best_actor(device, output_dir, best_model_name):
    ckpt = torch.load(os.path.join(output_dir, best_model_name), map_location=device)
    actor = Actor(
        ckpt["state_dim"],
        ckpt["action_dim"],
        ckpt["hidden"]
    ).to(device)
    actor.load_state_dict(ckpt["actor_state_dict"])
    obs_mean = ckpt["obs_mean"].to(device)
    obs_std = ckpt["obs_std"].to(device)
    print(
        f"Loaded best actor from epoch {ckpt['epoch']}, "
        f"best_val_critic_loss={ckpt['best_val_critic_loss']:.6f}"
    )
    return actor, obs_mean, obs_std, ckpt
```

**解释：**

- `def load_best_actor(device, output_dir, best_model_name):`：函数作用：加载训练过程中保存的最佳策略模型，用于评估或推理。
- `actor = Actor(`：创建策略网络。它输入状态，输出动作，是后续训练要优化的核心模型。
- `).to(device)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `obs_mean = ckpt["obs_mean"].to(device)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `obs_std = ckpt["obs_std"].to(device)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `print(`：把当前训练或评估信息输出到终端。
- `return actor, obs_mean, obs_std, ckpt`：返回训练结果、数据和标准化参数，供主流程继续评估或加载模型。

### 函数 `main`
**代码：**

```python
def main():
    cfg = build_config_from_args()
    set_seed(cfg.seed)
    ensure_dirs(cfg.output_dir)
    print("Using config:")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")
    dataset = OfflineDataset(cfg.hdf5_file)
    total_size = len(dataset)
    val_size = int(total_size * cfg.val_ratio)
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.seed)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda")
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda")
    )
    device = torch.device(cfg.device)
    obs_mean, obs_std = compute_obs_stats(train_dataset)
    obs_mean = obs_mean.to(device)
    obs_std = obs_std.to(device)
    reward_mean = dataset.reward_mean.to(device)
    reward_std = dataset.reward_std.to(device)
    state_dim = dataset.obs.shape[1]
    action_dim = dataset.actions.shape[1]
    agent = OfflineAWDDPGAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr_actor=cfg.lr_actor,
        lr_critic=cfg.lr_critic,
        gamma=cfg.gamma,
        tau=cfg.tau,
        hidden=cfg.hidden,
        device=device,
        weight_decay=cfg.weight_decay,
        grad_clip=cfg.grad_clip,
        reward_scale=cfg.reward_scale,
        policy_noise=cfg.policy_noise,
        noise_clip=cfg.noise_clip,
        policy_delay=cfg.policy_delay,
        aw_lambda=cfg.aw_lambda,
        weight_clip=cfg.weight_clip
    )
    actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        agent.actor_optimizer,
        T_max=cfg.num_epochs
    )
    critic_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        agent.critic_optimizer,
        T_max=cfg.num_epochs
    )
    agent.set_schedulers(actor_scheduler, critic_scheduler)
    train_ddpg_improved(
        agent=agent,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=cfg.num_epochs,
        output_dir=cfg.output_dir,
        best_model_name=cfg.best_model_name,
        records_csv_name=cfg.records_csv_name,
        actor_curve_name=cfg.actor_curve_name,
        critic_curve_name=cfg.critic_curve_name,
        weight_curve_name=cfg.weight_curve_name,
        obs_mean=obs_mean,
        obs_std=obs_std,
        reward_mean=reward_mean,
        reward_std=reward_std,
        state_dim=state_dim,
        action_dim=action_dim,
        hidden=cfg.hidden
    )
    load_best_actor(device, cfg.output_dir, cfg.best_model_name)
```

**解释：**

- `def main():`：函数作用：脚本主入口，负责读取配置、启动训练、加载最佳模型、做最终评估并保存评估结果。
- `cfg = build_config_from_args()`：读取命令行参数并生成配置对象 `cfg`，后面的训练、保存和评估都使用这份配置。
- `set_seed(cfg.seed)`：设置随机种子，让数据划分、初始化和训练过程尽量可复现。
- `print("Using config:")`：把当前训练或评估信息输出到终端。
- `for k, v in asdict(cfg).items():`：遍历配置对象中的每个参数，把本次实验配置打印出来。
- `print(f"  {k}: {v}")`：把当前训练或评估信息输出到终端。
- `batch_size=cfg.batch_size,`：batch_size 的作用：每个 batch 的样本数。
- `device = torch.device(cfg.device)`：device 的作用：计算设备，优先使用 GPU，否则使用 CPU。
- `obs_mean, obs_std = compute_obs_stats(train_dataset)`：只用训练集计算状态均值和标准差，后面用它们对状态做标准化。
- `obs_mean = obs_mean.to(device)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `obs_std = obs_std.to(device)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `reward_mean = dataset.reward_mean.to(device)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `reward_std = dataset.reward_std.to(device)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `state_dim = dataset.obs.shape[1]`：state_dim 的作用：状态维度，本项目通常是 17。
- `action_dim = dataset.actions.shape[1]`：action_dim 的作用：动作维度，本项目通常是 6。
- `lr_actor=cfg.lr_actor,`：创建策略网络。它输入状态，输出动作，是后续训练要优化的核心模型。
- `gamma=cfg.gamma,`：gamma 的作用：折扣因子，控制未来奖励在 TD 目标中的权重。
- `tau=cfg.tau,`：tau 的作用：目标网络软更新系数。
- `device=device,`：device 的作用：计算设备，优先使用 GPU，否则使用 CPU。
- `grad_clip=cfg.grad_clip,`：grad_clip 的作用：梯度裁剪阈值，防止梯度爆炸。
- `reward_scale=cfg.reward_scale,`：reward_scale 的作用：是否对奖励做标准化或缩放。
- `policy_noise=cfg.policy_noise,`：policy_noise 的作用：TD3 目标动作平滑噪声强度。
- `noise_clip=cfg.noise_clip,`：noise_clip 的作用：目标动作噪声裁剪范围。
- `policy_delay=cfg.policy_delay,`：policy_delay 的作用：TD3 中 actor 延迟更新间隔。
- `weight_clip=cfg.weight_clip`：weight_clip 的作用：优势权重裁剪上限，避免少数样本权重过大。
- `actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(`：创建学习率调度器，让学习率随训练轮数变化，后期训练更稳定。
- `critic_scheduler = optim.lr_scheduler.CosineAnnealingLR(`：创建学习率调度器，让学习率随训练轮数变化，后期训练更稳定。
- `num_epochs=cfg.num_epochs,`：num_epochs 的作用：训练轮数。
- `output_dir=cfg.output_dir,`：output_dir 的作用：模型、日志和图片的输出目录。
- `best_model_name=cfg.best_model_name,`：best_model_name 的作用：最佳模型 checkpoint 文件名。


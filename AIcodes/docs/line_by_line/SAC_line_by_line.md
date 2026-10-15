# SAC.py 逐行代码解释附录

这是 [`SAC.md`](../SAC.md) 的细节附录，保留对应源码每一行非空代码的解释。主文档负责理解流程；这里适合查某一行具体在做什么。

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
import math
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# ===============================
# ========== Utils ==============
# ===============================
# ===============================
# ========== Config =============
# ===============================
# ===============================
# ========== Networks ===========
# ===============================
# ===============================
# ========== Dataset ============
# ===============================
# ===============================
# ========== Agent ==============
# ===============================
# ===============================
# ========== Training ===========
# ===============================
# ===============================
# ========== Save History =======
# ===============================
# ===============================
# ========== Load Best Actor ====
# ===============================
# ===============================
# ========== Main ===============
# ===============================
if __name__ == "__main__":
    main()
```

**解释：**

- `import os`：导入依赖库，供后续数据读取、模型训练、优化、绘图或参数解析使用。
- `import json`：导入依赖库，供后续数据读取、模型训练、优化、绘图或参数解析使用。
- `import random`：导入依赖库，供后续数据读取、模型训练、优化、绘图或参数解析使用。
- `import argparse`：导入依赖库，供后续数据读取、模型训练、优化、绘图或参数解析使用。
- `from dataclasses import dataclass, asdict, field`：导入依赖库，供后续数据读取、模型训练、优化、绘图或参数解析使用。
- `import math`：导入依赖库，供后续数据读取、模型训练、优化、绘图或参数解析使用。
- `import h5py`：导入依赖库，供后续数据读取、模型训练、优化、绘图或参数解析使用。
- `import torch`：导入依赖库，供后续数据读取、模型训练、优化、绘图或参数解析使用。
- `import torch.nn as nn`：导入依赖库，供后续数据读取、模型训练、优化、绘图或参数解析使用。
- `import torch.optim as optim`：导入依赖库，供后续数据读取、模型训练、优化、绘图或参数解析使用。
- `from torch.utils.data import TensorDataset, DataLoader, random_split`：导入依赖库，供后续数据读取、模型训练、优化、绘图或参数解析使用。
- `import numpy as np`：导入依赖库，供后续数据读取、模型训练、优化、绘图或参数解析使用。
- `import pandas as pd`：导入依赖库，供后续数据读取、模型训练、优化、绘图或参数解析使用。
- `import matplotlib.pyplot as plt`：导入依赖库，供后续数据读取、模型训练、优化、绘图或参数解析使用。
- `if __name__ == "__main__":`：判断当前文件是否被直接运行；只有直接运行时才启动 main。
- `main()`：调用 main 函数，正式启动整个脚本。

### 函数 `ensure_dirs`
**代码：**

```python
def ensure_dirs(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    return save_dir
```

**解释：**

- `def ensure_dirs(save_dir):`：函数作用：封装一段可复用逻辑，供主训练流程调用。
- `return save_dir`：返回训练结果、数据和标准化参数，供主流程继续评估或加载模型。

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
    for param, target_param in zip(source_net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
```

**解释：**

- `def soft_update(source_net, target_net, tau):`：函数作用：软更新目标网络，让目标网络缓慢跟随当前网络，稳定 TD 目标。

### 函数 `gaussian_log_prob`
**代码：**

```python
def gaussian_log_prob(noise, log_std):
    return (-0.5 * (noise.pow(2) + 2 * log_std + math.log(2 * math.pi))).sum(dim=-1, keepdim=True)
```

**解释：**

- `def gaussian_log_prob(noise, log_std):`：函数作用：封装一段可复用逻辑，供主训练流程调用。
- `return (-0.5 * (noise.pow(2) + 2 * log_std + math.log(2 * math.pi))).sum(dim=-1, keepdim=True)`：返回训练结果、数据和标准化参数，供主流程继续评估或加载模型。

### 函数 `parse_hidden_sizes`
**代码：**

```python
def parse_hidden_sizes(value):
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

- `def parse_hidden_sizes(value):`：函数作用：把命令行里的隐藏层配置转换成整数列表，用来搭建神经网络。
- `return result`：返回训练结果、数据和标准化参数，供主流程继续评估或加载模型。
- `return [256, 256]`：返回训练结果、数据和标准化参数，供主流程继续评估或加载模型。
- `return [int(x.strip()) for x in value.split(",") if x.strip()]`：返回训练结果、数据和标准化参数，供主流程继续评估或加载模型。

### 函数 `is_finite_tensor`
**代码：**

```python
def is_finite_tensor(x):
    return torch.isfinite(x).all().item()
```

**解释：**

- `def is_finite_tensor(x):`：函数作用：封装一段可复用逻辑，供主训练流程调用。
- `return torch.isfinite(x).all().item()`：返回训练结果、数据和标准化参数，供主流程继续评估或加载模型。

### 函数 `sanitize_tensor`
**代码：**

```python
def sanitize_tensor(x, clamp_min=-1e6, clamp_max=1e6):
    x = torch.nan_to_num(x, nan=0.0, posinf=clamp_max, neginf=clamp_min)
    x = torch.clamp(x, clamp_min, clamp_max)
    return x
```

**解释：**

- `def sanitize_tensor(x, clamp_min=-1e6, clamp_max=1e6):`：函数作用：封装一段可复用逻辑，供主训练流程调用。
- `x = torch.clamp(x, clamp_min, clamp_max)`：做数值裁剪，防止动作、标准差或权重超出安全范围。
- `return x`：返回训练结果、数据和标准化参数，供主流程继续评估或加载模型。

### 类 `Config`
**代码：**

```python
@dataclass
class Config:
    env_name: str = "halfcheetah-medium-v2"
    hidden_sizes: list = field(default_factory=lambda: [256, 256])
    gamma: float = 0.99
    tau: float = 0.005
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    batch_size: int = 256
    num_epochs: int = 150
    val_ratio: float = 0.1
    seed: int = 42
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    data_path: str = "halfcheetah_medium-v2.hdf5"
    save_dir: str = "checkpoints/SAC"
    reward_scale: bool = True
    alpha_init: float = 0.2
    learn_alpha: bool = True
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    awac_lambda: float = 1.0
    weight_clip: float = 20.0
    best_model_name: str = "best_model.pt"
    history_csv_name: str = "training_history.csv"
```

**解释：**

- `@dataclass`：让配置类自动生成初始化方法，写超参数更方便。
- `class Config:`：配置类，集中保存数据路径、学习率、batch_size、网络结构、设备和输出目录等超参数。
- `hidden_sizes: list = field(default_factory=lambda: [256, 256])`：hidden_sizes 的作用：隐藏层宽度列表，用来决定 MLP 结构。
- `gamma: float = 0.99`：gamma 的作用：折扣因子，控制未来奖励在 TD 目标中的权重。
- `tau: float = 0.005`：tau 的作用：目标网络软更新系数。
- `batch_size: int = 256`：batch_size 的作用：每个 batch 的样本数。
- `num_epochs: int = 150`：num_epochs 的作用：训练轮数。
- `val_ratio: float = 0.1`：val_ratio 的作用：验证集比例。
- `seed: int = 42`：seed 的作用：随机种子，用于复现实验。
- `weight_decay: float = 1e-4`：weight_decay 的作用：L2 正则化系数，用于减轻过拟合。
- `grad_clip: float = 5.0`：grad_clip 的作用：梯度裁剪阈值，防止梯度爆炸。
- `device: str = "cuda" if torch.cuda.is_available() else "cpu"`：device 的作用：计算设备，优先使用 GPU，否则使用 CPU。
- `data_path: str = "halfcheetah_medium-v2.hdf5"`：data_path 的作用：数据集路径，指向 HDF5 离线强化学习数据文件。
- `save_dir: str = "checkpoints/SAC"`：save_dir 的作用：模型、日志和图片的输出目录。
- `reward_scale: bool = True`：reward_scale 的作用：是否对奖励做标准化或缩放。
- `weight_clip: float = 20.0`：weight_clip 的作用：优势权重裁剪上限，避免少数样本权重过大。
- `best_model_name: str = "best_model.pt"`：best_model_name 的作用：最佳模型 checkpoint 文件名。

### 函数 `build_config_from_args`
**代码：**

```python
def build_config_from_args():
    parser = argparse.ArgumentParser(description="Offline AWAC-SAC Training")
    parser.add_argument("--env-name", type=str, default="halfcheetah-medium-v2")
    parser.add_argument("--data-path", type=str, default="./halfcheetah_medium-v2.hdf5")
    parser.add_argument("--save-dir", type=str, default="checkpoints/SAC")
    parser.add_argument("--hidden-sizes", type=str, default="256,256")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--lr-actor", type=float, default=3e-4)
    parser.add_argument("--lr-critic", type=float, default=3e-4)
    parser.add_argument("--lr-alpha", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-epochs", type=int, default=150)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--alpha-init", type=float, default=0.2)
    parser.add_argument("--log-std-min", type=float, default=-5.0)
    parser.add_argument("--log-std-max", type=float, default=2.0)
    parser.add_argument("--awac-lambda", type=float, default=1.0)
    parser.add_argument("--weight-clip", type=float, default=20.0)
    parser.add_argument("--best-model-name", type=str, default="best_model.pt")
    parser.add_argument("--history-csv-name", type=str, default="training_history.csv")
    parser.add_argument("--reward-scale", dest="reward_scale", action="store_true")
    parser.add_argument("--no-reward-scale", dest="reward_scale", action="store_false")
    parser.set_defaults(reward_scale=True)
    parser.add_argument("--learn-alpha", dest="learn_alpha", action="store_true")
    parser.add_argument("--no-learn-alpha", dest="learn_alpha", action="store_false")
    parser.set_defaults(learn_alpha=True)
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
    if args.alpha_init <= 0:
        raise ValueError("--alpha-init 必须 > 0")
    if args.awac_lambda <= 0:
        raise ValueError("--awac-lambda 必须 > 0")
    if args.weight_clip <= 0:
        raise ValueError("--weight-clip 必须 > 0")
    cfg = Config(
        env_name=args.env_name,
        hidden_sizes=parse_hidden_sizes(args.hidden_sizes),
        gamma=args.gamma,
        tau=args.tau,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        lr_alpha=args.lr_alpha,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        val_ratio=args.val_ratio,
        seed=args.seed,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        num_workers=args.num_workers,
        device=device,
        data_path=args.data_path,
        save_dir=args.save_dir,
        reward_scale=args.reward_scale,
        alpha_init=args.alpha_init,
        learn_alpha=args.learn_alpha,
        log_std_min=args.log_std_min,
        log_std_max=args.log_std_max,
        awac_lambda=args.awac_lambda,
        weight_clip=args.weight_clip,
        best_model_name=args.best_model_name,
        history_csv_name=args.history_csv_name,
    )
    ensure_dirs(cfg.save_dir)
    save_cfg = asdict(cfg)
    with open(os.path.join(cfg.save_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(save_cfg, f, ensure_ascii=False, indent=2)
    return cfg
```

**解释：**

- `def build_config_from_args():`：函数作用：解析命令行参数，生成统一配置对象。
- `device = "cpu"`：device 的作用：计算设备，优先使用 GPU，否则使用 CPU。
- `device = "cuda"`：device 的作用：计算设备，优先使用 GPU，否则使用 CPU。
- `device = "cuda" if torch.cuda.is_available() else "cpu"`：device 的作用：计算设备，优先使用 GPU，否则使用 CPU。
- `hidden_sizes=parse_hidden_sizes(args.hidden_sizes),`：hidden_sizes 的作用：隐藏层宽度列表，用来决定 MLP 结构。
- `gamma=args.gamma,`：gamma 的作用：折扣因子，控制未来奖励在 TD 目标中的权重。
- `tau=args.tau,`：tau 的作用：目标网络软更新系数。
- `lr_actor=args.lr_actor,`：创建策略网络。它输入状态，输出动作，是后续训练要优化的核心模型。
- `batch_size=args.batch_size,`：batch_size 的作用：每个 batch 的样本数。
- `num_epochs=args.num_epochs,`：num_epochs 的作用：训练轮数。
- `val_ratio=args.val_ratio,`：val_ratio 的作用：验证集比例。
- `seed=args.seed,`：seed 的作用：随机种子，用于复现实验。
- `grad_clip=args.grad_clip,`：grad_clip 的作用：梯度裁剪阈值，防止梯度爆炸。
- `device=device,`：device 的作用：计算设备，优先使用 GPU，否则使用 CPU。
- `data_path=args.data_path,`：data_path 的作用：数据集路径，指向 HDF5 离线强化学习数据文件。
- `save_dir=args.save_dir,`：save_dir 的作用：模型、日志和图片的输出目录。
- `reward_scale=args.reward_scale,`：reward_scale 的作用：是否对奖励做标准化或缩放。
- `weight_clip=args.weight_clip,`：weight_clip 的作用：优势权重裁剪上限，避免少数样本权重过大。
- `best_model_name=args.best_model_name,`：best_model_name 的作用：最佳模型 checkpoint 文件名。
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
    def __init__(self, input_dim, output_dim, hidden_sizes):
        super().__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        layers.append(nn.Linear(last_dim, output_dim))
        self.model = nn.Sequential(*layers)
```

**解释：**

- `def __init__(self, input_dim, output_dim, hidden_sizes):`：函数作用：初始化当前类，创建网络层、优化器或训练需要的成员变量。

### 方法 `MLP.forward`
**代码：**

```python
    def forward(self, x):
        return self.model(x)
```

**解释：**

- `def forward(self, x):`：函数作用：定义网络前向传播，说明输入张量如何变成输出。
- `return self.model(x)`：返回训练结果、数据和标准化参数，供主流程继续评估或加载模型。

### 类 `GaussianActor`
**代码：**

```python
class GaussianActor(nn.Module):
```

**解释：**

- `class GaussianActor(nn.Module):`：SAC 的高斯策略网络，输出动作分布的均值和标准差。

### 方法 `GaussianActor.__init__`
**代码：**

```python
    def __init__(self, state_dim, action_dim, hidden_sizes, log_std_min=-5.0, log_std_max=2.0):
        super().__init__()
        layers = []
        last_dim = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        self.trunk = nn.Sequential(*layers)
        self.mu_layer = nn.Linear(last_dim, action_dim)
        self.log_std_layer = nn.Linear(last_dim, action_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.eps = 1e-6
```

**解释：**

- `def __init__(self, state_dim, action_dim, hidden_sizes, log_std_min=-5.0, log_std_max=2.0):`：函数作用：初始化当前类，创建网络层、优化器或训练需要的成员变量。

### 方法 `GaussianActor.forward`
**代码：**

```python
    def forward(self, state):
        h = self.trunk(state)
        mu = self.mu_layer(h)
        log_std = self.log_std_layer(h)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std
```

**解释：**

- `def forward(self, state):`：函数作用：定义网络前向传播，说明输入张量如何变成输出。
- `log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)`：做数值裁剪，防止动作、标准差或权重超出安全范围。
- `return mu, log_std`：返回训练结果、数据和标准化参数，供主流程继续评估或加载模型。

### 方法 `GaussianActor.sample`
**代码：**

```python
    def sample(self, state):
        mu, log_std = self(state)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        noise = torch.randn_like(mu)
        pre_tanh = mu + std * noise
        action = torch.tanh(pre_tanh)
        log_prob = gaussian_log_prob(noise, log_std)
        correction = torch.log(torch.clamp(1 - action.pow(2), min=self.eps))
        log_prob = log_prob - correction.sum(dim=-1, keepdim=True)
        log_prob = torch.clamp(log_prob, min=-100.0, max=100.0)
        mu_action = torch.tanh(mu)
        return action, log_prob, mu_action
```

**解释：**

- `def sample(self, state):`：函数作用：从策略分布中采样动作，并返回动作概率相关信息。
- `log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)`：做数值裁剪，防止动作、标准差或权重超出安全范围。
- `noise = torch.randn_like(mu)`：生成并裁剪目标动作噪声，这是 TD3 的 target policy smoothing。
- `correction = torch.log(torch.clamp(1 - action.pow(2), min=self.eps))`：做数值裁剪，防止动作、标准差或权重超出安全范围。
- `log_prob = torch.clamp(log_prob, min=-100.0, max=100.0)`：做数值裁剪，防止动作、标准差或权重超出安全范围。
- `return action, log_prob, mu_action`：返回训练结果、数据和标准化参数，供主流程继续评估或加载模型。

### 方法 `GaussianActor.log_prob_of_action`
**代码：**

```python
    def log_prob_of_action(self, state, action):
        mu, log_std = self(state)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        clipped_action = torch.clamp(action, -1.0 + self.eps, 1.0 - self.eps)
        pre_tanh = 0.5 * (torch.log1p(clipped_action) - torch.log1p(-clipped_action))
        noise = (pre_tanh - mu) / (std + self.eps)
        log_prob = gaussian_log_prob(noise, log_std)
        correction = torch.log(torch.clamp(1 - clipped_action.pow(2), min=self.eps))
        log_prob = log_prob - correction.sum(dim=-1, keepdim=True)
        log_prob = torch.clamp(log_prob, min=-100.0, max=100.0)
        return log_prob
```

**解释：**

- `def log_prob_of_action(self, state, action):`：函数作用：封装一段可复用逻辑，供主训练流程调用。
- `log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)`：做数值裁剪，防止动作、标准差或权重超出安全范围。
- `clipped_action = torch.clamp(action, -1.0 + self.eps, 1.0 - self.eps)`：做数值裁剪，防止动作、标准差或权重超出安全范围。
- `noise = (pre_tanh - mu) / (std + self.eps)`：生成并裁剪目标动作噪声，这是 TD3 的 target policy smoothing。
- `correction = torch.log(torch.clamp(1 - clipped_action.pow(2), min=self.eps))`：做数值裁剪，防止动作、标准差或权重超出安全范围。
- `log_prob = torch.clamp(log_prob, min=-100.0, max=100.0)`：做数值裁剪，防止动作、标准差或权重超出安全范围。
- `return log_prob`：返回训练结果、数据和标准化参数，供主流程继续评估或加载模型。

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
    def __init__(self, state_dim, action_dim, hidden_sizes):
        super().__init__()
        self.net = MLP(state_dim + action_dim, 1, hidden_sizes)
```

**解释：**

- `def __init__(self, state_dim, action_dim, hidden_sizes):`：函数作用：初始化当前类，创建网络层、优化器或训练需要的成员变量。

### 方法 `Critic.forward`
**代码：**

```python
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)
```

**解释：**

- `def forward(self, state, action):`：函数作用：定义网络前向传播，说明输入张量如何变成输出。
- `return self.net(x)`：返回训练结果、数据和标准化参数，供主流程继续评估或加载模型。

### 函数 `load_hdf5_dataset`
**代码：**

```python
def load_hdf5_dataset(path):
    with h5py.File(path, "r") as f:
        obs = torch.tensor(f["observations"][:], dtype=torch.float32)
        actions = torch.tensor(f["actions"][:], dtype=torch.float32)
        rewards = torch.tensor(f["rewards"][:], dtype=torch.float32)
        next_obs = torch.tensor(f["next_observations"][:], dtype=torch.float32)
        terminals = torch.tensor(f["terminals"][:], dtype=torch.float32)
    dataset = TensorDataset(obs, actions, rewards, next_obs, terminals)
    state_dim = obs.shape[1]
    action_dim = actions.shape[1]
    reward_mean = rewards.mean()
    reward_std = rewards.std().clamp(min=1e-6)
    return dataset, state_dim, action_dim, reward_mean, reward_std
```

**解释：**

- `def load_hdf5_dataset(path):`：函数作用：读取离线 HDF5 数据集，并构造训练集、验证集或 DataLoader。
- `state_dim = obs.shape[1]`：state_dim 的作用：状态维度，本项目通常是 17。
- `action_dim = actions.shape[1]`：action_dim 的作用：动作维度，本项目通常是 6。
- `return dataset, state_dim, action_dim, reward_mean, reward_std`：返回训练结果、数据和标准化参数，供主流程继续评估或加载模型。

### 函数 `split_dataset`
**代码：**

```python
def split_dataset(dataset, val_ratio=0.1, seed=42):
    total_size = len(dataset)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    return train_dataset, val_dataset
```

**解释：**

- `def split_dataset(dataset, val_ratio=0.1, seed=42):`：函数作用：封装一段可复用逻辑，供主训练流程调用。
- `return train_dataset, val_dataset`：返回训练结果、数据和标准化参数，供主流程继续评估或加载模型。

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

### 类 `OfflineAWACSACAgent`
**代码：**

```python
class OfflineAWACSACAgent:
```

**解释：**

- `class OfflineAWACSACAgent:`：训练器/智能体类，封装 actor、critic、优化器以及 train_step/eval_step 等核心流程。

### 方法 `OfflineAWACSACAgent.__init__`
**代码：**

```python
    def __init__(self, cfg, state_dim, action_dim):
        self.cfg = cfg
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.eps = 1e-6
        self.max_tensor_abs = 1e6
        self.max_logprob_abs = 100.0
        self.max_log_alpha = 5.0
        self.min_log_alpha = -10.0
        self.actor = GaussianActor(
            state_dim, action_dim, cfg.hidden_sizes,
            cfg.log_std_min, cfg.log_std_max
        ).to(cfg.device)
        self.critic1 = Critic(state_dim, action_dim, cfg.hidden_sizes).to(cfg.device)
        self.critic2 = Critic(state_dim, action_dim, cfg.hidden_sizes).to(cfg.device)
        self.target_critic1 = Critic(state_dim, action_dim, cfg.hidden_sizes).to(cfg.device)
        self.target_critic2 = Critic(state_dim, action_dim, cfg.hidden_sizes).to(cfg.device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=cfg.lr_actor,
            weight_decay=cfg.weight_decay
        )
        self.critic1_optimizer = optim.Adam(
            self.critic1.parameters(),
            lr=cfg.lr_critic,
            weight_decay=cfg.weight_decay
        )
        self.critic2_optimizer = optim.Adam(
            self.critic2.parameters(),
            lr=cfg.lr_critic,
            weight_decay=cfg.weight_decay
        )
        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.tensor(
            np.log(cfg.alpha_init),
            dtype=torch.float32,
            device=cfg.device,
            requires_grad=True
        )
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=cfg.lr_alpha)
        self.actor_scheduler = None
        self.critic1_scheduler = None
        self.critic2_scheduler = None
```

**解释：**

- `def __init__(self, cfg, state_dim, action_dim):`：函数作用：初始化当前类，创建网络层、优化器或训练需要的成员变量。
- `self.actor = GaussianActor(`：创建策略网络。它输入状态，输出动作，是后续训练要优化的核心模型。
- `).to(cfg.device)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `self.critic1 = Critic(state_dim, action_dim, cfg.hidden_sizes).to(cfg.device)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `self.critic2 = Critic(state_dim, action_dim, cfg.hidden_sizes).to(cfg.device)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `self.target_critic1 = Critic(state_dim, action_dim, cfg.hidden_sizes).to(cfg.device)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `self.target_critic2 = Critic(state_dim, action_dim, cfg.hidden_sizes).to(cfg.device)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `self.actor_optimizer = optim.Adam(`：创建 Adam 优化器，后面根据梯度更新网络参数。
- `self.critic1_optimizer = optim.Adam(`：创建 Adam 优化器，后面根据梯度更新网络参数。
- `self.critic2_optimizer = optim.Adam(`：创建 Adam 优化器，后面根据梯度更新网络参数。
- `device=cfg.device,`：device 的作用：计算设备，优先使用 GPU，否则使用 CPU。
- `self.alpha_optimizer = optim.Adam([self.log_alpha], lr=cfg.lr_alpha)`：创建 Adam 优化器，后面根据梯度更新网络参数。
- `self.actor_scheduler = None`：创建学习率调度器，让学习率随训练轮数变化，后期训练更稳定。
- `self.critic1_scheduler = None`：创建学习率调度器，让学习率随训练轮数变化，后期训练更稳定。
- `self.critic2_scheduler = None`：创建学习率调度器，让学习率随训练轮数变化，后期训练更稳定。

### 方法 `OfflineAWACSACAgent.alpha`
**代码：**

```python
    @property
    def alpha(self):
        return self.log_alpha.exp()
```

**解释：**

- `def alpha(self):`：函数作用：封装一段可复用逻辑，供主训练流程调用。
- `return self.log_alpha.exp()`：返回训练结果、数据和标准化参数，供主流程继续评估或加载模型。

### 方法 `OfflineAWACSACAgent.set_schedulers`
**代码：**

```python
    def set_schedulers(self, actor_scheduler, critic1_scheduler, critic2_scheduler):
        self.actor_scheduler = actor_scheduler
        self.critic1_scheduler = critic1_scheduler
        self.critic2_scheduler = critic2_scheduler
```

**解释：**

- `def set_schedulers(self, actor_scheduler, critic1_scheduler, critic2_scheduler):`：函数作用：封装一段可复用逻辑，供主训练流程调用。
- `self.actor_scheduler = actor_scheduler`：创建学习率调度器，让学习率随训练轮数变化，后期训练更稳定。
- `self.critic1_scheduler = critic1_scheduler`：创建学习率调度器，让学习率随训练轮数变化，后期训练更稳定。
- `self.critic2_scheduler = critic2_scheduler`：创建学习率调度器，让学习率随训练轮数变化，后期训练更稳定。

### 方法 `OfflineAWACSACAgent._compute_awac_weights`
**代码：**

```python
    def _compute_awac_weights(self, data_q, v_pi):
        advantage = data_q - v_pi
        adv_scaled = advantage / max(self.cfg.awac_lambda, self.eps)
        max_log_weight = math.log(max(self.cfg.weight_clip, 1.0))
        adv_scaled = torch.clamp(adv_scaled, min=-20.0, max=max_log_weight)
        weights = torch.exp(adv_scaled)
        weights = sanitize_tensor(weights, 0.0, self.cfg.weight_clip)
        return advantage, weights
```

**解释：**

- `def _compute_awac_weights(self, data_q, v_pi):`：函数作用：封装一段可复用逻辑，供主训练流程调用。
- `adv_scaled = torch.clamp(adv_scaled, min=-20.0, max=max_log_weight)`：做数值裁剪，防止动作、标准差或权重超出安全范围。
- `return advantage, weights`：返回训练结果、数据和标准化参数，供主流程继续评估或加载模型。

### 方法 `OfflineAWACSACAgent.train_step`
**代码：**

```python
    def train_step(self, batch, obs_mean, obs_std, reward_mean, reward_std):
        state, action, reward, next_state, done = [x.to(self.cfg.device, non_blocking=True) for x in batch]
        state = (state - obs_mean) / obs_std
        next_state = (next_state - obs_mean) / obs_std
        if self.cfg.reward_scale:
            reward = (reward - reward_mean) / reward_std
        reward = reward.unsqueeze(-1)
        done = done.unsqueeze(-1)
        # -------- Critic update --------
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_state)
            next_log_prob = torch.clamp(next_log_prob, min=-self.max_logprob_abs, max=self.max_logprob_abs)
            target_q1 = self.target_critic1(next_state, next_action)
            target_q2 = self.target_critic2(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha.detach() * next_log_prob
            target_q = reward + self.cfg.gamma * (1 - done) * target_q
            target_q = sanitize_tensor(target_q, -self.max_tensor_abs, self.max_tensor_abs)
        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)
        if (not is_finite_tensor(current_q1)) or (not is_finite_tensor(current_q2)):
            raise RuntimeError("critic 输出出现 NaN/Inf，请减小 lr 或检查数据。")
        critic1_loss = nn.functional.mse_loss(current_q1, target_q)
        critic2_loss = nn.functional.mse_loss(current_q2, target_q)
        if (not is_finite_tensor(critic1_loss)) or (not is_finite_tensor(critic2_loss)):
            raise RuntimeError("critic loss 出现 NaN/Inf。")
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        nn.utils.clip_grad_norm_(self.critic1.parameters(), self.cfg.grad_clip)
        self.critic1_optimizer.step()
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        nn.utils.clip_grad_norm_(self.critic2.parameters(), self.cfg.grad_clip)
        self.critic2_optimizer.step()
        # -------- Actor update: AWAC style --------
        with torch.no_grad():
            data_q1 = self.critic1(state, action)
            data_q2 = self.critic2(state, action)
            data_q = torch.min(data_q1, data_q2)
            pi_action, _, _ = self.actor.sample(state)
            pi_q1 = self.critic1(state, pi_action)
            pi_q2 = self.critic2(state, pi_action)
            v_pi = torch.min(pi_q1, pi_q2)
            advantage, weights = self._compute_awac_weights(data_q, v_pi)
        log_prob_data = self.actor.log_prob_of_action(state, action)
        log_prob_data = torch.clamp(log_prob_data, min=-self.max_logprob_abs, max=self.max_logprob_abs)
        actor_loss = -(weights * log_prob_data).mean()
        if (not is_finite_tensor(weights)) or (not is_finite_tensor(log_prob_data)) or (not is_finite_tensor(actor_loss)):
            raise RuntimeError("actor 更新出现 NaN/Inf。")
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_clip)
        self.actor_optimizer.step()
        # -------- Alpha update --------
        alpha_loss = torch.tensor(0.0, device=self.cfg.device)
        if self.cfg.learn_alpha:
            _, log_prob, _ = self.actor.sample(state)
            log_prob = torch.clamp(log_prob, min=-self.max_logprob_abs, max=self.max_logprob_abs)
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            if not is_finite_tensor(alpha_loss):
                raise RuntimeError("alpha_loss 出现 NaN/Inf。")
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            with torch.no_grad():
                self.log_alpha.data.clamp_(min=self.min_log_alpha, max=self.max_log_alpha)
        # -------- Soft update --------
        soft_update(self.critic1, self.target_critic1, self.cfg.tau)
        soft_update(self.critic2, self.target_critic2, self.cfg.tau)
        return {
            "critic1_loss": critic1_loss.item(),
            "critic2_loss": critic2_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item() if self.cfg.learn_alpha else 0.0,
            "alpha": self.alpha.item(),
            "reward": reward.mean().item(),
            "Q1": current_q1.mean().item(),
            "Q2": current_q2.mean().item(),
            "log_prob_data": log_prob_data.mean().item(),
            "advantage": advantage.mean().item(),
            "weight": weights.mean().item(),
        }
```

**解释：**

- `def train_step(self, batch, obs_mean, obs_std, reward_mean, reward_std):`：函数作用：执行一次 SAC 更新：训练双 critic、更新随机策略 actor，并处理熵温度或 BC/AWAC 正则。
- `state, action, reward, next_state, done = [x.to(self.cfg.device, non_blocking=True) for x in batch]`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `state = (state - obs_mean) / obs_std`：对当前状态做标准化：减去训练集均值，再除以训练集标准差。
- `next_state = (next_state - obs_mean) / obs_std`：对下一状态做同样标准化，保证 TD 目标计算使用相同输入尺度。
- `if self.cfg.reward_scale:`：如果启用奖励标准化/缩放，就对 reward 做归一化处理。
- `reward = (reward - reward_mean) / reward_std`：对奖励做标准化，减少奖励尺度过大或过小带来的训练不稳定。
- `with torch.no_grad():`：进入不记录梯度的区域，通常用于计算 TD 目标或验证指标。
- `next_action, next_log_prob, _ = self.actor.sample(next_state)`：用目标 actor 计算下一状态动作，并加入噪声后裁剪到动作范围内。
- `next_log_prob = torch.clamp(next_log_prob, min=-self.max_logprob_abs, max=self.max_logprob_abs)`：做数值裁剪，防止动作、标准差或权重超出安全范围。
- `target_q = torch.min(target_q1, target_q2) - self.alpha.detach() * next_log_prob`：用目标 critic 估计下一状态动作的 Q 值。
- `target_q = reward + self.cfg.gamma * (1 - done) * target_q`：构造 TD 目标：即时奖励加上折扣后的下一状态价值；done 为 1 时不再加未来价值。
- `self.critic1_optimizer.zero_grad()`：清空上一轮梯度，避免梯度累积影响当前 batch。
- `critic1_loss.backward()`：对当前损失反向传播，计算网络参数梯度。
- `nn.utils.clip_grad_norm_(self.critic1.parameters(), self.cfg.grad_clip)`：对梯度做裁剪，防止梯度过大导致训练不稳定。
- `self.critic1_optimizer.step()`：优化器根据梯度更新参数。
- `self.critic2_optimizer.zero_grad()`：清空上一轮梯度，避免梯度累积影响当前 batch。
- `critic2_loss.backward()`：对当前损失反向传播，计算网络参数梯度。
- `nn.utils.clip_grad_norm_(self.critic2.parameters(), self.cfg.grad_clip)`：对梯度做裁剪，防止梯度过大导致训练不稳定。
- `self.critic2_optimizer.step()`：优化器根据梯度更新参数。
- `log_prob_data = torch.clamp(log_prob_data, min=-self.max_logprob_abs, max=self.max_logprob_abs)`：做数值裁剪，防止动作、标准差或权重超出安全范围。
- `actor_loss = -(weights * log_prob_data).mean()`：合成 actor 损失，通常由 Q 项、熵项或 BC 正则项组成。
- `self.actor_optimizer.zero_grad()`：清空上一轮梯度，避免梯度累积影响当前 batch。
- `actor_loss.backward()`：对当前损失反向传播，计算网络参数梯度。
- `nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_clip)`：对梯度做裁剪，防止梯度过大导致训练不稳定。
- `self.actor_optimizer.step()`：优化器根据梯度更新参数。
- `log_prob = torch.clamp(log_prob, min=-self.max_logprob_abs, max=self.max_logprob_abs)`：做数值裁剪，防止动作、标准差或权重超出安全范围。
- `self.alpha_optimizer.zero_grad()`：清空上一轮梯度，避免梯度累积影响当前 batch。
- `alpha_loss.backward()`：对当前损失反向传播，计算网络参数梯度。
- `self.alpha_optimizer.step()`：优化器根据梯度更新参数。
- `soft_update(self.critic1, self.target_critic1, self.cfg.tau)`：软更新目标 critic，让目标 Q 网络缓慢跟随当前 critic。
- `soft_update(self.critic2, self.target_critic2, self.cfg.tau)`：软更新目标 critic，让目标 Q 网络缓慢跟随当前 critic。
- `return {`：返回本次更新或评估的指标字典，供训练日志记录。

### 方法 `OfflineAWACSACAgent.eval_step`
**代码：**

```python
    @torch.no_grad()
    def eval_step(self, batch, obs_mean, obs_std, reward_mean, reward_std):
        state, action, reward, next_state, done = [x.to(self.cfg.device, non_blocking=True) for x in batch]
        state = (state - obs_mean) / obs_std
        next_state = (next_state - obs_mean) / obs_std
        if self.cfg.reward_scale:
            reward = (reward - reward_mean) / reward_std
        reward = reward.unsqueeze(-1)
        done = done.unsqueeze(-1)
        next_action, next_log_prob, _ = self.actor.sample(next_state)
        next_log_prob = torch.clamp(next_log_prob, min=-self.max_logprob_abs, max=self.max_logprob_abs)
        target_q1 = self.target_critic1(next_state, next_action)
        target_q2 = self.target_critic2(next_state, next_action)
        target_q = torch.min(target_q1, target_q2) - self.alpha.detach() * next_log_prob
        target_q = reward + self.cfg.gamma * (1 - done) * target_q
        target_q = sanitize_tensor(target_q, -self.max_tensor_abs, self.max_tensor_abs)
        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)
        current_q1 = sanitize_tensor(current_q1, -self.max_tensor_abs, self.max_tensor_abs)
        current_q2 = sanitize_tensor(current_q2, -self.max_tensor_abs, self.max_tensor_abs)
        critic1_loss = nn.functional.mse_loss(current_q1, target_q)
        critic2_loss = nn.functional.mse_loss(current_q2, target_q)
        data_q1 = self.critic1(state, action)
        data_q2 = self.critic2(state, action)
        data_q = torch.min(data_q1, data_q2)
        pi_action, _, _ = self.actor.sample(state)
        pi_q1 = self.critic1(state, pi_action)
        pi_q2 = self.critic2(state, pi_action)
        v_pi = torch.min(pi_q1, pi_q2)
        advantage, weights = self._compute_awac_weights(data_q, v_pi)
        log_prob_data = self.actor.log_prob_of_action(state, action)
        log_prob_data = torch.clamp(log_prob_data, min=-self.max_logprob_abs, max=self.max_logprob_abs)
        actor_loss = -(weights * log_prob_data).mean()
        return {
            "critic1_loss": critic1_loss.item(),
            "critic2_loss": critic2_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha.item(),
            "reward": reward.mean().item(),
            "Q1": current_q1.mean().item(),
            "Q2": current_q2.mean().item(),
            "log_prob_data": log_prob_data.mean().item(),
            "advantage": advantage.mean().item(),
            "weight": weights.mean().item(),
        }
```

**解释：**

- `@torch.no_grad()`：装饰器作用：该函数执行时不记录梯度，适合验证、评估或构造目标值。
- `def eval_step(self, batch, obs_mean, obs_std, reward_mean, reward_std):`：函数作用：在指定数据集或 batch 上评估模型，不更新参数。
- `state, action, reward, next_state, done = [x.to(self.cfg.device, non_blocking=True) for x in batch]`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `state = (state - obs_mean) / obs_std`：对当前状态做标准化：减去训练集均值，再除以训练集标准差。
- `next_state = (next_state - obs_mean) / obs_std`：对下一状态做同样标准化，保证 TD 目标计算使用相同输入尺度。
- `if self.cfg.reward_scale:`：如果启用奖励标准化/缩放，就对 reward 做归一化处理。
- `reward = (reward - reward_mean) / reward_std`：对奖励做标准化，减少奖励尺度过大或过小带来的训练不稳定。
- `next_action, next_log_prob, _ = self.actor.sample(next_state)`：用目标 actor 计算下一状态动作，并加入噪声后裁剪到动作范围内。
- `next_log_prob = torch.clamp(next_log_prob, min=-self.max_logprob_abs, max=self.max_logprob_abs)`：做数值裁剪，防止动作、标准差或权重超出安全范围。
- `target_q = torch.min(target_q1, target_q2) - self.alpha.detach() * next_log_prob`：用目标 critic 估计下一状态动作的 Q 值。
- `target_q = reward + self.cfg.gamma * (1 - done) * target_q`：构造 TD 目标：即时奖励加上折扣后的下一状态价值；done 为 1 时不再加未来价值。
- `log_prob_data = torch.clamp(log_prob_data, min=-self.max_logprob_abs, max=self.max_logprob_abs)`：做数值裁剪，防止动作、标准差或权重超出安全范围。
- `actor_loss = -(weights * log_prob_data).mean()`：合成 actor 损失，通常由 Q 项、熵项或 BC 正则项组成。
- `return {`：返回本次更新或评估的指标字典，供训练日志记录。

### 方法 `OfflineAWACSACAgent.save_best`
**代码：**

```python
    def save_best(self, obs_mean, obs_std, reward_mean, reward_std, epoch, best_val_loss):
        torch.save({
            "actor_state_dict": self.actor.state_dict(),
            "critic1_state_dict": self.critic1.state_dict(),
            "critic2_state_dict": self.critic2.state_dict(),
            "obs_mean": obs_mean.cpu(),
            "obs_std": obs_std.cpu(),
            "reward_mean": reward_mean.cpu(),
            "reward_std": reward_std.cpu(),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_sizes": list(self.cfg.hidden_sizes),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "alpha": self.alpha.item(),
            "reward_scale": self.cfg.reward_scale,
            "log_std_min": self.cfg.log_std_min,
            "log_std_max": self.cfg.log_std_max,
            "awac_lambda": self.cfg.awac_lambda,
            "weight_clip": self.cfg.weight_clip,
        }, os.path.join(self.cfg.save_dir, self.cfg.best_model_name))
```

**解释：**

- `def save_best(self, obs_mean, obs_std, reward_mean, reward_std, epoch, best_val_loss):`：函数作用：保存当前最佳模型，同时保存标准化参数和网络结构信息，便于之后恢复。

### 函数 `train_sac`
**代码：**

```python
def train_sac(agent, train_loader, val_loader, obs_mean, obs_std, reward_mean, reward_std, cfg):
    history = {
        "epoch": [],
        "train_critic1_loss": [],
        "train_critic2_loss": [],
        "train_actor_loss": [],
        "train_reward": [],
        "train_Q1": [],
        "train_Q2": [],
        "train_log_prob_data": [],
        "train_alpha": [],
        "train_advantage": [],
        "train_weight": [],
        "val_critic1_loss": [],
        "val_critic2_loss": [],
        "val_actor_loss": [],
        "val_reward": [],
        "val_Q1": [],
        "val_Q2": [],
        "val_log_prob_data": [],
        "val_alpha": [],
        "val_advantage": [],
        "val_weight": [],
        "actor_lr": [],
        "critic1_lr": [],
        "critic2_lr": []
    }
    best_val_loss = float("inf")
    for epoch in range(1, cfg.num_epochs + 1):
        train_metrics_sum = {
            "critic1_loss": 0.0,
            "critic2_loss": 0.0,
            "actor_loss": 0.0,
            "reward": 0.0,
            "Q1": 0.0,
            "Q2": 0.0,
            "log_prob_data": 0.0,
            "alpha": 0.0,
            "advantage": 0.0,
            "weight": 0.0
        }
        for batch in train_loader:
            out = agent.train_step(batch, obs_mean, obs_std, reward_mean, reward_std)
            for k in train_metrics_sum:
                train_metrics_sum[k] += out[k]
        num_train_batches = len(train_loader)
        train_metrics = {k: v / num_train_batches for k, v in train_metrics_sum.items()}
        val_metrics_sum = {
            "critic1_loss": 0.0,
            "critic2_loss": 0.0,
            "actor_loss": 0.0,
            "reward": 0.0,
            "Q1": 0.0,
            "Q2": 0.0,
            "log_prob_data": 0.0,
            "alpha": 0.0,
            "advantage": 0.0,
            "weight": 0.0
        }
        for batch in val_loader:
            out = agent.eval_step(batch, obs_mean, obs_std, reward_mean, reward_std)
            for k in val_metrics_sum:
                val_metrics_sum[k] += out[k]
        num_val_batches = len(val_loader)
        val_metrics = {k: v / num_val_batches for k, v in val_metrics_sum.items()}
        actor_lr = agent.actor_optimizer.param_groups[0]["lr"]
        critic1_lr = agent.critic1_optimizer.param_groups[0]["lr"]
        critic2_lr = agent.critic2_optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch}/{cfg.num_epochs} | "
            f"Train Actor {train_metrics['actor_loss']:.6f} | "
            f"Train C1 {train_metrics['critic1_loss']:.6f} | "
            f"Train C2 {train_metrics['critic2_loss']:.6f} | "
            f"Val Actor {val_metrics['actor_loss']:.6f} | "
            f"Val C1 {val_metrics['critic1_loss']:.6f} | "
            f"Val C2 {val_metrics['critic2_loss']:.6f} | "
            f"Alpha {train_metrics['alpha']:.4f}"
        )
        history["epoch"].append(epoch)
        history["train_critic1_loss"].append(train_metrics["critic1_loss"])
        history["train_critic2_loss"].append(train_metrics["critic2_loss"])
        history["train_actor_loss"].append(train_metrics["actor_loss"])
        history["train_reward"].append(train_metrics["reward"])
        history["train_Q1"].append(train_metrics["Q1"])
        history["train_Q2"].append(train_metrics["Q2"])
        history["train_log_prob_data"].append(train_metrics["log_prob_data"])
        history["train_alpha"].append(train_metrics["alpha"])
        history["train_advantage"].append(train_metrics["advantage"])
        history["train_weight"].append(train_metrics["weight"])
        history["val_critic1_loss"].append(val_metrics["critic1_loss"])
        history["val_critic2_loss"].append(val_metrics["critic2_loss"])
        history["val_actor_loss"].append(val_metrics["actor_loss"])
        history["val_reward"].append(val_metrics["reward"])
        history["val_Q1"].append(val_metrics["Q1"])
        history["val_Q2"].append(val_metrics["Q2"])
        history["val_log_prob_data"].append(val_metrics["log_prob_data"])
        history["val_alpha"].append(val_metrics["alpha"])
        history["val_advantage"].append(val_metrics["advantage"])
        history["val_weight"].append(val_metrics["weight"])
        history["actor_lr"].append(actor_lr)
        history["critic1_lr"].append(critic1_lr)
        history["critic2_lr"].append(critic2_lr)
        val_total_critic_loss = val_metrics["critic1_loss"] + val_metrics["critic2_loss"]
        if val_total_critic_loss < best_val_loss:
            best_val_loss = val_total_critic_loss
            agent.save_best(obs_mean, obs_std, reward_mean, reward_std, epoch, best_val_loss)
            print(f"Best model saved at epoch {epoch}, val_total_critic_loss={best_val_loss:.6f}")
        if agent.actor_scheduler is not None:
            agent.actor_scheduler.step()
        if agent.critic1_scheduler is not None:
            agent.critic1_scheduler.step()
        if agent.critic2_scheduler is not None:
            agent.critic2_scheduler.step()
    return history
```

**解释：**

- `def train_sac(agent, train_loader, val_loader, obs_mean, obs_std, reward_mean, reward_std, cfg):`：函数作用：封装一段可复用逻辑，供主训练流程调用。
- `history = {`：创建日志容器，用来记录每个 epoch 的训练损失、验证损失和学习率。
- `best_val_loss = float("inf")`：初始化最佳验证指标为无穷大，后面只要验证损失更小就保存模型。
- `for epoch in range(1, cfg.num_epochs + 1):`：开始按 epoch 训练模型，每轮都会训练一次、验证一次，并记录指标。
- `print(`：把当前训练或评估信息输出到终端。
- `history["epoch"].append(epoch)`：记录当前 epoch 的训练指标，后面保存成表格和曲线。
- `history["train_critic1_loss"].append(train_metrics["critic1_loss"])`：记录当前 epoch 的训练指标，后面保存成表格和曲线。
- `history["train_critic2_loss"].append(train_metrics["critic2_loss"])`：记录当前 epoch 的训练指标，后面保存成表格和曲线。
- `history["train_actor_loss"].append(train_metrics["actor_loss"])`：记录当前 epoch 的训练指标，后面保存成表格和曲线。
- `history["train_reward"].append(train_metrics["reward"])`：记录当前 epoch 的训练指标，后面保存成表格和曲线。
- `history["train_Q1"].append(train_metrics["Q1"])`：记录当前 epoch 的训练指标，后面保存成表格和曲线。
- `history["train_Q2"].append(train_metrics["Q2"])`：记录当前 epoch 的训练指标，后面保存成表格和曲线。
- `history["train_log_prob_data"].append(train_metrics["log_prob_data"])`：记录当前 epoch 的训练指标，后面保存成表格和曲线。
- `history["train_alpha"].append(train_metrics["alpha"])`：记录当前 epoch 的训练指标，后面保存成表格和曲线。
- `history["train_advantage"].append(train_metrics["advantage"])`：记录当前 epoch 的训练指标，后面保存成表格和曲线。
- `history["train_weight"].append(train_metrics["weight"])`：记录当前 epoch 的训练指标，后面保存成表格和曲线。
- `history["val_critic1_loss"].append(val_metrics["critic1_loss"])`：记录当前 epoch 的训练指标，后面保存成表格和曲线。
- `history["val_critic2_loss"].append(val_metrics["critic2_loss"])`：记录当前 epoch 的训练指标，后面保存成表格和曲线。
- `history["val_actor_loss"].append(val_metrics["actor_loss"])`：记录当前 epoch 的训练指标，后面保存成表格和曲线。
- `history["val_reward"].append(val_metrics["reward"])`：记录当前 epoch 的训练指标，后面保存成表格和曲线。
- `history["val_Q1"].append(val_metrics["Q1"])`：记录当前 epoch 的训练指标，后面保存成表格和曲线。
- `history["val_Q2"].append(val_metrics["Q2"])`：记录当前 epoch 的训练指标，后面保存成表格和曲线。
- `history["val_log_prob_data"].append(val_metrics["log_prob_data"])`：记录当前 epoch 的训练指标，后面保存成表格和曲线。
- `history["val_alpha"].append(val_metrics["alpha"])`：记录当前 epoch 的训练指标，后面保存成表格和曲线。
- `history["val_advantage"].append(val_metrics["advantage"])`：记录当前 epoch 的训练指标，后面保存成表格和曲线。
- `history["val_weight"].append(val_metrics["weight"])`：记录当前 epoch 的训练指标，后面保存成表格和曲线。
- `history["actor_lr"].append(actor_lr)`：记录当前 epoch 的训练指标，后面保存成表格和曲线。
- `history["critic1_lr"].append(critic1_lr)`：记录当前 epoch 的训练指标，后面保存成表格和曲线。
- `history["critic2_lr"].append(critic2_lr)`：记录当前 epoch 的训练指标，后面保存成表格和曲线。
- `val_total_critic_loss = val_metrics["critic1_loss"] + val_metrics["critic2_loss"]`：计算 critic 的 TD 回归损失，让当前 Q 值逼近 TD 目标。
- `if val_total_critic_loss < best_val_loss:`：如果当前验证结果优于历史最好结果，就保存当前模型。
- `best_val_loss = val_total_critic_loss`：初始化最佳验证指标为无穷大，后面只要验证损失更小就保存模型。
- `agent.save_best(obs_mean, obs_std, reward_mean, reward_std, epoch, best_val_loss)`：保存当前最佳模型，同时保存标准化参数和网络结构。
- `print(f"Best model saved at epoch {epoch}, val_total_critic_loss={best_val_loss:.6f}")`：计算 critic 的 TD 回归损失，让当前 Q 值逼近 TD 目标。
- `agent.actor_scheduler.step()`：优化器根据梯度更新参数。
- `agent.critic1_scheduler.step()`：优化器根据梯度更新参数。
- `agent.critic2_scheduler.step()`：优化器根据梯度更新参数。
- `return history`：返回训练结果、数据和标准化参数，供主流程继续评估或加载模型。

### 函数 `save_history_csv`
**代码：**

```python
def save_history_csv(history, cfg):
    df = pd.DataFrame(history)
    df.to_csv(os.path.join(cfg.save_dir, cfg.history_csv_name), index=False)
    curve_keys = [
        ("train_critic1_loss", "val_critic1_loss", "critic1_loss"),
        ("train_critic2_loss", "val_critic2_loss", "critic2_loss"),
        ("train_actor_loss", "val_actor_loss", "actor_loss"),
        ("train_reward", "val_reward", "reward"),
        ("train_Q1", "val_Q1", "Q1"),
        ("train_Q2", "val_Q2", "Q2"),
        ("train_log_prob_data", "val_log_prob_data", "log_prob_data"),
        ("train_alpha", "val_alpha", "alpha"),
        ("train_advantage", "val_advantage", "advantage"),
        ("train_weight", "val_weight", "weight"),
    ]
    for train_key, val_key, fig_name in curve_keys:
        plt.figure(figsize=(8, 5))
        plt.plot(df["epoch"], df[train_key], label=f"Train {fig_name}")
        plt.plot(df["epoch"], df[val_key], label=f"Val {fig_name}")
        plt.title(fig_name)
        plt.xlabel("epoch")
        plt.ylabel(fig_name)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.save_dir, f"{fig_name}.png"))
        plt.close()
```

**解释：**

- `def save_history_csv(history, cfg):`：函数作用：保存训练日志和曲线，方便观察训练过程和写论文结果。
- `df = pd.DataFrame(history)`：把训练记录转换成 pandas 表格，便于保存 CSV 和画曲线。
- `df.to_csv(os.path.join(cfg.save_dir, cfg.history_csv_name), index=False)`：把表格结果保存成 CSV 文件，方便论文记录、画图或复查。

### 函数 `load_best_actor`
**代码：**

```python
def load_best_actor(cfg):
    ckpt = torch.load(os.path.join(cfg.save_dir, cfg.best_model_name), map_location=cfg.device)
    actor = GaussianActor(
        ckpt["state_dim"],
        ckpt["action_dim"],
        ckpt["hidden_sizes"],
        ckpt["log_std_min"],
        ckpt["log_std_max"]
    ).to(cfg.device)
    actor.load_state_dict(ckpt["actor_state_dict"])
    obs_mean = ckpt["obs_mean"].to(cfg.device)
    obs_std = ckpt["obs_std"].to(cfg.device)
    print(
        f"Loaded best actor from epoch {ckpt['epoch']}, "
        f"best_val_loss={ckpt['best_val_loss']:.6f}"
    )
    return actor, obs_mean, obs_std, ckpt
```

**解释：**

- `def load_best_actor(cfg):`：函数作用：加载训练过程中保存的最佳策略模型，用于评估或推理。
- `actor = GaussianActor(`：创建策略网络。它输入状态，输出动作，是后续训练要优化的核心模型。
- `).to(cfg.device)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `obs_mean = ckpt["obs_mean"].to(cfg.device)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `obs_std = ckpt["obs_std"].to(cfg.device)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `print(`：把当前训练或评估信息输出到终端。
- `return actor, obs_mean, obs_std, ckpt`：返回训练结果、数据和标准化参数，供主流程继续评估或加载模型。

### 函数 `main`
**代码：**

```python
def main():
    cfg = build_config_from_args()
    set_seed(cfg.seed)
    ensure_dirs(cfg.save_dir)
    print("Using config:")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")
    dataset, state_dim, action_dim, reward_mean, reward_std = load_hdf5_dataset(cfg.data_path)
    train_dataset, val_dataset = split_dataset(dataset, cfg.val_ratio, cfg.seed)
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
    obs_mean, obs_std = compute_obs_stats(train_dataset)
    obs_mean = obs_mean.to(cfg.device)
    obs_std = obs_std.to(cfg.device)
    reward_mean = reward_mean.to(cfg.device)
    reward_std = reward_std.to(cfg.device)
    agent = OfflineAWACSACAgent(cfg, state_dim, action_dim)
    actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        agent.actor_optimizer, T_max=cfg.num_epochs
    )
    critic1_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        agent.critic1_optimizer, T_max=cfg.num_epochs
    )
    critic2_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        agent.critic2_optimizer, T_max=cfg.num_epochs
    )
    agent.set_schedulers(actor_scheduler, critic1_scheduler, critic2_scheduler)
    history = train_sac(agent, train_loader, val_loader, obs_mean, obs_std, reward_mean, reward_std, cfg)
    save_history_csv(history, cfg)
    load_best_actor(cfg)
    print("Training complete, all CSVs and plots saved in", cfg.save_dir)
```

**解释：**

- `def main():`：函数作用：脚本主入口，负责读取配置、启动训练、加载最佳模型、做最终评估并保存评估结果。
- `cfg = build_config_from_args()`：读取命令行参数并生成配置对象 `cfg`，后面的训练、保存和评估都使用这份配置。
- `set_seed(cfg.seed)`：设置随机种子，让数据划分、初始化和训练过程尽量可复现。
- `print("Using config:")`：把当前训练或评估信息输出到终端。
- `for k, v in asdict(cfg).items():`：遍历配置对象中的每个参数，把本次实验配置打印出来。
- `print(f"  {k}: {v}")`：把当前训练或评估信息输出到终端。
- `dataset, state_dim, action_dim, reward_mean, reward_std = load_hdf5_dataset(cfg.data_path)`：加载离线数据集，得到训练/验证 DataLoader、状态维度、动作维度以及原始状态动作数据。
- `batch_size=cfg.batch_size,`：batch_size 的作用：每个 batch 的样本数。
- `obs_mean, obs_std = compute_obs_stats(train_dataset)`：只用训练集计算状态均值和标准差，后面用它们对状态做标准化。
- `obs_mean = obs_mean.to(cfg.device)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `obs_std = obs_std.to(cfg.device)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `reward_mean = reward_mean.to(cfg.device)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `reward_std = reward_std.to(cfg.device)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(`：创建学习率调度器，让学习率随训练轮数变化，后期训练更稳定。
- `critic1_scheduler = optim.lr_scheduler.CosineAnnealingLR(`：创建学习率调度器，让学习率随训练轮数变化，后期训练更稳定。
- `critic2_scheduler = optim.lr_scheduler.CosineAnnealingLR(`：创建学习率调度器，让学习率随训练轮数变化，后期训练更稳定。
- `history = train_sac(agent, train_loader, val_loader, obs_mean, obs_std, reward_mean, reward_std, cfg)`：创建日志容器，用来记录每个 epoch 的训练损失、验证损失和学习率。
- `load_best_actor(cfg)`：训练结束后加载验证集表现最好的模型，而不是简单使用最后一轮模型。
- `print("Training complete, all CSVs and plots saved in", cfg.save_dir)`：把当前训练或评估信息输出到终端。


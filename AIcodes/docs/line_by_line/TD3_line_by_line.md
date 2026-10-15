# TD3.py 逐行代码解释附录

这是 [`TD3.md`](../TD3.md) 的细节附录，保留对应源码每一行非空代码的解释。主文档负责理解流程；这里适合查某一行具体在做什么。

## 逐行代码解释
下面按源码分块展示代码：每个小节先给出干净源码，再用简洁中文解释这段代码的作用、关键变量含义和训练逻辑。

### 模块级代码、导入和入口
**代码：**

```python
import os
import json
import h5py
import random
import argparse
from dataclasses import dataclass, asdict, field
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
import pandas as pd
# ------------------------
# 工具函数
# ------------------------
# ------------------------
# 参数配置
# ------------------------
# ------------------------
# Actor & Critic 网络
# ------------------------
# ------------------------
# Utility: 计算训练集统计量
# ------------------------
# ------------------------
# Utility: soft update
# ------------------------
# ------------------------
# TD3 Trainer
# ------------------------
# ------------------------
# Utility: plot & save CSV
# ------------------------
# ------------------------
# Main function
# ------------------------
if __name__ == "__main__":
    main()
```

**解释：**

- `import os`：导入依赖库，供后续数据读取、模型训练、优化、绘图或参数解析使用。
- `import json`：导入依赖库，供后续数据读取、模型训练、优化、绘图或参数解析使用。
- `import h5py`：导入依赖库，供后续数据读取、模型训练、优化、绘图或参数解析使用。
- `import random`：导入依赖库，供后续数据读取、模型训练、优化、绘图或参数解析使用。
- `import argparse`：导入依赖库，供后续数据读取、模型训练、优化、绘图或参数解析使用。
- `from dataclasses import dataclass, asdict, field`：导入依赖库，供后续数据读取、模型训练、优化、绘图或参数解析使用。
- `import torch`：导入依赖库，供后续数据读取、模型训练、优化、绘图或参数解析使用。
- `import torch.nn as nn`：导入依赖库，供后续数据读取、模型训练、优化、绘图或参数解析使用。
- `import torch.optim as optim`：导入依赖库，供后续数据读取、模型训练、优化、绘图或参数解析使用。
- `import numpy as np`：导入依赖库，供后续数据读取、模型训练、优化、绘图或参数解析使用。
- `from torch.utils.data import TensorDataset, DataLoader, random_split`：导入依赖库，供后续数据读取、模型训练、优化、绘图或参数解析使用。
- `import matplotlib.pyplot as plt`：导入依赖库，供后续数据读取、模型训练、优化、绘图或参数解析使用。
- `import pandas as pd`：导入依赖库，供后续数据读取、模型训练、优化、绘图或参数解析使用。
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

### 函数 `parse_hidden`
**代码：**

```python
def parse_hidden(value):
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

- `def parse_hidden(value):`：函数作用：把命令行里的隐藏层配置转换成整数列表，用来搭建神经网络。
- `return result`：返回训练结果、数据和标准化参数，供主流程继续评估或加载模型。
- `return [256, 256]`：返回训练结果、数据和标准化参数，供主流程继续评估或加载模型。
- `return [int(x.strip()) for x in value.split(",") if x.strip()]`：返回训练结果、数据和标准化参数，供主流程继续评估或加载模型。

### 函数 `build_config_from_args`
**代码：**

```python
def build_config_from_args():
    parser = argparse.ArgumentParser(description="Offline TD3 Training")
    parser.add_argument("--data-path", type=str, default="halfcheetah_medium-v2.hdf5")
    parser.add_argument("--save-dir", type=str, default="checkpoints/TD3")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr-actor", type=float, default=1e-3)
    parser.add_argument("--lr-critic", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--policy-noise", type=float, default=0.2)
    parser.add_argument("--noise-clip", type=float, default=0.5)
    parser.add_argument("--policy-delay", type=int, default=2)
    parser.add_argument("--max-action", type=float, default=1.0)
    parser.add_argument("--num-epochs", type=int, default=150)
    parser.add_argument("--hidden", type=str, default="256,256")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--best-model-name", type=str, default="best_model.pt")
    parser.add_argument("--log-csv-name", type=str, default="training_log.csv")
    parser.add_argument("--critic-curve-name", type=str, default="critic_loss.png")
    parser.add_argument("--actor-curve-name", type=str, default="actor_loss.png")
    parser.add_argument("--q-curve-name", type=str, default="q_value.png")
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
        data_path=args.data_path,
        save_dir=args.save_dir,
        device=device,
        batch_size=args.batch_size,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        gamma=args.gamma,
        tau=args.tau,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        policy_delay=args.policy_delay,
        max_action=args.max_action,
        num_epochs=args.num_epochs,
        hidden=parse_hidden(args.hidden),
        seed=args.seed,
        val_ratio=args.val_ratio,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        num_workers=args.num_workers,
        best_model_name=args.best_model_name,
        log_csv_name=args.log_csv_name,
        critic_curve_name=args.critic_curve_name,
        actor_curve_name=args.actor_curve_name,
        q_curve_name=args.q_curve_name,
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
- `data_path=args.data_path,`：data_path 的作用：数据集路径，指向 HDF5 离线强化学习数据文件。
- `save_dir=args.save_dir,`：save_dir 的作用：模型、日志和图片的输出目录。
- `device=device,`：device 的作用：计算设备，优先使用 GPU，否则使用 CPU。
- `batch_size=args.batch_size,`：batch_size 的作用：每个 batch 的样本数。
- `lr_actor=args.lr_actor,`：创建策略网络。它输入状态，输出动作，是后续训练要优化的核心模型。
- `gamma=args.gamma,`：gamma 的作用：折扣因子，控制未来奖励在 TD 目标中的权重。
- `tau=args.tau,`：tau 的作用：目标网络软更新系数。
- `policy_noise=args.policy_noise,`：policy_noise 的作用：TD3 目标动作平滑噪声强度。
- `noise_clip=args.noise_clip,`：noise_clip 的作用：目标动作噪声裁剪范围。
- `policy_delay=args.policy_delay,`：policy_delay 的作用：TD3 中 actor 延迟更新间隔。
- `num_epochs=args.num_epochs,`：num_epochs 的作用：训练轮数。
- `seed=args.seed,`：seed 的作用：随机种子，用于复现实验。
- `val_ratio=args.val_ratio,`：val_ratio 的作用：验证集比例。
- `grad_clip=args.grad_clip,`：grad_clip 的作用：梯度裁剪阈值，防止梯度爆炸。
- `best_model_name=args.best_model_name,`：best_model_name 的作用：最佳模型 checkpoint 文件名。
- `return cfg`：返回训练结果、数据和标准化参数，供主流程继续评估或加载模型。

### 类 `Config`
**代码：**

```python
@dataclass
class Config:
    data_path: str = "halfcheetah_medium-v2.hdf5"
    save_dir: str = "checkpoints/TD3"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 256
    lr_actor: float = 1e-3
    lr_critic: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    max_action: float = 1.0
    num_epochs: int = 100
    hidden: list = field(default_factory=lambda: [256, 256])
    seed: int = 42
    val_ratio: float = 0.1
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    num_workers: int = 0
    best_model_name: str = "best_model.pt"
    log_csv_name: str = "training_log.csv"
    critic_curve_name: str = "critic_loss.png"
    actor_curve_name: str = "actor_loss.png"
    q_curve_name: str = "q_value.png"
```

**解释：**

- `@dataclass`：让配置类自动生成初始化方法，写超参数更方便。
- `class Config:`：配置类，集中保存数据路径、学习率、batch_size、网络结构、设备和输出目录等超参数。
- `data_path: str = "halfcheetah_medium-v2.hdf5"`：data_path 的作用：数据集路径，指向 HDF5 离线强化学习数据文件。
- `save_dir: str = "checkpoints/TD3"`：save_dir 的作用：模型、日志和图片的输出目录。
- `device: str = "cuda" if torch.cuda.is_available() else "cpu"`：device 的作用：计算设备，优先使用 GPU，否则使用 CPU。
- `batch_size: int = 256`：batch_size 的作用：每个 batch 的样本数。
- `gamma: float = 0.99`：gamma 的作用：折扣因子，控制未来奖励在 TD 目标中的权重。
- `tau: float = 0.005`：tau 的作用：目标网络软更新系数。
- `policy_noise: float = 0.2`：policy_noise 的作用：TD3 目标动作平滑噪声强度。
- `noise_clip: float = 0.5`：noise_clip 的作用：目标动作噪声裁剪范围。
- `policy_delay: int = 2`：policy_delay 的作用：TD3 中 actor 延迟更新间隔。
- `num_epochs: int = 100`：num_epochs 的作用：训练轮数。
- `seed: int = 42`：seed 的作用：随机种子，用于复现实验。
- `val_ratio: float = 0.1`：val_ratio 的作用：验证集比例。
- `weight_decay: float = 1e-4`：weight_decay 的作用：L2 正则化系数，用于减轻过拟合。
- `grad_clip: float = 5.0`：grad_clip 的作用：梯度裁剪阈值，防止梯度爆炸。
- `best_model_name: str = "best_model.pt"`：best_model_name 的作用：最佳模型 checkpoint 文件名。

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
    def __init__(self, state_dim, action_dim, max_action, hidden=(256, 256)):
        super().__init__()
        layers = []
        dims = [state_dim] + list(hidden)
        for i in range(len(hidden)):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden[-1], action_dim))
        layers.append(nn.Tanh())
        self.model = nn.Sequential(*layers)
        self.max_action = max_action
```

**解释：**

- `def __init__(self, state_dim, action_dim, max_action, hidden=(256, 256)):`：函数作用：初始化当前类，创建网络层、优化器或训练需要的成员变量。

### 方法 `Actor.forward`
**代码：**

```python
    def forward(self, state):
        return self.model(state) * self.max_action
```

**解释：**

- `def forward(self, state):`：函数作用：定义网络前向传播，说明输入张量如何变成输出。
- `return self.model(state) * self.max_action`：返回训练结果、数据和标准化参数，供主流程继续评估或加载模型。

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
        layers1 = []
        dims1 = [state_dim + action_dim] + list(hidden)
        for i in range(len(hidden)):
            layers1.append(nn.Linear(dims1[i], dims1[i + 1]))
            layers1.append(nn.ReLU())
        layers1.append(nn.Linear(hidden[-1], 1))
        self.q1 = nn.Sequential(*layers1)
        layers2 = []
        dims2 = [state_dim + action_dim] + list(hidden)
        for i in range(len(hidden)):
            layers2.append(nn.Linear(dims2[i], dims2[i + 1]))
            layers2.append(nn.ReLU())
        layers2.append(nn.Linear(hidden[-1], 1))
        self.q2 = nn.Sequential(*layers2)
```

**解释：**

- `def __init__(self, state_dim, action_dim, hidden=(256, 256)):`：函数作用：初始化当前类，创建网络层、优化器或训练需要的成员变量。

### 方法 `Critic.forward`
**代码：**

```python
    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)
```

**解释：**

- `def forward(self, state, action):`：函数作用：定义网络前向传播，说明输入张量如何变成输出。
- `return self.q1(sa), self.q2(sa)`：返回训练结果、数据和标准化参数，供主流程继续评估或加载模型。

### 方法 `Critic.q1_forward`
**代码：**

```python
    def q1_forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa)
```

**解释：**

- `def q1_forward(self, state, action):`：函数作用：封装一段可复用逻辑，供主训练流程调用。
- `return self.q1(sa)`：返回训练结果、数据和标准化参数，供主流程继续评估或加载模型。

### 函数 `compute_obs_stats`
**代码：**

```python
def compute_obs_stats(train_dataset):
    obs_list = []
    for item in train_dataset:
        obs_list.append(item[0].unsqueeze(0))
    obs_tensor = torch.cat(obs_list, dim=0)
    obs_mean = obs_tensor.mean(dim=0)
    obs_std = obs_tensor.std(dim=0)
    obs_std = torch.clamp(obs_std, min=1e-6)
    return obs_mean, obs_std
```

**解释：**

- `def compute_obs_stats(train_dataset):`：函数作用：统计训练集状态均值和标准差，后续用于状态标准化。
- `obs_mean = obs_tensor.mean(dim=0)`：obs_mean 的作用：训练集 17 个状态维度的均值，用于状态标准化。
- `obs_std = obs_tensor.std(dim=0)`：obs_std 的作用：训练集 17 个状态维度的标准差，用于状态标准化。
- `obs_std = torch.clamp(obs_std, min=1e-6)`：做数值裁剪，防止动作、标准差或权重超出安全范围。
- `return obs_mean, obs_std`：返回训练结果、数据和标准化参数，供主流程继续评估或加载模型。

### 函数 `soft_update`
**代码：**

```python
def soft_update(source_net, target_net, tau):
    for param, target_param in zip(source_net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
```

**解释：**

- `def soft_update(source_net, target_net, tau):`：函数作用：软更新目标网络，让目标网络缓慢跟随当前网络，稳定 TD 目标。

### 类 `TD3Trainer`
**代码：**

```python
class TD3Trainer:
```

**解释：**

- `class TD3Trainer:`：训练器/智能体类，封装 actor、critic、优化器以及 train_step/eval_step 等核心流程。

### 方法 `TD3Trainer.__init__`
**代码：**

```python
    def __init__(self, state_dim, action_dim, max_action, args):
        self.actor = Actor(state_dim, action_dim, max_action, args.hidden).to(args.device)
        self.actor_target = Actor(state_dim, action_dim, max_action, args.hidden).to(args.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic = Critic(state_dim, action_dim, args.hidden).to(args.device)
        self.critic_target = Critic(state_dim, action_dim, args.hidden).to(args.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=args.lr_actor,
            weight_decay=args.weight_decay
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=args.lr_critic,
            weight_decay=args.weight_decay
        )
        self.actor_scheduler = None
        self.critic_scheduler = None
        self.max_action = max_action
        self.gamma = args.gamma
        self.tau = args.tau
        self.policy_noise = args.policy_noise
        self.noise_clip = args.noise_clip
        self.policy_delay = args.policy_delay
        self.total_it = 0
        self.device = args.device
        self.grad_clip = args.grad_clip
```

**解释：**

- `def __init__(self, state_dim, action_dim, max_action, args):`：函数作用：初始化当前类，创建网络层、优化器或训练需要的成员变量。
- `self.actor = Actor(state_dim, action_dim, max_action, args.hidden).to(args.device)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `self.actor_target = Actor(state_dim, action_dim, max_action, args.hidden).to(args.device)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `self.critic = Critic(state_dim, action_dim, args.hidden).to(args.device)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `self.critic_target = Critic(state_dim, action_dim, args.hidden).to(args.device)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `self.actor_optimizer = optim.Adam(`：创建 Adam 优化器，后面根据梯度更新网络参数。
- `self.critic_optimizer = optim.Adam(`：创建 Adam 优化器，后面根据梯度更新网络参数。
- `self.actor_scheduler = None`：创建学习率调度器，让学习率随训练轮数变化，后期训练更稳定。
- `self.critic_scheduler = None`：创建学习率调度器，让学习率随训练轮数变化，后期训练更稳定。

### 方法 `TD3Trainer.set_schedulers`
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

### 方法 `TD3Trainer.train_step`
**代码：**

```python
    def train_step(self, batch, obs_mean, obs_std):
        self.total_it += 1
        state, action, reward, next_state, done = batch
        state = state.to(self.device, non_blocking=True)
        action = action.to(self.device, non_blocking=True)
        reward = reward.to(self.device, non_blocking=True).unsqueeze(1)
        next_state = next_state.to(self.device, non_blocking=True)
        done = done.to(self.device, non_blocking=True).unsqueeze(1)
        state = (state - obs_mean) / obs_std
        next_state = (next_state - obs_mean) / obs_std
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.gamma * target_Q
        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_Q1, target_Q) + nn.MSELoss()(current_Q2, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_optimizer.step()
        actor_loss = None
        actor_updated = 0
        if self.total_it % self.policy_delay == 0:
            actor_loss_tensor = -self.critic.q1_forward(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss_tensor.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
            self.actor_optimizer.step()
            soft_update(self.critic, self.critic_target, self.tau)
            soft_update(self.actor, self.actor_target, self.tau)
            actor_loss = actor_loss_tensor.item()
            actor_updated = 1
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss,
            "actor_updated": actor_updated,
            "target_q": target_Q.mean().item(),
            "current_q1": current_Q1.mean().item(),
            "current_q2": current_Q2.mean().item(),
            "reward": reward.mean().item()
        }
```

**解释：**

- `def train_step(self, batch, obs_mean, obs_std):`：函数作用：执行一次 TD3 更新：构造带平滑噪声的 TD 目标，训练双 critic，并按延迟策略更新 actor。
- `state, action, reward, next_state, done = batch`：把一个 batch 拆成状态、动作、奖励、下一状态和终止标志。
- `state = state.to(self.device, non_blocking=True)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `action = action.to(self.device, non_blocking=True)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `reward = reward.to(self.device, non_blocking=True).unsqueeze(1)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `next_state = next_state.to(self.device, non_blocking=True)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `done = done.to(self.device, non_blocking=True).unsqueeze(1)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `state = (state - obs_mean) / obs_std`：对当前状态做标准化：减去训练集均值，再除以训练集标准差。
- `next_state = (next_state - obs_mean) / obs_std`：对下一状态做同样标准化，保证 TD 目标计算使用相同输入尺度。
- `with torch.no_grad():`：进入不记录梯度的区域，通常用于计算 TD 目标或验证指标。
- `noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)`：生成并裁剪目标动作噪声，这是 TD3 的 target policy smoothing。
- `next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)`：用目标 actor 计算下一状态动作，并加入噪声后裁剪到动作范围内。
- `target_Q1, target_Q2 = self.critic_target(next_state, next_action)`：用目标 critic 估计下一状态动作的 Q 值。
- `target_Q = torch.min(target_Q1, target_Q2)`：双 Q 取较小值，降低 Q 值过估计风险。
- `target_Q = reward + (1 - done) * self.gamma * target_Q`：构造 TD 目标：即时奖励加上折扣后的下一状态价值；done 为 1 时不再加未来价值。
- `current_Q1, current_Q2 = self.critic(state, action)`：用当前 critic 估计数据集中真实状态动作的 Q 值。
- `critic_loss = nn.MSELoss()(current_Q1, target_Q) + nn.MSELoss()(current_Q2, target_Q)`：计算 critic 的 TD 回归损失，让当前 Q 值逼近 TD 目标。
- `self.critic_optimizer.zero_grad()`：清空上一轮梯度，避免梯度累积影响当前 batch。
- `critic_loss.backward()`：对当前损失反向传播，计算网络参数梯度。
- `nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)`：对梯度做裁剪，防止梯度过大导致训练不稳定。
- `self.critic_optimizer.step()`：优化器根据梯度更新参数。
- `actor_loss = None`：合成 actor 损失，通常由 Q 项、熵项或 BC 正则项组成。
- `self.actor_optimizer.zero_grad()`：清空上一轮梯度，避免梯度累积影响当前 batch。
- `actor_loss_tensor.backward()`：对当前损失反向传播，计算网络参数梯度。
- `nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)`：对梯度做裁剪，防止梯度过大导致训练不稳定。
- `self.actor_optimizer.step()`：优化器根据梯度更新参数。
- `soft_update(self.critic, self.critic_target, self.tau)`：软更新目标 critic，让目标 Q 网络缓慢跟随当前 critic。
- `soft_update(self.actor, self.actor_target, self.tau)`：软更新目标 actor，让目标策略网络缓慢跟随当前 actor。
- `actor_loss = actor_loss_tensor.item()`：合成 actor 损失，通常由 Q 项、熵项或 BC 正则项组成。
- `return {`：返回本次更新或评估的指标字典，供训练日志记录。

### 方法 `TD3Trainer.eval_step`
**代码：**

```python
    def eval_step(self, batch, obs_mean, obs_std):
        state, action, reward, next_state, done = batch
        state = state.to(self.device, non_blocking=True)
        action = action.to(self.device, non_blocking=True)
        reward = reward.to(self.device, non_blocking=True).unsqueeze(1)
        next_state = next_state.to(self.device, non_blocking=True)
        done = done.to(self.device, non_blocking=True).unsqueeze(1)
        state = (state - obs_mean) / obs_std
        next_state = (next_state - obs_mean) / obs_std
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.gamma * target_Q
            current_Q1, current_Q2 = self.critic(state, action)
            critic_loss = nn.MSELoss()(current_Q1, target_Q) + nn.MSELoss()(current_Q2, target_Q)
            actor_proxy = -self.critic.q1_forward(state, self.actor(state)).mean()
        return {
            "critic_loss": critic_loss.item(),
            "actor_proxy": actor_proxy.item(),
            "target_q": target_Q.mean().item(),
            "current_q1": current_Q1.mean().item(),
            "current_q2": current_Q2.mean().item(),
            "reward": reward.mean().item()
        }
```

**解释：**

- `def eval_step(self, batch, obs_mean, obs_std):`：函数作用：在指定数据集或 batch 上评估模型，不更新参数。
- `state, action, reward, next_state, done = batch`：把一个 batch 拆成状态、动作、奖励、下一状态和终止标志。
- `state = state.to(self.device, non_blocking=True)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `action = action.to(self.device, non_blocking=True)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `reward = reward.to(self.device, non_blocking=True).unsqueeze(1)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `next_state = next_state.to(self.device, non_blocking=True)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `done = done.to(self.device, non_blocking=True).unsqueeze(1)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `state = (state - obs_mean) / obs_std`：对当前状态做标准化：减去训练集均值，再除以训练集标准差。
- `next_state = (next_state - obs_mean) / obs_std`：对下一状态做同样标准化，保证 TD 目标计算使用相同输入尺度。
- `with torch.no_grad():`：进入不记录梯度的区域，通常用于计算 TD 目标或验证指标。
- `noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)`：生成并裁剪目标动作噪声，这是 TD3 的 target policy smoothing。
- `next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)`：用目标 actor 计算下一状态动作，并加入噪声后裁剪到动作范围内。
- `target_Q1, target_Q2 = self.critic_target(next_state, next_action)`：用目标 critic 估计下一状态动作的 Q 值。
- `target_Q = torch.min(target_Q1, target_Q2)`：双 Q 取较小值，降低 Q 值过估计风险。
- `target_Q = reward + (1 - done) * self.gamma * target_Q`：构造 TD 目标：即时奖励加上折扣后的下一状态价值；done 为 1 时不再加未来价值。
- `current_Q1, current_Q2 = self.critic(state, action)`：用当前 critic 估计数据集中真实状态动作的 Q 值。
- `critic_loss = nn.MSELoss()(current_Q1, target_Q) + nn.MSELoss()(current_Q2, target_Q)`：计算 critic 的 TD 回归损失，让当前 Q 值逼近 TD 目标。
- `return {`：返回本次更新或评估的指标字典，供训练日志记录。

### 方法 `TD3Trainer.save_best`
**代码：**

```python
    def save_best(self, save_path, obs_mean, obs_std, epoch, best_val_loss, args):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "obs_mean": obs_mean.detach().cpu(),
            "obs_std": obs_std.detach().cpu(),
            "state_dim": args.state_dim,
            "action_dim": args.action_dim,
            "max_action": args.max_action,
            "hidden": list(args.hidden),
            "epoch": epoch,
            "best_val_loss": best_val_loss
        }, save_path)
```

**解释：**

- `def save_best(self, save_path, obs_mean, obs_std, epoch, best_val_loss, args):`：函数作用：保存当前最佳模型，同时保存标准化参数和网络结构信息，便于之后恢复。

### 函数 `save_plot_csv`
**代码：**

```python
def save_plot_csv(df, cfg):
    os.makedirs(cfg.save_dir, exist_ok=True)
    df.to_csv(os.path.join(cfg.save_dir, cfg.log_csv_name), index=False)
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_critic_loss"], label="train_critic_loss")
    plt.plot(df["epoch"], df["val_critic_loss"], label="val_critic_loss")
    plt.title("critic_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.save_dir, cfg.critic_curve_name))
    plt.close()
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_actor_loss"], label="train_actor_loss")
    plt.plot(df["epoch"], df["val_actor_proxy"], label="val_actor_proxy")
    plt.title("actor_loss")
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.save_dir, cfg.actor_curve_name))
    plt.close()
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_target_q"], label="train_target_q")
    plt.plot(df["epoch"], df["val_target_q"], label="val_target_q")
    plt.plot(df["epoch"], df["train_current_q1"], label="train_current_q1")
    plt.plot(df["epoch"], df["val_current_q1"], label="val_current_q1")
    plt.title("q_value")
    plt.xlabel("epoch")
    plt.ylabel("Q")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.save_dir, cfg.q_curve_name))
    plt.close()
```

**解释：**

- `def save_plot_csv(df, cfg):`：函数作用：保存训练日志和曲线，方便观察训练过程和写论文结果。
- `df.to_csv(os.path.join(cfg.save_dir, cfg.log_csv_name), index=False)`：把表格结果保存成 CSV 文件，方便论文记录、画图或复查。

### 函数 `main`
**代码：**

```python
def main():
    cfg = build_config_from_args()
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)
    print("Using config:")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")
    with h5py.File(cfg.data_path, "r") as f:
        obs = torch.tensor(f["observations"][:], dtype=torch.float32)
        actions = torch.tensor(f["actions"][:], dtype=torch.float32)
        rewards = torch.tensor(f["rewards"][:], dtype=torch.float32)
        next_obs = torch.tensor(f["next_observations"][:], dtype=torch.float32)
        terminals = torch.tensor(f["terminals"][:], dtype=torch.float32)
    dataset = TensorDataset(obs, actions, rewards, next_obs, terminals)
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
    obs_mean, obs_std = compute_obs_stats(train_dataset)
    obs_mean = obs_mean.to(cfg.device)
    obs_std = obs_std.to(cfg.device)
    state_dim = obs.shape[1]
    action_dim = actions.shape[1]
    cfg.state_dim = state_dim
    cfg.action_dim = action_dim
    trainer = TD3Trainer(state_dim, action_dim, cfg.max_action, cfg)
    actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(trainer.actor_optimizer, T_max=cfg.num_epochs)
    critic_scheduler = optim.lr_scheduler.CosineAnnealingLR(trainer.critic_optimizer, T_max=cfg.num_epochs)
    trainer.set_schedulers(actor_scheduler, critic_scheduler)
    history = {
        "epoch": [],
        "train_critic_loss": [],
        "train_actor_loss": [],
        "train_target_q": [],
        "train_current_q1": [],
        "train_current_q2": [],
        "val_critic_loss": [],
        "val_actor_proxy": [],
        "val_target_q": [],
        "val_current_q1": [],
        "val_current_q2": [],
        "actor_lr": [],
        "critic_lr": []
    }
    best_val_loss = float("inf")
    best_model_path = os.path.join(cfg.save_dir, cfg.best_model_name)
    for epoch in range(1, cfg.num_epochs + 1):
        train_sum = {
            "critic_loss": 0.0,
            "target_q": 0.0,
            "current_q1": 0.0,
            "current_q2": 0.0,
            "reward": 0.0
        }
        actor_loss_sum = 0.0
        actor_loss_count = 0
        for batch in train_loader:
            out = trainer.train_step(batch, obs_mean, obs_std)
            train_sum["critic_loss"] += out["critic_loss"]
            train_sum["target_q"] += out["target_q"]
            train_sum["current_q1"] += out["current_q1"]
            train_sum["current_q2"] += out["current_q2"]
            train_sum["reward"] += out["reward"]
            if out["actor_loss"] is not None:
                actor_loss_sum += out["actor_loss"]
                actor_loss_count += 1
        n_train = len(train_loader)
        train_metrics = {
            "critic_loss": train_sum["critic_loss"] / n_train,
            "actor_loss": actor_loss_sum / max(actor_loss_count, 1),
            "target_q": train_sum["target_q"] / n_train,
            "current_q1": train_sum["current_q1"] / n_train,
            "current_q2": train_sum["current_q2"] / n_train,
            "reward": train_sum["reward"] / n_train
        }
        val_sum = {
            "critic_loss": 0.0,
            "actor_proxy": 0.0,
            "target_q": 0.0,
            "current_q1": 0.0,
            "current_q2": 0.0,
            "reward": 0.0
        }
        for batch in val_loader:
            out = trainer.eval_step(batch, obs_mean, obs_std)
            for k in val_sum:
                val_sum[k] += out[k]
        n_val = len(val_loader)
        val_metrics = {k: v / n_val for k, v in val_sum.items()}
        actor_lr = trainer.actor_optimizer.param_groups[0]["lr"]
        critic_lr = trainer.critic_optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch}/{cfg.num_epochs} | "
            f"Train Critic {train_metrics['critic_loss']:.6f} | "
            f"Train Actor {train_metrics['actor_loss']:.6f} | "
            f"Val Critic {val_metrics['critic_loss']:.6f} | "
            f"Val ActorProxy {val_metrics['actor_proxy']:.6f}"
        )
        history["epoch"].append(epoch)
        history["train_critic_loss"].append(train_metrics["critic_loss"])
        history["train_actor_loss"].append(train_metrics["actor_loss"])
        history["train_target_q"].append(train_metrics["target_q"])
        history["train_current_q1"].append(train_metrics["current_q1"])
        history["train_current_q2"].append(train_metrics["current_q2"])
        history["val_critic_loss"].append(val_metrics["critic_loss"])
        history["val_actor_proxy"].append(val_metrics["actor_proxy"])
        history["val_target_q"].append(val_metrics["target_q"])
        history["val_current_q1"].append(val_metrics["current_q1"])
        history["val_current_q2"].append(val_metrics["current_q2"])
        history["actor_lr"].append(actor_lr)
        history["critic_lr"].append(critic_lr)
        if val_metrics["critic_loss"] < best_val_loss:
            best_val_loss = val_metrics["critic_loss"]
            trainer.save_best(best_model_path, obs_mean, obs_std, epoch, best_val_loss, cfg)
            print(f"Best model saved at epoch {epoch}, val_critic_loss={best_val_loss:.6f}")
        if trainer.actor_scheduler is not None:
            trainer.actor_scheduler.step()
        if trainer.critic_scheduler is not None:
            trainer.critic_scheduler.step()
    df = pd.DataFrame(history)
    save_plot_csv(df, cfg)
    print(f"Training finished. Best model, logs and plots saved to {cfg.save_dir}")
```

**解释：**

- `def main():`：函数作用：脚本主入口，负责读取配置、启动训练、加载最佳模型、做最终评估并保存评估结果。
- `cfg = build_config_from_args()`：读取命令行参数并生成配置对象 `cfg`，后面的训练、保存和评估都使用这份配置。
- `set_seed(cfg.seed)`：设置随机种子，让数据划分、初始化和训练过程尽量可复现。
- `print("Using config:")`：把当前训练或评估信息输出到终端。
- `for k, v in asdict(cfg).items():`：遍历配置对象中的每个参数，把本次实验配置打印出来。
- `print(f"  {k}: {v}")`：把当前训练或评估信息输出到终端。
- `batch_size=cfg.batch_size,`：batch_size 的作用：每个 batch 的样本数。
- `obs_mean, obs_std = compute_obs_stats(train_dataset)`：只用训练集计算状态均值和标准差，后面用它们对状态做标准化。
- `obs_mean = obs_mean.to(cfg.device)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `obs_std = obs_std.to(cfg.device)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `state_dim = obs.shape[1]`：state_dim 的作用：状态维度，本项目通常是 17。
- `action_dim = actions.shape[1]`：action_dim 的作用：动作维度，本项目通常是 6。
- `actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(trainer.actor_optimizer, T_max=cfg.num_epochs)`：创建学习率调度器，让学习率随训练轮数变化，后期训练更稳定。
- `critic_scheduler = optim.lr_scheduler.CosineAnnealingLR(trainer.critic_optimizer, T_max=cfg.num_epochs)`：创建学习率调度器，让学习率随训练轮数变化，后期训练更稳定。
- `history = {`：创建日志容器，用来记录每个 epoch 的训练损失、验证损失和学习率。
- `best_val_loss = float("inf")`：初始化最佳验证指标为无穷大，后面只要验证损失更小就保存模型。
- `best_model_path = os.path.join(cfg.save_dir, cfg.best_model_name)`：初始化最佳模型保存路径，真正保存模型后会更新它。
- `for epoch in range(1, cfg.num_epochs + 1):`：开始按 epoch 训练模型，每轮都会训练一次、验证一次，并记录指标。
- `print(`：把当前训练或评估信息输出到终端。
- `history["epoch"].append(epoch)`：记录当前 epoch 的训练指标，后面保存成表格和曲线。
- `history["train_critic_loss"].append(train_metrics["critic_loss"])`：记录当前 epoch 的训练指标，后面保存成表格和曲线。
- `history["train_actor_loss"].append(train_metrics["actor_loss"])`：记录当前 epoch 的训练指标，后面保存成表格和曲线。
- `history["train_target_q"].append(train_metrics["target_q"])`：记录当前 epoch 的训练指标，后面保存成表格和曲线。
- `history["train_current_q1"].append(train_metrics["current_q1"])`：记录当前 epoch 的训练指标，后面保存成表格和曲线。
- `history["train_current_q2"].append(train_metrics["current_q2"])`：记录当前 epoch 的训练指标，后面保存成表格和曲线。
- `history["val_critic_loss"].append(val_metrics["critic_loss"])`：记录当前 epoch 的训练指标，后面保存成表格和曲线。
- `history["val_actor_proxy"].append(val_metrics["actor_proxy"])`：记录当前 epoch 的训练指标，后面保存成表格和曲线。
- `history["val_target_q"].append(val_metrics["target_q"])`：记录当前 epoch 的训练指标，后面保存成表格和曲线。
- `history["val_current_q1"].append(val_metrics["current_q1"])`：记录当前 epoch 的训练指标，后面保存成表格和曲线。
- `history["val_current_q2"].append(val_metrics["current_q2"])`：记录当前 epoch 的训练指标，后面保存成表格和曲线。
- `history["actor_lr"].append(actor_lr)`：记录当前 epoch 的训练指标，后面保存成表格和曲线。
- `history["critic_lr"].append(critic_lr)`：记录当前 epoch 的训练指标，后面保存成表格和曲线。
- `if val_metrics["critic_loss"] < best_val_loss:`：如果当前验证结果优于历史最好结果，就保存当前模型。
- `best_val_loss = val_metrics["critic_loss"]`：初始化最佳验证指标为无穷大，后面只要验证损失更小就保存模型。
- `trainer.save_best(best_model_path, obs_mean, obs_std, epoch, best_val_loss, cfg)`：保存当前最佳模型，同时保存标准化参数和网络结构。
- `print(f"Best model saved at epoch {epoch}, val_critic_loss={best_val_loss:.6f}")`：计算 critic 的 TD 回归损失，让当前 Q 值逼近 TD 目标。
- `trainer.actor_scheduler.step()`：优化器根据梯度更新参数。
- `trainer.critic_scheduler.step()`：优化器根据梯度更新参数。
- `df = pd.DataFrame(history)`：把训练记录转换成 pandas 表格，便于保存 CSV 和画曲线。
- `save_plot_csv(df, cfg)`：根据训练日志保存损失曲线，方便观察训练是否收敛。
- `print(f"Training finished. Best model, logs and plots saved to {cfg.save_dir}")`：把当前训练或评估信息输出到终端。


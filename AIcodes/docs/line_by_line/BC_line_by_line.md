# BC.py 逐行代码解释附录

这是 [`BC.md`](../BC.md) 的细节附录，保留对应源码每一行非空代码的解释。主文档负责理解流程；这里适合查某一行具体在做什么。

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
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# ====================
# 1. 参数配置
# ====================
#集中管理训练行为克隆（BC）模型时所需的所有 超参数 和 配置项 。
#将用户从命令行输入的 字符串 （或列表）解析为神经网络可以理解的 整数列表 。
#解析命令行参数 、 决定计算设备 、 初始化配置对象 并 记录训练元数据 。
# ====================
# 2. 随机种子
# ====================
# ====================
# 3. 数据加载与划分
# ====================
#将磁盘上的原始数据文件转换为深度学习模型可以直接读取的格式 ，并完成训练集与验证集的划分。
# ====================
# 4. 计算标准化参数
# ====================
# ====================
# 5. BC Policy
# ====================
#用于预测动作的 神经网络模型 。
# ====================
# 6. 单轮训练
# ====================
#整个训练过程的 核心执行单元 。它的任务是遍历一遍训练数据集，计算损失并更新神经网络的权重。
# ====================
# 7. 验证
# ====================
#在验证集（或测试集）上评估模型的性能
# ====================
# 8. 保存最佳模型
# ====================
#将模型及其“配套信息”完整地保存到磁盘文件中 。
# ====================
# 9. 加载最佳模型
# ====================
#从磁盘读取保存的模型文件，并完美还原出一个可以立即使用的模型对象 。
# ====================
# 10. 子集评估
# ====================
#在给定的子集（如验证集或测试集）上评估模型的性能
# ====================
# 11. 保存训练图
# ====================
# ====================
# 12. 主训练函数
# ====================
#算法的 核心工作流 。它负责把数据、模型、优化器和训练循环有机地结合在一起。
# ====================
# 13. 主流程
# ====================
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
- `from torch.utils.data import TensorDataset, DataLoader, random_split`：导入依赖库，供后续数据读取、模型训练、优化、绘图或参数解析使用。
- `import matplotlib.pyplot as plt`：导入依赖库，供后续数据读取、模型训练、优化、绘图或参数解析使用。
- `import pandas as pd`：导入依赖库，供后续数据读取、模型训练、优化、绘图或参数解析使用。
- `import numpy as np`：导入依赖库，供后续数据读取、模型训练、优化、绘图或参数解析使用。
- `if __name__ == "__main__":`：判断当前文件是否被直接运行；只有直接运行时才启动 main。
- `main()`：调用 main 函数，正式启动整个脚本。

### 类 `Config`
**代码：**

```python
@dataclass
class Config:
    data_path: str = "halfcheetah_medium-v2.hdf5"
    batch_size: int = 256
    lr: float = 3e-4
    num_epochs: int = 150
    hidden_layers: list = field(default_factory=lambda: [256, 256])
    output_dir: str = "checkpoints/BC"
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed: int = 42
    val_ratio: float = 0.1
    weight_decay: float = 1e-4#L2 正则化
    grad_clip: float = 5.0#梯度裁剪
    best_model_name: str = "best_model.pt"

    use_smooth_l1: bool = True
    num_workers: int = 0
```

**解释：**

- `@dataclass`：让配置类自动生成初始化方法，写超参数更方便。
- `class Config:`：配置类，集中保存数据路径、学习率、batch_size、网络结构、设备和输出目录等超参数。
- `data_path: str = "halfcheetah_medium-v2.hdf5"`：data_path 的作用：数据集路径，指向 HDF5 离线强化学习数据文件。
- `batch_size: int = 256`：batch_size 的作用：每个 batch 的样本数。
- `lr: float = 3e-4`：lr 的作用：学习率，控制参数更新步长。
- `num_epochs: int = 150`：num_epochs 的作用：训练轮数。
- `hidden_layers: list = field(default_factory=lambda: [256, 256])`：hidden_layers 的作用：隐藏层宽度列表，用来决定 MLP 结构。
- `output_dir: str = "checkpoints/BC"`：output_dir 的作用：模型、日志和图片的输出目录。
- `device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`：device 的作用：计算设备，优先使用 GPU，否则使用 CPU。
- `seed: int = 42`：seed 的作用：随机种子，用于复现实验。
- `val_ratio: float = 0.1`：val_ratio 的作用：验证集比例。
- `weight_decay: float = 1e-4#L2 正则化`：weight_decay 的作用：L2 正则化系数，用于减轻过拟合。
- `grad_clip: float = 5.0#梯度裁剪`：grad_clip 的作用：梯度裁剪阈值，防止梯度爆炸。
- `best_model_name: str = "best_model.pt"`：best_model_name 的作用：最佳模型 checkpoint 文件名。

### 函数 `parse_hidden_layers`
**代码：**

```python
def parse_hidden_layers(value):
    """
    支持：
    --hidden-layers 256,256,128
    --hidden-layers 256 256 128   （argparse会传list时可自行扩展，这里先按字符串处理）
    """
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

### 函数 `build_config_from_args`
**代码：**

```python
def build_config_from_args():
    parser = argparse.ArgumentParser(description="Behavior Cloning Training")

    parser.add_argument("--data-path", type=str, default="halfcheetah_medium-v2.hdf5")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-epochs", type=int, default=150)
    parser.add_argument("--hidden-layers", type=str, default="256,256")
    parser.add_argument("--output-dir", type=str, default="checkpoints/BC")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--best-model-name", type=str, default="best_model.pt")
    parser.add_argument("--action-clip-eps", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--use-layernorm", action="store_true", help="启用 LayerNorm")
    parser.add_argument("--use-smooth-l1", action="store_true", help="启用 SmoothL1Loss")
    parser.add_argument("--use-mse", action="store_true", help="启用 MSELoss（优先级高于 use_smooth_l1）")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--gpu", action="store_true", help="强制使用 GPU")
    group.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    args = parser.parse_args()

    if args.cpu:
        device = torch.device("cpu")
    elif args.gpu:
        if not torch.cuda.is_available():
            raise RuntimeError("你指定了 --gpu，但当前环境没有可用的 CUDA/GPU。")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    use_smooth_l1 = True
    if args.use_mse:
        use_smooth_l1 = False
    elif args.use_smooth_l1:
        use_smooth_l1 = True
    cfg = Config(
        data_path=args.data_path,
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.num_epochs,
        hidden_layers=parse_hidden_layers(args.hidden_layers),
        output_dir=args.output_dir,
        device=device,
        seed=args.seed,
        val_ratio=args.val_ratio,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        best_model_name=args.best_model_name,
        use_layernorm=args.use_layernorm,
        use_smooth_l1=use_smooth_l1,
        action_clip_eps=args.action_clip_eps,
        num_workers=args.num_workers,
    )
    os.makedirs(cfg.output_dir, exist_ok=True)
  
    save_cfg = asdict(cfg)
    save_cfg["device"] = str(cfg.device)
    with open(os.path.join(cfg.output_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(save_cfg, f, ensure_ascii=False, indent=2)
    return cfg
```

**解释：**

- `def build_config_from_args():`：函数作用：解析命令行参数，生成统一配置对象。
- `device = torch.device("cpu")`：device 的作用：计算设备，优先使用 GPU，否则使用 CPU。
- `device = torch.device("cuda")`：device 的作用：计算设备，优先使用 GPU，否则使用 CPU。
- `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`：device 的作用：计算设备，优先使用 GPU，否则使用 CPU。
- `data_path=args.data_path,`：data_path 的作用：数据集路径，指向 HDF5 离线强化学习数据文件。
- `batch_size=args.batch_size,`：batch_size 的作用：每个 batch 的样本数。
- `num_epochs=args.num_epochs,`：num_epochs 的作用：训练轮数。
- `hidden_layers=parse_hidden_layers(args.hidden_layers),`：hidden_layers 的作用：隐藏层宽度列表，用来决定 MLP 结构。
- `output_dir=args.output_dir,`：output_dir 的作用：模型、日志和图片的输出目录。
- `device=device,`：device 的作用：计算设备，优先使用 GPU，否则使用 CPU。
- `seed=args.seed,`：seed 的作用：随机种子，用于复现实验。
- `val_ratio=args.val_ratio,`：val_ratio 的作用：验证集比例。
- `grad_clip=args.grad_clip,`：grad_clip 的作用：梯度裁剪阈值，防止梯度爆炸。
- `best_model_name=args.best_model_name,`：best_model_name 的作用：最佳模型 checkpoint 文件名。
- `return cfg`：返回训练结果、数据和标准化参数，供主流程继续评估或加载模型。

### 函数 `set_seed`
**代码：**

```python
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

**解释：**

- `def set_seed(seed):`：函数作用：设置随机种子，让实验尽量可复现。

### 函数 `load_dataset`
**代码：**

```python
def load_dataset(config):
    with h5py.File(config.data_path, "r") as f:
        obs = torch.tensor(f["observations"][:], dtype=torch.float32)
        actions = torch.tensor(f["actions"][:], dtype=torch.float32)
    state_dim = obs.shape[1]
    action_dim = actions.shape[1]
    dataset = TensorDataset(obs, actions)
    total_size = len(dataset)
    val_size = int(total_size * config.val_ratio)
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=config.num_workers,
        pin_memory=(config.device.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.num_workers,
        pin_memory=(config.device.type == "cuda")
    )
    return train_loader, val_loader, state_dim, action_dim, obs, actions, train_dataset, val_dataset
```

**解释：**

- `def load_dataset(config):`：函数作用：读取离线 HDF5 数据集，并构造训练集、验证集或 DataLoader。
- `state_dim = obs.shape[1]`：state_dim 的作用：状态维度，本项目通常是 17。
- `action_dim = actions.shape[1]`：action_dim 的作用：动作维度，本项目通常是 6。
- `batch_size=config.batch_size,`：batch_size 的作用：每个 batch 的样本数。
- `return train_loader, val_loader, state_dim, action_dim, obs, actions, train_dataset, val_dataset`：返回训练结果、数据和标准化参数，供主流程继续评估或加载模型。

### 函数 `compute_obs_stats`
**代码：**

```python
def compute_obs_stats(train_dataset):
    obs_list = []
    for obs_item, _ in train_dataset:
        obs_list.append(obs_item.unsqueeze(0))
    obs_tensor = torch.cat(obs_list, dim=0)
    obs_mean = obs_tensor.mean(dim=0)
    obs_std = torch.clamp(obs_tensor.std(dim=0), min=1e-6)
    return obs_mean, obs_std
```

**解释：**

- `def compute_obs_stats(train_dataset):`：函数作用：统计训练集状态均值和标准差，后续用于状态标准化。
- `obs_mean = obs_tensor.mean(dim=0)`：obs_mean 的作用：训练集 17 个状态维度的均值，用于状态标准化。
- `obs_std = torch.clamp(obs_tensor.std(dim=0), min=1e-6)`：做数值裁剪，防止动作、标准差或权重超出安全范围。
- `return obs_mean, obs_std`：返回训练结果、数据和标准化参数，供主流程继续评估或加载模型。

### 类 `BCPolicy`
**代码：**

```python
class BCPolicy(nn.Module):
```

**解释：**

- `class BCPolicy(nn.Module):`：BC 策略网络，输入状态，输出动作。

### 方法 `BCPolicy.__init__`
**代码：**

```python
    def __init__(self, input_dim, output_dim, hidden_layers, use_layernorm=False):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            if use_layernorm:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            in_dim = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, output_dim)
```

**解释：**

- `def __init__(self, input_dim, output_dim, hidden_layers, use_layernorm=False):`：函数作用：初始化当前类，创建网络层、优化器或训练需要的成员变量。

### 方法 `BCPolicy.forward`
**代码：**

```python
    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        x = torch.tanh(x)
        return x
```

**解释：**

- `def forward(self, x):`：函数作用：定义网络前向传播，说明输入张量如何变成输出。
- `return x`：返回训练结果、数据和标准化参数，供主流程继续评估或加载模型。

### 函数 `train_one_epoch`
**代码：**

```python
def train_one_epoch(model, loader, optimizer, loss_fn, obs_mean, obs_std, config):
    model.train()
    total_loss = 0.0
    total_count = 0
    for batch_obs, batch_act in loader:
        batch_obs = batch_obs.to(config.device, non_blocking=True)
        batch_act = batch_act.to(config.device, non_blocking=True)
        batch_obs = (batch_obs - obs_mean) / obs_std
        batch_act = torch.clamp(batch_act, -1.0 + config.action_clip_eps, 1.0 - config.action_clip_eps)
        pred_act = model(batch_obs)
        loss = loss_fn(pred_act, batch_act)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        bs = batch_obs.size(0)
        total_loss += loss.item() * bs
        total_count += bs
    return total_loss / total_count
```

**解释：**

- `def train_one_epoch(model, loader, optimizer, loss_fn, obs_mean, obs_std, config):`：函数作用：训练一个 epoch，完成前向预测、计算损失、反向传播和参数更新。
- `batch_obs = batch_obs.to(config.device, non_blocking=True)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `batch_act = batch_act.to(config.device, non_blocking=True)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `batch_act = torch.clamp(batch_act, -1.0 + config.action_clip_eps, 1.0 - config.action_clip_eps)`：做数值裁剪，防止动作、标准差或权重超出安全范围。
- `optimizer.zero_grad()`：清空上一轮梯度，避免梯度累积影响当前 batch。
- `loss.backward()`：对当前损失反向传播，计算网络参数梯度。
- `nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)`：对梯度做裁剪，防止梯度过大导致训练不稳定。
- `optimizer.step()`：优化器根据梯度更新参数。
- `return total_loss / total_count`：返回训练结果、数据和标准化参数，供主流程继续评估或加载模型。

### 函数 `evaluate_loss`
**代码：**

```python
def evaluate_loss(model, loader, loss_fn, obs_mean, obs_std, config):
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch_obs, batch_act in loader:
            batch_obs = batch_obs.to(config.device, non_blocking=True)
            batch_act = batch_act.to(config.device, non_blocking=True)
            batch_obs = (batch_obs - obs_mean) / obs_std
            batch_act = torch.clamp(batch_act, -1.0 + config.action_clip_eps, 1.0 - config.action_clip_eps)
            pred_act = model(batch_obs)
            loss = loss_fn(pred_act, batch_act)
            bs = batch_obs.size(0)
            total_loss += loss.item() * bs
            total_count += bs
    return total_loss / total_count
```

**解释：**

- `def evaluate_loss(model, loader, loss_fn, obs_mean, obs_std, config):`：函数作用：在验证集上计算平均损失；只前向评估，不反向传播。
- `with torch.no_grad():`：进入不记录梯度的区域，通常用于计算 TD 目标或验证指标。
- `batch_obs = batch_obs.to(config.device, non_blocking=True)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `batch_act = batch_act.to(config.device, non_blocking=True)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `batch_act = torch.clamp(batch_act, -1.0 + config.action_clip_eps, 1.0 - config.action_clip_eps)`：做数值裁剪，防止动作、标准差或权重超出安全范围。
- `return total_loss / total_count`：返回训练结果、数据和标准化参数，供主流程继续评估或加载模型。

### 函数 `save_best_model`
**代码：**

```python
def save_best_model(model, obs_mean, obs_std, state_dim, action_dim, config, epoch, best_val_loss):
    save_path = os.path.join(config.output_dir, config.best_model_name)
    torch.save({
        "model_state_dict": model.state_dict(),
        "obs_mean": obs_mean.detach().cpu(),
        "obs_std": obs_std.detach().cpu(),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hidden_layers": list(config.hidden_layers),
        "use_layernorm": config.use_layernorm,
        "best_val_loss": best_val_loss,
        "epoch": epoch
    }, save_path)
    return save_path
```

**解释：**

- `def save_best_model(model, obs_mean, obs_std, state_dim, action_dim, config, epoch, best_val_loss):`：函数作用：保存当前最佳模型，同时保存标准化参数和网络结构信息，便于之后恢复。
- `return save_path`：返回训练结果、数据和标准化参数，供主流程继续评估或加载模型。

### 函数 `load_best_model`
**代码：**

```python
def load_best_model(config):
    checkpoint_path = os.path.join(config.output_dir, config.best_model_name)
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    model = BCPolicy(
        checkpoint["state_dim"],
        checkpoint["action_dim"],
        checkpoint["hidden_layers"],
        use_layernorm=checkpoint.get("use_layernorm", False)
    ).to(config.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    obs_mean = checkpoint["obs_mean"].to(config.device)
    obs_std = checkpoint["obs_std"].to(config.device)
    print(
        f"Loaded best model from epoch {checkpoint['epoch']}, "
        f"best_val_loss={checkpoint['best_val_loss']:.6f}"
    )
    return model, obs_mean, obs_std, checkpoint
```

**解释：**

- `def load_best_model(config):`：函数作用：加载训练过程中保存的最佳策略模型，用于评估或推理。
- `model = BCPolicy(`：创建策略网络。它输入状态，输出动作，是后续训练要优化的核心模型。
- `).to(config.device)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `obs_mean = checkpoint["obs_mean"].to(config.device)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `obs_std = checkpoint["obs_std"].to(config.device)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `print(`：把当前训练或评估信息输出到终端。
- `return model, obs_mean, obs_std, checkpoint`：返回训练结果、数据和标准化参数，供主流程继续评估或加载模型。

### 函数 `evaluate_subset`
**代码：**

```python
def evaluate_subset(model, subset, obs_mean, obs_std, config):
    loader = DataLoader(
        subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(config.device.type == "cuda")
    )
    model.eval()
    pred_list = []
    true_list = []
    with torch.no_grad():
        for batch_obs, batch_act in loader:
            batch_obs = batch_obs.to(config.device, non_blocking=True)
            batch_act = batch_act.to(config.device, non_blocking=True)
            batch_obs = (batch_obs - obs_mean) / obs_std
            pred_act = model(batch_obs)
            pred_list.append(pred_act.cpu())
            true_list.append(batch_act.cpu())
    pred = torch.cat(pred_list, dim=0)
    true = torch.cat(true_list, dim=0)
    mse = torch.mean((pred - true) ** 2).item()
    mae = torch.mean(torch.abs(pred - true)).item()
    max_abs_err = torch.max(torch.abs(pred - true)).item()
    return {
        "mse": mse,
        "mae": mae,
        "max_abs_err": max_abs_err
    }
```

**解释：**

- `def evaluate_subset(model, subset, obs_mean, obs_std, config):`：函数作用：在指定数据集或 batch 上评估模型，不更新参数。
- `batch_size=config.batch_size,`：batch_size 的作用：每个 batch 的样本数。
- `with torch.no_grad():`：进入不记录梯度的区域，通常用于计算 TD 目标或验证指标。
- `batch_obs = batch_obs.to(config.device, non_blocking=True)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `batch_act = batch_act.to(config.device, non_blocking=True)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `return {`：返回本次更新或评估的指标字典，供训练日志记录。

### 函数 `save_plots`
**代码：**

```python
def save_plots(df, config):
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], label="train_loss")
    plt.plot(df["epoch"], df["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("BC Train / Val Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(config.output_dir, "train_val_loss.png"))
    plt.close()
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["lr"], label="lr")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(config.output_dir, "lr_curve.png"))
    plt.close()
```

**解释：**

- `def save_plots(df, config):`：函数作用：保存训练日志和曲线，方便观察训练过程和写论文结果。

### 函数 `train_bc`
**代码：**

```python
def train_bc(config):
    set_seed(config.seed)
    (
        train_loader,
        val_loader,
        state_dim,
        action_dim,
        obs_all,
        act_all,
        train_dataset,
        val_dataset
    ) = load_dataset(config)
    obs_mean, obs_std = compute_obs_stats(train_dataset)
    obs_mean = obs_mean.to(config.device)
    obs_std = obs_std.to(config.device)
    model = BCPolicy(
        state_dim,
        action_dim,
        config.hidden_layers,
        use_layernorm=config.use_layernorm
    ).to(config.device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs
    )
    loss_fn = nn.SmoothL1Loss() if config.use_smooth_l1 else nn.MSELoss()
    results = []
    best_val_loss = float("inf")
    best_model_path = None
    for epoch in range(1, config.num_epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, obs_mean, obs_std, config
        )
        val_loss = evaluate_loss(
            model, val_loader, loss_fn, obs_mean, obs_std, config
        )
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch}/{config.num_epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"LR: {current_lr:.8f}"
        )
        results.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": current_lr
        })
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = save_best_model(
                model=model,
                obs_mean=obs_mean,
                obs_std=obs_std,
                state_dim=state_dim,
                action_dim=action_dim,
                config=config,
                epoch=epoch,
                best_val_loss=best_val_loss
            )
            print(f"Best model saved at epoch {epoch}, val_loss={val_loss:.6f}")
        scheduler.step()
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(config.output_dir, "training_log.csv"), index=False)
    save_plots(df, config)
    print(f"\nTraining finished.")
    print(f"Best model saved to: {best_model_path}")
    return model, obs_all, act_all, obs_mean, obs_std, train_dataset, val_dataset
```

**解释：**

- `def train_bc(config):`：函数作用：BC 训练总流程，负责加载数据、计算标准化参数、创建策略网络、循环训练验证、保存最佳模型和训练曲线。
- `set_seed(config.seed)`：设置随机种子，让数据划分、初始化和训练过程尽量可复现。
- `load_dataset(config)`：加载离线数据集，得到训练/验证 DataLoader、状态维度、动作维度以及原始状态动作数据。
- `obs_mean, obs_std = compute_obs_stats(train_dataset)`：只用训练集计算状态均值和标准差，后面用它们对状态做标准化。
- `obs_mean = obs_mean.to(config.device)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `obs_std = obs_std.to(config.device)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `model = BCPolicy(`：创建策略网络。它输入状态，输出动作，是后续训练要优化的核心模型。
- `).to(config.device)`：把张量移动到 CPU/GPU 设备上，保证和模型在同一设备计算。
- `optimizer = optim.Adam(`：创建 Adam 优化器，后面根据梯度更新网络参数。
- `scheduler = optim.lr_scheduler.CosineAnnealingLR(`：创建学习率调度器，让学习率随训练轮数变化，后期训练更稳定。
- `loss_fn = nn.SmoothL1Loss() if config.use_smooth_l1 else nn.MSELoss()`：选择行为克隆损失函数：SmoothL1 更抗异常值，MSE 直接惩罚平方误差。
- `results = []`：创建日志容器，用来记录每个 epoch 的训练损失、验证损失和学习率。
- `best_val_loss = float("inf")`：初始化最佳验证指标为无穷大，后面只要验证损失更小就保存模型。
- `best_model_path = None`：初始化最佳模型保存路径，真正保存模型后会更新它。
- `for epoch in range(1, config.num_epochs + 1):`：开始按 epoch 训练模型，每轮都会训练一次、验证一次，并记录指标。
- `train_loss = train_one_epoch(`：训练一个 epoch：前向预测动作、计算损失、反向传播并更新策略网络参数。
- `val_loss = evaluate_loss(`：在验证集上计算损失，只评估不更新参数，用它判断模型是否变好。
- `current_lr = optimizer.param_groups[0]["lr"]`：读取当前学习率，用于日志打印和训练记录。
- `print(`：把当前训练或评估信息输出到终端。
- `results.append({`：记录当前 epoch 的训练指标，后面保存成表格和曲线。
- `if val_loss < best_val_loss:`：如果当前验证结果优于历史最好结果，就保存当前模型。
- `best_val_loss = val_loss`：初始化最佳验证指标为无穷大，后面只要验证损失更小就保存模型。
- `best_model_path = save_best_model(`：初始化最佳模型保存路径，真正保存模型后会更新它。
- `print(f"Best model saved at epoch {epoch}, val_loss={val_loss:.6f}")`：把当前训练或评估信息输出到终端。
- `scheduler.step()`：优化器根据梯度更新参数。
- `df = pd.DataFrame(results)`：把训练记录转换成 pandas 表格，便于保存 CSV 和画曲线。
- `df.to_csv(os.path.join(config.output_dir, "training_log.csv"), index=False)`：把表格结果保存成 CSV 文件，方便论文记录、画图或复查。
- `save_plots(df, config)`：根据训练日志保存损失曲线，方便观察训练是否收敛。
- `print(f"\nTraining finished.")`：把当前训练或评估信息输出到终端。
- `print(f"Best model saved to: {best_model_path}")`：把当前训练或评估信息输出到终端。
- `return model, obs_all, act_all, obs_mean, obs_std, train_dataset, val_dataset`：返回训练结果、数据和标准化参数，供主流程继续评估或加载模型。

### 函数 `main`
**代码：**

```python
def main():
    cfg = build_config_from_args()
    print("Using config:")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")
    _, obs_all, act_all, _, _, train_dataset, val_dataset = train_bc(cfg)
    model, obs_mean, obs_std, checkpoint = load_best_model(cfg)
  
    all_metrics = evaluate_subset(model, TensorDataset(obs_all, act_all), obs_mean, obs_std, cfg)
    train_metrics = evaluate_subset(model, train_dataset, obs_mean, obs_std, cfg)
    val_metrics = evaluate_subset(model, val_dataset, obs_mean, obs_std, cfg)
    summary_df = pd.DataFrame([
        {"split": "all", **all_metrics},
        {"split": "train", **train_metrics},
        {"split": "val", **val_metrics},
    ])
    summary_df.to_csv(os.path.join(cfg.output_dir, "evaluation_summary.csv"), index=False)
    print("\nEvaluation summary:")
    print(summary_df.to_string(index=False))
```

**解释：**

- `def main():`：函数作用：脚本主入口，负责读取配置、启动训练、加载最佳模型、做最终评估并保存评估结果。
- `cfg = build_config_from_args()`：读取命令行参数并生成配置对象 `cfg`，后面的训练、保存和评估都使用这份配置。
- `print("Using config:")`：把当前训练或评估信息输出到终端。
- `for k, v in asdict(cfg).items():`：遍历配置对象中的每个参数，把本次实验配置打印出来。
- `print(f"  {k}: {v}")`：把当前训练或评估信息输出到终端。
- `_, obs_all, act_all, _, _, train_dataset, val_dataset = train_bc(cfg)`：调用 `train_bc(cfg)` 启动训练，并取出后续评估需要的数据集和状态动作数据；下划线表示该返回值暂时不用。
- `model, obs_mean, obs_std, checkpoint = load_best_model(cfg)`：训练结束后加载验证集表现最好的模型，而不是简单使用最后一轮模型。
- `all_metrics = evaluate_subset(model, TensorDataset(obs_all, act_all), obs_mean, obs_std, cfg)`：在全部数据上评估模型预测动作和真实动作的差距。
- `train_metrics = evaluate_subset(model, train_dataset, obs_mean, obs_std, cfg)`：在训练集上评估模型拟合情况。
- `val_metrics = evaluate_subset(model, val_dataset, obs_mean, obs_std, cfg)`：在验证集上评估模型泛化能力。
- `summary_df = pd.DataFrame([`：把 all、train、val 三组评估指标整理成表格。
- `summary_df.to_csv(os.path.join(cfg.output_dir, "evaluation_summary.csv"), index=False)`：把表格结果保存成 CSV 文件，方便论文记录、画图或复查。
- `print("\nEvaluation summary:")`：把当前训练或评估信息输出到终端。
- `print(summary_df.to_string(index=False))`：把当前训练或评估信息输出到终端。


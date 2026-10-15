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

    # 算法小改进
    
    use_smooth_l1: bool = True
    

    num_workers: int = 0

#将用户从命令行输入的 字符串 （或列表）解析为神经网络可以理解的 整数列表 。
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

#解析命令行参数 、 决定计算设备 、 初始化配置对象 并 记录训练元数据 。
def build_config_from_args():
    parser = argparse.ArgumentParser(description="Behavior Cloning Training")

    # 基础参数
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

    # 布尔参数
    parser.add_argument("--use-layernorm", action="store_true", help="启用 LayerNorm")
    parser.add_argument("--use-smooth-l1", action="store_true", help="启用 SmoothL1Loss")
    parser.add_argument("--use-mse", action="store_true", help="启用 MSELoss（优先级高于 use_smooth_l1）")

    # 设备选择
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--gpu", action="store_true", help="强制使用 GPU")
    group.add_argument("--cpu", action="store_true", help="强制使用 CPU")

    args = parser.parse_args()

    # 设备逻辑
    if args.cpu:
        device = torch.device("cpu")
    elif args.gpu:
        if not torch.cuda.is_available():
            raise RuntimeError("你指定了 --gpu，但当前环境没有可用的 CUDA/GPU。")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # loss 逻辑
    # 默认 True，但如果显式指定 use_mse，则强制 False
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

    # 保存本次训练配置，方便网页侧追踪
    save_cfg = asdict(cfg)
    save_cfg["device"] = str(cfg.device)
    with open(os.path.join(cfg.output_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(save_cfg, f, ensure_ascii=False, indent=2)

    return cfg


# ====================
# 2. 随机种子
# ====================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ====================
# 3. 数据加载与划分
# ====================
#将磁盘上的原始数据文件转换为深度学习模型可以直接读取的格式 ，并完成训练集与验证集的划分。
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


# ====================
# 4. 计算标准化参数
# ====================
def compute_obs_stats(train_dataset):
    obs_list = []
    for obs_item, _ in train_dataset:
        obs_list.append(obs_item.unsqueeze(0))
    obs_tensor = torch.cat(obs_list, dim=0)

    obs_mean = obs_tensor.mean(dim=0)
    obs_std = torch.clamp(obs_tensor.std(dim=0), min=1e-6)
    return obs_mean, obs_std


# ====================
# 5. BC Policy
# ====================
#用于预测动作的 神经网络模型 。
class BCPolicy(nn.Module):
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

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        x = torch.tanh(x)   # 动作范围限制到 [-1, 1]
        return x


# ====================
# 6. 单轮训练
# ====================
#整个训练过程的 核心执行单元 。它的任务是遍历一遍训练数据集，计算损失并更新神经网络的权重。
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


# ====================
# 7. 验证
# ====================
#在验证集（或测试集）上评估模型的性能
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


# ====================
# 8. 保存最佳模型
# ====================
#将模型及其“配套信息”完整地保存到磁盘文件中 。
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


# ====================
# 9. 加载最佳模型
# ====================
#从磁盘读取保存的模型文件，并完美还原出一个可以立即使用的模型对象 。
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


# ====================
# 10. 子集评估
# ====================
#在给定的子集（如验证集或测试集）上评估模型的性能
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


# ====================
# 11. 保存训练图
# ====================
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


# ====================
# 12. 主训练函数
# ====================
#算法的 核心工作流 。它负责把数据、模型、优化器和训练循环有机地结合在一起。
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


# ====================
# 13. 主流程
# ====================
def main():
    cfg = build_config_from_args()

    print("Using config:")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")

    _, obs_all, act_all, _, _, train_dataset, val_dataset = train_bc(cfg)

    model, obs_mean, obs_std, checkpoint = load_best_model(cfg)

    # 统一评估输出
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


if __name__ == "__main__":
    main()
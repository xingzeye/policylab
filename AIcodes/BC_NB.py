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
@dataclass
class Config:
    data_path: str = "halfcheetah_medium-v2.hdf5"
    batch_size: int = 256
    lr: float = 3e-4
    num_epochs: int = 150
    hidden_layers: list = field(default_factory=lambda: [256, 256])
    output_dir: str = "checkpoints/BC_NB"
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed: int = 42
    val_ratio: float = 0.1
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    best_model_name: str = "best_model.pt"

    # 算法改进参数
    use_layernorm: bool = False
    loss_type: str = "smooth_l1"   # "mse" or "smooth_l1"

    # Return-weighted BC
    weight_temperature: float = 3.0
    min_weight: float = 0.2
    max_weight: float = 5.0

    action_clip_eps: float = 1e-6
    num_workers: int = 0


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


def build_config_from_args():
    parser = argparse.ArgumentParser(description="Return-weighted Behavior Cloning Training")

    # 基础参数
    parser.add_argument("--data-path", type=str, default="halfcheetah_medium-v2.hdf5")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-epochs", type=int, default=150)
    parser.add_argument("--hidden-layers", type=str, default="256,256")
    parser.add_argument("--output-dir", type=str, default="checkpoints/BC_NB")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--best-model-name", type=str, default="best_model.pt")
    parser.add_argument("--action-clip-eps", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=0)

    # loss / 网络
    parser.add_argument("--use-layernorm", action="store_true")
    parser.add_argument("--loss-type", type=str, default="smooth_l1", choices=["mse", "smooth_l1"])

    # Return-weighted BC
    parser.add_argument("--weight-temperature", type=float, default=3.0)
    parser.add_argument("--min-weight", type=float, default=0.2)
    parser.add_argument("--max-weight", type=float, default=5.0)

    # 设备
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
        loss_type=args.loss_type,
        weight_temperature=args.weight_temperature,
        min_weight=args.min_weight,
        max_weight=args.max_weight,
        action_clip_eps=args.action_clip_eps,
        num_workers=args.num_workers,
    )

    os.makedirs(cfg.output_dir, exist_ok=True)

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
# 3. 根据 terminals 切分 episode 并计算 return-based 权重
# ====================
def build_sample_weights(rewards, terminals, config):
    rewards_np = rewards.cpu().numpy()
    terminals_np = terminals.cpu().numpy()

    n = len(rewards_np)
    episode_ranges = []
    start = 0

    for i in range(n):
        if terminals_np[i] > 0.5:
            episode_ranges.append((start, i))
            start = i + 1

    if start < n:
        episode_ranges.append((start, n - 1))

    episode_returns = np.zeros(len(episode_ranges), dtype=np.float32)

    for ep_id, (l, r) in enumerate(episode_ranges):
        episode_returns[ep_id] = rewards_np[l:r + 1].sum()

    ret_mean = episode_returns.mean()
    ret_std = episode_returns.std() + 1e-6
    norm_returns = (episode_returns - ret_mean) / ret_std

    episode_weights = np.exp(norm_returns / config.weight_temperature)
    episode_weights = np.clip(
        episode_weights,
        config.min_weight,
        config.max_weight
    )

    sample_weights = np.zeros(n, dtype=np.float32)
    for ep_id, (l, r) in enumerate(episode_ranges):
        sample_weights[l:r + 1] = episode_weights[ep_id]

    sample_weights = sample_weights / (sample_weights.mean() + 1e-6)

    return torch.tensor(sample_weights, dtype=torch.float32)


# ====================
# 4. 数据加载与划分
# ====================
def load_dataset(config):
    with h5py.File(config.data_path, "r") as f:
        obs = torch.tensor(f["observations"][:], dtype=torch.float32)
        actions = torch.tensor(f["actions"][:], dtype=torch.float32)
        rewards = torch.tensor(f["rewards"][:], dtype=torch.float32)
        terminals = torch.tensor(f["terminals"][:], dtype=torch.float32)

    sample_weights = build_sample_weights(rewards, terminals, config)

    state_dim = obs.shape[1]
    action_dim = actions.shape[1]

    dataset = TensorDataset(obs, actions, sample_weights)

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

    return (
        train_loader,
        val_loader,
        state_dim,
        action_dim,
        obs,
        actions,
        sample_weights,
        train_dataset,
        val_dataset
    )


# ====================
# 5. 计算状态标准化参数
# ====================
def compute_obs_stats(train_dataset):
    obs_list = []
    for obs_item, _, _ in train_dataset:
        obs_list.append(obs_item.unsqueeze(0))
    obs_tensor = torch.cat(obs_list, dim=0)

    obs_mean = obs_tensor.mean(dim=0)
    obs_std = torch.clamp(obs_tensor.std(dim=0), min=1e-6)
    return obs_mean, obs_std


# ====================
# 6. Policy 网络
# ====================
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
        x = torch.tanh(x)
        return x


# ====================
# 7. 逐样本损失
# ====================
def per_sample_loss(pred, target, loss_type="smooth_l1"):
    if loss_type == "mse":
        return ((pred - target) ** 2).mean(dim=1)
    elif loss_type == "smooth_l1":
        return nn.functional.smooth_l1_loss(pred, target, reduction="none").mean(dim=1)
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")


# ====================
# 8. 单轮训练
# ====================
def train_one_epoch(model, loader, optimizer, obs_mean, obs_std, config):
    model.train()
    total_loss = 0.0
    total_count = 0

    for batch_obs, batch_act, batch_w in loader:
        batch_obs = batch_obs.to(config.device, non_blocking=True)
        batch_act = batch_act.to(config.device, non_blocking=True)
        batch_w = batch_w.to(config.device, non_blocking=True)

        batch_obs = (batch_obs - obs_mean) / obs_std
        batch_act = torch.clamp(batch_act, -1.0 + config.action_clip_eps, 1.0 - config.action_clip_eps)

        pred_act = model(batch_obs)

        sample_loss = per_sample_loss(pred_act, batch_act, config.loss_type)
        loss = (sample_loss * batch_w).mean()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        bs = batch_obs.size(0)
        total_loss += loss.item() * bs
        total_count += bs

    return total_loss / total_count


# ====================
# 9. 测试/验证
# ====================
def evaluate_loss(model, loader, obs_mean, obs_std, config):
    model.eval()
    total_weighted_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for batch_obs, batch_act, batch_w in loader:
            batch_obs = batch_obs.to(config.device, non_blocking=True)
            batch_act = batch_act.to(config.device, non_blocking=True)
            batch_w = batch_w.to(config.device, non_blocking=True)

            batch_obs = (batch_obs - obs_mean) / obs_std
            batch_act = torch.clamp(batch_act, -1.0 + config.action_clip_eps, 1.0 - config.action_clip_eps)

            pred_act = model(batch_obs)

            sample_loss = per_sample_loss(pred_act, batch_act, config.loss_type)
            weighted_loss = (sample_loss * batch_w).mean()

            bs = batch_obs.size(0)
            total_weighted_loss += weighted_loss.item() * bs
            total_count += bs

    return {
        "weighted_loss": total_weighted_loss / total_count
    }


# ====================
# 10. 保存最佳模型
# ====================
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
        "loss_type": config.loss_type,
        "weight_temperature": config.weight_temperature,
        "min_weight": config.min_weight,
        "max_weight": config.max_weight,
        "best_val_loss": best_val_loss,
        "epoch": epoch
    }, save_path)
    return save_path


# ====================
# 11. 加载最佳模型
# ====================
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
# 12. 子集评估
# ====================
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
    weight_list = []

    with torch.no_grad():
        for batch_obs, batch_act, batch_w in loader:
            batch_obs = batch_obs.to(config.device, non_blocking=True)
            batch_act = batch_act.to(config.device, non_blocking=True)
            batch_w = batch_w.to(config.device, non_blocking=True)

            batch_obs = (batch_obs - obs_mean) / obs_std
            pred_act = model(batch_obs)

            pred_list.append(pred_act.cpu())
            true_list.append(batch_act.cpu())
            weight_list.append(batch_w.cpu())

    pred = torch.cat(pred_list, dim=0)
    true = torch.cat(true_list, dim=0)
    weights = torch.cat(weight_list, dim=0)

    per_mse = ((pred - true) ** 2).mean(dim=1)
    per_mae = torch.abs(pred - true).mean(dim=1)

    mse = per_mse.mean().item()
    mae = per_mae.mean().item()
    weighted_mse = (per_mse * weights).mean().item()
    weighted_mae = (per_mae * weights).mean().item()
    max_abs_err = torch.max(torch.abs(pred - true)).item()

    return {
        "mse": mse,
        "mae": mae,
        "weighted_mse": weighted_mse,
        "weighted_mae": weighted_mae,
        "max_abs_err": max_abs_err
    }


# ====================
# 13. 保存图表
# ====================
def save_plots(df, config):
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], label="train_loss")
    plt.plot(df["epoch"], df["test_loss"], label="test_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("RWBC Train / Test Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(config.output_dir, "train_test_loss.png"))
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
# 14. 训练主函数
# ====================
def train_bc(config):
    set_seed(config.seed)

    (
        train_loader,
        val_loader,
        state_dim,
        action_dim,
        obs_all,
        act_all,
        sample_weights,
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

    results = []
    best_val_loss = float("inf")
    best_model_path = None

    for epoch in range(1, config.num_epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, obs_mean, obs_std, config
        )

        test_info = evaluate_loss(
            model, val_loader, obs_mean, obs_std, config
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch}/{config.num_epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Test Loss: {test_info['weighted_loss']:.6f} | "
            f"LR: {current_lr:.8f}"
        )

        results.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_info["weighted_loss"],
            "lr": current_lr
        })

        if test_info["weighted_loss"] < best_val_loss:
            best_val_loss = test_info["weighted_loss"]
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
            print(f"Best model saved at epoch {epoch}, test_loss={best_val_loss:.6f}")

        scheduler.step()

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(config.output_dir, "training_log.csv"), index=False)
    save_plots(df, config)

    print("\nTraining finished.")
    print(f"Best model saved to: {best_model_path}")

    return model, obs_all, act_all, sample_weights, obs_mean, obs_std, train_dataset, val_dataset


# ====================
# 15. 主流程
# ====================
def main():
    cfg = build_config_from_args()

    print("Using config:")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")

    _, obs_all, act_all, sample_weights, _, _, train_dataset, val_dataset = train_bc(cfg)

    model, obs_mean, obs_std, checkpoint = load_best_model(cfg)

    full_dataset = TensorDataset(obs_all, act_all, sample_weights)

    all_metrics = evaluate_subset(model, full_dataset, obs_mean, obs_std, cfg)
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
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
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


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


# ------------------------
# 参数配置
# ------------------------
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


# ------------------------
# Actor & Critic 网络
# ------------------------
class Actor(nn.Module):
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

    def forward(self, state):
        return self.model(state) * self.max_action


class Critic(nn.Module):
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

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)

    def q1_forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa)


# ------------------------
# Utility: 计算训练集统计量
# ------------------------
def compute_obs_stats(train_dataset):
    obs_list = []
    for item in train_dataset:
        obs_list.append(item[0].unsqueeze(0))
    obs_tensor = torch.cat(obs_list, dim=0)
    obs_mean = obs_tensor.mean(dim=0)
    obs_std = obs_tensor.std(dim=0)
    obs_std = torch.clamp(obs_std, min=1e-6)
    return obs_mean, obs_std


# ------------------------
# Utility: soft update
# ------------------------
def soft_update(source_net, target_net, tau):
    for param, target_param in zip(source_net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


# ------------------------
# TD3 Trainer
# ------------------------
class TD3Trainer:
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

    def set_schedulers(self, actor_scheduler, critic_scheduler):
        self.actor_scheduler = actor_scheduler
        self.critic_scheduler = critic_scheduler

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


# ------------------------
# Utility: plot & save CSV
# ------------------------
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


# ------------------------
# Main function
# ------------------------
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


if __name__ == "__main__":
    main()
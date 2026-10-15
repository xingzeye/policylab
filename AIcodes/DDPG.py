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
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def ensure_dirs(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


# =====================
# 1. 参数配置
# =====================
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

    # 算法增强
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


# =====================
# 2. Actor & Critic 网络
# =====================
class Actor(nn.Module):
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

    def forward(self, state):
        return self.model(state)


class Critic(nn.Module):
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

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.model(x)


# =====================
# 3. Offline Dataset
# =====================
class OfflineDataset(torch.utils.data.Dataset):
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

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.terminals[idx]
        )


# =====================
# 4. 计算训练集状态统计量
# =====================
def compute_obs_stats(subset):
    obs_list = []
    for item in subset:
        obs_list.append(item[0].unsqueeze(0))
    obs_tensor = torch.cat(obs_list, dim=0)

    obs_mean = obs_tensor.mean(dim=0)
    obs_std = torch.clamp(obs_tensor.std(dim=0), min=1e-6)
    return obs_mean, obs_std


# =====================
# 5. Advantage-Weighted DDPG_NB Agent
# =====================
class OfflineAWDDPGAgent:
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

    def set_schedulers(self, actor_scheduler, critic_scheduler):
        self.actor_scheduler = actor_scheduler
        self.critic_scheduler = critic_scheduler

    def soft_update(self, net, net_target):
        for param, target_param in zip(net.parameters(), net_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def select_action(self, state, obs_mean=None, obs_std=None):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        if obs_mean is not None and obs_std is not None:
            state = (state - obs_mean) / obs_std
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]
        return np.clip(action, -1.0, 1.0)

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


# =====================
# 6. 训练与验证
# =====================
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


# =====================
# 7. 加载最优模型
# =====================
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


# =====================
# 8. Main
# =====================
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


if __name__ == "__main__":
    main()
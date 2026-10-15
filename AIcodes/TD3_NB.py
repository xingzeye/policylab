import os
import json
import h5py
import random
import argparse
from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split


# ------------------------
# 工具函数
# ------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


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


# ------------------------
# 参数配置
# ------------------------
@dataclass
class Config:
    data_path: str = "halfcheetah_medium-v2.hdf5"
    save_dir: str = "checkpoints/TD3_NB"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size: int = 256
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2

    max_action: float = 1.0
    num_epochs: int = 150
    hidden: list = field(default_factory=lambda: [256, 256])

    seed: int = 42
    val_ratio: float = 0.1
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    num_workers: int = 0

    # TD3+BC 参数
    alpha_bc: float = 2.5
    bc_coef: float = 1.0
    reward_scale: bool = True

    best_model_name: str = "best_model.pt"
    log_csv_name: str = "training_log.csv"
    critic_curve_name: str = "critic_loss.png"
    actor_curve_name: str = "actor_loss.png"
    q_curve_name: str = "q_value.png"


def build_config_from_args():
    parser = argparse.ArgumentParser(description="Offline TD3_NB / TD3+BC Training")

    parser.add_argument("--data-path", type=str, default="halfcheetah_medium-v2.hdf5")
    parser.add_argument("--save-dir", type=str, default="checkpoints/TD3_NB")

    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr-actor", type=float, default=3e-4)
    parser.add_argument("--lr-critic", type=float, default=3e-4)
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

    parser.add_argument("--alpha-bc", type=float, default=2.5, help="Q项自适应缩放系数")
    parser.add_argument("--bc-coef", type=float, default=1.0, help="行为克隆项权重")

    parser.add_argument("--reward-scale", dest="reward_scale", action="store_true")
    parser.add_argument("--no-reward-scale", dest="reward_scale", action="store_false")
    parser.set_defaults(reward_scale=True)

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
        alpha_bc=args.alpha_bc,
        bc_coef=args.bc_coef,
        reward_scale=args.reward_scale,
        best_model_name=args.best_model_name,
        log_csv_name=args.log_csv_name,
        critic_curve_name=args.critic_curve_name,
        actor_curve_name=args.actor_curve_name,
        q_curve_name=args.q_curve_name,
    )

    ensure_dir(cfg.save_dir)

    save_cfg = asdict(cfg)
    with open(os.path.join(cfg.save_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(save_cfg, f, ensure_ascii=False, indent=2)

    return cfg


# ------------------------
# Actor & Critic
# ------------------------
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden=(256, 256)):
        super().__init__()
        dims = [state_dim] + list(hidden)
        layers = []

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

        dims = [state_dim + action_dim] + list(hidden)

        q1_layers = []
        for i in range(len(hidden)):
            q1_layers.append(nn.Linear(dims[i], dims[i + 1]))
            q1_layers.append(nn.ReLU())
        q1_layers.append(nn.Linear(hidden[-1], 1))
        self.q1 = nn.Sequential(*q1_layers)

        q2_layers = []
        for i in range(len(hidden)):
            q2_layers.append(nn.Linear(dims[i], dims[i + 1]))
            q2_layers.append(nn.ReLU())
        q2_layers.append(nn.Linear(hidden[-1], 1))
        self.q2 = nn.Sequential(*q2_layers)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)

    def q1_forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa)


# ------------------------
# Utils
# ------------------------
def compute_obs_stats(train_dataset):
    obs_list = []
    reward_list = []
    for item in train_dataset:
        obs_list.append(item[0].unsqueeze(0))
        reward_list.append(item[2].view(1))

    obs_tensor = torch.cat(obs_list, dim=0)
    reward_tensor = torch.cat(reward_list, dim=0)

    obs_mean = obs_tensor.mean(dim=0)
    obs_std = torch.clamp(obs_tensor.std(dim=0), min=1e-6)

    reward_mean = reward_tensor.mean()
    reward_std = torch.clamp(reward_tensor.std(), min=1e-6)

    return obs_mean, obs_std, reward_mean, reward_std


def soft_update(source_net, target_net, tau):
    for p, tp in zip(source_net.parameters(), target_net.parameters()):
        tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)


# ------------------------
# TD3 Trainer
# N = normalization
# B = behavior cloning regularization
# ------------------------
class TD3NBTrainer:
    def __init__(self, state_dim, action_dim, max_action, args):
        self.device = args.device
        self.max_action = max_action

        self.actor = Actor(state_dim, action_dim, max_action, args.hidden).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action, args.hidden).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim, args.hidden).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, args.hidden).to(self.device)
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

        self.actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.actor_optimizer, T_max=args.num_epochs
        )
        self.critic_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.critic_optimizer, T_max=args.num_epochs
        )

        self.gamma = args.gamma
        self.tau = args.tau
        self.policy_noise = args.policy_noise
        self.noise_clip = args.noise_clip
        self.policy_delay = args.policy_delay
        self.grad_clip = args.grad_clip

        self.alpha = args.alpha_bc
        self.bc_coef = args.bc_coef

        self.total_it = 0

    def train_step(self, batch, obs_mean, obs_std, reward_mean, reward_std, reward_scale):
        self.total_it += 1

        state, action, reward, next_state, done = batch
        state = state.to(self.device, non_blocking=True)
        action = action.to(self.device, non_blocking=True)
        reward = reward.to(self.device, non_blocking=True).unsqueeze(1)
        next_state = next_state.to(self.device, non_blocking=True)
        done = done.to(self.device, non_blocking=True).unsqueeze(1)

        state = (state - obs_mean) / obs_std
        next_state = (next_state - obs_mean) / obs_std

        if reward_scale:
            reward = (reward - reward_mean) / reward_std

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1.0 - done) * self.gamma * target_q

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = nn.functional.mse_loss(current_q1, target_q) + nn.functional.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_optimizer.step()

        actor_loss_value = None
        q_term_value = None
        bc_term_value = None
        actor_updated = 0

        if self.total_it % self.policy_delay == 0:
            pi = self.actor(state)
            q_pi = self.critic.q1_forward(state, pi)

            lam = self.alpha / (q_pi.abs().mean().detach() + 1e-6)

            q_term = -lam * q_pi.mean()
            bc_term = self.bc_coef * nn.functional.mse_loss(pi, action)
            actor_loss = q_term + bc_term

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
            self.actor_optimizer.step()

            soft_update(self.actor, self.actor_target, self.tau)
            soft_update(self.critic, self.critic_target, self.tau)

            actor_loss_value = actor_loss.item()
            q_term_value = q_term.item()
            bc_term_value = bc_term.item()
            actor_updated = 1

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss_value,
            "actor_updated": actor_updated,
            "q_term": q_term_value,
            "bc_term": bc_term_value,
            "target_q": target_q.mean().item(),
            "current_q1": current_q1.mean().item(),
            "current_q2": current_q2.mean().item(),
            "reward": reward.mean().item(),
        }

    @torch.no_grad()
    def eval_step(self, batch, obs_mean, obs_std, reward_mean, reward_std, reward_scale):
        state, action, reward, next_state, done = batch
        state = state.to(self.device, non_blocking=True)
        action = action.to(self.device, non_blocking=True)
        reward = reward.to(self.device, non_blocking=True).unsqueeze(1)
        next_state = next_state.to(self.device, non_blocking=True)
        done = done.to(self.device, non_blocking=True).unsqueeze(1)

        state = (state - obs_mean) / obs_std
        next_state = (next_state - obs_mean) / obs_std

        if reward_scale:
            reward = (reward - reward_mean) / reward_std

        noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

        target_q1, target_q2 = self.critic_target(next_state, next_action)
        target_q = torch.min(target_q1, target_q2)
        target_q = reward + (1.0 - done) * self.gamma * target_q

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = nn.functional.mse_loss(current_q1, target_q) + nn.functional.mse_loss(current_q2, target_q)

        pi = self.actor(state)
        q_pi = self.critic.q1_forward(state, pi)
        lam = self.alpha / (q_pi.abs().mean().detach() + 1e-6)
        q_term = -lam * q_pi.mean()
        bc_term = self.bc_coef * nn.functional.mse_loss(pi, action)
        actor_proxy = q_term + bc_term

        return {
            "critic_loss": critic_loss.item(),
            "actor_proxy": actor_proxy.item(),
            "q_term": q_term.item(),
            "bc_term": bc_term.item(),
            "target_q": target_q.mean().item(),
            "current_q1": current_q1.mean().item(),
            "current_q2": current_q2.mean().item(),
            "reward": reward.mean().item(),
        }

    def save_best(self, save_path, obs_mean, obs_std, reward_mean, reward_std, epoch, best_val_loss, args):
        ensure_dir(os.path.dirname(save_path))
        torch.save({
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "obs_mean": obs_mean.detach().cpu(),
            "obs_std": obs_std.detach().cpu(),
            "reward_mean": reward_mean.detach().cpu(),
            "reward_std": reward_std.detach().cpu(),
            "state_dim": args.state_dim,
            "action_dim": args.action_dim,
            "max_action": args.max_action,
            "hidden": list(args.hidden),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "alpha_bc": args.alpha_bc,
            "bc_coef": args.bc_coef,
            "reward_scale": args.reward_scale
        }, save_path)


# ------------------------
# Plot & CSV
# ------------------------
def save_plot_csv(df, cfg):
    ensure_dir(cfg.save_dir)
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
# Main
# ------------------------
def main():
    cfg = build_config_from_args()

    set_seed(cfg.seed)
    ensure_dir(cfg.save_dir)

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

    obs_mean, obs_std, reward_mean, reward_std = compute_obs_stats(train_dataset)
    obs_mean = obs_mean.to(cfg.device)
    obs_std = obs_std.to(cfg.device)
    reward_mean = reward_mean.to(cfg.device)
    reward_std = reward_std.to(cfg.device)

    state_dim = obs.shape[1]
    action_dim = actions.shape[1]
    cfg.state_dim = state_dim
    cfg.action_dim = action_dim

    trainer = TD3NBTrainer(state_dim, action_dim, cfg.max_action, cfg)

    history = {
        "epoch": [],
        "train_critic_loss": [],
        "train_actor_loss": [],
        "train_q_term": [],
        "train_bc_term": [],
        "train_target_q": [],
        "train_current_q1": [],
        "train_current_q2": [],
        "val_critic_loss": [],
        "val_actor_proxy": [],
        "val_q_term": [],
        "val_bc_term": [],
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
            "current_q2": 0.0
        }

        actor_loss_sum = 0.0
        q_term_sum = 0.0
        bc_term_sum = 0.0
        actor_count = 0

        for batch in train_loader:
            out = trainer.train_step(
                batch=batch,
                obs_mean=obs_mean,
                obs_std=obs_std,
                reward_mean=reward_mean,
                reward_std=reward_std,
                reward_scale=cfg.reward_scale
            )

            train_sum["critic_loss"] += out["critic_loss"]
            train_sum["target_q"] += out["target_q"]
            train_sum["current_q1"] += out["current_q1"]
            train_sum["current_q2"] += out["current_q2"]

            if out["actor_updated"] == 1:
                actor_loss_sum += out["actor_loss"]
                q_term_sum += out["q_term"]
                bc_term_sum += out["bc_term"]
                actor_count += 1

        n_train = len(train_loader)

        train_metrics = {
            "critic_loss": train_sum["critic_loss"] / n_train,
            "actor_loss": actor_loss_sum / max(actor_count, 1),
            "q_term": q_term_sum / max(actor_count, 1),
            "bc_term": bc_term_sum / max(actor_count, 1),
            "target_q": train_sum["target_q"] / n_train,
            "current_q1": train_sum["current_q1"] / n_train,
            "current_q2": train_sum["current_q2"] / n_train,
        }

        val_sum = {
            "critic_loss": 0.0,
            "actor_proxy": 0.0,
            "q_term": 0.0,
            "bc_term": 0.0,
            "target_q": 0.0,
            "current_q1": 0.0,
            "current_q2": 0.0
        }

        for batch in val_loader:
            out = trainer.eval_step(
                batch=batch,
                obs_mean=obs_mean,
                obs_std=obs_std,
                reward_mean=reward_mean,
                reward_std=reward_std,
                reward_scale=cfg.reward_scale
            )
            for k in val_sum:
                val_sum[k] += out[k]

        n_val = len(val_loader)
        val_metrics = {k: v / n_val for k, v in val_sum.items()}

        actor_lr = trainer.actor_optimizer.param_groups[0]["lr"]
        critic_lr = trainer.critic_optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d}/{cfg.num_epochs} | "
            f"Train Critic {train_metrics['critic_loss']:.6f} | "
            f"Train Actor {train_metrics['actor_loss']:.6f} | "
            f"Val Critic {val_metrics['critic_loss']:.6f} | "
            f"Val ActorProxy {val_metrics['actor_proxy']:.6f}"
        )

        history["epoch"].append(epoch)
        history["train_critic_loss"].append(train_metrics["critic_loss"])
        history["train_actor_loss"].append(train_metrics["actor_loss"])
        history["train_q_term"].append(train_metrics["q_term"])
        history["train_bc_term"].append(train_metrics["bc_term"])
        history["train_target_q"].append(train_metrics["target_q"])
        history["train_current_q1"].append(train_metrics["current_q1"])
        history["train_current_q2"].append(train_metrics["current_q2"])

        history["val_critic_loss"].append(val_metrics["critic_loss"])
        history["val_actor_proxy"].append(val_metrics["actor_proxy"])
        history["val_q_term"].append(val_metrics["q_term"])
        history["val_bc_term"].append(val_metrics["bc_term"])
        history["val_target_q"].append(val_metrics["target_q"])
        history["val_current_q1"].append(val_metrics["current_q1"])
        history["val_current_q2"].append(val_metrics["current_q2"])

        history["actor_lr"].append(actor_lr)
        history["critic_lr"].append(critic_lr)

        if val_metrics["critic_loss"] < best_val_loss:
            best_val_loss = val_metrics["critic_loss"]
            trainer.save_best(
                save_path=best_model_path,
                obs_mean=obs_mean,
                obs_std=obs_std,
                reward_mean=reward_mean,
                reward_std=reward_std,
                epoch=epoch,
                best_val_loss=best_val_loss,
                args=cfg
            )
            print(f"  -> Best model saved to {best_model_path}")

        trainer.actor_scheduler.step()
        trainer.critic_scheduler.step()

    df = pd.DataFrame(history)
    save_plot_csv(df, cfg)

    print("\nTraining finished.")
    print(f"Best model saved to: {best_model_path}")
    print(f"Logs & plots saved to: {cfg.save_dir}")


if __name__ == "__main__":
    main()
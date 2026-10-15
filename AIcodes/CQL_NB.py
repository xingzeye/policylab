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
def ensure_dirs(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def soft_update(source_net, target_net, tau):
    for source_param, target_param in zip(source_net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


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


# ========================
# Config
# ========================
@dataclass
class Config:
    dataset_path: str = "halfcheetah_medium-v2.hdf5"
    output_dir: str = os.path.join("checkpoints", "CQL_NB")

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

    # 算法增强
    reward_scale: bool = True
    num_random_actions: int = 10
    alpha_bc: float = 2.5
    bc_coef: float = 1.0

    # TD3-style target smoothing
    policy_noise: float = 0.2
    noise_clip: float = 0.5


def build_config_from_args():
    parser = argparse.ArgumentParser(description="CQL_NB Training")

    parser.add_argument("--dataset-path", type=str, default="halfcheetah_medium-v2.hdf5")
    parser.add_argument("--output-dir", type=str, default=os.path.join("checkpoints", "CQL_NB"))

    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-epochs", type=int, default=150)

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

    parser.add_argument("--policy-noise", type=float, default=0.2)
    parser.add_argument("--noise-clip", type=float, default=0.5)

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
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
    )

    ensure_dirs(cfg.output_dir)

    save_cfg = asdict(cfg)
    with open(os.path.join(cfg.output_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(save_cfg, f, ensure_ascii=False, indent=2)

    return cfg


# ========================
# Network
# ========================
class MLP(nn.Module):
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

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        if self.use_tanh:
            x = torch.tanh(x)
        return x


# ========================
# Load Dataset + Split
# ========================
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


# ========================
# Compute Observation Stats
# ========================
def compute_obs_stats(train_dataset):
    obs_list = []
    for item in train_dataset:
        obs_list.append(item[0].unsqueeze(0))
    obs_tensor = torch.cat(obs_list, dim=0)

    obs_mean = obs_tensor.mean(dim=0)
    obs_std = obs_tensor.std(dim=0).clamp(min=1e-6)
    return obs_mean, obs_std


# ========================
# CQL penalty: multi-action logsumexp
# ========================
def compute_cql_penalty(critic, actor, s, s_next, a_data, num_random_actions=10,
                        policy_noise=0.2, noise_clip=0.5):
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

    noise = torch.randn_like(a_data) * policy_noise
    noise = noise.clamp(-noise_clip, noise_clip)
    next_pi_action = (actor(s_next) + noise).clamp(-1.0, 1.0)
    q_next_pi = critic(torch.cat([s_next, next_pi_action], dim=1))

    q_data = critic(torch.cat([s, a_data], dim=1))

    cat_q = torch.cat([q_rand, q_pi, q_next_pi], dim=1)
    penalty = torch.logsumexp(cat_q, dim=1, keepdim=True).mean() - q_data.mean()
    return penalty


# ========================
# Evaluate
# ========================
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

            noise = torch.randn_like(a) * args.policy_noise
            noise = noise.clamp(-args.noise_clip, args.noise_clip)
            next_a = (actor(s_next) + noise).clamp(-1.0, 1.0)

            q_target = r.unsqueeze(1) + args.gamma * (1.0 - done.unsqueeze(1)) * torch.min(
                target_critic1(torch.cat([s_next, next_a], dim=1)),
                target_critic2(torch.cat([s_next, next_a], dim=1))
            )

            q1_pred = critic1(torch.cat([s, a], dim=1))
            q2_pred = critic2(torch.cat([s, a], dim=1))

            mse_q1 = nn.functional.mse_loss(q1_pred, q_target)
            mse_q2 = nn.functional.mse_loss(q2_pred, q_target)
            critic_mse = mse_q1 + mse_q2

            cql1 = compute_cql_penalty(
                critic1, actor, s, s_next, a,
                args.num_random_actions, args.policy_noise, args.noise_clip
            )
            cql2 = compute_cql_penalty(
                critic2, actor, s, s_next, a,
                args.num_random_actions, args.policy_noise, args.noise_clip
            )
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


# ========================
# Train CQL
# ========================
def train_cql_improved(train_loader, val_loader, state_dim, action_dim, train_dataset,
                       reward_mean, reward_std, args, save_dir):
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
                noise = torch.randn_like(a) * args.policy_noise
                noise = noise.clamp(-args.noise_clip, args.noise_clip)
                next_a = (actor(s_next) + noise).clamp(-1.0, 1.0)

                q_target = r.unsqueeze(1) + args.gamma * (1.0 - done.unsqueeze(1)) * torch.min(
                    target_critic1(torch.cat([s_next, next_a], dim=1)),
                    target_critic2(torch.cat([s_next, next_a], dim=1))
                )

            q1_pred = critic1(torch.cat([s, a], dim=1))
            q2_pred = critic2(torch.cat([s, a], dim=1))

            mse_q1 = nn.functional.mse_loss(q1_pred, q_target)
            mse_q2 = nn.functional.mse_loss(q2_pred, q_target)
            critic_mse = mse_q1 + mse_q2

            cql1 = compute_cql_penalty(
                critic1, actor, s, s_next, a,
                args.num_random_actions, args.policy_noise, args.noise_clip
            )
            cql2 = compute_cql_penalty(
                critic2, actor, s, s_next, a,
                args.num_random_actions, args.policy_noise, args.noise_clip
            )
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
            f"Val Actor Loss: {val_metrics['actor_loss']:.6f} | "
            f"Train CQL: {train_metrics['cql_penalty']:.6f} | "
            f"Val CQL: {val_metrics['cql_penalty']:.6f}"
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
                "reward_scale": args.reward_scale,
                "num_random_actions": args.num_random_actions,
                "policy_noise": args.policy_noise,
                "noise_clip": args.noise_clip,
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


# ========================
# Load Best Model
# ========================
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
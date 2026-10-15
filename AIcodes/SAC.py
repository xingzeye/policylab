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
def ensure_dirs(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def soft_update(source_net, target_net, tau):
    for param, target_param in zip(source_net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def gaussian_log_prob(noise, log_std):
    return (-0.5 * (noise.pow(2) + 2 * log_std + math.log(2 * math.pi))).sum(dim=-1, keepdim=True)


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


def is_finite_tensor(x):
    return torch.isfinite(x).all().item()


def sanitize_tensor(x, clamp_min=-1e6, clamp_max=1e6):
    x = torch.nan_to_num(x, nan=0.0, posinf=clamp_max, neginf=clamp_min)
    x = torch.clamp(x, clamp_min, clamp_max)
    return x


# ===============================
# ========== Config =============
# ===============================
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


# ===============================
# ========== Networks ===========
# ===============================
class MLP(nn.Module):
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

    def forward(self, x):
        return self.model(x)


class GaussianActor(nn.Module):
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

    def forward(self, state):
        h = self.trunk(state)
        mu = self.mu_layer(h)
        log_std = self.log_std_layer(h)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std

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


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes):
        super().__init__()
        self.net = MLP(state_dim + action_dim, 1, hidden_sizes)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


# ===============================
# ========== Dataset ============
# ===============================
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


def compute_obs_stats(train_dataset):
    obs_list = []
    for item in train_dataset:
        obs_list.append(item[0].unsqueeze(0))
    obs_tensor = torch.cat(obs_list, dim=0)

    obs_mean = obs_tensor.mean(dim=0)
    obs_std = obs_tensor.std(dim=0).clamp(min=1e-6)
    return obs_mean, obs_std


# ===============================
# ========== Agent ==============
# ===============================
class OfflineAWACSACAgent:
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

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def set_schedulers(self, actor_scheduler, critic1_scheduler, critic2_scheduler):
        self.actor_scheduler = actor_scheduler
        self.critic1_scheduler = critic1_scheduler
        self.critic2_scheduler = critic2_scheduler

    def _compute_awac_weights(self, data_q, v_pi):
        advantage = data_q - v_pi
        adv_scaled = advantage / max(self.cfg.awac_lambda, self.eps)

        max_log_weight = math.log(max(self.cfg.weight_clip, 1.0))
        adv_scaled = torch.clamp(adv_scaled, min=-20.0, max=max_log_weight)

        weights = torch.exp(adv_scaled)
        weights = sanitize_tensor(weights, 0.0, self.cfg.weight_clip)

        return advantage, weights

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


# ===============================
# ========== Training ===========
# ===============================
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


# ===============================
# ========== Save History =======
# ===============================
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


# ===============================
# ========== Load Best Actor ====
# ===============================
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


# ===============================
# ========== Main ===============
# ===============================
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


if __name__ == "__main__":
    main()
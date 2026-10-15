# 算法公式与代码对照讲解

本文档集中说明 `AIcodes` 目录中 10 个离线强化学习脚本的核心公式、公式含义、对应代码片段与代码解释。阅读时可以把它当作答辩复习提纲：先看算法直觉，再看公式，最后对照源码说明公式是怎样落到 PyTorch 代码里的。

## 1. 符号说明

| 符号 | 代码变量 | 含义 |
| --- | --- | --- |
| $s$ | `state` / `obs` / `batch_obs` | 当前状态，也就是 HDF5 中的 `observations`。 |
| $a$ | `action` / `actions` / `batch_act` | 离线数据集中已有的动作，也叫数据动作或行为动作。 |
| $r$ | `reward` / `rewards` / `r` | 当前转移的即时奖励。 |
| $s'$ | `next_state` / `next_obs` / `s_next` | 下一状态，也就是 `next_observations`。 |
| $d$ | `done` / `terminals` | 终止标记，终止时为 1，未终止时为 0。 |
| $\pi_\theta(s)$ | `actor(state)` / `model(batch_obs)` | 策略网络输出的动作。BC 中叫 `model`，actor-critic 算法中叫 `actor`。 |
| $Q_\phi(s,a)$ | `critic(state, action)` | critic 网络估计的状态-动作价值。 |
| $\gamma$ | `gamma` / `self.gamma` | 折扣因子，控制未来奖励的重要程度。 |
| $\tau$ | `tau` / `self.tau` | target 网络软更新系数。 |
| $\alpha$ | `alpha` / `self.alpha` | SAC 中的熵温度系数，控制策略随机性。 |
| $\lambda$ | `lam` / `aw_lambda` | 损失缩放或优势加权温度。不同算法含义略有差异。 |

## 2. 通用离线强化学习流程

大部分脚本都遵循下面的流程：

1. 从 `halfcheetah_medium-v2.hdf5` 读取离线数据。
2. 划分训练集和验证集。
3. 用训练集状态计算 `obs_mean` 和 `obs_std`，训练时执行状态标准化。
4. 构建策略网络 actor 和价值网络 critic。
5. 根据算法公式计算 actor loss、critic loss 或 BC loss。
6. 执行 `zero_grad()`、`backward()`、`step()` 更新网络参数。
7. 记录训练日志、曲线图和 `best_model.pt`。

actor-critic 算法中常见的 TD 目标形式是：

$$
y = r + \gamma (1-d) Q_{\bar{\phi}}(s', a')
$$

其中 $Q_{\bar{\phi}}$ 表示 target critic，$a'$ 表示下一状态动作。如果样本已经终止，即 $d=1$，后续价值会被截断，只保留当前奖励 $r$。

target 网络软更新公式为：

$$
\bar{\theta} \leftarrow \tau \theta + (1-\tau)\bar{\theta}
$$

它让 target 网络缓慢跟随当前网络，避免训练目标变化过快。

---

## 3. BC.py：行为克隆

### 算法直觉

BC（Behavior Cloning）不使用奖励，也不估计 Q 值。它把离线强化学习转成监督学习：输入状态 $s$，直接让网络输出动作 $\pi_\theta(s)$，并尽量接近数据集中的真实动作 $a$。

### 核心公式

$$
L_{BC} = \frac{1}{N}\sum_{i=1}^{N}\ell(\pi_\theta(s_i), a_i)
$$

代码中 $\ell$ 可以是 Smooth L1 Loss 或 MSE Loss：

$$
\ell_{MSE}(\pi(s),a)=\|\pi(s)-a\|_2^2
$$

### 公式解释

- $s_i$：第 $i$ 个离线样本的状态。
- $a_i$：第 $i$ 个离线样本中的真实动作。
- $\pi_\theta(s_i)$：策略网络预测动作。
- $L_{BC}$：动作拟合误差。越小表示策略越像离线数据中的行为。
- 该公式只解决“像不像数据动作”的问题，不判断动作未来收益高不高。

### 对应代码

来源：`AIcodes/BC.py`

```python
class BCPolicy(nn.Module):
    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        x = torch.tanh(x)
        return x

def train_one_epoch(model, loader, optimizer, loss_fn, obs_mean, obs_std, config):
    for batch_obs, batch_act in loader:
        batch_obs = (batch_obs - obs_mean) / obs_std
        batch_act = torch.clamp(
            batch_act,
            -1.0 + config.action_clip_eps,
            1.0 - config.action_clip_eps
        )

        pred_act = model(batch_obs)
        loss = loss_fn(pred_act, batch_act)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
```

### 代码解释

下面按代码出现顺序逐行解释。

- `class BCPolicy(nn.Module):` 定义行为克隆策略网络类，并继承 PyTorch 的 `nn.Module`，这样该类才能被优化器识别参数、执行前向传播和保存模型权重。
- `def forward(self, x):` 定义网络的前向传播过程。这里的 `x` 就是输入状态，对应公式中的 $s$。
- `x = self.backbone(x)` 把状态输入到主干 MLP 网络中，提取隐藏特征。`backbone` 一般由多层 `Linear + ReLU` 组成。
- `x = self.head(x)` 把隐藏特征映射到动作维度。HalfCheetah 的动作是 6 维，所以这里输出的是 6 个动作分量。
- `x = torch.tanh(x)` 使用 `tanh` 把网络输出压到 $[-1,1]$，对应连续控制环境中的动作范围限制。
- `return x` 返回预测动作，也就是公式里的 $\pi_\theta(s)$。
- `def train_one_epoch(...)` 定义单轮训练函数，一轮训练表示完整遍历一次训练集。
- `for batch_obs, batch_act in loader:` 从 DataLoader 中按 batch 取出状态和动作。`batch_obs` 对应 $s_i$，`batch_act` 对应 $a_i$。
- `batch_obs = (batch_obs - obs_mean) / obs_std` 对状态做标准化，使不同状态维度的数值尺度更接近，训练更稳定。
- `batch_act = torch.clamp(...)` 对数据动作做裁剪，避免动作正好落在边界外。
- `batch_act,` 表示被裁剪的对象是数据集中真实动作。
- `-1.0 + config.action_clip_eps,` 是动作下界，稍微大于 `-1`，留出一个很小的数值余量。
- `1.0 - config.action_clip_eps` 是动作上界，稍微小于 `1`，同样用于避免边界数值问题。
- `)` 结束 `torch.clamp` 调用，得到裁剪后的真实动作标签。
- `pred_act = model(batch_obs)` 将标准化后的状态输入策略网络，得到预测动作 $\pi_\theta(s)$。
- `loss = loss_fn(pred_act, batch_act)` 计算预测动作和真实动作之间的误差，对应 $L_{BC}$。
- `optimizer.zero_grad()` 清空上一轮保留的梯度，避免 PyTorch 默认的梯度累积影响当前更新。
- `loss.backward()` 根据 BC 损失执行反向传播，计算每个网络参数的梯度。
- `nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)` 对梯度做范数裁剪，防止梯度过大导致训练不稳定。
- `optimizer.step()` 根据刚刚计算出的梯度更新策略网络参数。

### 与改进版差异

BC 对所有样本一视同仁。它的优点是简单稳定，缺点是如果数据集中有差动作，也会照样模仿。

---

## 4. BC_NB.py：回报加权行为克隆

### 算法直觉

BC_NB 仍然是行为克隆，但它不再让所有样本同等重要。它先根据 `terminals` 切分 episode，再计算每条轨迹的累计回报。高回报轨迹中的动作权重更高，模型会更愿意模仿这些动作。

### 核心公式

轨迹回报：

$$
G_e = \sum_{t \in e} r_t
$$

标准化回报：

$$
\hat{G}_e = \frac{G_e - \mu_G}{\sigma_G + \epsilon}
$$

轨迹权重：

$$
w_e = \mathrm{clip}\left(\exp\left(\frac{\hat{G}_e}{T}\right), w_{\min}, w_{\max}\right)
$$

加权行为克隆损失：

$$
L_{RWBC} = \frac{1}{N}\sum_{i=1}^{N} w_i \ell(\pi_\theta(s_i), a_i)
$$

### 公式解释

- $G_e$：第 $e$ 条轨迹的累计奖励。
- $\hat{G}_e$：标准化后的轨迹回报，用来消除 reward 数值尺度影响。
- $T$：`weight_temperature`，温度越大，权重差异越平缓。
- $w_{\min}, w_{\max}$：权重裁剪范围，避免极端样本主导训练。
- $w_i$：第 $i$ 个样本所属 episode 的权重。
- 该公式解决普通 BC 的问题：差轨迹也会被模仿。BC_NB 让高回报轨迹对梯度贡献更大。

### 对应代码

来源：`AIcodes/BC_NB.py`

```python
def build_sample_weights(rewards, terminals, config):
    for i in range(n):
        if terminals_np[i] > 0.5:
            episode_ranges.append((start, i))
            start = i + 1

    for ep_id, (l, r) in enumerate(episode_ranges):
        episode_returns[ep_id] = rewards_np[l:r + 1].sum()

    norm_returns = (episode_returns - ret_mean) / ret_std
    episode_weights = np.exp(norm_returns / config.weight_temperature)
    episode_weights = np.clip(
        episode_weights,
        config.min_weight,
        config.max_weight
    )

    for ep_id, (l, r) in enumerate(episode_ranges):
        sample_weights[l:r + 1] = episode_weights[ep_id]

    sample_weights = sample_weights / (sample_weights.mean() + 1e-6)
    return torch.tensor(sample_weights, dtype=torch.float32)

# ------------------------------------------------------------
# 继续：同一算法的另一段核心代码。
# ------------------------------------------------------------

def per_sample_loss(pred, target, loss_type="smooth_l1"):
    if loss_type == "mse":
        return ((pred - target) ** 2).mean(dim=1)
    elif loss_type == "smooth_l1":
        return nn.functional.smooth_l1_loss(
            pred, target, reduction="none"
        ).mean(dim=1)

def train_one_epoch(model, loader, optimizer, obs_mean, obs_std, config):
    for batch_obs, batch_act, batch_w in loader:
        pred_act = model(batch_obs)
        sample_loss = per_sample_loss(pred_act, batch_act, config.loss_type)
        loss = (sample_loss * batch_w).mean()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
```

### 代码解释

下面按代码出现顺序逐行解释。

- `def build_sample_weights(rewards, terminals, config):` 定义样本权重构造函数，输入奖励、终止标记和配置项。
- `for i in range(n):` 遍历全部样本索引，用来根据 `terminals` 找出每条 episode 的边界。
- `if terminals_np[i] > 0.5:` 判断第 `i` 个样本是否为 episode 终止点。终止标记接近 1 时表示一条轨迹结束。
- `episode_ranges.append((start, i))` 把当前 episode 的起止索引保存下来，范围是从 `start` 到 `i`。
- `start = i + 1` 下一条 episode 从当前终止点的下一个样本开始。
- `for ep_id, (l, r) in enumerate(episode_ranges):` 遍历每条 episode，`l` 是起始索引，`r` 是结束索引。
- `episode_returns[ep_id] = rewards_np[l:r + 1].sum()` 计算该 episode 的累计奖励，对应公式 $G_e=\sum r_t$。
- `norm_returns = (episode_returns - ret_mean) / ret_std` 对 episode return 做标准化，对应 $\hat{G}_e$。
- `episode_weights = np.exp(norm_returns / config.weight_temperature)` 把标准化回报转成指数权重，对应 $w_e=\exp(\hat{G}_e/T)$。
- `episode_weights = np.clip(...)` 对权重做上下界裁剪，避免极高回报轨迹权重过大。
- `episode_weights,` 表示被裁剪的是刚刚计算出的 episode 权重。
- `config.min_weight,` 是权重下界，保证低回报轨迹不会完全失效。
- `config.max_weight` 是权重上界，防止高回报轨迹独占梯度。
- `)` 结束 `np.clip` 调用。
- `for ep_id, (l, r) in enumerate(episode_ranges):` 再次遍历每条 episode，把 episode 权重分配给其中所有样本。
- `sample_weights[l:r + 1] = episode_weights[ep_id]` 将同一条 episode 内的所有样本赋予同一个权重。
- `sample_weights = sample_weights / (sample_weights.mean() + 1e-6)` 将样本权重归一化到均值约为 1，避免整体 loss 尺度变化太大。
- `return torch.tensor(sample_weights, dtype=torch.float32)` 把 NumPy 权重转成 PyTorch 张量，供 DataLoader 和训练过程使用。
- `# ------------------------------------------------------------` 是文档里的分隔注释，用来说明下面进入同一算法的另一段核心代码。
- `# 继续：同一算法的另一段核心代码。` 表示接下来展示的是加权损失如何真正参与训练。
- `# ------------------------------------------------------------` 结束分隔注释。
- `def per_sample_loss(pred, target, loss_type="smooth_l1"):` 定义逐样本损失函数，输入预测动作、真实动作和损失类型。
- `if loss_type == "mse":` 判断是否使用 MSE 损失。
- `return ((pred - target) ** 2).mean(dim=1)` 对每个样本的动作维度求均方误差，返回形状为 batch 大小的逐样本 loss。
- `elif loss_type == "smooth_l1":` 判断是否使用 Smooth L1 损失。
- `return nn.functional.smooth_l1_loss(...)` 使用 Smooth L1 计算误差，它比 MSE 对异常值更不敏感。
- `pred, target, reduction="none"` 表示先保留每个动作维度的损失，不直接求全局平均。
- `).mean(dim=1)` 对动作维度求平均，得到每个样本自己的损失。
- `def train_one_epoch(...)` 定义 BC_NB 的单轮训练函数。
- `for batch_obs, batch_act, batch_w in loader:` 每个 batch 除了状态和动作，还额外取出样本权重 `batch_w`。
- `pred_act = model(batch_obs)` 用策略网络预测动作，对应 $\pi_\theta(s)$。
- `sample_loss = per_sample_loss(pred_act, batch_act, config.loss_type)` 计算每个样本自己的动作拟合误差。
- `loss = (sample_loss * batch_w).mean()` 将逐样本 loss 乘以权重后再平均，对应 $L_{RWBC}$。
- `optimizer.zero_grad()` 清空旧梯度。
- `loss.backward()` 对加权 BC 损失反向传播。
- `nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)` 裁剪策略网络梯度，提升训练稳定性。
- `optimizer.step()` 更新策略网络参数。

### 与基础版差异

BC 的损失是普通平均，BC_NB 的损失是加权平均。BC_NB 多用了 `rewards` 和 `terminals`，但它仍然不是 actor-critic 方法，也没有 TD target。

---

## 5. CQL.py：保守 Q 学习

### 算法直觉

CQL（Conservative Q-Learning）用于离线强化学习。离线数据不能主动探索，如果 critic 把没见过的动作估得过高，actor 会被错误的高 Q 值吸引到数据分布外。CQL 的核心是压低候选动作的 Q，同时保留数据动作的 Q。

### 核心公式

TD 目标：

$$
y = r + \gamma(1-d)\min(Q_{\bar{\phi}_1}(s', \pi(s')), Q_{\bar{\phi}_2}(s', \pi(s')))
$$

critic MSE：

$$
L_Q = \mathrm{MSE}(Q_{\phi_1}(s,a), y) + \mathrm{MSE}(Q_{\phi_2}(s,a), y)
$$

CQL 保守项：

$$
L_{CQL} = \log\sum_{\tilde{a}}\exp(Q(s,\tilde{a})) - Q(s,a_{data})
$$

总 critic 损失：

$$
L_{critic} = L_Q + \lambda_{cql}(L_{CQL}^{(1)} + L_{CQL}^{(2)})
$$

actor 损失：

$$
L_\pi = -\lambda Q_{\phi_1}(s,\pi_\theta(s)) + \beta\mathrm{MSE}(\pi_\theta(s), a_{data})
$$

### 公式解释

- $y$：Bellman TD 目标。
- $\min(Q_1,Q_2)$：双 Q 取小值，降低 Q 过估计。
- $L_{CQL}$：让随机动作、策略动作等候选动作的 Q 不要整体高于数据动作 Q。
- $\lambda_{cql}$：保守项权重，越大越保守。
- actor loss 中的第一项鼓励 actor 输出高 Q 动作，第二项 BC 正则防止 actor 偏离离线动作太远。

### 对应代码

来源：`AIcodes/CQL.py`

```python
def compute_cql_penalty(critic, actor, s, a_data, num_random_actions=10):
    random_actions = torch.empty(
        batch_size, num_random_actions, action_dim, device=s.device
    ).uniform_(-1.0, 1.0)

    q_rand = critic(
        torch.cat([s_repeat, random_actions], dim=-1)
        .view(batch_size * num_random_actions, -1)
    ).view(batch_size, num_random_actions)

    pi_action = actor(s)
    q_pi = critic(torch.cat([s, pi_action], dim=1))
    q_data = critic(torch.cat([s, a_data], dim=1))

    cat_q = torch.cat([q_rand, q_pi], dim=1)
    penalty = torch.logsumexp(cat_q, dim=1, keepdim=True).mean() - q_data.mean()
    return penalty

# ------------------------------------------------------------
# 继续：同一算法的另一段核心代码。
# ------------------------------------------------------------

with torch.no_grad():
    next_a = actor(s_next)
    q_target = r.unsqueeze(1) + args.gamma * (1.0 - done.unsqueeze(1)) * torch.min(
        target_critic1(torch.cat([s_next, next_a], dim=1)),
        target_critic2(torch.cat([s_next, next_a], dim=1))
    )

q1_pred = critic1(torch.cat([s, a], dim=1))
q2_pred = critic2(torch.cat([s, a], dim=1))
mse_q1 = nn.functional.mse_loss(q1_pred, q_target)
mse_q2 = nn.functional.mse_loss(q2_pred, q_target)
critic_mse = mse_q1 + mse_q2

cql1 = compute_cql_penalty(critic1, actor, s, a, args.num_random_actions)
cql2 = compute_cql_penalty(critic2, actor, s, a, args.num_random_actions)
cql_penalty = args.lambda_cql * (cql1 + cql2)
total_critic_loss = critic_mse + cql_penalty

pred_a = actor(s)
q_pi = critic1(torch.cat([s, pred_a], dim=1))
lam = args.alpha_bc / (q_pi.abs().mean().detach() + 1e-6)
bc_loss = nn.functional.mse_loss(pred_a, a)
actor_loss = -lam * q_pi.mean() + args.bc_coef * bc_loss
```

### 代码解释

下面按代码出现顺序逐行解释。

- `def compute_cql_penalty(...)` 定义 CQL 保守惩罚函数，输入 critic、actor、当前状态、数据动作和随机动作数量。
- `random_actions = torch.empty(...)` 创建随机动作张量，用来模拟数据分布外动作。
- `batch_size, num_random_actions, action_dim, device=s.device` 指定随机动作形状和设备，保证它和当前 batch 在同一 GPU/CPU 上。
- `).uniform_(-1.0, 1.0)` 将随机动作采样到 $[-1,1]$，匹配动作空间范围。
- `q_rand = critic(...)` 计算随机动作对应的 Q 值。
- `torch.cat([s_repeat, random_actions], dim=-1)` 把状态和随机动作拼接成 critic 输入。
- `.view(batch_size * num_random_actions, -1)` 将三维张量展平成 critic 可处理的二维 batch。
- `).view(batch_size, num_random_actions)` 再把输出恢复为“每个状态对应多个随机动作 Q 值”的形状。
- `pi_action = actor(s)` 用当前 actor 生成策略动作 $\pi(s)$。
- `q_pi = critic(torch.cat([s, pi_action], dim=1))` 计算策略动作的 Q 值。
- `q_data = critic(torch.cat([s, a_data], dim=1))` 计算离线数据真实动作的 Q 值。
- `cat_q = torch.cat([q_rand, q_pi], dim=1)` 汇总随机动作和策略动作的 Q 值，作为需要保守压低的候选动作。
- `penalty = torch.logsumexp(...) - q_data.mean()` 计算 CQL penalty，即让候选动作整体 Q 不要高于数据动作 Q。
- `return penalty` 返回保守惩罚项。
- `# ------------------------------------------------------------` 是文档分隔注释。
- `# 继续：同一算法的另一段核心代码。` 表示下面进入训练中 TD 目标、critic loss 和 actor loss 的计算。
- `# ------------------------------------------------------------` 结束分隔注释。
- `with torch.no_grad():` 表示下面构造 TD 目标时不计算梯度，target 只作为训练参考答案。
- `next_a = actor(s_next)` 用 actor 在下一状态 $s'$ 上生成下一动作 $a'$。
- `q_target = r.unsqueeze(1) + ...` 构造 TD 目标 $y$。
- `args.gamma * (1.0 - done.unsqueeze(1))` 表示未来价值折扣，终止样本会把未来价值乘成 0。
- `torch.min(...)` 对两个 target critic 的 Q 值取较小值，降低过估计。
- `target_critic1(torch.cat([s_next, next_a], dim=1))` 计算第一个 target critic 的下一状态价值。
- `target_critic2(torch.cat([s_next, next_a], dim=1))` 计算第二个 target critic 的下一状态价值。
- `q1_pred = critic1(torch.cat([s, a], dim=1))` 用第一个 critic 估计当前数据动作 Q 值。
- `q2_pred = critic2(torch.cat([s, a], dim=1))` 用第二个 critic 估计当前数据动作 Q 值。
- `mse_q1 = nn.functional.mse_loss(q1_pred, q_target)` 计算第一个 critic 与 TD 目标的 MSE。
- `mse_q2 = nn.functional.mse_loss(q2_pred, q_target)` 计算第二个 critic 与 TD 目标的 MSE。
- `critic_mse = mse_q1 + mse_q2` 汇总两个 critic 的 Bellman 误差。
- `cql1 = compute_cql_penalty(...)` 计算第一个 critic 的保守惩罚。
- `cql2 = compute_cql_penalty(...)` 计算第二个 critic 的保守惩罚。
- `cql_penalty = args.lambda_cql * (cql1 + cql2)` 用 `lambda_cql` 调整保守惩罚强度。
- `total_critic_loss = critic_mse + cql_penalty` 得到 critic 总损失，对应 $L_{critic}$。
- `pred_a = actor(s)` 用 actor 在当前状态下预测动作。
- `q_pi = critic1(torch.cat([s, pred_a], dim=1))` 评估 actor 预测动作的 Q 值。
- `lam = args.alpha_bc / (q_pi.abs().mean().detach() + 1e-6)` 根据 Q 值尺度自适应计算 Q 项权重。
- `bc_loss = nn.functional.mse_loss(pred_a, a)` 计算 actor 动作和数据动作之间的 BC 正则。
- `actor_loss = -lam * q_pi.mean() + args.bc_coef * bc_loss` 组合 Q 最大化项和 BC 约束项，得到 actor 损失。

### 与改进版差异

CQL 使用当前策略动作和随机动作构造保守项。CQL_NB 会加入带噪声的下一状态策略动作，并在 TD target 中使用目标动作平滑。

---

## 6. CQL_NB.py：带目标动作平滑的 CQL

### 算法直觉

CQL_NB 可以理解成 CQL 加上 TD3 风格的 target policy smoothing。它不仅保守估计 Q，还让下一状态动作附近的小扰动也进入训练目标，从而让 critic 对动作变化更平滑。

### 核心公式

目标动作平滑：

$$
\epsilon \sim \mathcal{N}(0,\sigma^2), \quad
\epsilon = \mathrm{clip}(\epsilon, -c, c)
$$

$$
a' = \mathrm{clip}(\pi_\theta(s') + \epsilon, -1, 1)
$$

TD 目标：

$$
y = r + \gamma(1-d)\min(Q_{\bar{\phi}_1}(s',a'), Q_{\bar{\phi}_2}(s',a'))
$$

CQL_NB 保守项：

$$
L_{CQL\_NB} =
\log\sum \exp(Q(s,a_{rand}), Q(s,\pi(s)), Q(s',a')) - Q(s,a_{data})
$$

actor 损失仍为：

$$
L_\pi = -\lambda Q_{\phi_1}(s,\pi_\theta(s)) + \beta\mathrm{MSE}(\pi_\theta(s), a_{data})
$$

### 公式解释

- $\epsilon$：目标动作噪声。
- $c$：噪声裁剪阈值，对应 `noise_clip`。
- $a'$：加噪并裁剪后的下一状态动作。
- 加噪可以避免 Q 网络只在单点动作上学习，使目标值更平滑。
- CQL_NB 的保守项比 CQL 多考虑 `s_next` 上的噪声策略动作。

### 对应代码

来源：`AIcodes/CQL_NB.py`

```python
def compute_cql_penalty(
    critic, actor, s, s_next, a_data,
    num_random_actions=10, policy_noise=0.2, noise_clip=0.5
):
    random_actions = torch.empty(
        batch_size, num_random_actions, action_dim, device=s.device
    ).uniform_(-1.0, 1.0)

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

# ------------------------------------------------------------
# 继续：同一算法的另一段核心代码。
# ------------------------------------------------------------

with torch.no_grad():
    noise = torch.randn_like(a) * args.policy_noise
    noise = noise.clamp(-args.noise_clip, args.noise_clip)
    next_a = (actor(s_next) + noise).clamp(-1.0, 1.0)

    q_target = r.unsqueeze(1) + args.gamma * (1.0 - done.unsqueeze(1)) * torch.min(
        target_critic1(torch.cat([s_next, next_a], dim=1)),
        target_critic2(torch.cat([s_next, next_a], dim=1))
    )

cql_penalty = args.lambda_cql * (cql1 + cql2)
total_critic_loss = critic_mse + cql_penalty

pred_a = actor(s)
q_pi = critic1(torch.cat([s, pred_a], dim=1))
lam = args.alpha_bc / (q_pi.abs().mean().detach() + 1e-6)
bc_loss = nn.functional.mse_loss(pred_a, a)
actor_loss = -lam * q_pi.mean() + args.bc_coef * bc_loss
```

### 代码解释

下面按代码出现顺序逐行解释。

- `def compute_cql_penalty(...)` 定义 CQL_NB 的保守惩罚函数，比 CQL 多接收 `s_next`、`policy_noise` 和 `noise_clip`。
- `random_actions = torch.empty(...)` 创建随机动作，用来检测分布外动作的 Q 是否被高估。
- `batch_size, num_random_actions, action_dim, device=s.device` 指定随机动作形状和设备。
- `).uniform_(-1.0, 1.0)` 在动作空间范围内均匀采样随机动作。
- `pi_action = actor(s)` 生成当前状态下的策略动作。
- `q_pi = critic(torch.cat([s, pi_action], dim=1))` 计算当前策略动作的 Q 值。
- `noise = torch.randn_like(a_data) * policy_noise` 生成和动作同形状的高斯噪声，对应 $\epsilon$。
- `noise = noise.clamp(-noise_clip, noise_clip)` 裁剪噪声，避免扰动过大。
- `next_pi_action = (actor(s_next) + noise).clamp(-1.0, 1.0)` 得到带噪声的下一状态策略动作，对应平滑目标动作 $a'$。
- `q_next_pi = critic(torch.cat([s_next, next_pi_action], dim=1))` 计算下一状态扰动动作的 Q 值。
- `q_data = critic(torch.cat([s, a_data], dim=1))` 计算数据动作的 Q 值。
- `cat_q = torch.cat([q_rand, q_pi, q_next_pi], dim=1)` 汇总随机动作、当前策略动作和下一状态扰动动作的 Q 值。
- `penalty = torch.logsumexp(...) - q_data.mean()` 计算增强版 CQL penalty，让更多候选动作都受到保守约束。
- `return penalty` 返回保守惩罚项。
- `# ------------------------------------------------------------` 是文档分隔注释。
- `# 继续：同一算法的另一段核心代码。` 表示下面展示训练中的目标动作平滑和损失计算。
- `# ------------------------------------------------------------` 结束分隔注释。
- `with torch.no_grad():` 构造 TD 目标时关闭梯度。
- `noise = torch.randn_like(a) * args.policy_noise` 生成 target action 噪声。
- `noise = noise.clamp(-args.noise_clip, args.noise_clip)` 将噪声限制在允许范围内。
- `next_a = (actor(s_next) + noise).clamp(-1.0, 1.0)` 在下一状态动作上加入噪声并裁剪到合法动作范围。
- `q_target = r.unsqueeze(1) + ...` 构造 CQL_NB 的 TD 目标。
- `args.gamma * (1.0 - done.unsqueeze(1))` 表示只有非终止样本才保留未来价值。
- `torch.min(...)` 对两个 target critic 的估计取较小值。
- `target_critic1(...)` 计算第一个 target critic 的目标 Q。
- `target_critic2(...)` 计算第二个 target critic 的目标 Q。
- `cql_penalty = args.lambda_cql * (cql1 + cql2)` 将两个 critic 的 CQL penalty 加权汇总。
- `total_critic_loss = critic_mse + cql_penalty` 得到 critic 总损失。
- `pred_a = actor(s)` 生成当前状态动作。
- `q_pi = critic1(torch.cat([s, pred_a], dim=1))` 用 critic 评估 actor 动作。
- `lam = args.alpha_bc / (q_pi.abs().mean().detach() + 1e-6)` 根据 Q 值尺度计算 Q 项缩放系数。
- `bc_loss = nn.functional.mse_loss(pred_a, a)` 计算策略动作和数据动作的 BC 距离。
- `actor_loss = -lam * q_pi.mean() + args.bc_coef * bc_loss` 得到 actor 总损失。

### 与基础版差异

CQL_NB 比 CQL 多了 `policy_noise` 和 `noise_clip`。它更接近 CQL + TD3 稳定技巧，重点是减少目标动作过尖导致的 Q 波动。

---

## 7. DDPG.py：优势加权 DDPG

### 算法直觉

项目中的 `DDPG.py` 不是最原始的在线 DDPG，而是离线 Advantage-Weighted DDPG。critic 仍然用 TD 目标学习 Q，actor 不直接最大化 Q，而是根据优势给 BC 损失加权：如果数据动作比当前 actor 动作的 Q 更高，就更值得模仿。

### 核心公式

目标动作平滑：

$$
a' = \mathrm{clip}(\pi_{\bar{\theta}}(s')+\epsilon,-1,1)
$$

TD 目标：

$$
y = r + \gamma(1-d)\min(Q_{\bar{\phi}_1}(s',a'), Q_{\bar{\phi}_2}(s',a'))
$$

critic 损失：

$$
L_Q = \mathrm{MSE}(Q_{\phi_1}(s,a),y)+\mathrm{MSE}(Q_{\phi_2}(s,a),y)
$$

优势：

$$
A(s,a)=\min(Q_1(s,a),Q_2(s,a))-\min(Q_1(s,\pi(s)),Q_2(s,\pi(s)))
$$

优势权重：

$$
w=\mathrm{clip}\left(\exp\left(\frac{A(s,a)}{\lambda_{aw}}\right),0,w_{\max}\right)
$$

actor 损失：

$$
L_\pi=\mathbb{E}\left[w\cdot\|\pi_\theta(s)-a\|_2^2\right]
$$

### 公式解释

- $A(s,a)$：数据动作相对当前策略动作的优势。
- 如果 $Q(s,a_{data}) > Q(s,\pi(s))$，说明数据动作更好，权重大。
- 如果数据动作并不比当前策略好，权重较小。
- 这种设计适合离线场景，因为 actor 仍然被限制在数据动作附近，不会轻易跑出数据分布。

### 对应代码

来源：`AIcodes/DDPG.py`

```python
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

# ------------------------------------------------------------
# 继续：同一算法的另一段核心代码。
# ------------------------------------------------------------

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
```

### 代码解释

下面按代码出现顺序逐行解释。

- `with torch.no_grad():` 构造 target 时关闭梯度，避免 target 网络参与反向传播。
- `noise = torch.randn_like(actions) * self.policy_noise` 生成目标动作平滑噪声。
- `noise = noise.clamp(-self.noise_clip, self.noise_clip)` 限制噪声幅度，避免目标动作偏离太远。
- `next_actions = (self.actor_target(next_obs) + noise).clamp(-1.0, 1.0)` 用 target actor 生成下一动作并加入噪声，再裁剪到动作范围。
- `q1_target_next = self.critic1_target(next_obs, next_actions)` 用第一个 target critic 估计下一状态动作价值。
- `q2_target_next = self.critic2_target(next_obs, next_actions)` 用第二个 target critic 估计下一状态动作价值。
- `q_target_next = torch.min(q1_target_next, q2_target_next)` 取双 Q 较小值，降低过估计。
- `td_target = rewards + self.gamma * (1 - terminals) * q_target_next` 构造 TD 目标，终止样本不保留未来价值。
- `q1_val = self.critic1(obs, actions)` 用第一个 critic 估计当前数据动作 Q。
- `q2_val = self.critic2(obs, actions)` 用第二个 critic 估计当前数据动作 Q。
- `critic1_loss = nn.functional.mse_loss(q1_val, td_target)` 计算第一个 critic 的 Bellman 误差。
- `critic2_loss = nn.functional.mse_loss(q2_val, td_target)` 计算第二个 critic 的 Bellman 误差。
- `critic_loss = critic1_loss + critic2_loss` 汇总 critic 损失。
- `# ------------------------------------------------------------` 是文档分隔注释。
- `# 继续：同一算法的另一段核心代码。` 表示下面展示 actor 的优势加权更新。
- `# ------------------------------------------------------------` 结束分隔注释。
- `if self.total_it % self.policy_delay == 0:` 判断是否到了 actor 更新步。DDPG/TD3 类算法通常延迟更新 actor。
- `with torch.no_grad():` 计算优势权重时不需要对 critic 求梯度。
- `q_data = torch.min(...)` 计算数据动作的双 Q 较小值。
- `self.critic1(obs, actions)` 计算第一个 critic 对数据动作的评价。
- `self.critic2(obs, actions)` 计算第二个 critic 对数据动作的评价。
- `pi_actions = self.actor(obs)` 用当前 actor 生成策略动作。
- `q_pi = torch.min(...)` 计算策略动作的双 Q 较小值。
- `self.critic1(obs, pi_actions)` 计算第一个 critic 对策略动作的评价。
- `self.critic2(obs, pi_actions)` 计算第二个 critic 对策略动作的评价。
- `advantage = q_data - q_pi` 计算数据动作相对策略动作的优势 $A(s,a)$。
- `weights = torch.exp(advantage / self.aw_lambda)` 将优势转成指数权重。
- `weights = torch.clamp(weights, max=self.weight_clip)` 裁剪权重，避免极端 advantage 产生过大梯度。
- `pred_actions = self.actor(obs)` 重新计算 actor 动作，这一次需要保留梯度用于更新 actor。
- `per_sample_bc = ((pred_actions - actions) ** 2).mean(dim=1, keepdim=True)` 计算每个样本的 BC 动作误差。
- `actor_loss = (weights * per_sample_bc).mean()` 用优势权重加权 BC loss，高优势样本对 actor 影响更大。

### 与改进版差异

DDPG.py 的 actor 更新是优势加权 BC。DDPG_NB.py 则显式使用 `Q 最大化 + BC 正则`，不再用优势权重。

---

## 8. DDPG_NB.py：DDPG/TD3-lite + BC 正则

### 算法直觉

DDPG_NB 的 actor 一边追求更高 Q，一边保持接近数据动作。Q 项负责提升性能，BC 项负责防止策略偏离离线数据分布。

### 核心公式

critic 目标与 DDPG.py 类似：

$$
y = r + \gamma(1-d)\min(Q_{\bar{\phi}_1}(s',a'), Q_{\bar{\phi}_2}(s',a'))
$$

actor 的 Q 项：

$$
L_Q^\pi = -\lambda \mathbb{E}[Q_{\phi_1}(s,\pi_\theta(s))]
$$

BC 项：

$$
L_{BC}^\pi = \beta\mathrm{MSE}(\pi_\theta(s),a_{data})
$$

actor 总损失：

$$
L_\pi = L_Q^\pi + L_{BC}^\pi
$$

其中：

$$
\lambda = \frac{\alpha_{bc}}{\mathbb{E}[|Q_{\phi_1}(s,\pi(s))|]+\epsilon}
$$

### 公式解释

- $L_Q^\pi$：最大化 actor 动作的 Q 值。因为优化器默认最小化 loss，所以写成负号。
- $L_{BC}^\pi$：约束 actor 动作不要离数据动作太远。
- $\lambda$：根据 Q 值尺度自动调整 Q 项权重，防止 Q 项数值过大。
- $\beta$：`bc_coef`，控制 BC 约束强度。

### 对应代码

来源：`AIcodes/DDPG_NB.py`

```python
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

# ------------------------------------------------------------
# 继续：同一算法的另一段核心代码。
# ------------------------------------------------------------

if self.total_it % self.policy_delay == 0:
    pred_actions = self.actor(obs)
    q_pi = self.critic1(obs, pred_actions)

    lam = self.alpha_bc / (q_pi.abs().mean().detach() + 1e-6)
    q_loss = -lam * q_pi.mean()
    bc_loss = self.bc_coef * nn.functional.mse_loss(pred_actions, actions)
    actor_loss = q_loss + bc_loss

    self.actor_optimizer.zero_grad()
    actor_loss.backward()
    nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
    self.actor_optimizer.step()
```

### 代码解释

下面按代码出现顺序逐行解释。

- `with torch.no_grad():` 构造 TD 目标时不记录梯度。
- `noise = torch.randn_like(actions) * self.policy_noise` 生成目标动作扰动。
- `noise = noise.clamp(-self.noise_clip, self.noise_clip)` 对扰动做裁剪。
- `next_actions = (self.actor_target(next_obs) + noise).clamp(-1.0, 1.0)` target actor 输出下一动作，加噪并限制动作范围。
- `q1_target_next = self.critic1_target(next_obs, next_actions)` 第一个 target critic 估计下一 Q。
- `q2_target_next = self.critic2_target(next_obs, next_actions)` 第二个 target critic 估计下一 Q。
- `q_target_next = torch.min(q1_target_next, q2_target_next)` 取两个 target Q 的较小值。
- `td_target = rewards + self.gamma * (1 - terminals) * q_target_next` 得到 critic 学习的 TD 目标。
- `q1_val = self.critic1(obs, actions)` 第一个 critic 预测当前数据动作 Q。
- `q2_val = self.critic2(obs, actions)` 第二个 critic 预测当前数据动作 Q。
- `critic1_loss = nn.functional.mse_loss(q1_val, td_target)` 第一个 critic 的 MSE loss。
- `critic2_loss = nn.functional.mse_loss(q2_val, td_target)` 第二个 critic 的 MSE loss。
- `critic_loss = critic1_loss + critic2_loss` 汇总 critic 损失。
- `# ------------------------------------------------------------` 是文档分隔注释。
- `# 继续：同一算法的另一段核心代码。` 表示下面进入 actor 的 Q+BC 更新。
- `# ------------------------------------------------------------` 结束分隔注释。
- `if self.total_it % self.policy_delay == 0:` 只有达到延迟更新间隔时才更新 actor。
- `pred_actions = self.actor(obs)` 生成当前状态下的 actor 动作。
- `q_pi = self.critic1(obs, pred_actions)` 用第一个 critic 评价 actor 动作。
- `lam = self.alpha_bc / (q_pi.abs().mean().detach() + 1e-6)` 根据 Q 值平均绝对值计算归一化系数。
- `q_loss = -lam * q_pi.mean()` 构造 Q 最大化项。负号表示最小化该 loss 等价于最大化 Q。
- `bc_loss = self.bc_coef * nn.functional.mse_loss(pred_actions, actions)` 构造 BC 正则项，约束 actor 动作接近数据动作。
- `actor_loss = q_loss + bc_loss` 合并 Q 项和 BC 项。
- `self.actor_optimizer.zero_grad()` 清空 actor 旧梯度。
- `actor_loss.backward()` 对 actor loss 反向传播。
- `nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)` 裁剪 actor 梯度。
- `self.actor_optimizer.step()` 更新 actor 参数。

### 与基础版差异

DDPG.py 是优势加权 BC；DDPG_NB.py 是 Q 最大化加 BC 正则。前者更强调“哪些数据动作值得模仿”，后者更强调“提升 Q 的同时不要离数据太远”。

---

## 9. SAC.py：最大熵 critic + AWAC actor

### 算法直觉

SAC（Soft Actor-Critic）使用最大熵思想，让策略既追求高 Q，也保持一定随机性。项目中的 `SAC.py` 在 critic 上保留 SAC 的熵正则 TD 目标，但 actor 更新采用 AWAC 风格：提高高优势数据动作在当前策略下的概率。

### 核心公式

SAC critic 目标：

$$
y = r + \gamma(1-d)\left[\min(Q_{\bar{\phi}_1}(s',a'),Q_{\bar{\phi}_2}(s',a')) - \alpha\log\pi(a'|s')\right]
$$

critic 损失：

$$
L_Q = \mathrm{MSE}(Q_{\phi_1}(s,a),y)+\mathrm{MSE}(Q_{\phi_2}(s,a),y)
$$

AWAC 优势：

$$
A(s,a)=Q(s,a_{data})-V_\pi(s)
$$

其中：

$$
V_\pi(s)\approx Q(s,\pi(s))
$$

优势权重：

$$
w=\mathrm{clip}\left(\exp\left(\frac{A(s,a)}{\lambda_{awac}}\right),0,w_{\max}\right)
$$

actor 损失：

$$
L_\pi = -\mathbb{E}[w\log\pi_\theta(a_{data}|s)]
$$

温度参数损失：

$$
L_\alpha = -\log\alpha\cdot(\log\pi(a|s)+\mathcal{H}_{target})
$$

### 公式解释

- `- alpha * log_prob` 是最大熵项，鼓励策略保留随机性。
- $A(s,a)$ 衡量数据动作是否比当前策略动作更有价值。
- $w\log\pi(a_{data}|s)$ 表示高优势数据动作应该在当前策略中获得更高概率。
- 由于优化器最小化 loss，所以 actor loss 前面有负号。

### 对应代码

来源：`AIcodes/SAC.py`

```python
with torch.no_grad():
    next_action, next_log_prob, _ = self.actor.sample(next_state)
    next_log_prob = torch.clamp(
        next_log_prob,
        min=-self.max_logprob_abs,
        max=self.max_logprob_abs
    )

    target_q1 = self.target_critic1(next_state, next_action)
    target_q2 = self.target_critic2(next_state, next_action)
    target_q = torch.min(target_q1, target_q2) - self.alpha.detach() * next_log_prob
    target_q = reward + self.cfg.gamma * (1 - done) * target_q

# ------------------------------------------------------------
# 继续：同一算法的另一段核心代码。
# ------------------------------------------------------------

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
log_prob_data = torch.clamp(
    log_prob_data,
    min=-self.max_logprob_abs,
    max=self.max_logprob_abs
)

actor_loss = -(weights * log_prob_data).mean()

# ------------------------------------------------------------
# 继续：同一算法的另一段核心代码。
# ------------------------------------------------------------

if self.cfg.learn_alpha:
    _, log_prob, _ = self.actor.sample(state)
    alpha_loss = -(
        self.log_alpha * (log_prob + self.target_entropy).detach()
    ).mean()

    self.alpha_optimizer.zero_grad()
    alpha_loss.backward()
    self.alpha_optimizer.step()
```

### 代码解释

下面按代码出现顺序逐行解释。

- `with torch.no_grad():` 构造 critic 的目标值时关闭梯度，不让 target 计算参与参数更新。
- `next_action, next_log_prob, _ = self.actor.sample(next_state)` 在下一状态采样动作，并得到该动作的对数概率。
- `next_log_prob = torch.clamp(...)` 对 log probability 做裁剪，防止极端概率导致数值不稳定。
- `next_log_prob,` 表示被裁剪的是下一动作的 log probability。
- `min=-self.max_logprob_abs,` 设置裁剪下界。
- `max=self.max_logprob_abs` 设置裁剪上界。
- `)` 结束 `torch.clamp` 调用。
- `target_q1 = self.target_critic1(next_state, next_action)` 第一个 target critic 估计下一状态动作价值。
- `target_q2 = self.target_critic2(next_state, next_action)` 第二个 target critic 估计下一状态动作价值。
- `target_q = torch.min(target_q1, target_q2) - self.alpha.detach() * next_log_prob` 构造最大熵目标中的软 Q 值，双 Q 取小后减去熵温度项。
- `target_q = reward + self.cfg.gamma * (1 - done) * target_q` 加上当前奖励并折扣未来价值，得到 SAC critic 的 TD 目标。
- `# ------------------------------------------------------------` 是文档分隔注释。
- `# 继续：同一算法的另一段核心代码。` 表示下面展示 AWAC 风格 actor 更新。
- `# ------------------------------------------------------------` 结束分隔注释。
- `with torch.no_grad():` 计算优势和权重时关闭梯度，因为权重只作为 actor loss 的系数。
- `data_q1 = self.critic1(state, action)` 第一个 critic 评估数据动作。
- `data_q2 = self.critic2(state, action)` 第二个 critic 评估数据动作。
- `data_q = torch.min(data_q1, data_q2)` 取数据动作双 Q 的较小值，作为保守的数据动作价值。
- `pi_action, _, _ = self.actor.sample(state)` 从当前策略采样动作。
- `pi_q1 = self.critic1(state, pi_action)` 第一个 critic 评估策略采样动作。
- `pi_q2 = self.critic2(state, pi_action)` 第二个 critic 评估策略采样动作。
- `v_pi = torch.min(pi_q1, pi_q2)` 用策略动作 Q 近似当前状态价值 $V_\pi(s)$。
- `advantage, weights = self._compute_awac_weights(data_q, v_pi)` 根据 $Q(s,a_{data})-V_\pi(s)$ 计算优势和 AWAC 权重。
- `log_prob_data = self.actor.log_prob_of_action(state, action)` 计算数据动作在当前策略分布下的 log probability。
- `log_prob_data = torch.clamp(...)` 对数据动作 log probability 做裁剪。
- `log_prob_data,` 表示被裁剪的是数据动作的 log probability。
- `min=-self.max_logprob_abs,` 设置下界。
- `max=self.max_logprob_abs` 设置上界。
- `)` 结束裁剪。
- `actor_loss = -(weights * log_prob_data).mean()` 构造 AWAC actor loss，高优势数据动作的 log probability 会被更强地提高。
- `# ------------------------------------------------------------` 是文档分隔注释。
- `# 继续：同一算法的另一段核心代码。` 表示下面展示 SAC 的温度参数更新。
- `# ------------------------------------------------------------` 结束分隔注释。
- `if self.cfg.learn_alpha:` 判断是否自动学习熵温度系数。
- `_, log_prob, _ = self.actor.sample(state)` 从当前状态采样动作并取得 log probability。
- `alpha_loss = -(...).mean()` 构造温度参数损失，使策略熵接近目标熵。
- `self.log_alpha * (log_prob + self.target_entropy).detach()` 表示只更新 `log_alpha`，不让该损失反向影响 actor。
- `self.alpha_optimizer.zero_grad()` 清空温度参数优化器的旧梯度。
- `alpha_loss.backward()` 对温度损失反向传播。
- `self.alpha_optimizer.step()` 更新 `log_alpha`，从而间接更新 $\alpha$。

### 与改进版差异

SAC.py 的 actor 是 AWAC 风格，直接提高数据动作概率。SAC_NB.py 使用更标准的 SAC actor loss，再额外加 BC 正则。

---

## 10. SAC_NB.py：SAC + BC 正则

### 算法直觉

SAC_NB 保留标准 SAC actor loss，同时加入 BC 正则。这样 actor 既能最大化软 Q 目标，又不会在离线训练中跑到数据分布外。

### 核心公式

critic 目标：

$$
y = r + \gamma(1-d)\left[\min(Q_{\bar{\phi}_1}(s',a'),Q_{\bar{\phi}_2}(s',a')) - \alpha\log\pi(a'|s')\right]
$$

标准 SAC actor loss：

$$
L_{SAC} = \mathbb{E}[\alpha\log\pi_\theta(a_\theta|s)-\min(Q_1(s,a_\theta),Q_2(s,a_\theta))]
$$

BC 正则：

$$
L_{BC} = \mathrm{MSE}(a_\theta,a_{data})
$$

actor 总损失：

$$
L_\pi = L_{SAC} + \beta L_{BC}
$$

### 公式解释

- $a_\theta$：从当前 Gaussian actor 中采样的动作。
- $\alpha\log\pi$：熵项，鼓励策略随机性。
- $-Q$：最大化 Q。因为最小化 loss，所以写成负 Q。
- $L_{BC}$：把采样动作拉向离线数据动作。
- $\beta$：`bc_coef`，控制 BC 正则强度。

### 对应代码

来源：`AIcodes/SAC_NB.py`

```python
with torch.no_grad():
    next_action, next_log_prob, _ = self.actor.sample(next_state)
    target_q1 = self.target_critic1(next_state, next_action)
    target_q2 = self.target_critic2(next_state, next_action)
    target_q = torch.min(target_q1, target_q2) - self.alpha.detach() * next_log_prob
    target_q = reward + self.cfg.gamma * (1 - done) * target_q

current_q1 = self.critic1(state, action)
current_q2 = self.critic2(state, action)
critic1_loss = nn.functional.mse_loss(current_q1, target_q)
critic2_loss = nn.functional.mse_loss(current_q2, target_q)

# ------------------------------------------------------------
# 继续：同一算法的另一段核心代码。
# ------------------------------------------------------------

sampled_action, log_prob, _ = self.actor.sample(state)
q1_pi = self.critic1(state, sampled_action)
q2_pi = self.critic2(state, sampled_action)
q_pi = torch.min(q1_pi, q2_pi)

sac_loss = (self.alpha.detach() * log_prob - q_pi).mean()
bc_loss = nn.functional.mse_loss(sampled_action, action)
actor_loss = sac_loss + self.cfg.bc_coef * bc_loss

self.actor_optimizer.zero_grad()
actor_loss.backward()
nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_clip)
self.actor_optimizer.step()
```

### 代码解释

下面按代码出现顺序逐行解释。

- `with torch.no_grad():` 构造 TD 目标时关闭梯度。
- `next_action, next_log_prob, _ = self.actor.sample(next_state)` 在下一状态采样动作并计算 log probability。
- `target_q1 = self.target_critic1(next_state, next_action)` 第一个 target critic 估计下一动作价值。
- `target_q2 = self.target_critic2(next_state, next_action)` 第二个 target critic 估计下一动作价值。
- `target_q = torch.min(target_q1, target_q2) - self.alpha.detach() * next_log_prob` 计算最大熵软目标值，包含双 Q 取小和熵项。
- `target_q = reward + self.cfg.gamma * (1 - done) * target_q` 加上当前奖励，得到 critic 的 TD 目标。
- `current_q1 = self.critic1(state, action)` 第一个 critic 估计当前数据动作价值。
- `current_q2 = self.critic2(state, action)` 第二个 critic 估计当前数据动作价值。
- `critic1_loss = nn.functional.mse_loss(current_q1, target_q)` 第一个 critic 拟合 TD 目标。
- `critic2_loss = nn.functional.mse_loss(current_q2, target_q)` 第二个 critic 拟合 TD 目标。
- `# ------------------------------------------------------------` 是文档分隔注释。
- `# 继续：同一算法的另一段核心代码。` 表示下面展示 actor 更新。
- `# ------------------------------------------------------------` 结束分隔注释。
- `sampled_action, log_prob, _ = self.actor.sample(state)` 从当前策略中采样动作，并计算该动作的 log probability。
- `q1_pi = self.critic1(state, sampled_action)` 第一个 critic 评估采样动作。
- `q2_pi = self.critic2(state, sampled_action)` 第二个 critic 评估采样动作。
- `q_pi = torch.min(q1_pi, q2_pi)` 取双 Q 的较小值作为 actor 更新使用的价值。
- `sac_loss = (self.alpha.detach() * log_prob - q_pi).mean()` 标准 SAC actor loss：最小化熵项减 Q，等价于提高软 Q。
- `bc_loss = nn.functional.mse_loss(sampled_action, action)` 计算采样动作和数据动作之间的 BC 误差。
- `actor_loss = sac_loss + self.cfg.bc_coef * bc_loss` 将 SAC 目标和 BC 正则相加。
- `self.actor_optimizer.zero_grad()` 清空 actor 旧梯度。
- `actor_loss.backward()` 对 actor loss 反向传播。
- `nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_clip)` 裁剪 actor 梯度。
- `self.actor_optimizer.step()` 更新 actor 参数。

### 与基础版差异

SAC.py 使用 AWAC 权重更新 actor；SAC_NB.py 使用标准 SAC actor loss 加 BC 正则。SAC_NB 的形式更直观：一项追求高软 Q，一项约束动作接近离线数据。

---

## 11. TD3.py：双延迟确定性策略梯度

### 算法直觉

TD3 是 DDPG 的稳定版。它用三个技巧提高稳定性：双 Q 取小值减少过估计、目标动作平滑让 Q 更平滑、延迟 actor 更新让 critic 先学稳。

### 核心公式

目标动作平滑：

$$
a' = \mathrm{clip}(\pi_{\bar{\theta}}(s')+\mathrm{clip}(\epsilon,-c,c), -a_{\max}, a_{\max})
$$

TD 目标：

$$
y = r + \gamma(1-d)\min(Q_{\bar{\phi}_1}(s',a'),Q_{\bar{\phi}_2}(s',a'))
$$

critic 损失：

$$
L_Q = \mathrm{MSE}(Q_{\phi_1}(s,a),y)+\mathrm{MSE}(Q_{\phi_2}(s,a),y)
$$

actor 损失：

$$
L_\pi = -\mathbb{E}[Q_{\phi_1}(s,\pi_\theta(s))]
$$

### 公式解释

- 双 Q 取最小值可以抑制过估计。
- 目标动作平滑减少 critic 对某个精确动作点的过拟合。
- actor loss 只使用 `Q1`，目标是让 actor 输出更高 Q 的动作。
- actor 延迟更新通过 `policy_delay` 实现。

### 对应代码

来源：`AIcodes/TD3.py`

```python
with torch.no_grad():
    noise = (
        torch.randn_like(action) * self.policy_noise
    ).clamp(-self.noise_clip, self.noise_clip)
    next_action = (
        self.actor_target(next_state) + noise
    ).clamp(-self.max_action, self.max_action)

    target_Q1, target_Q2 = self.critic_target(next_state, next_action)
    target_Q = torch.min(target_Q1, target_Q2)
    target_Q = reward + (1 - done) * self.gamma * target_Q

current_Q1, current_Q2 = self.critic(state, action)
critic_loss = (
    nn.MSELoss()(current_Q1, target_Q)
    + nn.MSELoss()(current_Q2, target_Q)
)

# ------------------------------------------------------------
# 继续：同一算法的另一段核心代码。
# ------------------------------------------------------------

if self.total_it % self.policy_delay == 0:
    actor_loss_tensor = -self.critic.q1_forward(
        state,
        self.actor(state)
    ).mean()

    self.actor_optimizer.zero_grad()
    actor_loss_tensor.backward()
    nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
    self.actor_optimizer.step()

    soft_update(self.critic, self.critic_target, self.tau)
    soft_update(self.actor, self.actor_target, self.tau)
```

### 代码解释

下面按代码出现顺序逐行解释。

- `with torch.no_grad():` 构造 target Q 时关闭梯度，target 网络不通过这一步更新。
- `noise = (...)` 生成目标动作平滑噪声。
- `torch.randn_like(action) * self.policy_noise` 生成与动作形状相同的高斯噪声，并按 `policy_noise` 缩放。
- `.clamp(-self.noise_clip, self.noise_clip)` 将噪声裁剪到指定范围内。
- `next_action = (...)` 构造下一状态动作。
- `self.actor_target(next_state) + noise` 用 target actor 输出下一动作并加上平滑噪声。
- `.clamp(-self.max_action, self.max_action)` 将下一动作限制在合法动作范围内。
- `target_Q1, target_Q2 = self.critic_target(next_state, next_action)` 用 target critic 的双 Q 网络估计下一状态动作价值。
- `target_Q = torch.min(target_Q1, target_Q2)` 取双 Q 较小值，减少过估计。
- `target_Q = reward + (1 - done) * self.gamma * target_Q` 构造 TD 目标。
- `current_Q1, current_Q2 = self.critic(state, action)` 用当前 critic 估计数据动作的两个 Q 值。
- `critic_loss = (...)` 构造 critic 损失。
- `nn.MSELoss()(current_Q1, target_Q)` 第一个 Q 网络拟合 TD 目标。
- `+ nn.MSELoss()(current_Q2, target_Q)` 第二个 Q 网络拟合 TD 目标。
- `# ------------------------------------------------------------` 是文档分隔注释。
- `# 继续：同一算法的另一段核心代码。` 表示下面展示延迟 actor 更新。
- `# ------------------------------------------------------------` 结束分隔注释。
- `if self.total_it % self.policy_delay == 0:` 每隔 `policy_delay` 步才更新 actor。
- `actor_loss_tensor = -self.critic.q1_forward(...)` 构造 actor loss，负号表示最大化 Q。
- `state,` 是 actor 输入状态。
- `self.actor(state)` 是 actor 当前输出动作 $\pi(s)$。
- `).mean()` 对 batch 中所有样本的 Q 值取平均。
- `self.actor_optimizer.zero_grad()` 清空 actor 旧梯度。
- `actor_loss_tensor.backward()` 对 actor loss 反向传播。
- `nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)` 裁剪 actor 梯度。
- `self.actor_optimizer.step()` 更新 actor 参数。
- `soft_update(self.critic, self.critic_target, self.tau)` 软更新 target critic。
- `soft_update(self.actor, self.actor_target, self.tau)` 软更新 target actor。

### 与改进版差异

TD3.py 没有 BC 项。它直接沿 Q 梯度提升策略，因此在纯离线场景下可能更容易偏离数据分布。

---

## 12. TD3_NB.py：TD3 + BC 正则

### 算法直觉

TD3_NB 在 TD3 基础上加入 BC 正则。它保留 TD3 的双 Q、目标动作平滑和延迟更新，同时让 actor 输出动作不要离数据动作太远，更适合离线训练。

### 核心公式

TD 目标与 TD3 相同：

$$
y = r + \gamma(1-d)\min(Q_{\bar{\phi}_1}(s',a'),Q_{\bar{\phi}_2}(s',a'))
$$

actor 的 Q 项：

$$
L_Q^\pi = -\lambda \mathbb{E}[Q_{\phi_1}(s,\pi_\theta(s))]
$$

BC 项：

$$
L_{BC}^\pi = \beta\mathrm{MSE}(\pi_\theta(s),a_{data})
$$

actor 总损失：

$$
L_\pi = L_Q^\pi + L_{BC}^\pi
$$

其中：

$$
\lambda = \frac{\alpha}{\mathbb{E}[|Q_{\phi_1}(s,\pi(s))|]+\epsilon}
$$

### 公式解释

- $L_Q^\pi$：让 actor 选择 Q 值更高的动作。
- $L_{BC}^\pi$：约束 actor 输出贴近离线数据动作。
- $\lambda$：归一化 Q 项，避免 Q 尺度影响 actor loss。
- TD3_NB 的核心是“价值提升 + 分布约束”。

### 对应代码

来源：`AIcodes/TD3_NB.py`

```python
with torch.no_grad():
    noise = (
        torch.randn_like(action) * self.policy_noise
    ).clamp(-self.noise_clip, self.noise_clip)
    next_action = (
        self.actor_target(next_state) + noise
    ).clamp(-self.max_action, self.max_action)

    target_q1, target_q2 = self.critic_target(next_state, next_action)
    target_q = torch.min(target_q1, target_q2)
    target_q = reward + (1.0 - done) * self.gamma * target_q

current_q1, current_q2 = self.critic(state, action)
critic_loss = (
    nn.functional.mse_loss(current_q1, target_q)
    + nn.functional.mse_loss(current_q2, target_q)
)

# ------------------------------------------------------------
# 继续：同一算法的另一段核心代码。
# ------------------------------------------------------------

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
```

### 代码解释

下面按代码出现顺序逐行解释。

- `with torch.no_grad():` 构造 TD 目标时关闭梯度。
- `noise = (...)` 生成 TD3 目标动作平滑噪声。
- `torch.randn_like(action) * self.policy_noise` 生成与动作同形状的高斯噪声。
- `.clamp(-self.noise_clip, self.noise_clip)` 裁剪噪声范围。
- `next_action = (...)` 构造下一状态动作。
- `self.actor_target(next_state) + noise` 表示 target actor 动作加噪声。
- `.clamp(-self.max_action, self.max_action)` 将动作限制到合法范围。
- `target_q1, target_q2 = self.critic_target(next_state, next_action)` target critic 输出两个 Q 估计。
- `target_q = torch.min(target_q1, target_q2)` 取双 Q 较小值。
- `target_q = reward + (1.0 - done) * self.gamma * target_q` 构造 TD 目标。
- `current_q1, current_q2 = self.critic(state, action)` 当前 critic 估计数据动作的两个 Q 值。
- `critic_loss = (...)` 构造 critic 损失。
- `nn.functional.mse_loss(current_q1, target_q)` 第一个 Q 的 TD 误差。
- `+ nn.functional.mse_loss(current_q2, target_q)` 第二个 Q 的 TD 误差。
- `# ------------------------------------------------------------` 是文档分隔注释。
- `# 继续：同一算法的另一段核心代码。` 表示下面展示 TD3_NB 的 actor 更新。
- `# ------------------------------------------------------------` 结束分隔注释。
- `if self.total_it % self.policy_delay == 0:` 每隔固定步数更新一次 actor。
- `pi = self.actor(state)` 用 actor 生成当前状态动作。
- `q_pi = self.critic.q1_forward(state, pi)` 用第一个 Q 网络评价 actor 动作。
- `lam = self.alpha / (q_pi.abs().mean().detach() + 1e-6)` 根据 Q 值尺度计算归一化系数。
- `q_term = -lam * q_pi.mean()` 构造 Q 最大化项。
- `bc_term = self.bc_coef * nn.functional.mse_loss(pi, action)` 构造 BC 正则项，使 actor 动作接近数据动作。
- `actor_loss = q_term + bc_term` 合并 Q 项和 BC 项。
- `self.actor_optimizer.zero_grad()` 清空 actor 旧梯度。
- `actor_loss.backward()` 对 actor 总损失反向传播。
- `nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)` 裁剪 actor 梯度。
- `self.actor_optimizer.step()` 更新 actor 参数。
- `soft_update(self.actor, self.actor_target, self.tau)` 软更新 target actor。
- `soft_update(self.critic, self.critic_target, self.tau)` 软更新 target critic。

### 与基础版差异

TD3.py 的 actor 只最大化 Q；TD3_NB.py 额外加入 BC 项。这个 BC 项是离线强化学习中的安全约束，可以减少策略跑到数据分布外导致的 Q 过估计风险。

---

## 13. 总结对比

| 算法 | 核心目标 | 是否训练 critic | 是否有 BC 约束 | 主要改进点 |
| --- | --- | --- | --- | --- |
| BC | 监督学习模仿动作 | 否 | 是 | 最简单的行为克隆基线。 |
| BC_NB | 回报加权模仿动作 | 否 | 是 | 高回报轨迹权重更大。 |
| CQL | 保守估计 Q 并约束 actor | 是 | 是 | CQL penalty 压低分布外动作 Q。 |
| CQL_NB | 平滑目标动作的 CQL | 是 | 是 | 加入 TD3 风格目标动作噪声。 |
| DDPG | 优势加权 BC | 是 | 是 | 用 $Q(a_{data})-Q(\pi(s))$ 生成权重。 |
| DDPG_NB | Q 最大化 + BC | 是 | 是 | 显式组合 Q 项和 BC 项。 |
| SAC | 最大熵 critic + AWAC actor | 是 | 是 | 用优势权重提升数据动作概率。 |
| SAC_NB | 标准 SAC + BC | 是 | 是 | SAC actor loss 叠加 BC 正则。 |
| TD3 | 双 Q + 延迟更新 + 目标平滑 | 是 | 否 | 稳定版 DDPG。 |
| TD3_NB | TD3 + BC | 是 | 是 | 离线场景下增加数据分布约束。 |

从答辩角度可以这样记：

- BC 类：只看动作拟合，不看未来收益。
- CQL 类：重点是“保守”，防止 critic 高估分布外动作。
- DDPG 类：确定性策略，项目中分别体现优势加权和 BC 约束。
- SAC 类：最大熵，既看 Q，也看策略随机性。
- TD3 类：DDPG 稳定版，核心是双 Q、延迟更新和目标动作平滑。
- `_NB` 改进版：总体目标都是让离线训练更稳，常见手段包括加权样本、加噪平滑、BC 正则和 reward 标准化。

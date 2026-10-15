# 每个算法的训练代码与神经网络讲解

这份文档用于答辩复习，重点解释项目中 10 个算法脚本分别在哪里、神经网络在哪里、训练逻辑是什么、输出什么结果。

项目里的算法不是 ResNet、Unet 这类图像网络，而是离线强化学习常见的 MLP、Actor、Critic、GaussianActor。训练代码统一放在 `AIcodes/` 目录下，训练结果统一保存在 `AIcodes/checkpoints/算法名/` 目录下。

## 1. 统一理解

### 1.1 数据是什么

项目使用的是 `halfcheetah_medium-v2.hdf5` 离线强化学习数据集。可以把一条训练样本理解成：

```text
(state, action, reward, next_state, done)
```

对应代码里的变量通常是：

| 符号 | 代码变量 | 含义 |
| --- | --- | --- |
| `state` | `obs` / `state` / `s` | 当前状态，也就是 observations |
| `action` | `actions` / `action` / `a` | 数据集里已有的动作 |
| `reward` | `rewards` / `reward` / `r` | 当前动作得到的奖励 |
| `next_state` | `next_obs` / `next_state` / `s_next` | 下一时刻状态 |
| `done` | `terminals` / `done` | 是否结束 |

BC 类算法主要用 `state + action`，其他 actor-critic 算法一般会用完整 transition：`state, action, reward, next_state, done`。

### 1.2 神经网络是什么

这个项目的神经网络主要有三类：

| 网络 | 作用 |
| --- | --- |
| `BCPolicy` | 行为克隆策略网络，输入状态，输出动作 |
| `Actor` / `GaussianActor` | 策略网络，负责根据状态生成动作 |
| `Critic` | 价值网络，输入状态和动作，输出 Q 值 |

简单说：

```text
Actor / Policy：负责“做动作”
Critic：负责“评价这个动作好不好”
```

### 1.3 训练代码一般分几层

大多数脚本都有类似结构：

```text
Config / build_config_from_args
    -> 读取训练参数

load_dataset / load_hdf5_dataset
    -> 读取 HDF5 数据集

Actor / Critic / Policy
    -> 定义神经网络

train_step / train_one_epoch
    -> 单个 batch 或单个 epoch 的训练更新

train_xxx / main
    -> 整体训练流程，包括循环 epoch、验证、保存模型、保存曲线
```

### 1.4 输出文件是什么

训练完成后，结果保存在：

```text
AIcodes/checkpoints/算法名/
```

常见输出包括：

| 文件 | 含义 |
| --- | --- |
| `best_model.pt` | PyTorch 保存的最佳模型 checkpoint |
| `train_config.json` | 本次训练使用的参数配置 |
| `training_log.csv` / `training_records.csv` / `training_history.csv` | 训练日志 |
| `evaluation_summary.csv` | 评估汇总，主要在 BC 系列里有 |
| `*.png` | 训练曲线图，比如 loss、Q 值、reward、alpha |

## 2. 代码位置总览

| 算法 | 代码文件 | 神经网络位置 | 训练入口 |
| --- | --- | --- | --- |
| BC | `AIcodes/BC.py` | `BCPolicy`，约第 215 行 | `train_one_epoch`，约第 242 行；`train_bc`，约第 418 行 |
| BC_NB | `AIcodes/BC_NB.py` | `BCPolicy`，约第 267 行 | `train_one_epoch`，约第 305 行；`train_bc`，约第 498 行 |
| CQL | `AIcodes/CQL.py` | `MLP`，约第 185 行 | `train_cql_improved`，约第 396 行 |
| CQL_NB | `AIcodes/CQL_NB.py` | `MLP`，约第 194 行 | `train_cql_improved`，约第 420 行 |
| DDPG | `AIcodes/DDPG.py` | `Actor`，约第 177 行；`Critic`，约第 194 行 | `train_step`，约第 332 行；`train_ddpg_improved`，约第 529 行 |
| DDPG_NB | `AIcodes/DDPG_NB.py` | `Actor`，约第 177 行；`Critic`，约第 194 行 | `train_step`，约第 333 行；`train_ddpg_improved`，约第 510 行 |
| SAC | `AIcodes/SAC.py` | `GaussianActor`，约第 224 行；`Critic`，约第 284 行 | `train_step`，约第 415 行；`train_sac`，约第 608 行 |
| SAC_NB | `AIcodes/SAC_NB.py` | `GaussianActor`，约第 204 行；`Critic`，约第 244 行 | `train_step`，约第 357 行；`train_sac`，约第 503 行 |
| TD3 | `AIcodes/TD3.py` | `Actor`，约第 166 行；`Critic`，约第 183 行 | `train_step`，约第 275 行；`main`，约第 430 行 |
| TD3_NB | `AIcodes/TD3_NB.py` | `Actor`，约第 182 行；`Critic`，约第 202 行 | `train_step`，约第 306 行；`main`，约第 491 行 |

## 3. BC：行为克隆

### 3.1 代码在哪里

```text
训练脚本：AIcodes/BC.py
神经网络：BCPolicy
单轮训练：train_one_epoch
整体训练：train_bc
```

### 3.2 它解决什么问题

BC 全称 Behavior Cloning，中文可以说“行为克隆”。它是最像监督学习的算法。

它不看 reward，也不训练 critic。它只做一件事：

```text
输入状态 state
    -> 网络预测动作 pred_action
    -> 和数据集里的真实动作 action 比较
    -> 让预测动作越来越接近真实动作
```

### 3.3 神经网络是什么

网络类是：

```python
class BCPolicy(nn.Module):
```

它是一个 MLP 策略网络：

```text
输入：17 维状态 observations
隐藏层：默认多层全连接层 + ReLU
输出：6 维动作 actions
输出层：tanh，把动作限制在 [-1, 1]
```

可以理解成：

```text
BCPolicy(state) -> action
```

### 3.4 训练逻辑

核心训练函数是：

```python
def train_one_epoch(model, loader, optimizer, loss_fn, obs_mean, obs_std, config):
```

关键逻辑是：

```python
batch_obs = (batch_obs - obs_mean) / obs_std
pred_act = model(batch_obs)
loss = loss_fn(pred_act, batch_act)
loss.backward()
optimizer.step()
```

意思是：

```text
先对状态做标准化
再用 BCPolicy 预测动作
然后计算预测动作和真实动作之间的误差
最后反向传播更新网络参数
```

### 3.5 Loss 怎么理解

BC 的 loss 就是动作拟合误差：

```text
loss = predicted_action 和 dataset_action 的误差
```

可以是 SmoothL1Loss，也可以是 MSELoss。

### 3.6 输出文件

```text
AIcodes/checkpoints/BC/best_model.pt
AIcodes/checkpoints/BC/training_log.csv
AIcodes/checkpoints/BC/evaluation_summary.csv
AIcodes/checkpoints/BC/train_val_loss.png
AIcodes/checkpoints/BC/lr_curve.png
AIcodes/checkpoints/BC/train_config.json
```

### 3.7 答辩说法

BC 是本项目中最基础的行为克隆方法。它把离线强化学习简化成监督学习，输入 HalfCheetah 的状态向量，输出动作向量，用数据集里的动作作为标签进行训练。它的网络是 `BCPolicy`，本质是 MLP，最后用 `tanh` 限制动作范围。

## 4. BC_NB：加权行为克隆

### 4.1 代码在哪里

```text
训练脚本：AIcodes/BC_NB.py
神经网络：BCPolicy
样本权重：build_sample_weights
单轮训练：train_one_epoch
整体训练：train_bc
```

### 4.2 它和 BC 的区别

BC_NB 还是行为克隆，但是它不再把所有样本同等对待。

它会根据一条 episode 的累计 reward 计算权重：

```text
高回报轨迹里的样本权重大
低回报轨迹里的样本权重小
```

这样模型会更重视“表现好的轨迹”。

### 4.3 神经网络是什么

BC_NB 的网络仍然是：

```python
class BCPolicy(nn.Module):
```

结构和 BC 类似：

```text
state -> MLP -> tanh -> action
```

区别不在网络结构，而在 loss 计算方式。

### 4.4 权重怎么来

核心函数：

```python
def build_sample_weights(rewards, terminals, config):
```

逻辑是：

```text
根据 terminals 把数据切成多条 episode
    -> 计算每条 episode 的累计 reward
    -> 对累计 reward 标准化
    -> 用 exp 生成 episode 权重
    -> 每个样本继承所在 episode 的权重
```

### 4.5 训练逻辑

核心代码：

```python
sample_loss = per_sample_loss(pred_act, batch_act, config.loss_type)
loss = (sample_loss * batch_w).mean()
```

含义：

```text
先算每个样本自己的动作误差
再乘以样本权重
最后求平均
```

### 4.6 Loss 怎么理解

BC_NB 的 loss 是加权行为克隆损失：

```text
weighted_loss = sample_weight * action_prediction_error
```

如果一个样本来自高回报轨迹，它的误差会被放大，模型会更认真地学习它。

### 4.7 输出文件

```text
AIcodes/checkpoints/BC_NB/best_model.pt
AIcodes/checkpoints/BC_NB/training_log.csv
AIcodes/checkpoints/BC_NB/evaluation_summary.csv
AIcodes/checkpoints/BC_NB/train_test_loss.png
AIcodes/checkpoints/BC_NB/lr_curve.png
AIcodes/checkpoints/BC_NB/train_config.json
```

### 4.8 答辩说法

BC_NB 是 BC 的改进版。普通 BC 对所有样本一视同仁，而 BC_NB 会根据轨迹累计回报给样本加权，让高回报轨迹中的动作对训练影响更大。它的网络结构和 BC 基本一致，主要改进体现在样本权重和加权 loss 上。

## 5. CQL：保守 Q 学习

### 5.1 代码在哪里

```text
训练脚本：AIcodes/CQL.py
神经网络：MLP
CQL 惩罚项：compute_cql_penalty
整体训练：train_cql_improved
```

### 5.2 它解决什么问题

CQL 全称 Conservative Q-Learning，中文可以说“保守 Q 学习”。

离线强化学习不能在线试错，所以如果模型对数据集中没见过的动作给出很高 Q 值，就可能学到危险策略。CQL 的核心就是：

```text
对数据外动作保持保守
不要随便给没见过的动作高分
```

### 5.3 神经网络是什么

CQL 用同一个 `MLP` 类构建不同网络：

```text
actor：输入 state，输出 action
critic1：输入 state + action，输出 Q 值
critic2：输入 state + action，输出 Q 值
target_critic1 / target_critic2：目标 critic，用于稳定训练
```

可以理解成：

```text
actor(state) -> action
critic(state, action) -> Q value
```

### 5.4 训练逻辑

训练函数：

```python
def train_cql_improved(...):
```

每个 batch 的主要步骤：

```text
读取 state, action, reward, next_state, done
    -> 标准化 state
    -> actor 生成 next_action
    -> target critic 计算 TD target
    -> critic 拟合 TD target
    -> 加上 CQL penalty
    -> actor 根据 Q 值和 BC 项更新
    -> soft update target critic
```

### 5.5 Loss 怎么理解

CQL 的 critic loss 由两部分组成：

```text
critic_loss = TD 误差 + CQL 保守惩罚项
```

CQL penalty 的直觉：

```text
比较随机动作、当前策略动作和数据动作的 Q 值
    -> 压低随机动作和策略动作的过高 Q 值
    -> 保留数据动作的价值
```

actor loss 大致是：

```text
actor_loss = 让策略动作 Q 值更高 + 行为克隆约束
```

行为克隆约束是为了让 actor 不要偏离离线数据太远。

### 5.6 输出文件

```text
AIcodes/checkpoints/CQL/best_model.pt
AIcodes/checkpoints/CQL/training_log.csv
AIcodes/checkpoints/CQL/critic_loss_curve.png
AIcodes/checkpoints/CQL/actor_loss_curve.png
AIcodes/checkpoints/CQL/cql_penalty_curve.png
AIcodes/checkpoints/CQL/train_config.json
```

### 5.7 答辩说法

CQL 是典型的离线强化学习算法。它不仅训练 actor，还训练 critic 来估计 Q 值。由于离线数据不能在线探索，CQL 增加了保守惩罚项，避免 critic 对数据集外的动作产生过高估计，从而提高离线训练稳定性。

## 6. CQL_NB：增强版 CQL

### 6.1 代码在哪里

```text
训练脚本：AIcodes/CQL_NB.py
神经网络：MLP
CQL 惩罚项：compute_cql_penalty
整体训练：train_cql_improved
```

### 6.2 它和 CQL 的区别

CQL_NB 是在 CQL 基础上的增强版。主要区别是：

```text
CQL penalty 覆盖动作更多
加入 next policy action
加入 target smoothing noise
```

普通 CQL 主要比较：

```text
随机动作 + 当前策略动作 + 数据动作
```

CQL_NB 进一步加入：

```text
下一状态策略动作 next policy action
```

### 6.3 神经网络是什么

仍然使用 `MLP` 构建：

```text
actor
critic1
critic2
target_critic1
target_critic2
```

网络结构不是主要变化点，主要变化在训练目标和 penalty 计算。

### 6.4 训练逻辑

CQL_NB 在计算下一状态动作时会加噪声：

```text
next_action = actor(next_state) + noise
```

然后做裁剪：

```text
next_action 被限制在合法动作范围内
```

这样做类似 TD3 的 target smoothing，可以减少 critic 被某个尖锐高值动作误导。

### 6.5 Loss 怎么理解

CQL_NB 的 critic loss 仍然是：

```text
critic_loss = TD 误差 + CQL penalty
```

但是 CQL penalty 更完整：

```text
随机动作
当前策略动作
下一状态策略动作
数据动作
```

这让 critic 对更多可能的数据外动作保持保守。

### 6.6 输出文件

```text
AIcodes/checkpoints/CQL_NB/best_model.pt
AIcodes/checkpoints/CQL_NB/training_log.csv
AIcodes/checkpoints/CQL_NB/critic_loss_curve.png
AIcodes/checkpoints/CQL_NB/actor_loss_curve.png
AIcodes/checkpoints/CQL_NB/cql_penalty_curve.png
AIcodes/checkpoints/CQL_NB/train_config.json
```

### 6.7 答辩说法

CQL_NB 是 CQL 的改进版，主要增强了保守约束和目标动作平滑。它把下一状态策略动作也纳入 CQL penalty，并给目标动作加噪声，目的是让 Q 值估计更稳，减少离线强化学习中策略跑到数据分布外的问题。

## 7. DDPG：离线增强版 DDPG

### 7.1 代码在哪里

```text
训练脚本：AIcodes/DDPG.py
神经网络：Actor、Critic
Agent 类：OfflineAWDDPGAgent
单步训练：train_step
整体训练：train_ddpg_improved
```

### 7.2 它解决什么问题

DDPG 是确定性策略梯度方法，基本思想是：

```text
Actor 负责输出动作
Critic 负责评价动作的 Q 值
Actor 根据 Critic 的评价改进动作
```

项目里的 `DDPG.py` 已经不是最原始的 DDPG，而是更适合离线场景的增强版本，包含双 critic、target network、目标动作噪声、优势加权等设计。

### 7.3 神经网络是什么

`Actor`：

```text
输入 state
输出 action
最后使用 tanh 限制动作范围
```

`Critic`：

```text
输入 state + action
输出 Q 值
```

项目中有两个 critic：

```text
critic1
critic2
```

这样可以减少 Q 值过高估计。

### 7.4 训练逻辑

`train_step` 里主要做：

```text
1. 标准化 state 和 next_state
2. 对 reward 做标准化
3. target actor 生成 next_action，并加噪声
4. target critic 计算 TD target
5. 当前 critic 拟合 TD target
6. 计算 advantage 权重
7. 用加权 BC loss 更新 actor
8. soft update target 网络
```

### 7.5 Loss 怎么理解

critic loss：

```text
critic_loss = MSE(Q1(state, action), target_q)
            + MSE(Q2(state, action), target_q)
```

actor loss 不是简单最大化 Q，而是 advantage-weighted BC：

```text
actor_loss = weight * MSE(actor(state), data_action)
```

其中 weight 来自 advantage：

```text
如果数据动作比当前策略动作更好，权重更大
```

### 7.6 输出文件

```text
AIcodes/checkpoints/DDPG/best_model.pt
AIcodes/checkpoints/DDPG/training_records.csv
AIcodes/checkpoints/DDPG/critic_loss_curve.png
AIcodes/checkpoints/DDPG/actor_loss_curve.png
AIcodes/checkpoints/DDPG/weight_curve.png
AIcodes/checkpoints/DDPG/train_config.json
```

### 7.7 答辩说法

DDPG 这部分采用 actor-critic 结构，actor 输出动作，critic 评价动作价值。项目中的实现加入了双 critic、target 网络和 advantage-weighted 行为克隆思想，让它更适合离线数据训练，避免策略过度偏离数据集。

## 8. DDPG_NB：DDPG + BC 正则

### 8.1 代码在哪里

```text
训练脚本：AIcodes/DDPG_NB.py
神经网络：Actor、Critic
Agent 类：OfflineDDPGBCAgent
单步训练：train_step
整体训练：train_ddpg_improved
```

### 8.2 它和 DDPG 的区别

DDPG_NB 更明确地把 actor loss 分成两部分：

```text
Q 项：鼓励 actor 输出高 Q 值动作
BC 项：约束 actor 不要偏离数据集动作太远
```

它的核心是：

```text
既想让动作价值更高
又不能让策略跑出离线数据分布
```

### 8.3 神经网络是什么

仍然是：

```text
Actor(state) -> action
Critic(state, action) -> Q value
```

并且使用双 critic 和 target network。

### 8.4 训练逻辑

`train_step` 中：

```text
先更新 critic
    -> 用 TD target 拟合 Q 值

再延迟更新 actor
    -> 计算 q_loss
    -> 计算 bc_loss
    -> actor_loss = q_loss + bc_loss
```

### 8.5 Loss 怎么理解

critic loss：

```text
critic_loss = TD 误差
```

actor loss：

```text
q_loss = -lambda * Q(state, actor(state))
bc_loss = bc_coef * MSE(actor(state), data_action)
actor_loss = q_loss + bc_loss
```

解释：

```text
q_loss 越小，表示 actor 更倾向输出高 Q 动作
bc_loss 越小，表示 actor 越接近数据集动作
```

### 8.6 输出文件

```text
AIcodes/checkpoints/DDPG_NB/best_model.pt
AIcodes/checkpoints/DDPG_NB/training_records.csv
AIcodes/checkpoints/DDPG_NB/critic_loss_curve.png
AIcodes/checkpoints/DDPG_NB/actor_loss_curve.png
AIcodes/checkpoints/DDPG_NB/bc_loss_curve.png
AIcodes/checkpoints/DDPG_NB/train_config.json
```

### 8.7 答辩说法

DDPG_NB 是在 DDPG 思路上加入 BC 正则的改进版本。它的 actor 更新同时考虑 Q 值和行为克隆误差，既希望策略动作价值高，又希望动作不要偏离离线数据太远。

## 9. SAC：AWAC-SAC 风格方法

### 9.1 代码在哪里

```text
训练脚本：AIcodes/SAC.py
神经网络：MLP、GaussianActor、Critic
Agent 类：OfflineAWACSACAgent
单步训练：train_step
整体训练：train_sac
```

### 9.2 它解决什么问题

SAC 全称 Soft Actor-Critic。它通常强调：

```text
既追求高 reward
又保留一定随机性
```

项目里的 `SAC.py` 更接近 AWAC-SAC 风格，在 actor 更新时使用 advantage 权重和数据动作的 log probability。

### 9.3 神经网络是什么

`GaussianActor`：

```text
输入 state
输出动作分布的 mean 和 log_std
从高斯分布中采样动作
动作经过 tanh 限制范围
```

`Critic`：

```text
输入 state + action
输出 Q 值
```

项目中使用：

```text
critic1
critic2
target_critic1
target_critic2
```

### 9.4 训练逻辑

`train_step` 中主要做：

```text
1. 标准化 state、next_state、reward
2. 用 target critic 计算 soft TD target
3. 更新 critic1 和 critic2
4. 计算数据动作的 advantage
5. 根据 advantage 得到权重
6. 用加权 log probability 更新 actor
7. 更新 alpha 熵系数
8. soft update target critic
```

### 9.5 Loss 怎么理解

critic target 中包含 entropy 项：

```text
target_q = reward + gamma * (min(target_Q1, target_Q2) - alpha * log_prob)
```

actor loss 是 AWAC 风格：

```text
actor_loss = -mean(weight * log_prob_data)
```

解释：

```text
如果数据动作 advantage 高，就提高它在当前策略下的概率
```

### 9.6 输出文件

```text
AIcodes/checkpoints/SAC/best_model.pt
AIcodes/checkpoints/SAC/training_history.csv
AIcodes/checkpoints/SAC/actor_loss.png
AIcodes/checkpoints/SAC/critic1_loss.png
AIcodes/checkpoints/SAC/critic2_loss.png
AIcodes/checkpoints/SAC/alpha.png
AIcodes/checkpoints/SAC/reward.png
AIcodes/checkpoints/SAC/Q1.png
AIcodes/checkpoints/SAC/Q2.png
AIcodes/checkpoints/SAC/advantage.png
AIcodes/checkpoints/SAC/weight.png
AIcodes/checkpoints/SAC/log_prob_data.png
AIcodes/checkpoints/SAC/train_config.json
```

### 9.7 答辩说法

SAC 采用随机策略 actor 和双 critic。项目中的实现结合了 AWAC 思想，通过 advantage 权重提高优质数据动作的概率，同时维护 alpha 熵系数，让策略训练更加稳定。

## 10. SAC_NB：SAC + BC 正则

### 10.1 代码在哪里

```text
训练脚本：AIcodes/SAC_NB.py
神经网络：MLP、GaussianActor、Critic
Agent 类：OfflineSACAgent
单步训练：train_step
整体训练：train_sac
```

### 10.2 它和 SAC 的区别

SAC_NB 更直接地把 actor loss 写成：

```text
actor_loss = sac_loss + bc_coef * bc_loss
```

也就是：

```text
SAC 原本的策略优化
    + 行为克隆约束
```

### 10.3 神经网络是什么

仍然是：

```text
GaussianActor(state) -> sampled_action
Critic(state, action) -> Q value
```

`GaussianActor` 输出的是动作分布，不是单个固定动作，所以它能计算 log probability，也能支持 SAC 的熵项。

### 10.4 训练逻辑

`train_step` 中：

```text
1. 计算 target_q
2. 更新 critic1 和 critic2
3. actor 采样动作 sampled_action
4. 计算 sac_loss
5. 计算 bc_loss
6. actor_loss = sac_loss + bc_coef * bc_loss
7. 更新 alpha
8. soft update target critic
```

### 10.5 Loss 怎么理解

SAC loss：

```text
sac_loss = alpha * log_prob - Q
```

BC loss：

```text
bc_loss = MSE(sampled_action, data_action)
```

总 loss：

```text
actor_loss = sac_loss + bc_coef * bc_loss
```

解释：

```text
SAC 项让策略追求高 Q 和合适随机性
BC 项让策略动作靠近离线数据动作
```

### 10.6 输出文件

```text
AIcodes/checkpoints/SAC_NB/best_model.pt
AIcodes/checkpoints/SAC_NB/training_history.csv
AIcodes/checkpoints/SAC_NB/actor_loss.png
AIcodes/checkpoints/SAC_NB/critic1_loss.png
AIcodes/checkpoints/SAC_NB/critic2_loss.png
AIcodes/checkpoints/SAC_NB/alpha.png
AIcodes/checkpoints/SAC_NB/reward.png
AIcodes/checkpoints/SAC_NB/Q1.png
AIcodes/checkpoints/SAC_NB/Q2.png
AIcodes/checkpoints/SAC_NB/bc_loss.png
AIcodes/checkpoints/SAC_NB/log_prob.png
AIcodes/checkpoints/SAC_NB/train_config.json
```

### 10.7 答辩说法

SAC_NB 是在 SAC 基础上加入 BC 正则的版本。它保留 SAC 的最大熵策略优化，同时用行为克隆损失约束策略不要偏离离线数据动作，适合离线强化学习场景。

## 11. TD3：双 Q 延迟策略更新

### 11.1 代码在哪里

```text
训练脚本：AIcodes/TD3.py
神经网络：Actor、Critic
Trainer 类：TD3Trainer
单步训练：train_step
整体训练：main
```

### 11.2 它解决什么问题

TD3 可以理解成更稳定的 DDPG。它的核心技巧有三个：

```text
双 critic
目标动作加噪声
延迟更新 actor
```

这些设计都是为了减少 Q 值高估和训练抖动。

### 11.3 神经网络是什么

`Actor`：

```text
输入 state
输出 action
```

`Critic`：

```text
输入 state + action
输出两个 Q 值：Q1 和 Q2
```

TD3 会取两个 Q 值中较小的一个：

```text
target_Q = min(Q1, Q2)
```

这样可以更保守，避免高估。

### 11.4 训练逻辑

`train_step` 中：

```text
1. 标准化 state 和 next_state
2. target actor 生成 next_action
3. 给 next_action 加噪声并裁剪
4. target critic 计算 target_Q1 和 target_Q2
5. 取 min(target_Q1, target_Q2)
6. 当前 critic 拟合 target_Q
7. 每隔 policy_delay 步更新一次 actor
8. soft update actor 和 critic 的 target 网络
```

### 11.5 Loss 怎么理解

critic loss：

```text
critic_loss = MSE(current_Q1, target_Q)
            + MSE(current_Q2, target_Q)
```

actor loss：

```text
actor_loss = -Q1(state, actor(state))
```

最小化 `-Q` 等价于让 actor 输出 critic 认为价值更高的动作。

### 11.6 输出文件

```text
AIcodes/checkpoints/TD3/best_model.pt
AIcodes/checkpoints/TD3/training_log.csv
AIcodes/checkpoints/TD3/critic_loss.png
AIcodes/checkpoints/TD3/actor_loss.png
AIcodes/checkpoints/TD3/q_value.png
AIcodes/checkpoints/TD3/train_config.json
```

### 11.7 答辩说法

TD3 是 DDPG 的稳定化版本。它使用双 critic 减少 Q 值高估，给目标动作加噪声做平滑，并延迟更新 actor，让 critic 先学得更稳定，再指导策略更新。

## 12. TD3_NB：TD3 + BC 正则

### 12.1 代码在哪里

```text
训练脚本：AIcodes/TD3_NB.py
神经网络：Actor、Critic
Trainer 类：TD3NBTrainer
单步训练：train_step
整体训练：main
```

### 12.2 它和 TD3 的区别

TD3_NB 在 TD3 的基础上加入了 BC 正则和 reward scaling。

普通 TD3 的 actor 主要追求：

```text
输出高 Q 动作
```

TD3_NB 的 actor 同时追求：

```text
输出高 Q 动作
动作不要偏离数据集动作太远
```

### 12.3 神经网络是什么

仍然是：

```text
Actor(state) -> action
Critic(state, action) -> Q1, Q2
```

网络结构和 TD3 类似，主要改进在 actor loss。

### 12.4 训练逻辑

`train_step` 中：

```text
1. 标准化 state、next_state
2. 可选标准化 reward
3. target actor 生成 next_action，并加入噪声
4. target critic 计算 target_q
5. critic 拟合 target_q
6. 延迟更新 actor
7. actor_loss = q_term + bc_term
8. soft update target 网络
```

### 12.5 Loss 怎么理解

critic loss：

```text
critic_loss = MSE(current_q1, target_q)
            + MSE(current_q2, target_q)
```

actor loss：

```text
q_term = -lambda * Q(state, actor(state))
bc_term = bc_coef * MSE(actor(state), data_action)
actor_loss = q_term + bc_term
```

解释：

```text
q_term 负责提高动作价值
bc_term 负责限制策略偏离离线数据
```

### 12.6 输出文件

```text
AIcodes/checkpoints/TD3_NB/best_model.pt
AIcodes/checkpoints/TD3_NB/training_log.csv
AIcodes/checkpoints/TD3_NB/critic_loss.png
AIcodes/checkpoints/TD3_NB/actor_loss.png
AIcodes/checkpoints/TD3_NB/q_value.png
AIcodes/checkpoints/TD3_NB/train_config.json
```

### 12.7 答辩说法

TD3_NB 是 TD3 的离线强化学习改进版。它保留双 critic、目标动作噪声和延迟更新，同时在 actor loss 中加入 BC 项，让策略在追求高 Q 值时仍然贴近数据集动作。

## 13. Base 和 NB 的整体区别

项目中很多算法都成对出现：

```text
BC / BC_NB
CQL / CQL_NB
DDPG / DDPG_NB
SAC / SAC_NB
TD3 / TD3_NB
```

可以整体理解为：

```text
Base 版：基础算法逻辑
NB 版：在基础算法上加入更适合离线场景的改进
```

常见改进包括：

| 改进 | 出现位置 | 作用 |
| --- | --- | --- |
| 样本加权 | BC_NB | 高回报轨迹更重要 |
| 更强 CQL penalty | CQL_NB | 约束更多数据外动作 |
| target smoothing | CQL_NB、TD3 系列 | 让目标 Q 更平滑 |
| BC 正则 | DDPG_NB、SAC_NB、TD3_NB | 防止 actor 偏离离线数据 |
| reward scaling | 多个 actor-critic 脚本 | 降低 reward 尺度影响 |

## 14. 答辩时怎么串起来

可以这样讲：

```text
我的算法训练代码都在 AIcodes 目录下，每个算法一个 Python 脚本。
BC 系列是行为克隆，只需要状态和动作。
CQL、DDPG、SAC、TD3 系列是 actor-critic 方法，需要完整 transition 数据。
其中 actor 负责生成动作，critic 负责评价动作价值。
训练完成后，脚本会保存 best_model.pt、训练日志 CSV 和曲线图。
Django 页面不会实时训练，而是读取 checkpoints 下的结果文件进行展示。
```

再补一句更具体的：

```text
Base 版本展示基础算法思路，NB 版本主要加入样本加权、保守约束、目标平滑或 BC 正则等改进，用来说明离线强化学习中如何让策略更稳定、更不容易偏离数据分布。
```

## 15. 老师可能问的问题

### 15.1 你的神经网络在哪里

回答：

```text
神经网络都在 AIcodes 目录的各个算法脚本里。BC 系列是 BCPolicy，CQL 系列用 MLP 构建 actor 和 critic，DDPG/TD3 是 Actor 和 Critic，SAC 系列是 GaussianActor 和 Critic。
```

### 15.2 你的模型保存在哪里

回答：

```text
训练好的模型保存在 AIcodes/checkpoints/算法名/best_model.pt。每个算法目录下还有训练日志 CSV、配置文件 JSON 和评价曲线 PNG。
```

### 15.3 Django 有没有直接训练模型

回答：

```text
Django 不直接训练模型。训练由 AIcodes 里的 PyTorch 脚本完成。Django 主要负责读取 checkpoints 下已经生成的模型、日志和曲线图，然后展示到网页上。
```

### 15.4 train_sac 和 train_step 有什么区别

回答：

```text
train_sac 是整体训练流程，负责 epoch 循环、训练集和验证集遍历、日志记录、模型保存和曲线生成。train_step 是单个 batch 的训练更新，负责一次 critic、actor 和 alpha 的参数更新。
```

### 15.5 为什么要有 NB 版本

回答：

```text
NB 版本是对 Base 版本的改进，主要是为了更适合离线强化学习。因为离线数据不能在线试错，所以需要加入样本加权、保守约束、BC 正则等方法，让策略不要过度偏离数据集。
```


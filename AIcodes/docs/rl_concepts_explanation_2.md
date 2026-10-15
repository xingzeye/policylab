# 强化学习核心概念解释 2：从 TD Target 开始

这份文档用来解释强化学习公式里经常一起出现的概念，尤其是：

- TD target
- TD error
- $Q(s,a)$
- $V(s)$
- Advantage
- Bellman equation
- Actor / Critic
- Target network
- Offline RL
- BC / CQL / SAC / TD3 / DDPG 里常见的训练项

阅读这份文档时，可以把离线数据集中的一条样本统一记成：

$$
(s,a,r,s_{\text{next}},d)
$$

含义：

- $s$：当前状态
- $a$：当前状态下执行的动作
- $r$：执行动作后得到的即时奖励
- $s_{\text{next}}$：执行动作后到达的下一状态
- $d$：是否终止，终止时 $d=1$，否则 $d=0$

---

## 1. TD Target 是什么

TD target 可以理解成：critic 要学习的目标答案。

在 DDPG、TD3、SAC、CQL 这类 actor-critic 算法里，critic 的任务是估计：

$$
Q(s,a)
$$

也就是“在状态 $s$ 下执行动作 $a$，未来大概能获得多少总回报”。

但是训练 critic 时，需要有一个“目标值”来告诉它应该往哪里靠。这个目标值就是 TD target。

最常见的 TD target 形式：

$$
y = r + \gamma(1-d)Q_{\text{target}}(s_{\text{next}},a_{\text{next}})
$$

解释：

- $y$：TD target，也就是 critic 的监督目标
- $r$：当前这一步已经拿到的奖励
- $\gamma$：折扣因子，控制未来奖励的重要性
- $(1-d)$：如果已经终止，就不再看未来
- $Q_{\text{target}}(s_{\text{next}},a_{\text{next}})$：下一状态继续走下去的未来价值估计

所以 TD target 的意思是：

$$
\text{目标价值}=\text{当前奖励}+\text{折扣后的未来价值}
$$

如果当前样本已经结束，即 $d=1$：

$$
y=r
$$

因为没有下一步了。

---

## 2. TD Target 在项目里的典型形式

项目里的 DDPG、DDPG_NB、TD3、TD3_NB、CQL、CQL_NB 都用了类似结构：

$$
y = r + \gamma(1-d)
\min\left(
Q_{1,\text{target}}(s_{\text{next}},a_{\text{next}}),
Q_{2,\text{target}}(s_{\text{next}},a_{\text{next}})
\right)
$$

这里比最基础的 TD target 多了一个：

$$
\min(Q_{1,\text{target}},Q_{2,\text{target}})
$$

原因是这些算法用了双 critic。

双 critic 会分别估计下一步价值：

$$
\begin{aligned}
Q_{1,\text{target}}(s_{\text{next}},a_{\text{next}}) \\
Q_{2,\text{target}}(s_{\text{next}},a_{\text{next}})
\end{aligned}
$$

然后取较小的那个：

$$
\min(Q_{1,\text{target}},Q_{2,\text{target}})
$$

这样做是为了减少 Q 值过估计。

直观理解：两个 critic 中，谁估得保守就信谁。

这样 TD target 不容易被虚高的 Q 值带偏。

---

## 3. TD Target 和 Critic Loss 的关系

critic 当前预测的是：

$$
Q(s,a)
$$

TD target 是：

$$
y
$$

critic loss 通常就是：

$$
\mathcal{L}_{\text{critic}}
=\operatorname{MSE}\left(Q(s,a),y\right)
$$

如果是双 critic：

$$
\mathcal{L}_{\text{critic}}
=\operatorname{MSE}\left(Q_1(s,a),y\right)
+\operatorname{MSE}\left(Q_2(s,a),y\right)
$$

也就是说：

$$
\text{critic 的训练}=\text{让当前 Q 预测值靠近 TD target}
$$

TD target 是“训练标签”，$Q(s,a)$ 是“模型预测”。

---

## 4. TD Error 是什么

TD error 是当前预测和 TD target 的差距。

常见写法：

$$
\delta = y - Q(s,a)
$$

或者：

$$
\delta = r+\gamma Q(s_{\text{next}},a_{\text{next}})-Q(s,a)
$$

含义：

- $\delta>0$：说明当前 Q 估低了，应该调高
- $\delta<0$：说明当前 Q 估高了，应该调低
- $\delta=0$：说明当前预测和目标一致

critic loss 本质上就是让 TD error 变小：

$$
\mathcal{L}_{\text{critic}}=\delta^2
$$

如果用 MSE，就是一批样本 TD error 平方的平均值。

---

## 5. TD 是什么

TD 是 Temporal Difference，中文常译为“时序差分”。

它的核心思想是：

用当前一步奖励 + 下一状态的估计价值，来更新当前状态的价值。

也就是：

$$
\text{当前估计}\leftarrow \text{当前奖励}+\text{未来估计}
$$

TD 的特点是：它不需要等整个 episode 结束，就可以更新价值函数。

这和 Monte Carlo 方法不同。

---

## 6. Monte Carlo Target 和 TD Target 的区别

Monte Carlo target 使用一整条轨迹的真实回报：

$$
G_t=r_t+\gamma r_{t+1}+\gamma^2 r_{t+2}+\cdots
$$

TD target 使用一步真实奖励加下一步估计：

$$
y_t=r_t+\gamma V(s_{t+1})
$$

对比：

| 概念 | 使用什么作为目标 | 优点 | 缺点 |
|---|---|---|---|
| Monte Carlo target | 完整未来真实回报 | 偏差较小 | 方差大，必须等轨迹结束 |
| TD target | 当前奖励 + 下一步价值估计 | 更新快，方差较小 | 有估计偏差 |

一句话：Monte Carlo 看完整未来，TD 用下一步估计未来。

---

## 7. Bootstrapping 是什么

Bootstrapping 指的是：用模型自己的估计值，构造新的训练目标。

TD target 中的：

$$
Q_{\text{target}}(s_{\text{next}},a_{\text{next}})
$$

就是一个模型估计值。

所以 TD 学习属于 bootstrapping。

直观理解：不是等完整答案出来，而是先用模型对下一步的估计，拼出当前这一步的答案。

优点：

- 可以边走边学
- 不需要完整轨迹
- 训练效率高

风险：

- 如果下一步 Q 估错了，当前 TD target 也会跟着错
- 离线强化学习里，这种错误可能被反复放大

---

## 8. Reward 和 Return

reward 是一步奖励：

$$
r_t
$$

return 是从某一步开始的累计折扣奖励：

$$
G_t=r_t+\gamma r_{t+1}+\gamma^2 r_{t+2}+\cdots
$$

区别：

> reward 是眼前这一口收益；return 是从现在开始一路走下去的总收益。

强化学习真正想最大化的通常不是单步 reward，而是长期 return。

---

## 9. Gamma 折扣因子

$\gamma$ 用来控制未来奖励的重要程度：

$$
0 \le \gamma \le 1
$$

如果：

$$
\gamma=0
$$

模型只关心当前奖励：

$$
y=r
$$

如果：

当 $\gamma$ 接近 $1$ 时：

模型会更重视长期奖励。

常见值：

$$
\gamma=0.99
$$

直观理解：

> $\gamma$ 越大，越有长远眼光；$\gamma$ 越小，越短视。

---

## 10. Value Function：V(s)

$V(s)$ 表示状态价值函数。

它回答的问题是：

在状态 $s$ 下，按照当前策略继续行动，未来大概能拿多少总回报？

公式上可以理解为：

$$
V(s)=\mathbb{E}\left[\text{return}\mid s\right]
$$

它只看状态，不指定动作。

例子：

当 $V(s)$ 高时：

说明这个状态整体比较有前途。

---

## 11. Action-Value Function：Q(s, a)

$Q(s,a)$ 表示动作价值函数。

它回答的问题是：

在状态 $s$ 下先执行动作 $a$，之后按照某个策略继续走，未来大概能拿多少总回报？

公式上可以理解为：

$$
Q(s,a)=\mathbb{E}\left[\text{return}\mid s,a\right]
$$

它同时看状态和动作。

对 actor-critic 方法来说，critic 通常学的就是 $Q(s,a)$。

---

## 12. V(s) 和 Q(s, a) 的区别

| 概念 | 输入 | 回答的问题 |
|---|---|---|
| $V(s)$ | 状态 | 这个状态好不好 |
| $Q(s,a)$ | 状态 + 动作 | 在这个状态下做这个动作好不好 |

如果策略是 $\pi(a\mid s)$，那么：

$$
V(s)=\mathbb{E}_{a\sim\pi(\cdot\mid s)}\left[Q(s,a)\right]
$$

意思是：

$$
\text{状态价值}=\text{在这个状态下，按策略可能采取的动作价值的平均}
$$

如果是确定性策略 $a=\pi(s)$，可以近似理解为：

$$
V(s)\approx Q(s,\pi(s))
$$

---

## 13. Advantage 优势函数

优势函数表示：

优势函数表示：某个动作比当前策略平均水平好多少。

常见定义：

$$
A(s,a)=Q(s,a)-V(s)
$$

在你项目里的 DDPG 和 SAC 相关实现中，经常用类似写法：

$$
A(s,a)=Q(s,a)-Q(s,\pi(s))
$$

含义：

- $Q(s,a)$：数据集动作的价值
- $Q(s,\pi(s))$：当前策略动作的价值
- $A>0$：数据动作比当前策略动作更好
- $A<0$：数据动作不如当前策略动作

优势函数常用于给样本加权：

$$
w=\exp\left(\frac{A}{\lambda}\right)
$$

优势越大，权重越大，模型越应该学习这个动作。

---

## 14. Bellman Equation 贝尔曼方程

贝尔曼方程是强化学习里价值函数的递推关系。

对 Q 函数来说，可以写成：

$$
Q(s,a)=r+\gamma\mathbb{E}\left[Q(s_{\text{next}},a_{\text{next}})\right]
$$

它表达的是：

$$
\text{当前动作价值}=\text{当前奖励}+\text{下一步价值}
$$

TD target 就是贝尔曼方程在训练里的具体目标值：

$$
y=r+\gamma Q_{\text{target}}(s_{\text{next}},a_{\text{next}})
$$

所以可以这样理解：

> Bellman equation 是理论关系；TD target 是训练时拿来拟合的目标。

---

## 15. Policy 策略

策略决定 agent 在某个状态下怎么选动作。

确定性策略：

$$
a=\pi(s)
$$

意思是：给定状态 $s$，直接输出一个动作。

DDPG、TD3 通常使用确定性策略。

随机策略：

$$
a\sim\pi(\cdot\mid s)
$$

意思是：给定状态 $s$，输出一个动作分布，再从中采样动作。

SAC 使用随机策略。

---

## 16. Actor 和 Critic

Actor 是策略网络，负责选动作：

$$
a=\pi(s)
$$

Critic 是价值网络，负责评价动作：

$$
Q(s,a)
$$

两者关系：

> actor 负责做决定；critic 负责打分；actor 根据 critic 的评分改进自己。

在 DDPG、TD3、SAC、CQL 里，常见训练方式是：

1. 先训练 critic，让它更会评价动作。
2. 再训练 actor，让它选择 critic 认为更好的动作。

---

## 17. Actor Loss

actor loss 用来更新策略。

不同算法的 actor loss 不一样。

### 最大化 Q 值

TD3 的 actor loss：

$$
\mathcal{L}_{\text{actor}}
=-\mathbb{E}\left[Q_1(s,\pi(s))\right]
$$

因为训练时是最小化 loss，所以加负号后，等价于最大化 Q。

### Q 最大化 + BC 约束

DDPG_NB、TD3_NB 常见形式：

$$
\mathcal{L}_{\text{actor}}
=-\lambda\,\mathbb{E}\left[Q(s,\pi(s))\right]
\;+\;\beta_{\text{BC}}\operatorname{MSE}\left(\pi(s),a\right)
$$

含义：

- 第一项让 actor 追求高 Q 动作。
- 第二项让 actor 不要离数据集动作太远。

### 优势加权行为克隆

DDPG 项目实现中的 actor loss：

$$
\mathcal{L}_{\text{actor}}
=\mathbb{E}\left[
w\,\operatorname{MSE}\left(\pi(s),a\right)
\right]
$$

其中：

$$
w=\exp\left(\frac{A}{\lambda}\right)
$$

优势越大，模仿权重越大。

### 加权最大似然

SAC 项目实现中的 AWAC 风格 actor loss：

$$
\mathcal{L}_{\text{actor}}
=-\mathbb{E}\left[
w\log\pi(a\mid s)
\right]
$$

含义：

让策略提高高优势数据动作的概率。

---

## 18. Critic Loss

critic loss 通常是 TD error 的 MSE。

单 critic：

$$
\mathcal{L}_{\text{critic}}
=\operatorname{MSE}\left(Q(s,a),y\right)
$$

双 critic：

$$
\mathcal{L}_{\text{critic}}
=\operatorname{MSE}\left(Q_1(s,a),y\right)
+\operatorname{MSE}\left(Q_2(s,a),y\right)
$$

其中 $y$ 就是 TD target。

所以 critic loss 的核心问题永远是：

> 我现在估计的 Q 值，和 TD target 差多少？

---

## 19. Target Network 目标网络

target network 是一份更新更慢的网络副本。

例如：

$$
Q_{\text{target}},\quad \pi_{\text{target}}
$$

它们用于计算 TD target：

$$
y=r+\gamma Q_{\text{target}}\left(s_{\text{next}},\pi_{\text{target}}(s_{\text{next}})\right)
$$

为什么不用当前正在训练的网络直接算？

因为如果目标值也跟着当前网络剧烈变化，训练会非常不稳定。

target network 的作用：

让 TD target 变化慢一点，训练更稳定。

---

## 20. Soft Update 软更新

target network 通常不是每次直接复制，而是慢慢靠近当前网络。

公式：

$$
\theta_{\text{target}}
\leftarrow
\tau\theta+(1-\tau)\theta_{\text{target}}
$$

其中：

- $\theta$：当前网络参数
- $\theta_{\text{target}}$：目标网络参数
- $\tau$：软更新系数

如果：

$$
\tau=0.005
$$

表示目标网络每次只更新一点点。

---

## 21. Double Q / Twin Critics

双 critic 指同时训练两个 Q 网络：

$$
Q_1(s,a),\quad Q_2(s,a)
$$

计算 TD target 时取较小值：

$$
\min(Q_{1,\text{target}},Q_{2,\text{target}})
$$

目的：

减少 Q 值过估计。

为什么会过估计？

因为 actor 会主动找 critic 认为价值高的动作。如果 critic 错把某些动作估得很高，actor 就会钻这个空子，导致策略变差。

双 critic 取最小值，相当于更保守地评价动作。

---

## 22. Target Policy Smoothing

TD3、DDPG_NB、CQL_NB 等实现中常看到：

$$
\begin{aligned}
\epsilon
&=\operatorname{clip}\left(\mathcal{N}(0,\sigma_{\text{policy}}),-c,c\right) \\
a_{\text{next}}
&=\operatorname{clip}\left(\pi_{\text{target}}(s_{\text{next}})+\epsilon,-1,1\right)
\end{aligned}
$$

这叫目标策略平滑。

作用：

不要只让 critic 学某一个精确动作点，而是学动作附近一小片区域的价值。

好处：

- 降低 Q 函数对动作小变化的敏感性
- 减少过拟合
- 让 TD target 更平滑

---

## 23. Policy Delay 延迟策略更新

TD3、DDPG、DDPG_NB、TD3_NB 里常见：

$$
\text{policy\_delay}=2
$$

意思是：

critic 更新 2 次，actor 更新 1 次。

为什么要延迟更新 actor？

因为 actor 是根据 critic 的评价来更新的。如果 critic 还不准，actor 太频繁更新，就会追着错误 Q 值跑。

所以：

先让 critic 稳一点，再更新 actor。

---

## 24. Entropy 熵

SAC 里有熵的概念。

熵表示策略的随机程度。

如果策略很确定：

每次几乎都选同一个动作。

熵低。

如果策略比较随机：

会尝试多种动作。

熵高。

SAC 的目标里会出现：

$$
-\alpha\log\pi(a\mid s)
$$

它鼓励策略保留一定随机性。

---

## 25. Alpha 温度参数

SAC 中的 $\alpha$ 控制熵项强度。

在 TD target 里：

$$
y=r+\gamma\left(\min(Q_1,Q_2)-\alpha\log\pi(a\mid s)\right)
$$

在 actor loss 里：

$$
\mathcal{L}_{\text{actor}}
=\mathbb{E}\left[\alpha\log\pi(a\mid s)-Q(s,a)\right]
$$

含义：

- $\alpha$ 大：更鼓励随机性，策略更保守、更探索
- $\alpha$ 小：更重视 Q 值，策略更确定

如果 `learn_alpha=True`，代码会自动学习 $\alpha$。

---

## 26. On-policy 和 Off-policy

On-policy：

用当前策略采集的数据，训练当前策略。

典型特点：

- 数据新鲜
- 训练稳定
- 样本利用率低

Off-policy：

可以用旧策略或其他策略采集的数据，训练当前策略。

典型特点：

- 样本利用率高
- 可以使用 replay buffer 或离线数据
- 容易出现分布偏移问题

DDPG、TD3、SAC、CQL 都属于 off-policy 思路。

---

## 27. Offline RL 离线强化学习

离线强化学习指：

只用已有数据集训练，不再和环境交互采样。

数据集里已经有：

$$
(s,a,r,s_{\text{next}},d)
$$

训练时不能再试新动作。

最大问题是：

actor 可能生成数据集里没见过的动作，critic 对这些动作估计不准。

这叫分布外动作问题，也常叫 OOD action 问题。

---

## 28. Behavior Policy 行为策略

behavior policy 是采集离线数据的策略。

离线数据里的动作 $a$ 来自 behavior policy：

$$
a\sim\pi_{\text{behavior}}(\cdot\mid s)
$$

而训练出来的新策略是：

$$
\pi(\cdot\mid s)
$$

两者可能不同。

离线 RL 的风险是：

新策略 $\pi$ 可能走到 behavior policy 很少覆盖的动作区域。

这会让 critic 的 Q 估计不可靠。

---

## 29. Distribution Shift 分布偏移

分布偏移指训练数据分布和模型实际使用时遇到的分布不同。

在离线 RL 中：

训练数据动作来自数据集；actor 更新后可能输出数据集中很少出现的动作。

这样 critic 对这些动作没有充分训练，Q 值可能乱估。

所以离线 RL 常常加入保守约束：

- BC 约束
- CQL penalty
- Advantage-weighted BC
- TD3+BC
- SAC+BC

---

## 30. OOD Action 分布外动作

OOD 是 Out-of-Distribution。

OOD action 指：

数据集中很少出现，甚至没有出现过的动作。

critic 对 OOD action 的 Q 值可能不准。

如果错误地把某个 OOD action 估得很高，actor 就可能被吸引过去。

这就是离线强化学习中很常见的失败模式。

---

## 31. BC 行为克隆

BC 是 Behavior Cloning。

核心 loss：

$$
\mathcal{L}_{\text{BC}}
=\operatorname{MSE}\left(\pi(s),a\right)
$$

含义：

让 actor 输出动作接近数据集动作。

优点：

- 简单稳定
- 不依赖 Q 估计
- 不容易产生 OOD action

缺点：

- 只会模仿数据
- 如果数据里有差动作，也会学进去
- 不能主动利用 reward 做策略改进

---

## 32. BC Regularization

BC regularization 是在强化学习 actor loss 中加入行为克隆约束。

常见形式：

$$
\mathcal{L}_{\text{actor}}
=\mathcal{L}_{\text{RL}}
\;+\;
\beta_{\text{BC}}\operatorname{MSE}\left(\pi(s),a\right)
$$

作用：

让策略在追求高 Q 的同时，不要偏离离线数据太远。

$\beta_{\text{BC}}$ 越大，策略越保守。

---

## 33. CQL Penalty

CQL 的核心思想是保守估计 Q 值。

它希望：

数据动作的 Q 值相对高；随机动作、策略动作等非数据动作的 Q 值不要虚高。

常见 penalty：

$$
\mathcal{L}_{\text{CQL}}
=\operatorname{logsumexp}\left(
Q(s,a_{\text{random}}),
Q(s,\pi(s))
\right)
-Q(s,a_{\text{data}})
$$

直观理解：

如果模型把随机动作或策略动作估得太高，就惩罚它；如果模型给数据动作较高 Q，就相对鼓励。

这样可以缓解 OOD action 的 Q 过估计。

---

## 34. AWAC / Advantage-Weighted 思想

AWAC 的核心思想是：

不是模仿所有数据动作，而是更重视比当前策略更好的动作。

先算优势：

$$
A=Q(s,a_{\text{data}})-Q(s,\pi(s))
$$

再算权重：

$$
w=\exp\left(\frac{A}{\lambda}\right)
$$

然后加权模仿：

$$
\mathcal{L}
=\mathbb{E}\left[w\,\mathcal{L}_{\text{BC}}\right]
$$

或者对随机策略使用加权最大似然：

$$
\mathcal{L}
=-\mathbb{E}\left[w\log\pi(a_{\text{data}}\mid s)\right]
$$

解释：

- $A>0$：数据动作比当前策略动作好，应该多学
- $A<0$：数据动作不如当前策略动作，少学

---

## 35. Reward Scale 奖励标准化

项目里 `reward_scale=True` 时，会做：

$$
r_{\text{norm}}
=\frac{r-\mu_r}{\sigma_r}
$$

然后 TD target 使用 $r_{\text{norm}}$：

$$
y=r_{\text{norm}}+\gamma Q_{\text{future}}
$$

作用：

让 reward 和 Q 值的数值尺度更稳定。

如果 reward 绝对值很大，TD target 也会很大，critic 训练可能不稳定。

---

## 36. Batch、Epoch、Gradient

batch 是一次训练拿出来的一小批样本。

$$
\text{batch}=\{(s_i,a_i,r_i,s_{i+1},d_i)\}_{i=1}^{B}
$$

epoch 是完整遍历一遍训练集。

gradient 是 loss 对模型参数的导数。

训练神经网络时：

$$
\text{计算 loss}\rightarrow\text{反向传播 gradient}\rightarrow\text{optimizer 更新参数}
$$

强化学习里也是一样，只是 loss 的构造比普通监督学习复杂。

---

## 37. Gradient Clipping 梯度裁剪

梯度裁剪用来限制梯度最大范数。

例如：

$$
\text{grad\_clip}=5.0
$$

作用：

防止某次更新的梯度太大，把网络参数冲坏。

在 critic target 波动较大时，梯度裁剪能提升训练稳定性。

---

## 38. Weight Decay 权重衰减

weight decay 是一种 L2 正则。

作用：

限制模型参数过大，降低过拟合。

在项目中通常作为 optimizer 的参数出现：

$$
\text{weight\_decay}=10^{-4}
$$

---

## 39. State Normalization 状态标准化

项目中通常会对状态做：

$$
s_{\text{norm}}
=\frac{s-\mu_{\text{obs}}}{\sigma_{\text{obs}}}
$$

作用：

让不同维度的状态处在相近数值尺度上。

这样神经网络更容易训练。

注意：

> 状态标准化影响网络输入；reward 标准化影响 TD target。

---

## 40. 常见概念关系图

可以把 actor-critic 的训练理解成：

1. 数据样本：

$$
(s,a,r,s_{\text{next}},d)
$$

2. 计算 TD target：

$$
y
$$

3. 训练 critic：

$$
Q(s,a)\rightarrow y
$$

4. critic 给 actor 的动作打分。

5. 训练 actor：选择更高价值或更接近数据的动作。

critic 的主线：

$$
\text{TD target}\rightarrow\text{TD error}\rightarrow\text{critic loss}
$$

actor 的主线：

$$
\text{Q value / log probability / BC constraint}
\rightarrow
\text{actor loss}
$$

---

## 41. 项目中 10 个算法与概念对应

| 算法 | 重点概念 | 主要 loss 思路 |
|---|---|---|
| `BC` | 行为克隆 | 直接最小化 $\pi(s)$ 和数据动作 $a$ 的差距 |
| `BC_NB` | Return-weighted BC | 高回报轨迹样本权重更大 |
| `CQL` | TD target + CQL penalty | critic 拟合 TD target，同时压低非数据动作 Q |
| `CQL_NB` | CQL + target smoothing | 在 CQL 基础上加入目标动作扰动和扩展 penalty |
| `DDPG` | Advantage-weighted BC | 根据优势函数给 BC loss 加权 |
| `DDPG_NB` | Q 最大化 + BC | actor 同时追求高 Q 和贴近数据动作 |
| `SAC` | 熵正则 + AWAC | critic 用熵正则 TD target，actor 用优势加权 log probability |
| `SAC_NB` | SAC + BC | 标准 SAC actor loss 加行为克隆约束 |
| `TD3` | 双 critic + policy delay | critic 拟合 TD target，actor 最大化 $Q_1(s,\pi(s))$ |
| `TD3_NB` | TD3+BC | TD3 actor loss 加 BC 约束 |

---

## 42. 几组容易混淆的概念

### TD Target vs Q(s, a)

> TD target 是目标答案；$Q(s,a)$ 是 critic 当前预测。

训练 critic 时，就是让 $Q(s,a)$ 靠近 TD target。

### TD Target vs TD Error

> TD target 是 $y$；TD error 是 $y-Q(s,a)$。

TD target 是目标，TD error 是差距。

### Reward vs Return

> reward 是一步奖励；return 是长期累计奖励。

强化学习优化的是长期 return。

### V(s) vs Q(s, a)

> $V(s)$ 评价状态；$Q(s,a)$ 评价状态下的具体动作。

### Actor Loss vs Critic Loss

> critic loss 训练评价器；actor loss 训练决策器。

critic 学会打分，actor 学会选高分动作。

### Online RL vs Offline RL

> online RL 可以继续和环境交互；offline RL 只能用已有数据集。

离线 RL 更需要防止策略偏离数据分布。

---

## 43. 用一句话串起来

强化学习训练可以这样理解：

reward 告诉模型这一步好不好；return 表示长期总收益；$Q$ 函数估计某个动作的长期收益；TD target 给 $Q$ 函数提供训练目标；TD error 衡量 $Q$ 估计错了多少；critic 通过 TD error 学会评价动作；actor 根据 critic 的评价改进策略；离线 RL 还要用 BC、CQL、优势加权等方法，防止 actor 跑到数据集没覆盖的危险动作区域。

# 10 个离线强化学习算法的 Loss 解释

本文按项目 `AIcodes/` 中 10 个训练脚本的实际实现整理 loss。算法包括：

- `BC`
- `BC_NB`
- `CQL`
- `CQL_NB`
- `DDPG`
- `DDPG_NB`
- `SAC`
- `SAC_NB`
- `TD3`
- `TD3_NB`

## 统一符号

离线数据集中的一条样本记为：

$$
(s,a,r,s_{\text{next}},d)
$$

含义如下：

- $s$：当前状态
- $a$：数据集里的动作，也叫 data action
- $r$：奖励
- $s_{\text{next}}$：下一状态
- $d$：终止标记，终止时为 1，否则为 0
- $\pi(s)$：actor 在状态 $s$ 下输出或采样的动作
- $Q_1,Q_2$：双 critic
- $Q_{1,\text{target}},Q_{2,\text{target}}$：目标 critic
- $\gamma$：折扣因子
- $\alpha$：SAC 温度参数
- $\beta_{\text{BC}}$：行为克隆约束系数，对应代码中的 `bc_coef`
- $\operatorname{MSE}(x,y)$：均方误差

如果 `reward_scale=True`，代码会先对 reward 做标准化：

$$
r_{\text{norm}}=\frac{r-\mu_r}{\sigma_r}
$$

之后 TD target 中使用的是标准化后的 $r_{\text{norm}}$。

---

## 1. BC

代码文件：`AIcodes/BC.py`

BC 只做行为克隆，不训练 critic。它直接让 actor 输出的动作接近离线数据里的动作。

预测动作：

$$
a_{\text{pred}}=\pi(s)
$$

默认 loss 是 `SmoothL1Loss`，如果使用 `--use-mse`，则改成 `MSELoss`。

默认形式：

$$
\mathcal{L}_{\text{BC}}
=\operatorname{SmoothL1}(a_{\text{pred}},a)
$$

如果切换为 MSE：

$$
\mathcal{L}_{\text{BC}}
=\operatorname{MSE}(a_{\text{pred}},a)
$$

解释：

- $a_{\text{pred}}$ 是模型预测动作。
- $a$ 是数据集动作。
- loss 越小，说明模型越像数据集里的行为。
- BC 不关心 reward，也不估计 Q 值，只做监督学习。

---

## 2. BC_NB

代码文件：`AIcodes/BC_NB.py`

BC_NB 是 Return-weighted BC。它仍然是行为克隆，但会给高回报 episode 中的样本更大权重。

先计算每个 episode 的总回报：

$$
R_{\text{episode}}=\sum_t r_t
$$

对 episode return 做标准化：

$$
R_{\text{norm}}
=\frac{R_{\text{episode}}-\operatorname{mean}(R_{\text{episode}})}
{\operatorname{std}(R_{\text{episode}})}
$$

再计算 episode 权重：

$$
\begin{aligned}
w_{\text{episode}}
&=\exp\left(\frac{R_{\text{norm}}}{T_{\text{weight}}}\right) \\
w_{\text{episode}}
&=\operatorname{clip}\left(w_{\text{episode}},w_{\min},w_{\max}\right)
\end{aligned}
$$

每条样本继承所在 episode 的权重，并再除以平均权重做归一化：

$$
w_{\text{sample}}
=\frac{w_{\text{episode}}}{\operatorname{mean}(w_{\text{episode}})}
$$

每条样本的动作误差为：

$$
\ell_i=\operatorname{MSE}\left(\pi(s_i),a_i\right)
$$

或者：

$$
\ell_i=\operatorname{SmoothL1}\left(\pi(s_i),a_i\right)
$$

最终 loss：

$$
\mathcal{L}_{\text{BC\_NB}}
=\mathbb{E}_i\left[w_{\text{sample},i}\ell_i\right]
$$

解释：

- 普通 BC 对所有样本一视同仁。
- BC_NB 会更重视高回报轨迹里的动作。
- `weight_temperature` 越小，权重差异越明显。
- `min_weight` 和 `max_weight` 防止权重过小或过大。

---

## 3. CQL

代码文件：`AIcodes/CQL.py`

CQL 同时训练 critic 和 actor。critic loss 由 TD 误差和 CQL 保守惩罚组成；actor loss 由 Q 最大化和 BC 约束组成。

### Critic TD Target

$$
\begin{aligned}
a_{\text{next}}&=\pi(s_{\text{next}}) \\
y&=r+\gamma(1-d)
\min\left(
Q_{1,\text{target}}(s_{\text{next}},a_{\text{next}}),
Q_{2,\text{target}}(s_{\text{next}},a_{\text{next}})
\right)
\end{aligned}
$$

### Critic MSE Loss

$$
\mathcal{L}_{\text{mse}}
=\operatorname{MSE}\left(Q_1(s,a),y\right)
+\operatorname{MSE}\left(Q_2(s,a),y\right)
$$

### CQL Penalty

CQL 会额外比较随机动作、当前策略动作和数据动作的 Q 值。

$$
\begin{aligned}
q_{\text{rand}}&=Q(s,a_{\text{random}}) \\
q_{\pi}&=Q(s,\pi(s)) \\
q_{\text{data}}&=Q(s,a)
\end{aligned}
$$

单个 critic 的 CQL 惩罚：

$$
\mathcal{L}_{\text{cql-one}}
=\operatorname{logsumexp}(q_{\text{rand}},q_{\pi})
-\mathbb{E}\left[q_{\text{data}}\right]
$$

两个 critic 合起来：

$$
\mathcal{L}_{\text{cql}}
=\lambda_{\text{cql}}
\left(\mathcal{L}_{\text{cql},1}+\mathcal{L}_{\text{cql},2}\right)
$$

critic 总 loss：

$$
\mathcal{L}_{\text{critic}}
=\mathcal{L}_{\text{mse}}+\mathcal{L}_{\text{cql}}
$$

解释：

- $\mathcal{L}_{\text{mse}}$ 让 Q 值拟合 TD target。
- $\mathcal{L}_{\text{cql}}$ 会压低随机动作和策略动作的过高 Q 值。
- 它鼓励模型更相信数据集动作，减少离线 RL 中的 Q 过估计。

### Actor Loss

$$
\begin{aligned}
a_{\text{pred}}&=\pi(s) \\
q_{\pi}&=Q_1(s,a_{\text{pred}}) \\
\lambda&=\frac{\alpha_{\text{BC}}}{\mathbb{E}[|q_{\pi}|]+\varepsilon} \\
\mathcal{L}_{\text{actor}}
&=-\lambda\,\mathbb{E}[q_{\pi}]
\;+\;\beta_{\text{BC}}\operatorname{MSE}(a_{\text{pred}},a)
\end{aligned}
$$

解释：

- $-\lambda\,\mathbb{E}[q_{\pi}]$：最小化这个项等价于最大化 Q 值。
- $\beta_{\text{BC}}\operatorname{MSE}(a_{\text{pred}},a)$：限制 actor 不要离数据集动作太远。
- $\lambda$ 会根据 Q 值大小自动缩放，避免 Q 项数值过大或过小。

---

## 4. CQL_NB

代码文件：`AIcodes/CQL_NB.py`

CQL_NB 在 CQL 的基础上加入了目标动作扰动和更丰富的 CQL penalty。

### Critic TD Target

先给下一状态动作加噪声：

$$
\begin{aligned}
\epsilon
&=\operatorname{clip}\left(\mathcal{N}(0,\sigma_{\text{policy}}),-c,c\right) \\
a_{\text{next}}
&=\operatorname{clip}\left(\pi(s_{\text{next}})+\epsilon,-1,1\right)
\end{aligned}
$$

TD target：

$$
y=r+\gamma(1-d)
\min\left(
Q_{1,\text{target}}(s_{\text{next}},a_{\text{next}}),
Q_{2,\text{target}}(s_{\text{next}},a_{\text{next}})
\right)
$$

### Critic MSE Loss

$$
\mathcal{L}_{\text{mse}}
=\operatorname{MSE}\left(Q_1(s,a),y\right)
+\operatorname{MSE}\left(Q_2(s,a),y\right)
$$

### CQL_NB Penalty

CQL_NB 的 penalty 比 CQL 多看了下一状态策略动作：

$$
\begin{aligned}
q_{\text{rand}}&=Q(s,a_{\text{random}}) \\
q_{\pi}&=Q(s,\pi(s)) \\
q_{\text{next-}\pi}&=Q(s_{\text{next}},\pi(s_{\text{next}})+\epsilon) \\
q_{\text{data}}&=Q(s,a)
\end{aligned}
$$

单个 critic：

$$
\mathcal{L}_{\text{cql-one}}
=\operatorname{logsumexp}(q_{\text{rand}},q_{\pi},q_{\text{next-}\pi})
-\mathbb{E}\left[q_{\text{data}}\right]
$$

critic 总 loss：

$$
\mathcal{L}_{\text{critic}}
=\mathcal{L}_{\text{mse}}
\;+\;
\lambda_{\text{cql}}
\left(\mathcal{L}_{\text{cql},1}+\mathcal{L}_{\text{cql},2}\right)
$$

解释：

- 普通 CQL 主要压制当前状态下随机动作和策略动作的过高 Q。
- CQL_NB 还把下一状态的策略动作也纳入惩罚。
- 目标动作扰动让 TD target 更平滑，降低对某个精确动作点的过拟合。

### Actor Loss

CQL_NB 的 actor loss 和 CQL 一样：

$$
\begin{aligned}
a_{\text{pred}}&=\pi(s) \\
q_{\pi}&=Q_1(s,a_{\text{pred}}) \\
\lambda&=\frac{\alpha_{\text{BC}}}{\mathbb{E}[|q_{\pi}|]+\varepsilon} \\
\mathcal{L}_{\text{actor}}
&=-\lambda\,\mathbb{E}[q_{\pi}]
\;+\;\beta_{\text{BC}}\operatorname{MSE}(a_{\text{pred}},a)
\end{aligned}
$$

---

## 5. DDPG

代码文件：`AIcodes/DDPG.py`

项目中的 DDPG 不是原始 DDPG 的普通 `-Q` actor loss，而是用了优势加权行为克隆。

### Critic TD Target

先做目标动作平滑：

$$
\begin{aligned}
\epsilon
&=\operatorname{clip}\left(\mathcal{N}(0,\sigma_{\text{policy}}),-c,c\right) \\
a_{\text{next}}
&=\operatorname{clip}\left(\pi_{\text{target}}(s_{\text{next}})+\epsilon,-1,1\right)
\end{aligned}
$$

TD target：

$$
y=r+\gamma(1-d)
\min\left(
Q_{1,\text{target}}(s_{\text{next}},a_{\text{next}}),
Q_{2,\text{target}}(s_{\text{next}},a_{\text{next}})
\right)
$$

critic loss：

$$
\mathcal{L}_{\text{critic}}
=\operatorname{MSE}\left(Q_1(s,a),y\right)
+\operatorname{MSE}\left(Q_2(s,a),y\right)
$$

解释：

- 双 critic 取较小值，降低 Q 过估计。
- 目标动作加噪声，降低 critic 对动作点的过拟合。

### Actor Loss

先计算数据动作和当前策略动作的相对优势：

$$
\begin{aligned}
q_{\text{data}}
&=\min\left(Q_1(s,a),Q_2(s,a)\right) \\
q_{\pi}
&=\min\left(Q_1(s,\pi(s)),Q_2(s,\pi(s))\right) \\
A&=q_{\text{data}}-q_{\pi}
\end{aligned}
$$

优势越大，说明数据动作 $a$ 比当前策略动作 $\pi(s)$ 更好。

计算权重：

$$
\begin{aligned}
w&=\exp\left(\frac{A}{\lambda_{\text{AW}}}\right) \\
w&=\operatorname{clip}\left(w,\text{max}=w_{\max}\right)
\end{aligned}
$$

actor loss：

$$
\mathcal{L}_{\text{actor}}
=\mathbb{E}\left[
w\cdot \left\|\pi(s)-a\right\|_2^2
\right]
$$

解释：

- DDPG 这里本质是 Advantage-weighted BC。
- 如果某条数据动作明显优于当前策略动作，它的 `w` 更大，actor 更强烈地模仿它。
- 如果数据动作不如当前策略动作，权重变小，actor 少学它。
- actor 每隔 `policy_delay` 次 critic 更新才更新一次。

---

## 6. DDPG_NB

代码文件：`AIcodes/DDPG_NB.py`

DDPG_NB 的 critic 和 DDPG 类似，actor 改成 “Q 最大化 + BC 约束”。

### Critic Loss

目标动作：

$$
\begin{aligned}
\epsilon
&=\operatorname{clip}\left(\mathcal{N}(0,\sigma_{\text{policy}}),-c,c\right) \\
a_{\text{next}}
&=\operatorname{clip}\left(\pi_{\text{target}}(s_{\text{next}})+\epsilon,-1,1\right)
\end{aligned}
$$

TD target：

$$
y=r+\gamma(1-d)
\min\left(
Q_{1,\text{target}}(s_{\text{next}},a_{\text{next}}),
Q_{2,\text{target}}(s_{\text{next}},a_{\text{next}})
\right)
$$

critic loss：

$$
\mathcal{L}_{\text{critic}}
=\operatorname{MSE}\left(Q_1(s,a),y\right)
+\operatorname{MSE}\left(Q_2(s,a),y\right)
$$

### Actor Loss

$$
\begin{aligned}
a_{\text{pred}}&=\pi(s) \\
q_{\pi}&=Q_1(s,a_{\text{pred}}) \\
\lambda&=\frac{\alpha_{\text{BC}}}{\mathbb{E}[|q_{\pi}|]+\varepsilon}
\end{aligned}
$$

Q 最大化项：

$$
\mathcal{L}_q=-\lambda\,\mathbb{E}[q_{\pi}]
$$

BC 约束项：

$$
\mathcal{L}_{\text{BC}}
=\beta_{\text{BC}}\operatorname{MSE}(a_{\text{pred}},a)
$$

总 actor loss：

$$
\mathcal{L}_{\text{actor}}
=\mathcal{L}_q+\mathcal{L}_{\text{BC}}
$$

解释：

- $\mathcal{L}_q$ 让 actor 追求更高 Q 值。
- $\mathcal{L}_{\text{BC}}$ 让 actor 不要偏离离线数据动作太远。
- $\alpha_{\text{BC}}$ 通过 $\lambda$ 控制 Q 项尺度。
- $\beta_{\text{BC}}$ 控制行为克隆约束强度。
- actor 同样按 `policy_delay` 延迟更新。

---

## 7. SAC

代码文件：`AIcodes/SAC.py`

项目中的 SAC critic 使用 SAC 的熵正则 TD target，但 actor 使用的是 AWAC 风格的加权最大似然。

### Critic TD Target

从策略中采样下一动作：

$$
\begin{aligned}
a_{\text{next}}&\sim\pi(\cdot\mid s_{\text{next}}) \\
\log p_{\text{next}}&=\log\pi(a_{\text{next}}\mid s_{\text{next}})
\end{aligned}
$$

TD target：

$$
y=r+\gamma(1-d)
\left[
\min\left(
Q_{1,\text{target}}(s_{\text{next}},a_{\text{next}}),
Q_{2,\text{target}}(s_{\text{next}},a_{\text{next}})
\right)
-\alpha\log p_{\text{next}}
\right]
$$

critic loss：

$$
\begin{aligned}
\mathcal{L}_{\text{critic1}}
&=\operatorname{MSE}\left(Q_1(s,a),y\right) \\
\mathcal{L}_{\text{critic2}}
&=\operatorname{MSE}\left(Q_2(s,a),y\right)
\end{aligned}
$$

解释：

- $-\alpha\log p_{\text{next}}$ 是 SAC 的熵项。
- 它鼓励策略保留一定随机性，不要过早变得确定。

### Actor Loss

先计算数据动作相对当前策略动作的优势：

$$
\begin{aligned}
q_{\text{data}}
&=\min\left(Q_1(s,a),Q_2(s,a)\right) \\
a_{\pi}&\sim\pi(\cdot\mid s) \\
v_{\pi}
&=\min\left(Q_1(s,a_{\pi}),Q_2(s,a_{\pi})\right) \\
A&=q_{\text{data}}-v_{\pi}
\end{aligned}
$$

优势加权：

$$
\begin{aligned}
w&=\exp\left(\frac{A}{\lambda_{\text{AWAC}}}\right) \\
w&=\operatorname{clip}\left(w,\text{max}=w_{\max}\right)
\end{aligned}
$$

计算数据动作在当前策略下的 log probability：

$$
\log p_{\text{data}}=\log\pi(a\mid s)
$$

actor loss：

$$
\mathcal{L}_{\text{actor}}
=-\mathbb{E}\left[w\log p_{\text{data}}\right]
$$

解释：

- 这是加权最大似然。
- 高优势数据动作的权重大，策略会更努力提高这些动作的概率。
- 低优势动作权重小，策略不会盲目模仿。

### Alpha Loss

如果 `learn_alpha=True`，温度参数也会学习：

$$
\mathcal{L}_{\alpha}
=-\mathbb{E}\left[
\log\alpha\cdot
\operatorname{detach}\left(\log\pi(a\mid s)+\mathcal{H}_{\text{target}}\right)
\right]
$$

解释：

- $\alpha$ 控制探索熵的强度。
- 这个 loss 让策略熵接近目标熵 $\mathcal{H}_{\text{target}}$。

---

## 8. SAC_NB

代码文件：`AIcodes/SAC_NB.py`

SAC_NB 使用标准 SAC actor 目标，并额外加入 BC 约束。

### Critic Loss

TD target 与 SAC 一样：

$$
\begin{aligned}
a_{\text{next}}&\sim\pi(\cdot\mid s_{\text{next}}) \\
\log p_{\text{next}}&=\log\pi(a_{\text{next}}\mid s_{\text{next}}) \\
y&=r+\gamma(1-d)
\left[
\min\left(
Q_{1,\text{target}}(s_{\text{next}},a_{\text{next}}),
Q_{2,\text{target}}(s_{\text{next}},a_{\text{next}})
\right)
-\alpha\log p_{\text{next}}
\right]
\end{aligned}
$$

critic loss：

$$
\begin{aligned}
\mathcal{L}_{\text{critic1}}
&=\operatorname{MSE}\left(Q_1(s,a),y\right) \\
\mathcal{L}_{\text{critic2}}
&=\operatorname{MSE}\left(Q_2(s,a),y\right)
\end{aligned}
$$

### Actor Loss

从当前策略采样动作：

$$
\begin{aligned}
a_{\pi}&\sim\pi(\cdot\mid s) \\
\log p_{\pi}&=\log\pi(a_{\pi}\mid s) \\
q_{\pi}&=\min\left(Q_1(s,a_{\pi}),Q_2(s,a_{\pi})\right)
\end{aligned}
$$

SAC 策略项：

$$
\mathcal{L}_{\text{SAC}}
=\mathbb{E}\left[\alpha\log p_{\pi}-q_{\pi}\right]
$$

BC 约束项：

$$
\mathcal{L}_{\text{BC}}
=\operatorname{MSE}(a_{\pi},a)
$$

总 actor loss：

$$
\mathcal{L}_{\text{actor}}
=\mathcal{L}_{\text{SAC}}
\beta_{\text{BC}}\mathcal{L}_{\text{BC}}
$$

解释：

- $\mathcal{L}_{\text{SAC}}$ 让策略在最大化 Q 的同时保留熵。
- $\mathcal{L}_{\text{BC}}$ 把采样动作拉回离线数据动作附近。
- $\beta_{\text{BC}}$ 越大，策略越保守，越像数据集。

### Alpha Loss

$$
\mathcal{L}_{\alpha}
=-\mathbb{E}\left[
\log\alpha\cdot
\operatorname{detach}\left(\log\pi(a\mid s)+\mathcal{H}_{\text{target}}\right)
\right]
$$

---

## 9. TD3

代码文件：`AIcodes/TD3.py`

TD3 使用双 critic、目标动作平滑和延迟 actor 更新。

### Critic TD Target

目标动作：

$$
\begin{aligned}
\epsilon
&=\operatorname{clip}\left(\mathcal{N}(0,\sigma_{\text{policy}}),-c,c\right) \\
a_{\text{next}}
&=\operatorname{clip}\left(
\pi_{\text{target}}(s_{\text{next}})+\epsilon,
-a_{\max},a_{\max}
\right)
\end{aligned}
$$

TD target：

$$
y=r+\gamma(1-d)
\min\left(
Q_{1,\text{target}}(s_{\text{next}},a_{\text{next}}),
Q_{2,\text{target}}(s_{\text{next}},a_{\text{next}})
\right)
$$

critic loss：

$$
\mathcal{L}_{\text{critic}}
=\operatorname{MSE}\left(Q_1(s,a),y\right)
+\operatorname{MSE}\left(Q_2(s,a),y\right)
$$

解释：

- 双 critic 取最小值，减少过估计。
- 目标动作加噪声，平滑 target。

### Actor Loss

$$
\mathcal{L}_{\text{actor}}
=-\mathbb{E}\left[Q_1(s,\pi(s))\right]
$$

解释：

- 训练是最小化 loss。
- 因为前面有负号，所以最小化 $-Q$ 等价于最大化 $Q_1(s,\pi(s))$。
- actor 每隔 `policy_delay` 次 critic 更新才更新一次。

---

## 10. TD3_NB

代码文件：`AIcodes/TD3_NB.py`

TD3_NB 是 TD3+BC 风格：critic 和 TD3 类似，actor 同时包含 Q 最大化和 BC 约束。

### Critic Loss

如果开启 `reward_scale`，先标准化 reward：

$$
r_{\text{norm}}=\frac{r-\mu_r}{\sigma_r}
$$

目标动作：

$$
\begin{aligned}
\epsilon
&=\operatorname{clip}\left(\mathcal{N}(0,\sigma_{\text{policy}}),-c,c\right) \\
a_{\text{next}}
&=\operatorname{clip}\left(
\pi_{\text{target}}(s_{\text{next}})+\epsilon,
-a_{\max},a_{\max}
\right)
\end{aligned}
$$

TD target：

$$
y=r+\gamma(1-d)
\min\left(
Q_{1,\text{target}}(s_{\text{next}},a_{\text{next}}),
Q_{2,\text{target}}(s_{\text{next}},a_{\text{next}})
\right)
$$

critic loss：

$$
\mathcal{L}_{\text{critic}}
=\operatorname{MSE}\left(Q_1(s,a),y\right)
+\operatorname{MSE}\left(Q_2(s,a),y\right)
$$

### Actor Loss

$$
\begin{aligned}
a_{\text{pred}}&=\pi(s) \\
q_{\pi}&=Q_1(s,a_{\text{pred}}) \\
\lambda&=\frac{\alpha_{\text{BC}}}{\mathbb{E}[|q_{\pi}|]+\varepsilon}
\end{aligned}
$$

Q 最大化项：

$$
\mathcal{L}_q=-\lambda\,\mathbb{E}[q_{\pi}]
$$

BC 约束项：

$$
\mathcal{L}_{\text{BC}}
=\beta_{\text{BC}}\operatorname{MSE}(a_{\text{pred}},a)
$$

总 actor loss：

$$
\mathcal{L}_{\text{actor}}
=\mathcal{L}_q+\mathcal{L}_{\text{BC}}
$$

解释：

- $\mathcal{L}_q$ 继承 TD3 的思想，让 actor 选择高 Q 动作。
- $\mathcal{L}_{\text{BC}}$ 是离线强化学习里的保守约束，让动作不要偏离数据集太远。
- $\alpha_{\text{BC}}$ 控制 Q 项的归一化尺度。
- $\beta_{\text{BC}}$ 控制模仿数据动作的强度。

---

## 总览表

| 算法 | Critic Loss | Actor / Policy Loss | 主要约束思想 |
|---|---|---|---|
| `BC` | 无 | $\operatorname{SmoothL1/MSE}(\pi(s),a)$ | 直接模仿数据动作 |
| `BC_NB` | 无 | $\mathbb{E}[w\cdot\operatorname{SmoothL1/MSE}(\pi(s),a)]$ | 高回报轨迹权重更大 |
| `CQL` | TD MSE + CQL penalty | $-\lambda Q+\beta_{\text{BC}}\operatorname{MSE}$ | 压低 OOD 动作 Q 值 |
| `CQL_NB` | TD MSE + 扩展 CQL penalty | $-\lambda Q+\beta_{\text{BC}}\operatorname{MSE}$ | 加入目标动作扰动和下一状态动作惩罚 |
| `DDPG` | 双 critic TD MSE | $\mathbb{E}[w_{\text{adv}}\operatorname{MSE}(\pi(s),a)]$ | 优势加权行为克隆 |
| `DDPG_NB` | 双 critic TD MSE | $-\lambda Q+\beta_{\text{BC}}\operatorname{MSE}$ | Q 最大化 + BC 约束 |
| `SAC` | 熵正则 TD MSE | $-\mathbb{E}[w_{\text{adv}}\log\pi(a\mid s)]$ | AWAC 风格加权最大似然 |
| `SAC_NB` | 熵正则 TD MSE | $\mathcal{L}_{\text{SAC}}+\beta_{\text{BC}}\mathcal{L}_{\text{BC}}$ | SAC 目标 + BC 约束 |
| `TD3` | 双 critic TD MSE | $-\mathbb{E}[Q_1(s,\pi(s))]$ | 最大化 critic 评价 |
| `TD3_NB` | 双 critic TD MSE | $-\lambda Q+\beta_{\text{BC}}\operatorname{MSE}$ | TD3+BC 风格保守 actor |

## 最容易混淆的点

1. `BC` 和 `BC_NB` 没有 critic，loss 就是动作监督误差。
2. `DDPG` 在本项目里不是普通 DDPG actor loss，而是优势加权 BC。
3. `SAC` 在本项目里 actor 用 AWAC 风格的加权 log probability，不是普通 SAC actor loss。
4. `DDPG_NB`、`TD3_NB`、`CQL`、`CQL_NB` 的 actor loss 都有类似结构：

$$
\mathcal{L}_{\text{actor}}
=-\lambda Q+\beta_{\text{BC}}\mathcal{L}_{\text{BC}}
$$

5. `CQL` 和 `CQL_NB` 的关键区别在 critic penalty：`CQL_NB` 多考虑了带噪声的下一状态策略动作。
6. `reward_scale` 不改变 loss 形式，但会改变 TD target 中 reward 的数值尺度，从而影响 critic 学到的 Q 值大小。

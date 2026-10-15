# 10 个算法策略总结

本文用于答辩复习，按项目 `AIcodes/` 目录中的 10 个算法脚本，总结每个算法采用的关键训练策略。这里的重点不是逐行解释代码，而是回答老师可能会问的：

```text
这个算法用了什么改进策略？
为什么要加这个策略？
它解决了离线强化学习中的什么问题？
```

## 1. 总览表

| 算法 | 代码文件 | 算法类型 | 用到的主要策略 | 策略作用 |
| --- | --- | --- | --- | --- |
| BC | `AIcodes/BC.py` | 行为克隆 | 监督学习式动作拟合、状态标准化、`tanh` 动作约束、SmoothL1/MSE 损失、余弦退火学习率 | 把离线强化学习先简化成动作预测任务，作为稳定基线 |
| BC_NB | `AIcodes/BC_NB.py` | 加权行为克隆 | 轨迹累计回报加权、逐样本 loss、权重归一化、余弦退火学习率 | 让高回报轨迹中的样本对训练影响更大，避免平均模仿所有轨迹 |
| CQL | `AIcodes/CQL.py` | 保守离线 Actor-Critic | 双 critic、target critic、TD 目标、CQL 保守惩罚项、actor 中加入 BC 约束、reward scaling、软更新、余弦退火学习率 | 压低数据分布外动作的虚高 Q 值，降低离线策略偏移风险 |
| CQL_NB | `AIcodes/CQL_NB.py` | 增强版 CQL | 更强 CQL penalty、下一状态策略动作、target smoothing noise、噪声裁剪、双 critic、BC 约束、reward scaling | 在 CQL 保守约束基础上加入目标动作平滑，让 Q 估计更稳定 |
| DDPG | `AIcodes/DDPG.py` | 离线增强 DDPG | 确定性 actor、双 critic、target network、目标动作噪声、延迟更新 actor、advantage-weighted BC、reward scaling、软更新 | 用优势权重选择更值得模仿的数据动作，减少策略偏离离线数据 |
| DDPG_NB | `AIcodes/DDPG_NB.py` | DDPG + BC 正则 | 双 critic、target network、目标动作噪声、延迟更新 actor、Q 项 + BC 项、Q 项自适应缩放、reward scaling、软更新 | 让 actor 既追求高 Q 动作，又不要离数据动作太远 |
| SAC | `AIcodes/SAC.py` | 离线 AWAC-SAC 风格 | 随机高斯 actor、双 critic、target critic、熵正则、自动 alpha、advantage 权重、加权 log probability、reward scaling、软更新 | 在保留 SAC 随机策略的同时，让策略更偏向高优势的数据动作 |
| SAC_NB | `AIcodes/SAC_NB.py` | SAC + BC 正则 | 随机高斯 actor、双 critic、target critic、熵正则、自动 alpha、SAC loss + BC loss、reward scaling、软更新 | 用 BC 正则约束随机策略，防止采样动作偏离离线数据分布 |
| TD3 | `AIcodes/TD3.py` | 双 Q 延迟策略算法 | 双 critic、取较小 Q 值、target policy smoothing、目标动作噪声、噪声裁剪、延迟更新 actor、target network、软更新、余弦退火学习率 | 减少 Q 值过估计，让 critic 先稳定再更新 actor |
| TD3_NB | `AIcodes/TD3_NB.py` | TD3 + BC 正则 | 双 critic、取较小 Q 值、target policy smoothing、延迟更新 actor、Q 项 + BC 项、Q 项自适应缩放、reward scaling、软更新 | 在 TD3 稳定化基础上加入离线数据约束，让策略优化更稳 |

## 2. 按策略分类记忆

| 策略 | 出现算法 | 简单解释 | 解决的问题 |
| --- | --- | --- | --- |
| 状态标准化 | 10 个算法基本都使用 | 用训练集均值和标准差规范输入状态 | 减少不同状态维度尺度差异，提升训练稳定性 |
| 余弦退火学习率 | BC、BC_NB、CQL、CQL_NB、DDPG、DDPG_NB、SAC、SAC_NB、TD3、TD3_NB | 学习率按余弦曲线逐渐下降 | 前期较快学习，后期更平稳收敛 |
| 行为克隆损失 | BC、BC_NB、CQL、CQL_NB、DDPG、DDPG_NB、SAC_NB、TD3_NB | 让策略输出动作接近数据集动作 | 防止策略脱离离线数据分布 |
| 样本/轨迹加权 | BC_NB | 根据 episode 累计回报给样本加权 | 让高回报轨迹更重要，低质量轨迹影响更小 |
| CQL penalty | CQL、CQL_NB | 对随机动作、策略动作等候选动作的 Q 值加保守惩罚 | 避免 critic 对数据外动作给出虚高估计 |
| 更强 penalty 覆盖 | CQL_NB | penalty 中额外考虑下一状态策略动作 | 约束更多可能的数据外动作 |
| 双 critic / 双 Q | CQL、CQL_NB、DDPG、DDPG_NB、SAC、SAC_NB、TD3、TD3_NB | 同时训练两个 critic，并在 target 中取较小值或共同约束 | 降低 Q 值过估计 |
| target network | CQL、CQL_NB、DDPG、DDPG_NB、SAC、SAC_NB、TD3、TD3_NB | 维护一套慢更新的目标网络 | 避免目标值随当前网络剧烈变化 |
| 软更新 | CQL、CQL_NB、DDPG、DDPG_NB、SAC、SAC_NB、TD3、TD3_NB | 用 `tau` 小步更新 target network | 让目标网络平滑跟随当前网络 |
| target policy smoothing | CQL_NB、DDPG、DDPG_NB、TD3、TD3_NB | 给 target action 加裁剪噪声 | 避免 critic 对某个尖锐动作点过度敏感 |
| 延迟更新 actor | DDPG、DDPG_NB、TD3、TD3_NB | critic 更新多次后，actor 才更新一次 | 让 actor 不要追随尚不稳定的 Q 估计 |
| advantage-weighted BC | DDPG、SAC | 根据优势值给数据动作加权 | 更重视高价值数据动作，而不是平均模仿 |
| Q 项 + BC 项 | DDPG_NB、TD3_NB | actor loss 同时包含高 Q 目标和 BC 约束 | 兼顾“动作价值高”和“别偏离数据” |
| 熵正则 | SAC、SAC_NB | 在目标中加入 `alpha * log_prob` | 保持策略随机性，避免过早确定化 |
| 自动 alpha | SAC、SAC_NB | 自动调节熵温度系数 | 平衡探索随机性和价值优化 |
| reward scaling | CQL、CQL_NB、DDPG、DDPG_NB、SAC、SAC_NB、TD3_NB 等 | 对 reward 做缩放或标准化 | 降低 reward 尺度对 TD target 的影响 |

## 3. Base 与 NB 的主要区别

| 算法组 | Base 版本重点 | NB 版本重点 | 一句话区别 |
| --- | --- | --- | --- |
| BC / BC_NB | 普通行为克隆，所有样本基本同等对待 | 根据轨迹累计回报给样本加权 | BC_NB 不再平均模仿，而是更重视高回报轨迹 |
| CQL / CQL_NB | 使用 CQL penalty 保守估计 Q 值 | penalty 覆盖更多动作，并加入 target smoothing | CQL_NB 在“保守”的基础上进一步强调“平滑” |
| DDPG / DDPG_NB | advantage-weighted BC，按优势值模仿数据动作 | 直接用 Q 项 + BC 项约束 actor | DDPG_NB 更明确地把“追求高 Q”和“贴近数据”合在 actor loss 中 |
| SAC / SAC_NB | AWAC-SAC 风格，用优势权重提高数据动作概率 | SAC loss 外加 BC loss | SAC_NB 更直接地用 BC 正则限制随机策略 |
| TD3 / TD3_NB | 双 Q、目标动作平滑、延迟策略更新 | 保留 TD3 技巧，同时加入 BC 正则和 reward scaling | TD3_NB 是更适合离线数据的 TD3+BC |

## 4. 答辩速记版

可以按下面这段回答：

```text
本项目一共实现了 BC、CQL、DDPG、SAC、TD3 及其 NB 改进版 10 个算法。

BC 是行为克隆基线，主要用监督学习拟合数据动作；BC_NB 加入轨迹回报加权，让高回报样本更重要。

CQL 通过 CQL penalty 对数据分布外动作保持保守，防止 Q 值过估计；CQL_NB 在此基础上加入 target smoothing noise，并扩大 penalty 覆盖范围。

DDPG、TD3 都使用双 critic、target network、目标动作噪声和延迟更新等稳定化策略；DDPG 更强调 advantage-weighted BC，TD3 更强调双 Q、目标策略平滑和延迟策略更新。

SAC 使用随机高斯策略、熵正则和自动 alpha，项目中还结合 AWAC 思想，让高优势数据动作概率更高。

DDPG_NB、SAC_NB、TD3_NB 的共同点是加入 BC 正则，让 actor 在追求高 Q 或最大熵目标时不要偏离离线数据分布，因此更适合离线强化学习场景。
```

## 5. 最容易混淆的点

| 易混点 | 正确理解 |
| --- | --- |
| `actor loss` 能不能跨算法直接比大小 | 不能。不同算法的 actor loss 定义不同，只能看同一算法内部的趋势和稳定性 |
| `critic loss` 越低是否一定代表策略越好 | 不一定。critic loss 主要反映价值拟合误差，还要结合 actor、BC loss 和曲线稳定性 |
| CQL 的惩罚项是不是普通正则化 | 不是普通防过拟合正则，它专门用于压低数据外动作的 Q 值 |
| BC 正则和 BC 算法是不是一回事 | 不是。BC 算法只做行为克隆；BC 正则是在 actor-critic 算法中额外加入动作约束 |
| 延迟更新 actor 的目的 | 让 critic 先学稳定，再让 actor 根据更可靠的 Q 值更新 |
| target smoothing 的目的 | 给目标动作加小噪声，避免 Q 网络只对某个尖锐动作点过度乐观 |

## 6. 一句话总括

```text
这些算法的核心改进都围绕离线强化学习的两个风险展开：
一是 Q 值容易对数据外动作过估计，所以需要 CQL penalty、双 Q、target smoothing；
二是 actor 容易偏离离线数据分布，所以需要 BC 正则、样本加权和 advantage-weighted 更新。
```

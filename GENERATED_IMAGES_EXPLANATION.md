# 项目生成图片说明

本文用于解释项目中已经生成的图片文件：它们放在哪里、对应哪个页面、每张图在表达什么，以及阅读时应该重点看什么。

生成时间背景：本文整理于 2026-05-22，依据当前项目目录、Django 页面配置和 `AIcodes/checkpoints/` 下的训练日志。

## 解释范围

本文解释项目自有图片，不包含依赖包自带图标。

| 图片目录 | 数量 | 主要用途 | 页面关系 |
| --- | ---: | --- | --- |
| `media/datainfo/` | 3 | HalfCheetah 数据集分布图 | `/datainfo/` 数据可视化 |
| `AIcodes/checkpoints/` | 42 | 各算法训练曲线、诊断曲线、结果图 | `/results/` 和 `/compare/` |
| `output/playwright/` | 6 | Playwright 生成的页面截图和标注图 | 页面验收、模块映射、答辩材料 |

未纳入解释的图片：`output/playwright/node_modules/` 里的 Playwright 内置图标和浏览器 logo。

## 读图通用规则

1. 先看图片属于哪一类：数据分布图、训练结果图、页面截图图。不同类型不能直接比较数值大小。
2. Loss 曲线通常看趋势是否下降、是否收敛、是否剧烈震荡；最终值只是参考。
3. Actor loss 不一定是“越小越好”。很多 actor loss 前面带负号，下降可能表示 actor 正在追求更高 Q 值。
4. Critic loss 用来观察 Q 网络是否稳定拟合 TD target。它受 reward 标准化、Q 值尺度、算法公式影响，不适合跨算法直接比大小。
5. Q 值曲线不是越大越好。突然爆大、爆小或持续发散，都可能表示训练不稳定。
6. `reward.png` 或日志里的 `train_reward`、`val_reward` 在本项目里主要用于观察训练批次中的奖励数据，不等同于环境 rollout 的最终回报。
7. `lr_curve.png` 是学习率调度曲线，不是模型效果曲线。

## 文件名速查

| 文件名关键词 | 含义 | 重点看什么 |
| --- | --- | --- |
| `train_val_loss` | 训练集和验证集行为克隆误差 | 两条线是否都下降，训练/验证差距是否过大 |
| `train_test_loss` | 训练集和测试集行为克隆误差 | 测试误差是否跟随训练误差下降 |
| `lr_curve` | 学习率变化 | 学习率是否按计划逐渐降低 |
| `critic_loss` | critic 的 TD 误差 | 是否收敛、是否爆炸、是否异常震荡 |
| `critic1_loss` / `critic2_loss` | SAC 双 critic 的两条损失 | 两个 critic 是否量级接近、是否同步稳定 |
| `actor_loss` | actor 策略更新目标 | 结合算法公式解释，不能只按大小判断 |
| `cql_penalty` | CQL 保守项 | 是否稳定，是否帮助压制分布外动作的高 Q |
| `Q1` / `Q2` / `q_value` | Q 值估计 | 是否发散，双 Q 是否量级接近 |
| `alpha` | SAC 熵温度系数 | 是否稳定，是否过大或过小 |
| `advantage` | 数据动作相对策略动作的优势 | 是否出现极端波动 |
| `weight` | 优势权重或样本权重 | 是否被极端样本主导 |
| `bc_loss` | 行为克隆正则项 | actor 是否仍贴近离线数据动作 |
| `log_prob` / `log_prob_data` | 策略对动作的对数概率 | 策略分布是否稳定 |
| `reward` | 批次奖励均值或记录值 | 观察数据奖励尺度，不直接等同算法最终性能 |

## 数据集图片

这 3 张图位于 `media/datainfo/`，由 `/datainfo/` 页面读取，用来解释 HalfCheetah 数据集本身。

| 文件 | 图像大小 | 展示内容 | 怎么解释 |
| --- | --- | --- | --- |
| [`media/datainfo/observations_subplots.png`](media/datainfo/observations_subplots.png) | 2000 x 2000 | 17 维 observation 的分布子图 | 每个子图对应一个状态维度。用于观察状态值是否集中、是否偏移、是否存在长尾或异常范围。训练前做 observation 标准化，就是为了减弱这些维度尺度差异。 |
| [`media/datainfo/actions_subplots.png`](media/datainfo/actions_subplots.png) | 1500 x 1000 | 6 维 action 的分布子图 | 每个子图对应一个动作维度。HalfCheetah 动作一般被限制在 `[-1, 1]` 附近，这张图可以帮助判断动作是否经常贴近边界，以及行为克隆要学习的动作标签大致分布。 |
| [`media/datainfo/reward_terminal_subplots.png`](media/datainfo/reward_terminal_subplots.png) | 1500 x 500 | reward 分布和 terminal 标记 | 左侧看每步奖励的大致范围，右侧看 episode 终止标记。BC_NB 会用 reward 和 terminal 划分轨迹并给高回报轨迹更高权重。 |

## 训练结果图片总览

训练结果图片位于 `AIcodes/checkpoints/<算法名>/`。这些图由训练脚本生成，`/results/` 页面按算法展示，`/compare/` 页面按 Base / NB 成对对比。

所有训练图当前都是 `800 x 500` 的 PNG。

### BC

算法定位：行为克隆基线，只学习 `state -> action` 的监督映射，不训练 critic。

| 文件 | 含义 | 读图方法 |
| --- | --- | --- |
| [`AIcodes/checkpoints/BC/train_val_loss.png`](AIcodes/checkpoints/BC/train_val_loss.png) | 训练误差与验证误差曲线 | 重点看两条曲线是否一起下降并趋于稳定。如果训练误差很低但验证误差明显升高，可能是过拟合。 |
| [`AIcodes/checkpoints/BC/lr_curve.png`](AIcodes/checkpoints/BC/lr_curve.png) | 学习率变化曲线 | 用来确认学习率调度是否正常执行。学习率降低通常意味着训练后期进入细调阶段。 |

当前日志提示：第 150 轮 `val_loss = 0.016500`，验证集 `mse = 0.033168`，`mae = 0.113791`。这类数值主要说明动作拟合误差，不说明策略真实环境回报。

### BC_NB

算法定位：回报加权行为克隆。仍然是监督学习，但高回报轨迹样本权重更高。

| 文件 | 含义 | 读图方法 |
| --- | --- | --- |
| [`AIcodes/checkpoints/BC_NB/train_test_loss.png`](AIcodes/checkpoints/BC_NB/train_test_loss.png) | 训练误差与测试误差曲线 | 重点看测试误差是否跟随训练误差下降。相比 BC，它体现加权样本后整体拟合是否稳定。 |
| [`AIcodes/checkpoints/BC_NB/lr_curve.png`](AIcodes/checkpoints/BC_NB/lr_curve.png) | 学习率变化曲线 | 用来确认学习率衰减是否按训练计划完成。 |

当前日志提示：第 150 轮 `test_loss = 0.016508`，验证集 `mse = 0.033184`，`weighted_mse = 0.033184`。与 BC 结果非常接近，说明在当前数据和参数下，加权策略没有带来明显数值差距，但它更适合解释“高质量轨迹权重”的改进思想。

### CQL

算法定位：保守 Q 学习，通过 CQL penalty 降低分布外动作被高估的风险。

| 文件 | 含义 | 读图方法 |
| --- | --- | --- |
| [`AIcodes/checkpoints/CQL/critic_loss_curve.png`](AIcodes/checkpoints/CQL/critic_loss_curve.png) | critic TD 损失 | 主图之一。看 Q 网络拟合 TD target 是否稳定，震荡过大时要谨慎解释结果。 |
| [`AIcodes/checkpoints/CQL/cql_penalty_curve.png`](AIcodes/checkpoints/CQL/cql_penalty_curve.png) | CQL 保守项 | 主图之一。它不是简单越小越好，要结合 critic loss 看是否在稳定地约束 Q 估计。 |
| [`AIcodes/checkpoints/CQL/actor_loss_curve.png`](AIcodes/checkpoints/CQL/actor_loss_curve.png) | actor 损失 | actor 目标包含 Q 最大化和 BC 约束，重点看是否稳定，不要单独用最终大小判断优劣。 |

当前日志提示：第 120 轮 `val_critic_loss = 20.678396`，`val_actor_loss = 2.532608`，`val_cql_penalty = -4.789153`。CQL penalty 出现负值并不必然表示错误，它和实现中的 `logsumexp - q_data` 以及 Q 值尺度有关。

### CQL_NB

算法定位：增强版 CQL，加入目标动作扰动和更丰富的保守项计算。

| 文件 | 含义 | 读图方法 |
| --- | --- | --- |
| [`AIcodes/checkpoints/CQL_NB/critic_loss_curve.png`](AIcodes/checkpoints/CQL_NB/critic_loss_curve.png) | critic TD 损失 | 对比 CQL 时看量级、波动和最终是否收敛。 |
| [`AIcodes/checkpoints/CQL_NB/cql_penalty_curve.png`](AIcodes/checkpoints/CQL_NB/cql_penalty_curve.png) | 扩展 CQL 保守项 | 因为 NB 版 penalty 计算更丰富，数值不能直接和 CQL 的 penalty 简单比大小。 |
| [`AIcodes/checkpoints/CQL_NB/actor_loss_curve.png`](AIcodes/checkpoints/CQL_NB/actor_loss_curve.png) | actor 损失 | 看策略更新是否平稳，是否和 critic 曲线出现同步异常。 |

当前日志提示：第 150 轮 `val_critic_loss = 39.217671`，`val_actor_loss = 2.531301`，`val_cql_penalty = 10.979043`。相比 CQL，critic loss 和 penalty 量级更大，解释时应强调“公式和约束项不同”，不要直接得出 NB 更差。

### DDPG

算法定位：项目中的 DDPG 采用优势加权行为克隆思想，actor 更重视优势更高的数据动作。

| 文件 | 含义 | 读图方法 |
| --- | --- | --- |
| [`AIcodes/checkpoints/DDPG/critic_loss_curve.png`](AIcodes/checkpoints/DDPG/critic_loss_curve.png) | critic 损失 | 观察双 Q 对 TD target 的拟合是否稳定。 |
| [`AIcodes/checkpoints/DDPG/actor_loss_curve.png`](AIcodes/checkpoints/DDPG/actor_loss_curve.png) | actor 损失 | 本项目 DDPG 的 actor 本质接近 advantage-weighted BC，重点看趋势是否稳定。 |
| [`AIcodes/checkpoints/DDPG/weight_curve.png`](AIcodes/checkpoints/DDPG/weight_curve.png) | 优势权重曲线 | 观察样本权重是否过大或波动剧烈。权重过极端时，训练可能被少数样本主导。 |

当前日志提示：第 150 轮 `val_critic_loss = 19.396916`，`val_actor_loss = 0.031106`，`val_weight = 0.893640`。权重接近 1，说明最后阶段没有明显极端放大。

### DDPG_NB

算法定位：DDPG + BC 正则，actor 同时包含 Q 最大化项和行为克隆约束项。

| 文件 | 含义 | 读图方法 |
| --- | --- | --- |
| [`AIcodes/checkpoints/DDPG_NB/critic_loss_curve.png`](AIcodes/checkpoints/DDPG_NB/critic_loss_curve.png) | critic 损失 | 与 DDPG 对比时可看是否更稳定、最终量级是否更低。 |
| [`AIcodes/checkpoints/DDPG_NB/actor_loss_curve.png`](AIcodes/checkpoints/DDPG_NB/actor_loss_curve.png) | actor 总损失 | 由于包含负 Q 项，数值为负不代表错误。 |
| [`AIcodes/checkpoints/DDPG_NB/bc_loss_curve.png`](AIcodes/checkpoints/DDPG_NB/bc_loss_curve.png) | 行为克隆正则损失 | 用来解释 NB 版如何约束 actor 不要偏离数据动作。 |

当前日志提示：第 150 轮 `val_critic_loss = 14.947542`，`val_actor_loss = -2.158637`，`val_bc_loss = 0.050972`，`val_Q1` 和 `val_Q2` 约为 49.26。负的 actor loss 来自 Q 最大化项，是该目标形式的正常现象。

### SAC

算法定位：项目中的 SAC 使用最大熵 critic，并采用 AWAC 风格 actor 更新。

| 文件 | 含义 | 读图方法 |
| --- | --- | --- |
| [`AIcodes/checkpoints/SAC/critic1_loss.png`](AIcodes/checkpoints/SAC/critic1_loss.png) | 第一个 critic 损失 | 与 `critic2_loss.png` 一起看，判断双 critic 是否同步稳定。 |
| [`AIcodes/checkpoints/SAC/critic2_loss.png`](AIcodes/checkpoints/SAC/critic2_loss.png) | 第二个 critic 损失 | 如果两个 critic 量级差异很大或都快速发散，需要谨慎使用结果。 |
| [`AIcodes/checkpoints/SAC/actor_loss.png`](AIcodes/checkpoints/SAC/actor_loss.png) | actor 损失 | SAC 这里是 AWAC 风格加权 log probability，数值大小需结合优势权重解释。 |
| [`AIcodes/checkpoints/SAC/alpha.png`](AIcodes/checkpoints/SAC/alpha.png) | 熵温度系数 | alpha 控制策略随机性。过大可能让熵项影响过强。 |
| [`AIcodes/checkpoints/SAC/reward.png`](AIcodes/checkpoints/SAC/reward.png) | 奖励记录 | 主要看训练/验证批次奖励尺度，不当作最终 policy return。 |
| [`AIcodes/checkpoints/SAC/Q1.png`](AIcodes/checkpoints/SAC/Q1.png) | Q1 估计 | 观察 Q 值是否发散。 |
| [`AIcodes/checkpoints/SAC/Q2.png`](AIcodes/checkpoints/SAC/Q2.png) | Q2 估计 | 与 Q1 一起看双 Q 是否接近。 |
| [`AIcodes/checkpoints/SAC/advantage.png`](AIcodes/checkpoints/SAC/advantage.png) | 优势值 | 反映数据动作相对策略动作的价值差。 |
| [`AIcodes/checkpoints/SAC/weight.png`](AIcodes/checkpoints/SAC/weight.png) | AWAC 权重 | 看权重是否极端，极端权重会放大少数样本影响。 |
| [`AIcodes/checkpoints/SAC/log_prob_data.png`](AIcodes/checkpoints/SAC/log_prob_data.png) | 数据动作 log probability | 观察策略对离线数据动作的概率估计是否稳定。 |

当前日志提示：第 150 轮 `val_critic1_loss = 986091.320412`，`val_critic2_loss = 1006871.440575`，`val_Q1` 和 `val_Q2` 约为 `-103672`，`val_alpha = 148.413162`。这些值量级非常大，虽然当前日志未显示 NaN，但解释时应把 SAC Base 结果视为训练稳定性风险较高的结果。

### SAC_NB

算法定位：SAC + BC 正则，actor 使用 SAC 目标并额外约束动作接近离线数据。

| 文件 | 含义 | 读图方法 |
| --- | --- | --- |
| [`AIcodes/checkpoints/SAC_NB/critic1_loss.png`](AIcodes/checkpoints/SAC_NB/critic1_loss.png) | 第一个 critic 损失 | 看是否平稳下降，是否比 SAC Base 更容易控制量级。 |
| [`AIcodes/checkpoints/SAC_NB/critic2_loss.png`](AIcodes/checkpoints/SAC_NB/critic2_loss.png) | 第二个 critic 损失 | 与 critic1 一起看双 critic 稳定性。 |
| [`AIcodes/checkpoints/SAC_NB/actor_loss.png`](AIcodes/checkpoints/SAC_NB/actor_loss.png) | actor 总损失 | 包含 SAC 策略项和 BC 正则项，负值不等于异常。 |
| [`AIcodes/checkpoints/SAC_NB/alpha.png`](AIcodes/checkpoints/SAC_NB/alpha.png) | 熵温度系数 | 看温度参数是否收敛在合理范围。 |
| [`AIcodes/checkpoints/SAC_NB/reward.png`](AIcodes/checkpoints/SAC_NB/reward.png) | 奖励记录 | 用于观察数据批次奖励，不等于最终策略回报。 |
| [`AIcodes/checkpoints/SAC_NB/Q1.png`](AIcodes/checkpoints/SAC_NB/Q1.png) | Q1 估计 | 看 Q 值是否稳定。 |
| [`AIcodes/checkpoints/SAC_NB/Q2.png`](AIcodes/checkpoints/SAC_NB/Q2.png) | Q2 估计 | 与 Q1 一起观察双 Q 的一致性。 |
| [`AIcodes/checkpoints/SAC_NB/bc_loss.png`](AIcodes/checkpoints/SAC_NB/bc_loss.png) | BC 正则损失 | 说明策略是否仍受离线动作约束。 |
| [`AIcodes/checkpoints/SAC_NB/log_prob.png`](AIcodes/checkpoints/SAC_NB/log_prob.png) | 策略采样动作 log probability | 用来判断策略分布是否稳定。 |

当前日志提示：第 120 轮 `val_critic1_loss = 3.349683`，`val_critic2_loss = 3.293697`，`val_actor_loss = -87.841180`，`val_bc_loss = 0.263219`，`val_alpha = 0.108365`。相比 SAC Base，这组数值尺度明显更可控，更适合在页面中作为“BC 正则稳定策略”的例子。

### TD3

算法定位：双 Q、目标动作平滑、延迟 actor 更新的确定性策略算法。

| 文件 | 含义 | 读图方法 |
| --- | --- | --- |
| [`AIcodes/checkpoints/TD3/critic_loss.png`](AIcodes/checkpoints/TD3/critic_loss.png) | critic 损失 | 主图之一，用来观察双 Q 对 TD target 的拟合是否稳定。 |
| [`AIcodes/checkpoints/TD3/q_value.png`](AIcodes/checkpoints/TD3/q_value.png) | Q 值曲线 | 主图之一。Q 值过大或剧烈波动时要警惕过估计或尺度问题。 |
| [`AIcodes/checkpoints/TD3/actor_loss.png`](AIcodes/checkpoints/TD3/actor_loss.png) | actor 损失 | TD3 actor loss 通常是 `-Q`，负值越大不一定越差，要和 Q 值一起解释。 |

当前日志提示：第 150 轮 `val_critic_loss = 21.227167`，`val_actor_proxy = -632.385568`，`val_target_q = 627.117407`。Q 值量级较大，解释时建议强调曲线稳定性，而不是只比较最终 Q 值。

### TD3_NB

算法定位：TD3 + BC 正则。相比 TD3，actor 既追求高 Q，也保持接近离线数据动作。

| 文件 | 含义 | 读图方法 |
| --- | --- | --- |
| [`AIcodes/checkpoints/TD3_NB/critic_loss.png`](AIcodes/checkpoints/TD3_NB/critic_loss.png) | critic 损失 | 与 TD3 对比时看是否更稳定，是否量级更可控。 |
| [`AIcodes/checkpoints/TD3_NB/q_value.png`](AIcodes/checkpoints/TD3_NB/q_value.png) | Q 值曲线 | 看 Q 值是否稳定，不要直接和 TD3 的 Q 值做绝对大小比较。 |
| [`AIcodes/checkpoints/TD3_NB/actor_loss.png`](AIcodes/checkpoints/TD3_NB/actor_loss.png) | actor 总损失 | 包含 Q 项和 BC 项，通常需要结合日志中的 `val_q_term`、`val_bc_term` 解释。 |

当前日志提示：第 150 轮 `val_critic_loss = 14.764948`，`val_actor_proxy = -2.164941`，`val_bc_term = 0.050942`，`val_target_q = 49.747568`。与 TD3 Base 相比，Q 值尺度更小，适合解释“BC 正则让离线策略更保守”。

## Base / NB 对比图片关系

`/compare/` 页面会把 Base 和 NB 的关键图片并排展示。下面是每组默认对比关系。

| 算法对 | 对比图 | 解释重点 |
| --- | --- | --- |
| `BC / BC_NB` | `train_val_loss.png` vs `train_test_loss.png`，`lr_curve.png` | 看普通行为克隆和回报加权行为克隆的误差差异。当前结果非常接近，适合讲算法思想差异。 |
| `CQL / CQL_NB` | `critic_loss_curve.png`，`actor_loss_curve.png`，`cql_penalty_curve.png` | 看 NB 版目标动作扰动和扩展 penalty 对训练曲线的影响。 |
| `DDPG / DDPG_NB` | `critic_loss_curve.png`，`actor_loss_curve.png`，`weight_curve.png` / `bc_loss_curve.png` | Base 看优势权重，NB 看 BC 正则。重点是改进项不同，不是单纯同名曲线比大小。 |
| `SAC / SAC_NB` | `actor_loss.png`，`critic1_loss.png`，`critic2_loss.png`，`alpha.png` | 当前 SAC Base 数值尺度明显异常偏大，SAC_NB 更适合展示稳定化效果。 |
| `TD3 / TD3_NB` | `critic_loss.png`，`actor_loss.png`，`q_value.png` | TD3_NB 的 Q 值尺度更可控，但需结合 BC 项解释，不能只看 Q 值大小。 |

## Playwright 页面截图和标注图

这些图片位于 `output/playwright/`，它们不是算法训练结果，而是页面验收、模块映射或答辩用截图。

| 文件 | 图像大小 | 含义 |
| --- | ---: | --- |
| [`output/playwright/annotated/01-data-management-mapping.png`](output/playwright/annotated/01-data-management-mapping.png) | 1500 x 1500 | `/datainfo/` 页面模块映射图。标注了数据集概览统计卡、基础信息表、CSV 预览、字段说明入口和数据可视化入口。 |
| [`output/playwright/annotated/02-modelsinfo-mapping.png`](output/playwright/annotated/02-modelsinfo-mapping.png) | 1500 x 1100 | `/modelsinfo/` 页面模块映射图。标注了算法组切换、方法流程、核心设计、参数说明、代码讲解、输入输出与产物、Base / NB 对比和小结。 |
| [`output/playwright/annotated/03-results-mapping.png`](output/playwright/annotated/03-results-mapping.png) | 1500 x 1300 | `/results/` 页面模块映射图。标注了算法选择、结果状态、概览标签页、核心图表、详细指标、文件下载和结果理解步骤。 |
| [`output/playwright/annotated/04-compare-mapping.png`](output/playwright/annotated/04-compare-mapping.png) | 1500 x 1250 | `/compare/` 页面模块映射图。标注了算法对选择、概览、关键指标对比、核心曲线对比、文件与产物、差异解读和一句话结论。 |
| [`output/playwright/modelsinfo-bc-conclusion-points.png`](output/playwright/modelsinfo-bc-conclusion-points.png) | 1296 x 576 | `/modelsinfo/` 页面中 BC 小结分点展示的视觉验收截图，用于确认小结从短段落改成分点说明后可读性是否合适。 |
| [`output/playwright/modelsinfo-cql-core-design.png`](output/playwright/modelsinfo-cql-core-design.png) | 1500 x 900 | `/modelsinfo/` 页面中 CQL 核心设计模块的视觉验收截图，用于确认“通俗理解 + 技术对应”的分层说明是否正确渲染。 |

## 页面使用关系

| 页面 | 使用哪些图片 | 代码位置 |
| --- | --- | --- |
| `/datainfo/` | `media/datainfo/*.png` | `apps/datainfo/views.py` 和 `templates/datainfo/halfcheetah_overview.html` |
| `/results/` | 当前选择算法目录下的主图和进阶图 | `apps/results/views.py` 和 `templates/results/results.html` |
| `/compare/` | Base / NB 两个算法目录下的核心对比图 | `apps/compare/views.py` 和 `templates/compare/compare.html` |
| 页面验收截图 | `output/playwright/*.png` | `output/playwright/annotate-pages.js` |

## 答辩或报告中的推荐说法

可以按下面顺序介绍这些图片：

1. 先用 `media/datainfo/` 的 3 张图说明数据集结构：状态 17 维、动作 6 维、奖励和终止标记。
2. 再用 `BC` 和 `BC_NB` 的 loss 曲线说明最基础的行为克隆与加权行为克隆。
3. 接着按算法家族解释结果图：CQL 看保守项，DDPG 看优势权重或 BC 正则，SAC 看双 critic、alpha 和策略分布，TD3 看 critic 与 Q 值。
4. 最后用 `/compare/` 的 Base / NB 并列图说明改进版不一定每个数字都更好，但它提供了更适合离线训练的约束或稳定化设计。

一句话总结：这些生成图片不是孤立的“效果截图”，而是项目从数据集理解、算法训练、结果展示到页面验收的完整证据链。

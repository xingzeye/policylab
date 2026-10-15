# 项目算法读书笔记

本文整理的是本项目 `AIcodes/` 目录下 10 个算法脚本的读书笔记。每一篇围绕一个算法展开，重点不是重新抄一遍公式，而是结合 HalfCheetah 离线数据集、训练脚本和项目展示页面，说明这个算法在项目中想解决什么问题、代码怎么体现、阅读时应该抓住什么。

## 1. BC 行为克隆

读 `AIcodes/BC.py` 时，最直接的感受是它把强化学习问题先降成了监督学习问题。项目中的 HalfCheetah 数据集本来包含状态、动作、奖励、下一状态和终止标记，但 BC 主要只使用 `observations` 和 `actions`。也就是说，它并不判断一个动作未来能带来多少回报，而是学习“在这个状态下，数据集里的动作是什么”。这种思路很适合作为离线强化学习的入门算法，因为训练链路清楚：读取状态和动作，计算训练集状态均值和标准差，建立 `BCPolicy` 网络，用 MLP 从状态预测动作，再用 SmoothL1Loss 或 MSELoss 衡量预测动作和数据动作的差距。

BC 的优点是稳定、简单、容易解释。项目里的 `BCPolicy` 最后用 `tanh` 把动作限制在合理范围内，这和 HalfCheetah 连续动作控制任务相匹配。它的训练目标可以理解成 `loss = distance(pi(s), a_data)`，只要验证集 loss 下降，就保存 `best_model.pt`。但是它的局限也很明显：如果离线数据中既有好轨迹也有差轨迹，BC 会一起模仿。它不会主动分辨哪些动作更优，也不会利用 reward 学到“更长远”的选择。因此我理解 BC 更像是项目中的基线方法，它提供了一个最低复杂度的参照，让后面的 CQL、DDPG、SAC、TD3 都能在“如何利用奖励和 Q 值”这一点上展开对比。

## 2. BC_NB 回报加权行为克隆

`AIcodes/BC_NB.py` 仍然属于行为克隆，但它比普通 BC 多了一层“样本质量意识”。普通 BC 把所有状态动作对等同看待，数据中只要出现过，模型就会尽量模仿。BC_NB 的改进点在于，它读取 `rewards` 和 `terminals`，先按 episode 切分轨迹，再计算每条轨迹的累计回报。高回报轨迹中的样本会得到更大的权重，低回报轨迹中的样本权重会被压低。这样一来，训练仍然是动作拟合，但梯度贡献不再平均分配。

代码中关键函数是 `build_sample_weights`。它把 episode return 标准化后放进指数函数，再用 `min_weight` 和 `max_weight` 控制权重范围，避免极端轨迹把训练完全带偏。训练时也不是直接对整个 batch 求一个普通 loss，而是先通过 `per_sample_loss` 得到每个样本自己的动作误差，再乘以样本权重。这个设计让我觉得很有启发：它没有引入复杂的 critic，也没有做 Bellman 更新，却已经开始利用 reward 给模仿学习加方向。

从读书笔记角度看，BC_NB 可以理解为 BC 和强化学习之间的过渡。它承认“模仿”仍然重要，但不再盲目模仿所有行为。对于离线数据来说，这种想法很现实，因为模型不能和环境互动，只能从历史轨迹里挑更值得学的部分。它的风险在于 episode return 是比较粗的质量信号，同一条轨迹里并不是每个动作都一定好。不过作为项目中的 NB 改进版，它很好地展示了如何在保持代码简单的同时，让算法更关注高质量经验。

## 3. CQL 保守 Q 学习

读 `AIcodes/CQL.py` 时，重点要从“模仿动作”转到“估计动作价值”。CQL 使用完整的离线转移数据，包括状态、动作、奖励、下一状态和 done。它建立 actor、两个 critic 以及 target critic，通过 Bellman target 学习 Q 值。和普通 actor-critic 不同的是，CQL 特别关心离线训练中的一个危险：数据集中没出现过的动作，critic 可能会给出虚高 Q 值，而 actor 一旦追着这些虚高值走，就会跑到数据分布外，策略反而变差。

项目中 `compute_cql_penalty` 就是在处理这个问题。它会比较随机动作、策略动作和数据动作的 Q 值，用类似 `logsumexp(Q候选动作) - Q数据动作` 的惩罚项压低分布外动作的价值。critic 的损失不只是 Bellman MSE，还要加上 `lambda_cql * CQL_penalty`。actor 的更新也不是单纯最大化 Q，而是加入 BC 项，让 `pi(s)` 不要离数据动作太远。这样 CQL 的核心思想就很清楚：既要学习高回报方向，也要保持保守。

我认为 CQL 是项目中最能体现“离线强化学习特性”的算法之一。在线强化学习可以探索，估错了还有机会用新数据修正；离线强化学习没有这个机会，所以宁可保守一些，也不能让策略被不可靠的 Q 值吸引。阅读这部分代码时，最应该抓住的是三条线：双 critic 如何计算 TD 目标，CQL penalty 如何约束分布外动作，actor loss 如何在高 Q 和贴近数据之间取平衡。

## 4. CQL_NB 带目标平滑的 CQL

`AIcodes/CQL_NB.py` 可以看成是在 CQL 基础上吸收了 TD3 的稳定化技巧。普通 CQL 已经通过保守惩罚降低分布外动作的 Q 值，但在连续控制任务中，动作空间是连续的，某个动作附近的小扰动也可能影响 Q 估计。CQL_NB 在计算下一状态动作时加入 `policy_noise`，并用 `noise_clip` 限制噪声范围，相当于让 target action 附近的一小片区域都被考虑进来。

项目文档把它概括为“CQL 和 TD3 平滑技巧的结合”，这个说法很贴切。它仍然保留 CQL 的核心：critic loss 等于 Bellman MSE 加 CQL penalty，actor 仍然通过 Q 项和 BC 项共同更新。但 CQL_NB 的 penalty 会比较更多类型的动作，包括随机动作、当前策略动作以及带噪声的下一状态策略动作。这样做的直觉是，如果只看一个精确动作点，Q 网络可能在局部产生尖锐误差；加入裁剪噪声后，critic 被迫学习更平滑的价值估计。

读这个算法时，我觉得它展示了一种常见的算法改造方式：不是推翻原算法，而是把另一个算法中经过验证的稳定技巧迁移过来。离线强化学习尤其需要这种稳定性，因为训练数据固定，错误的 Q 估计不能靠环境交互修复。CQL_NB 的价值就在于，一方面继续坚持 CQL 的保守原则，另一方面通过 target policy smoothing 降低连续动作空间中的估计震荡。它比 CQL 多了一些参数，但逻辑仍然围绕“保守”和“平滑”两个关键词展开。

## 5. DDPG Advantage-weighted 版本

项目里的 `AIcodes/DDPG.py` 并不是最原始的在线 DDPG，而是适配离线数据的 advantage-weighted 写法。DDPG 本身是确定性策略梯度算法，actor 输入状态输出一个确定性动作，critic 评价状态动作对的 Q 值。在离线场景中，如果 actor 直接最大化 critic 给出的 Q 值，很容易被过高估计带偏，所以项目代码没有让 actor 无约束地追求高 Q，而是通过优势权重来决定“哪些数据动作更值得模仿”。

这里最关键的思想是比较 `Q(s, a_data)` 和 `Q(s, pi(s))`。如果数据动作的 Q 值比当前 actor 动作更高，说明这条数据动作相对更有学习价值，于是权重变大；如果数据动作并不占优，权重就变小。公式可以记成 `advantage = Q(s, a_data) - Q(s, pi(s))`，再通过 `exp(advantage / aw_lambda)` 转成行为克隆权重。actor loss 本质上仍是加权 MSE，让策略贴近数据动作，但贴近程度由优势来控制。

这个算法给我的启发是，离线强化学习里“学得更好”和“别跑太远”常常要同时考虑。DDPG 的 actor-critic 框架负责引入长期回报判断，而 advantage-weighted BC 又把策略拉回数据分布中。代码中还使用双 Q、target policy smoothing、延迟更新等技巧，使 critic 更稳定。阅读时可以先看 `Actor` 和 `Critic` 的结构，再看 `train_step` 中 critic 更新、advantage 计算和 actor 加权更新的顺序，这样会比直接追每一行张量形状更容易理解。

## 6. DDPG_NB / TD3-lite + BC

`AIcodes/DDPG_NB.py` 的核心可以概括为“用 Q 项提升策略，用 BC 项约束策略”。它仍然采用确定性 actor 和双 critic，也使用 target 网络与延迟策略更新，但 actor 的损失形式和普通 DDPG 不同。普通思路可能会让 actor 最大化 `Q(s, pi(s))`，而离线场景下 critic 对未见动作可能过度乐观，所以 DDPG_NB 加入了 `MSE(pi(s), a_data)` 作为行为克隆正则。

项目中 actor loss 可以理解为两部分：`q_loss = -lam * mean(Q(s, pi(s)))`，负责推动策略选择高价值动作；`bc_loss = bc_coef * MSE(pi(s), a_data)`，负责让动作不要偏离数据集太远。其中 `lam = alpha_bc / mean(abs(Q))` 是一个自适应缩放项，用来平衡 Q 项的数值尺度。这个细节很重要，因为 Q 值大小可能随 reward scale 和训练阶段变化，如果不缩放，Q 项或 BC 项都可能在某些阶段过强。

和前一篇 DDPG 的 advantage-weighted 写法相比，DDPG_NB 更像 TD3+BC 的简化表达。它没有只挑高优势样本模仿，而是直接把“追求高 Q”和“贴近数据动作”合成一个 actor 目标。这种设计读起来更直观，也更容易和 TD3_NB 对照。我的理解是，DDPG_NB 的意义在于说明离线确定性策略算法不能只相信 critic，必须给 actor 加上数据分布约束。否则策略可能在训练日志里看似 Q 值提高，实际却选择了数据中很少出现、缺乏可靠依据的动作。

## 7. SAC 离线 AWAC 风格版本

`AIcodes/SAC.py` 保留了 SAC 的随机策略和熵正则思想，但项目里的写法更适合离线数据。标准 SAC 鼓励策略在高回报和高熵之间平衡，actor 会从高斯策略中采样动作，并用 `alpha * log_prob` 控制探索程度。可是离线训练不能真正探索环境，如果 actor 仅根据 critic 更新，就可能采到数据分布外的动作。因此项目中的 SAC 使用 AWAC 风格的 actor 更新，让策略更倾向于提高高优势数据动作的概率。

代码里 `GaussianActor` 是理解 SAC 的关键。它输出动作分布，而不是一个固定动作，所以除了动作本身，还要计算 log probability。critic target 中包含 `min(Q1,Q2) - alpha * log_pi`，体现了 SAC 的熵正则。actor 更新时，则计算数据动作和当前策略动作之间的 advantage，把 advantage 变成权重，再最大化加权后的 `log_pi(a_data | s)`。换句话说，策略不是盲目追求 critic 认为好的新动作，而是提高那些“在数据中出现过且价值较高”的动作概率。

我觉得 SAC 这部分最值得学习的是“随机策略在离线场景中的改造”。随机性原本服务于探索，但离线任务不能探索，所以随机策略必须被数据约束住。项目还支持自动学习 `alpha`，让熵权重能随训练调整。阅读时可以抓住三点：GaussianActor 如何采样并算 log probability，critic target 如何加入熵项，actor loss 如何通过优势权重回到数据动作上。

## 8. SAC_NB + BC 正则

`AIcodes/SAC_NB.py` 比 `SAC.py` 更像“普通 SAC 加行为克隆约束”。它同样使用 `GaussianActor`、双 critic 和 target critic，也保留 SAC 的 actor loss：让采样动作在高 Q 和熵项之间取得平衡。但为了适配离线训练，它额外计算采样动作和数据动作之间的 MSE，把 `bc_coef * bc_loss` 加到 actor loss 中。

这个设计的直觉很容易理解：SAC 的随机策略有能力产生数据集中没出现过的动作，在线环境中这可能是探索，离线环境中却可能是风险。BC 正则就像一个约束，把采样动作往数据动作附近拉。actor loss 可以记成 `sac_loss + bc_coef * MSE(sampled_action, a_data)`。critic 仍然通过 SAC target 学习价值，target 中含有熵项；actor 则同时考虑“这个动作 Q 值高不高”“策略是否保留足够随机性”“动作是否偏离数据太远”。

和 SAC 的 AWAC 风格版本相比，SAC_NB 的学习方式更直接。它不是通过高优势数据动作的 log probability 来更新 actor，而是把行为克隆误差作为一个明确的正则项加入目标函数。这样写的好处是代码和公式都更清晰，适合解释“离线 SAC 为什么需要约束”。不足是 BC 项的权重需要调好，太弱会约束不住策略，太强又可能退化成模仿学习。读这个脚本时，我会重点看 `train_step` 中 critic loss、actor loss、bc loss 和 alpha 更新的顺序，这正是它区别于普通 SAC 的地方。

## 9. TD3 离线训练

`AIcodes/TD3.py` 展示的是 TD3 的基础机制。TD3 可以看成 DDPG 的稳定版，它解决的主要问题是 actor-critic 训练中 Q 值过估计和策略更新过快。项目里的 TD3 直接使用离线数据训练，不和环境交互，所以它更像是用固定数据集复现 TD3 的核心技巧：双 Q、target policy smoothing 和 delayed policy update。

双 Q 的作用是防止 critic 对动作价值过于乐观。计算 TD target 时，代码会取两个 target critic 输出的较小值，即 `min(Q1_target, Q2_target)`。target policy smoothing 则是在 target actor 给出的下一动作上加入裁剪噪声，使 Q 学习不要只对某个精确动作点敏感。延迟 actor 更新的含义是先让 critic 多学几步，再更新 actor，避免 actor 追随尚不稳定的价值估计。actor loss 则是 `-mean(Q1(s, actor(s)))`，表示让 actor 选择 critic 认为更高价值的动作。

我认为 TD3.py 在项目中的意义是提供一个“稳定化技巧集合”的参照。它没有 BC 正则，所以在纯离线场景下仍存在策略偏离数据分布的风险，但这也让它更适合作为理解 TD3 原理的基础版本。阅读时可以先看 `TD3Trainer.train_step`：先构造带噪声的 target action，再更新 critic，隔一段步数更新 actor，最后软更新 target 网络。把这个顺序理清楚，TD3 的三大技巧就不会显得零散。

## 10. TD3_NB / TD3+BC

`AIcodes/TD3_NB.py` 是 TD3 在离线强化学习中更稳妥的改进版本。它保留 TD3 的双 Q、target policy smoothing、延迟 actor 更新和 target 网络软更新，同时在 actor loss 中加入 BC 正则。这样一来，actor 既追求 critic 给出的高 Q 动作，又不会离数据集动作太远。项目文档把它概括为“Q 项负责让动作变好，BC 项负责别离数据太远”，这是理解 TD3+BC 最简洁的方式。

代码中 actor loss 分为 `q_term` 和 `bc_term`。`q_term = -lam * mean(Q1(s, pi(s)))`，推动策略往高价值方向走；`bc_term = bc_coef * MSE(pi(s), a_data)`，把策略限制在数据动作附近。这里的 `lam = alpha_bc / mean(abs(Q))` 用于自动调整 Q 项尺度，避免 Q 值大小让 actor loss 失衡。相比基础 TD3，TD3_NB 还支持 reward 标准化，使 TD target 的数值更稳定。

读这个算法时，我最大的体会是，离线强化学习不是简单把在线算法搬到静态数据上。TD3 在在线场景中可以通过探索修正错误估计，但离线场景没有新反馈，所以必须主动限制 actor 的动作范围。TD3_NB 的 BC 项正是这个限制。它比普通 BC 更有提升空间，因为它利用 Q 值判断动作长期价值；又比纯 TD3 更稳，因为它不让 actor 过度相信 critic。作为项目最后一类算法，它很好地总结了离线连续控制中的核心矛盾：既要优化回报，又要尊重数据分布。

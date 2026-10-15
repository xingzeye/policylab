from django.shortcuts import render


def design_card(title, plain, technical):
    return {
        'title': title,
        'plain': plain,
        'technical': technical,
    }


ALGORITHM_GROUPS = {
    'bc': {
        'key': 'bc',
        'title': 'BC / BC_NB',
        'base_name': 'BC',
        'nb_name': 'BC_NB',
        'summary': '这一组方法适合新手理解“离线强化学习里最像监督学习”的思路：直接根据数据集中已有的状态和动作来学策略。',
        'sections': {
            'beginner': {
                'title': '小白先看',
                'lead': '先把它理解成“照着老师给的标准答案学动作”。BC 直接学动作，BC_NB 则更重视高质量轨迹里的样本。',
                'bullets': [
                    'BC：把状态当输入，把数据集里的动作当标签，训练一个策略网络。',
                    'BC_NB：在 BC 基础上给样本加权，让高回报轨迹中的样本更重要。',
                    '适合用来做入门基线，因为训练流程清楚、结果容易解释。',
                ],
            },
            'flow': {
                'title': '方法流程',
                'steps': [
                    '读取 HDF5 数据集。',
                    '划分训练集与验证集。',
                    '根据训练集计算状态标准化参数。',
                    '构建 MLP 策略网络。',
                    '执行训练、验证并保存最佳模型。',
                    '输出日志、评估汇总与曲线图。',
                ],
            },
            'design': {
                'title': '核心设计',
                'cards': [
                    design_card(
                        '数据处理',
                        '可以把 BC 看成“照着老师示范做动作”：网页把每条状态当作题目，把数据集里对应的动作当作标准答案。BC_NB 会额外看看这段轨迹最后表现好不好，好样本会被看得更重。',
                        'BC 主要读取 observations 和 actions；BC_NB 额外读取 rewards 与 terminals，并根据 episode return 构造 sample weights，让高回报轨迹在损失中占更大权重。',
                    ),
                    design_card(
                        '网络结构',
                        '网络就像一个动作预测器：输入机器人当前姿态和速度等状态信息，输出 6 个动作数值，告诉 HalfCheetah 每个关节大概要怎么发力。',
                        '两者都使用 MLP 策略网络，隐藏层默认 256,256，ReLU 负责非线性表达，输出层经过 tanh 将动作限制在 [-1, 1]，与环境动作范围匹配。',
                    ),
                    design_card(
                        '损失函数',
                        '训练时就是不断比较“模型给出的动作”和“数据里的动作”差多少。BC_NB 的区别是：来自高回报轨迹的样本，错一点会被罚得更明显。',
                        'BC 使用普通动作回归损失；BC_NB 先计算每个样本自己的 MSE 或 SmoothL1 损失，再乘以 sample weight，最后做加权平均。',
                    ),
                    design_card(
                        '稳定性设计',
                        '这些设计都是为了让训练别忽上忽下：先把状态数值整理到相近范围，再限制梯度过大，学习率也会慢慢变小。',
                        '两者都包含状态标准化、梯度裁剪、权重衰减和余弦退火学习率；BC 还支持 SmoothL1Loss 与可选 LayerNorm，减少异常样本和尺度差异带来的影响。',
                    ),
                ],
            },
            'params': {
                'title': '参数说明',
                'basic': [
                    ('data_path', '数据集路径'),
                    ('batch_size', '批大小'),
                    ('lr', '学习率'),
                    ('num_epochs', '训练轮数'),
                    ('hidden_layers', '隐藏层结构'),
                    ('seed', '随机种子'),
                    ('val_ratio', '验证集比例'),
                    ('output_dir', '输出目录'),
                ],
                'advanced': [
                    ('BC.use_layernorm', '是否使用 LayerNorm'),
                    ('BC.use_smooth_l1', '是否使用 SmoothL1Loss'),
                    ('BC_NB.loss_type', '逐样本损失类型，支持 mse / smooth_l1'),
                    ('BC_NB.weight_temperature', '控制样本权重差异程度'),
                    ('BC_NB.min_weight / max_weight', '限制样本权重上下界'),
                    ('action_clip_eps', '动作截断稳定项'),
                ],
            },
            'code': {
                'title': '代码块讲解',
                'items': [
                    ('AIcodes/BC.py -> Config / build_config_from_args', '定义训练参数并将配置写入 train_config.json。'),
                    ('AIcodes/BC.py -> load_dataset', '读取 observations/actions，切分训练集与验证集。'),
                    ('AIcodes/BC.py -> BCPolicy', 'MLP 策略网络，输出动作。'),
                    ('AIcodes/BC.py -> train_one_epoch / train_bc', '完成训练、验证、最佳模型保存和日志导出。'),
                    ('AIcodes/BC_NB.py -> build_sample_weights', '根据 rewards 与 terminals 计算 episode return，并得到样本权重。'),
                    ('AIcodes/BC_NB.py -> per_sample_loss', '先算逐样本损失，再结合权重进行训练。'),
                ],
            },
            'io': {
                'title': '输入输出与产物',
                'inputs': [
                    '代码文件：AIcodes/BC.py、AIcodes/BC_NB.py',
                    '数据文件：halfcheetah_medium-v2.hdf5',
                    '命令行参数：学习率、轮数、隐藏层、输出目录等',
                ],
                'outputs': [
                    'checkpoints/BC/ 或 checkpoints/BC_NB/',
                    'best_model.pt',
                    'train_config.json',
                    'training_log.csv',
                    'evaluation_summary.csv',
                    'train_val_loss.png / train_test_loss.png / lr_curve.png',
                ],
            },
            'compare': {
                'title': 'Base / NB 对比',
                'rows': [
                    ('核心思路', '普通行为克隆，直接回归动作', '加权行为克隆，优先学习高回报轨迹样本'),
                    ('输入字段', 'observations, actions', 'observations, actions, rewards, terminals'),
                    ('损失方式', '普通回归损失', '逐样本损失 × sample weights'),
                    ('典型参数', 'use_layernorm, use_smooth_l1', 'loss_type, weight_temperature, min_weight, max_weight'),
                    ('适用理解', '适合作为最基础基线', '适合展示“在基线上做改进”的思路'),
                ],
            },
            'conclusion': {
                'title': '小结',
                'text': '如果你第一次接触离线强化学习，建议先从 BC 看起，再看 BC_NB。这样能先理解“模仿学习”主线，再理解“好样本更重要”这一类改进方式。',
                'points': [
                    '建议先看 BC：它最像监督学习，把状态当输入、把数据集里的动作当标准答案，重点是理解“模仿已有数据”。',
                    '再看 BC_NB：它在 BC 基础上加入样本权重，让高回报轨迹里的样本更重要，适合理解“好样本应该多学一点”。',
                    '这一组适合作为离线强化学习入门基线，因为训练流程清楚，结果主要看动作预测误差和训练/验证曲线。',
                    '如果要做算法对比，可以把 BC 看成基础版本，把 BC_NB 看成“在基础模仿上加入轨迹质量判断”的改进版本。',
                ],
            },
        },
    },
    'cql': {
        'key': 'cql',
        'title': 'CQL / CQL_NB',
        'base_name': 'CQL',
        'nb_name': 'CQL_NB',
        'summary': '这一组方法属于更典型的离线强化学习算法：不仅学策略，还学 Q 值，并通过保守约束减少“离线数据外动作”的高估问题。',
        'sections': {
            'beginner': {
                'title': '小白先看',
                'lead': '可以把 CQL 理解成“学动作之前，先更谨慎地评价动作值不值”。CQL_NB 则是在这个基础上进一步增强训练稳定性。',
                'bullets': [
                    'CQL：学习 actor 和双 critic，并给 critic 加保守约束。',
                    'CQL_NB：在 CQL 上加入更多动作参与 penalty，并引入类似 TD3 的 target smoothing。',
                    '比 BC 更复杂，但更接近真正的离线强化学习方法。',
                ],
            },
            'flow': {
                'title': '方法流程',
                'steps': [
                    '读取包含 state、action、reward、next_state、done 的完整 transition 数据。',
                    '划分训练集与验证集。',
                    '计算状态标准化参数和 reward 统计量。',
                    '构建 actor、critic1、critic2 及 target critics。',
                    '更新 critic，并加上 CQL penalty。',
                    '更新 actor，并用 BC 项辅助训练。',
                    '保存最佳模型、日志和曲线图。',
                ],
            },
            'design': {
                'title': '核心设计',
                'cards': [
                    design_card(
                        '数据处理',
                        'CQL 不只看“当前应该做什么动作”，还要看“做完这个动作后会到哪里、能拿多少奖励”。所以它需要一整条状态转移记录。',
                        '两者都使用完整 transition：observations、actions、rewards、next_observations、terminals，用它们构造 TD target 和 critic 训练样本。',
                    ),
                    design_card(
                        '网络结构',
                        'actor 像做动作的学生，critic 像评分老师。两个 critic 相当于两个老师同时打分，最后取更谨慎的那个分数，避免把动作评价得太乐观。',
                        '两者都包含一个 actor 和两个 critic；critic 输入 state+action 输出 Q 值，actor 输入 state 输出 action；target critics 用软更新提供更稳定的目标值。',
                    ),
                    design_card(
                        '保守约束',
                        '离线数据里没出现过的动作，模型其实没亲自试过。CQL 的核心就是提醒 critic：别随便给没见过的动作打高分。',
                        'CQL 用 logsumexp penalty 压低随机动作和当前策略动作的 Q 值，同时保留数据动作的价值；CQL_NB 进一步把 next policy 动作也纳入 penalty 覆盖范围。',
                    ),
                    design_card(
                        '稳定性增强',
                        '这一组方法很容易因为 Q 值估计不稳而抖动，所以会对奖励、梯度和目标动作都做处理，让训练过程更像慢慢校准而不是猛跳。',
                        '两者都支持 reward scaling、梯度裁剪和 BC 辅助 actor；CQL_NB 额外加入 target smoothing noise，在 next action 上加截断噪声后再计算目标 Q。',
                    ),
                ],
            },
            'params': {
                'title': '参数说明',
                'basic': [
                    ('dataset_path', '数据集路径'), ('batch_size', '批大小'), ('num_epochs', '训练轮数'), ('lr_actor / lr_critic', 'actor/critic 学习率'), ('gamma / tau', '折扣系数与软更新系数'), ('hidden_layers', '隐藏层结构'), ('output_dir', '输出目录')
                ],
                'advanced': [
                    ('lambda_cql', '保守约束权重'), ('num_random_actions', '构造 penalty 时采样的随机动作数'), ('alpha_bc / bc_coef', 'actor 中 BC 相关参数'), ('reward_scale', '是否对 reward 标准化'), ('CQL_NB.policy_noise / noise_clip', 'target smoothing 噪声参数')
                ],
            },
            'code': {
                'title': '代码块讲解',
                'items': [
                    ('AIcodes/CQL.py -> Config', '定义 CQL 训练参数并保存配置。'),
                    ('AIcodes/CQL.py -> load_hdf5_dataset', '加载完整 transition 数据。'),
                    ('AIcodes/CQL.py -> compute_cql_penalty', '构造保守 Q 约束。'),
                    ('AIcodes/CQL.py -> train_cql_improved', '执行双 critic + actor 训练。'),
                    ('AIcodes/CQL_NB.py -> compute_cql_penalty', '加入 next policy action 和 smoothing noise。'),
                    ('AIcodes/CQL_NB.py -> train_cql_improved', '把 target smoothing 和更丰富的 penalty 用到训练中。'),
                ],
            },
            'io': {
                'title': '输入输出与产物',
                'inputs': [
                    '代码文件：AIcodes/CQL.py、AIcodes/CQL_NB.py',
                    '数据文件：halfcheetah_medium-v2.hdf5',
                    '完整 transition 输入：state/action/reward/next_state/done',
                ],
                'outputs': [
                    'checkpoints/CQL/ 或 checkpoints/CQL_NB/',
                    'best_model.pt', 'train_config.json', 'training_log.csv',
                    'critic_loss_curve.png', 'actor_loss_curve.png', 'cql_penalty_curve.png',
                ],
            },
            'compare': {
                'title': 'Base / NB 对比',
                'rows': [
                    ('核心思路', '标准保守 Q 学习 + BC 辅助 actor', '在 CQL 基础上增加更强的 penalty 覆盖和 smoothing'),
                    ('Penalty 动作集合', '随机动作 + 当前策略动作', '随机动作 + 当前策略动作 + next policy 动作'),
                    ('稳定性设计', 'reward scaling、双 critic、软更新', '额外加入 TD3 风格 target smoothing'),
                    ('适合理解', '理解离线 RL 中“保守”的意义', '理解如何在基线 CQL 上继续增强稳定性'),
                ],
            },
            'conclusion': {
                'title': '小结',
                'text': '先把 CQL 理解成“更谨慎地学 Q 值”，再看 CQL_NB 如何把这个思路做得更稳。这样更容易接受离线强化学习里 critic 设计的重要性。',
                'points': [
                    '建议先把 CQL 理解成“先给动作打分，再决定怎么学动作”：critic 负责判断动作值不值，actor 再根据这个评价调整策略。',
                    'CQL 的核心价值是保守：离线数据里没出现过的动作不能随便给高分，否则策略可能学到数据外的危险动作。',
                    'CQL_NB 在 CQL 基础上扩大 penalty 覆盖范围，并加入 target smoothing，让随机动作、当前策略动作和下一步策略动作都被更谨慎地约束。',
                    '这一组适合理解离线强化学习为什么重视 critic：不是只看动作像不像，还要看 Q 值估计是否稳定、是否过度乐观。',
                ],
            },
        },
    },
    'ddpg': {
        'key': 'ddpg',
        'title': 'DDPG / DDPG_NB',
        'base_name': 'DDPG',
        'nb_name': 'DDPG_NB',
        'summary': '这一组页面按照当前代码命名来展示，但从实现上看，二者都已经不是最原始的 DDPG，而是离线场景下做过增强的 actor-critic 版本。',
        'sections': {
            'beginner': {
                'title': '小白先看',
                'lead': '可以把它理解成“用 critic 告诉 actor 什么动作更好”。这一对方法的重点，是看代码里如何在 DDPG 主线上加入更稳的离线训练约束。',
                'bullets': [
                    'DDPG：当前脚本实现更接近 advantage-weighted 的离线 actor-critic。',
                    'DDPG_NB：当前脚本实现更接近 BC 正则化的 DDPG/TD3-lite。',
                    '这一对的价值主要在于展示两种不同的增强思路。',
                ],
            },
            'flow': {
                'title': '方法流程',
                'steps': [
                    '读取完整 transition 数据。',
                    '划分训练集与验证集。',
                    '计算状态标准化和 reward 统计量。',
                    '构建 actor、双 critic 及其 target 网络。',
                    '先更新 critic，再按延迟策略更新 actor。',
                    '保存模型、训练记录和曲线。',
                ],
            },
            'design': {
                'title': '核心设计',
                'cards': [
                    design_card(
                        '数据处理',
                        'DDPG 这组方法也需要知道“做了动作以后发生了什么”，所以它看的不是单独一行状态，而是一条完整的交互记录。',
                        '两者都使用 observations、actions、rewards、next_observations、terminals，并据此计算 critic 的 TD 目标和 actor 的更新方向。',
                    ),
                    design_card(
                        '网络结构',
                        'actor 负责提出动作，critic 负责判断动作值不值。target network 像一个更新慢一点的参考答案，防止当前网络一边改一边把目标也带偏。',
                        '两者都采用 actor + 双 critic + target network 的结构，critic 使用 state-action 输入，target actor/critic 通过 tau 做软更新。',
                    ),
                    design_card(
                        'Base 版本增强',
                        'Base 版更像“优先模仿那些看起来更有优势的动作”：如果某个数据动作比当前策略动作更值得做，它的学习权重就更高。',
                        'DDPG.py 中 actor 更新采用 advantage-weighted BC 风格，根据 data action 与 policy action 的 Q 值差构造 advantage，再计算加权行为克隆损失。',
                    ),
                    design_card(
                        'NB 版本增强',
                        'NB 版更像给 actor 同时提两个要求：一方面动作要让 critic 觉得好，另一方面不要离数据集里的真实动作太远。',
                        'DDPG_NB.py 中 actor loss 由 q_term 与 bc_term 组合而成，并用 alpha_bc 自适应缩放 Q 项，使策略优化和行为克隆约束共同生效。',
                    ),
                ],
            },
            'params': {
                'title': '参数说明',
                'basic': [
                    ('hdf5_file', '数据集路径'), ('batch_size', '批大小'), ('num_epochs', '训练轮数'), ('lr_actor / lr_critic', '学习率'), ('gamma / tau', '折扣与软更新'), ('hidden', '隐藏层结构'), ('output_dir', '输出目录')
                ],
                'advanced': [
                    ('reward_scale', '是否做 reward 标准化'), ('policy_noise / noise_clip', '目标动作扰动'), ('policy_delay', '延迟更新 actor'), ('DDPG.aw_lambda / weight_clip', '优势加权参数'), ('DDPG_NB.alpha_bc / bc_coef', 'BC 正则参数')
                ],
            },
            'code': {
                'title': '代码块讲解',
                'items': [
                    ('AIcodes/DDPG.py -> OfflineAWDDPGAgent', 'Base 版本的核心 agent，actor 更新使用 advantage-weighted BC。'),
                    ('AIcodes/DDPG.py -> train_step', '先更新 critic，再按权重更新 actor。'),
                    ('AIcodes/DDPG_NB.py -> OfflineDDPGBCAgent', 'NB 版本的核心 agent，actor 更新使用 q_loss + bc_loss。'),
                    ('AIcodes/DDPG_NB.py -> train_step', '通过 alpha_bc 自适应缩放 Q 项，再叠加行为克隆项。'),
                ],
            },
            'io': {
                'title': '输入输出与产物',
                'inputs': ['代码文件：AIcodes/DDPG.py、AIcodes/DDPG_NB.py', '数据文件：halfcheetah_medium-v2.hdf5', '完整 transition 数据'],
                'outputs': ['checkpoints/DDPG/ 或 checkpoints/DDPG_NB/', 'best_model.pt', 'train_config.json', 'training_records.csv', 'actor/critic/bc 或 weight 曲线图'],
            },
            'compare': {
                'title': 'Base / NB 对比',
                'rows': [
                    ('actor 更新方式', 'advantage-weighted BC', 'Q 项 + BC 项的组合更新'),
                    ('增强重点', '根据 advantage 调整样本权重', '直接给 actor 加 BC 正则'),
                    ('可解释重点', '为什么高优势样本更重要', '为什么要让 actor 不偏离数据动作太远'),
                    ('工程理解', '更偏“加权学习”', '更偏“约束学习”'),
                ],
            },
            'conclusion': {
                'title': '小结',
                'text': '这一对页面更适合拿来展示“同一主线下的两种增强方式”：一种是优势加权，一种是 BC 正则。对理解离线 actor 更新很有帮助。',
                'points': [
                    'DDPG 这一组可以理解成“actor 负责做动作，critic 负责打分”，训练重点是让 actor 学会产生 critic 认为更好的动作。',
                    'Base 版本更偏优势加权：如果数据里的某个动作比当前策略动作更有优势，就让模型更重视这个样本。',
                    'DDPG_NB 更偏 BC 正则：actor 追求高 Q 值的同时，还要尽量贴近离线数据里的真实动作，减少乱跑到数据外区域。',
                    '这一组适合用来比较两种离线 actor 更新思路：一种是“好样本多学”，另一种是“策略别偏太远”。',
                ],
            },
        },
    },
    'sac': {
        'key': 'sac',
        'title': 'SAC / SAC_NB',
        'base_name': 'SAC',
        'nb_name': 'SAC_NB',
        'summary': '这一组方法体现了“最大熵 actor-critic”在离线强化学习中的工程化实现：既看收益，也保留一定随机性。',
        'sections': {
            'beginner': {
                'title': '小白先看',
                'lead': '可以把 SAC 理解成“既想学到高收益动作，也不想太死板”。SAC_NB 则在这个基础上再加一层行为克隆约束。',
                'bullets': [
                    'SAC：当前脚本实现更接近 AWAC-SAC，即 actor 更新里带有 advantage-weighted 思路。',
                    'SAC_NB：当前脚本实现更接近 SAC + BC。',
                    '这组方法适合展示随机策略网络和 alpha 自动调节。',
                ],
            },
            'flow': {
                'title': '方法流程',
                'steps': [
                    '读取完整 transition 数据。',
                    '划分训练集与验证集。',
                    '计算状态标准化和 reward 统计量。',
                    '构建 Gaussian actor、双 critic 与 target critics。',
                    '更新 critic、actor 和 alpha。',
                    '保存训练历史和模型。',
                ],
            },
            'design': {
                'title': '核心设计',
                'cards': [
                    design_card(
                        '随机策略网络',
                        'SAC 不会只给一个死板动作，而是给出一类可能动作，再从里面采样。这样策略既能追求高收益，也能保留一点灵活性。',
                        '两者都使用 GaussianActor，网络输出 mu 与 log_std，通过重参数采样得到动作，并用 tanh 将动作压到合法范围。',
                    ),
                    design_card(
                        'Base 版本增强',
                        'Base 版会更愿意学习那些“看起来比当前策略更好”的数据动作，相当于从离线数据里挑更值得模仿的经验。',
                        'SAC.py 的 actor 更新采用 AWAC 风格，根据 data_q 与 v_pi 的差值计算 advantage 权重，再对数据动作的 log_prob 做加权优化。',
                    ),
                    design_card(
                        'NB 版本增强',
                        'NB 版除了 SAC 自己的策略优化，还加了一条“别偏离数据动作太远”的约束，适合离线场景里减少乱试动作。',
                        'SAC_NB.py 的 actor 更新采用 sac_loss + bc_loss：前者提升策略价值并考虑熵项，后者用行为克隆约束策略靠近数据集动作。',
                    ),
                    design_card(
                        '温度参数',
                        'alpha 可以理解成“探索和收益的平衡旋钮”：alpha 大时策略更随机，alpha 小时策略更专注于当前认为好的动作。',
                        '两者都支持 alpha_init 与 learn_alpha，通过目标熵自动调整 alpha，使策略随机性和 Q 值优化保持平衡。',
                    ),
                ],
            },
            'params': {
                'title': '参数说明',
                'basic': [
                    ('data_path / save_dir', '数据路径与输出目录'), ('hidden_sizes', '隐藏层结构'), ('lr_actor / lr_critic / lr_alpha', '学习率'), ('batch_size / num_epochs', '训练规模'), ('gamma / tau', '折扣与软更新')
                ],
                'advanced': [
                    ('alpha_init / learn_alpha', '温度参数及是否学习'), ('log_std_min / log_std_max', '策略方差范围'), ('SAC.awac_lambda / weight_clip', 'AWAC 权重相关参数'), ('SAC_NB.bc_coef', '行为克隆约束系数'), ('reward_scale', 'reward 标准化开关')
                ],
            },
            'code': {
                'title': '代码块讲解',
                'items': [
                    ('AIcodes/SAC.py -> GaussianActor', '实现随机策略，输出动作和对数概率。'),
                    ('AIcodes/SAC.py -> OfflineAWACSACAgent', 'Base 版本的核心 agent，带 AWAC 风格 actor 更新。'),
                    ('AIcodes/SAC_NB.py -> OfflineSACAgent', 'NB 版本的核心 agent，actor 更新为 sac_loss + bc_loss。'),
                    ('AIcodes/SAC.py / SAC_NB.py -> alpha 更新', '自动调节温度参数 alpha。'),
                ],
            },
            'io': {
                'title': '输入输出与产物',
                'inputs': ['代码文件：AIcodes/SAC.py、AIcodes/SAC_NB.py', '数据文件：halfcheetah_medium-v2.hdf5', '完整 transition 数据'],
                'outputs': ['checkpoints/SAC/ 或 checkpoints/SAC_NB/', 'best_model.pt', 'train_config.json', 'training_history.csv', '训练曲线图'],
            },
            'compare': {
                'title': 'Base / NB 对比',
                'rows': [
                    ('actor 更新方式', 'AWAC 风格加权 log_prob 学习', 'SAC loss + BC loss'),
                    ('增强重点', '根据 advantage 调整学习权重', '直接加入行为克隆约束'),
                    ('共同特征', 'Gaussian actor、双 critic、alpha 自动调节', 'Gaussian actor、双 critic、alpha 自动调节'),
                    ('适合理解', '理解离线随机策略如何加权学习', '理解随机策略如何加入 BC 约束'),
                ],
            },
            'conclusion': {
                'title': '小结',
                'text': 'SAC 这一对页面最适合帮助用户理解“随机策略 + 双 critic + alpha”这条主线，以及在离线训练时如何加入额外约束。',
                'points': [
                    'SAC 可以理解成“既追求高收益，也保留一定随机性”：它不会只输出一个固定动作，而是从一个动作分布中采样。',
                    '双 critic 用来降低 Q 值高估，alpha 温度参数像一个旋钮，用来平衡“多探索一点”和“更贪心一点”。',
                    'SAC 的 Base 版本更偏 AWAC 风格，会更重视 advantage 更高的数据动作；SAC_NB 则额外加入 BC loss，强调不要偏离离线数据动作。',
                    '这一组适合理解随机策略在离线场景里的难点：既要学到高收益动作，也要避免策略跑到数据集没有覆盖的地方。',
                ],
            },
        },
    },
    'td3': {
        'key': 'td3',
        'title': 'TD3 / TD3_NB',
        'base_name': 'TD3',
        'nb_name': 'TD3_NB',
        'summary': '这一组方法体现了双 critic、目标动作加噪声和延迟策略更新这些经典稳定化技巧。',
        'sections': {
            'beginner': {
                'title': '小白先看',
                'lead': '如果把 DDPG 理解成“critic 指导 actor”，那 TD3 可以看成“让这个指导过程更稳一些”。TD3_NB 则在此基础上再加入 BC 正则。',
                'bullets': [
                    'TD3：双 critic，目标动作加噪声，actor 延迟更新。',
                    'TD3_NB：在 TD3 主线上加入 BC 项和 reward scaling。',
                    '很适合用来展示“经典稳定化技巧 + 离线增强”的组合。',
                ],
            },
            'flow': {
                'title': '方法流程',
                'steps': [
                    '读取完整 transition 数据。',
                    '划分训练集与验证集。',
                    '计算状态标准化；TD3_NB 还计算 reward 统计量。',
                    '构建 actor、critic 与 target networks。',
                    '先更新 critic，再按 policy_delay 更新 actor。',
                    '保存训练日志、图像与最佳模型。',
                ],
            },
            'design': {
                'title': '核心设计',
                'cards': [
                    design_card(
                        '双 critic',
                        'TD3 像找两个老师给同一个动作打分，然后采用更低的分数。这样模型不会因为某个老师过于乐观就误以为动作特别好。',
                        '两者都使用两个 Q 网络，并在目标值中取 min(Q1, Q2)，这是 TD3 用来降低 Q 值高估的核心技巧。',
                    ),
                    design_card(
                        '目标动作扰动',
                        '给目标动作加一点小噪声，相当于检查“附近动作是不是也还可以”。这样 actor 不容易钻某个 critic 评分尖峰的空子。',
                        '两者都会对 target action 加 policy_noise，并用 noise_clip 截断，再将动作限制在合法范围内，用于平滑 target critic 的估计。',
                    ),
                    design_card(
                        '延迟更新',
                        'critic 要先多学几步，把评分标准变稳一点，actor 再根据这个评分去改动作。否则 actor 太早跟着不稳定评分跑，训练会抖。',
                        'actor 不会每一步都更新，而是按 policy_delay 延迟更新；critic 更新更频繁，actor 和 target networks 按固定间隔更新。',
                    ),
                    design_card(
                        'NB 增强',
                        'NB 版在“追求高 Q 值”的同时，加上“别偏离数据动作太远”的提醒，更适合不能在线试错的离线数据训练。',
                        'TD3_NB 在 actor 更新中组合 q_term 与 bc_term，并支持 reward_scale，让策略优化、行为克隆约束和奖励尺度处理同时参与训练。',
                    ),
                ],
            },
            'params': {
                'title': '参数说明',
                'basic': [
                    ('data_path / save_dir', '路径参数'), ('batch_size', '批大小'), ('lr_actor / lr_critic', '学习率'), ('gamma / tau', '折扣与软更新'), ('policy_noise / noise_clip', '目标动作噪声'), ('policy_delay', '延迟更新步长'), ('hidden', '隐藏层结构')
                ],
                'advanced': [
                    ('max_action', '动作范围上限'), ('TD3_NB.alpha_bc / bc_coef', 'BC 正则参数'), ('TD3_NB.reward_scale', 'reward 标准化'), ('grad_clip / weight_decay', '训练稳定性参数')
                ],
            },
            'code': {
                'title': '代码块讲解',
                'items': [
                    ('AIcodes/TD3.py -> TD3Trainer', 'Base 版本训练器，实现双 critic 与延迟更新。'),
                    ('AIcodes/TD3.py -> train_step', '目标动作加噪声并更新 critic，再按延迟更新 actor。'),
                    ('AIcodes/TD3_NB.py -> TD3NBTrainer', 'NB 版本训练器，在 actor 更新中引入 BC 项。'),
                    ('AIcodes/TD3_NB.py -> train_step', '将 q_term 与 bc_term 组合成 actor_loss。'),
                ],
            },
            'io': {
                'title': '输入输出与产物',
                'inputs': ['代码文件：AIcodes/TD3.py、AIcodes/TD3_NB.py', '数据文件：halfcheetah_medium-v2.hdf5', '完整 transition 数据'],
                'outputs': ['checkpoints/TD3/ 或 checkpoints/TD3_NB/', 'best_model.pt', 'train_config.json', 'training_log.csv', 'critic_loss.png / actor_loss.png / q_value.png'],
            },
            'compare': {
                'title': 'Base / NB 对比',
                'rows': [
                    ('核心主线', '经典 TD3 稳定化训练', 'TD3 + BC 正则'),
                    ('critic 设计', '双 critic', '双 critic'),
                    ('actor 更新', '纯 Q 导向更新', 'Q 项 + BC 项'),
                    ('额外增强', '目标动作加噪声、延迟更新', '再加 reward_scale 与 BC 正则'),
                ],
            },
            'conclusion': {
                'title': '小结',
                'text': 'TD3 这一对页面适合拿来说明“经典 RL 稳定技巧”如何迁移到离线场景，以及在离线训练中为什么常常还要额外加 BC 约束。',
                'points': [
                    'TD3 可以看成更稳的 DDPG：它用双 critic、目标动作加噪声和延迟更新，尽量减少训练中的 Q 值高估和策略抖动。',
                    '目标动作扰动让模型不要只盯着某个很尖锐的高分动作，延迟更新则让 critic 先学稳一点，再指导 actor。',
                    'TD3_NB 在 TD3 主线基础上加入 BC 正则和 reward scaling，让 actor 在追求高 Q 值时仍然靠近离线数据动作。',
                    '这一组适合展示“经典稳定化技巧 + 离线约束”的组合：先让评价更稳，再让策略别偏离数据太远。',
                ],
            },
        },
    },
}


SECTION_ORDER = ['beginner', 'flow', 'design', 'params', 'code', 'io', 'compare', 'conclusion']


def modelsinfo(request):
    pair = request.GET.get('pair', 'bc')
    if pair not in ALGORITHM_GROUPS:
        pair = 'bc'

    current = ALGORITHM_GROUPS[pair]
    pairs = [ALGORITHM_GROUPS[key] for key in ['bc', 'cql', 'ddpg', 'sac', 'td3']]

    context = {
        'page_title': '算法介绍',
        'pairs': pairs,
        'current_pair': current,
        'current_key': pair,
        'section_order': SECTION_ORDER,
    }
    return render(request, 'modelsinfo/modelsinfo.html', context)

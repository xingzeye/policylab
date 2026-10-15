# 答辩追问 Q&A 整理

这份文档整理你前面问过的问题和对应回答，并补充了一些答辩时容易被追问的点。它适合答辩前快速复习，尤其是老师问“代码里怎么实现的”“这些数字怎么来的”“loss 怎么理解”的时候。

## 1. 项目整体怎么介绍

### Q1：这个项目一句话怎么说

可以这样说：

```text
本项目是一个离线强化学习实验展示系统。训练部分使用 PyTorch 在 AIcodes 目录中实现 BC、CQL、DDPG、SAC、TD3 及其 NB 改进版本；展示部分使用 Django 读取训练产生的 checkpoint、CSV 日志和曲线图，在网页中展示数据集说明、算法介绍、单算法结果和 Base/NB 对比。
```

注意：不要说这是 ResNet、Unet 图像识别项目。这个项目不是图像识别，而是离线强化学习控制任务。

### Q2：如果老师问“模型在哪里”

回答：

```text
训练好的模型保存在 AIcodes/checkpoints/算法名/best_model.pt 里。例如 BC 的模型在 AIcodes/checkpoints/BC/best_model.pt，TD3_NB 的模型在 AIcodes/checkpoints/TD3_NB/best_model.pt。
```

这些 `.pt` 文件是 PyTorch checkpoint。

### Q3：如果老师问“Django 有没有直接训练模型”

回答：

```text
Django 不直接训练模型。模型训练由 AIcodes 目录下的 PyTorch 脚本完成。Django 主要负责读取训练好的结果文件，包括 best_model.pt、training_log.csv、training_history.csv 和曲线图片，然后展示到页面上。
```

## 2. Django 页面是怎么显示出来的

### Q4：Django 的页面请求流程是什么

通用流程：

```text
浏览器访问 URL
    -> qlearn/urls.py 匹配路由
    -> include 到对应 app 的 urls.py
    -> 调用 views.py 中的 view 函数
    -> view 函数读取文件或配置字典
    -> 组装 context
    -> render(request, template, context)
    -> 模板生成 HTML
    -> 浏览器显示页面
```

项目总路由在 `qlearn/urls.py`：

```python
urlpatterns = [
    path('admin/', admin.site.urls),
    path("datainfo/", include("apps.datainfo.urls")),
    path("modelsinfo/", include("apps.modelsinfo.urls")),
    path("results/", include("apps.results.urls")),
    path("compare/", include("apps.compare.urls")),
]
```

对应关系：

| URL | 作用 |
| --- | --- |
| `/datainfo/` | 数据集介绍 |
| `/modelsinfo/` | 算法说明 |
| `/results/` | 单算法训练结果 |
| `/compare/` | Base/NB 算法对比 |
| `/admin/` | Django 后台 |

## 3. Django 的 context 是什么

### Q5：context 是什么

`context` 可以理解为：**后端传给 HTML 模板的数据包**。

例如后端：

```python
context = {
    "selected_algorithm": "BC",
    "summary_cards": [
        {"label": "验证误差", "value": "0.016500"},
        {"label": "验证集 MSE", "value": "0.033168"},
    ],
}
return render(request, "results/results.html", context)
```

模板里：

```html
{{ selected_algorithm }}

{% for card in summary_cards %}
    {{ card.label }}
    {{ card.value }}
{% endfor %}
```

所以答辩说法：

```text
Django 的 context 是后端 view 函数传给前端模板的数据字典。我的项目里，算法名称、训练指标、图片路径、文件列表等内容都会先放进 context，然后模板通过 {{ 变量名 }} 或 {% for %} 渲染到页面上。
```

## 4. view 函数如何读取本地文件和配置字典

### Q6：view 函数如何读取 Python 配置字典

例如算法介绍页 `modelsinfo` 的内容写在：

```text
apps/modelsinfo/views.py
```

核心是 `ALGORITHM_GROUPS` 字典：

```python
ALGORITHM_GROUPS = {
    "bc": {
        "title": "BC / BC_NB",
        "summary": "...",
        "sections": {
            "beginner": {...},
            "flow": {...},
            "design": {...},
        },
    },
}
```

view 函数根据 URL 参数取内容：

```python
pair = request.GET.get("pair", "bc")
current = ALGORITHM_GROUPS[pair]
```

访问：

```text
/modelsinfo/?pair=td3
```

就会取：

```python
ALGORITHM_GROUPS["td3"]
```

### Q7：view 函数如何读取本地实验文件

例如 `results` 页面读取：

```text
AIcodes/checkpoints/BC/training_log.csv
AIcodes/checkpoints/BC/evaluation_summary.csv
AIcodes/checkpoints/BC/train_val_loss.png
```

后端会先拼路径：

```python
root = os.path.join(settings.BASE_DIR, "AIcodes/checkpoints")
algo_dir = os.path.join(root, selected_algorithm)
```

再读取 CSV：

```python
pd.read_csv(csv_path)
```

如果是图片或模型文件，不一定直接读内容，而是生成 URL，让浏览器再请求文件接口：

```text
/results/file/BC/train_val_loss.png/
```

然后后端用：

```python
FileResponse(open(file_path, "rb"))
```

把文件返回给浏览器。

答辩说法：

```text
说明类页面主要读取 Python 字典，结果类页面主要读取 checkpoint 目录下的本地文件。view 函数把这些内容整理成 context，最后交给模板显示。
```

## 5. compare 页面路由怎么解释

### Q8：这段 urlpatterns 是什么意思

代码：

```python
urlpatterns = [
    path('', views.compare_view, name='compare'),
    path('asset/<path:relative_path>/', views.compare_asset, name='asset'),
]
```

解释：

| 路由 | 调用函数 | 作用 |
| --- | --- | --- |
| `/compare/` | `compare_view` | 显示算法对比主页面 |
| `/compare/asset/BC/xxx.png/` | `compare_asset` | 返回对比页面需要的图片或文件 |

`<path:relative_path>` 是路径参数。

例如：

```text
/compare/asset/TD3/critic_loss.png/
```

Django 会调用：

```python
compare_asset(request, relative_path="TD3/critic_loss.png")
```

然后函数去 `AIcodes/checkpoints/TD3/critic_loss.png` 找文件并返回。

## 6. compare_view 函数做了什么

### Q9：compare_view 的整体逻辑

核心代码：

```python
def compare_view(request):
    pair_key = request.GET.get('pair', 'BC')
    if pair_key not in PAIR_CONFIGS:
        pair_key = 'BC'

    pair_cfg = PAIR_CONFIGS[pair_key]
    base_summary = _method_summary(pair_cfg['base_dir'])
    nb_summary = _method_summary(pair_cfg['nb_dir'])

    metric_rows = _build_metric_rows(pair_cfg, base_summary, nb_summary)
    chart_rows = _build_chart_rows(pair_cfg, base_summary, nb_summary)
    conclusion = _compare_conclusion(...)
```

流程：

```text
读取 URL 参数 pair
    -> 例如 /compare/?pair=TD3
    -> pair_key = TD3
    -> 从 PAIR_CONFIGS 中取 TD3 / TD3_NB 配置
    -> 分别汇总 TD3 和 TD3_NB 的结果目录
    -> 生成指标对比表
    -> 生成曲线对比图数据
    -> 生成一句话结论
    -> context 传给 compare.html
```

答辩说法：

```text
compare_view 根据用户选择的算法对，读取 Base 和 NB 两个 checkpoint 目录，整理状态、指标、图表和结论，最后通过 context 传给 compare.html 渲染成对比页面。
```

### Q10：compare_view 里的 context 字段分别是什么

```python
context = {
    'pair_key': pair_key,
    'pair_cfg': pair_cfg,
    'pair_options': pair_options,
    'base_summary': base_summary,
    'nb_summary': nb_summary,
    'metric_rows': metric_rows,
    'chart_rows': chart_rows,
    'conclusion': conclusion,
    'checkpoints_root': _checkpoints_dir(),
}
```

解释：

| 字段 | 含义 |
| --- | --- |
| `pair_key` | 当前选择的算法对，例如 `TD3` |
| `pair_cfg` | 当前算法对的配置，例如标题、Base 目录、NB 目录、核心指标 |
| `pair_options` | 页面下拉框选项列表 |
| `base_summary` | Base 算法结果汇总，例如 TD3 的状态、文件、指标 |
| `nb_summary` | NB 算法结果汇总，例如 TD3_NB 的状态、文件、指标 |
| `metric_rows` | 关键指标对比表 |
| `chart_rows` | 曲线图对比数据 |
| `conclusion` | 页面顶部的一句话结论 |
| `checkpoints_root` | 结果文件根目录 `AIcodes/checkpoints` |

## 7. 为什么后台管理端能显示这些页面

### Q11：为什么能在管理端看到实验页面

因为项目用了 `django-simpleui`，并在 `settings.py` 里配置了后台菜单：

```python
SIMPLEUI_CONFIG = {
    'menus': [
        {
            'name': '实验介绍',
            'models': [
                {'name': '数据集介绍', 'url': '/datainfo/'},
                {'name': '算法说明', 'url': '/modelsinfo/'},
            ]
        },
        {
            'name': '实验结果',
            'models': [
                {'name': '单算法结果展示', 'url': '/results/'},
                {'name': '结果对比分析', 'url': '/compare/'},
            ]
        },
    ]
}
```

所以后台左侧菜单只是提供入口链接。

注意：

```text
这些页面不是通过 admin.py 注册的模型管理页。
它们是普通 Django 页面，只是被 simpleui 菜单链接到了后台里。
```

答辩说法：

```text
后台能显示这些页面，是因为 simpleui 的菜单配置把 /datainfo/、/modelsinfo/、/results/、/compare/ 加到了后台菜单中。点击菜单后，本质上还是访问普通 Django 路由，由对应 views.py 渲染页面。
```

## 8. 页面上的数字是怎么传进来的

### Q12：结果页上“验证误差、MSE、MAE”这些数字怎么来的

以 BC 页面为例，数字来源：

| 页面显示 | 来源文件 | 字段 |
| --- | --- | --- |
| 验证误差 | `AIcodes/checkpoints/BC/training_log.csv` | `val_loss` |
| 验证集 MSE | `AIcodes/checkpoints/BC/evaluation_summary.csv` | `mse`，`split=val` |
| 验证集 MAE | `AIcodes/checkpoints/BC/evaluation_summary.csv` | `mae`，`split=val` |

后端流程：

```text
访问 /results/?algorithm=BC
    -> results_overview()
    -> pandas 读取 training_log.csv 和 evaluation_summary.csv
    -> _build_summary_cards() 提取关键指标
    -> summary_cards 放进 context
    -> results.html 循环 summary_cards
    -> 页面显示数字
```

模板中：

```html
{% for card in summary_cards %}
    <div class="label-text">{{ card.label }}</div>
    <div class="value-text">{{ card.value }}</div>
{% endfor %}
```

答辩说法：

```text
这些数字不是前端写死的，而是后端从训练日志 CSV 中读取后传给模板的。
```

## 9. 每个算法训练代码和神经网络在哪里

### Q13：训练代码和神经网络位置

| 算法 | 训练代码 | 神经网络 |
| --- | --- | --- |
| BC | `AIcodes/BC.py` 的 `train_one_epoch`、`train_bc` | `BCPolicy` |
| BC_NB | `AIcodes/BC_NB.py` 的 `train_one_epoch`、`train_bc` | `BCPolicy` |
| CQL | `AIcodes/CQL.py` 的 `train_cql_improved` | `MLP` 构建 actor 和 critic |
| CQL_NB | `AIcodes/CQL_NB.py` 的 `train_cql_improved` | `MLP` |
| DDPG | `AIcodes/DDPG.py` 的 `train_step`、`train_ddpg_improved` | `Actor`、`Critic` |
| DDPG_NB | `AIcodes/DDPG_NB.py` 的 `train_step`、`train_ddpg_improved` | `Actor`、`Critic` |
| SAC | `AIcodes/SAC.py` 的 `train_step`、`train_sac` | `GaussianActor`、`Critic` |
| SAC_NB | `AIcodes/SAC_NB.py` 的 `train_step`、`train_sac` | `GaussianActor`、`Critic` |
| TD3 | `AIcodes/TD3.py` 的 `train_step`、`main` | `Actor`、`Critic` |
| TD3_NB | `AIcodes/TD3_NB.py` 的 `train_step`、`main` | `Actor`、`Critic` |

答辩说法：

```text
训练代码统一在 AIcodes 目录下。BC 系列是策略网络 BCPolicy，CQL 使用 MLP 构建 actor 和 critic，DDPG/TD3 是 Actor 和 Critic，SAC 是 GaussianActor 和 Critic。
```

## 10. train_sac 和 train_step 有什么区别

### Q14：train_sac 和 train_step 的区别

一句话：

```text
train_sac 是整体训练流程，train_step 是一次 batch 更新。
```

`train_sac` 负责：

```text
循环 epoch
遍历训练集
遍历验证集
记录日志
保存 best_model.pt
保存 CSV 和曲线图
```

`train_step` 负责：

```text
取一个 batch
更新 critic
更新 actor
更新 alpha
soft update target critic
返回当前 batch 的 loss 和指标
```

答辩说法：

```text
train_sac 是外层训练函数，控制整个训练过程；train_step 是 agent 内部的单步训练函数，每次处理一个 batch，并完成一次网络参数更新。
```

## 11. 训练日志 CSV 里的参数是什么意思

### Q15：CSV 里的字段怎么看

这些 CSV 是训练日志，不是训练超参数。真正的超参数一般在 `train_config.json`。

通用字段：

| 字段 | 含义 |
| --- | --- |
| `epoch` | 第几轮训练 |
| `train_...` | 训练集指标 |
| `val_...` | 验证集指标 |
| `..._loss` | 损失值 |
| `..._lr` / `lr` | 当前学习率 |

### BC 日志字段

文件：

```text
AIcodes/checkpoints/BC/training_log.csv
```

| 字段 | 含义 |
| --- | --- |
| `train_loss` | 训练集动作预测误差 |
| `val_loss` | 验证集动作预测误差 |
| `lr` | 当前学习率 |

### CQL 日志字段

| 字段 | 含义 |
| --- | --- |
| `train_actor_loss` / `val_actor_loss` | actor 策略损失 |
| `train_critic_loss` / `val_critic_loss` | critic 价值网络损失 |
| `train_cql_penalty` / `val_cql_penalty` | CQL 保守惩罚项 |
| `train_q1_mse` / `train_q2_mse` | 两个 critic 的 MSE |
| `train_bc_loss` / `val_bc_loss` | actor 的行为克隆约束误差 |
| `actor_lr` / `critic_lr` | actor 和 critic 学习率 |

### DDPG 日志字段

| 字段 | 含义 |
| --- | --- |
| `actor_loss` | actor 损失 |
| `critic_loss` | critic 损失 |
| `bc_loss` | 行为克隆误差 |
| `advantage` | 数据动作相对当前策略动作的优势 |
| `weight` | 根据 advantage 算出的权重 |
| `Q1` / `Q2` | 两个 critic 的 Q 值 |

### SAC 日志字段

| 字段 | 含义 |
| --- | --- |
| `critic1_loss` / `critic2_loss` | 两个 critic 的损失 |
| `actor_loss` | actor 策略损失 |
| `Q1` / `Q2` | critic 估计的 Q 值 |
| `log_prob_data` | 数据动作在当前策略下的概率对数 |
| `alpha` | SAC 熵温度系数 |
| `advantage` / `weight` | AWAC 风格权重相关指标 |

### TD3 日志字段

| 字段 | 含义 |
| --- | --- |
| `critic_loss` | 双 critic 的 TD 损失 |
| `actor_loss` | actor 损失，通常是 `-Q` |
| `actor_proxy` | 验证阶段的 actor 代理指标 |
| `target_q` | TD target 目标 Q 值 |
| `current_q1` / `current_q2` | 当前 critic 预测 Q 值 |

## 12. TD3 页面三个指标怎么解释

### Q16：验证 critic 损失、验证 actor 代理指标、验证 target Q 是什么

页面上这三个通常来自 TD3 的：

```text
val_critic_loss
val_actor_proxy
val_target_q
```

解释：

| 页面字段 | 作用 |
| --- | --- |
| 验证 critic 损失 | 看 critic 价值网络在验证集上拟合 TD target 的误差 |
| 验证 actor 代理指标 | 用当前 actor 生成动作，再用 critic 评价；验证阶段不更新 actor，所以叫代理指标 |
| 验证 target Q | critic 学习时参考的目标 Q 值 |

答辩说法：

```text
critic loss 看评价网络学得准不准；actor proxy 看当前策略在 critic 眼里的价值；target Q 是训练 critic 时的目标值。它们要结合曲线看，不能只凭某一个数判断好坏。
```

## 13. actor loss 是怎么求的

### Q17：每个算法 actor loss 怎么算

| 算法 | actor / policy loss |
| --- | --- |
| BC | `loss = loss_fn(pred_act, batch_act)` |
| BC_NB | `loss = (sample_loss * batch_w).mean()` |
| CQL | `actor_loss = -lam * q_pi.mean() + bc_coef * bc_loss` |
| CQL_NB | 同 CQL |
| DDPG | `actor_loss = (weights * per_sample_bc).mean()` |
| DDPG_NB | `actor_loss = q_loss + bc_loss` |
| SAC | `actor_loss = -(weights * log_prob_data).mean()` |
| SAC_NB | `actor_loss = sac_loss + bc_coef * bc_loss` |
| TD3 | `actor_loss = -Q1(state, actor(state)).mean()` |
| TD3_NB | `actor_loss = q_term + bc_term` |

可以按三类理解：

```text
BC 类：动作监督误差
-Q 类：让 actor 输出高 Q 动作
-Q + BC 类：既追求高 Q，又不偏离数据动作
```

## 14. actor loss 是 Q 值吗

### Q18：actor loss 是不是 Q 值

回答：

```text
actor loss 不是 Q 值本身，而是用来训练 actor 的目标函数。有些算法会用 Q 值构造 actor loss，比如 TD3 的 actor loss = -Q。因为优化器最小化 loss，所以最小化 -Q 等价于最大化 Q。
```

举例：

```text
TD3:
actor_loss = -Q(state, actor(state))
```

但是 NB 版本通常还有 BC 项：

```text
actor_loss = -Q + BC_loss
```

所以不能简单说 actor loss 就是 Q 值。

## 15. MSE、SmoothL1 这些损失函数怎么说

### Q19：MSE 在项目里有什么作用

MSE 是均方误差：

```text
MSE = 平均((预测值 - 真实值)^2)
```

它在项目里主要有三种用途：

| 用途 | 出现位置 |
| --- | --- |
| 动作预测误差 | BC、BC_NB |
| critic TD 误差 | CQL、DDPG、SAC、TD3 |
| BC 正则项 | DDPG_NB、SAC_NB、TD3_NB、CQL 系列 |

答辩说法：

```text
MSE 是具体的误差计算方式。BC 系列用它衡量预测动作和数据动作的差距；actor-critic 算法用它训练 critic，让当前 Q 值接近 TD target；NB 改进版还用 MSE 作为 BC 正则，限制 actor 不要偏离离线数据动作。
```

### Q20：SmoothL1 是什么

SmoothL1 可以理解成比 MSE 更稳一点的回归损失。

答辩时简单说：

```text
SmoothL1 也是衡量预测动作和真实动作差距的损失函数，相比 MSE 对异常值没那么敏感，可以让训练更稳定。
```

## 16. critic loss 和 actor loss 的关系

### Q21：critic loss 和 actor loss 有什么关系

一句话：

```text
critic loss 训练“评价动作的人”，actor loss 训练“做动作的人”。
```

critic 负责：

```text
输入 state + action
输出 Q 值
```

critic loss 衡量：

```text
当前 Q 预测 和 TD target 差多少
```

actor 负责：

```text
输入 state
输出 action
```

actor loss 让 actor：

```text
输出 critic 认为价值更高的动作
```

训练关系：

```text
先训练 critic，让它学会评价动作价值
再训练 actor，让它根据 critic 的评价改进动作
```

如果 critic 不准，actor 就可能被错误引导。所以项目里有：

```text
TD3 的双 critic
CQL 的保守惩罚
NB 版的 BC 正则
```

这些都是为了让 actor 不要被不可靠的 Q 值带偏。

答辩说法：

```text
critic loss 用来训练价值网络，actor loss 用来训练策略网络。critic 给 actor 提供评价标准，actor 根据 critic 的评价更新策略。在离线强化学习里，为了防止 actor 只追求高 Q 而跑出数据分布，很多 NB 版本会在 actor loss 里加入 BC 正则。
```

## 17. Base 和 NB 怎么整体解释

### Q22：为什么有 NB 版本

可以这样说：

```text
Base 版本用于展示基础算法逻辑，NB 版本是在基础算法上加入更适合离线强化学习的改进，例如样本加权、BC 正则、保守约束或目标动作平滑。因为离线强化学习不能在线试错，所以 NB 版本的目的通常是让策略更稳定，不要过度偏离已有数据。
```

对应关系：

| 算法对 | NB 改进点 |
| --- | --- |
| BC / BC_NB | 高回报轨迹样本加权 |
| CQL / CQL_NB | 更强 CQL penalty，加入 next policy action 和 target smoothing |
| DDPG / DDPG_NB | 加入更明确的 Q 项 + BC 正则 |
| SAC / SAC_NB | SAC 目标 + BC 正则 |
| TD3 / TD3_NB | TD3 目标 + BC 正则和 reward scaling |

## 18. 答辩快速背诵版

### 18.1 项目整体

```text
我的项目分为训练和展示两部分。训练代码在 AIcodes 目录下，用 PyTorch 实现离线强化学习算法；展示部分用 Django 读取 checkpoints 下的模型、日志和曲线图，并渲染到网页上。
```

### 18.2 页面数据怎么来的

```text
后端 view 函数读取 Python 配置字典或本地 CSV、PNG、PT 文件，把数据整理成 context，再通过 render 传给 HTML 模板显示。
```

### 18.3 loss 怎么讲

```text
critic loss 训练 Q 值评价网络，actor loss 训练策略网络。MSE 常用于 critic 的 TD 误差和 BC 约束。actor loss 有些算法是 -Q，有些是 -Q + BC，有些是加权行为克隆，本质都是让策略输出更好的动作，同时在离线场景下不要偏离数据集。
```

### 18.4 后台为什么能打开页面

```text
后台使用 simpleui 配置了菜单入口，菜单链接到 /datainfo/、/modelsinfo/、/results/、/compare/。这些不是 admin 模型页，而是普通 Django 页面。
```

## 19. 最容易说错的地方

1. 不要说项目是 ResNet/Unet 图像识别；它是离线强化学习展示系统。
2. 不要说 Django 实时训练模型；训练由 `AIcodes/*.py` 完成。
3. 不要说 actor loss 就是 Q 值；准确说是用 Q 值和其他约束构造的训练目标。
4. 不要说后台页面都是 admin 注册模型；这些实验页面只是通过 simpleui 菜单链接进后台。
5. 不要只看单个 loss 数值判断算法好坏；强化学习结果要结合曲线、Q 值、是否 NaN、是否稳定一起看。


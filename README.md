# 策学 · PolicyLab

这是一个面向离线强化学习课程或毕业设计展示的 Django 项目，用于说明 HalfCheetah 数据集、算法原理、训练结果和算法对比。

「策学」寓意学习策略。PolicyLab 将算法直觉、代码讲解和真实训练产物放到同一个学习入口。

- GPT Sites：<https://policylab.xzy666999.chatgpt.site>（仅站点所有者可访问，需登录对应 ChatGPT 账号）
- GitHub Pages：<https://xingzeye.github.io/policylab/>
- GitHub 源码：<https://github.com/xingzeye/policylab>
- 算法轻松学：<https://xingzeye.github.io/policylab/learn/>

## 算法轻松学

首页 `/` 和新页面 `/learn/` 以生活类比、三步流程、可展开的简式公式、小问答和对照表讲解 BC、DDPG、TD3、SAC、CQL。页面单独说明项目中的 NB 改动，并区分经典算法与项目的离线改写。支持手机浏览、算法锚点直达和浏览器返回；关闭 JavaScript 时仍可阅读全部内容。

## 页面入口

- 首页 / 算法轻松学：`/`、`/learn/`
- 数据管理：`/datainfo/`
- 算法介绍：`/modelsinfo/`
- 单算法结果展示：`/results/`
- 算法对比分析：`/compare/`
- 管理后台：`/admin/`

## 算法介绍页更新

`/modelsinfo/` 中的“核心设计”模块已经改为更通俗的分层讲解形式。每个核心设计卡片包含：

- `通俗理解`：用更接近口语的方式解释算法为什么这样设计。
- `技术对应`：说明该设计在代码、数据流、网络结构或损失函数中的对应实现。

`/modelsinfo/` 中的“小结”模块也已改为分点说明。每组算法会用 3-4 条要点总结学习顺序、核心思想、NB 改进价值和适合理解方式。

覆盖的算法对包括：

- `BC / BC_NB`
- `CQL / CQL_NB`
- `DDPG / DDPG_NB`
- `SAC / SAC_NB`
- `TD3 / TD3_NB`

## 本地运行

推荐使用项目已有的 conda 环境：

```powershell
conda run -n benke python manage.py check
conda run -n benke python manage.py runserver 8005
```

然后访问：

```text
http://127.0.0.1:8005/
```

新环境可使用 Python 3.9 安装展示系统依赖：

```powershell
python -m pip install -r requirements.txt
python manage.py migrate
python manage.py runserver 8005
```

训练脚本需要 PyTorch、MuJoCo 等额外环境，展示和静态导出不需要运行训练。项目原有 Django 2.2 环境用于本地展示；GitHub Pages 上只发布静态文件。需要保持本地登录会话时，请通过环境变量 `DJANGO_SECRET_KEY` 提供稳定随机密钥；缺省时会生成进程内临时密钥。

## GitHub Pages 部署与更新

GitHub Pages 仅运行 HTML / CSS / JavaScript。仓库保留 Django 源码，`docs/` 保存导出的公开网站，Pages 的发布源为 `main` 分支的 `/docs`。推送该目录后，GitHub 自动构建和发布，不需要服务器或平台密钥。

在项目根目录运行：

```powershell
conda run -n benke python manage.py check
conda run -n benke python manage.py test apps.modelsinfo.tests -v 2
conda run -n benke python manage.py export_pages
git add -u
git add docs
git commit -m "Update PolicyLab learning site"
git push
```

新增源码文件应按实际改动显式 `git add`。`export_pages` 导出首页、学习页、数据页、5 组原理说明、10 个算法结果和 5 组对比的全部选择项，检查站内文件链接，并使用相对路径兼容仓库子路径。线上算法选择跳转到对应 HTML，无需 Django。

网站保留训练曲线、CSV 日志、配置及已有模型下载。完整数据集、账号密码、本地数据库、安装包和本地输出不进入仓库。数据页使用 `data/dataset_preview.json` 的完整行数与前 8 条样本，下载文件明确标为样本。若本地完整 CSV 更新，先运行：

```powershell
conda run -n benke python manage.py snapshot_dataset
```

然后重新导出、提交、推送。线上不提供管理后台、训练任务或动态文件管理；后台继续通过本地 `/admin/` 使用。撤回站点更新时，使用 `git revert <commit>` 后推送，以保留历史。

## GPT Sites 部署与更新

Sites 与 GitHub Pages 使用同一套静态导出，不改变原有学习内容与训练产物。Sites 的身份和静态目录记录在 `.openai/hosting.json`，项目 ID 为 `appgprj_6ad0bc9a25448191ba9741e7b32d1803`，后续更新应复用此 ID。`dist/` 是被 Git 忽略的 Sites 打包目录，`docs/` 仍供 GitHub Pages 使用。

在项目根目录生成 Sites 发布文件：

```powershell
conda run -n benke python manage.py export_pages --output dist
```

然后使用 Codex 的 Sites 技能获取此站点的临时源码凭据，通过官方 source helper 同步精确源码提交、打包并发布。临时凭据只通过隐藏的标准输入传递，不写入仓库或文件。GitHub 推送和 Sites 发布是两个独立动作；仅推送 GitHub 不会更新 Sites。

新 Sites 默认仅所有者可访问；该地址需要对应 ChatGPT 账号登录。更改可见范围应通过 Sites 的共享设置另行操作。Sites 仍是静态展示，不运行 Django 后台或训练任务。

## 本地文件注意事项

项目中包含较大的数据和模型相关文件，操作时不要批量删除文件或目录。

## 本次验证记录（2026-10-15）

Django 系统检查无问题，7 项测试通过；26 个公开页面成功导出，站内文件链接检查通过。从 GitHub 全新克隆后，在没有完整 CSV 和本地数据库的情况下仍可导出。桌面与手机布局、算法切换及三个原有页面的选择器经过浏览器检查；GitHub Pages 构建部署成功，主要线上页面、图片和样本下载返回 HTTP 200。本次没有重新训练模型或验证 MuJoCo 环境回报。

# AGENT 工作说明

## 项目概览

当前项目是一个 Django 展示系统，核心功能包括：

- HalfCheetah 数据集展示
- 离线强化学习算法说明
- 单算法训练结果展示
- Base / NB 算法对比分析

## 当前页面说明

项目公开名称为「策学 · PolicyLab」，仓库为 `xingzeye/policylab`。首页与 `/learn/` 使用 `apps/modelsinfo/learning.py` 和 `templates/modelsinfo/learn.html`；简洁教学内容放在 `LESSONS`，现有完整说明继续放在 `ALGORITHM_GROUPS`。不要把两种阅读层级合并成密集的长页面。

学习内容需区分经典算法与本地实现：DDPG.py 有优势加权模仿，SAC.py 为 AWAC 风格 Actor 目标；NB 是本项目版本标识。修改讲解应先核对 `AIcodes/`，不能根据文件名推断算法原样实现。

`apps.modelsinfo` 负责算法介绍页。该页面的数据主要写在 `apps/modelsinfo/views.py` 的 `ALGORITHM_GROUPS` 中，模板为 `templates/modelsinfo/modelsinfo.html`。

“核心设计”模块采用 `design.cards` 渲染。每张卡片应优先使用以下字段：

- `title`：卡片标题
- `plain`：通俗理解，面向初学者
- `technical`：技术对应，说明代码或算法含义

模板仍保留 `content` 字段兼容，避免旧数据缺少新字段时页面空白。

“小结”模块采用 `conclusion.points` 渲染分点说明。每组算法的小结应保持 3 条以上要点，覆盖学习顺序、核心思想、改进版价值或适合理解方式；模板仍保留 `conclusion.text` 作为兜底。

## 开发约束

- 禁止批量删除文件或目录。
- 不要使用 `del /s`、`rd /s`、`rmdir /s`、`Remove-Item -Recurse`、`rm -rf`。
- 如果必须删除文件，只能一次删除一个明确路径的文件。
- 修改代码时，同步维护 `PROJECT_GUIDE.md`、`README.md` 和 `AGENT.md`。
- 推荐在 conda 环境 `benke` 下运行检查和服务。

## 验证建议

修改算法介绍页后，至少运行：

```powershell
conda run -n benke python manage.py check
conda run -n benke python manage.py test apps.modelsinfo.tests -v 2
conda run -n benke python manage.py export_pages
```

学习页 / 静态部署的接续检查：

- `LearningAndPagesTests` 验证两个入口、五种算法、数据快照、完整静态导出、选择器改写和私密文件排除；现有四项原理页测试继续保留。
- 浏览器检查桌面及手机排版、算法点击切换、直接 `#sac` 锚点、展开问答、返回、原理页选择器、结果页与对比页选择器及图片加载。
- 导出后同时提交源码与 `docs/`；GitHub Pages 从 `main:/docs` 发布。推送后检查 Pages 构建完成及线上 HTTP / 浏览器结果。
- 原始 CSV 更新时先执行 `snapshot_dataset`。线上只展示 8 条预览，不能标成完整数据下载。
- `.gitignore` 排除账号密码、数据库、原始大数据、安装包和输出；发布前核对暂存文件列表。公开资源仅复制静态目录、数据预览图、样本 CSV 和结果文件白名单。
- `DJANGO_SECRET_KEY` 来自环境变量，缺省为进程内随机值；不要把真实密钥加入源码。临时值会使会话在服务重启后失效。
- `docs/` 为生成目录；导出器不会清空目录，也拒绝覆盖未标记且非空的目录。更换输出位置时必须指定独立目录。
- 当前仍沿用本地旧版 Django 依赖；若未来部署真实 Django 服务，应另立任务升级和验证，不能把 Pages 发布视为后端生产部署验证。

2026-10-15 发布已完成：Pages 构建 / 部署成功，主要线上路径返回 HTTP 200，浏览器学习页控制台无错误。桌面 / 手机布局、三种旧页面的静态选择器和图片已检查；全新克隆可在无完整数据、无数据库时导出。截图留在已忽略的 `output/playwright/`，不随源码发布。

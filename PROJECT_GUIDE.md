# 策学 · PolicyLab 项目说明

本项目是一个用于展示离线强化学习数据集、算法说明、单算法训练结果和算法对比结果的 Django 应用。

## 主要页面

- `/`、`learn/`：新增算法轻松学，统一品牌和全站导航。
- `datainfo/`：展示 HalfCheetah 数据集基础信息、字段说明、CSV 样本预览和数据分布图。
- `modelsinfo/`：展示 BC、CQL、DDPG、SAC、TD3 及其 NB 改进版本的算法说明。
- `results/`：展示单个算法的训练结果、核心图表、详细指标和结果文件。
- `compare/`：展示基础算法与改进算法的指标、曲线和结果文件对比。

## 算法说明页

`modelsinfo/` 页面按算法对展示内容，包括：

- 小白先看
- 方法流程
- 核心设计
- 参数说明
- 代码块讲解
- 输入输出与产物
- Base / NB 对比
- 小结

其中“核心设计”已采用分层解释方式：每张卡片同时展示“通俗理解”和“技术对应”。通俗理解用于帮助初学者快速明白设计目的，技术对应用于说明它在训练数据、网络结构、损失函数或稳定化技巧中的具体含义。

“小结”已采用分点式详细说明。每组算法会从学习顺序、核心思想、改进版价值和适合理解方式几个角度总结，避免只用一段短文字概括。

## 数据与算法文件

- 数据文件主要位于 `media/datainfo/` 和 `AIcodes/`。
- 算法脚本位于 `AIcodes/`。
- 训练结果位于 `AIcodes/checkpoints/`。

## 学习页与发布结构

- `apps/modelsinfo/learning.py`：5 组算法的精简教学内容及视图；经典思想与项目改写分开说明。
- `templates/modelsinfo/learn.html`：类比、流程、可展开公式、自测及横向对照表。
- `templates/includes/site_nav.html`：所有公开页面的统一导航。
- `static/policylab/`：独立学习页样式、渐进增强的算法切换和导航样式。
- `data/dataset_preview.json`、`media/datainfo/halfcheetah_preview.csv`：真实完整 CSV 的行数与 8 条预览样本。当前完整数据为 1,000,000 行；快照不等于完整数据文件。
- `apps/modelsinfo/management/commands/snapshot_dataset.py`：在拥有完整 CSV 的本地环境刷新快照。
- `apps/modelsinfo/management/commands/export_pages.py`：调用现有 Django 视图，导出公开页面和白名单资源、改写查询选择器及资源地址，并检查站内链接。
- `docs/`：GitHub Pages 发布产物。包含 26 个公开 HTML 页面和 1 个 404 页面，不手工编辑此目录中的业务内容。

发布仓库为 `xingzeye/policylab`，Pages 使用 `main:/docs`。导出沿用现有原理、结果和对比页面，不重写训练逻辑。算法选择器将原有查询参数映射到各自的静态 HTML；结果图片与下载转到 `assets/results/`。相对路径可同时支持本地子目录与 GitHub 项目路径。

本地 Django 保留原有后台和文件视图。Pages 不运行 Python，不连接数据库，也不启动训练。完整 CSV / HDF5 文件因体积较大留在本地；公开站的数据页明确显示快照说明。训练日志的 NaN 等异常提示按现有逻辑保留，不能把训练损失或图表当作独立回报评估。

## GPT Sites 发布配置

- `.openai/hosting.json`：持久化 Site ID `appgprj_6ad0bc9a25448191ba9741e7b32d1803`，声明 `static.directory: dist`。
- `dist/`：通过现有 `export_pages --output dist` 生成，沿用 26 个公开页面及结果文件白名单，不进入 Git。
- Sites 地址：`https://policylab.xzy666999.chatgpt.site`；新建站点保持仅所有者可访问，未改为公开共享。
- 源码同时保留 GitHub `origin` 与 Sites 平台源码同步。官方 helper 使用临时凭据推送确切提交，再从该提交构建的 `dist/` 打包发布。
- 原 `docs/` 与 GitHub Pages 发布设置继续保留。未来更新内容时，应分别导出两个目录并完成两端发布。

## 既有实现与验证

- 已实现：中文算法学习页、跨页面导航、首页入口、数据预览快照、公开页面静态导出和查询切换适配。
- 已验证：Django 系统检查、7 项测试、全部 26 个公开页面导出及本地文件链接检查。
- 已验证（2026-10-15）：1440px 桌面与 390px 手机浏览；学习页算法切换、直接锚点，原理 / 结果 / 对比页选择器，结果和对比图片加载；手机学习页无整页横向溢出。
- 已验证：从 GitHub 全新克隆的源码在没有完整 CSV 和本地数据库时，可再次导出 26 个公开页面，全部站内文件链接通过检查。
- 已部署：`https://xingzeye.github.io/policylab/`，GitHub Pages 构建及部署成功。首页、学习页、数据页、SAC 原理页、TD3_NB 结果页、CQL 对比页、结果图片及样本 CSV 均已确认 HTTP 200。
- 尚未验证：重新训练算法及在 MuJoCo 环境中复现回报；本次没有修改训练脚本。

## 开发注意事项

- 修改代码后，需要同步维护 `PROJECT_GUIDE.md`、`README.md` 和 `AGENT.md`。
- 不要批量删除文件或目录；如需删除多个文件，应让用户手动确认和处理。
- 推荐使用 conda 环境 `benke` 运行 Django 检查和本地服务。

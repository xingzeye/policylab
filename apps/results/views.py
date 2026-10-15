# -*- coding: utf-8 -*-
import math
import mimetypes
import os

import pandas as pd
from django.conf import settings
from django.http import FileResponse, Http404
from django.shortcuts import render
from django.utils.encoding import smart_str


ALGORITHM_META = {
    'BC': {
        'title': 'BC',
        'nickname': '行为克隆',
        'category': '模仿学习 / 基线方法',
        'summary': '直接学习“状态到动作”的映射，适合当作最基础的可解释基线。',
        'primary_metrics': [
            ('val_loss', '验证误差'),
            ('mse', '验证集 MSE'),
            ('mae', '验证集 MAE'),
        ],
        'main_charts': ['train_val_loss.png'],
        'more_charts': ['lr_curve.png'],
        'log_files': ['training_log.csv', 'evaluation_summary.csv'],
    },
    'BC_NB': {
        'title': 'BC_NB',
        'nickname': '加权行为克隆',
        'category': '模仿学习 / 改进方法',
        'summary': '在 BC 基础上，对高回报轨迹给予更高权重，更重视“好样本”。',
        'primary_metrics': [
            ('test_loss', '测试误差'),
            ('weighted_mse', '加权 MSE'),
            ('weighted_mae', '加权 MAE'),
        ],
        'main_charts': ['train_test_loss.png'],
        'more_charts': ['lr_curve.png'],
        'log_files': ['training_log.csv', 'evaluation_summary.csv'],
    },
    'CQL': {
        'title': 'CQL',
        'nickname': '保守 Q 学习',
        'category': '离线强化学习 / 基线方法',
        'summary': '通过更保守地学习 Q 值，减少对离线数据外动作的过高估计。',
        'primary_metrics': [
            ('val_critic_loss', '验证 critic 损失'),
            ('val_actor_loss', '验证 actor 损失'),
            ('val_cql_penalty', '验证保守项'),
        ],
        'main_charts': ['critic_loss_curve.png', 'cql_penalty_curve.png'],
        'more_charts': ['actor_loss_curve.png'],
        'log_files': ['training_log.csv'],
    },
    'CQL_NB': {
        'title': 'CQL_NB',
        'nickname': '增强型保守 Q 学习',
        'category': '离线强化学习 / 改进方法',
        'summary': '在 CQL 上加入更强的稳定化设计，便于观察保守项和训练波动。',
        'primary_metrics': [
            ('val_critic_loss', '验证 critic 损失'),
            ('val_actor_loss', '验证 actor 损失'),
            ('val_cql_penalty', '验证保守项'),
        ],
        'main_charts': ['critic_loss_curve.png', 'cql_penalty_curve.png'],
        'more_charts': ['actor_loss_curve.png'],
        'log_files': ['training_log.csv'],
    },
    'DDPG': {
        'title': 'DDPG',
        'nickname': '优势加权 DDPG',
        'category': '离线强化学习 / 基线方法',
        'summary': '通过 actor-critic 方式学习策略，并额外观察 advantage 与样本权重变化。',
        'primary_metrics': [
            ('val_critic_loss', '验证 critic 损失'),
            ('val_actor_loss', '验证 actor 损失'),
            ('val_weight', '验证平均权重'),
        ],
        'main_charts': ['critic_loss_curve.png', 'actor_loss_curve.png'],
        'more_charts': ['weight_curve.png'],
        'log_files': ['training_records.csv'],
    },
    'DDPG_NB': {
        'title': 'DDPG_NB',
        'nickname': 'DDPG + BC 正则',
        'category': '离线强化学习 / 改进方法',
        'summary': '在 DDPG 上加入行为克隆正则，更容易看出策略学习项与模仿项的平衡。',
        'primary_metrics': [
            ('val_critic_loss', '验证 critic 损失'),
            ('val_actor_loss', '验证 actor 损失'),
            ('val_bc_loss', '验证 BC 损失'),
        ],
        'main_charts': ['critic_loss_curve.png', 'actor_loss_curve.png'],
        'more_charts': ['bc_loss_curve.png'],
        'log_files': ['training_records.csv'],
    },
    'SAC': {
        'title': 'SAC',
        'nickname': 'AWAC-SAC',
        'category': '离线强化学习 / 基线方法',
        'summary': '兼顾收益与探索，但如果日志出现 NaN，要优先视为训练异常而不是“结果更好”。',
        'primary_metrics': [
            ('val_critic1_loss', '验证 critic1 损失'),
            ('val_critic2_loss', '验证 critic2 损失'),
            ('val_actor_loss', '验证 actor 损失'),
        ],
        'main_charts': ['critic1_loss.png', 'critic2_loss.png', 'actor_loss.png'],
        'more_charts': ['alpha.png', 'reward.png', 'Q1.png', 'Q2.png', 'advantage.png', 'weight.png', 'log_prob_data.png'],
        'log_files': ['training_history.csv'],
    },
    'SAC_NB': {
        'title': 'SAC_NB',
        'nickname': 'SAC + BC 正则',
        'category': '离线强化学习 / 改进方法',
        'summary': '在 SAC 上叠加行为克隆约束，更适合向初学者解释“强化学习 + 模仿项”共同作用。',
        'primary_metrics': [
            ('val_critic1_loss', '验证 critic1 损失'),
            ('val_critic2_loss', '验证 critic2 损失'),
            ('val_actor_loss', '验证 actor 损失'),
        ],
        'main_charts': ['critic1_loss.png', 'critic2_loss.png', 'actor_loss.png'],
        'more_charts': ['alpha.png', 'reward.png', 'Q1.png', 'Q2.png', 'bc_loss.png', 'log_prob.png'],
        'log_files': ['training_history.csv'],
    },
    'TD3': {
        'title': 'TD3',
        'nickname': '双 Q 延迟策略更新',
        'category': '离线强化学习 / 基线方法',
        'summary': '通过双 Q 和延迟更新减少训练抖动，结果页重点看 critic 曲线和 Q 值走势。',
        'primary_metrics': [
            ('val_critic_loss', '验证 critic 损失'),
            ('val_actor_proxy', '验证 actor 代理指标'),
            ('val_target_q', '验证 target Q'),
        ],
        'main_charts': ['critic_loss.png', 'q_value.png'],
        'more_charts': ['actor_loss.png'],
        'log_files': ['training_log.csv'],
    },
    'TD3_NB': {
        'title': 'TD3_NB',
        'nickname': 'TD3 + BC 正则',
        'category': '离线强化学习 / 改进方法',
        'summary': '在 TD3 上加入行为克隆项，结果页既能看主损失，也能看 BC 项的影响。',
        'primary_metrics': [
            ('val_critic_loss', '验证 critic 损失'),
            ('val_actor_proxy', '验证 actor 代理指标'),
            ('val_bc_term', '验证 BC 项'),
        ],
        'main_charts': ['critic_loss.png', 'q_value.png'],
        'more_charts': ['actor_loss.png'],
        'log_files': ['training_log.csv'],
    },
}


FRIENDLY_LABELS = {
    'epoch': '训练轮次',
    'train_loss': '训练误差',
    'val_loss': '验证误差',
    'test_loss': '测试误差',
    'mse': 'MSE',
    'mae': 'MAE',
    'weighted_mse': '加权 MSE',
    'weighted_mae': '加权 MAE',
    'max_abs_err': '最大绝对误差',
    'val_actor_loss': '验证 actor 损失',
    'val_critic_loss': '验证 critic 损失',
    'val_cql_penalty': '验证保守项',
    'val_critic1_loss': '验证 critic1 损失',
    'val_critic2_loss': '验证 critic2 损失',
    'val_weight': '验证平均权重',
    'val_bc_loss': '验证 BC 损失',
    'val_actor_proxy': '验证 actor 代理指标',
    'val_target_q': '验证 target Q',
    'val_bc_term': '验证 BC 项',
}


def _results_root():
    custom_root = getattr(settings, 'RESULTS_ROOT', None)
    if custom_root:
        return custom_root
    return os.path.join(settings.BASE_DIR, 'AIcodes/checkpoints')


def _algorithm_dir(algorithm):
    return os.path.abspath(os.path.join(_results_root(), algorithm))


def _safe_join_algorithm_file(algorithm, filename):
    algo_dir = _algorithm_dir(algorithm)
    target = os.path.abspath(os.path.join(algo_dir, filename))
    if not target.startswith(algo_dir + os.sep):
        raise Http404('非法文件路径')
    if not os.path.exists(target):
        raise Http404('文件不存在')
    return target


def _discover_algorithms():
    root = _results_root()
    if not os.path.isdir(root):
        return []
    names = []
    for name in sorted(os.listdir(root)):
        full = os.path.join(root, name)
        if os.path.isdir(full):
            names.append(name)
    return names


def _safe_read_csv(csv_path):
    try:
        return pd.read_csv(csv_path)
    except Exception:
        return None


def _last_record(df):
    if df is None or df.empty:
        return {}
    return df.tail(1).to_dict(orient='records')[0]


def _evaluation_val_record(df):
    if df is None or df.empty:
        return {}
    if 'split' in df.columns:
        val_df = df[df['split'].astype(str).str.lower() == 'val']
        if not val_df.empty:
            return val_df.tail(1).to_dict(orient='records')[0]
    return _last_record(df)


def _is_nan_value(value):
    try:
        return pd.isna(value)
    except Exception:
        return False


def _format_metric_value(value):
    if _is_nan_value(value):
        return 'NaN'
    if isinstance(value, (int, float)):
        abs_v = abs(float(value))
        if abs_v >= 1000:
            return '{:.2f}'.format(value)
        if abs_v >= 1:
            return '{:.4f}'.format(value)
        return '{:.6f}'.format(value)
    return smart_str(value)


def _collect_files(algorithm):
    algo_dir = _algorithm_dir(algorithm)
    if not os.path.isdir(algo_dir):
        return []
    items = []
    for name in sorted(os.listdir(algo_dir)):
        path = os.path.join(algo_dir, name)
        if os.path.isfile(path):
            items.append({
                'name': name,
                'size': os.path.getsize(path),
                'is_image': name.lower().endswith('.png'),
                'is_csv': name.lower().endswith('.csv'),
                'is_model': name.lower().endswith('.pt'),
            })
    return items


def _human_size(size_bytes):
    size = float(size_bytes)
    units = ['B', 'KB', 'MB', 'GB']
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return '{:.2f} {}'.format(size, unit)
        size /= 1024.0


def _build_chart_items(algorithm, names):
    items = []
    for name in names:
        path = os.path.join(_algorithm_dir(algorithm), name)
        if os.path.exists(path):
            items.append({
                'name': name,
                'title': os.path.splitext(name)[0].replace('_', ' ').title(),
                'url': 'result-file-placeholder:' + name,
            })
    return items


def _inject_chart_urls(algorithm, items):
    for item in items:
        item['url'] = 'file/{}/{}/'.format(algorithm, item['name'])
    return items


def _detect_nan(csv_data_list):
    for df in csv_data_list:
        if df is not None:
            numeric_df = df.select_dtypes(include=['number'])
            if not numeric_df.empty and numeric_df.isna().any().any():
                return True
    return False


def _build_summary_cards(algorithm, meta, log_df, eval_df):
    last_log = _last_record(log_df)
    eval_val = _evaluation_val_record(eval_df)

    cards = []
    for field, label in meta.get('primary_metrics', []):
        value = None
        if field in eval_val:
            value = eval_val.get(field)
        elif field in last_log:
            value = last_log.get(field)
        cards.append({
            'label': label,
            'value': _format_metric_value(value) if value is not None else '-',
        })
    return cards


def _friendly_status(files, has_nan, algorithm):
    names = [f['name'] for f in files]
    has_model = 'best_model.pt' in names
    has_csv = any(name.endswith('.csv') for name in names)

    if has_nan:
        return {
            'type': 'danger',
            'title': '结果异常',
            'text': '日志中检测到 NaN，建议先把这次训练视为异常结果，再决定是否纳入后续对比。',
        }
    if has_model and has_csv:
        return {
            'type': 'success',
            'title': '结果完整',
            'text': '{} 的模型文件和日志文件都已生成，可以先看摘要和主图。'.format(algorithm),
        }
    return {
        'type': 'warning',
        'title': '结果不完整',
        'text': '当前目录下部分关键文件缺失，页面会尽量展示已存在内容。',
    }


def _beginner_summary(algorithm, meta, files, has_nan, summary_cards):
    if has_nan:
        return '这次训练虽然产出了文件，但关键日志里已经出现异常数值，因此不建议直接把它当成可用结果。'

    if algorithm == 'BC':
        return '可以把它理解为“模仿数据里的动作”。结果页里最值得先看的，是验证误差和 train/val loss 曲线。'
    if algorithm == 'BC_NB':
        return '这是加权版 BC。除了普通误差，还能看加权误差，用来判断“高质量样本”是否被更好地学习到了。'
    if algorithm.startswith('CQL'):
        return '这一类结果先看 critic 损失，再看保守项。保守项不是越小越好，而是用来帮助判断训练是否过于激进。'
    if algorithm.startswith('DDPG'):
        return '这一类结果先看 critic 损失，再看 actor 相关指标。若同时有权重项或 BC 项，说明结果页会额外展示改进模块的影响。'
    if algorithm.startswith('SAC'):
        return '这一类结果建议先看 critic 和 actor 是否稳定，再看 alpha 或 BC 项等进阶指标。'
    if algorithm.startswith('TD3'):
        return '这一类结果建议先看 critic loss 和 Q value 图，先判断训练是否稳定，再看 actor 或 BC 项。'
    return meta.get('summary', '')


def _build_metric_table(log_df, eval_df):
    sections = []
    if log_df is not None and not log_df.empty:
        last = _last_record(log_df)
        rows = []
        for key, value in last.items():
            rows.append({
                'name': FRIENDLY_LABELS.get(key, key),
                'raw_name': key,
                'value': _format_metric_value(value),
            })
        sections.append({'title': '最新训练记录', 'rows': rows})

    if eval_df is not None and not eval_df.empty:
        rows = []
        for _, record in eval_df.iterrows():
            row = {'split': record.get('split', '-')}
            values = []
            for col in eval_df.columns:
                if col == 'split':
                    continue
                values.append({
                    'name': FRIENDLY_LABELS.get(col, col),
                    'raw_name': col,
                    'value': _format_metric_value(record.get(col)),
                })
            row['values'] = values
            rows.append(row)
        sections.append({'title': '评估汇总', 'summary_rows': rows})
    return sections


def results_overview(request):
    algorithms = _discover_algorithms()
    selected = request.GET.get('algorithm')
    if not selected or selected not in algorithms:
        selected = algorithms[0] if algorithms else None

    if not selected:
        return render(request, 'results/results.html', {
            'algorithms': [],
            'selected_algorithm': None,
            'page_empty': True,
        })

    meta = ALGORITHM_META.get(selected, {
        'title': selected,
        'nickname': selected,
        'category': '算法结果',
        'summary': '该算法已生成结果文件，页面将自动读取并展示。',
        'primary_metrics': [],
        'main_charts': [],
        'more_charts': [],
        'log_files': [],
    })

    files = _collect_files(selected)
    file_names = [item['name'] for item in files]

    log_df = None
    eval_df = None
    for name in meta.get('log_files', []):
        path = os.path.join(_algorithm_dir(selected), name)
        if os.path.exists(path):
            if name == 'evaluation_summary.csv':
                eval_df = _safe_read_csv(path)
            else:
                log_df = _safe_read_csv(path)
                break

    eval_path = os.path.join(_algorithm_dir(selected), 'evaluation_summary.csv')
    if eval_df is None and os.path.exists(eval_path):
        eval_df = _safe_read_csv(eval_path)

    has_nan = _detect_nan([log_df, eval_df])
    status = _friendly_status(files, has_nan, selected)
    summary_cards = _build_summary_cards(selected, meta, log_df, eval_df)
    beginner_summary = _beginner_summary(selected, meta, files, has_nan, summary_cards)

    main_charts = _inject_chart_urls(selected, _build_chart_items(selected, meta.get('main_charts', [])))
    more_charts = _inject_chart_urls(selected, _build_chart_items(selected, meta.get('more_charts', [])))

    file_items = []
    for item in files:
        file_items.append({
            'name': item['name'],
            'size': _human_size(item['size']),
            'is_image': item['is_image'],
            'is_csv': item['is_csv'],
            'is_model': item['is_model'],
            'url': 'file/{}/{}/'.format(selected, item['name']),
        })

    metric_sections = _build_metric_table(log_df, eval_df)

    context = {
        'algorithms': algorithms,
        'selected_algorithm': selected,
        'meta': meta,
        'status': status,
        'beginner_summary': beginner_summary,
        'summary_cards': summary_cards,
        'main_charts': main_charts,
        'more_charts': more_charts,
        'metric_sections': metric_sections,
        'file_items': file_items,
        'page_empty': False,
        'results_root': _results_root(),
    }
    return render(request, 'results/results.html', context)


def result_file(request, algorithm, filename):
    file_path = _safe_join_algorithm_file(algorithm, filename)
    content_type, _ = mimetypes.guess_type(file_path)
    content_type = content_type or 'application/octet-stream'
    as_attachment = request.GET.get('download') == '1'
    response = FileResponse(open(file_path, 'rb'), content_type=content_type)
    if as_attachment:
        response['Content-Disposition'] = 'attachment; filename="{}"'.format(os.path.basename(file_path))
    else:
        response['Content-Disposition'] = 'inline; filename="{}"'.format(os.path.basename(file_path))
    return response

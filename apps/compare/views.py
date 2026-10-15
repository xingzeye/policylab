
import os
from collections import OrderedDict

import pandas as pd

from django.conf import settings
from django.http import FileResponse, Http404
from django.shortcuts import render
from django.urls import reverse


PAIR_CONFIGS = OrderedDict([
    ('BC', {
        'title': 'BC / BC_NB',
        'base_dir': 'BC',
        'nb_dir': 'BC_NB',
        'family': '行为克隆',
        'beginner': '这一组最适合初学者理解：Base 版直接模仿数据中的动作，NB 版在此基础上让高回报轨迹样本更重要。',
        'difference': 'NB 版增加了 return-weighted 权重设计，因此除了普通误差外，还可以额外观察加权误差。',
        'core_charts': [
            ('loss', '损失曲线', ['train_val_loss.png', 'train_test_loss.png']),
            ('lr', '学习率曲线', ['lr_curve.png']),
        ],
        'primary_metrics': [
            {'key': 'val_loss', 'label': '最终验证损失', 'source': 'train', 'pref': 'lower'},
            {'key': 'val_mse', 'label': '验证集 MSE', 'source': 'eval', 'pref': 'lower'},
            {'key': 'val_mae', 'label': '验证集 MAE', 'source': 'eval', 'pref': 'lower'},
            {'key': 'val_weighted_mse', 'label': '验证集加权 MSE', 'source': 'eval', 'pref': 'lower', 'only': 'nb'},
        ],
    }),
    ('CQL', {
        'title': 'CQL / CQL_NB',
        'base_dir': 'CQL',
        'nb_dir': 'CQL_NB',
        'family': '保守 Q 学习',
        'beginner': '这一组关注“学得稳不稳”：它不只看动作，还要学习 Q 值，并通过保守项限制过高估计。',
        'difference': 'NB 版在保守项计算和目标动作处理上更激进/更完整，适合用来看改进版的训练变化。',
        'core_charts': [
            ('critic', 'Critic Loss', ['critic_loss_curve.png']),
            ('actor', 'Actor Loss', ['actor_loss_curve.png']),
            ('cql', 'CQL Penalty', ['cql_penalty_curve.png']),
        ],
        'primary_metrics': [
            {'key': 'val_actor_loss', 'label': '最终验证 Actor Loss', 'source': 'train', 'pref': 'neutral'},
            {'key': 'val_critic_loss', 'label': '最终验证 Critic Loss', 'source': 'train', 'pref': 'lower'},
            {'key': 'val_cql_penalty', 'label': '最终验证保守项', 'source': 'train', 'pref': 'neutral'},
            {'key': 'val_bc_loss', 'label': '最终验证 BC Loss', 'source': 'train', 'pref': 'lower'},
        ],
    }),
    ('DDPG', {
        'title': 'DDPG / DDPG_NB',
        'base_dir': 'DDPG',
        'nb_dir': 'DDPG_NB',
        'family': '确定性策略方法',
        'beginner': '这一组的核心是“Actor 产出动作、Critic 评价动作”。对小白来说，先看训练是否稳定，再看改进版引入了什么额外约束。',
        'difference': 'Base 与 NB 的训练目标不完全相同：Base 更强调 advantage-weighted BC，NB 更强调 BC 正则和 q_loss 的组合。',
        'core_charts': [
            ('critic', 'Critic Loss', ['critic_loss_curve.png']),
            ('actor', 'Actor Loss', ['actor_loss_curve.png']),
            ('extra', '附加项', ['weight_curve.png', 'bc_loss_curve.png']),
        ],
        'primary_metrics': [
            {'key': 'val_actor_loss', 'label': '最终验证 Actor Loss', 'source': 'train', 'pref': 'neutral'},
            {'key': 'val_critic_loss', 'label': '最终验证 Critic Loss', 'source': 'train', 'pref': 'lower'},
            {'key': 'val_weight', 'label': '最终验证权重', 'source': 'train', 'pref': 'neutral', 'only': 'base'},
            {'key': 'val_bc_loss', 'label': '最终验证 BC Loss', 'source': 'train', 'pref': 'lower', 'only': 'nb'},
        ],
    }),
    ('SAC', {
        'title': 'SAC / SAC_NB',
        'base_dir': 'SAC',
        'nb_dir': 'SAC_NB',
        'family': '最大熵 Actor-Critic',
        'beginner': '这一组除了学动作和 Q 值，还会维护一个熵系数 alpha，用来平衡“多探索一点”和“收益更高”之间的关系。',
        'difference': 'NB 版在 SAC 损失外加入了 BC 项，通常更容易解释为“在策略优化的同时别偏离数据太多”。',
        'core_charts': [
            ('actor', 'Actor Loss', ['actor_loss.png']),
            ('critic1', 'Critic1 Loss', ['critic1_loss.png']),
            ('critic2', 'Critic2 Loss', ['critic2_loss.png']),
            ('alpha', 'Alpha', ['alpha.png']),
        ],
        'primary_metrics': [
            {'key': 'val_actor_loss', 'label': '最终验证 Actor Loss', 'source': 'train', 'pref': 'neutral'},
            {'key': 'val_critic1_loss', 'label': '最终验证 Critic1 Loss', 'source': 'train', 'pref': 'lower'},
            {'key': 'val_critic2_loss', 'label': '最终验证 Critic2 Loss', 'source': 'train', 'pref': 'lower'},
            {'key': 'val_alpha', 'label': '最终验证 Alpha', 'source': 'train', 'pref': 'neutral'},
            {'key': 'val_bc_loss', 'label': '最终验证 BC Loss', 'source': 'train', 'pref': 'lower', 'only': 'nb'},
        ],
    }),
    ('TD3', {
        'title': 'TD3 / TD3_NB',
        'base_dir': 'TD3',
        'nb_dir': 'TD3_NB',
        'family': '双 Q 延迟更新',
        'beginner': '这一组重点看“训练稳不稳”和“Q 值有没有越来越离谱”。NB 版多了一项 BC 正则，更容易解释改进从哪里来。',
        'difference': 'NB 版在 TD3 基础上额外加入了 BC 项，因此不仅能看 actor/critic，还能单独看 q_term 和 bc_term。',
        'core_charts': [
            ('critic', 'Critic Loss', ['critic_loss.png']),
            ('actor', 'Actor Loss', ['actor_loss.png']),
            ('q', 'Q Value', ['q_value.png']),
        ],
        'primary_metrics': [
            {'key': 'val_critic_loss', 'label': '最终验证 Critic Loss', 'source': 'train', 'pref': 'lower'},
            {'key': 'val_actor_proxy', 'label': '最终验证 Actor Proxy', 'source': 'train', 'pref': 'neutral'},
            {'key': 'val_bc_term', 'label': '最终验证 BC Term', 'source': 'train', 'pref': 'lower', 'only': 'nb'},
        ],
    }),
])


def _checkpoints_dir():
    return os.path.join(str(settings.BASE_DIR), 'AIcodes/checkpoints')


def _safe_join_under(base_dir, relative_path):
    base_dir = os.path.abspath(base_dir)
    abs_path = os.path.abspath(os.path.join(base_dir, relative_path))
    if not abs_path.startswith(base_dir + os.sep) and abs_path != base_dir:
        raise Http404('非法路径')
    return abs_path


def compare_asset(request, relative_path):
    abs_path = _safe_join_under(_checkpoints_dir(), relative_path)
    if not os.path.exists(abs_path) or not os.path.isfile(abs_path):
        raise Http404('文件不存在')
    return FileResponse(open(abs_path, 'rb'))


def _fmt_value(value):
    if value is None:
        return '-'
    try:
        if pd.isna(value):
            return 'NaN'
    except Exception:
        pass
    try:
        value = float(value)
        if abs(value) >= 1000:
            return '{:.2f}'.format(value)
        if abs(value) >= 1:
            return '{:.4f}'.format(value)
        return '{:.6f}'.format(value)
    except Exception:
        return str(value)


def _file_size(path):
    if not os.path.exists(path):
        return '-'
    size = float(os.path.getsize(path))
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024 or unit == 'GB':
            return '{:.2f} {}'.format(size, unit)
        size /= 1024.0


def _read_csv_if_exists(dir_path, filename):
    path = os.path.join(dir_path, filename)
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


def _get_training_df(dir_path):
    for name in ['training_log.csv', 'training_records.csv', 'training_history.csv']:
        df = _read_csv_if_exists(dir_path, name)
        if df is not None:
            return name, df
    return None, None


def _get_eval_df(dir_path):
    return _read_csv_if_exists(dir_path, 'evaluation_summary.csv')


def _get_val_row(eval_df):
    if eval_df is None or 'split' not in eval_df.columns:
        return None
    val_df = eval_df[eval_df['split'].astype(str).str.lower() == 'val']
    if val_df.empty:
        return None
    return val_df.iloc[0].to_dict()


def _has_nan(df):
    if df is None:
        return False
    numeric_df = df.select_dtypes(include='number')
    if numeric_df.empty:
        return False
    return numeric_df.isna().any().any()


def _series_tail_dict(df):
    if df is None or df.empty:
        return {}
    return df.iloc[-1].to_dict()


def _metric_from_sources(metric_key, train_last, eval_val):
    if metric_key in train_last:
        return train_last.get(metric_key)
    if metric_key.startswith('val_'):
        plain_key = metric_key.replace('val_', '', 1)
        if plain_key in eval_val:
            return eval_val.get(plain_key)
    return eval_val.get(metric_key)


def _status_info(method_dir):
    best_model = os.path.join(method_dir, 'best_model.pt')
    csv_name, train_df = _get_training_df(method_dir)
    has_model = os.path.exists(best_model)
    has_train = train_df is not None

    if not os.path.exists(method_dir):
        return 'missing', '结果目录缺失', 'label-default'
    if _has_nan(train_df):
        return 'warning', '训练结果异常', 'label-warning'
    if has_model and has_train:
        return 'ok', '结果完整', 'label-success'
    if has_model or has_train:
        return 'partial', '结果不完整', 'label-info'
    return 'empty', '暂无结果', 'label-default'


def _method_summary(method_dir_name):
    root = _checkpoints_dir()
    dir_path = os.path.join(root, method_dir_name)
    status_key, status_text, status_class = _status_info(dir_path)
    train_csv_name, train_df = _get_training_df(dir_path)
    eval_df = _get_eval_df(dir_path)

    train_last = _series_tail_dict(train_df)
    eval_val = _get_val_row(eval_df) or {}

    image_files = []
    if os.path.isdir(dir_path):
        for fname in sorted(os.listdir(dir_path)):
            if fname.lower().endswith('.png'):
                rel = method_dir_name + '/' + fname
                image_files.append({
                    'name': fname,
                    'relative_path': rel,
                    'url': reverse('compare:asset', kwargs={'relative_path': rel}),
                    'size': _file_size(os.path.join(dir_path, fname)),
                })

    files = []
    if os.path.isdir(dir_path):
        for fname in sorted(os.listdir(dir_path)):
            path = os.path.join(dir_path, fname)
            if os.path.isfile(path):
                rel = method_dir_name + '/' + fname
                files.append({
                    'name': fname,
                    'relative_path': rel,
                    'url': reverse('compare:asset', kwargs={'relative_path': rel}),
                    'size': _file_size(path),
                })

    return {
        'name': method_dir_name,
        'dir_name': method_dir_name,
        'dir_path': dir_path,
        'status_key': status_key,
        'status_text': status_text,
        'status_class': status_class,
        'train_csv_name': train_csv_name,
        'train_df': train_df,
        'eval_df': eval_df,
        'train_last': train_last,
        'eval_val': eval_val,
        'files': files,
        'images': image_files,
        'has_nan': _has_nan(train_df),
        'has_model': os.path.exists(os.path.join(dir_path, 'best_model.pt')),
    }


def _build_metric_rows(pair_cfg, base_summary, nb_summary):
    rows = []
    for item in pair_cfg['primary_metrics']:
        only = item.get('only')
        if only == 'base':
            base_val = _metric_from_sources(item['key'], base_summary['train_last'], base_summary['eval_val'])
            nb_val = None
        elif only == 'nb':
            base_val = None
            nb_val = _metric_from_sources(item['key'], nb_summary['train_last'], nb_summary['eval_val'])
        else:
            base_val = _metric_from_sources(item['key'], base_summary['train_last'], base_summary['eval_val'])
            nb_val = _metric_from_sources(item['key'], nb_summary['train_last'], nb_summary['eval_val'])

        diff_text = '-'
        if base_val is not None and nb_val is not None:
            try:
                diff = float(nb_val) - float(base_val)
                diff_text = _fmt_value(diff)
            except Exception:
                diff_text = '-'

        winner = '—'
        if base_val is not None and nb_val is not None and not pd.isna(base_val) and not pd.isna(nb_val):
            pref = item.get('pref', 'neutral')
            try:
                bv = float(base_val)
                nv = float(nb_val)
                if pref == 'lower':
                    winner = 'NB 更优' if nv < bv else ('Base 更优' if bv < nv else '接近')
                elif pref == 'higher':
                    winner = 'NB 更优' if nv > bv else ('Base 更优' if bv > nv else '接近')
                else:
                    winner = '需结合业务解释'
            except Exception:
                winner = '—'

        rows.append({
            'label': item['label'],
            'base_value': _fmt_value(base_val),
            'nb_value': _fmt_value(nb_val),
            'difference': diff_text,
            'winner': winner,
        })
    return rows


def _build_chart_rows(pair_cfg, base_summary, nb_summary):
    rows = []
    for key, label, candidates in pair_cfg['core_charts']:
        base_chart = None
        nb_chart = None
        for fname in candidates:
            bpath = os.path.join(base_summary['dir_path'], fname)
            npath = os.path.join(nb_summary['dir_path'], fname)
            if base_chart is None and os.path.exists(bpath):
                base_chart = {
                    'name': fname,
                    'url': reverse('compare:asset', kwargs={'relative_path': base_summary['dir_name'] + '/' + fname}),
                }
            if nb_chart is None and os.path.exists(npath):
                nb_chart = {
                    'name': fname,
                    'url': reverse('compare:asset', kwargs={'relative_path': nb_summary['dir_name'] + '/' + fname}),
                }
        rows.append({
            'label': label,
            'base_chart': base_chart,
            'nb_chart': nb_chart,
        })
    return rows


def _compare_conclusion(pair_key, pair_cfg, base_summary, nb_summary, metric_rows):
    if base_summary['status_key'] == 'warning' and nb_summary['status_key'] != 'warning':
        return '当前更推荐优先查看 {}，因为 Base 版日志中检测到了异常值，而 NB 版结果更完整。'.format(nb_summary['name'])
    if nb_summary['status_key'] == 'warning' and base_summary['status_key'] != 'warning':
        return '当前更推荐优先查看 {}，因为 NB 版日志中检测到了异常值，而 Base 版结果更完整。'.format(base_summary['name'])

    if pair_key == 'BC':
        mse_row = next((r for r in metric_rows if r['label'] == '验证集 MSE'), None)
        if mse_row:
            if mse_row['winner'] == 'NB 更优':
                return '这一对结果整体非常接近，但从验证集 MSE 看，NB 版略优，同时还能额外提供加权误差分析。'
            if mse_row['winner'] == 'Base 更优':
                return '这一对结果整体非常接近，从验证集 MSE 看，Base 版略优；NB 版的价值更多体现在“能分析高质量轨迹权重”。'
        return '这一对结果整体差异不大，更适合作为“基线版 vs 加权改进版”的教学展示。'

    if pair_key == 'SAC':
        return '这一组建议优先看训练是否稳定，再看损失数值；如果出现 NaN，应先把它当异常处理，而不是直接比较大小。'

    return '这一组更适合看“训练是否稳定、改进项是否有效”，不要只盯着某一个数字的大小。'


def compare_view(request):
    pair_key = request.GET.get('pair', 'BC')
    if pair_key not in PAIR_CONFIGS:
        pair_key = 'BC'

    pair_cfg = PAIR_CONFIGS[pair_key]
    base_summary = _method_summary(pair_cfg['base_dir'])
    nb_summary = _method_summary(pair_cfg['nb_dir'])

    metric_rows = _build_metric_rows(pair_cfg, base_summary, nb_summary)
    chart_rows = _build_chart_rows(pair_cfg, base_summary, nb_summary)
    conclusion = _compare_conclusion(pair_key, pair_cfg, base_summary, nb_summary, metric_rows)

    pair_options = []
    for key, info in PAIR_CONFIGS.items():
        pair_options.append({
            'key': key,
            'title': info['title'],
            'selected': key == pair_key,
        })

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
    return render(request, 'compare/compare.html', context)

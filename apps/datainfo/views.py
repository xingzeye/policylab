import csv
import json
import os

from django.conf import settings
from django.shortcuts import render


MEDIA_SUBDIR = 'datainfo'
DATASET_FILENAME = 'halfcheetah_medium-v2.hdf5'
CSV_FILENAME = 'halfcheetah_dataset.csv'
OBS_IMG_FILENAME = 'observations_subplots.png'
ACT_IMG_FILENAME = 'actions_subplots.png'
REWARD_IMG_FILENAME = 'reward_terminal_subplots.png'


OBS_COLUMNS = ['obs_{:d}'.format(i) for i in range(17)]
ACT_COLUMNS = ['act_{:d}'.format(i) for i in range(6)]
CSV_COLUMNS = OBS_COLUMNS + ACT_COLUMNS + ['reward', 'terminal']


FIELD_ROWS = [
    {
        'field_name': 'observations',
        'csv_mapping': 'obs_0 ~ obs_16',
        'shape_desc': '[N, 17]',
        'meaning': '环境状态向量，共 17 维',
        'used_by': '全部算法',
    },
    {
        'field_name': 'actions',
        'csv_mapping': 'act_0 ~ act_5',
        'shape_desc': '[N, 6]',
        'meaning': '动作向量，共 6 维，范围通常在 [-1, 1]',
        'used_by': '全部算法',
    },
    {
        'field_name': 'rewards',
        'csv_mapping': 'reward',
        'shape_desc': '[N]',
        'meaning': '每一步对应奖励值',
        'used_by': 'CQL / SAC / TD3 / DDPG 系列',
    },
    {
        'field_name': 'next_observations',
        'csv_mapping': '未导出到当前 CSV',
        'shape_desc': '[N, 17]',
        'meaning': '下一时刻状态，属于 transition 信息',
        'used_by': 'CQL / SAC / TD3 / DDPG 系列',
    },
    {
        'field_name': 'terminals',
        'csv_mapping': 'terminal',
        'shape_desc': '[N]',
        'meaning': '轨迹终止标记，0/1',
        'used_by': 'CQL / SAC / TD3 / DDPG 系列',
    },
]


CSV_GROUPS = [
    {
        'group_name': 'Observation 列',
        'prefix': 'obs_',
        'count': 17,
        'range_text': 'obs_0 ~ obs_16',
        'description': '对应 HDF5 中的 observations',
    },
    {
        'group_name': 'Action 列',
        'prefix': 'act_',
        'count': 6,
        'range_text': 'act_0 ~ act_5',
        'description': '对应 HDF5 中的 actions',
    },
    {
        'group_name': 'Reward 列',
        'prefix': 'reward',
        'count': 1,
        'range_text': 'reward',
        'description': '对应 HDF5 中的 rewards',
    },
    {
        'group_name': 'Terminal 列',
        'prefix': 'terminal',
        'count': 1,
        'range_text': 'terminal',
        'description': '对应 HDF5 中的 terminals',
    },
]


ALGO_REQUIREMENTS = [
    {
        'algo': 'BC / BC_NB',
        'required_fields': 'observations, actions',
    },
    {
        'algo': 'CQL / CQL_NB',
        'required_fields': 'observations, actions, rewards, next_observations, terminals',
    },
    {
        'algo': 'SAC / SAC_NB',
        'required_fields': 'observations, actions, rewards, next_observations, terminals',
    },
    {
        'algo': 'TD3 / TD3_NB',
        'required_fields': 'observations, actions, rewards, next_observations, terminals',
    },
    {
        'algo': 'DDPG 家族',
        'required_fields': 'observations, actions, rewards, next_observations, terminals',
    },
]


def _human_size(size_bytes):
    if size_bytes is None:
        return '-'
    units = ['B', 'KB', 'MB', 'GB']
    size = float(size_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return '{:.2f} {}'.format(size, unit)
        size /= 1024.0


def _media_file_info(filename):
    abs_path = os.path.join(settings.MEDIA_ROOT, MEDIA_SUBDIR, filename)
    url = '{}{}/{}'.format(settings.MEDIA_URL, MEDIA_SUBDIR, filename)
    exists = os.path.exists(abs_path)
    return {
        'name': filename,
        'path': abs_path,
        'url': url,
        'exists': exists,
        'size': _human_size(os.path.getsize(abs_path)) if exists else '-',
    }


def _read_csv_preview(csv_path, preview_rows=8):
    columns = []
    rows = []
    row_count = None
    error = ''

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            columns = next(reader, [])
            for index, row in enumerate(reader):
                if index < preview_rows:
                    rows.append(row)
                row_count = index + 1
            if row_count is None:
                row_count = 0
    except Exception as exc:
        error = str(exc)

    return columns, rows, row_count, error


def halfcheetah_overview(request):
    csv_file = _media_file_info(CSV_FILENAME)
    obs_img = _media_file_info(OBS_IMG_FILENAME)
    act_img = _media_file_info(ACT_IMG_FILENAME)
    reward_img = _media_file_info(REWARD_IMG_FILENAME)

    preview_columns = []
    preview_rows = []
    csv_preview_error = ''
    csv_row_count = None

    is_preview = getattr(settings, 'PAGES_EXPORT', False) or not csv_file['exists']
    if csv_file['exists'] and not is_preview:
        preview_columns, preview_rows, csv_row_count, csv_preview_error = _read_csv_preview(csv_file['path'])
    elif is_preview:
        snapshot_path = os.path.join(settings.BASE_DIR, 'data', 'dataset_preview.json')
        if os.path.isfile(snapshot_path):
            with open(snapshot_path, encoding='utf-8') as snapshot_file:
                snapshot = json.load(snapshot_file)
            preview_columns = snapshot['columns']
            preview_rows = snapshot['rows']
            csv_row_count = snapshot['row_count']
            csv_file = _media_file_info('halfcheetah_preview.csv')

    missing_files = [
        item['name'] for item in [csv_file, obs_img, act_img, reward_img] if not item['exists']
    ]

    context = {
        'dataset': {
            'is_preview': is_preview,
            'title': 'HalfCheetah Medium v2',
            'dataset_name': DATASET_FILENAME,
            'env_name': 'halfcheetah-medium-v2',
            'task_type': '离线强化学习数据集',
            'obs_dim': 17,
            'action_dim': 6,
            'csv_column_count': len(CSV_COLUMNS),
            'csv_row_count': csv_row_count,
            'media_dir': os.path.join(settings.MEDIA_ROOT, MEDIA_SUBDIR),
            'description': (
                '该页面用于展示 halfcheetah_medium-v2.hdf5 的基础信息、字段结构、CSV 预览及数据分布图。'
                '页面默认读取 media/datainfo 下的分析产物。'
            ),
        },
        'files': {
            'csv': csv_file,
            'obs_img': obs_img,
            'act_img': act_img,
            'reward_img': reward_img,
        },
        'missing_files': missing_files,
        'preview_columns': preview_columns,
        'preview_rows': preview_rows,
        'csv_preview_error': csv_preview_error,
        'field_rows': FIELD_ROWS,
        'csv_groups': CSV_GROUPS,
        'algo_requirements': ALGO_REQUIREMENTS,
    }
    return render(request, 'datainfo/halfcheetah_overview.html', context)

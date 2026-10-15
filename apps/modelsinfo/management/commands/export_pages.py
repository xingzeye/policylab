"""Render the existing Django views as a relocatable GitHub Pages site."""
import json
import os
import re
import shutil
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.test import Client, override_settings


class LocalLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        self.links.extend(value for key, value in attrs
                          if key in ('href', 'src', 'data-src') and value)


def validate_links(destination):
    """Check local links in every page, including all generated selector variants."""
    errors = []
    for page in destination.rglob('*.html'):
        parser = LocalLinks()
        parser.feed(page.read_text(encoding='utf-8'))
        for link in parser.links:
            parts = urlsplit(link)
            if parts.scheme or parts.netloc or not parts.path:
                continue
            target = (page.parent / unquote(parts.path)).resolve()
            if destination not in target.parents and target != destination:
                errors.append('{}: link escapes export: {}'.format(page.name, link))
            elif not target.exists():
                errors.append('{}: missing {}'.format(page.relative_to(destination), link))
    if errors:
        raise CommandError('\n'.join(errors[:30]))


class Command(BaseCommand):
    help = 'Export the public learning site and all result variants to docs/ (no backend required).'

    def add_arguments(self, parser):
        parser.add_argument('--output', default='docs')

    def handle(self, *args, **options):
        from apps.modelsinfo.views import ALGORITHM_GROUPS
        from apps.results.views import ALGORITHM_META
        from apps.compare.views import PAIR_CONFIGS

        root = Path(settings.BASE_DIR).resolve()
        destination = Path(options['output']).resolve()
        if destination == root or destination in root.parents:
            raise CommandError('The output must be a dedicated directory, never the project root.')
        if destination.exists() and any(destination.iterdir()) and not (destination / '.policylab-export').exists():
            raise CommandError('Output contains unrelated files. Choose an empty directory.')
        destination.mkdir(parents=True, exist_ok=True)
        (destination / '.policylab-export').write_text('PolicyLab static export\n', encoding='utf-8')
        (destination / '.nojekyll').touch()

        # Explicit public asset list: never copy source datasets, credentials, or the database.
        for folder in ('vendor', 'policylab'):
            shutil.copytree(root / 'static' / folder, destination / 'static' / folder, dirs_exist_ok=True)
        media_out = destination / 'media' / 'datainfo'
        media_out.mkdir(parents=True, exist_ok=True)
        for name in ('observations_subplots.png', 'actions_subplots.png',
                     'reward_terminal_subplots.png', 'halfcheetah_preview.csv'):
            shutil.copy2(root / 'media' / 'datainfo' / name, media_out / name)
        for algorithm in ALGORITHM_META:
            source = root / 'AIcodes' / 'checkpoints' / algorithm
            for asset in sorted(source.iterdir()) if source.is_dir() else []:
                if asset.is_file() and asset.suffix.lower() in ('.png', '.csv', '.json', '.pt'):
                    target = destination / 'assets' / 'results' / algorithm / asset.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    # Config downloads should use repository-relative paths on the public site.
                    if asset.suffix == '.json':
                        data = json.loads(asset.read_text(encoding='utf-8'))
                        def portable(value):
                            if isinstance(value, dict):
                                return {key: portable(item) for key, item in value.items()}
                            if isinstance(value, list):
                                return [portable(item) for item in value]
                            if isinstance(value, str):
                                return value.replace(str(root), '.').replace(root.as_posix(), '.')
                            return value
                        target.write_text(json.dumps(portable(data), ensure_ascii=False, indent=2), encoding='utf-8')
                    else:
                        shutil.copy2(asset, target)

        routes = [('/', 'index.html'), ('/learn/', 'learn/index.html'),
                  ('/datainfo/', 'datainfo/index.html'), ('/modelsinfo/', 'modelsinfo/index.html'),
                  ('/results/', 'results/index.html'), ('/compare/', 'compare/index.html')]
        routes += [('/modelsinfo/?pair=' + key, 'modelsinfo/' + key + '.html') for key in ALGORITHM_GROUPS]
        routes += [('/results/?algorithm=' + key, 'results/' + key + '.html') for key in ALGORITHM_META]
        routes += [('/compare/?pair=' + key, 'compare/' + key + '.html') for key in PAIR_CONFIGS]

        def rewrite_url(url, output_path):
            parts = urlsplit(url)
            if parts.scheme or parts.netloc or not parts.path.startswith('/'):
                return url
            target = parts.path.lstrip('/')
            query = parse_qs(parts.query)
            if target.startswith('results/file/'):
                target = 'assets/results/' + target[len('results/file/'):].rstrip('/')
            elif target.startswith('compare/asset/'):
                target = 'assets/results/' + target[len('compare/asset/'):].rstrip('/')
            elif target == 'modelsinfo/' and query.get('pair'):
                target += query['pair'][0] + '.html'
            elif target.endswith('/') or not target:
                target += 'index.html'
            relative = os.path.relpath(destination / target, output_path.parent).replace(os.sep, '/')
            return relative + ('#' + parts.fragment if parts.fragment else '')

        with override_settings(PAGES_EXPORT=True, ALLOWED_HOSTS=['testserver']):
            client = Client()
            for route, filename in routes:
                response = client.get(route)
                if response.status_code != 200:
                    raise CommandError('{} returned {}'.format(route, response.status_code))
                output_path = destination / filename
                output_path.parent.mkdir(parents=True, exist_ok=True)
                content = response.content.decode('utf-8')
                content = re.sub(r'((?:href|src|data-src)=)([\'"])(/[^\'"]*)\2',
                                 lambda m: m[1] + m[2] + rewrite_url(m[3], output_path) + m[2], content)
                content = content.replace("'?pair=' + $(this).val()", "$(this).val() + '.html'")
                content = content.replace('this.form.submit()', "window.location.href = this.value + '.html'")
                # Relative source paths are meaningful to visitors and disclose no local usernames.
                content = content.replace(str(root), '.').replace(root.as_posix(), '.')
                content = '\n'.join(line.rstrip() for line in content.splitlines()) + '\n'
                output_path.write_text(content, encoding='utf-8')
        (destination / '404.html').write_text(
            '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>页面未找到 · 策学</title><h1>这一页暂时找不到</h1>'
            '<p>请使用浏览器返回，或回到策学网站首页继续学习。</p></html>', encoding='utf-8')
        validate_links(destination)
        self.stdout.write(self.style.SUCCESS('Exported {} public pages; all local links exist.'.format(len(routes))))

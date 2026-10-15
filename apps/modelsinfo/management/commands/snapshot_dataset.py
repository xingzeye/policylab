"""Create the small, explicit dataset preview used by GitHub Pages."""
import csv
import json
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = 'Save the original CSV row count and first 8 rows without publishing the full dataset.'

    def handle(self, *args, **options):
        root = Path(settings.BASE_DIR)
        source = root / 'media' / 'datainfo' / 'halfcheetah_dataset.csv'
        if not source.is_file():
            raise CommandError('Full dataset CSV is needed to refresh its snapshot.')
        rows = []
        row_count = 0
        with source.open(encoding='utf-8', newline='') as stream:
            reader = csv.reader(stream)
            columns = next(reader)
            for row_count, row in enumerate(reader, 1):
                if row_count <= 8:
                    rows.append(row)
        snapshot = {'source': source.name, 'row_count': row_count,
                    'source_size_bytes': source.stat().st_size, 'columns': columns, 'rows': rows}
        (root / 'data').mkdir(exist_ok=True)
        (root / 'data' / 'dataset_preview.json').write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        with (source.parent / 'halfcheetah_preview.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            writer.writerows(rows)
        self.stdout.write(self.style.SUCCESS('Saved {} preview rows; full CSV has {} rows.'.format(len(rows), row_count)))

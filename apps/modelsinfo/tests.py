from django.test import SimpleTestCase, override_settings
from django.urls import reverse

from .views import ALGORITHM_GROUPS


class ModelsInfoDesignExplanationTests(SimpleTestCase):
    def test_all_core_design_cards_have_plain_and_technical_text(self):
        for key, group in ALGORITHM_GROUPS.items():
            cards = group["sections"]["design"]["cards"]
            self.assertGreater(len(cards), 0, key)
            for card in cards:
                with self.subTest(pair=key, card=card["title"]):
                    self.assertIn("plain", card)
                    self.assertIn("technical", card)
                    self.assertTrue(card["plain"].strip())
                    self.assertTrue(card["technical"].strip())

    def test_core_design_page_renders_layered_explanation_labels(self):
        for key in ALGORITHM_GROUPS:
            with self.subTest(pair=key):
                response = self.client.get(reverse("modelsinfo:modelsinfo"), {"pair": key})
                self.assertEqual(response.status_code, 200)
                self.assertContains(response, "通俗理解")
                self.assertContains(response, "技术对应")

    def test_all_conclusions_have_detailed_points(self):
        for key, group in ALGORITHM_GROUPS.items():
            conclusion = group["sections"]["conclusion"]
            points = conclusion.get("points", [])
            with self.subTest(pair=key):
                self.assertGreaterEqual(len(points), 3)
                for point in points:
                    self.assertTrue(point.strip())

    def test_conclusion_page_renders_point_list(self):
        for key in ALGORITHM_GROUPS:
            with self.subTest(pair=key):
                response = self.client.get(reverse("modelsinfo:modelsinfo"), {"pair": key})
                self.assertEqual(response.status_code, 200)
                self.assertContains(response, 'class="conclusion-points"')


class LearningAndPagesTests(SimpleTestCase):
    def test_learning_is_accessible_from_home_and_learn(self):
        for route in ('home', 'learn'):
            response = self.client.get(reverse(route))
            self.assertEqual(response.status_code, 200)
            self.assertContains(response, '策学')
            for key in ('bc', 'ddpg', 'td3', 'sac', 'cql'):
                self.assertContains(response, 'data-lesson="{}"'.format(key))
            self.assertContains(response, '不是原始 DDPG')
            self.assertContains(response, 'AWAC')

    @override_settings(PAGES_EXPORT=True)
    def test_dataset_preview_remains_available_without_full_csv(self):
        response = self.client.get(reverse('datainfo:index'))
        self.assertContains(response, '下载 8 条样本 CSV')
        self.assertNotContains(response, 'href="/media/datainfo/halfcheetah_dataset.csv"')

    def test_static_export_supports_all_variants_and_has_no_private_files(self):
        import io
        import tempfile
        from pathlib import Path
        from django.core.management import call_command
        from .management.commands.export_pages import validate_links
        with tempfile.TemporaryDirectory(prefix='policylab-test-') as temporary:
            destination = Path(temporary).resolve()
            call_command('export_pages', output=str(destination), stdout=io.StringIO())
            validate_links(destination)
            self.assertEqual(len(list(destination.rglob('*.html'))), 27)
            page = (destination / 'modelsinfo' / 'sac.html').read_text(encoding='utf-8')
            self.assertIn("$(this).val() + '.html'", page)
            self.assertIn('../learn/index.html', page)
            self.assertNotIn('D:\\benke', page)
            self.assertNotIn('this.form.submit()', (destination / 'compare' / 'BC.html').read_text(encoding='utf-8'))
            for forbidden in ('db.sqlite3', '账号密码.txt', 'halfcheetah_dataset.csv'):
                self.assertFalse(list(destination.rglob(forbidden)))

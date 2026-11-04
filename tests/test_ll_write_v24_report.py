import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('report', Path(__file__).parents[1] / 'harness/_ll_write_v24_report.py')
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def slot(target='knee', classification='Good', ccc=.7):
    return dict(target=target, view='side', classification=classification, ccc_lin=ccc)


class ReportTests(unittest.TestCase):
    def render(self, baseline, current, previous=None):
        data = {report.V17_BASELINE: {'slots': baseline},
                report.V18_PS: {'slots': []}, report.V20_PS: {'slots': []},
                report.V23_PS: {'slots': previous or []}, report.V24_PS: {'slots': current}}
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'REPORT.md'
            with patch.object(report, '_load', side_effect=data.__getitem__), patch.object(report, 'OUT_PATH', out):
                report.main()
            return out.read_text()

    def test_missing_baseline_never_promotes(self):
        text = self.render([], [slot()])
        self.assertIn('Promotions vs v17: 0', text)
        self.assertIn('Not assessable vs v17: 1', text)
        self.assertIn('| Not assessable |', text)

    def test_invalid_classifications_are_not_ranked(self):
        for classification in (None, '', 'Unknown', 1, [], {}):
            with self.subTest(classification=classification):
                text = self.render([slot(classification=classification)], [slot()])
                self.assertIn('Promotions vs v17: 0', text)
                self.assertIn('Not assessable vs v17: 1', text)
                text = self.render([slot()], [slot(classification=classification)])
                self.assertIn('Demotions vs v17: 0', text)
                self.assertIn('1 Not assessable', text)
        missing = slot()
        del missing['classification']
        self.assertEqual(report._tier(missing), 'Not assessable')

    def test_valid_tiers_and_loa_mapping_are_preserved(self):
        self.assertEqual(report._tier(slot(classification='LoA-only Moderate')), 'Poor')
        text = self.render([slot('up', 'Poor'), slot('down', 'Good'), slot('same')],
                           [slot('up'), slot('down', 'Poor'), slot('same')])
        self.assertIn('Promotions vs v17: 1 | Demotions vs v17: 1 | Unchanged vs v17: 1', text)

    def test_unassessable_ccc_never_negative_conclusion(self):
        for value in (None, float('nan'), float('inf'), -float('inf'), '0.7', True):
            for invalid_current in (False, True):
                with self.subTest(value=value, invalid_current=invalid_current):
                    text = self.render([], [slot(ccc=value if invalid_current else .7)],
                                       [slot(ccc=.7 if invalid_current else value)])
                    self.assertIn('0/1 v24 slots assessed; 1 not assessable', text)
                    self.assertEqual(text.count('Comparison not assessable: no paired finite CCC values.'), 2)
                    self.assertNotIn('no slot where', text.lower())

    def test_partial_coverage_qualifies_negative(self):
        text = self.render([], [slot('tie'), slot('missing')], [slot('tie')])
        self.assertIn('1/2 v24 slots assessed; 1 not assessable', text)
        self.assertIn('Among 1 assessed pairs, no slot where v24 beats v23', text)
        self.assertIn('Among 1 assessed pairs, no slot where v23 beats v24', text)

    def test_valid_comparisons_both_directions(self):
        text = self.render([], [slot('up', ccc=.8), slot('down', ccc=.6)],
                           [slot('up', ccc=.7), slot('down', ccc=.7)])
        self.assertIn('2/2 v24 slots assessed; 0 not assessable', text)
        self.assertIn('| up / side | 0.70 | 0.80 | 0.10 |', text)
        self.assertIn('| down / side | 0.70 | 0.60 | -0.10 |', text)

    def test_comparison_threshold_is_strict(self):
        text = self.render([], [slot('edge', ccc=.505), slot('within', ccc=.503)],
                           [slot('edge', ccc=.5), slot('within', ccc=.5)])
        self.assertIn('Among 2 assessed pairs, no slot where v24 beats v23', text)
        self.assertIn('Among 2 assessed pairs, no slot where v23 beats v24', text)

    def test_empty_input(self):
        text = self.render([], [])
        self.assertIn('0/0 v24 slots assessed', text)
        self.assertEqual(text.count('Comparison not assessable:'), 2)

    def test_head_to_head_ignores_invalid_numbers(self):
        text = self.render([], [slot(ccc=float('inf'))], [slot(ccc=.6)])
        self.assertIn('| knee / side | - | - | - | 0.60 | **-** | v23 |', text)


if __name__ == '__main__':
    unittest.main()

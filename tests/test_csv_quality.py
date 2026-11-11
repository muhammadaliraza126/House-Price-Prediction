import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from tools.csv_quality import inspect_csv

ROOT = Path(__file__).resolve().parents[1]

class QualityTests(unittest.TestCase):
    def check_text(self, text, numeric=()):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'input.csv'
            source.write_text(text, encoding='utf-8')
            return inspect_csv(source, numeric)

    def test_clean_and_bom(self):
        report = self.check_text('\ufeffname,amount\nAli,12\nSara,15\n', ['amount'])
        self.assertEqual(report['status'], 'passed')
        self.assertEqual(report['rows'], 2)

    def test_counts(self):
        report = self.check_text('name,amount\nAli,12\nAli,12\n ,bad\nSara,NaN\nshort\n', ['amount'])
        self.assertEqual(report['duplicate_rows'], 1)
        self.assertEqual(report['malformed_rows'], 1)
        self.assertEqual(report['blank_cells_by_column']['name'], 1)
        self.assertEqual(report['invalid_numeric_cells_by_column']['amount'], 2)

    def test_header_errors(self):
        for text in ['', 'a,a\n1,2\n', ',b\n1,2\n']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.check_text(text)
        with self.assertRaises(ValueError):
            self.check_text('a\n1\n', ['missing'])

    def test_header_only(self):
        self.assertTrue(self.check_text('a,b\n')['empty_dataset'])

    def test_quoted_comma_and_multiline(self):
        report = self.check_text('name,amount\n"Ali, Raza",12\n"two\nlines",13\n', ['amount'])
        self.assertEqual(report['rows'], 2)
        self.assertEqual(report['status'], 'passed')

    def test_cli_exit_codes_and_input_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'input.csv'
            for content, expected in [('a\n1\n', 0), ('a\n1\n1\n', 1)]:
                source.write_text(content)
                result = subprocess.run([sys.executable, str(ROOT/'tools/csv_quality.py'), str(source)], capture_output=True, text=True)
                self.assertEqual(result.returncode, expected, result.stderr)
                self.assertIn('status', json.loads(result.stdout))
            before = source.read_bytes()
            result = subprocess.run([sys.executable, str(ROOT/'tools/csv_quality.py'), str(source), '--output', str(source)], capture_output=True)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(source.read_bytes(), before)

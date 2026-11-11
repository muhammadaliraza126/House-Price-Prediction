import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

class CommandLineTests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, str(ROOT / 'houseprice.py'), *map(str, args)],
                              cwd=ROOT, capture_output=True, text=True)

    def test_csv_report_matches_stdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'nested' / 'report.json'
            result = self.run_cli('--csv', ROOT / 'examples/houses_synthetic.csv', '--output', output)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(output.read_text())
            self.assertEqual(report, json.loads(result.stdout))
            self.assertEqual((report['train_rows'], report['test_rows']), (80, 20))
            self.assertEqual(report['data_source'], 'user_csv')

    def test_missing_file_exits_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_cli('--csv', Path(tmp) / 'missing.csv')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('error:', result.stderr)
            self.assertNotIn('Traceback', result.stderr)

    def test_invalid_csv_does_not_write_report(self):
        for content, expected in [('wrong\n1\n', 'Missing columns'),
                                  ('size_sqft,bedrooms,age_years,price_usd\nlarge,2,5,200000\n', 'Unable to parse')]:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as tmp:
                source = Path(tmp) / 'bad.csv'
                output = Path(tmp) / 'report.json'
                source.write_text(content)
                result = self.run_cli('--csv', source, '--output', output)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected, result.stderr)
                self.assertFalse(output.exists())

if __name__ == '__main__':
    unittest.main()

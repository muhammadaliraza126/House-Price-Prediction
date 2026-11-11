"""Check CSV quality locally without modifying the input or requiring dependencies."""
import argparse
import csv
import json
import math
from pathlib import Path


def inspect_csv(path, numeric=()):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.reader(handle, strict=True)
        header = next(reader, None)
        if not header or any(not name.strip() for name in header):
            raise ValueError('A nonempty header with named columns is required.')
        if len(set(header)) != len(header):
            raise ValueError('Duplicate column names are not supported.')
        unknown = set(numeric) - set(header)
        if unknown:
            raise ValueError('Unknown numeric columns: ' + ', '.join(sorted(unknown)))
        blanks = dict.fromkeys(header, 0)
        invalid = dict.fromkeys(numeric, 0)
        seen = set()
        duplicates = malformed = count = 0
        for row in reader:
            count += 1
            if len(row) != len(header):
                malformed += 1
                continue
            key = tuple(row)
            if key in seen:
                duplicates += 1
            seen.add(key)
            for name, value in zip(header, row):
                if not value.strip():
                    blanks[name] += 1
                elif name in invalid:
                    try:
                        valid = math.isfinite(float(value))
                    except ValueError:
                        valid = False
                    if not valid:
                        invalid[name] += 1
        issues = not count or malformed or duplicates or any(blanks.values()) or any(invalid.values())
        return {'status': 'issues_found' if issues else 'passed',
                'rows': count, 'columns': header, 'malformed_rows': malformed,
                'duplicate_rows': duplicates, 'blank_cells_by_column': blanks,
                'invalid_numeric_cells_by_column': invalid,
                'empty_dataset': count == 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--numeric', nargs='+', default=[], help='Columns expected to contain finite numbers')
    parser.add_argument('--output', type=Path, help='Write JSON to a different file')
    args = parser.parse_args()
    try:
        if args.output and args.output.resolve() == args.csv.resolve():
            raise ValueError('Output must not overwrite the input CSV.')
        report = inspect_csv(args.csv, args.numeric)
        result = json.dumps(report, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open('x', encoding='utf-8') as output:
                output.write(result + '\n')
        print(result)
        return 1 if report['status'] == 'issues_found' else 0
    except (ValueError, OSError, UnicodeError, csv.Error) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    raise SystemExit(main())

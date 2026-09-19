"""Summarize sampled GPU activity without attributing device-wide peaks to tensors."""

import argparse
import csv
import json
import re
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('evidence', type=Path)
args = parser.parse_args()
audit = {}
for arm in ('N0', 'A0', 'P0', 'P1', 'A1', 'N1'):
    directory = args.evidence / arm
    with (directory / 'processes-start.csv').open() as source:
        residents = {int(row['pid']): row['process_name'] for row in csv.DictReader(source, skipinitialspace=True)}
    processes = {}
    for line in (directory / 'process-utilization.txt').read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        fields = line.split()
        if fields[1] == '-':
            continue
        pid = int(fields[1])
        record = processes.setdefault(pid, {'command': fields[-1], 'type': fields[2], 'samples': 0, 'numeric_sm_samples': 0, 'max_sm': 0, 'max_fb_mib': 0})
        record['samples'] += 1
        if fields[3] != '-':
            record['numeric_sm_samples'] += 1
            record['max_sm'] = max(record['max_sm'], int(fields[3]))
        if fields[9] != '-':
            record['max_fb_mib'] = max(record['max_fb_mib'], int(fields[9]))
    workers = {int(pid) for pid in re.findall(r'pid=(\d+)', (directory / 'run.log').read_text())}
    extra_active = [pid for pid, row in processes.items() if row['type'] == 'C' and row['max_sm'] > 0 and pid not in workers]
    audit[arm] = {'resident_processes': residents, 'worker_pids': sorted(workers), 'additional_active_compute_pids': extra_active, 'samples_by_pid': processes}
(args.evidence / 'process-audit.json').write_text(json.dumps(audit, indent=2) + '\n')
print(json.dumps(audit, indent=2))

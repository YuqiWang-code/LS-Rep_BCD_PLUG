"""
update_metrics.py — 每次训练结束后，把结果/指标写回 experiment_metrics.xlsx。

用法：
    python analyse/update_metrics.py --log <train_log.txt> --experiment "PLUG Run1 R1"

从 train_log.txt 里解析：
  - Experiment Configuration 块：dataset / model / train_params / batch_size / lr / max_steps
  - "Test Results (Best Model)" 块：Recall / Precision / F1 / IoU / OA / Kappa

然后把一行写入 docs/experiment_metrics.xlsx 的 "All Data" 表（按 Experiment+Dataset 去重，
已存在的行会更新，不重复追加）。
"""

import os
import re
import argparse

import openpyxl

XLSX = os.path.join(os.path.dirname(__file__), '..', 'docs', 'experiment_metrics.xlsx')

# "All Data" 表列顺序
ALL_DATA_COLS = [
    'Source', 'Model', 'Experiment', 'Dataset', 'Metric Split', 'Metric Log',
    'Backbone', 'Venue', 'Train Params (M)', 'Infer Params (M)',
    'Train FLOPs (G)', 'Infer FLOPs (G)', 'Trainable (M)', 'LR', 'Batch',
    'Epochs', 'Input Size', 'Best F1', 'Best Epoch', 'OA', 'IoU', 'F1',
    'Recall', 'Precision', 'Kappa',
]


def parse_log(log_file):
    """Parse config + test metrics out of a train_log.txt."""
    text = open(log_file, encoding='utf-8').read()

    cfg = {}
    for key in ('dataset', 'model', 'train_params', 'batch_size', 'lr', 'max_steps'):
        m = re.search(rf'^{key}:\s*(.+)$', text, re.MULTILINE)
        if m:
            cfg[key] = m.group(1).strip()

    metrics = {}
    for key in ('Recall', 'Precision', 'F1', 'IoU', 'OA', 'Kappa'):
        m = re.search(rf'^{key}:\s*([0-9.eE+-]+)', text, re.MULTILINE)
        if m:
            metrics[key] = float(m.group(1))

    return cfg, metrics


def parse_train_params(s):
    """'2.91M' -> 2.91 (float, in M)."""
    if not s:
        return None
    s = s.strip()
    m = re.match(r'([0-9.]+)\s*M', s, re.IGNORECASE)
    return float(m.group(1)) if m else None


def _col_idx(sheet, name):
    """Return 0-based column index for a header name; raise if missing."""
    for i, cell in enumerate(next(sheet.iter_rows(min_row=1, max_row=1))):
        if cell.value == name:
            return i
    raise KeyError(f'column {name!r} not found')


def upsert(wb, entry):
    ws = wb['All Data']
    header = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1))]
    idx = {name: i for i, name in enumerate(header)}

    exp = entry['Experiment']
    ds = entry['Dataset']

    # find existing row to overwrite
    target = None
    for r in range(2, ws.max_row + 1):
        if (ws.cell(r, idx['Experiment'] + 1).value == exp and
                ws.cell(r, idx['Dataset'] + 1).value == ds):
            target = r
            break

    row_vals = [None] * len(header)
    for key, val in entry.items():
        if key in idx:
            row_vals[idx[key]] = val

    if target is None:
        ws.append(row_vals)
        print(f'[added]   {exp} / {ds}')
    else:
        for i, val in enumerate(row_vals):
            if val is not None:
                ws.cell(target, i + 1).value = val
        print(f'[updated] {exp} / {ds}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', required=True, help='path to train_log.txt')
    ap.add_argument('--experiment', required=True, help='experiment label, e.g. "PLUG Run1 R1"')
    ap.add_argument('--xlsx', default=XLSX, help='path to experiment_metrics.xlsx')
    ap.add_argument('--source', default='Ours')
    args = ap.parse_args()

    cfg, metrics = parse_log(args.log)
    if not metrics:
        raise SystemExit('no "Test Results (Best Model)" block found in log')

    dataset = cfg.get('dataset', '')
    train_params = parse_train_params(cfg.get('train_params'))
    lr = float(cfg['lr']) if cfg.get('lr') else None
    batch = int(cfg['batch_size']) if cfg.get('batch_size') else None
    max_steps = int(cfg['max_steps']) if cfg.get('max_steps') else None

    entry = {
        'Source': args.source,
        'Model': cfg.get('model', 'A2Net_LWGANet_L0'),
        'Experiment': args.experiment,
        'Dataset': dataset + '-CD-256' if dataset and not dataset.endswith('-CD-256') else dataset,
        'Metric Split': 'test',
        'Metric Log': os.path.normpath(args.log).replace('\\', '/'),
        'Backbone': 'LWGANet-L0',
        'Venue': None,
        'Train Params (M)': train_params,
        'LR': lr,
        'Batch': batch,
        'Epochs': None,
        'Input Size': 256,
        'OA': metrics.get('OA'),
        'IoU': metrics.get('IoU'),
        'F1': metrics.get('F1'),
        'Recall': metrics.get('Recall'),
        'Precision': metrics.get('Precision'),
        'Kappa': metrics.get('Kappa'),
    }

    wb = openpyxl.load_workbook(args.xlsx)
    upsert(wb, entry)
    wb.save(args.xlsx)
    print(f'saved -> {args.xlsx}')


if __name__ == '__main__':
    main()

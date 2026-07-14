"""Run the DETERMINISTIC rule engine over ALL AllianceChicago candidate pairs.

This is the rules analogue of `predict_alliance_full.py`: same parquet inputs
(per-record MDM population + candidate-pairs blocking output), but instead of the
GPT-2 model it applies the `rules_alliance.classify_pair` cascade. No GPU, no
checkpoint -- pure deterministic Python, so the full population scores in seconds
to a couple of minutes rather than hours.

Output columns: PATID_A, PATID_B, rule_pred, rule_id, rule_reason.
(Join back to the records parquet on PATID for the underlying fields.)

Run from the AnyMatch repo root:
    python deterministic_rules/predict_alliance_rules_full.py `
        --records_parquet data/alliance/MDM_Population_cleaned_v4_2026_06_16.parquet `
        --pairs_parquet   data/alliance/candidate_pairs_v5_2026_06_16.parquet `
        --output_csv      data/alliance/deterministic_rules_predictions_full.csv `
        --chunk_size 20000
"""
from __future__ import annotations

import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')  # harmless here; parity w/ model path

import argparse
import csv
import sys
import time
from datetime import datetime, timedelta

import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # so `import rules_alliance` works

from utils.alliance_schema import id_str_dtypes, prep_record_df, FEATURE_COLS_SRC, FEATURE_COLS  # noqa: E402
import rules_alliance as R  # noqa: E402

OUTPUT_COLS = ['PATID_A', 'PATID_B', 'rule_pred', 'rule_id', 'rule_reason']


def parse_args():
    p = argparse.ArgumentParser(description='Deterministic rule scoring over all candidate pairs (resumable).')
    p.add_argument('--records_parquet', required=True, help='Cleaned MDM population parquet (per-record).')
    p.add_argument('--pairs_parquet', required=True, help='Blocking output parquet with PATID_A / PATID_B.')
    p.add_argument('--output_csv', required=True, help='Incremental predictions CSV (appended to; resumable).')
    p.add_argument('--chunk_size', type=int, default=20000, help='Pairs scored + flushed to disk per chunk.')
    return p.parse_args()


def log(msg):
    print(f'{datetime.now():%Y-%m-%d %H:%M:%S} [INFO] {msg}', flush=True)


def main():
    args = parse_args()
    assert os.path.exists(os.path.join(_REPO_ROOT, 'loo.py')), \
        f'Run from the AnyMatch/ directory (repo root not found at {_REPO_ROOT!r}).'

    log('=== Deterministic rules full scoring ===')

    # ---- 1. Records: load, force ID dtypes, filter valid, rename->friendly, index ----
    log(f'Loading records parquet: {args.records_parquet}')
    records = pd.read_parquet(args.records_parquet)
    for col, dt in id_str_dtypes(records.columns).items():
        records[col] = records[col].astype(dt)
    if 'valid_record' not in records.columns:
        raise KeyError("records parquet has no 'valid_record' column.")
    records_valid = records[records['valid_record']].copy()
    log(f'{len(records):,} records -> {len(records_valid):,} with valid_record=True')

    missing = [c for c in FEATURE_COLS_SRC if c not in records_valid.columns]
    if missing:
        raise KeyError(f'FEATURE_COLS_SRC not found in records: {missing}')

    records_valid = prep_record_df(records_valid).set_index('PATID')
    if not records_valid.index.is_unique:
        dup = records_valid.index[records_valid.index.duplicated()].unique()[:5]
        raise ValueError(f'PATID not unique in records. Examples: {list(dup)}')

    # ---- 2. Pairs: keep both-valid, deterministic order (for resume) ----
    log(f'Loading pairs parquet: {args.pairs_parquet}')
    pairs = pd.read_parquet(args.pairs_parquet)[['PATID_A', 'PATID_B']]
    n_raw = len(pairs)
    valid_ids = set(records_valid.index)
    pairs = pairs[pairs['PATID_A'].isin(valid_ids) & pairs['PATID_B'].isin(valid_ids)]
    pairs = pairs.sort_values(['PATID_A', 'PATID_B'], kind='stable').reset_index(drop=True)
    total = len(pairs)
    log(f'{n_raw:,} candidate pairs -> {total:,} with both sides valid (will be scored)')
    if total == 0:
        log('No pairs to score. Exiting.'); return

    # ---- 3. Resume: how many already written? ----
    done = 0
    file_exists = os.path.exists(args.output_csv) and os.path.getsize(args.output_csv) > 0
    if file_exists:
        with open(args.output_csv, 'r', newline='') as f:
            done = max(sum(1 for _ in f) - 1, 0)
        if done >= total:
            log(f'Output already has {done:,} rows >= {total:,} pairs. Nothing to do.')
            return
        log(f'Resuming: {done:,} already in {args.output_csv}; {total - done:,} remaining.')
    else:
        log(f'Fresh run -> {args.output_csv}')

    # ---- 4. Chunk loop: build _l/_r wide table, run cascade, append ----
    out_f = open(args.output_csv, 'a', newline='')
    writer = csv.writer(out_f)
    if not file_exists:
        writer.writerow(OUTPUT_COLS); out_f.flush(); os.fsync(out_f.fileno())

    session_start = time.perf_counter()
    session_done = 0
    try:
        for start in range(done, total, args.chunk_size):
            end = min(start + args.chunk_size, total)
            chunk = pairs.iloc[start:end].reset_index(drop=True)

            wide = chunk.copy()
            for side, patid_col in [('l', 'PATID_A'), ('r', 'PATID_B')]:
                side_df = (records_valid.loc[chunk[patid_col].values, FEATURE_COLS]
                           .reset_index(drop=True).add_suffix(f'_{side}'))
                wide = pd.concat([wide, side_df], axis=1)

            scored = R.classify_df(wide)
            for pa, pb, rp, rid, rr in zip(
                chunk['PATID_A'], chunk['PATID_B'],
                scored['rule_pred'], scored['rule_id'], scored['rule_reason'],
            ):
                writer.writerow([pa, pb, rp, rid, rr])
            out_f.flush(); os.fsync(out_f.fileno())

            session_done += (end - start)
            total_done = end
            elapsed = time.perf_counter() - session_start
            rate = session_done / elapsed if elapsed > 0 else 0.0
            remaining = total - total_done
            eta = remaining / rate if rate > 0 else None
            finish = (datetime.now() + timedelta(seconds=eta)).strftime('%H:%M:%S') if eta else 'unknown'
            log(f'{total_done:,}/{total:,} ({total_done/total*100:.2f}%) | '
                f'remaining={remaining:,} | rate={rate:.0f} pairs/s | ETA={finish}')
    finally:
        out_f.flush(); os.fsync(out_f.fileno()); out_f.close()

    log(f'DONE. {total:,} pairs scored -> {args.output_csv}')


if __name__ == '__main__':
    main()

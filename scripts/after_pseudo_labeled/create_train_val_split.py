"""
create_train_val_split.py

Assigns a train/val split column to data/after_pseudo_labeled/species_segment_index.csv.

Grouping unit: individual segment (dest_path row), stratified by
original_primary_label within each assigned_species group.

Rules per assigned_species group
---------------------------------
  N == 0  : skip
  N == 1  : train (species may not appear in val)
  N == 2  : 1 val, 1 train (shuffled)
  N >= 3  :
    n_val = max(1, floor(N * 0.10))

    Case A — no original_primary_label contains ";":
        Shuffle (fixed seed), first n_val -> val, rest -> train.

    Case B — any original_primary_label contains ";":
        Sub-group by unique original_primary_label, rarest first.
        Each sub-group claims 1 val slot from the budget (if budget > 0),
        rest of sub-group -> train.
        If budget remains after all sub-groups, pull more from train pool
        at random until budget is exhausted.

Input : data/after_pseudo_labeled/species_segment_index.csv
Output: data/after_pseudo_labeled/species_segment_index_PL_split.csv
"""

import math
import os
import random

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
INPUT_CSV  = os.path.join(PROJECT_ROOT, 'data', 'after_pseudo_labeled', 'species_segment_index.csv')
OUTPUT_CSV = os.path.join(PROJECT_ROOT, 'data', 'after_pseudo_labeled', 'species_segment_index_PL_split.csv')

SEED      = 42
VAL_RATIO = 0.10


def assign_split_for_group(rows: pd.DataFrame, rng: random.Random) -> list[str]:
    """Return list of 'train'/'val' (same length as rows) for one assigned_species group."""
    n = len(rows)

    if n == 0:
        return []

    if n == 1:
        return ['train']

    if n == 2:
        indices = [0, 1]
        rng.shuffle(indices)
        splits = ['train', 'train']
        splits[indices[0]] = 'val'
        return splits

    # N >= 3
    n_val  = max(1, math.floor(n * VAL_RATIO))
    labels = rows['original_primary_label'].tolist()
    has_multi = any(';' in str(lbl) for lbl in labels)

    splits = [None] * n

    if not has_multi:
        # Case A: shuffle + assign first n_val to val
        indices = list(range(n))
        rng.shuffle(indices)
        for rank, orig_idx in enumerate(indices):
            splits[orig_idx] = 'val' if rank < n_val else 'train'
        return splits

    # Case B: sub-group by unique label, rarest first
    label_to_positions: dict = {}
    for pos, lbl in enumerate(labels):
        label_to_positions.setdefault(lbl, []).append(pos)

    subgroups = sorted(label_to_positions.items(), key=lambda kv: (len(kv[1]), kv[0]))

    val_budget = n_val

    for _label, positions in subgroups:
        rng.shuffle(positions)
        if val_budget > 0:
            splits[positions[0]] = 'val'
            val_budget -= 1
        for pos in positions[1:]:
            splits[pos] = 'train'

    # Drain remaining budget from the train pool
    if val_budget > 0:
        train_positions = [i for i, s in enumerate(splits) if s == 'train']
        rng.shuffle(train_positions)
        for pos in train_positions[:val_budget]:
            splits[pos] = 'val'

    splits = ['train' if s is None else s for s in splits]
    return splits


def main():
    print(f"Loading {INPUT_CSV} ...")
    df = pd.read_csv(INPUT_CSV)
    n_species = df['assigned_species'].nunique()
    print(f"  {len(df):,} rows  |  {n_species} species")

    rng = random.Random(SEED)

    split_column = [None] * len(df)
    edge_counts  = {'skip': 0, 'single': 0, 'two': 0, 'case_a': 0, 'case_b': 0}

    for species, grp in df.groupby('assigned_species', sort=True):
        n = len(grp)

        if n == 0:
            edge_counts['skip'] += 1
            continue

        splits = assign_split_for_group(grp.reset_index(drop=True), rng)

        for list_pos, df_idx in enumerate(grp.index):
            split_column[df_idx] = splits[list_pos]

        if n == 1:
            edge_counts['single'] += 1
        elif n == 2:
            edge_counts['two'] += 1
        elif any(';' in str(lbl) for lbl in grp['original_primary_label']):
            edge_counts['case_b'] += 1
        else:
            edge_counts['case_a'] += 1

    df['split'] = split_column

    missing = df['split'].isna().sum()
    if missing:
        print(f"  WARNING: {missing} rows have no split assigned")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved to:\n  {OUTPUT_CSV}")

    # Summary
    print(f"\n{'='*60}")
    print("SPLIT SUMMARY")
    print(f"{'='*60}")
    print(f"  Species skipped (N=0)          : {edge_counts['skip']}")
    print(f"  Species N=1  (-> train only)   : {edge_counts['single']}")
    print(f"  Species N=2  (1 val / 1 train) : {edge_counts['two']}")
    print(f"  Species Case A (single-label)  : {edge_counts['case_a']}")
    print(f"  Species Case B (multi-label)   : {edge_counts['case_b']}")
    print()

    split_dist = df['split'].value_counts()
    total = len(df)
    for s in ['train', 'val']:
        cnt = split_dist.get(s, 0)
        pct = cnt / total * 100
        print(f"  {s:5s}: {cnt:>8,} segments  ({pct:.1f}%)")

    print()
    species_in_val = df[df['split'] == 'val']['assigned_species'].nunique()
    print(f"  Species with >= 1 val segment  : {species_in_val} / {n_species}")

    print(f"\n  Split distribution by source_type:")
    print(df.groupby(['source_type', 'split']).size().to_string())
    print(f"\n{'='*60}\n")


if __name__ == '__main__':
    main()

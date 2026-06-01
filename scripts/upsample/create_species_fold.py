"""
create_species_fold.py

Assigns a 3-fold cross-validation column to species_segment_index.csv.

Grouping unit: individual segment (dest_path row), stratified by
original_primary_label within each assigned_species group.

Rules per assigned_species group
---------------------------------
  N == 0  : skip (empty group — shouldn't occur but handled)
  N == 1  : fold 0
  N == 2  : fold 0, fold 1
  N >= 3  :
    Case A — all original_primary_label values are single-species (no ";"):
        Shuffle (fixed seed), then round-robin across 3 folds.
    Case B — any original_primary_label contains ";" (multi-species):
        Sub-group by unique original_primary_label value.
        Process sub-groups rarest-first (fewest segments first).
        Within each sub-group apply the N==1 / N==2 / round-robin edge rules,
        starting the round-robin from the currently least-populated fold.
        This guarantees at least one segment per fold for each unique label
        when N >= 3.

Input : data/train_upsampled_species/species_segment_index.csv
Output: data/train_upsampled_species/species_segment_index.csv  (fold column added in-place)
"""

import os
import sys
import random

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
INPUT_CSV    = os.path.join(PROJECT_ROOT, 'data', 'train_upsampled_species', 'species_segment_index.csv')

N_FOLDS = 3
SEED    = 42


def assign_folds_for_group(rows: pd.DataFrame, rng: random.Random) -> list[int]:
    """
    Return a list of fold ids (same length as rows) for one assigned_species group.
    Rows retain their original DataFrame order; the returned list is index-aligned.
    """
    n = len(rows)

    if n == 0:
        return []

    if n == 1:
        return [0]

    if n == 2:
        return [0, 1]

    # ── N >= 3 ────────────────────────────────────────────────────────────────
    labels = rows['original_primary_label'].tolist()
    has_multi = any(';' in str(lbl) for lbl in labels)

    fold_ids = [None] * n

    if not has_multi:
        # Case A: simple shuffle + round-robin
        indices = list(range(n))
        rng.shuffle(indices)
        for rank, orig_idx in enumerate(indices):
            fold_ids[orig_idx] = rank % N_FOLDS
        return fold_ids

    # Case B: sub-group by unique original_primary_label, rarest first
    fold_counts = [0] * N_FOLDS

    # Build sub-groups: {label -> [positional indices into rows]}
    label_to_positions: dict = {}
    for pos, lbl in enumerate(labels):
        label_to_positions.setdefault(lbl, []).append(pos)

    # Sort sub-groups ascending by size (rarest first), ties broken by label string
    subgroups = sorted(label_to_positions.items(), key=lambda kv: (len(kv[1]), kv[0]))

    for _label, positions in subgroups:
        m = len(positions)
        rng.shuffle(positions)  # shuffle within sub-group for reproducibility

        if m == 1:
            f = fold_counts.index(min(fold_counts))
            fold_ids[positions[0]] = f
            fold_counts[f] += 1

        elif m == 2:
            # Place into the two least-populated folds
            sorted_folds = sorted(range(N_FOLDS), key=lambda k: (fold_counts[k], k))
            for i, f in enumerate(sorted_folds[:2]):
                fold_ids[positions[i]] = f
                fold_counts[f] += 1

        else:
            # Round-robin starting from least-populated fold
            start = fold_counts.index(min(fold_counts))
            for i, pos in enumerate(positions):
                f = (start + i) % N_FOLDS
                fold_ids[pos] = f
                fold_counts[f] += 1

    return fold_ids


def main():
    print(f"Loading {INPUT_CSV} ...")
    df = pd.read_csv(INPUT_CSV)
    print(f"  {len(df):,} rows  |  {df['assigned_species'].nunique()} species")

    rng = random.Random(SEED)

    fold_column = [None] * len(df)

    edge_counts = {'skip': 0, 'single': 0, 'two': 0, 'case_a': 0, 'case_b': 0}

    for species, grp in df.groupby('assigned_species', sort=True):
        n = len(grp)

        if n == 0:
            edge_counts['skip'] += 1
            continue

        folds = assign_folds_for_group(grp.reset_index(drop=True), rng)

        for list_pos, df_idx in enumerate(grp.index):
            fold_column[df_idx] = folds[list_pos]

        if n == 1:
            edge_counts['single'] += 1
        elif n == 2:
            edge_counts['two'] += 1
        elif any(';' in str(lbl) for lbl in grp['original_primary_label']):
            edge_counts['case_b'] += 1
        else:
            edge_counts['case_a'] += 1

    df['fold'] = fold_column
    missing = df['fold'].isna().sum()
    if missing:
        print(f"  WARNING: {missing} rows could not be assigned a fold")

    df.to_csv(INPUT_CSV, index=False)
    print(f"Saved updated index to:\n  {INPUT_CSV}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FOLD ASSIGNMENT SUMMARY")
    print(f"{'='*60}")
    print(f"  Species skipped (N=0)        : {edge_counts['skip']}")
    print(f"  Species with 1 segment       : {edge_counts['single']}")
    print(f"  Species with 2 segments      : {edge_counts['two']}")
    print(f"  Species Case A (single-label): {edge_counts['case_a']}")
    print(f"  Species Case B (multi-label) : {edge_counts['case_b']}")
    print()

    fold_dist = df['fold'].value_counts().sort_index()
    for f, cnt in fold_dist.items():
        print(f"  Fold {f}: {cnt:>8,} segments")
    print()

    print("  Fold distribution by source_type:")
    print(df.groupby(['source_type', 'fold']).size().to_string())
    print(f"\n{'='*60}\n")


if __name__ == '__main__':
    main()

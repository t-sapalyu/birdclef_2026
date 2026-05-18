"""Generate label2idx.json - the canonical species-code -> index mapping."""

import os
import json
import pandas as pd

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)

INPUT_CSV   = os.path.join(PROJECT_ROOT, 'data', 'taxonomy.csv')
OUTPUT_JSON = os.path.join(PROJECT_ROOT, 'data', 'label2idx.json')

EXPECTED_N_CLASSES = 234


def main():
    print(f"Reading: {INPUT_CSV}")
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(
            f"Expected {INPUT_CSV}. Place taxonomy.csv in the data/ folder."
        )

    tax = pd.read_csv(INPUT_CSV)
    print(f"  Rows: {len(tax)}  columns: {list(tax.columns)}")

    if 'primary_label' not in tax.columns:
        raise ValueError(
            f"taxonomy.csv has no 'primary_label' column. "
            f"Found: {sorted(tax.columns)}"
        )

    # Sort the species codes so the mapping is deterministic - the SAME
    # input always produces the SAME ordering, on any machine.
    species = sorted(tax['primary_label'].astype(str).unique())
    label2idx = {code: idx for idx, code in enumerate(species)}

    # Sanity checks
    if len(label2idx) != EXPECTED_N_CLASSES:
        print(f"  WARNING: got {len(label2idx)} classes, "
              f"expected {EXPECTED_N_CLASSES}.")
    assert len(label2idx) == len(species), "duplicate species codes!"

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(label2idx, f, indent=2)

    print(f"\nWrote: {OUTPUT_JSON}")
    print(f"  {len(label2idx)} species mapped to indices "
          f"0..{len(label2idx) - 1}")
    print(f"  First 3:  "
          f"{dict(list(label2idx.items())[:3])}")
    print(f"  Last 3:   "
          f"{dict(list(label2idx.items())[-3:])}")


if __name__ == '__main__':
    main()
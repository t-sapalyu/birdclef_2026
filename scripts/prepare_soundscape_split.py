import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)
sys.path.insert(0, PROJECT_ROOT)

from birdclef_utils.folds import (
    create_soundscape_folds, check_soundscape_folds,
)

INPUT_CSV  = os.path.join(PROJECT_ROOT, 'data',
                          'train_soundscapes_labels.csv')
OUTPUT_CSV = os.path.join(PROJECT_ROOT, 'data',
                          'labeled_soundscapes_split.csv')

N_FOLDS = 3
SEED    = 42

# Columns the raw competition CSV is expected to contain
EXPECTED_COLS = {'filename', 'start', 'end', 'primary_label'}


def time_to_seconds(t):
    """Convert a time value to integer seconds.

    Handles:
      - time strings: '0:00:05', '0:01:00', '1:23'
      - plain numbers (already in seconds): 5, 10, 0
    """
    s = str(t).strip()

    # Already a plain number (no colon) -> treat as seconds directly
    if ':' not in s:
        return int(float(s))

    # Time string with colons
    parts = s.split(':')
    if len(parts) == 3:
        h, m, sec = parts
    elif len(parts) == 2:
        h, m, sec = '0', parts[0], parts[1]
    else:
        h, m, sec = '0', '0', parts[0]
    return int(h) * 3600 + int(m) * 60 + int(float(sec))


def main():
    # ---- Load ----
    print(f"Reading: {INPUT_CSV}")
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(
            f"Expected the CSV at {INPUT_CSV} but it is not there. "
            f"Place train_soundscapes_labels.csv in the data/ folder."
        )

    ss = pd.read_csv(INPUT_CSV)
    print(f"  Raw rows: {len(ss)}")
    print(f"  Columns:  {list(ss.columns)}")

    # ---- Structure check ----
    missing = EXPECTED_COLS - set(ss.columns)
    if missing:
        raise ValueError(
            f"Input CSV is missing expected columns: {sorted(missing)}. "
            f"Found columns: {sorted(ss.columns)}. "
            f"Make sure data/train_soundscapes_labels.csv is the raw "
            f"competition file."
        )
    extra = set(ss.columns) - EXPECTED_COLS
    if extra:
        print(f"  Note: extra columns present {sorted(extra)} - this CSV "
              f"may have been processed before. Proceeding anyway.")

    # ---- 1. Deduplicate ----
    before = len(ss)
    ss = ss.drop_duplicates(
        subset=['filename', 'start', 'end']
    ).reset_index(drop=True)
    removed = before - len(ss)
    print(f"\nDeduplication: {before} -> {len(ss)} rows "
          f"(removed {removed})")
    if removed == 0:
        print("  Source CSV had no duplicate rows.")
    else:
        print("  Source CSV contained duplicate rows - now removed.")

    # ---- 2. Numeric start / end seconds ----
    ss['start_sec'] = ss['start'].apply(time_to_seconds)
    ss['end_sec']   = ss['end'].apply(time_to_seconds)

    chunk_lens = ss['end_sec'] - ss['start_sec']
    print(f"\nChunk length distribution (seconds):")
    print(chunk_lens.value_counts().sort_index())
    if (chunk_lens != 5).any():
        n_odd = (chunk_lens != 5).sum()
        print(f"  WARNING: {n_odd} chunk(s) are not exactly 5 seconds.")

    # ---- 3. chunk_id ----
    # Format: <filename without .ogg>_<end second>
    # Matches the competition row_id convention.
    ss['chunk_id'] = (
        ss['filename'].str.replace('.ogg', '', regex=False)
        + '_' + ss['end_sec'].astype(str)
    )
    print(f"\nSample chunk_ids: {ss['chunk_id'].head(3).tolist()}")

    # Sanity: chunk_id should be unique after deduplication
    n_dup_ids = ss['chunk_id'].duplicated().sum()
    if n_dup_ids > 0:
        print(f"  WARNING: {n_dup_ids} duplicate chunk_id values remain - "
              f"two chunks share the same file and end second.")

    # ---- 4. File-grouped 3-fold split ----
    print(f"\nCreating file-grouped {N_FOLDS}-fold split...")
    ss = create_soundscape_folds(ss, n_folds=N_FOLDS, seed=SEED)
    check_soundscape_folds(ss, n_folds=N_FOLDS)

    # ---- 5. Save ----
    ss.to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote: {OUTPUT_CSV}")
    print(f"  Final shape:   {ss.shape}")
    print(f"  Final columns: {list(ss.columns)}")
    print(f"  Chunks per fold: "
          f"{ss['fold'].value_counts().sort_index().to_dict()}")


if __name__ == '__main__':
    main()
import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)
sys.path.insert(0, PROJECT_ROOT)

from birdclef_utils.folds import create_audio_folds, check_audio_folds

INPUT_CSV  = os.path.join(PROJECT_ROOT, 'data', 'train.csv')
OUTPUT_CSV = os.path.join(PROJECT_ROOT, 'data', 'train_folds.csv')

N_FOLDS = 3
SEED    = 42

# Columns the script relies on. train.csv has many more (rating, latitude,
# collection, ...) which are all preserved - these are just the required ones.
REQUIRED_COLS = {'filename', 'primary_label', 'author'}


def main():
    # ---- Load ----
    print(f"Reading: {INPUT_CSV}")
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(
            f"Expected the CSV at {INPUT_CSV} but it is not there. "
            f"Place train.csv in the data/ folder."
        )

    df = pd.read_csv(INPUT_CSV)
    print(f"  Rows:    {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    # ---- Structure check ----
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Input CSV is missing required columns: {sorted(missing)}. "
            f"Found columns: {sorted(df.columns)}. "
            f"Make sure data/train.csv is the raw competition file."
        )

    print(f"\n  Unique species: {df['primary_label'].nunique()}")
    print(f"  Unique authors: {df['author'].nunique()}")
    n_unknown = (
        df['author'].astype(str).str.strip().str.lower() == 'unknown'
    ).sum()
    n_nan = df['author'].isna().sum()
    print(f"  Author = 'Unknown' string: {n_unknown}")
    print(f"  Author = NaN:              {n_nan}")

    if 'secondary_labels' in df.columns:
        n_sec = df['secondary_labels'].notna().sum()
        print(f"  Rows with secondary_labels: {n_sec}")
    else:
        print("  Note: no 'secondary_labels' column found.")

    # ---- Create folds ----
    # Author-grouped, primary-species-stratified. Secondary labels are NOT
    # used for stratification (multi-label); they are preserved in the
    # output and used later in the model's target vector.
    print(f"\nCreating author-grouped {N_FOLDS}-fold split...")
    df = create_audio_folds(df, n_folds=N_FOLDS, seed=SEED)

    # ---- Verify (asserts zero author leakage internally) ----
    check_audio_folds(df, n_folds=N_FOLDS)

    # ---- Save ----
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote: {OUTPUT_CSV}")
    print(f"  Final shape:   {df.shape}")
    print(f"  Final columns: {list(df.columns)}")
    print(f"  Rows per fold: "
          f"{df['fold'].value_counts().sort_index().to_dict()}")


if __name__ == '__main__':
    main()
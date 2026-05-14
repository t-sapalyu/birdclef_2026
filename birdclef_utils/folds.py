import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

def create_folds(df, n_folds=3, seed=42):
    """
    Author-grouped, stratified k-fold for train_audio recordings.

    Handles missing authors stored as NaN, empty strings, OR the literal
    string "Unknown" (BirdCLEF uses the string form). Each missing-author
    row gets a unique placeholder so they distribute across all folds
    instead of clumping into one.
    """
    _validate_input(df, required_cols=['primary_label', 'author'])

    df = df.copy()
    df['fold'] = -1

    # Detect missing authors: NaN, empty string, or literal "Unknown"
    # (case-insensitive, whitespace-tolerant)
    author = df['author']
    is_missing = (
        author.isna()
        | (author.astype(str).str.strip().str.lower() == 'unknown')
        | (author.astype(str).str.strip() == '')
    )

    # Start author_filled as the original author string
    df['author_filled'] = author.astype(str)

    # Replace missing ones with UNIQUE placeholders (unknown_<row index>)
    # so each is its own group and can land in any fold
    df.loc[is_missing, 'author_filled'] = [
        f'unknown_{i}' for i in df.index[is_missing]
    ]

    sgkf = StratifiedGroupKFold(
        n_splits=n_folds, shuffle=True, random_state=seed
    )

    for fold_idx, (_, val_idx) in enumerate(
        sgkf.split(
            df,
            y=df['primary_label'],
            groups=df['author_filled'],
        )
    ):
        df.loc[df.index[val_idx], 'fold'] = fold_idx

    assert (df['fold'] != -1).all(), \
        "Some rows were not assigned to any fold - check input data."

    return df


def split_labeled_soundscapes(df, val_fraction=0.2, seed=42):
    """
    File-level 80/20 train/val split for labeled soundscapes.

    All chunks from the same soundscape file go to the same side of
    the split, to avoid within-file leakage (chunks from the same
    recording share location/weather/recorder characteristics).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'filename' column. Typically this is the result
        of reading `train_soundscapes_labels.csv`.
    val_fraction : float
        Fraction of *files* to assign to validation (default 0.2).
    seed : int
        Random state for reproducibility.

    Returns
    -------
    pd.DataFrame
        Copy of `df` with a new column:
          - 'split': either 'train' or 'val'.
    """
    _validate_input(df, required_cols=['filename'])

    df = df.copy()
    unique_files = df['filename'].unique()

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_files)

    n_val = int(round(len(shuffled) * val_fraction))
    val_files = set(shuffled[:n_val])

    df['split'] = df['filename'].apply(
        lambda f: 'val' if f in val_files else 'train'
    )

    return df


def check_fold_quality(df, n_folds=3, verbose=True):
    """
    Verify fold quality after `create_folds`.

    Checks performed:
      - Fold sizes (should be roughly balanced)
      - Author overlap between train and val (must be 0 for all folds)
      - Species coverage per fold (informational - rare species may be
        absent from some folds, which is expected)

    Returns
    -------
    dict
        Summary statistics.
    """
    _validate_input(df, required_cols=['primary_label', 'author_filled', 'fold'])

    results = {
        'fold_sizes': {},
        'species_per_fold': {},
        'author_overlap': {},
        'missing_species_per_fold': {},
    }

    all_species = set(df['primary_label'].unique())

    for fold in range(n_folds):
        in_fold = df['fold'] == fold
        not_in_fold = df['fold'] != fold

        # Fold size
        results['fold_sizes'][fold] = int(in_fold.sum())

        # Species coverage
        val_species = set(df.loc[in_fold, 'primary_label'])
        results['species_per_fold'][fold] = len(val_species)
        results['missing_species_per_fold'][fold] = len(
            all_species - val_species
        )

        # Author overlap (MUST be zero)
        val_authors = set(df.loc[in_fold, 'author_filled'])
        train_authors = set(df.loc[not_in_fold, 'author_filled'])
        overlap = val_authors & train_authors
        results['author_overlap'][fold] = len(overlap)

    if verbose:
        _print_fold_report(results, n_folds, total_species=len(all_species))

    # Hard assertion: author overlap is a correctness bug, not a warning
    for fold, n in results['author_overlap'].items():
        assert n == 0, (
            f"Author leakage detected in fold {fold}: "
            f"{n} authors appear in both train and val. "
            f"Fold creation is broken."
        )

    return results


def check_soundscape_split_quality(df, verbose=True):
    """
    Verify the labeled soundscape split.

    Returns summary stats for thesis reporting.
    """
    _validate_input(df, required_cols=['filename', 'split', 'primary_label'])

    n_train_files = df[df['split'] == 'train']['filename'].nunique()
    n_val_files = df[df['split'] == 'val']['filename'].nunique()
    n_train_chunks = (df['split'] == 'train').sum()
    n_val_chunks = (df['split'] == 'val').sum()

    # Check no file is in both sides
    train_files = set(df[df['split'] == 'train']['filename'])
    val_files = set(df[df['split'] == 'val']['filename'])
    overlap = train_files & val_files

    results = {
        'n_train_files': int(n_train_files),
        'n_val_files': int(n_val_files),
        'n_train_chunks': int(n_train_chunks),
        'n_val_chunks': int(n_val_chunks),
        'file_overlap': len(overlap),
    }

    if verbose:
        print("=== Labeled Soundscape Split ===")
        print(f"Train files:  {results['n_train_files']:>5}  "
              f"({results['n_train_chunks']:>6} chunks)")
        print(f"Val files:    {results['n_val_files']:>5}  "
              f"({results['n_val_chunks']:>6} chunks)")
        print(f"File overlap: {results['file_overlap']:>5}  (must be 0)")

    assert results['file_overlap'] == 0, (
        f"File leakage in soundscape split: {results['file_overlap']} files "
        f"appear in both train and val."
    )

    return results


def _validate_input(df, required_cols):
    """Check that a DataFrame has the columns we need."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}")
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(
            f"DataFrame is missing required columns: {sorted(missing)}. "
            f"Available columns: {sorted(df.columns)}"
        )


def _print_fold_report(results, n_folds, total_species):
    print("=== Fold Quality Report ===\n")

    print("Fold sizes:")
    for fold in range(n_folds):
        print(f"  Fold {fold}: {results['fold_sizes'][fold]:>6} recordings")

    print("\nSpecies coverage in validation:")
    for fold in range(n_folds):
        n_present = results['species_per_fold'][fold]
        n_missing = results['missing_species_per_fold'][fold]
        print(f"  Fold {fold}: {n_present:>3} / {total_species} species "
              f"present  ({n_missing} missing)")

    print("\nAuthor overlap (train ∩ val, must be 0):")
    for fold in range(n_folds):
        n = results['author_overlap'][fold]
        flag = "OK" if n == 0 else "LEAKAGE"
        print(f"  Fold {fold}: {n} authors overlap  [{flag}]")
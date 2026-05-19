import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold


def create_audio_folds(df, n_folds=3, seed=42):
    """Author-grouped, primary-species-stratified k-fold for train_audio."""

    _validate_input(df, required_cols=['primary_label', 'author'])

    df = df.copy()
    df['fold'] = -1

    # Treat NaN, empty string, and literal "Unknown" (any case) as missing.
    author = df['author']
    is_missing = (
        author.isna()
        | (author.astype(str).str.strip().str.lower() == 'unknown')
        | (author.astype(str).str.strip() == '')
    )
    # Each missing-author row gets a UNIQUE placeholder so it is its own
    # group and can land in any fold (rather than all clumping into one).
    df['author_filled'] = author.astype(str)
    df.loc[is_missing, 'author_filled'] = [
        f'unknown_{i}' for i in df.index[is_missing]
    ]

    sgkf = StratifiedGroupKFold(
        n_splits=n_folds, shuffle=True, random_state=seed
    )
    for fold_idx, (_, val_idx) in enumerate(
        sgkf.split(
            df,
            y=df['primary_label'],          # stratify on primary only
            groups=df['author_filled'],     # group by author
        )
    ):
        df.loc[df.index[val_idx], 'fold'] = fold_idx

    assert (df['fold'] != -1).all(), \
        "Some train_audio rows were not assigned to any fold."

    return df


def check_audio_folds(df, n_folds=3, verbose=True):
    """Verify train_audio folds: zero author leakage, balanced sizes."""

    _validate_input(
        df, required_cols=['primary_label', 'author_filled', 'fold']
    )

    results = {
        'fold_sizes': {},
        'species_per_fold': {},
        'missing_species_per_fold': {},
        'author_overlap': {},
    }
    all_species = set(df['primary_label'].unique())

    for fold in range(n_folds):
        in_fold = df['fold'] == fold
        not_in = df['fold'] != fold

        results['fold_sizes'][fold] = int(in_fold.sum())

        val_species = set(df.loc[in_fold, 'primary_label'])
        results['species_per_fold'][fold] = len(val_species)
        results['missing_species_per_fold'][fold] = len(
            all_species - val_species
        )

        val_authors = set(df.loc[in_fold, 'author_filled'])
        train_authors = set(df.loc[not_in, 'author_filled'])
        results['author_overlap'][fold] = len(val_authors & train_authors)

    if verbose:
        print("=== train_audio Fold Quality ===")
        for fold in range(n_folds):
            print(f"  Fold {fold}: {results['fold_sizes'][fold]:>6} rows | "
                  f"{results['species_per_fold'][fold]:>3} species in val | "
                  f"{results['missing_species_per_fold'][fold]:>3} species "
                  f"missing | author overlap = "
                  f"{results['author_overlap'][fold]}")

    for fold, n in results['author_overlap'].items():
        assert n == 0, (
            f"Author leakage in train_audio fold {fold}: {n} authors "
            f"in both train and val. Fold creation is broken."
        )

    return results


def create_soundscape_folds(df, n_folds=3, seed=42):
    """File-grouped k-fold for labeled soundscape chunks."""

    _validate_input(df, required_cols=['filename'])

    df = df.copy()
    df['fold'] = -1

    gkf = GroupKFold(n_splits=n_folds)
    for fold_idx, (_, val_idx) in enumerate(
        gkf.split(df, groups=df['filename'])
    ):
        df.loc[df.index[val_idx], 'fold'] = fold_idx

    assert (df['fold'] != -1).all(), \
        "Some soundscape chunks were not assigned to any fold."

    return df


def check_soundscape_folds(df, n_folds=3, verbose=True):
    """Verify soundscape folds: zero file leakage, balanced sizes."""

    _validate_input(df, required_cols=['filename', 'fold'])

    results = {'fold_sizes': {}, 'fold_files': {}, 'file_overlap': {}}

    for fold in range(n_folds):
        in_fold = df['fold'] == fold
        not_in = df['fold'] != fold

        results['fold_sizes'][fold] = int(in_fold.sum())
        results['fold_files'][fold] = int(
            df.loc[in_fold, 'filename'].nunique()
        )

        val_files = set(df.loc[in_fold, 'filename'])
        train_files = set(df.loc[not_in, 'filename'])
        results['file_overlap'][fold] = len(val_files & train_files)

    if verbose:
        print("=== labeled soundscape Fold Quality ===")
        for fold in range(n_folds):
            print(f"  Fold {fold}: {results['fold_sizes'][fold]:>5} chunks "
                  f"from {results['fold_files'][fold]:>3} files | "
                  f"file overlap = {results['file_overlap'][fold]}")

    for fold, n in results['file_overlap'].items():
        assert n == 0, (
            f"File leakage in soundscape fold {fold}: {n} files in both "
            f"train and val."
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
            f"Available: {sorted(df.columns)}"
        )
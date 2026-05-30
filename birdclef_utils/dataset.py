import os
import ast

import numpy as np
import torch
import librosa
from torch.utils.data import Dataset

from .audio import (
    load_audio, normalize_waveform, fix_length, audio_to_melspec_torch,
)
from .constants import SR, N_SAMPLES, NUM_CLASSES


def parse_label_list(value):
    """Parse a label field into a list of species code strings.

    Handles the formats seen in BirdCLEF CSVs:
      - Python-list string : "['stbflu1', 'barwha1']"
      - semicolon-separated: "22961;23158;24321"
      - space-separated    : "stbflu1 barwha1"
      - empty / missing    : "", "[]", NaN, None

    Returns
    -------
    list[str]
        Possibly-empty list of code strings.
    """
    if value is None:
        return []
    s = str(value).strip()
    if s in ('', '[]', 'nan', 'NaN', 'None'):
        return []

    # Try a Python literal list first (focal secondary_labels often is one)
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple)):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except (ValueError, SyntaxError):
        pass

    # Fall back: split on ';' or whitespace (soundscape primary_label uses ';')
    return [c.strip() for c in s.replace(';', ' ').split() if c.strip()]


def build_multihot(codes, label2idx, num_classes=NUM_CLASSES):
    """Build a multi-hot float32 target vector from a list of species codes.

    Codes not present in `label2idx` are silently skipped (e.g. a label
    that is not one of the NUM_CLASSES target classes).
    """
    label = torch.zeros(num_classes, dtype=torch.float32)
    for code in codes:
        idx = label2idx.get(code)
        if idx is not None:
            label[idx] = 1.0
    return label


class FocalDataset(Dataset):
    """Dataset for train_audio focal recordings."""

    def __init__(self, df, audio_dir, label2idx,
                 mode='train', use_secondary=True, max_duration=30.0,
                 waveform_transform=None, spec_transform=None):
        self.df = df.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.label2idx = label2idx
        self.mode = mode
        self.use_secondary = use_secondary
        self.max_duration = max_duration
        self.waveform_transform = waveform_transform
        self.spec_transform = spec_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        for _ in range(10):
            row = self.df.iloc[idx]
            filepath = os.path.join(self.audio_dir, row['filename'])

            # --- load ---
            y, _ = load_audio(filepath, max_duration=self.max_duration)
            if y is None:
                idx = np.random.randint(len(self))
                continue

            # --- preprocess ---
            y = normalize_waveform(y)

            # waveform-domain augmentation hook (Task 3)
            if self.mode == 'train' and self.waveform_transform is not None:
                y = self.waveform_transform(y)

            # exact length: random crop for training, center crop for val
            crop_mode = 'random' if self.mode == 'train' else 'center'
            y = fix_length(y, N_SAMPLES, crop_mode=crop_mode)

            # --- mel spectrogram (computed on the fly) ---
            spec = audio_to_melspec_torch(y)              # (1, N_MELS, time)

            # spectrogram-domain augmentation hook (Task 3, e.g. SpecAugment)
            if self.mode == 'train' and self.spec_transform is not None:
                spec = self.spec_transform(spec)

            # --- label ---
            codes = [row['primary_label']]
            if self.use_secondary and 'secondary_labels' in row:
                codes += parse_label_list(row.get('secondary_labels'))
            label = build_multihot(codes, self.label2idx)

            return spec, label

        raise RuntimeError(
            f"FocalDataset: could not load a valid sample after 10 attempts "
            f"(last tried: {filepath})"
        )


class SoundscapeChunkDataset(Dataset):
    """Dataset for labeled soundscape chunks.

    Parameters
    ----------
    df : pd.DataFrame
        Rows of labeled_soundscapes_split.csv. Needs 'filename',
        'start_sec', and 'primary_label' (a ';'-separated code list).
    audio_dir : str
        Directory containing the soundscape .ogg files.
    label2idx : dict
        Maps species code -> integer class index.
    mode : {'train', 'val'}
        Only affects whether augmentation hooks fire. The 5 s window is
        ALWAYS the exact chunk window - soundscape labels are strong and
        aligned to fixed 5 s boundaries, so the audio must match.
    waveform_transform, spec_transform : callable or None
        Same augmentation hooks as FocalDataset (Task 3).
    """

    def __init__(self, df, audio_dir, label2idx,
                 mode='train', waveform_transform=None, spec_transform=None):
        self.df = df.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.label2idx = label2idx
        self.mode = mode
        self.waveform_transform = waveform_transform
        self.spec_transform = spec_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        for _ in range(10):
            row = self.df.iloc[idx]
            filepath = os.path.join(self.audio_dir, row['filename'])

            # --- load the EXACT 5 s chunk window ---
            try:
                y, _ = librosa.load(
                    filepath,
                    sr=SR,
                    offset=float(row['start_sec']),
                    duration=5.0,
                    mono=True,
                )
                y = np.asarray(y, dtype=np.float32)
            except Exception as e:
                print(f"SoundscapeChunkDataset: failed to load "
                      f"{filepath}@{row['start_sec']}s: {e}")
                idx = np.random.randint(len(self))
                continue

            if y.size == 0:
                idx = np.random.randint(len(self))
                continue

            # --- preprocess ---
            y = normalize_waveform(y)

            if self.mode == 'train' and self.waveform_transform is not None:
                y = self.waveform_transform(y)

            # Pad/crop to exact length. crop_mode='start' because the chunk
            # window is already fixed - we just normalize any rounding drift.
            y = fix_length(y, N_SAMPLES, crop_mode='start')

            # --- mel spectrogram (on the fly) ---
            spec = audio_to_melspec_torch(y)              # (1, N_MELS, time)

            if self.mode == 'train' and self.spec_transform is not None:
                spec = self.spec_transform(spec)

            # --- label: soundscape primary_label is a ';'-separated list ---
            codes = parse_label_list(row['primary_label'])
            label = build_multihot(codes, self.label2idx)

            return spec, label

        raise RuntimeError(
            f"SoundscapeChunkDataset: could not load a valid sample after "
            f"10 attempts (last tried: {filepath}@{row['start_sec']}s)"
        )
import os
import sys
import ast
import random
import logging
import numpy as np
import pandas as pd
import soundfile as sf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from birdclef_utils.constants import SR
from birdclef_utils.audio import load_audio

# ── Paths ────────────────────────────────────────────────────────────────────
COUNTS_CSV   = os.path.join(PROJECT_ROOT, 'visualization', 'train_species_counts.csv')
TRAIN_CSV    = os.path.join(PROJECT_ROOT, 'data', 'train.csv')
AUDIO_DIR    = os.path.join(PROJECT_ROOT, 'data', 'train_audio')
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, 'data', 'train_audio_upsampled')
MANIFEST_CSV = os.path.join(PROJECT_ROOT, 'data', 'train_upsampled.csv')

# ── Config ───────────────────────────────────────────────────────────────────
SEGMENT_SEC     = 5
SEGMENT_SAMPLES = SEGMENT_SEC * SR   # 160 000 samples
SHORT_THRESHOLD = 8                  # sec; below this → fixed sequential starts
MAX_LOAD_SEC    = 30.0
SEED            = 42

# Count-range → (n_segments, fixed sequential starts in seconds)
TIER_TABLE = [
    (1,   100,  4, [0, 1, 2, 3]),
    (101, 300,  3, [0, 1, 2]),
    (301, 500,  2, [0, 1]),
    (501, 1200, 1, [0]),
]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_tier(count: int):
    """Return (n_segments, fixed_starts) for a species count, or None to skip."""
    for lo, hi, n_seg, fixed in TIER_TABLE:
        if lo <= count <= hi:
            return n_seg, fixed
    return None


def parse_secondary_labels(raw) -> list:
    """Parse secondary_labels string (e.g. \"['compau', 'saffin']\") to a list."""
    if not isinstance(raw, str) or raw.strip() in ('[]', ''):
        return []
    try:
        parsed = ast.literal_eval(raw)
        return [str(s) for s in parsed] if isinstance(parsed, list) else []
    except Exception:
        return []


def extract_segments(waveform: np.ndarray, n_segments: int, fixed_starts: list) -> list:
    """
    Return a list of (start_sec, chunk) tuples.

    - duration <= 0        → empty list (caller skips)
    - 0 < duration < 8 sec → fixed sequential starts, zero-pad each chunk
    - duration >= 8 sec    → n_segments random starts, zero-pad edge chunks
    """
    duration_sec = len(waveform) / SR

    if duration_sec <= 0:
        return []

    if duration_sec < SHORT_THRESHOLD:
        # Fixed starts; every chunk is padded to SEGMENT_SAMPLES
        segments = []
        for start_sec in fixed_starts:
            start_idx = int(start_sec * SR)
            chunk = waveform[start_idx: start_idx + SEGMENT_SAMPLES]
            if len(chunk) < SEGMENT_SAMPLES:
                chunk = np.pad(chunk, (0, SEGMENT_SAMPLES - len(chunk)), mode='constant')
            segments.append((float(start_sec), chunk))
        return segments

    # Random starts within the valid range
    max_start_sec = max(0.0, duration_sec - SEGMENT_SEC)
    starts = [random.uniform(0.0, max_start_sec) for _ in range(n_segments)]

    segments = []
    for start_sec in starts:
        start_idx = int(start_sec * SR)
        chunk = waveform[start_idx: start_idx + SEGMENT_SAMPLES]
        if len(chunk) < SEGMENT_SAMPLES:
            chunk = np.pad(chunk, (0, SEGMENT_SAMPLES - len(chunk)), mode='constant')
        segments.append((start_sec, chunk))
    return segments


def save_segment(chunk: np.ndarray, species: str, stem: str, seg_idx: int) -> str:
    """Write a segment to OUTPUT_DIR and return the relative path."""
    species_dir = os.path.join(OUTPUT_DIR, species)
    os.makedirs(species_dir, exist_ok=True)
    out_name = f"{stem}_seg{seg_idx}.ogg"
    out_path = os.path.join(species_dir, out_name)
    sf.write(out_path, chunk, SR, format='OGG', subtype='VORBIS')
    return os.path.join(species, out_name).replace('\\', '/')


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    random.seed(SEED)
    np.random.seed(SEED)

    # ── Load inputs ──────────────────────────────────────────────────────────
    counts_df = pd.read_csv(COUNTS_CSV)
    train_df  = pd.read_csv(TRAIN_CSV)

    # Parse secondary_labels once up front
    train_df['_secondary_list'] = train_df['secondary_labels'].apply(parse_secondary_labels)

    # ── Build species → tier map ──────────────────────────────────────────────
    rare_species = {}   # species_code → (n_segments, fixed_starts)
    for _, row in counts_df.iterrows():
        tier = get_tier(int(row['count']))
        if tier is not None:
            rare_species[str(row['species'])] = tier

    log.info("Rare species in scope: %d", len(rare_species))

    # ── Collect work: (species, filename, abs_path, n_segments, fixed_starts, was_secondary) ─
    processed_files = set()   # filenames already queued; prevents double-processing
    work_queue = []

    for species, (n_seg, fixed) in rare_species.items():

        # Primary pass
        primary_rows = train_df[train_df['primary_label'].astype(str) == species]
        for _, row in primary_rows.iterrows():
            fname = row['filename']
            if fname not in processed_files:
                processed_files.add(fname)
                work_queue.append((species, fname, n_seg, fixed, False))

        # Secondary pass — species appears in another file's secondary_labels
        secondary_rows = train_df[
            train_df['_secondary_list'].apply(lambda lst: species in lst)
        ]
        for _, row in secondary_rows.iterrows():
            fname = row['filename']
            if fname not in processed_files:   # already-chunked guard
                processed_files.add(fname)
                work_queue.append((species, fname, n_seg, fixed, True))

    log.info("Files queued for segmentation: %d", len(work_queue))

    # ── Process ───────────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    manifest_rows = []
    skipped = 0
    total_segments = 0

    for i, (species, fname, n_seg, fixed, was_secondary) in enumerate(work_queue, 1):
        audio_path = os.path.join(AUDIO_DIR, fname)

        if not os.path.exists(audio_path):
            log.warning("[%d/%d] Missing: %s", i, len(work_queue), audio_path)
            skipped += 1
            continue

        waveform, _ = load_audio(audio_path, max_duration=MAX_LOAD_SEC)
        if waveform is None:
            log.warning("[%d/%d] Load failed: %s", i, len(work_queue), audio_path)
            skipped += 1
            continue

        duration_sec = len(waveform) / SR

        if duration_sec <= 0:
            log.warning("[%d/%d] Empty waveform: %s", i, len(work_queue), audio_path)
            skipped += 1
            continue

        segments = extract_segments(waveform, n_seg, fixed)
        if not segments:
            skipped += 1
            continue

        stem = os.path.splitext(os.path.basename(fname))[0]

        for seg_idx, (start_sec, chunk) in enumerate(segments):
            rel_path = save_segment(chunk, species, stem, seg_idx)
            manifest_rows.append({
                'primary_label':       species,
                'filename':            rel_path,
                'source_file':         fname,
                'segment_index':       seg_idx,
                'source_duration_sec': round(duration_sec, 3),
                'start_sec':           round(start_sec, 3),
                'was_secondary':       was_secondary,
            })
            total_segments += 1

        if i % 500 == 0:
            log.info("Progress: %d / %d files | %d segments written", i, len(work_queue), total_segments)

    # ── Write manifest ────────────────────────────────────────────────────────
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(MANIFEST_CSV, index=False)

    log.info("Done.")
    log.info("  Files processed : %d", len(work_queue) - skipped)
    log.info("  Files skipped   : %d", skipped)
    log.info("  Segments written: %d", total_segments)
    log.info("  Manifest CSV    : %s", MANIFEST_CSV)
    log.info("  Output dir      : %s", OUTPUT_DIR)


if __name__ == '__main__':
    main()

"""
upsample_soundscape_segments.py

Segment-level upsampling for multi-label bird soundscape data.
Upsampling factor is determined by the rarest species in each segment,
preventing duplicate upsampling when multiple rare species share a segment.
Segments whose rarest species has count > 100 are left as-is.

Input:
  data/train_soundscapes/{filename}.ogg   — source 60-second soundscape files
  data/train_soundscapes_labels.csv       — original segment labels
  visualization/soundscape_species_counts.csv

Output:
  data/train_soundscapes_upsampled/       — extracted 5-second OGG clips
  data/train_soundscapes_labels_upsampled.csv
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import soundfile as sf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from birdclef_utils.constants import SR
from birdclef_utils.audio import load_audio

# ── Paths ─────────────────────────────────────────────────────────────────────
COUNTS_CSV        = os.path.join(PROJECT_ROOT, 'visualization', 'soundscape_species_counts.csv')
LABELS_CSV        = os.path.join(PROJECT_ROOT, 'data', 'train_soundscapes_labels.csv')
SOUNDSCAPES_DIR   = os.path.join(PROJECT_ROOT, 'data', 'train_soundscapes')
UPSAMPLED_DIR     = os.path.join(PROJECT_ROOT, 'data', 'train_soundscapes_upsampled')
OUTPUT_CSV        = os.path.join(PROJECT_ROOT, 'data', 'train_soundscapes_labels_upsampled.csv')
COUNTS_REPORT_CSV = os.path.join(PROJECT_ROOT, 'visualization', 'soundscape_species_counts_upsampled.csv')

# ── Config ────────────────────────────────────────────────────────────────────
AUDIO_DURATION   = 60                   # each soundscape is exactly 60 seconds
WINDOW_SIZE      = 5                    # fixed 5-second segments
SEGMENT_SAMPLES  = WINDOW_SIZE * SR     # samples per clip at 32 kHz
SEED             = 42

# Candidate second-level shifts; 0 is excluded to avoid identical windows
POSSIBLE_SHIFTS = [-3, -2, -1, 1, 2, 3]

# Deterministic shifts used only when min_count == 1
DETERMINISTIC_SHIFTS = [1, 2, 3, 4]


# ── Label helpers ─────────────────────────────────────────────────────────────

def parse_labels(label_str) -> list:
    """Split a semicolon-separated label string into a list; handles NaN safely."""
    if pd.isna(label_str) or str(label_str).strip() == '':
        return []
    return [s.strip() for s in str(label_str).split(';') if s.strip()]


def load_species_counts(path: str) -> dict:
    """Return {species_str: count} from soundscape_species_counts.csv."""
    df = pd.read_csv(path)
    df['species'] = df['species'].astype(str)
    return dict(zip(df['species'], df['count'].astype(int)))


def load_labels(path: str) -> pd.DataFrame:
    """
    Load train_soundscapes_labels.csv.
    Uses start_sec / end_sec columns (integers) for all numeric operations.
    Falls back to parsing HH:MM:SS from start / end if the sec columns are absent.
    """
    df = pd.read_csv(path)
    if 'start_sec' not in df.columns or 'end_sec' not in df.columns:
        df['start_sec'] = pd.to_timedelta(df['start']).dt.total_seconds().astype(int)
        df['end_sec']   = pd.to_timedelta(df['end']).dt.total_seconds().astype(int)
    else:
        df['start_sec'] = df['start_sec'].astype(int)
        df['end_sec']   = df['end_sec'].astype(int)
    return df


# ── Upsampling policy ─────────────────────────────────────────────────────────

def get_min_count(labels: list, species_counts: dict) -> int:
    """
    Return the minimum segment count among all species in the label list.
    Species absent from the counts file are treated as count=1 (ultra-rare).
    """
    counts = []
    for sp in labels:
        c = species_counts.get(str(sp))
        counts.append(int(c) if c is not None else 1)
    return min(counts) if counts else 0


def get_n_copies(min_count: int) -> int:
    """
    Map the rarest-species count to the number of additional copies to create.
    Returns 0 when min_count > 100 — no upsampling needed for common species.
    """
    if min_count <= 10:
        return 5
    elif min_count <= 50:
        return 4
    elif min_count <= 100:
        return 2
    elif min_count <= 200:
        return 1
    else:
        return 0


# ── Window generation ─────────────────────────────────────────────────────────

def generate_shifted_windows(
    filename: str,
    start: int,
    end: int,
    min_count: int,
    n_copies: int,
    existing_windows: set,
    rng: random.Random,
) -> list:
    """
    Generate up to n_copies shifted 5-second windows around (start, end).

    For min_count == 1  → deterministic shifts +1, +2, +3, +4.
    For min_count >= 2  → random sample from POSSIBLE_SHIFTS without replacement.

    Windows that violate the [0, 60] boundary or already exist are skipped.
    Returns a list of (new_start, new_end) tuples.
    """
    if min_count == 1:
        candidate_shifts = DETERMINISTIC_SHIFTS
    else:
        candidate_shifts = rng.sample(POSSIBLE_SHIFTS, len(POSSIBLE_SHIFTS))

    windows = []
    for shift in candidate_shifts:
        if len(windows) >= n_copies:
            break
        new_start = start + shift
        new_end   = new_start + WINDOW_SIZE
        if new_start < 0 or new_end > AUDIO_DURATION:
            continue
        key = (filename, new_start, new_end)
        if key in existing_windows:
            continue
        windows.append((new_start, new_end))
        existing_windows.add(key)

    return windows


def assign_labels_to_window(new_start: int, new_end: int, file_df: pd.DataFrame) -> str:
    """
    Compute the union of labels from all original segments that overlap [new_start, new_end].

    Overlap condition: seg.start < new_end  AND  seg.end > new_start
    Labels are sorted alphabetically and joined with ';'.
    """
    mask = (file_df['start_sec'] < new_end) & (file_df['end_sec'] > new_start)
    union = set()
    for label_str in file_df.loc[mask, 'primary_label']:
        union.update(parse_labels(label_str))
    return ';'.join(sorted(union))


def process_file_metadata(
    filename: str,
    file_df: pd.DataFrame,
    species_counts: dict,
    existing_windows: set,
    rng: random.Random,
) -> list:
    """
    Generate upsampled segment metadata for all segments in one audio file.
    Segments whose rarest species exceeds count 100 are skipped (n_copies == 0).
    Returns a list of dicts (without audio_path — filled in during extraction).
    """
    rows = []
    for row in file_df.itertuples(index=False):
        labels = parse_labels(row.primary_label)
        if not labels:
            continue

        min_count = get_min_count(labels, species_counts)
        if min_count == 0:
            continue

        n_copies = get_n_copies(min_count)
        if n_copies == 0:
            continue   # common species — no upsampling needed

        shifted = generate_shifted_windows(
            filename, row.start_sec, row.end_sec,
            min_count, n_copies, existing_windows, rng,
        )

        for new_start, new_end in shifted:
            new_label = assign_labels_to_window(new_start, new_end, file_df)
            rows.append({
                'filename':      filename,
                'start':         new_start,
                'end':           new_end,
                'primary_label': new_label,
                'source_type':   'upsampled',
                'audio_path':    '',   # filled in after audio extraction
            })
    return rows


# ── Audio extraction ──────────────────────────────────────────────────────────

def extract_clip(waveform: np.ndarray, start_sec: int, end_sec: int) -> np.ndarray:
    """Slice [start_sec, end_sec] from waveform and zero-pad if the slice is short."""
    start_sample = start_sec * SR
    end_sample   = end_sec   * SR
    chunk = waveform[start_sample:end_sample]
    if len(chunk) < SEGMENT_SAMPLES:
        chunk = np.pad(chunk, (0, SEGMENT_SAMPLES - len(chunk)), mode='constant')
    return chunk


def save_clip(chunk: np.ndarray, out_path: str) -> None:
    """Write a mono float32 waveform as an OGG/Vorbis file."""
    sf.write(out_path, chunk, SR, format='OGG', subtype='VORBIS')


def extract_all_file_clips(filename: str, segments: list) -> dict:
    """
    Load the source soundscape once, extract every (start, end) in segments,
    save each as an OGG clip in UPSAMPLED_DIR, and return a
    {(start, end): relative_audio_path} mapping for all segments.

    Skips writing a clip if the file already exists (idempotent re-runs).
    If the source soundscape cannot be loaded, paths are still returned so
    the CSV stays consistent — the clip files just won't exist on disk.
    """
    src_path = os.path.join(SOUNDSCAPES_DIR, filename)
    waveform, _ = load_audio(src_path, max_duration=float(AUDIO_DURATION))
    stem = os.path.splitext(filename)[0]

    path_map = {}
    for start, end in segments:
        clip_name = f"{stem}_{start}_{end}.ogg"
        out_path  = os.path.join(UPSAMPLED_DIR, clip_name)

        if waveform is not None and not os.path.exists(out_path):
            chunk = extract_clip(waveform, start, end)
            save_clip(chunk, out_path)

        path_map[(start, end)] = os.path.relpath(out_path, PROJECT_ROOT).replace('\\', '/')

    return path_map


# ── Statistics ────────────────────────────────────────────────────────────────

def species_counts_from_df(df: pd.DataFrame) -> dict:
    """Count how many segments each species appears in."""
    counts: dict = {}
    for label_str in df['primary_label']:
        for sp in parse_labels(label_str):
            counts[sp] = counts.get(sp, 0) + 1
    return counts


def print_statistics(original_df: pd.DataFrame, upsampled_df: pd.DataFrame) -> None:
    total_df = pd.concat([original_df, upsampled_df], ignore_index=True)

    print(f"\n{'='*65}")
    print("UPSAMPLING STATISTICS")
    print(f"{'='*65}")
    print(f"  Original segment count  : {len(original_df):>8,}")
    print(f"  Upsampled segment count : {len(upsampled_df):>8,}")
    print(f"  Total segment count     : {len(total_df):>8,}")

    before = species_counts_from_df(original_df)
    after  = species_counts_from_df(total_df)

    print(f"\n  Species counts BEFORE upsampling (rarest first):")
    print(f"  {'Species':<22} {'Count':>7}")
    print(f"  {'-'*30}")
    for sp, cnt in sorted(before.items(), key=lambda x: x[1]):
        print(f"  {sp:<22} {cnt:>7,}")

    print(f"\n  Species counts AFTER upsampling (rarest first):")
    print(f"  {'Species':<22} {'Count':>7}")
    print(f"  {'-'*30}")
    for sp, cnt in sorted(after.items(), key=lambda x: x[1]):
        print(f"  {sp:<22} {cnt:>7,}")

    all_species = set(before) | set(after)
    deltas = [
        (sp, before.get(sp, 0), after.get(sp, 0), after.get(sp, 0) - before.get(sp, 0))
        for sp in all_species
    ]
    deltas.sort(key=lambda x: -x[3])

    print(f"\n  Top 50 species by count increase:")
    print(f"  {'Species':<22} {'Before':>8} {'After':>8} {'Increase':>10}")
    print(f"  {'-'*52}")
    for sp, b, a, inc in deltas[:50]:
        print(f"  {sp:<22} {b:>8,} {a:>8,} {inc:>10,}")
    print(f"{'='*65}\n")


def save_species_counts_report(
    original_df: pd.DataFrame,
    upsampled_df: pd.DataFrame,
    counts_csv: str,
) -> None:
    """
    Build and save a per-species count report with columns:
      species, class_name, count_before, count_after_sample, increase

    Rows are sorted by count_before ascending (rarest first).
    Saved to COUNTS_REPORT_CSV.
    """
    total_df = pd.concat([original_df, upsampled_df], ignore_index=True)

    before = species_counts_from_df(original_df)
    after  = species_counts_from_df(total_df)

    all_species = set(before) | set(after)
    rows = [
        {
            'species':            sp,
            'count_before':       before.get(sp, 0),
            'count_after_sample': after.get(sp, 0),
            'increase':           after.get(sp, 0) - before.get(sp, 0),
        }
        for sp in all_species
    ]

    report = pd.DataFrame(rows)

    # Merge in class_name from the original counts file
    counts_ref = pd.read_csv(counts_csv)[['species', 'class_name']]
    counts_ref['species'] = counts_ref['species'].astype(str)
    report['species'] = report['species'].astype(str)
    report = report.merge(counts_ref, on='species', how='left')

    # Sort rarest first, then by species name for ties
    report = report.sort_values(['count_before', 'species'], ascending=[True, True])
    report = report[['species', 'class_name', 'count_before', 'count_after_sample', 'increase']]
    report = report.reset_index(drop=True)

    report.to_csv(COUNTS_REPORT_CSV, index=False)
    print(f"Species counts report saved to:\n  {COUNTS_REPORT_CSV}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rng = random.Random(SEED)
    os.makedirs(UPSAMPLED_DIR, exist_ok=True)

    print("Loading species counts...")
    species_counts = load_species_counts(COUNTS_CSV)
    print(f"  {len(species_counts)} species loaded  "
          f"(rarest: {min(species_counts.values())}, "
          f"most common: {max(species_counts.values())})")

    print("Loading soundscape labels...")
    df = load_labels(LABELS_CSV)
    print(f"  {len(df):,} segments across {df['filename'].nunique():,} files")

    # Seed duplicate-detection with every original (filename, start, end)
    existing_windows: set = set(zip(df['filename'], df['start_sec'], df['end_sec']))

    df['source_type'] = 'original'
    df['audio_path']  = ''   # filled in per-file below

    # Group by file for fast overlap lookup and single-load audio extraction
    file_groups   = {fn: grp.reset_index(drop=True) for fn, grp in df.groupby('filename')}
    total_files   = len(file_groups)
    all_upsampled: list = []
    original_paths: dict = {}   # (filename, start, end) -> audio_path

    print(f"Generating upsampled segments and extracting audio ({total_files} files)...")
    for i, (filename, file_df) in enumerate(file_groups.items(), start=1):
        if i % 20 == 0 or i == total_files:
            print(f"  [{i:>4}/{total_files}] {filename}")

        # Step 1: generate upsampled metadata for this file
        upsampled_rows = process_file_metadata(
            filename, file_df, species_counts, existing_windows, rng
        )

        # Step 2: collect all segments to extract — originals + upsampled
        orig_segments  = [(r.start_sec, r.end_sec) for r in file_df.itertuples(index=False)]
        upsamp_segments = [(r['start'], r['end']) for r in upsampled_rows]
        all_segments   = list(dict.fromkeys(orig_segments + upsamp_segments))  # dedup, order-stable

        # Step 3: load soundscape once, extract and save all clips, get path map
        path_map = extract_all_file_clips(filename, all_segments)

        # Step 4: store original paths for later assignment
        for start, end in orig_segments:
            original_paths[(filename, start, end)] = path_map[(start, end)]

        # Step 5: assign audio_path to upsampled rows
        for row in upsampled_rows:
            row['audio_path'] = path_map[(row['start'], row['end'])]

        all_upsampled.extend(upsampled_rows)

    print(f"  Generated {len(all_upsampled):,} upsampled segments")

    # Build output DataFrames with consistent columns
    original_out = df[['filename', 'start_sec', 'end_sec', 'primary_label',
                        'source_type', 'audio_path']].copy()
    original_out = original_out.rename(columns={'start_sec': 'start', 'end_sec': 'end'})

    # Apply extracted clip paths to original rows
    original_out['audio_path'] = original_out.apply(
        lambda r: original_paths.get((r['filename'], r['start'], r['end']), ''), axis=1
    )

    if all_upsampled:
        upsampled_out = pd.DataFrame(all_upsampled)
    else:
        upsampled_out = pd.DataFrame(
            columns=['filename', 'start', 'end', 'primary_label', 'source_type', 'audio_path']
        )

    result = pd.concat([original_out, upsampled_out], ignore_index=True)

    print(f"Saving {len(result):,} total segments to:\n  {OUTPUT_CSV}")
    result.to_csv(OUTPUT_CSV, index=False)

    print_statistics(original_out, upsampled_out)
    save_species_counts_report(original_out, upsampled_out, COUNTS_CSV)


if __name__ == '__main__':
    main()

"""
create_species_folders.py

Organises upsampled audio segments into per-species subfolders under
  data/train_upsampled_species/{species}/

Two phases — index CSV records are written in this order:
  Phase 1 — train_audio_upsampled  (single-label, already by species)
  Phase 2 — train_soundscapes_upsampled  (multi-label, greedy rarest-first assignment)

Phase 2 assignment strategy:
  1. Sort species by count_after_sample ASC (rarest gets first claim).
  2. Each species claims all segments not yet taken by a rarer species.
  3. Segments still unclaimed after all species are processed go to the
     most common (last in priority) matching species in their label.

Input:
  data/train_upsampled.csv
  data/train_audio_upsampled/{species}/*.ogg
  visualization/soundscape_species_counts_upsampled.csv
  data/train_soundscapes_labels_upsampled.csv

Output:
  data/train_upsampled_species/{species}/*.ogg
  data/train_upsampled_species/species_segment_index.csv
    Columns: source_type, assigned_species, dest_path, source_path, original_primary_label
"""

import os
import shutil
import pandas as pd

PROJECT_ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

TRAIN_AUDIO_CSV    = os.path.join(PROJECT_ROOT, 'data', 'train_upsampled.csv')
TRAIN_AUDIO_DIR    = os.path.join(PROJECT_ROOT, 'data', 'train_audio_upsampled')
COUNTS_CSV         = os.path.join(PROJECT_ROOT, 'visualization', 'soundscape_species_counts_upsampled.csv')
LABELS_CSV         = os.path.join(PROJECT_ROOT, 'data', 'train_soundscapes_labels_upsampled.csv')
OUTPUT_DIR         = os.path.join(PROJECT_ROOT, 'data', 'train_upsampled_species')
INDEX_CSV          = os.path.join(OUTPUT_DIR, 'species_segment_index.csv')


def parse_labels(label_str) -> list:
    if pd.isna(label_str) or str(label_str).strip() == '':
        return []
    return [s.strip() for s in str(label_str).split(';') if s.strip()]


# ── Phase 1: train_audio_upsampled ────────────────────────────────────────────

def phase1_train_audio() -> tuple[list, int]:
    """
    Copy files from train_audio_upsampled into train_upsampled_species/{species}/
    using train_upsampled.csv as the manifest.

    Returns (index_rows, missing_count).
    """
    train_df = pd.read_csv(TRAIN_AUDIO_CSV)
    index_rows = []
    missing_files = 0

    total = len(train_df)
    print(f"Phase 1 — train_audio: {total} segments across "
          f"{train_df['primary_label'].nunique()} species")

    for idx, row in enumerate(train_df.itertuples(index=False), start=1):
        sp       = str(row.primary_label)
        filename = str(row.filename)   # e.g. "grekis/XC47777_seg0.ogg"

        sp_dir = os.path.join(OUTPUT_DIR, sp)
        os.makedirs(sp_dir, exist_ok=True)

        src = os.path.join(TRAIN_AUDIO_DIR, filename.replace('/', os.sep))
        dst = os.path.join(sp_dir, os.path.basename(src))

        if os.path.exists(src):
            shutil.copy2(src, dst)
        else:
            missing_files += 1
            print(f"  WARNING: source not found: {src}")

        index_rows.append({
            'source_type':            'train_audio',
            'assigned_species':       sp,
            'dest_path':              os.path.relpath(dst, PROJECT_ROOT).replace('\\', '/'),
            'source_path':            os.path.join('data', 'train_audio_upsampled',
                                                   filename).replace('\\', '/'),
            'original_primary_label': sp,
        })

        if idx % 500 == 0 or idx == total:
            print(f"  [{idx:>5}/{total}] copied")

    print(f"Phase 1 done — {len(index_rows)} records, {missing_files} missing files\n")
    return index_rows, missing_files


# ── Phase 2: train_soundscapes_upsampled (greedy rarest-first) ────────────────

def phase2_soundscapes() -> tuple[list, int]:
    """
    Assign soundscape segments to species folders (greedy, rarest first),
    copy files, and return (index_rows, missing_count).
    """
    # Load species priority order
    counts_df = pd.read_csv(COUNTS_CSV)
    counts_df['species'] = counts_df['species'].astype(str)
    counts_df = counts_df.sort_values('count_after_sample', ascending=True)
    priority_list = counts_df['species'].tolist()
    priority_rank = {sp: i for i, sp in enumerate(priority_list)}  # 0 = rarest

    print(f"Phase 2 — soundscapes: {len(priority_list)} species "
          f"(rarest: {priority_list[0]}, most common: {priority_list[-1]})")

    # Load labels and build indices
    labels_df = pd.read_csv(LABELS_CSV)
    labels_df['audio_path'] = labels_df['audio_path'].astype(str)

    path_to_label = dict(zip(labels_df['audio_path'], labels_df['primary_label']))

    species_to_segments: dict[str, list] = {sp: [] for sp in priority_list}
    for _, row in labels_df.iterrows():
        for sp in parse_labels(row['primary_label']):
            if sp in species_to_segments:
                species_to_segments[sp].append(row['audio_path'])

    # Greedy assignment
    claimed: set = set()
    assignment: dict[str, list] = {sp: [] for sp in priority_list}

    for sp in priority_list:
        unclaimed = [seg for seg in species_to_segments[sp] if seg not in claimed]
        assignment[sp] = unclaimed
        claimed.update(unclaimed)

    # Handle unassigned segments → most common matching species
    unassigned = set(labels_df['audio_path'].tolist()) - claimed
    if unassigned:
        print(f"  {len(unassigned)} unassigned segments → assigning to most common matching species")
        for seg_path in unassigned:
            species_in_seg = parse_labels(path_to_label.get(seg_path, ''))
            ranked = [(priority_rank[sp], sp) for sp in species_in_seg if sp in priority_rank]
            if ranked:
                _, last_sp = max(ranked)
                assignment[last_sp].append(seg_path)
    else:
        print("  All segments assigned — no unassigned remainder.")

    # Create folders, copy files, collect rows
    index_rows = []
    missing_files = 0
    total_species = len(priority_list)

    for idx, sp in enumerate(priority_list, start=1):
        sp_dir = os.path.join(OUTPUT_DIR, sp)
        os.makedirs(sp_dir, exist_ok=True)

        segs = assignment[sp]
        if idx % 10 == 0 or idx == total_species:
            print(f"  [{idx:>3}/{total_species}] {sp:<20} — {len(segs)} segments")

        for seg_path in segs:
            src = os.path.join(PROJECT_ROOT, seg_path.replace('/', os.sep))
            dst = os.path.join(sp_dir, os.path.basename(src))

            if os.path.exists(src):
                shutil.copy2(src, dst)
            else:
                missing_files += 1
                print(f"  WARNING: source not found: {src}")

            index_rows.append({
                'source_type':            'soundscape',
                'assigned_species':       sp,
                'dest_path':              os.path.relpath(dst, PROJECT_ROOT).replace('\\', '/'),
                'source_path':            seg_path,
                'original_primary_label': path_to_label.get(seg_path, ''),
            })

    print(f"Phase 2 done — {len(index_rows)} records, {missing_files} missing files\n")
    return index_rows, missing_files


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    p1_rows, p1_missing = phase1_train_audio()
    p2_rows, p2_missing = phase2_soundscapes()

    # Phase 1 (train_audio) records first, then Phase 2 (soundscape)
    all_rows = p1_rows + p2_rows
    index_df = pd.DataFrame(all_rows, columns=[
        'source_type', 'assigned_species', 'dest_path',
        'source_path', 'original_primary_label',
    ])
    index_df.to_csv(INDEX_CSV, index=False)

    print(f"{'='*60}")
    print("DONE")
    print(f"  Phase 1 (train_audio) records : {len(p1_rows)}")
    print(f"  Phase 2 (soundscape)  records : {len(p2_rows)}")
    print(f"  Total records                 : {len(all_rows)}")
    print(f"  Total missing source files    : {p1_missing + p2_missing}")
    print(f"  Index CSV saved to            : {INDEX_CSV}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

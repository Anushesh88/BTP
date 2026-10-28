"""
Phase 2: Leak-Free Grouped Splitting Strategy
IMPORTANT:
Never use random train_test_split. Miniatures from the same manuscript set, series, or album
share distinct paletting, paper tone, border decoration, and artist hands.
Random splitting causes severe data leakage and artificially inflated metrics.

- 2.1 Identify Grouping Keys:
  Group by manuscript series, collection, or album ID extracted from sources.txt and caption provenance.
- 2.2 Implement Grouped Stratified Partition:
  - 70% Train, 15% Validation, 15% Test.
  - Ensure strictly disjoint groups: paintings from the same manuscript folio/album/series never appear across splits.
  - Verify label distributions across splits for each task (category, school, period, artist, raga).
  - Export frozen split manifests:
      data/metadata/train_split.jsonl
      data/metadata/val_split.jsonl
      data/metadata/test_split.jsonl
"""

import json
import random
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd

def perform_grouped_split(data_dir: str = "data", seed: int = 42, refine_groups: bool = True,
                          use_image_similarity: bool = True):
    data_path = Path(data_dir)
    metadata_dir = data_path / "metadata"
    manifest_path = metadata_dir / "dataset_manifest.parquet"
    
    if not manifest_path.exists():
        manifest_path = metadata_dir / "dataset_manifest.csv"
        df = pd.read_csv(manifest_path)
    else:
        df = pd.read_parquet(manifest_path)

    # 2.1 refinement: series named in titles/captions, single-source collections,
    # near-duplicate images, exact duplicates (see src/group_refinement.py)
    if refine_groups:
        from src.group_refinement import compute_all_embeddings, refine_group_keys
        embeddings = compute_all_embeddings(df) if use_image_similarity else None
        df, refinement_report = refine_group_keys(df, embeddings)
        print(f"[Split] Group refinement: {refinement_report['groups_before']} -> {refinement_report['groups_after']} groups, "
              f"{refinement_report['paintings_regrouped']} paintings regrouped, "
              f"{len(refinement_report['exact_duplicates_removed'])} exact duplicate(s) removed")
        with open(metadata_dir / "group_refinement_report.json", "w", encoding="utf-8") as f:
            json.dump(refinement_report, f, indent=2)

    records = df.to_dict(orient="records")
    total_samples = len(records)
    print(f"[Split] Total records to partition: {total_samples}")

    # Group records by group_key
    groups = defaultdict(list)
    for r in records:
        groups[r['group_key']].append(r)

    print(f"[Split] Total distinct disjoint groups: {len(groups)}")

    # Target counts: 70% Train, 15% Val, 15% Test
    target_train = round(0.70 * total_samples)
    target_val = round(0.15 * total_samples)
    target_test = total_samples - target_train - target_val
    print(f"[Split] Target split counts: Train={target_train} (70%), Val={target_val} (15%), Test={target_test} (15%)")

    # Multi-objective optimization for optimal grouped stratified partition
    # Groups larger than the val/test targets (e.g. the 153-folio Tuti-nama, the 116 detail
    # studies) cannot fit in val or test, so they are assigned to Train
    fixed_train = sorted(g for g in groups if len(groups[g]) > target_val)
    print(f"[Split] Groups too large for val/test, fixed to Train: {[(g, len(groups[g])) for g in fixed_train]}")
    other_groups = [g for g in groups if g not in fixed_train]

    minority_label_tests = [
        lambda p: p['school'] == 'Rajasthani',
        lambda p: str(p['raga']) not in ('None', 'nan', ''),
        lambda p: p['category'] == 'Ragamala Paintings',
        # Periods outside the two large fixed-to-Train groups (Akbar-era Tuti-nama, detail studies)
        lambda p: p['period'] in ('Late Mughal / 18th Century', '18th Century'),
        lambda p: p['period'] == 'Shah Jahan Period (c. 1628-1658)',
        lambda p: p['period'] == 'Aurangzeb Period (c. 1658-1707)',
        lambda p: p['period'] == 'Jahangir Period (c. 1605-1627)',
    ]

    random.seed(seed)
    best_score = float('inf')
    best_splits = None

    for trial in range(10000):
        random.shuffle(other_groups)
        train_g = list(fixed_train)
        val_g = []
        test_g = []
        
        tr_cnt = sum(len(groups[g]) for g in train_g)
        va_cnt = 0
        te_cnt = 0
        
        for g in other_groups:
            g_len = len(groups[g])
            def_tr = max(0, target_train - tr_cnt)
            def_va = max(0, target_val - va_cnt)
            def_te = max(0, target_test - te_cnt)
            
            choices = []
            if def_tr > 0: choices.append(('tr', def_tr))
            if def_va > 0: choices.append(('va', def_va * 1.25))
            if def_te > 0: choices.append(('te', def_te * 1.25))
            
            if not choices:
                choices = [('tr', 1)]
                
            chosen = max(choices, key=lambda x: x[1] * random.uniform(0.75, 1.25))[0]
            if chosen == 'tr':
                train_g.append(g)
                tr_cnt += g_len
            elif chosen == 'va':
                val_g.append(g)
                va_cnt += g_len
            else:
                test_g.append(g)
                te_cnt += g_len
                
        tr_items = [p for g in train_g for p in groups[g]]
        va_items = [p for g in val_g for p in groups[g]]
        te_items = [p for g in test_g for p in groups[g]]
        
        # Penalties:
        # 1. Size deviation
        sz_pen = (len(tr_items) - target_train)**2 + (len(va_items) - target_val)**2 + (len(te_items) - target_test)**2
        
        # 2. Minority-label balance: each split should hold its share of the rare labels the
        # model must learn (Rajasthani school, Ragamala paintings). Crop balance is not targeted:
        # the 116 single-source detail studies are one group and must sit in Train.
        cat_pen = 0
        for share_in in minority_label_tests:
            ideal = sum(1 for p in records if share_in(p)) / total_samples
            for items in (tr_items, va_items, te_items):
                cat_pen += (sum(1 for p in items if share_in(p)) / max(1, len(items)) - ideal) ** 2

        # 3. School presence
        school_pen = 0
        for sch in ['Mughal', 'Rajasthani']:
            s_tr = sum(1 for p in tr_items if p['school'] == sch)
            s_va = sum(1 for p in va_items if p['school'] == sch)
            s_te = sum(1 for p in te_items if p['school'] == sch)
            if s_tr == 0 or s_va == 0 or s_te == 0:
                school_pen += 1000

        # 4. Period presence (ensure major periods in all splits)
        period_pen = 0
        for per in ['Akbar Period (c. 1556-1605)', 'Jahangir Period (c. 1605-1627)', 'Classical Mughal (16th-17th Century)']:
            p_tr = sum(1 for p in tr_items if p['period'] == per)
            p_va = sum(1 for p in va_items if p['period'] == per)
            p_te = sum(1 for p in te_items if p['period'] == per)
            if p_tr == 0 or p_va == 0 or p_te == 0:
                period_pen += 500

        total_penalty = sz_pen + cat_pen * 10000 + school_pen + period_pen
        if total_penalty < best_score:
            best_score = total_penalty
            best_splits = (train_g, val_g, test_g, tr_items, va_items, te_items)

    train_groups, val_groups, test_groups, train_items, val_items, test_items = best_splits

    # Verification of Disjoint Groups
    overlap_tr_va = set(train_groups).intersection(set(val_groups))
    overlap_tr_te = set(train_groups).intersection(set(test_groups))
    overlap_va_te = set(val_groups).intersection(set(test_groups))

    assert len(overlap_tr_va) == 0, f"LEAKAGE DETECTED between Train and Val: {overlap_tr_va}"
    assert len(overlap_tr_te) == 0, f"LEAKAGE DETECTED between Train and Test: {overlap_tr_te}"
    assert len(overlap_va_te) == 0, f"LEAKAGE DETECTED between Val and Test: {overlap_va_te}"

    print(f"\n[Split] Group Disjointness Assertion: PASSED (0% group overlap across splits)")
    print(f"[Split] Partitioned counts:")
    print(f"  - Train: {len(train_items)} images ({len(train_items)/total_samples*100:.2f}%) across {len(train_groups)} groups")
    print(f"  - Val:   {len(val_items)} images ({len(val_items)/total_samples*100:.2f}%) across {len(val_groups)} groups")
    print(f"  - Test:  {len(test_items)} images ({len(test_items)/total_samples*100:.2f}%) across {len(test_groups)} groups")

    # Add 'split' field to items
    for item in train_items: item['split'] = 'train'
    for item in val_items: item['split'] = 'val'
    for item in test_items: item['split'] = 'test'

    # Export frozen split manifests
    train_file = metadata_dir / "train_split.jsonl"
    val_file = metadata_dir / "val_split.jsonl"
    test_file = metadata_dir / "test_split.jsonl"

    for file_path, items in [(train_file, train_items), (val_file, val_items), (test_file, test_items)]:
        with open(file_path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"[Split] Exported frozen manifest: {file_path} ({len(items)} items)")

    # Generate split verification statistics dictionary
    stats = {
        "summary": {
            "total_samples": total_samples,
            "train_samples": len(train_items),
            "train_ratio": round(len(train_items) / total_samples, 4),
            "val_samples": len(val_items),
            "val_ratio": round(len(val_items) / total_samples, 4),
            "test_samples": len(test_items),
            "test_ratio": round(len(test_items) / total_samples, 4),
            "train_groups": len(train_groups),
            "val_groups": len(val_groups),
            "test_groups": len(test_groups),
            "group_leakage": False
        },
        "distributions": {
            "category": {
                "train": dict(Counter(p['category'] for p in train_items)),
                "val": dict(Counter(p['category'] for p in val_items)),
                "test": dict(Counter(p['category'] for p in test_items))
            },
            "school": {
                "train": dict(Counter(p['school'] for p in train_items)),
                "val": dict(Counter(p['school'] for p in val_items)),
                "test": dict(Counter(p['school'] for p in test_items))
            },
            "period": {
                "train": dict(Counter(p['period'] for p in train_items)),
                "val": dict(Counter(p['period'] for p in val_items)),
                "test": dict(Counter(p['period'] for p in test_items))
            },
            "artist": {
                "train": dict(Counter(p['artist'] for p in train_items)),
                "val": dict(Counter(p['artist'] for p in val_items)),
                "test": dict(Counter(p['artist'] for p in test_items))
            },
            "raga": {
                "train": dict(Counter(p['raga'] for p in train_items)),
                "val": dict(Counter(p['raga'] for p in val_items)),
                "test": dict(Counter(p['raga'] for p in test_items))
            }
        }
    }

    stats_file = metadata_dir / "split_verification_report.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[Split] Split verification report saved to: {stats_file}")

    return stats

if __name__ == "__main__":
    perform_grouped_split()


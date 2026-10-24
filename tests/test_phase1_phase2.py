"""
Comprehensive Verification Tests for Phase 1 & Phase 2
"""

import json
from pathlib import Path
import pandas as pd

def test_directories_exist():
    base = Path("data")
    assert (base / "raw" / "images").is_dir(), "Images directory missing"
    assert (base / "raw" / "captions").is_dir(), "Captions directory missing"
    assert (base / "metadata").is_dir(), "Metadata directory missing"
    print("[OK] Directory structure verified")

def test_file_counts():
    img_dir = Path("data/raw/images")
    cap_dir = Path("data/raw/captions")
    imgs = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.jpeg")) + list(img_dir.glob("*.png"))
    caps = list(cap_dir.glob("*.txt"))
    assert len(imgs) == 564, f"Expected 564 images, got {len(imgs)}"
    assert len(caps) == 564, f"Expected 564 captions, got {len(caps)}"
    print(f"[OK] File counts verified: {len(imgs)} images, {len(caps)} captions")

def test_manifest_integrity():
    meta_dir = Path("data/metadata")
    pq_path = meta_dir / "dataset_manifest.parquet"
    csv_path = meta_dir / "dataset_manifest.csv"
    assert pq_path.exists(), "Parquet manifest missing"
    assert csv_path.exists(), "CSV manifest missing"
    
    df_pq = pd.read_parquet(pq_path)
    df_csv = pd.read_csv(csv_path)
    assert len(df_pq) == 564, f"Expected 564 rows in parquet, got {len(df_pq)}"
    assert len(df_csv) == 564, f"Expected 564 rows in csv, got {len(df_csv)}"

    required_cols = [
        'image_id', 'file_path', 'category', 'raga', 'school', 
        'period', 'artist', 'collection_id', 'caption', 'has_box_labels'
    ]
    for col in required_cols:
        assert col in df_pq.columns, f"Missing column {col} in manifest"
        
    # Check 3 categories
    cats = df_pq['category'].value_counts().to_dict()
    assert "Mughal Miniature Paintings" in cats and cats["Mughal Miniature Paintings"] > 0
    assert "Painting Crops" in cats and cats["Painting Crops"] > 0
    assert "Ragamala Paintings" in cats and cats["Ragamala Paintings"] > 0
    print(f"[OK] Manifest columns and 3 categories verified: {cats}")

def test_leak_free_disjoint_splits():
    meta_dir = Path("data/metadata")
    train_path = meta_dir / "train_split.jsonl"
    val_path = meta_dir / "val_split.jsonl"
    test_path = meta_dir / "test_split.jsonl"
    
    assert train_path.exists() and val_path.exists() and test_path.exists(), "Split files missing"

    with open(train_path, "r", encoding="utf-8") as f:
        train_data = [json.loads(line) for line in f]
    with open(val_path, "r", encoding="utf-8") as f:
        val_data = [json.loads(line) for line in f]
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f]

    assert len(train_data) + len(val_data) + len(test_data) == 564, "Total split counts do not equal 564"
    assert abs(len(train_data) / 564 - 0.70) < 0.02, f"Train split not ~70%: {len(train_data)}"
    assert abs(len(val_data) / 564 - 0.15) < 0.02, f"Val split not ~15%: {len(val_data)}"
    assert abs(len(test_data) / 564 - 0.15) < 0.02, f"Test split not ~15%: {len(test_data)}"

    # Check for ZERO duplicate images
    train_ids = set(x['image_id'] for x in train_data)
    val_ids = set(x['image_id'] for x in val_data)
    test_ids = set(x['image_id'] for x in test_data)

    assert len(train_ids.intersection(val_ids)) == 0, "Duplicate images in train and val"
    assert len(train_ids.intersection(test_ids)) == 0, "Duplicate images in train and test"
    assert len(val_ids.intersection(test_ids)) == 0, "Duplicate images in val and test"

    # Check for ZERO group leakage
    train_groups = set(x['group_key'] for x in train_data)
    val_groups = set(x['group_key'] for x in val_data)
    test_groups = set(x['group_key'] for x in test_data)

    assert len(train_groups.intersection(val_groups)) == 0, "Group leakage detected between Train and Val"
    assert len(train_groups.intersection(test_groups)) == 0, "Group leakage detected between Train and Test"
    assert len(val_groups.intersection(test_groups)) == 0, "Group leakage detected between Val and Test"

    print(f"[OK] Split counts verified: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    print("[OK] Leak-Free Disjoint Groups Assertion: PASSED (0% group overlap)")

if __name__ == "__main__":
    test_directories_exist()
    test_file_counts()
    test_manifest_integrity()
    test_leak_free_disjoint_splits()
    print("\nALL 4 VERIFICATION TEST SUITES PASSED SUCCESSFULLY!")


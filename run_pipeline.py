"""
Pipeline Runner for Phase 1 and Phase 2
Checklist:
Phase 1: Data Ingestion, Extraction & Metadata Parsing
  1.1 Download and Stage Dataset
  1.2 Metadata Ingestion & Alignment
  1.3 Data Integrity & Validation
Phase 2: Leak-Free Grouped Splitting Strategy
  2.1 Identify Grouping Keys
  2.2 Implement Grouped Stratified Partition
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.stage_dataset import stage_dataset
from src.metadata_parser import parse_metadata_and_captions
from src.data_validation import validate_dataset
from src.grouped_splitting import perform_grouped_split

def main():
    print("=" * 70)
    print("STARTING PHASE 1 & PHASE 2 PIPELINE")
    print("=" * 70)
    
    print("\n>>> STEP 1.1: Staging Dataset...")
    already_staged = any(Path("data/raw/images").glob("*")) and Path("data/metadata/metadata.jsonl").exists()
    if not Path("Dataset").exists() and already_staged:
        print("[Stage] 'Dataset/' not present but data/raw and data/metadata are already staged. Skipping copy.")
    else:
        stage_info = stage_dataset(source_dir="Dataset", target_base_dir="data")
    
    print("\n>>> STEP 1.2: Metadata Ingestion & Alignment...")
    manifest_df = parse_metadata_and_captions(data_dir="data")
    
    print("\n>>> STEP 1.3: Data Integrity & Validation...")
    val_report = validate_dataset(data_dir="data")
    
    print("\n>>> STEP 2.1 & 2.2: Leak-Free Grouped Splitting Strategy...")
    split_report = perform_grouped_split(data_dir="data", seed=42)
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("Generated Artifacts:")
    print("  1. Manifests:")
    print("     - data/metadata/dataset_manifest.parquet")
    print("     - data/metadata/dataset_manifest.csv")
    print("  2. Quality & Validation Report:")
    print("     - data/metadata/data_validation_report.json")
    print("  3. Frozen Leak-Free Split Manifests:")
    print("     - data/metadata/train_split.jsonl")
    print("     - data/metadata/val_split.jsonl")
    print("     - data/metadata/test_split.jsonl")
    print("  4. Split Verification Report:")
    print("     - data/metadata/split_verification_report.json")
    print("=" * 70)

if __name__ == "__main__":
    main()


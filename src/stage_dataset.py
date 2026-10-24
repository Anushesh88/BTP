"""
Phase 1.1: Download & Stage Dataset
- Organize local directory structure:
    data/raw/images
    data/raw/captions
    data/metadata
- Copy and stage files from downloaded dataset archive
- Verify image presence across the 3 categories:
    1. Mughal Miniature Paintings
    2. Painting Crops
    3. Ragamala Paintings
"""

import os
import shutil
from pathlib import Path
import json

def stage_dataset(source_dir: str = "Dataset", target_base_dir: str = "data"):
    source_path = Path(source_dir)
    target_path = Path(target_base_dir)
    
    images_dir = target_path / "raw" / "images"
    captions_dir = target_path / "raw" / "captions"
    metadata_dir = target_path / "metadata"
    
    # Create directories
    for d in [images_dir, captions_dir, metadata_dir]:
        d.mkdir(parents=True, exist_ok=True)
        print(f"[Stage] Ensured directory: {d}")

    # Identify image folder in source
    img_src_dir = source_path / "mughal_final_dataset_new"
    if not img_src_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {img_src_dir}")

    # Copy metadata and sources
    meta_src = source_path / "metadata.jsonl"
    sources_src = source_path / "sources.txt"
    
    if meta_src.exists():
        shutil.copy2(meta_src, metadata_dir / "metadata.jsonl")
        print(f"[Stage] Copied {meta_src.name} -> {metadata_dir}")
    if sources_src.exists():
        shutil.copy2(sources_src, metadata_dir / "sources.txt")
        print(f"[Stage] Copied {sources_src.name} -> {metadata_dir}")

    # Copy images and caption txt files
    all_files = list(img_src_dir.iterdir())
    img_copied = 0
    cap_copied = 0
    
    for f in all_files:
        ext = f.suffix.lower()
        if ext in ['.jpg', '.jpeg', '.png']:
            dest = images_dir / f.name
            if not dest.exists() or dest.stat().st_size != f.stat().st_size:
                shutil.copy2(f, dest)
            img_copied += 1
        elif ext == '.txt':
            dest = captions_dir / f.name
            if not dest.exists() or dest.stat().st_size != f.stat().st_size:
                shutil.copy2(f, dest)
            cap_copied += 1

    print(f"[Stage] Total staged images: {img_copied}")
    print(f"[Stage] Total staged captions: {cap_copied}")

    # Verify category presence
    with open(metadata_dir / "metadata.jsonl", "r", encoding="utf-8") as f:
        meta_entries = [json.loads(line) for line in f]

    cats = {"Mughal Miniature Paintings": 0, "Painting Crops": 0, "Ragamala Paintings": 0}
    for e in meta_entries:
        fn = e['file_name']
        cap = e['caption']
        is_crop = (
            fn.startswith('mughal_style_') or 
            fn.startswith('mughalstyle_') or 
            'crop' in fn.lower() or 
            'cropped' in fn.lower()
        )
        is_rgmla = cap.startswith('RGMLA') or 'ragamala' in fn.lower() or 'ragini' in fn.lower() or 'raga' in fn.lower()
        
        if is_crop:
            cats["Painting Crops"] += 1
        elif is_rgmla:
            cats["Ragamala Paintings"] += 1
        else:
            cats["Mughal Miniature Paintings"] += 1

    print("[Stage] Category Verification:")
    for c, cnt in cats.items():
        print(f"  - {c}: {cnt} images present")
        assert cnt > 0, f"Category {c} has 0 images!"

    return {
        "images_dir": str(images_dir),
        "captions_dir": str(captions_dir),
        "metadata_dir": str(metadata_dir),
        "image_count": img_copied,
        "caption_count": cap_copied,
        "category_counts": cats
    }

if __name__ == "__main__":
    stage_dataset()


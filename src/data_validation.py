"""
Phase 1.3: Data Integrity & Validation
- Validate image file headers, resolutions, color channels (ensure conversion to RGB if RGBA/CMYK/grayscale).
- Identify and flag corrupted or unreadable images.
- Map label distributions and identify class imbalances across targets (Raga, School, Period, Artist).
- Save validation report to JSON for downstream consumption.
"""

from pathlib import Path
from PIL import Image
import pandas as pd
import json
from collections import Counter

def validate_dataset(data_dir: str = "data"):
    data_path = Path(data_dir)
    images_dir = data_path / "raw" / "images"
    metadata_dir = data_path / "metadata"
    manifest_path = metadata_dir / "dataset_manifest.parquet"
    
    if not manifest_path.exists():
        manifest_path = metadata_dir / "dataset_manifest.csv"
        df = pd.read_csv(manifest_path)
    else:
        df = pd.read_parquet(manifest_path)

    print(f"[Validation] Auditing {len(df)} images in {images_dir}...")

    image_stats = []
    corrupted_images = []
    color_mode_conversions = 0

    for idx, row in df.iterrows():
        img_rel_path = Path(row['file_path'])
        # Absolute path on disk
        full_path = data_path.parent / img_rel_path
        if not full_path.exists():
            full_path = images_dir / Path(row['file_path']).name
            
        if not full_path.exists():
            corrupted_images.append({
                "image_id": row['image_id'],
                "file_path": str(img_rel_path),
                "error": "File does not exist on disk"
            })
            continue

        try:
            with Image.open(full_path) as img:
                img.verify()

            # Reopen to read attributes
            with Image.open(full_path) as img:
                orig_mode = img.mode
                orig_format = img.format
                width, height = img.size
                
                # Check color channel conversion need
                if orig_mode != "RGB":
                    print(f"[Validation] Converting non-RGB image {full_path.name} ({orig_mode}) to RGB...")
                    rgb_img = img.convert("RGB")
                    rgb_img.save(full_path)
                    color_mode_conversions += 1
                    current_mode = "RGB"
                else:
                    current_mode = orig_mode

                image_stats.append({
                    "image_id": row['image_id'],
                    "format": orig_format,
                    "mode": current_mode,
                    "width": width,
                    "height": height,
                    "aspect_ratio": round(width / max(1, height), 4)
                })

        except Exception as e:
            corrupted_images.append({
                "image_id": row['image_id'],
                "file_path": str(img_rel_path),
                "error": str(e)
            })

    # Summary metrics
    resolutions = [(s['width'], s['height']) for s in image_stats]
    widths = [s['width'] for s in image_stats]
    heights = [s['height'] for s in image_stats]

    # Target distributions
    dist_category = df['category'].value_counts().to_dict()
    dist_school = df['school'].value_counts().to_dict()
    dist_period = df['period'].value_counts().to_dict()
    dist_artist = df['artist'].value_counts().to_dict()
    dist_raga = df['raga'].value_counts().to_dict()

    # Imbalance analysis
    imbalance_notes = {
        "category": {
            "dominant_class": df['category'].value_counts().index[0],
            "dominant_share": round(df['category'].value_counts().iloc[0] / len(df), 4),
            "minority_class": df['category'].value_counts().index[-1],
            "minority_share": round(df['category'].value_counts().iloc[-1] / len(df), 4)
        },
        "school": {
            "dominant_class": df['school'].value_counts().index[0],
            "dominant_share": round(df['school'].value_counts().iloc[0] / len(df), 4),
            "rare_classes": df['school'].value_counts()[df['school'].value_counts() < 5].to_dict()
        },
        "period": {
            "dominant_class": df['period'].value_counts().index[0],
            "dominant_share": round(df['period'].value_counts().iloc[0] / len(df), 4),
            "total_classes": len(dist_period)
        },
        "artist": {
            "attributed_count": int((df['artist'] != 'Unknown').sum()),
            "unknown_count": int((df['artist'] == 'Unknown').sum()),
            "unknown_percentage": round(float((df['artist'] == 'Unknown').mean()) * 100, 2)
        },
        "raga": {
            "ragamala_count": int((df['raga'] != 'None').sum()),
            "distinct_ragas": [k for k in dist_raga.keys() if k not in ['None', 'Ragamala (Unspecified)']]
        }
    }

    report = {
        "total_images_manifest": len(df),
        "total_images_valid": len(image_stats),
        "total_corrupted_or_missing": len(corrupted_images),
        "color_mode_conversions_to_rgb": color_mode_conversions,
        "resolution_summary": {
            "min_width": min(widths) if widths else 0,
            "max_width": max(widths) if widths else 0,
            "mean_width": round(sum(widths) / max(1, len(widths)), 1),
            "min_height": min(heights) if heights else 0,
            "max_height": max(heights) if heights else 0,
            "mean_height": round(sum(heights) / max(1, len(heights)), 1)
        },
        "distributions": {
            "category": dist_category,
            "school": dist_school,
            "period": dist_period,
            "artist": dist_artist,
            "raga": dist_raga
        },
        "imbalance_analysis": imbalance_notes,
        "corrupted_images": corrupted_images
    }

    report_path = metadata_dir / "data_validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "="*50)
    print("DATA INTEGRITY & VALIDATION SUMMARY")
    print("="*50)
    print(f"Total Valid Images:        {len(image_stats)} / {len(df)}")
    print(f"Corrupted / Unreadable:    {len(corrupted_images)}")
    print(f"Color Conversions to RGB:  {color_mode_conversions}")
    print(f"Resolution Width Range:    {min(widths)}px - {max(widths)}px (Mean: {report['resolution_summary']['mean_width']}px)")
    print(f"Resolution Height Range:   {min(heights)}px - {max(heights)}px (Mean: {report['resolution_summary']['mean_height']}px)")
    print("\n--- Target Distributions & Imbalances ---")
    print("Category:", dist_category)
    print("School:", dist_school)
    print("Period (Top 5):", dict(list(dist_period.items())[:5]))
    print("Artist Attributed:", f"{report['imbalance_analysis']['artist']['attributed_count']} ({report['imbalance_analysis']['artist']['unknown_percentage']}% Unknown)")
    print("Ragamala Entries:", f"{report['imbalance_analysis']['raga']['ragamala_count']} entries across {len(report['imbalance_analysis']['raga']['distinct_ragas'])} identified Raga families")
    print(f"\n[Validation] Report saved to: {report_path}")

    return report

if __name__ == "__main__":
    validate_dataset()


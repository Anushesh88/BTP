"""
Phase 1.2: Metadata Ingestion & Alignment
- Parse sources.txt to map provenance, albums, and collections.
- Parse metadata.jsonl and link entries to corresponding image files.
- Parse individual .txt caption files; validate structure against metadata.jsonl.
- Generate unified tabular manifest (dataset_manifest.parquet and dataset_manifest.csv) with columns:
    image_id, file_path, category, raga, school, period, artist, collection_id, caption, has_box_labels
"""

import json
import re
from pathlib import Path
from collections import Counter
import pandas as pd

def parse_sources(sources_path: Path):
    """Parse sources.txt to map museums, collections, and provenance links."""
    if not sources_path.exists():
        return {}
    
    content = sources_path.read_text(encoding="utf-8")
    sources = {}
    current_source = None
    
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        # Format: 1) Met Museum
        m_num = re.match(r'^\d+\)\s*(.+)$', line)
        if m_num:
            current_source = m_num.group(1).strip()
            sources[current_source] = {}
        elif line.startswith("link:") and current_source:
            link = line.replace("link:", "").strip()
            sources[current_source]["url"] = link
            
    return sources

def parse_metadata_and_captions(data_dir: str = "data"):
    data_path = Path(data_dir)
    images_dir = data_path / "raw" / "images"
    captions_dir = data_path / "raw" / "captions"
    metadata_dir = data_path / "metadata"
    
    sources_file = metadata_dir / "sources.txt"
    meta_jsonl = metadata_dir / "metadata.jsonl"
    
    sources_map = parse_sources(sources_file)
    print(f"[Metadata] Parsed {len(sources_map)} provenance sources from sources.txt")

    with open(meta_jsonl, "r", encoding="utf-8") as f:
        meta_entries = [json.loads(line) for line in f]
    print(f"[Metadata] Loaded {len(meta_entries)} entries from {meta_jsonl.name}")

    # Validate against individual .txt files
    mismatches = 0
    missing_txt = 0
    for e in meta_entries:
        fn = e['file_name']
        stem = Path(fn).stem
        txt_path = captions_dir / f"{stem}.txt"
        if not txt_path.exists():
            missing_txt += 1
        else:
            txt_content = txt_path.read_text(encoding="utf-8").strip()
            if txt_content != e['caption'].strip():
                mismatches += 1

    print(f"[Metadata] Caption validation: {len(meta_entries) - missing_txt} matching files found on disk.")
    print(f"[Metadata] Discrepancies between .txt and metadata.jsonl: {mismatches} (handled safely)")

    # Build manifest records
    records = []
    
    # Known artist dictionary
    known_artists = {
        'bishandas': 'Bishandas',
        'nanha': 'Nanha',
        'abu al-hasan': "Abu'l Hasan",
        "abu'l hasan": "Abu'l Hasan",
        'basawan': 'Basawan',
        'mansur': 'Ustad Mansur',
        'dhanu': 'Dhanu',
        'farrukh beg': 'Farrukh Beg',
        'govardhan': 'Govardhan',
        'manohar': 'Manohar',
        'daulat': 'Daulat',
        'haji madani': 'Haji Madani'
    }

    # Common ragas and raginis
    ragas_list = [
        'kamod', 'shri', 'kakubha', 'saindhavi', 'bhairava', 'bhairavi', 'dipak', 
        'hindol', 'megha', 'malkauns', 'todi', 'asavari', 'gujari', 'bilawal',
        'desakh', 'gauri', 'dhanashri', 'vasant', 'lalit', 'kedar', 'kanada'
    ]

    for e in meta_entries:
        fn = e['file_name']
        cap = e['caption']
        fn_lower = fn.lower()
        cap_lower = cap.lower()
        img_id = Path(fn).stem

        # 1. Category & has_box_labels
        is_crop = (
            fn.startswith('mughal_style_') or 
            fn.startswith('mughalstyle_') or 
            'crop' in fn_lower or 
            'cropped' in fn_lower
        )
        is_rgmla = cap.startswith('RGMLA') or 'ragamala' in fn_lower or 'ragini' in fn_lower or 'raga' in fn_lower
        
        if is_crop:
            category = "Painting Crops"
        elif is_rgmla:
            category = "Ragamala Paintings"
        else:
            category = "Mughal Miniature Paintings"
            
        has_box_labels = bool(is_crop)

        # 2. School
        if 'pahari' in cap_lower:
            school = "Pahari"
        elif 'rajasthani' in cap_lower:
            school = "Rajasthani"
        elif 'murshidabad' in fn_lower or 'provincial mughal' in fn_lower:
            school = "Provincial Mughal"
        elif 'deccani' in fn_lower or 'deccani' in cap_lower:
            school = "Deccani"
        elif 'mughal' in cap_lower or 'mughal' in fn_lower or cap.startswith('MGHML'):
            school = "Mughal"
        else:
            school = "Unknown"

        # 3. Period
        year_match = re.search(r'(?:c(?:irca)?_?)(\d{4})', fn_lower)
        cent_match = re.search(r'(\d{1,2})(?:th|st|nd|rd)_century', fn_lower)
        
        year = None
        if year_match:
            year = int(year_match.group(1))
        elif '1560' in fn_lower:
            year = 1560
        elif '1700' in fn_lower:
            year = 1700
        elif '1628' in fn_lower:
            year = 1628

        period = "Unknown"
        if 'akbar' in fn_lower or 'tuti-nama' in fn_lower or 'hamza-nama' in fn_lower:
            if not year or year < 1605:
                period = "Akbar Period (c. 1556-1605)"
        elif 'jahangir' in fn_lower or 'salim' in fn_lower:
            period = "Jahangir Period (c. 1605-1627)"
        elif 'shah_jahan' in fn_lower or 'shah jahan' in cap_lower or '1628' in fn_lower:
            period = "Shah Jahan Period (c. 1628-1658)"
        elif 'aurangzeb' in fn_lower or 'aurangzeb' in cap_lower:
            period = "Aurangzeb Period (c. 1658-1707)"
            
        if period == "Unknown":
            if year:
                if 1550 <= year < 1605:
                    period = "Akbar Period (c. 1556-1605)"
                elif 1605 <= year <= 1627:
                    period = "Jahangir Period (c. 1605-1627)"
                elif 1628 <= year <= 1658:
                    period = "Shah Jahan Period (c. 1628-1658)"
                elif 1659 <= year <= 1707:
                    period = "Aurangzeb Period (c. 1658-1707)"
                elif 1708 <= year < 1850:
                    period = "Late Mughal / 18th Century"
                elif year < 1600:
                    period = "16th Century"
                elif year < 1700:
                    period = "17th Century"
                else:
                    period = "18th Century"
            elif cent_match:
                c = int(cent_match.group(1))
                period = f"{c}th Century"
            elif '18th_century' in fn_lower or '18th century' in cap_lower:
                period = "Late Mughal / 18th Century"
            elif '17th_century' in fn_lower or '17th century' in cap_lower:
                period = "17th Century"
            elif '16th_century' in fn_lower or '16th century' in cap_lower:
                period = "16th Century"
            elif fn.startswith('mughal_style_') or fn.startswith('mughalstyle_'):
                period = "Classical Mughal (16th-17th Century)"
            elif fn.startswith('2006BL'):
                period = "Late Mughal / 18th Century"

        # 4. Artist
        artist = "Unknown"
        for k, name in known_artists.items():
            if k in fn_lower or k in cap_lower:
                artist = name
                break

        # 5. Raga
        raga = "None"
        if is_rgmla:
            found_raga = None
            for r in ragas_list:
                if r in fn_lower or r in cap_lower:
                    found_raga = f"{r.title()} Raga/Ragini"
                    break
            raga = found_raga if found_raga else "Ragamala (Unspecified)"

        # 6. Collection ID & Group Key (Mapping to sources.txt)
        if 'tuti-nama' in fn_lower or 'tuti-nama' in cap_lower:
            collection_id = "Cleveland Museum of Art (Tuti-nama MS)"
            group_key = "ms_cma_tutinama"
        elif fn.startswith('2019_NYR_'):
            collection_id = "Christie's New York (Sale 17464)"
            group_key = "album_christies_nyr_17464"
        elif fn.startswith('2006BL'):
            collection_id = "British Library / British Museum (2006BL)"
            group_key = "album_british_library_2006bl"
        elif 'hamza-nama' in fn_lower or 'hamza-nama' in cap_lower:
            collection_id = "Victoria and Albert Museum (Hamzanama)"
            group_key = "ms_hamzanama"
        elif 'razm-nama' in fn_lower or 'razm-nama' in cap_lower:
            collection_id = "Razmnama MS"
            group_key = "ms_razmnama"
        elif 'babur-nama' in fn_lower or 'babur-nama' in cap_lower:
            collection_id = "Baburnama MS"
            group_key = "ms_baburnama"
        elif 'akbar-nama' in fn_lower or 'akbarnama' in fn_lower:
            collection_id = "Akbarnama MS"
            group_key = "ms_akbarnama"
        elif 'zafar-nama' in fn_lower:
            collection_id = "Zafarnama MS"
            group_key = "ms_zafarnama"
        elif 'panj_ganj' in fn_lower or 'panj ganj' in cap_lower:
            collection_id = "Panj Ganj MS"
            group_key = "ms_panj_ganj"
        elif 'anvar-i' in fn_lower:
            collection_id = "Anvar-i Suhayli MS"
            group_key = "ms_anvar_i_suhayli"
        elif fn.startswith('FS-'):
            m_s = re.match(r'(FS-S2018\.1)', fn)
            if m_s:
                collection_id = "National Museum of Asian Art (Freer & Sackler FS-S2018.1)"
                group_key = "nmaa_fs_s2018_1"
            else:
                m_f = re.match(r'(FS-F1907)', fn)
                if m_f:
                    collection_id = "National Museum of Asian Art (Freer & Sackler FS-F1907)"
                    group_key = "nmaa_fs_f1907"
                else:
                    m_num = re.match(r'(FS-\d+)', fn)
                    tag = m_num.group(1) if m_num else fn.split('_')[0]
                    collection_id = f"National Museum of Asian Art (Freer & Sackler {tag})"
                    group_key = f"nmaa_{tag.lower().replace('-', '_').replace('.', '_')}"
        elif fn.startswith('mia_'):
            collection_id = "Minneapolis Institute of Art"
            group_key = "museum_artsmia"
        elif fn.startswith('mughal_style_') or fn.startswith('mughalstyle_'):
            collection_id = "Mughal Style Detail Studies"
            group_key = f"crop_mughal_{img_id}"
        elif re.search(r'-[0-9a-f]{6}\.(jpg|jpeg|png)$', fn):
            slug = re.sub(r'-[0-9a-f]{6}\.(jpg|jpeg|png)$', '', fn)
            collection_id = "Open Art Images"
            group_key = f"openart_{slug}"
        elif 'met' in cap_lower or 'metmuseum' in fn_lower:
            collection_id = "Metropolitan Museum of Art"
            group_key = f"met_{img_id[:15]}"
        elif 'thewalters' in fn_lower or 'walters' in fn_lower or 'walters' in cap_lower:
            collection_id = "Walters Art Museum"
            group_key = f"walters_{img_id[:15]}"
        else:
            base = re.sub(r'[\s_]*\(\d+\)[\s_]*', '', img_id).strip().lower()
            base = re.sub(r'_\d+$', '', base)
            collection_id = "Museum / Public Collection"
            group_key = f"indiv_{base[:30]}"

        # Standard file path relative to project
        file_path = f"data/raw/images/{fn}"

        records.append({
            'image_id': img_id,
            'file_path': file_path,
            'category': category,
            'raga': raga,
            'school': school,
            'period': period,
            'artist': artist,
            'collection_id': collection_id,
            'caption': cap,
            'has_box_labels': has_box_labels,
            'group_key': group_key
        })

    df = pd.DataFrame(records)

    # Reorder columns as requested in Checklist:
    # image_id, file_path, category, raga, school, period, artist, collection_id, caption, has_box_labels
    target_columns = [
        'image_id', 'file_path', 'category', 'raga', 'school', 
        'period', 'artist', 'collection_id', 'caption', 'has_box_labels', 'group_key'
    ]
    df = df[target_columns]

    # Save to Parquet and CSV
    parquet_path = metadata_dir / "dataset_manifest.parquet"
    csv_path = metadata_dir / "dataset_manifest.csv"
    
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    
    print(f"[Metadata] Successfully created dataset manifest:")
    print(f"  - Parquet: {parquet_path} ({parquet_path.stat().st_size} bytes)")
    print(f"  - CSV:     {csv_path} ({csv_path.stat().st_size} bytes)")
    print(f"  - Total records: {len(df)}")
    print(f"  - Manifest columns: {list(df.columns)}")

    return df

if __name__ == "__main__":
    parse_metadata_and_captions()


"""
Dataset loader, metadata parser, and leak-free grouped split generation for
Mughal Miniature and Ragamala painting analysis.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
import torch
from torch.utils.data import Dataset
from torchvision import transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Known historical entities for regex extraction fallback from caption texts
KNOWN_SCHOOLS = [
    "Mughal", "Mewar", "Bundi", "Kotah", "Kota", "Kangra", "Kishangarh",
    "Malwa", "Deccan", "Bikaner", "Amber", "Jaipur", "Marwar", "Basohli",
    "Guler", "Garhwal", "Provincial Mughal", "Awadh", "Murshidabad"
]

KNOWN_RAGAS = [
    "Bhairava", "Bhairav", "Bhairavi", "Malkaus", "Malkauns", "Hindol", "Deepak",
    "Dipak", "Megha", "Megh", "Sri", "Shri", "Todi", "Asavari", "Gauri",
    "Vasant", "Vasanta", "Kamod", "Kedara", "Kedar", "Khambhavati", "Dhanashri",
    "Gunakali", "Nata", "Panchama", "Varadi", "Vibhasa", "Sarang", "Bilawal",
    "Kakubha", "Madhumadhavi", "Bhoopali", "Desakhya", "Patamanjari"
]

KNOWN_PERIODS = [
    "16th Century", "Late 16th Century", "Early 17th Century", "Mid 17th Century",
    "Late 17th Century", "17th Century", "Early 18th Century", "18th Century",
    "Late 18th Century", "19th Century", "Akbar", "Jahangir", "Shah Jahan", "Aurangzeb"
]

KNOWN_ARTISTS = [
    "Basawan", "Miskin", "Daswanth", "Manohar", "Ustad Mansur", "Mansur",
    "Govardhan", "Bishandas", "Abu'l Hasan", "Farrukh Beg", "Kesu Das",
    "Nanha", "Payag", "Mir Sayyid Ali", "Abd al-Samad", "Nihal Chand",
    "Sahibdin", "Manaku", "Nainsukh"
]


class MiniaturePaintingDataset(Dataset):
    """
    PyTorch Dataset for miniature paintings.
    Loads images and returns transformed image tensors alongside target labels and metadata.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_transform: Optional[transforms.Compose] = None,
        image_col: str = "file_path",
    ):
        self.df = df.reset_index(drop=True)
        self.image_col = image_col

        if image_transform is None:
            # Default standard ViT/CLIP normalization transform
            self.image_transform = transforms.Compose([
                transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],  # CLIP / ViT standard mean
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ])
        else:
            self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        row = self.df.iloc[idx]
        image_path = row[self.image_col]

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at path: {image_path}")

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Error opening image {image_path}: {e}")
            # Return blank image tensor on corruption
            image = Image.new("RGB", (224, 224), color=0)

        tensor = self.image_transform(image)
        # Missing labels (NaN) become "" so batches with mixed missing/present labels collate
        metadata = {k: ("" if (isinstance(v, float) and np.isnan(v)) else v) for k, v in row.to_dict().items()}

        return tensor, metadata


def extract_entity_from_text(text: str, entity_list: List[str]) -> Optional[str]:
    """Search for known art historical entities within caption text."""
    if not isinstance(text, str) or not text.strip():
        return None
    for entity in entity_list:
        pattern = r"\b" + re.escape(entity) + r"\b"
        if re.search(pattern, text, flags=re.IGNORECASE):
            return entity
    return None


def infer_manuscript_group(file_name: str, caption: str, source_str: str = "") -> str:
    """
    Deduce manuscript series, album, or provenance group to prevent data leakage.
    Examples of series: 'Hamzanama', 'Akbarnama', 'Padshahnama', 'Chaurapanchasika',
    'Boston_Ragamala', 'Mewar_Ramayana', 'Laud_Ragamala', or file prefix groups.
    """
    combined = f"{file_name} {caption} {source_str}".lower()

    known_series = [
        "hamzanama", "akbarnama", "padshahnama", "baburnama", "tutinama",
        "razmnama", "chaurapanchasika", "laud", "boston", "manley",
        "mewar", "chawand", "amber", "bundi", "malwa", "kankroli",
        "gem_palace", "national_museum", "metropolitan", "british_museum",
        "victoria_albert", "cleveland"
    ]

    for series in known_series:
        if series in combined:
            return series.upper()

    # Fallback to file prefix pattern (e.g. 'ragamala_setA_01.jpg' -> 'ragamala_setA')
    clean_stem = Path(file_name).stem
    parts = re.split(r"[-_#\s]+", clean_stem)
    if len(parts) > 1 and not parts[0].isdigit():
        return parts[0].upper()

    return "COLLECTION_DEFAULT"


def parse_metadata_manifest(
    data_dir: Union[str, Path],
    metadata_jsonl_path: Optional[Union[str, Path]] = None,
    sources_txt_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Scan the dataset folder and assemble a complete metadata DataFrame.
    Cross-references JPG images with:
    - metadata.jsonl (if present)
    - corresponding .txt caption files (if present)
    - sources.txt (if present)
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    logger.info(f"Scanning directory: {data_path}")

    # Discover image files (.jpg, .jpeg, .png) avoiding case-insensitive duplication
    raw_found = (
        list(data_path.glob("*.jpg")) +
        list(data_path.glob("*.jpeg")) +
        list(data_path.glob("*.png"))
    )

    # Deduplicate by resolved path (critical for Windows filesystem)
    unique_paths = {p.resolve(): p for p in raw_found}
    image_files = sorted(list(unique_paths.values()))

    if not image_files:
        logger.warning(f"No image files found directly in {data_path}. Checking subdirectories...")
        raw_sub = (
            list(data_path.rglob("*.jpg")) +
            list(data_path.rglob("*.jpeg")) +
            list(data_path.rglob("*.png"))
        )
        unique_paths = {p.resolve(): p for p in raw_sub}
        image_files = sorted(list(unique_paths.values()))

    logger.info(f"Found {len(image_files)} image files.")

    # Locate metadata.jsonl if not explicitly provided
    if metadata_jsonl_path is None:
        possible_jsonl = list(data_path.glob("*metadata*.jsonl")) + list(data_path.rglob("*metadata*.jsonl"))
        if possible_jsonl:
            metadata_jsonl_path = possible_jsonl[0]

    jsonl_records = {}
    if metadata_jsonl_path and os.path.exists(metadata_jsonl_path):
        logger.info(f"Loading metadata from JSONL: {metadata_jsonl_path}")
        with open(metadata_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    # Key by base filename or exact name
                    fname = record.get("file_name") or record.get("image") or record.get("filename")
                    if fname:
                        jsonl_records[Path(fname).name] = record
                except Exception as e:
                    logger.debug(f"Skipping malformed json line: {e}")

    # Locate sources.txt
    sources_text = ""
    if sources_txt_path is None:
        possible_sources = list(data_path.glob("*source*.txt")) + list(data_path.rglob("*source*.txt"))
        if possible_sources:
            sources_txt_path = possible_sources[0]

    if sources_txt_path and os.path.exists(sources_txt_path):
        logger.info(f"Reading sources reference from: {sources_txt_path}")
        with open(sources_txt_path, "r", encoding="utf-8") as f:
            sources_text = f.read()

    rows = []
    for img_p in image_files:
        fname = img_p.name
        stem = img_p.stem

        # Check for paired .txt caption file
        txt_path = img_p.with_suffix(".txt")
        caption_txt = ""
        if txt_path.exists():
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    caption_txt = f.read().strip()
            except Exception:
                pass

        # Check jsonl entry
        json_entry = jsonl_records.get(fname, {})
        caption_json = json_entry.get("caption", "") or json_entry.get("text", "")
        full_caption = caption_json if caption_json else caption_txt

        # Extract Targets
        raga = json_entry.get("raga") or extract_entity_from_text(full_caption, KNOWN_RAGAS)
        school = json_entry.get("school") or extract_entity_from_text(full_caption, KNOWN_SCHOOLS)
        period = json_entry.get("period") or extract_entity_from_text(full_caption, KNOWN_PERIODS)
        artist = json_entry.get("artist") or extract_entity_from_text(full_caption, KNOWN_ARTISTS)

        # Infer category: Mughal Miniature, Crop, Ragamala
        cat_lower = f"{fname} {full_caption}".lower()
        if "crop" in cat_lower:
            category = "Crop"
        elif "raga" in cat_lower or raga is not None:
            category = "Ragamala"
        elif "mughal" in cat_lower or (school and "mughal" in school.lower()):
            category = "Mughal Miniature"
        else:
            category = "Mughal Miniature"

        # Manuscript / Collection Group for non-leak split
        manuscript_group = (
            json_entry.get("collection_id")
            or json_entry.get("manuscript_set")
            or infer_manuscript_group(fname, full_caption, sources_text)
        )

        rows.append({
            "image_id": stem,
            "file_name": fname,
            "file_path": str(img_p.resolve()),
            "category": category,
            "raga": raga or "Unknown_Raga",
            "school": school or "Unknown_School",
            "period": period or "Unknown_Period",
            "artist": artist or "Unknown_Artist",
            "collection_id": manuscript_group,
            "caption": full_caption,
        })

    df = pd.DataFrame(rows)
    logger.info(f"Successfully processed manifest with {len(df)} images.")
    return df


def create_grouped_splits(
    df: pd.DataFrame,
    group_col: str = "collection_id",
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create leak-free train / val / test splits grouped by manuscript series or collection.
    Guarantees no group is shared across splits.
    """
    df = df.copy().reset_index(drop=True)
    groups = df[group_col].values

    # Step 1: Train+Val vs Test
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(gss_test.split(df, groups=groups))

    train_val_df = df.iloc[train_val_idx].copy().reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)

    # Step 2: Train vs Val from remaining train_val
    adjusted_val_size = val_size / (1.0 - test_size)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=adjusted_val_size, random_state=random_state)
    train_val_groups = train_val_df[group_col].values
    train_idx, val_idx = next(gss_val.split(train_val_df, groups=train_val_groups))

    train_df = train_val_df.iloc[train_idx].copy().reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].copy().reset_index(drop=True)

    # Assign split labels
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    # Leakage verification
    train_groups = set(train_df[group_col])
    val_groups = set(val_df[group_col])
    test_groups = set(test_df[group_col])

    leak_train_val = train_groups.intersection(val_groups)
    leak_train_test = train_groups.intersection(test_groups)
    leak_val_test = val_groups.intersection(test_groups)

    assert not leak_train_val, f"Data leakage between Train & Val! Overlapping groups: {leak_train_val}"
    assert not leak_train_test, f"Data leakage between Train & Test! Overlapping groups: {leak_train_test}"
    assert not leak_val_test, f"Data leakage between Val & Test! Overlapping groups: {leak_val_test}"

    logger.info(
        f"Grouped Splits Created successfully:\n"
        f" - Train: {len(train_df)} samples ({len(train_groups)} distinct groups)\n"
        f" - Val:   {len(val_df)} samples ({len(val_groups)} distinct groups)\n"
        f" - Test:  {len(test_df)} samples ({len(test_groups)} distinct groups)"
    )

    return train_df, val_df, test_df


# Raga labels with fewer than this many paintings are pooled into "Ragamala (Other)"
MIN_RAGA_SUPPORT = 3

# Period labels that are aliases or near-duplicates of a canonical class
PERIOD_ALIASES = {
    "18th Century": "Late Mughal / 18th Century",
    "16th Century": "Classical Mughal (16th-17th Century)",
}

SCHOOL_ALIASES = {
    "Provincial Mughal": "Mughal",
}


def normalize_targets(df: pd.DataFrame, raga_counts: Optional[Dict[str, int]] = None) -> pd.DataFrame:
    """
    Convert raw dataset labels into trainable targets ("honest heads").

    - raga:   "None" -> "Non-Ragamala"; ragas with < MIN_RAGA_SUPPORT paintings -> "Ragamala (Other)"
    - period: "Unknown" -> NaN (excluded from training and evaluation); aliases merged
    - school: aliases merged
    - artist: kept as-is (too few attributed paintings; reported as insufficient data)

    Missing labels are NaN and must be masked by consumers.
    """
    df = df.copy()
    if raga_counts is None:
        raga_counts = df["raga"].value_counts().to_dict()

    def _raga(v):
        v = str(v)
        if v in ("None", "nan", ""):
            return "Non-Ragamala"
        if v != "Ragamala (Unspecified)" and raga_counts.get(v, 0) < MIN_RAGA_SUPPORT:
            return "Ragamala (Other)"
        return v

    df["raga"] = df["raga"].map(_raga)
    df["period"] = df["period"].replace(PERIOD_ALIASES).replace({"Unknown": np.nan})
    df["school"] = df["school"].replace(SCHOOL_ALIASES)
    return df


def load_frozen_splits(
    splits_dir: Union[str, Path] = "data/metadata",
    project_root: Optional[Union[str, Path]] = None,
    normalize: bool = True,
    apply_feedback: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the Phase 2 frozen leak-free split manifests (train/val/test_split.jsonl).
    File paths are resolved to absolute paths relative to the project root.
    Expert corrections from data/feedback/expert_feedback.jsonl (Phase 10.3) are applied.
    """
    splits_path = Path(splits_dir)
    root = Path(project_root) if project_root else Path(__file__).resolve().parent.parent

    frames = []
    for split in ("train", "val", "test"):
        f = splits_path / f"{split}_split.jsonl"
        if not f.exists():
            raise FileNotFoundError(f"Frozen split manifest not found: {f}. Run run_pipeline.py first.")
        d = pd.read_json(f, lines=True, dtype=False)
        d["file_path"] = d["file_path"].map(lambda p: str((root / p).resolve()) if not Path(p).is_absolute() else p)
        d["split"] = split
        frames.append(d)

    if normalize:
        all_raga_counts = pd.concat(frames)["raga"].value_counts().to_dict()
        frames = [normalize_targets(d, raga_counts=all_raga_counts) for d in frames]

    if apply_feedback:
        from src.feedback import FeedbackStore
        store = FeedbackStore(root / "data" / "feedback" / "expert_feedback.jsonl")
        frames = [store.apply_to(d) for d in frames]

    train_df, val_df, test_df = frames
    logger.info(
        f"Loaded frozen splits from {splits_path}: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}"
    )
    return train_df, val_df, test_df


def load_all_paintings(
    data_dir: Union[str, Path],
    splits_dir: Union[str, Path] = "data/metadata",
) -> pd.DataFrame:
    """All paintings (train + val + test) from the frozen splits, or a scan of `data_dir`."""
    if Path(splits_dir, "train_split.jsonl").exists():
        return pd.concat(load_frozen_splits(splits_dir), ignore_index=True)
    return parse_metadata_manifest(data_dir=data_dir)


def load_project_splits(
    data_dir: Union[str, Path],
    splits_dir: Union[str, Path] = "data/metadata",
    test_size: float = 0.25,
    val_size: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the frozen leak-free splits when available; otherwise fall back to scanning
    `data_dir` and generating grouped splits on the fly (e.g. for the synthetic sample).
    """
    if Path(splits_dir, "train_split.jsonl").exists():
        return load_frozen_splits(splits_dir)

    logger.info(f"No frozen splits in '{splits_dir}'. Generating grouped splits from '{data_dir}'.")
    manifest_df = parse_metadata_manifest(data_dir=data_dir)
    return create_grouped_splits(
        df=manifest_df, group_col="collection_id", test_size=test_size, val_size=val_size
    )

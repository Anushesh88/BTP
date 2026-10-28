"""
Phase 2.1 refinement: make the leak-free grouping actually leak-free.

metadata_parser.py derives `group_key` from the collection field. Paintings whose collection
is generic ("Museum / Public Collection") became one-painting groups (`indiv_...`) even when
their title or caption names the manuscript or album they belong to. An audit of the first
split found 5 Tuti-nama folios in test while 153 from the same manuscript were in train, two
near-duplicate images split across val and test, and one exact duplicate counted twice.

Refinements, applied before partitioning:
  1. Series named in the title / caption ("from a Tuti-nama", "from the late Shah Jahan album",
     "Akbarnama", ...) join one group per series (Tuti-nama joins the Cleveland manuscript group).
  2. One-source collections ("Mughal Style Detail Studies") form a single group.
  3. Near-duplicate images (frozen ViT-B/16 cosine >= 0.90) are merged into one group.
  4. Exact duplicates (cosine >= 0.98) keep only their first copy.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

NEAR_DUPLICATE_COS = 0.90
EXACT_DUPLICATE_COS = 0.98

# (pattern on normalized title + caption + collection text, group key)
SERIES_RULES: List[Tuple[str, str]] = [
    # File names are often truncated mid-word: "from_a_tuti_c_1560", "tuti_n", "tuti_nam"
    (r"\btuti(?: ?n(?:a(?:ma?)?)?)?\b", "ms_cma_tutinama"),
    (r"\bhamza ?nama\b", "series_hamzanama"),
    (r"\bakbar ?nama\b", "series_akbarnama"),
    (r"\bbabur ?nama\b", "series_baburnama"),
    (r"\bpadshah ?nama\b", "series_padshahnama"),
    (r"\brazm ?nama\b", "series_razmnama"),
    (r"\bzafar ?nama\b", "series_zafarnama"),
    (r"\banvar i suhayli\b", "series_anvar_i_suhayli"),
    (r"\bpanj ganj\b", "series_panj_ganj"),
]
ALBUM_PATTERN = re.compile(r"\bfrom the ((?:[a-z]+ ){0,3}?[a-z]+) album\b")
SINGLE_SOURCE_COLLECTIONS = {"Mughal Style Detail Studies": "coll_mughal_style_detail_studies"}


def _normalize(text: str) -> str:
    return re.sub(r"[\s_\-]+", " ", str(text).lower())


class _UnionFind:
    def __init__(self, items):
        self.parent = {i: i for i in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Keep the more descriptive (non-singleton) key as the representative
            keep, drop = (rb, ra) if ra.startswith("indiv_") and not rb.startswith("indiv_") else (ra, rb)
            self.parent[drop] = keep


def series_key(row) -> str:
    text = _normalize(f"{row['image_id']} {row.get('caption', '')} {row.get('collection_id', '')}")
    for pattern, key in SERIES_RULES:
        if re.search(pattern, text):
            return key
    m = ALBUM_PATTERN.search(text)
    if m:
        return "album_" + m.group(1).strip().replace(" ", "_")
    return None


def compute_all_embeddings(df: pd.DataFrame, cache_dir: str = "data/cache") -> np.ndarray:
    """Frozen ViT-B/16 embeddings for every painting (cached and validated by image ID)."""
    from src.feature_extractor import extract_embeddings_for_split

    d = df.copy()
    d["file_path"] = d["file_path"].map(lambda p: str(Path(p).resolve()))
    emb, _ = extract_embeddings_for_split(df=d, split_name="all", model_name="vit_b_16", cache_dir=cache_dir)
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)


def refine_group_keys(df: pd.DataFrame, embeddings: np.ndarray = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Return (df with refined `group_key` and exact duplicates removed, audit report).
    `embeddings` must be row-aligned with `df`; pass None to skip the image-similarity steps.
    """
    df = df.reset_index(drop=True).copy()
    original = df["group_key"].copy()
    report = {"series_merges": {}, "collection_merges": {}, "near_duplicate_pairs": [], "exact_duplicates_removed": []}

    # 1-2. Text and collection based keys. Every painting naming the same series is unioned,
    # including ones already in a manuscript group under a different key (e.g. "Razmnama MS").
    series_of = {}
    for i, row in df.iterrows():
        new = SINGLE_SOURCE_COLLECTIONS.get(row.get("collection_id"))
        if new and new != row["group_key"]:
            df.at[i, "group_key"] = new
            report["collection_merges"][new] = report["collection_merges"].get(new, 0) + 1
        series_of[i] = series_key(row)

    uf = _UnionFind(set(df["group_key"]) | {s for s in series_of.values() if s})
    for i, s in series_of.items():
        if s:
            if df.at[i, "group_key"] != s and str(df.at[i, "group_key"]).startswith("indiv_"):
                report["series_merges"][s] = report["series_merges"].get(s, 0) + 1
            uf.union(s, df.at[i, "group_key"])

    # 3-4. Image similarity
    drop = set()
    if embeddings is not None:
        sim = embeddings @ embeddings.T
        np.fill_diagonal(sim, 0)
        rows, cols = np.where(np.triu(sim, 1) >= NEAR_DUPLICATE_COS)
        for i, j in zip(rows, cols):
            report["near_duplicate_pairs"].append(
                {"a": df.at[i, "image_id"], "b": df.at[j, "image_id"], "cosine": round(float(sim[i, j]), 3)}
            )
            uf.union(df.at[i, "group_key"], df.at[j, "group_key"])
            if sim[i, j] >= EXACT_DUPLICATE_COS and i not in drop:
                drop.add(j)
                report["exact_duplicates_removed"].append({"kept": df.at[i, "image_id"], "removed": df.at[j, "image_id"]})
    df["group_key"] = df["group_key"].map(uf.find)

    report["groups_before"] = int(original.nunique())
    report["groups_after"] = int(df["group_key"].nunique())
    report["paintings_regrouped"] = int((df["group_key"] != original).sum())
    df = df.drop(index=sorted(drop)).reset_index(drop=True)
    logger.info(
        f"Group refinement: {report['groups_before']} -> {report['groups_after']} groups, "
        f"{report['paintings_regrouped']} paintings regrouped, {len(drop)} exact duplicates removed"
    )
    return df, report

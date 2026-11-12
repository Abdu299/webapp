"""Splits the OSV-5M test photos in images/test/00 into train, validation and test sets for the privacy network.
    These photos are unseen by the geolocation model, which was trained on the OSV-5M train split.
    The split uses only the photos' metadata, no geolocation model, so it is the same for every model.

    Does this:
      1. joins each photo with its coordinates and sequence from test.csv
      2. William's street-level test photos become the test set, and every other photo from their sequences is left out
      3. the rest is split by sequence into train (90% of the sequences) and validation (10%)

    Saves data/osv5m/<split>/photos.csv for split in train, validation, test, with name, lat, lon, sequence, path.
    Each model's distances on the unblurred photos are saved next to it by unblurred_km.py, and load_split joins them.

    Needs
    data/osv5m/raw/test.csv and the unzipped photos in data/osv5m/raw/images/test/00 (from download_osv5m.py)
    data/osv5m/raw/osv5m_street_test.csv (William's street-level test photos)

    Run from the repo root:
        python -m src.data_prep.privacy_split
"""

from pathlib import Path

import numpy as np
import pandas as pd


data_dir = Path("privacyData/osv5m")
raw_dir = data_dir / "raw"
image_dir = raw_dir / "images" / "test" / "00"
street_test = raw_dir / "osv5m_street_test.csv"


def photo_table():
    """Photos in image_dir with their coordinates and sequence from test.csv."""
    meta = pd.read_csv(raw_dir / "test.csv", dtype={"id": str, "sequence": str})
    names = [path.stem for path in image_dir.glob("*.jpg")]
    print(f"{len(names)} photos in {image_dir}")

    df = meta[meta["id"].isin(names)] # only photos in 00
    df = df.rename(columns={"id": "name", "latitude": "lat", "longitude": "lon"})[["name", "lat", "lon", "sequence"]]
    df["path"] = [str(image_dir / f"{name}.jpg") for name in df["name"]]
    return df.sort_values("name").reset_index(drop=True)


def split_by_sequence(df, val_fraction=0.1, seed=0):
    """Splits the photos into train and validation, with val_fraction of the sequences (not photos) in validation.
        A whole sequence goes to one split, so train and validation have no near identical photos.
    """
    sequences = df["sequence"].unique()
    rng = np.random.default_rng(seed)
    val_sequences = rng.choice(sequences, size=int(len(sequences) * val_fraction), replace=False)
    is_val = df["sequence"].isin(val_sequences)
    return df[~is_val], df[is_val]


def load_split(split, models=()):
    """Photos of one split, with a <model>_unblurred_km column for each model in models (saved by unblurred_km.py)."""
    df = pd.read_csv(data_dir / split / "photos.csv", dtype={"name": str, "sequence": str})
    for model in models:
        km = pd.read_csv(data_dir / split / f"unblurred_km_{model}.csv", dtype={"name": str})
        df = df.merge(km.rename(columns={"km": f"{model}_unblurred_km"}), on="name", how="left")
    return df


def save_split(df, split):
    """Saves the photos of one split as data/osv5m/<split>/photos.csv."""
    (data_dir / split).mkdir(parents=True, exist_ok=True)
    df.to_csv(data_dir / split / "photos.csv", index=False)


if __name__ == "__main__":
    df = photo_table()

    # William's street-level test photos are the test set
    street = pd.read_csv(street_test, dtype={"name": str})
    is_test = df["name"].isin(street["name"])
    test = df[is_test]

    # leave out the test photos' whole sequences, so train and validation have no near identical copies of test photos
    in_test_sequence = df["sequence"].isin(test["sequence"])
    rest = df[~in_test_sequence]
    train, val = split_by_sequence(rest)

    save_split(train, "train")
    save_split(val, "validation")
    save_split(test, "test")

    print(f"{(in_test_sequence & ~is_test).sum()} photos left out for sharing a sequence with a test photo")
    print(f"train: {len(train)} photos, {train['sequence'].nunique()} sequences")
    print(f"validation: {len(val)} photos, {val['sequence'].nunique()} sequences")
    print(f"test: {len(test)} photos (William's), {test['sequence'].nunique()} sequences")
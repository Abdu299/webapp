
#configure storage before importing Hugging Face or Matplotlib
from data.paths import PROJECT_DIR, DATA_DIR, DATASET_DIR, CSV_DIR, CACHE_DIR
import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath
from datetime import datetime

from huggingface_hub import hf_hub_download
import pandas as pd
import numpy as np
import matplotlib

# Use a non-interactive backend so plots can be saved on servers without a display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from data.split_dataset import split_dataset


# Limit the number of images used so we do not need to train on the full OSV-5M dataset.
TRAIN_SUBSET_SIZE = 1000_000
TEST_SUBSET_SIZE = 20_000

# The dataset stores images across numbered ZIP archives.
# We use 18 training archives and one test archive.
TRAIN_FOLDERS = [ "00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17" ]

TEST_FOLDERS = ["00"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}



# Check that a ZIP archive contains image entries.
# Store the result in a marker file so unchanged archives do not need to be scanned again.
def verify_archive(archive_path, marker_path):

    stats = archive_path.stat()

    # A cached result is reusable when the ZIP's size and modification time match.
    # This avoids repeatedly reading the ZIP directory during data preparation.
    if marker_path.is_file():
        try:
            with marker_path.open() as f:
                marker = json.load(f)

            if ( marker.get("size_bytes") == stats.st_size and marker.get("mtime_ns") == stats.st_mtime_ns and marker.get("image_count", 0) > 0 ):
                return marker["image_count"]

        # If the marker is missing valid information, scan the archive again.
        except (OSError, ValueError, AttributeError):
            pass

    image_count = 0

    #read and check the ZIP without extracting any files
    # Reading the ZIP directory lets us count image entries without creating individual files.
    with zipfile.ZipFile(archive_path) as z:
        

        for info in z.infolist():

            if not info.is_dir() and PurePosixPath(info.filename).suffix.lower() in IMAGE_SUFFIXES:
                image_count += 1



    # Save the image count and ZIP metadata for future runs.
    # Writing to a temporary file first reduces the risk of leaving a partial marker.
    marker = { "size_bytes": stats.st_size, "mtime_ns": stats.st_mtime_ns, "image_count": image_count }

    temporary_marker = marker_path.with_suffix(".json.tmp")

    with temporary_marker.open("w") as f:
        json.dump(marker, f)

    temporary_marker.replace(marker_path)
    return image_count


#keep images inside ZIP archives on project storage to avoid one inode per image
def downloading_dataSet_toDataSetFolder():

    marker_dir = DATASET_DIR / "archive_manifests"

    marker_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "test"]:

        csv_path = DATASET_DIR / f"{split}_metadata.csv"

        # Download metadata only if it is missing or empty.
        # The CSV contains information such as image IDs, coordinates and countries.
        if not csv_path.is_file() or csv_path.stat().st_size == 0:

            downloaded_csv = hf_hub_download( repo_id="osv5m/osv5m", filename=f"{split}.csv", repo_type="dataset", local_dir=str(DATASET_DIR), cache_dir=str(CACHE_DIR / "hub") )

            Path(downloaded_csv).replace(csv_path)

        folders = TEST_FOLDERS

        if split == "train":
            folders = TRAIN_FOLDERS

        for folder in folders:

            archive_path = DATASET_DIR / "images" / split / f"{folder}.zip"

            # Skip downloading archives that already exist and are not empty.
            # Hugging Face also uses a cache to avoid unnecessary downloads.
            if not archive_path.is_file() or archive_path.stat().st_size == 0:

                print(f"Downloading: {split}/{folder}.zip", flush=True)

                downloaded_zip = hf_hub_download(
                    repo_id="osv5m/osv5m",
                    filename=f"images/{split}/{folder}.zip",
                    repo_type="dataset",
                    local_dir=str(DATASET_DIR),

                    cache_dir=str(CACHE_DIR / "hub")
                )

                archive_path = Path(downloaded_zip)

            

            marker_path = marker_dir / f"{split}_{folder}_zip.json"
            image_count = verify_archive(archive_path, marker_path)

            print(f"{split}/{folder} images in ZIP:", image_count, flush=True)

        print(f"{split} metadata:", csv_path, flush=True)

        print(f"{split} ZIP directory:", DATASET_DIR / "images" / split, flush=True)


#match ZIP members to their coordinates without extracting images
#the existing function name is retained for compatibility
def extracrt_images_coordinates(split="train", subset_size=5000):

    csv_path = DATASET_DIR / f"{split}_metadata.csv"

    # Map each image ID to its ZIP archive and its internal filename.
    # The dataset loader can later use both values to read images from ZIP files.
    paths = {}
    members = {}

    folders = TEST_FOLDERS

    if split == "train":
        folders = TRAIN_FOLDERS

    for folder in folders:
        archive_path = DATASET_DIR / "images" / split / f"{folder}.zip"

        with zipfile.ZipFile(archive_path) as z:

            for info in z.infolist():
                name = PurePosixPath(info.filename)

                if info.is_dir() or name.suffix.lower() not in IMAGE_SUFFIXES:
                    continue

                # The image ID is taken from the filename without its extension.
                # It is used to match the image with its row in the metadata CSV.
                image_id = name.stem

                # Duplicate IDs would make the image-to-metadata mapping ambiguous.
                if image_id in paths:
                    raise RuntimeError(f"Duplicate image ID in {split}: {image_id}")

                paths[image_id] = str(archive_path.resolve())

                members[image_id] = info.filename

    parts = []

    # Process the large metadata CSV in chunks to reduce peak memory usage.
    # Only keep rows whose image IDs exist in the downloaded ZIP archives.
    for chunk in pd.read_csv(csv_path, dtype={"id": str}, chunksize=100_000):

        chunk = chunk[chunk["id"].isin(paths)].copy()

        chunk["image_path"] = chunk["id"].map(paths)
        chunk["archive_member"] = chunk["id"].map(members)

        # Remove samples missing a ZIP location or valid coordinate fields.
        chunk = chunk.dropna(subset=["image_path", "archive_member", "latitude", "longitude"])

        if len(chunk) > 0:
            parts.append(chunk)

    if len(parts) == 0:
        raise RuntimeError(f"no ZIP images matched the metadata for {split}")

    df = pd.concat(parts, ignore_index=True)
    print(f"{split} matched images:", len(df), flush=True)

    # Select a reproducible random subset rather than using the first rows.
    # random_state=42 ensures the same sampled images across repeated runs
    # when the available data and its ordering remain unchanged.
    df = df.sample(n=min(subset_size, len(df)), random_state=42).copy()

    return df


# Standardize country values so the same country receives the same label.
def clean_country_column(df, split):

    # OSV-5M already provides ISO-like two-letter country codes.
    if "country" not in df.columns:
        raise ValueError(f"{split} metadata has no country column")

    df = df.copy()

    # Convert country codes to uppercase and remove surrounding whitespace.
    # This prevents differences such as "no", "NO" and " NO " from creating
    # separate categories.
    df["country"] = df["country"].astype("string").str.strip().str.upper()
    df = df.dropna(subset=["country"])

    # Accept only two-letter alphabetic codes.
    # This checks the format, not whether the code is an officially assigned country.
    df = df[df["country"].str.fullmatch(r"[A-Z]{2}", na=False)].copy()

    if df.empty:
        raise ValueError(f"No valid country labels found for {split}")

    return df


# Create a plot showing the geographic distribution of sampled training images.
# This helps inspect where the dataset contains many or few observations.
def visualize(df):

    plt.figure(figsize=(12, 5))
    plt.scatter(
        df["longitude"],
        df["latitude"],
        s=4,
        alpha=0.5
    )

    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("OSV-5M training image locations")
    plt.grid(True, alpha=0.3)

    # Save the plot instead of displaying it, since training may run remotely.
    plt.savefig(CSV_DIR / "training_locations.png", dpi=150, bbox_inches="tight")
    plt.close()


# Make the project's local CSV directory point to the configured shared CSV storage.
def link_csv_folder():

    #keep existing training/test scripts that read data/csv working
    local_csv = Path(__file__).resolve().parent / "csv"

    # No symlink is needed if both paths already refer to the same location.
    if local_csv.resolve() == CSV_DIR.resolve():
        return

    # Preserve existing local CSV files before creating the symlink.
    # The timestamp gives each backup a unique name.
    if local_csv.exists() or local_csv.is_symlink():
        backup_name = "csv_before_shared_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        backup_path = DATA_DIR / backup_name
        shutil.move(str(local_csv), str(backup_path))
        print("Previous CSV files kept in:", backup_path, flush=True)

    # A symbolic link lets old code use its original path while the files
    # are physically stored in the configured CSV directory.
    local_csv.symlink_to(CSV_DIR, target_is_directory=True)
    print("Training and test CSV path:", local_csv, "->", CSV_DIR, flush=True)


# Run the complete data preparation pipeline before model training.
def main():

    print("Running file:", Path(__file__).resolve(), flush=True)
    print("Project storage:", PROJECT_DIR.resolve(), flush=True)
    print("Images and metadata:", DATASET_DIR.resolve(), flush=True)
    
    print("CSV files:", CSV_DIR.resolve(), flush=True)
    print("Download cache:", CACHE_DIR.resolve(), flush=True)
    print("Image storage: ZIP archives (no extraction)", flush=True)

    # Make sure the required metadata and image archives are available.
    downloading_dataSet_toDataSetFolder()

    # TRAINING DATA: classify countries instead of grid cells

    # Match training images with metadata, sample up to one million images,
    # and keep only samples with properly formatted country codes.
    train_df = extracrt_images_coordinates(split="train", subset_size=TRAIN_SUBSET_SIZE)
    train_df = clean_country_column(train_df, "train")

    # Assign one integer class ID to each country in the training data.
    # Sorting makes the mapping deterministic for the same set of countries.
    countries = sorted(train_df["country"].unique().tolist())
    country_to_label = {}

    for index, country in enumerate(countries):
        country_to_label[country] = index

    # Classification models predict numeric class indices rather than text codes.
    train_df["label"] = train_df["country"].map(country_to_label).astype(int)
    train_df.to_csv(CSV_DIR / "training_full.csv", index=False)


    # Save the mapping so training and evaluation use identical class IDs.
    # The mapping is also needed to convert predicted IDs back to country codes.
    with (CSV_DIR / "country_to_label.json").open("w") as file:
        json.dump(country_to_label, file, indent=2, sort_keys=True)

    print("Training images before train/validation split:", len(train_df), flush=True)
    print("Training countries:", len(countries), flush=True)
    visualize(train_df)

    # TEST DATA: reuse the training mapping, never fit a new one
    test_df = extracrt_images_coordinates(split="test", subset_size=TEST_SUBSET_SIZE)
    test_df = clean_country_column(test_df, "test")

    # Countries absent from training receive label -1.
    # The model has no output class for these countries, so evaluation
    # must handle them separately or count them as incorrect.
    # Creating a new mapping from the test data would give inconsistent labels.
    test_df["label"] = test_df["country"].map(country_to_label).fillna(-1).astype(int)
    test_df.to_csv(CSV_DIR / "test_evaluation.csv", index=False)

    print("Test images:", len(test_df), flush=True)
    print("Known countries:", (test_df["label"] != -1).sum(), flush=True)
    print("Unknown countries:", (test_df["label"] == -1).sum(), flush=True)

    # Create the training and validation splits from the prepared data.
    # The exact split strategy is implemented in data.split_dataset.
    split_dataset()
    
    # Keep the old cell CSV files untouched; country CSVs use csv_countries.
    print("All data is ready in:", DATA_DIR, flush=True)


if __name__ == "__main__":
    main()

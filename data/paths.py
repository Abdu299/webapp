
import os
import tempfile
from pathlib import Path


# Store datasets, generated CSV files and caches on shared project storage.
# This avoids filling up the limited storage in the home directory.
PROJECT_DIR = Path("/fp/projects01/ec674/ec-abdulrag")
DATA_DIR = PROJECT_DIR / "data"
DATASET_DIR = DATA_DIR / "osv5m"
CSV_DIR = DATA_DIR / "csv_countries"
CACHE_DIR = PROJECT_DIR / "hf_cache"
TMP_DIR = PROJECT_DIR / "tmp"


# Configure storage locations before other modules start downloading
# datasets, creating cache files or using temporary storage.
def setup_storage():

    # Hugging Face uses these environment variables to decide where
    # downloaded models, datasets and cache files should be stored.
    # Separating hub and Xet caches keeps the downloaded files organized.
    os.environ["HF_HOME"] = str(CACHE_DIR)
    os.environ["HF_HUB_CACHE"] = str(CACHE_DIR / "hub")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(CACHE_DIR / "hub")
    os.environ["HF_XET_CACHE"] = str(CACHE_DIR / "xet")

    # Redirect general application caches and Matplotlib configuration
    # away from the home directory and into project storage.
    os.environ["XDG_CACHE_HOME"] = str(PROJECT_DIR / "cache")
    os.environ["MPLCONFIGDIR"] = str(PROJECT_DIR / "cache" / "matplotlib")

    # Some libraries create temporary files during downloads or processing.
    # Using project storage helps avoid running out of local temporary space.
    os.environ["TMPDIR"] = str(TMP_DIR)

    # Create the required directories if they do not already exist.
    # exist_ok=True allows this setup to run multiple times without errors.
    for folder in [
        DATASET_DIR,
        CSV_DIR,
        CACHE_DIR / "hub",
        CACHE_DIR / "xet",
        Path(os.environ["MPLCONFIGDIR"]),
        TMP_DIR
    ]:
        folder.mkdir(parents=True, exist_ok=True)

    #tempfile may already have cached a different temporary directory
    # Updating TMPDIR alone may not affect Python's tempfile module
    # if it has already selected a temporary directory.
    # Setting tempfile.tempdir explicitly ensures it uses our project folder.
    tempfile.tempdir = str(TMP_DIR)


# Apply the storage configuration immediately when this module is imported.
# This is why data.paths should be imported before Hugging Face or Matplotlib.
setup_storage()

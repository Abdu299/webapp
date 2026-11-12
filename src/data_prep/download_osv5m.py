"""Downloads the OSV-5M test metadata (test.csv) and the first test zip (images/test/00.zip), then unzips it.
    These photos are unseen by the geolocation model (trained on the OSV-5M train split)
    and are used to train, validate and test the privacy network (see privacy_split.py).

    Run from the repo root:
        python -m src.data_prep.download_osv5m [out_dir]
    out_dir defaults to data/osv5m/raw.
"""

import sys
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download


out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("privacyData/osv5m/raw")


def download(filename, subfolder=None):
    """Downloads one file of the OSV-5M dataset repo into out_dir and returns its path."""
    path = hf_hub_download(repo_id="osv5m/osv5m", repo_type="dataset",
                           filename=filename, subfolder=subfolder, local_dir=out_dir)
    return Path(path)


def unzip(path):
    """Unzips next to the zip file, then deletes the zip to save space."""
    with zipfile.ZipFile(path) as z:
        z.extractall(path.parent)
    path.unlink()


if __name__ == "__main__":
    download("test.csv")
    unzip(download("00.zip", subfolder="images/test"))

from data.paths import DATA_DIR

import os
import zipfile
import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


# Custom PyTorch Dataset that reads images and labels from a CSV file.
# Supports both images stored inside ZIP archives and extracted image files.
class GeoDataset(Dataset):

    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.data_dir = DATA_DIR

        #each DataLoader worker opens its own archive handles, only when needed
        # Cache open ZIP archives so we do not reopen the same archive
        # every time an image is requested.
        self.archives = {}
        self.archive_pid = None

    def __len__(self):
        return len(self.df)

    # Return an open ZIP archive, reusing an existing handle when possible.
    def get_archive(self, archive_path):

        # Each running process has its own process ID (PID).
        pid = os.getpid()

        #forked workers must not share inherited ZIP file handles
        # If the process ID changes, close any inherited ZIP handles.
        # Sharing these handles between worker processes can cause problems
        # when multiple workers try to read from the same archive.
        if self.archive_pid != pid:
            self.close()
            self.archive_pid = pid

        key = str(archive_path)

        # Open the archive only the first time it is needed by this worker.
        # Later requests for the same ZIP reuse the existing handle.
        if key not in self.archives:
            self.archives[key] = zipfile.ZipFile(archive_path)

        return self.archives[key]

    # Called by PyTorch DataLoader to load one image and its label.
    def __getitem__(self, index):

        row = self.df.iloc[index]
        image_path = Path(row["image_path"])

        # Relative paths are resolved against our configured data directory.
        # Absolute paths can be used directly without modification.
        if not image_path.is_absolute():
            image_path = self.data_dir / image_path

        label = int(row["label"])

        # archive_member identifies the image's filename inside a ZIP archive.
        # This allows images to be loaded without extracting the entire ZIP.
        archive_member = row.get("archive_member")

        if pd.notna(archive_member) and str(archive_member) != "":
            archive = self.get_archive(image_path)

            # Read only the requested image from the ZIP archive.
            # Converting to RGB ensures a consistent three-channel image
            # regardless of the original image format.
            with archive.open(str(archive_member)) as image_file:

                with Image.open(image_file) as img:
                    image = img.convert("RGB")

        else:
            #also support older CSV files that point to extracted images
            # This keeps the Dataset compatible with our previous data format.
            with Image.open(image_path) as img:
                image = img.convert("RGB")


        # Apply preprocessing such as resizing, tensor conversion and
        # normalization before passing the image to the model.
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    # Close open archives to release file handles and system resources.
    def close(self):

        for archive in self.archives.values():
            archive.close()

        self.archives = {}

    # Control how the Dataset is serialized when DataLoader creates workers.
    def __getstate__(self):

        #spawned workers receive paths and metadata, never open ZIP handles
        # ZIP file objects should not be transferred between processes.
        # Each worker opens its own archives when it needs them.
        state = self.__dict__.copy()
        state["archives"] = {}
        state["archive_pid"] = None
        return state

    # Clean up any remaining open ZIP archives when the Dataset is destroyed.
    def __del__(self):

        if hasattr(self, "archives"):
            self.close()

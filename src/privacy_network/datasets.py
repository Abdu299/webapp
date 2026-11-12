import os

import lightning as pl
from PIL import Image
from torch.utils.data import DataLoader, Dataset, default_collate

from src.data_prep.privacy_split import load_split
from src.utils.common import radius_km


class PhotoDataset(Dataset):
    def __init__(self, df, transform):
        self.paths = list(df["path"])
        self.lats = list(df["lat"])  # degrees
        self.lons = list(df["lon"])
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            image = self.transform(Image.open(self.paths[i]).convert("RGB"))  # (3, 224, 224)
        except OSError:
            return None  # if unreadable photo
        return {"image": image, "lat": self.lats[i], "lon": self.lons[i], "row": i}


def collate(samples):
    return default_collate([s for s in samples if s is not None])  # leaves out unreadable photos


def photo_loader(df, transform, batch_size=64):
    return DataLoader(
        PhotoDataset(df, transform),
        batch_size=batch_size,
        num_workers=min(8, os.cpu_count()),
        collate_fn=collate,
    )


class PhotoDataModule(pl.LightningDataModule):
    def __init__(self, config, transform):
        super().__init__()
        self.judge = config["name"]
        self.batch_size = config["batch_size"]
        self.num_workers = config["num_workers"]
        self.transform = transform

    def setup(self, stage=None):
        self.train_set = PhotoDataset(load_split("train"), self.transform)

        # validate only on photos the judge locates without blur
        val = load_split("validation", models=[self.judge])
        val = val[val[f"{self.judge}_unblurred_km"] <= radius_km]
        self.val_set = PhotoDataset(val, self.transform)

    def loader(self, dataset, train):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=train,
            drop_last=train,
            num_workers=self.num_workers,
            collate_fn=collate,
            persistent_workers=True,
            pin_memory=True,
        )

    def train_dataloader(self):
        return self.loader(self.train_set, train=True)

    def val_dataloader(self):
        return self.loader(self.val_set, train=False)

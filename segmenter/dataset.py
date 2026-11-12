"""
Mapillary Vistas v1.2 dataset that returns (image, clue mask) pairs.

Each example is one street photo, resized so it fits DINOv2, and its label
converted from the 66 Vistas classes to our 10 clue classes (see clue_classes.py).
"""

import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from clue_classes import IGNORE_INDEX, build_lookup_table

# DINOv2 was trained on images normalised with the ImageNet mean and std,
# so our images must be normalised the same way.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class VistasClueDataset(Dataset):
    def __init__(self, root, split, size=(448, 336)):
        """
        root:  the Vistas folder, e.g. /fp/projects01/ec674/data/vistas
        split: "training" or "validation"
        size:  (width, height) to resize to. Both must be multiples of 14
               (DINOv2's patch size). 448x336 = 32x24 patches and keeps the
               4:3 shape of most Vistas photos.
        """
        self.size = size
        assert size[0] % 14 == 0 and size[1] % 14 == 0

        self.image_dir = os.path.join(root, split, "images")
        self.label_dir = os.path.join(root, split, "v1.2", "labels")

        self.ids = sorted(os.path.splitext(name)[0] for name in os.listdir(self.image_dir))

        self.lut = build_lookup_table(os.path.join(root, "config_v1.2.json"))



    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        image_id = self.ids[i]

        # Open the files
        image = Image.open(os.path.join(self.image_dir, image_id + ".jpg")).convert("RGB")
        label = Image.open(os.path.join(self.label_dir, image_id + ".png"))

        # Map the 66 Vistas classes to the 10 classes and convert bck to a PIL image
        mask = Image.fromarray(self.lut[np.array(label)])

        # Resize image and mask. BILINEAR blends neighbouring pixels to look smooth. NEAREST picks one existing pixel
        image = image.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        # Image to tensor
        # np.array(image) has shape [336, 448, 3]. Reorder to [3, 336, 448] since pytorch expects channels first
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255

        # Mask to tensor
        # Use long() since CrossEntropyLoss requires this type for class IDs
        mask = torch.from_numpy(np.array(mask)).long()

        # --- Step 5: normalise the image ---
        # 11: subtract IMAGENET_MEAN and divide by IMAGENET_STD per channel.
        #   Hint: make them tensors of shape [3, 1, 1] so they broadcast over
        #   H and W: torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        # Normalization
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        image = (image - mean) / std

        return image, mask


if __name__ == "__main__":
    # Quick check: run `python dataset.py` from the segmenter folder on Fox.
    root = "/fp/projects01/ec674/data/vistas"
    ds = VistasClueDataset(root, "validation")
    print("examples:", len(ds))  # expect 2000

    image, mask = ds[0]
    print("image:", image.shape, image.dtype)  # expect [3, 336, 448] float32
    print("mask: ", mask.shape, mask.dtype)    # expect [336, 448] int64
    print("image mean/std:", image.mean().item(), image.std().item())  # roughly 0 and 1
    print("clue ids in mask:", mask.unique().tolist())  # only 0-9 and 255
    print("ignore pixels:", (mask == IGNORE_INDEX).float().mean().item())

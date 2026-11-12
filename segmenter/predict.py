"""
Predict clue masks for photos that have no labels (the week 3 test sets).

For every photo in a test set CSV (from test_sets.py) this saves one PNG with the
photo's name, at the photo's original size, where each pixel is a clue class id 0-9.
Other code (class_blur.py, Victoria's blur) only needs these files, not the model.

Run from the segmenter folder:
    python predict.py --csv /fp/projects01/ec674/williaan/data/test_sets/im2gps.csv \
                      --out /fp/projects01/ec674/williaan/masks/im2gps
Quick test on a few photos (works on CPU):
    python predict.py --csv im2gps.csv --out masks_test --limit 4
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from clue_classes import CLUE_CLASSES
from dataset import IMAGENET_MEAN, IMAGENET_STD
from evaluate import MODELS, NUM_CLASSES
from visualize import colorize


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    # The best model so far (RESULTS.md, higher resolution 896x672 section)
    p.add_argument("--checkpoint", default="/fp/projects01/ec674/williaan/checkpoints/segmenter_896_seed0.pt")
    p.add_argument("--model", choices=MODELS, default="segmenter")
    # Must match the size the checkpoint was trained with (width height)
    p.add_argument("--size", type=int, nargs=2, default=[896, 672])
    # Also save "photo | mask" pictures for the first N photos, to check the masks by eye
    p.add_argument("--preview", type=int, default=20)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def load_image(path, size):
    """Photo -> (normalised tensor [3, H, W] at size, original (width, height)).
    Same resize and normalisation as dataset.py, so the model sees what it was trained on."""
    image = Image.open(path).convert("RGB")
    original_size = image.size

    image = image.resize(tuple(size), Image.BILINEAR)
    image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (image - mean) / std, original_size


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)

    df = pd.read_csv(args.csv, dtype={"name": str})
    if args.limit:
        df = df.head(args.limit)

    model = MODELS[args.model](num_classes=NUM_CLASSES).to(device).eval()
    model.decoder.load_state_dict(torch.load(args.checkpoint, map_location=device))

    os.makedirs(os.path.join(args.out, "preview"), exist_ok=True)

    # Pixels per class over all photos, to compare with the Vistas shares (domain shift check)
    counts = torch.zeros(NUM_CLASSES, dtype=torch.long)

    with torch.no_grad():
        for i, (name, path) in enumerate(zip(df["name"], df["path"])):
            image, (width, height) = load_image(path, args.size)
            scores = model(image[None].to(device))

            # Back to the original size before argmax, so the mask lines up with the photo.
            # Resizing the scores (not the class ids) gives smoother edges.
            scores = F.interpolate(scores, size=(height, width), mode="bilinear", align_corners=False)
            mask = scores.argmax(dim=1)[0].cpu()

            counts += torch.bincount(mask.flatten(), minlength=NUM_CLASSES)
            Image.fromarray(mask.numpy().astype(np.uint8)).save(os.path.join(args.out, name + ".png"))

            if i < args.preview:
                photo = np.array(Image.open(path).convert("RGB"))
                both = np.concatenate([photo, colorize(mask)], axis=1)
                Image.fromarray(both).save(os.path.join(args.out, "preview", name + ".jpg"))

            if i % 100 == 0:
                print(f"{i}/{len(df)}", flush=True)

    print(f"saved {len(df)} masks to {args.out}")
    print(f"{'class':<28} {'pixels':>7}")
    for (_, class_name, _), count in zip(CLUE_CLASSES, counts):
        print(f"{class_name:<28} {count.item() / counts.sum().item():7.1%}")


if __name__ == "__main__":
    main()

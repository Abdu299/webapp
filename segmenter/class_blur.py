"""
Week 3 experiment: blur one clue class at a time and measure how far the geolocation
model's guess moves away from the true location.

For every photo the model guesses the location of 21 versions:
- the original photo
- 10 "blur" versions: all pixels of one clue class blurred (masks from predict.py)
- 10 "random" versions: a random region with the same area as that class blurred.
  This is the area control: blurring a third of a photo (e.g. sky) can hurt the guess
  just because so much is gone. A class is only a real clue if blurring it hurts more
  than blurring the same amount of random pixels.

Geolocation model (--model):
- osv5m: the OSV-5M baseline, the same stand-in Victoria uses (src/baseline.py). Saves km.
- geovit: Abdu's country model (geovit.py). Saves the guessed country and probabilities.
One row per photo and version is saved; clue_ranking.py turns them into the ranking.

Run from the segmenter folder:
    python class_blur.py --csv .../test_sets/im2gps.csv --masks .../masks/im2gps \
                         --out /fp/projects01/ec674/williaan/results/class_blur_im2gps.csv
Quick test on a few photos:
    python class_blur.py --csv ... --masks ... --out test.csv --limit 4
Abdu's model:
    python class_blur.py --model geovit --csv ... --masks ... --out .../class_blur_geovit_osv5m.csv
"""

import argparse
import contextlib
import math
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from clue_classes import CLUE_CLASSES
from evaluate import NUM_CLASSES
from geovit import load_geovit, prepare_geovit


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--masks", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", choices=["osv5m", "geovit"], default="osv5m")
    # A clone of https://github.com/gastruc/osv5m, which has the code for the baseline model
    p.add_argument("--osv5m-code", default="/fp/projects01/ec674/williaan/osv5m")
    # The baseline resizes the short side to 224 and keeps the centre square, so we work on
    # that same square, at 2x the model's size so the blur is done on a sharper image.
    p.add_argument("--work-size", type=int, default=448)
    # Victoria uses sigma 20 on 224x224 images. 40 on 448x448 is the same blur.
    p.add_argument("--sigma", type=float, default=40)
    # Grow every mask by this many pixels (on the 448 square), because predicted masks have
    # rounded edges and the outline of a sign is still a clue if it stays sharp.
    p.add_argument("--dilate", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def load_osv5m_baseline(code_dir, device):
    """The frozen OSV-5M baseline, loaded like Victoria's load_model()."""
    sys.path.insert(0, code_dir)
    from models.huggingface import Geolocalizer

    # The model's config uses paths relative to the osv5m folder
    with contextlib.chdir(code_dir):
        model = Geolocalizer.from_pretrained("osv5m/baseline")
    return model.to(device).eval().requires_grad_(False)


def centre_square(image, size, interpolation):
    """Short side -> size, then the centre size x size square: the part the baseline sees."""
    image = TF.resize(image, size, interpolation=interpolation)
    return TF.center_crop(image, size)


def gaussian_blur(x, sigma):
    """Blur the whole image. Two 1D blurs (rows, then columns) give the same result as one
    2D blur but are much faster for a big sigma, as in Victoria's full_blur."""
    kernel_size = 2 * math.ceil(3 * sigma) + 1
    x = TF.gaussian_blur(x, [kernel_size, 1], sigma)
    return TF.gaussian_blur(x, [1, kernel_size], sigma)


def dilate(mask, pixels):
    """Grow a [H, W] 0/1 mask by pixels in every direction (max over a square window)."""
    if pixels == 0:
        return mask
    return F.max_pool2d(mask[None, None], 2 * pixels + 1, stride=1, padding=pixels)[0, 0]


def random_region(area, size, generator, device):
    """A [size, size] 0/1 mask covering the given share of pixels, as a few random blobs.
    Smooth random noise is cut at the right level, so the blobs look more like objects
    than single scattered pixels do, and the area is exact."""
    if area == 0:
        return torch.zeros(size, size, device=device)

    noise = torch.rand(1, 1, size // 32, size // 32, generator=generator).to(device)
    noise = F.interpolate(noise, size=(size, size), mode="bilinear", align_corners=False)[0, 0]

    # The highest `area` share of the noise values becomes the region
    threshold = noise.flatten().sort().values[min(int((1 - area) * noise.numel()), noise.numel() - 1)]
    return (noise >= threshold).float()


def haversine(lat1, lon1, lat2, lon2):
    """Distance in km between points in degrees (works on numbers and numpy arrays)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)

    df = pd.read_csv(args.csv, dtype={"name": str})
    if args.limit:
        df = df.head(args.limit)

    if args.model == "osv5m":
        model = load_osv5m_baseline(args.osv5m_code, device)
    else:
        model, label_to_country = load_geovit(device)
        country_to_label = {country: label for label, country in label_to_country.items()}
    # Same random regions every run
    generator = torch.Generator().manual_seed(args.seed)

    rows = []
    for i, photo in enumerate(df.itertuples()):
        image = centre_square(Image.open(photo.path).convert("RGB"), args.work_size, TF.InterpolationMode.BICUBIC)
        # NEAREST so every pixel keeps a real class id
        mask = centre_square(Image.open(os.path.join(args.masks, photo.name + ".png")), args.work_size,
                             TF.InterpolationMode.NEAREST)

        x = TF.to_tensor(image).to(device)                       # [3, S, S], values 0-1
        mask = torch.from_numpy(np.array(mask)).to(device)       # [S, S], class ids
        blurred = gaussian_blur(x, args.sigma)

        # (version, class name, region to blur). Region None = the original photo.
        versions = [("original", "", None)]
        for class_id, class_name, _ in CLUE_CLASSES:
            region = dilate((mask == class_id).float(), args.dilate)
            area = region.mean().item()
            versions.append(("blur", class_name, region))
            versions.append(("random", class_name, random_region(area, args.work_size, generator, device)))

        # Blurred pixels inside the region, original pixels outside
        version_images = [x if region is None else region * blurred + (1 - region) * x
                          for _, _, region in versions]
        common = [{"name": photo.name, "continent": photo.continent, "version": version, "class": class_name,
                   "area": 0.0 if region is None else region.mean().item()}
                  for version, class_name, region in versions]

        if args.model == "osv5m":
            # The baseline's own transform (resize to 224, normalise), on the CPU like Victoria's
            images = [model.transform(TF.to_pil_image(v.cpu())) for v in version_images]
            with torch.no_grad():
                # One batch of 21 versions. The baseline returns (lat, lon) in radians.
                guesses = torch.rad2deg(model(torch.stack(images).to(device))).cpu().numpy()

            km = haversine(photo.lat, photo.lon, guesses[:, 0], guesses[:, 1])
            for row, (pred_lat, pred_lon), distance in zip(common, guesses, km):
                rows.append({**row, "pred_lat": pred_lat, "pred_lon": pred_lon, "km": distance})
        else:
            with torch.no_grad():
                probs = torch.softmax(model(prepare_geovit(torch.stack(version_images))), dim=1).cpu()
            guess = probs[0].argmax().item()                # its country for the original photo
            true = country_to_label.get(photo.country)      # None = a country it never saw
            for row, p in zip(common, probs):
                rows.append({**row, "country": photo.country,
                             "pred_country": label_to_country[p.argmax().item()],
                             "correct": p.argmax().item() == true,
                             # How sure it still is of its original guess: drops when a clue is hidden
                             "p_guess": p[guess].item(),
                             "p_true": 0.0 if true is None else p[true].item()})

        if i % 50 == 0:
            print(f"{i}/{len(df)}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"saved {len(rows)} rows ({len(df)} photos x {1 + 2 * NUM_CLASSES} versions) to {args.out}")


if __name__ == "__main__":
    main()

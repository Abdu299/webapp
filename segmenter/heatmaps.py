"""
Week 4: occlusion heatmaps. For every photo, blur one window at a time and record how much the
model's guess changes. Big change = the model used that part.
- osv5m (OSV-5M baseline): km the guess moves from its original guess
- geovit (Abdu's country model): drop in the probability of its original country guess
Saves one [448, 448] heatmap per photo as .npy; heatmap_ranking.py compares them with the masks.

Run from the segmenter folder:
    python heatmaps.py --csv .../test_sets/osv5m.csv --out /fp/projects01/ec674/williaan/heatmaps/osv5m
Quick test:
    python heatmaps.py --csv ... --out test_heatmaps --limit 2
Abdu's model:
    python heatmaps.py --model geovit --csv ... --out /fp/projects01/ec674/williaan/heatmaps_geovit/osv5m
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from class_blur import load_osv5m_baseline, centre_square, gaussian_blur, haversine
from geovit import load_geovit, prepare_geovit


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", choices=["osv5m", "geovit"], default="osv5m")
    p.add_argument("--osv5m-code", default="/fp/projects01/ec674/williaan/osv5m")
    p.add_argument("--work-size", type=int, default=448)
    p.add_argument("--sigma", type=float, default=40)
    # Window 64 px (2% of the square), moved 16 px at a time: 27 x 27 = 729 guesses per photo
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def prepare(batch):
    """batch: [N, 3, 448, 448], values 0-1 -> what the model expects: [N, 3, 224, 224], normalised.
    Same as model.transform, but for a whole batch on the GPU."""
    batch = F.interpolate(batch, size=224, mode="bicubic", antialias=True)
    return (batch - MEAN.to(batch.device)) / STD.to(batch.device)


def osv5m_scorer(model):
    """guess: batch -> [N, 2] lat/lon in degrees. change: km each new guess is from the original."""
    def guess(batch):
        return torch.rad2deg(model(prepare(batch))).cpu().numpy()

    def change(original, new):
        return haversine(original[0, 0], original[0, 1], new[:, 0], new[:, 1])
    return guess, change


def geovit_scorer(model):
    """guess: batch -> [N, countries] probabilities. change: how much the original top country lost."""
    def guess(batch):
        return torch.softmax(model(prepare_geovit(batch)), dim=1).cpu().numpy()

    def change(original, new):
        top = original[0].argmax()                  # the country it guessed for the full photo
        drop = original[0, top] - new[:, top]
        # Hiding a distracting spot can make it more sure; count that as 0, like km can't be negative
        return np.clip(drop, 0, None)
    return guess, change


def occlusion_heatmap(guess, change, x, blurred, window, stride, batch_size):

    S = x.shape[-1]
    original = guess(x)

    # Top left corner of every window
    starts = range(0, S - window + 1, stride)
    corners = [(r, c) for r in starts for c in starts]

    heat_sum = torch.zeros(S, S, device=x.device)
    count = torch.zeros(S, S, device=x.device)

    for i in range(0, len(corners), batch_size):
        batch = corners[i:i + batch_size]       # the next 64 corners (fewer in the last batch)

        # One mask per window: [N, 1, S, S]. The 1 lets one mask cover all 3 colour channels
        regions = torch.zeros(len(batch), 1, S, S, device=x.device)
        for j, (r, c) in enumerate(batch):
            regions[j, 0, r:r + window, c:c + window] = 1

        # Same mixing as before. x and blurred are [1, 3, S, S] and get repeated for all N masks
        versions = regions * blurred + (1 - regions) * x       # [N, 3, S, S]

        # How much the guess changed, for all N windows at once
        moved = change(original, guess(versions))

        for j in range(len(batch)):
            heat_sum += float(moved[j]) * regions[j, 0]
            count += regions[j, 0]

    return heat_sum / count


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)

    df = pd.read_csv(args.csv, dtype={"name": str})
    if args.limit:
        df = df.head(args.limit)
    os.makedirs(args.out, exist_ok=True)

    if args.model == "osv5m":
        guess, change = osv5m_scorer(load_osv5m_baseline(args.osv5m_code, device))
    else:
        guess, change = geovit_scorer(load_geovit(device)[0])

    start = time.time()
    for i, photo in enumerate(df.itertuples()):
        # Same 448 square as class_blur.py, so heatmap pixels line up with the mask pixels
        image = centre_square(Image.open(photo.path).convert("RGB"), args.work_size, TF.InterpolationMode.BICUBIC)
        x = TF.to_tensor(image).unsqueeze(0).to(device)        # [1, 3, S, S]
        blurred = gaussian_blur(x, args.sigma)

        with torch.no_grad():
            heat = occlusion_heatmap(guess, change, x, blurred, args.window, args.stride, args.batch_size)

        # float16 halves the file size (~400 KB per photo); km or probabilities don't need more
        np.save(os.path.join(args.out, photo.name + ".npy"), heat.cpu().numpy().astype(np.float16))

        if i % 50 == 0:
            print(f"{i}/{len(df)}  {time.time() - start:.0f} s", flush=True)

    print(f"saved {len(df)} heatmaps to {args.out}")


if __name__ == "__main__":
    main()

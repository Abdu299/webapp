"""
Week 4: heatmap-based clue ranking. For each photo and clue class:
    heat ratio  = mean heat on the class's pixels / mean heat of the whole photo
    excess heat = (heat ratio - 1) * area
Heat ratio is per pixel: 1 = the model's guess depends on this class as much as on an average
pixel, >1 = more. Excess heat is for the whole class: how much more heat it holds than an
average region of the same size. That's the heatmap version of class_blur's clue effect
(blur the class vs blur a random region of the same size), so it is the one to compare.
Works for both models' heatmaps (the ratio has no unit).

Uses the heatmaps from heatmaps.py and the masks from predict.py. No GPU needed:
    python heatmap_ranking.py --csv .../test_sets/osv5m.csv --heatmaps .../heatmaps/osv5m \
        --masks .../masks/osv5m --out .../results/heatmap_ratio_osv5m.csv \
        --blur .../results/class_blur_osv5m.csv
--blur (optional): the class_blur.py rows of the same model, to compare the two rankings.
"""

import argparse
import os

import numpy as np
import pandas as pd
import torchvision.transforms.functional as TF
from PIL import Image

from class_blur import centre_square
from clue_classes import CLUE_CLASSES
from clue_ranking import CLASS_NAMES, confidence_interval, effects


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--heatmaps", required=True)
    p.add_argument("--masks", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--blur", default="")
    p.add_argument("--work-size", type=int, default=448)
    p.add_argument("--min-area", type=float, default=0.001)   # same rule as clue_ranking.py
    p.add_argument("--min-photos", type=int, default=10)      # same as clue_ranking.py
    p.add_argument("--min-class-photos", type=int, default=20)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def per_photo_rows(df, args):
    """One row per photo and class present: area, heat ratio, excess heat."""
    rows = []
    for photo in df.itertuples():
        heat = np.load(os.path.join(args.heatmaps, photo.name + ".npy")).astype(np.float32)
        # Same square and resizing as class_blur.py, so mask pixels line up with heat pixels
        mask = centre_square(Image.open(os.path.join(args.masks, photo.name + ".png")), args.work_size,
                             TF.InterpolationMode.NEAREST)
        mask = np.array(mask)

        photo_mean = heat.mean()            # mean heat of the whole photo
        if photo_mean == 0:                 # guess never moved: nothing to rank
            continue

        for class_id, class_name, _ in CLUE_CLASSES:
            in_class = mask == class_id     # True where the mask is this class
            area = in_class.mean()          # share of the photo this class covers (0-1)
            if area < args.min_area:
                continue
            ratio = heat[in_class].mean() / photo_mean
            rows.append({"name": photo.name, "continent": photo.continent, "class": class_name,
                         "area": area, "ratio": ratio, "excess": (ratio - 1) * area})
    return pd.DataFrame(rows)


def ranking(rows):
    """Mean heat ratio and excess heat per class, each with a 95% interval."""
    table = []
    for class_name, group in rows.groupby("class"):
        ratio_low, ratio_high = confidence_interval(group["ratio"].to_numpy())
        excess_low, excess_high = confidence_interval(group["excess"].to_numpy())
        table.append({"class": class_name, "photos": len(group), "area": group["area"].mean(),
                      "heat ratio": group["ratio"].mean(), "ratio low": ratio_low, "ratio high": ratio_high,
                      "excess heat": group["excess"].mean(), "excess low": excess_low, "excess high": excess_high})
    return pd.DataFrame(table).set_index("class")


def spearman(a, b):
    """Rank correlation: 1 = same order, 0 = unrelated, -1 = reversed. Ranks + ordinary correlation."""
    both = pd.concat([a, b], axis=1).dropna()
    return both.iloc[:, 0].rank().corr(both.iloc[:, 1].rank()), len(both)


def main():
    args = parse_args()
    df = pd.read_csv(args.csv, dtype={"name": str}, keep_default_na=False)
    if args.limit:
        df = df.head(args.limit)

    rows = per_photo_rows(df, args)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rows.to_csv(args.out, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.3f}".format)

    print(f"{args.heatmaps}: {rows['name'].nunique()} photos\n")
    print("Heat ranking, all photos (sorted by excess heat; 95% intervals from the bootstrap).")
    print("heat ratio > 1 = more heat per pixel than average; excess heat > 0 = more than a same-size region:")
    table = ranking(rows).sort_values("excess heat", ascending=False)
    print(table.to_string(), "\n")

    # Excess heat per class (rows) and continent (columns), like clue_ranking.py
    labels = {}
    for continent, group in rows.groupby("continent"):
        photos = (df["continent"] == continent).sum()
        if photos < args.min_photos:
            continue
        t = ranking(group)
        t = t[t["photos"] >= args.min_class_photos]
        # * = the 95% interval does not include 0
        significant = (t["excess low"] > 0) | (t["excess high"] < 0)
        labels[f"{continent} ({photos})"] = pd.Series(
            [f"{v:.4f}{'*' if s else ''}" for v, s in zip(t["excess heat"], significant)], index=t.index)
    print(f"Excess heat per continent (photos in brackets; * = interval does not include 0;")
    print(f"empty = class in fewer than {args.min_class_photos} photos):")
    print(pd.DataFrame(labels).reindex(CLASS_NAMES).fillna("").to_string(), "\n")

    if args.blur:
        blur = effects(pd.read_csv(args.blur, dtype={"name": str}, keep_default_na=False),
                       args.min_area).set_index("class")
        print(f"Compared with the blur ranking ({args.blur}):")
        both = pd.DataFrame({"blur clue effect": blur["clue effect"], "excess heat": table["excess heat"],
                             "heat ratio": table["heat ratio"]})
        both["blur rank"] = both["blur clue effect"].rank(ascending=False).astype(int)
        both["excess rank"] = both["excess heat"].rank(ascending=False).astype(int)
        print(both.sort_values("blur rank").to_string(), "\n")
        for column in ["excess heat", "heat ratio"]:
            rho, n = spearman(both["blur clue effect"], both[column])
            print(f"Spearman, blur clue effect vs {column}: {rho:.2f} ({n} classes)")


if __name__ == "__main__":
    main()

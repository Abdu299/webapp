"""
Turn the class_blur.py rows into the clue ranking: which clue classes give the location away.

For each class, only photos that contain it are used (blurring a class that is not
there cannot move the guess). For those photos:
- blur damage:   how much worse the guess gets when the class is blurred
                 osv5m: extra km of error. geovit: drop in the probability of its original country.
- random damage: the same when a random region of the same area is blurred
- clue effect:   blur damage - random damage. The ranking uses this number, so a
                 class does not rank high just because it covers a lot of the photo.
- ok:            share of photos still right (osv5m: within 750 km, geovit: right country)
Which model made the CSV is seen from its columns (km = osv5m, p_guess = geovit).

Printed overall and per continent. No GPU needed, run on the login node:
    python clue_ranking.py /fp/projects01/ec674/williaan/results/class_blur_im2gps.csv \
        | tee /fp/projects01/ec674/williaan/results/clue_ranking_im2gps.txt
"""

import argparse

import numpy as np
import pandas as pd

from clue_classes import CLUE_CLASSES

CLASS_NAMES = [name for _, name, _ in CLUE_CLASSES]
# Same distance thresholds as Victoria's results (street, city, region, country, continent)
THRESHOLDS = [1, 25, 200, 750, 2500]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    # A class counts as present if it covers at least this share of the square the model sees.
    # 0.001 = about 200 pixels of 448x448, so a few stray pixels do not count as a sign.
    p.add_argument("--min-area", type=float, default=0.001)
    # Leave out continents with fewer photos than this from the per-continent table
    p.add_argument("--min-photos", type=int, default=10)
    # In the per-continent table, leave out a class found in fewer photos than this:
    # a mean over a handful of photos is mostly noise
    p.add_argument("--min-class-photos", type=int, default=20)
    return p.parse_args()


def confidence_interval(values, repeats=1000, seed=0):
    """95% bootstrap interval of the mean: resample the photos with replacement many times
    and keep the middle 95% of the means. A few photos with huge km changes can move a mean
    a lot, so this shows which differences are bigger than chance."""
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(values), size=(repeats, len(values)))
    means = values[samples].mean(axis=1)
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def damage_and_ok(df):
    """Per row: damage = how much worse than the photo's original version, ok = still right."""
    original = df[df["version"] == "original"].set_index("name")
    if "km" in df.columns:
        damage = df["km"] - df["name"].map(original["km"])
        ok = df["km"] <= 750                        # country level, like Victoria's thresholds
    else:
        damage = df["name"].map(original["p_guess"]) - df["p_guess"]
        ok = df["correct"].astype(str) == "True"    # read back from the CSV as text
    return df.assign(damage=damage, ok=ok)


def effects(df, min_area):
    """One row per class: photos, mean area, blur/random damage, clue effect with its
    95% interval, share still right."""
    df = damage_and_ok(df)
    original = df[df["version"] == "original"].set_index("name")

    rows = []
    for class_name in CLASS_NAMES:
        blur = df[(df["version"] == "blur") & (df["class"] == class_name)].set_index("name")
        random = df[(df["version"] == "random") & (df["class"] == class_name)].set_index("name")

        present = blur.index[blur["area"] >= min_area]
        if len(present) == 0:
            continue
        blur, random = blur.loc[present], random.loc[present]

        # Clue effect per photo: how much more the guess suffers with the class blurred
        # than with the same area blurred at random
        per_photo = (blur["damage"] - random["damage"]).to_numpy()
        low, high = confidence_interval(per_photo)

        rows.append({
            "class": class_name,
            "photos": len(present),
            "area": blur["area"].mean(),
            "blur damage": blur["damage"].mean(),
            "random damage": random["damage"].mean(),
            "clue effect": per_photo.mean(),
            "95% low": low,
            "95% high": high,
            "ok before": original.loc[present, "ok"].mean(),
            "ok blur": blur["ok"].mean(),
            "ok random": random["ok"].mean(),
        })
    return pd.DataFrame(rows).sort_values("clue effect", ascending=False)


def main():
    args = parse_args()
    df = pd.read_csv(args.csv, dtype={"name": str}, keep_default_na=False)
    original = df[df["version"] == "original"]

    km = "km" in df.columns
    unit = "km" if km else "probability"
    print(f"{args.csv}: {len(original)} photos, model: {'OSV-5M baseline' if km else 'GeoVit'}\n")

    if km:
        print("Geolocation without blur (share of photos within each distance):")
        for t in THRESHOLDS:
            print(f"  within {t:>4} km: {(original['km'] <= t).mean():6.1%}")
        print(f"  median error: {original['km'].median():.0f} km\n")
        print("damage = extra km of error, ok = within 750 km\n")
    else:
        print(f"Country accuracy without blur: {(original['correct'].astype(str) == 'True').mean():.1%}")
        print(f"Mean probability of its own guess: {original['p_guess'].mean():.3f}\n")
        print("damage = drop in the probability of its original country guess, ok = right country\n")

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}" if abs(v) < 1 else f"{v:.0f}")

    print("Clue ranking, all photos (highest clue effect = gives the location away most).")
    print("If the 95% interval does not include 0, the effect is bigger than chance:")
    print(effects(df, args.min_area).to_string(index=False), "\n")

    # Clue effect per class (rows) and continent (columns)
    columns, labels = {}, {}
    for continent, group in df.groupby("continent"):
        photos = (group["version"] == "original").sum()
        if photos < args.min_photos:
            continue
        table = effects(group, args.min_area).set_index("class")
        table = table[table["photos"] >= args.min_class_photos]

        name = f"{continent} ({photos})"
        columns[name] = table["clue effect"]
        # * = the 95% interval does not include 0
        significant = (table["95% low"] > 0) | (table["95% high"] < 0)
        labels[name] = [(f"{effect:.0f}" if km else f"{effect:.3f}") + ("*" if sig else "")
                        for effect, sig in zip(table["clue effect"], significant)]
        labels[name] = pd.Series(labels[name], index=table.index)

    per_continent = pd.DataFrame(columns).reindex(CLASS_NAMES)
    print(f"Clue effect ({unit}) per continent (photos in brackets; * = bigger than chance;")
    print(f"empty = class in fewer than {args.min_class_photos} photos):")
    print(pd.DataFrame(labels).reindex(CLASS_NAMES).fillna("").to_string(), "\n")

    print("Top 3 clues per continent:")
    for continent, effect in per_continent.items():
        print(f"  {continent}: {', '.join(effect.dropna().sort_values(ascending=False).index[:3])}")


if __name__ == "__main__":
    main()

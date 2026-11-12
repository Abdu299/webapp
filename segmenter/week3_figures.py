"""
Week 3 figures, made from the class_blur.py results (no GPU needed, runs on the login node):

1. week3_ranking_<set>.png: clue effect per class with its 95% interval (the clue ranking as a chart)
2. week3_blur_examples_<set>.png: a few photos with their mask, the model's top clue blurred, a random
   region of the same size blurred, and a second clue blurred, each with the model's guess

--model osv5m (default): OSV-5M baseline, km. Blurs vegetation and signs in the examples.
--model geovit: Abdu's country model, drop in probability. Blurs road and road markings in the
                examples (its two top clues). Files get "_geovit" in their names.

The values in figure 2 are the ones from the experiment. The random regions are made again
exactly as class_blur.py made them (same seed, same order), so the picture shows the region
that was really tested.

Run from the segmenter folder:
    python week3_figures.py
    python week3_figures.py --model geovit
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from class_blur import centre_square, dilate, gaussian_blur, random_region
from clue_classes import CLUE_CLASSES
from clue_ranking import damage_and_ok, effects
from visualize import colorize

# Bar colours by what the interval says. Red/blue are opposite poles, grey = no clear effect.
CLUE = "#e34948"
LESS_THAN_RANDOM = "#2a78d6"
UNCLEAR = "#b5b4ae"
TEXT = "#52514e"

CLASS_ID = {name: i for i, name, _ in CLUE_CLASSES}

# Per model: name in titles, how a clue effect is printed, axis text, and the two classes in the examples
MODELS = {
    "osv5m": {"name": "OSV-5M baseline", "file": "class_blur_{}.csv", "suffix": "",
              "value": "{:+.0f} km",
              "xlabel": "Clue effect (km): extra error when the class is blurred,\n"
                        "minus the extra error when a random region of the same size is blurred",
              "examples": ("vegetation_and_terrain", "signs_and_text")},
    "geovit": {"name": "GeoVit (Abdu)", "file": "class_blur_geovit_{}.csv", "suffix": "_geovit",
               "value": "{:+.3f}",
               "xlabel": "Clue effect: drop in the probability of its country guess when the class is blurred,\n"
                         "minus the drop when a random region of the same size is blurred",
               "examples": ("road", "road_markings")},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--set", default="osv5m")
    p.add_argument("--model", choices=list(MODELS), default="osv5m")
    p.add_argument("--results", default="/fp/projects01/ec674/williaan/results")
    p.add_argument("--data", default="/fp/projects01/ec674/williaan/data/test_sets")
    p.add_argument("--masks", default="/fp/projects01/ec674/williaan/masks")
    p.add_argument("--out", default="/fp/projects01/ec674/williaan/figures")
    p.add_argument("--num-examples", type=int, default=3)
    # Must be the same as the class_blur.py run, so the figure shows what was tested
    p.add_argument("--work-size", type=int, default=448)
    p.add_argument("--sigma", type=float, default=40)
    p.add_argument("--dilate", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def pretty(class_name):
    return class_name.replace("_and_", " & ").replace("_", " ")


def bar_colours(table):
    """Red = clue (interval above 0), blue = hurts less than random (below 0), grey = unclear."""
    return [CLUE if low > 0 else LESS_THAN_RANDOM if high < 0 else UNCLEAR
            for low, high in zip(table["95% low"], table["95% high"])]


def draw_ranking(ax, table, value_format, with_photos=True):
    """Horizontal clue effect bars with 95% intervals and a value next to each bar, in the given row order."""
    labels = [f"{pretty(name)}  ({photos} photos)" if with_photos else pretty(name)
              for name, photos in zip(table["class"], table["photos"])]
    y = np.arange(len(table))
    effect = table["clue effect"].to_numpy()
    error = [effect - table["95% low"].to_numpy(), table["95% high"].to_numpy() - effect]

    ax.barh(y, effect, color=bar_colours(table), height=0.6, edgecolor="white", linewidth=2)
    ax.errorbar(effect, y, xerr=error, fmt="none", ecolor=TEXT, elinewidth=1.2, capsize=3)
    ax.axvline(0, color=TEXT, linewidth=1)

    # Space around the bars scales with the data, so the chart works for any range
    low_end, high_end = table["95% low"].min(), table["95% high"].max()
    gap = 0.02 * (high_end - low_end)

    # Value next to each bar, outside the interval so it never overlaps the error bar
    for yi, value, low, high in zip(y, effect, table["95% low"], table["95% high"]):
        x, align = (high + gap, "left") if value >= 0 else (low - gap, "right")
        ax.text(x, yi, value_format.format(value), va="center", ha=align, fontsize=9, color=TEXT)

    ax.set_yticks(y, labels, fontsize=9)
    ax.grid(axis="x", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.set_xlim(min(low_end, 0) - 14 * gap, max(high_end, 0) + 14 * gap)


def legend_handles():
    return [Patch(color=CLUE, label="clue: hurts more than random"),
            Patch(color=UNCLEAR, label="no clear difference from random"),
            Patch(color=LESS_THAN_RANDOM, label="hurts less than random")]


def ranking_figure(rows, path, title, model):
    table = effects(rows, min_area=0.001).sort_values("clue effect")

    fig, ax = plt.subplots(figsize=(9, 5.2))
    draw_ranking(ax, table, model["value"])
    ax.set_xlabel(model["xlabel"], fontsize=9)
    ax.set_title(title, fontsize=11, loc="left")
    ax.legend(handles=legend_handles(), loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print("saved", path)


def pick_examples(rows, n, model, km):
    """Photos the model gets right that contain both example classes, with the biggest clue effect of the
    first class, one per continent so the examples are spread out."""
    first, second = model["examples"]
    rows = damage_and_ok(rows)
    wide = rows.pivot_table(index="name", columns=["version", "class"], values=["damage", "area"], aggfunc="first")
    original = rows[rows["version"] == "original"].set_index("name")
    right = original["km"] <= 200 if km else original["ok"]
    candidates = pd.DataFrame({
        "continent": original["continent"],
        "right": right,
        "first area": wide[("area", "blur", first)],
        "second area": wide[("area", "blur", second)],
        "effect": wide[("damage", "blur", first)] - wide[("damage", "random", first)],
    })
    strict = candidates[candidates["right"] & (candidates["first area"] >= 0.15)
                        & (candidates["second area"] >= 0.005)]
    # Fall back to any photo it gets right if too few have a lot of both classes
    if len(strict) >= n:
        candidates = strict
    elif candidates["right"].any():
        candidates = candidates[candidates["right"]]
    best = candidates.sort_values("effect", ascending=False).groupby("continent").head(1)
    return best.head(n).index.tolist()


def replay_regions(test_set, names, args):
    """Make the class and random regions of the chosen photos exactly as class_blur.py did.
    The random regions come from one generator shared by all photos in order, so every photo
    before the chosen ones is replayed too (masks only, no model)."""
    generator = torch.Generator().manual_seed(args.seed)
    last = max(test_set.index[test_set["name"].isin(names)])
    regions = {}

    for photo in test_set.loc[:last].itertuples():
        mask = centre_square(Image.open(os.path.join(args.masks, args.set, photo.name + ".png")),
                             args.work_size, TF.InterpolationMode.NEAREST)
        mask = torch.from_numpy(np.array(mask))

        for class_id, _, _ in CLUE_CLASSES:
            if photo.name in names:
                region = dilate((mask == class_id).float(), args.dilate)
                regions[(photo.name, class_id, "blur")] = region
                regions[(photo.name, class_id, "random")] = random_region(
                    region.mean().item(), args.work_size, generator, "cpu")
            elif (mask == class_id).any():
                # Other photos: only use up the same random numbers random_region would
                # (it draws once when the class is there, and not at all when it is not)
                torch.rand(1, 1, args.work_size // 32, args.work_size // 32, generator=generator)
        if photo.name in names:
            regions[(photo.name, "mask")] = mask
    return regions


def guess_text(rows, km):
    """Text for each (photo, version, class): the error in km, or the country guess and how sure it is."""
    rows = rows.set_index(["name", "version", "class"])
    if km:
        return {key: f"error {value:,.0f} km" for key, value in rows["km"].items()}
    texts = {}
    for key, row in rows.iterrows():
        original = rows.loc[(key[0], "original", "")]
        if row["pred_country"] == original["pred_country"]:
            texts[key] = f"{row['pred_country']}, {row['p_guess']:.0%} sure"
        else:                               # it switched country: show the new guess too
            texts[key] = f"now {row['pred_country']} ({original['pred_country']} {row['p_guess']:.0%})"
    return texts


def examples_figure(rows, test_set, path, model, km, args):
    first, second = model["examples"]
    names = pick_examples(rows, args.num_examples, model, km)
    regions = replay_regions(test_set, names, args)
    texts = guess_text(rows[rows["name"].isin(names)], km)
    original = rows[rows["version"] == "original"].set_index("name")

    columns = ["photo", "clue mask", f"{pretty(first)} blurred", "random region,\nsame size",
               f"{pretty(second)} blurred"]
    fig, axes = plt.subplots(len(names), len(columns), figsize=(3 * len(columns), 3.2 * len(names)), squeeze=False)

    for row, name in enumerate(names):
        photo = test_set[test_set["name"] == name].iloc[0]
        image = centre_square(Image.open(photo.path).convert("RGB"), args.work_size, TF.InterpolationMode.BICUBIC)
        x = TF.to_tensor(image)
        blurred = gaussian_blur(x, args.sigma)

        def blur_with(region):
            return (region * blurred + (1 - region) * x).permute(1, 2, 0).numpy()

        panels = [
            (x.permute(1, 2, 0).numpy(), texts[(name, "original", "")]),
            (colorize(regions[(name, "mask")]), None),
            (blur_with(regions[(name, CLASS_ID[first], "blur")]), texts[(name, "blur", first)]),
            (blur_with(regions[(name, CLASS_ID[first], "random")]), texts[(name, "random", first)]),
            (blur_with(regions[(name, CLASS_ID[second], "blur")]), texts[(name, "blur", second)]),
        ]
        for col, (panel, text) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(panel)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(columns[col], fontsize=11)
            if text is not None:
                ax.text(0.03, 0.04, text, transform=ax.transAxes, fontsize=10,
                        color="black", bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=2))
        label = original.loc[name, "continent"]
        if not km:
            label += f"\ntrue: {original.loc[name, 'country']}"
        axes[row, 0].set_ylabel(label, fontsize=11)

    legend = [Patch(facecolor=np.array(color) / 255, edgecolor="black", linewidth=0.5, label=name)
              for _, name, color in CLUE_CLASSES]
    fig.legend(handles=legend, loc="lower center", ncol=5, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=150)
    print("saved", path, "photos:", names)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    model = MODELS[args.model]

    rows = pd.read_csv(os.path.join(args.results, model["file"].format(args.set)), dtype={"name": str},
                       keep_default_na=False)
    test_set = pd.read_csv(os.path.join(args.data, f"{args.set}.csv"), dtype={"name": str}, keep_default_na=False)
    photos = (rows["version"] == "original").sum()
    km = "km" in rows.columns

    set_name = {"osv5m": "OSV-5M", "im2gps": "Im2GPS"}.get(args.set, args.set)
    title = f"Which clues give the location away? {model['name']}, {photos:,} {set_name} photos"
    suffix = model["suffix"]
    ranking_figure(rows, os.path.join(args.out, f"week3_ranking{suffix}_{args.set}.png"), title, model)
    examples_figure(rows, test_set, os.path.join(args.out, f"week3_blur_examples{suffix}_{args.set}.png"),
                    model, km, args)


if __name__ == "__main__":
    main()

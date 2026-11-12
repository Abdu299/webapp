"""
Week 4 figures, made from the heatmaps and heatmap_ranking.py results (no GPU, login node):

1. week4_heat_ranking_<set>.png: excess heat per class with its 95% interval, one bar per model
2. week4_heatmap_examples_<set>.png: a few photos with their clue mask and each model's heatmap.
   Also the check that masks and heatmaps line up: the heat should sit on real objects.
3. week4_model_comparison_<set>.png: blur clue effect (week 3 method) and excess heat (week 4 method)
   for each model, in the week 3 style (red = clue, grey = unclear, blue = less than a random region).
   Same class order in every panel, so differences between models and methods are easy to spot.

Uses every model whose files exist (OSV-5M baseline, Abdu's GeoVit).

Run from the segmenter folder:
    python week4_figures.py
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

from class_blur import centre_square, gaussian_blur
from clue_classes import CLUE_CLASSES
from clue_ranking import effects
from heatmap_ranking import ranking
from week3_figures import MODELS as BLUR_MODELS, draw_ranking, legend_handles
from visualize import colorize

ROOT = "/fp/projects01/ec674/williaan"
# (key, name in the figure, heatmap folder, per-photo ratio file, bar colour, how to print the max heat)
MODELS = [
    ("osv5m", "OSV-5M baseline", "heatmaps", "heatmap_ratio_{}.csv", "#2a78d6", "max {:,.0f} km"),
    ("geovit", "GeoVit (Abdu)", "heatmaps_geovit", "heatmap_ratio_geovit_{}.csv", "#eb6834", "max drop {:.3f}"),
]
HEAT = np.array([227, 73, 72]) / 255     # the "clue" red from week3_figures.py
TEXT = "#52514e"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--set", default="osv5m")
    p.add_argument("--root", default=ROOT)                   # data, masks, heatmaps, results live here
    p.add_argument("--num-examples", type=int, default=4)
    p.add_argument("--names", nargs="*", default=[])         # pick the photos yourself
    p.add_argument("--work-size", type=int, default=448)     # same square as heatmaps.py
    return p.parse_args()


def available_models(args):
    """The models that have both heatmaps and a ratio file for this test set."""
    found = []
    for key, name, folder, ratio_file, colour, max_label in MODELS:
        heat_dir = f"{args.root}/{folder}/{args.set}"
        ratio_path = f"{args.root}/results/{ratio_file.format(args.set)}"
        if os.path.isdir(heat_dir) and os.path.exists(ratio_path):
            rows = pd.read_csv(ratio_path, dtype={"name": str}, keep_default_na=False)
            found.append({"key": key, "name": name, "heat_dir": heat_dir, "rows": rows, "colour": colour,
                          "max_label": max_label})
        else:
            print(f"skipped {name}: no {heat_dir} or {ratio_path}")
    return found


def pretty(class_name):
    return class_name.replace("_and_", " & ").replace("_", " ")


def comparison_figure(models, path, title, args):
    """Rows: blur (week 3) and heat (week 4). Columns: models. Classes in the order of the first model's blur."""
    blur = {}
    for model in models:
        blur_path = f"{args.root}/results/{BLUR_MODELS[model['key']]['file'].format(args.set)}"
        if os.path.exists(blur_path):
            rows = pd.read_csv(blur_path, dtype={"name": str}, keep_default_na=False)
            blur[model["key"]] = effects(rows, min_area=0.001).set_index("class")
    if not blur:
        print("skipped the comparison: no class_blur results")
        return
    first = blur.get(models[0]["key"], next(iter(blur.values())))
    order = first.sort_values("clue effect").index      # biggest clue of the first model at the top

    fig, axes = plt.subplots(2, len(models), figsize=(6.5 * len(models), 10), squeeze=False)
    for col, model in enumerate(models):
        # Top: blur clue effect, in the model's own unit
        ax = axes[0, col]
        if model["key"] in blur:
            draw_ranking(ax, blur[model["key"]].reindex(order).reset_index(), BLUR_MODELS[model["key"]]["value"],
                         with_photos=False)
        ax.set_title(f"{model['name']}: blur one class (week 3)", fontsize=11, loc="left")
        ax.set_xlabel("clue effect (km)" if model["key"] == "osv5m" else "clue effect (probability drop)", fontsize=9)

        # Bottom: excess heat, same colours (its interval decides red / grey / blue)
        heat = ranking(model["rows"]).rename(columns={"excess heat": "clue effect", "excess low": "95% low",
                                                       "excess high": "95% high"})
        draw_ranking(axes[1, col], heat.reindex(order).reset_index(), "{:+.3f}", with_photos=False)
        axes[1, col].set_title(f"{model['name']}: heatmaps (week 4)", fontsize=11, loc="left")
        axes[1, col].set_xlabel("excess heat (share of the heatmap)", fontsize=9)

    fig.suptitle(title, fontsize=13, x=0.01, ha="left")
    fig.legend(handles=legend_handles(), loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(path, dpi=200)
    print("saved", path)


def ranking_figure(models, path, title):
    tables = [ranking(m["rows"]) for m in models]
    order = tables[0].sort_values("excess heat").index        # sorted by the first model

    y = np.arange(len(order))
    height = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, (model, table) in enumerate(zip(models, tables)):
        t = table.reindex(order)
        offset = (i - (len(models) - 1) / 2) * height
        effect = t["excess heat"].to_numpy()
        error = [effect - t["excess low"].to_numpy(), t["excess high"].to_numpy() - effect]
        ax.barh(y - offset, effect, height=height, color=model["colour"], edgecolor="white", linewidth=2,
                label=model["name"])
        ax.errorbar(effect, y - offset, xerr=error, fmt="none", ecolor=TEXT, elinewidth=1, capsize=2)

    ax.axvline(0, color=TEXT, linewidth=1)
    ax.set_yticks(y, [pretty(c) for c in order], fontsize=9)
    ax.set_xlabel("Excess heat: extra share of the heatmap on the class,\n"
                  "compared with an average region of the same size (0 = no clue)", fontsize=9)
    ax.set_title(title, fontsize=11, loc="left", pad=24)     # pad leaves room for the legend below it
    ax.grid(axis="x", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    # Between the title and the plot, so it never covers a bar
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1), ncol=len(models), fontsize=9, frameon=False,
              borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print("saved", path)


def pick_examples(rows, n):
    """Photos with many clue classes, including signs, one per continent so they're spread out."""
    present = rows.groupby("name").agg(continent=("continent", "first"), classes=("class", "nunique"))
    signs = rows[(rows["class"] == "signs_and_text") & (rows["area"] >= 0.005)]["name"]
    present = present[present.index.isin(signs)]
    best = present.sort_values("classes", ascending=False).groupby("continent").head(1)
    return best.head(n).index.tolist()


def heat_overlay(gray, heat):
    """Grey photo with red on top: the more the guess changes there, the stronger the red."""
    # Light smoothing for display only: the heat is measured in 16 px steps, so this hides nothing real
    heat = gaussian_blur(torch.from_numpy(heat)[None, None], 8)[0, 0].numpy()
    # Scale by the 99th percentile, so one very hot spot doesn't make the rest invisible
    strength = np.clip(heat / max(np.percentile(heat, 99), 1e-12), 0, 1)[..., None] * 0.75
    return (1 - strength) * gray + strength * HEAT


def examples_figure(models, test_set, names, path, args):
    columns = ["photo", "clue mask"] + [f"heatmap:\n{m['name']}" for m in models]
    fig, axes = plt.subplots(len(names), len(columns), figsize=(3 * len(columns), 3.2 * len(names)), squeeze=False)

    for row, name in enumerate(names):
        photo = test_set[test_set["name"] == name].iloc[0]
        # Same square as heatmaps.py and heatmap_ranking.py, so all panels show the same pixels
        image = np.array(centre_square(Image.open(photo.path).convert("RGB"), args.work_size,
                                       TF.InterpolationMode.BICUBIC)) / 255
        mask = np.array(centre_square(Image.open(os.path.join(args.root, "masks", args.set, name + ".png")),
                                      args.work_size, TF.InterpolationMode.NEAREST))
        gray = image.mean(axis=2, keepdims=True).repeat(3, axis=2)

        panels = [image, 0.4 * image + 0.6 * colorize(torch.from_numpy(mask)) / 255]
        notes = [None, None]                # text in the corner of each panel
        for model in models:
            heat = np.load(os.path.join(model["heat_dir"], name + ".npy")).astype(np.float32)
            panels.append(heat_overlay(gray, heat))
            notes.append(model["max_label"].format(heat.max()))     # how strong the hottest spot is

        for col, (panel, note) in enumerate(zip(panels, notes)):
            ax = axes[row, col]
            ax.imshow(np.clip(panel, 0, 1))
            ax.set_xticks([])
            ax.set_yticks([])
            if note:
                ax.text(0.03, 0.04, note, transform=ax.transAxes, fontsize=10, color="black",
                        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=2))
            if row == 0:
                ax.set_title(columns[col], fontsize=11)
        axes[row, 0].set_ylabel(photo.continent, fontsize=11)

    legend = [Patch(facecolor=np.array(color) / 255, edgecolor="black", linewidth=0.5, label=pretty(name))
              for _, name, color in CLUE_CLASSES]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.03), ncol=5, fontsize=9, frameon=False)
    fig.text(0.5, 0.005, "Heatmaps: stronger red = the guess changes more when that spot is blurred",
             ha="center", va="bottom", fontsize=9, color=TEXT)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path, dpi=150)
    print("saved", path, "photos:", names)


def main():
    args = parse_args()
    out = os.path.join(args.root, "figures")
    os.makedirs(out, exist_ok=True)

    models = available_models(args)
    if not models:
        raise SystemExit("no heatmaps + ratio files found; run heatmaps.py and heatmap_ranking.py first")
    test_set = pd.read_csv(os.path.join(args.root, "data", "test_sets", f"{args.set}.csv"), dtype={"name": str}, keep_default_na=False)

    set_name = {"osv5m": "OSV-5M", "im2gps": "Im2GPS"}.get(args.set, args.set)
    photos = models[0]["rows"]["name"].nunique()
    ranking_figure(models, os.path.join(out, f"week4_heat_ranking_{args.set}.png"),
                   f"Where do the models look? Heatmaps of {photos:,} {set_name} photos")

    comparison_figure(models, os.path.join(out, f"week4_model_comparison_{args.set}.png"),
                      f"Which clues give the location away? Two models, two methods, {photos:,} {set_name} photos", args)

    names = args.names or pick_examples(models[0]["rows"], args.num_examples)
    examples_figure(models, test_set, names, os.path.join(out, f"week4_heatmap_examples_{args.set}.png"), args)


if __name__ == "__main__":
    main()

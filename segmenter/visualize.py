"""
Save example figures
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import numpy as np

import torch

from clue_classes import CLUE_CLASSES, IGNORE_INDEX
from dataset import IMAGENET_MEAN, IMAGENET_STD, VistasClueDataset
from evaluate import MODELS, NUM_CLASSES
from mask2former_eval import Mask2FormerClues

# Focusing on signs_and_text and poles
FOCUS_CLASSES = [2, 3]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=os.environ.get("VISTAS_ROOT", "/fp/projects01/ec674/data/vistas"))
    p.add_argument("--checkpoints", default="/fp/projects01/ec674/williaan/checkpoints")
    p.add_argument("--models", nargs="+", default=["linear_seed0", "segmenter_weighted_seed0"])
    p.add_argument("--out", default="/fp/projects01/ec674/williaan/figures/week2_examples.png")
    p.add_argument("--num-images", type=int, default=4)
    # How many validation images to search through for many poles and signs
    p.add_argument("--search", type=int, default=300)
    return p.parse_args()




def colorize(mask):
    """
    [H, W] clue ids -> [H, W, 3] RGB image with ignored pixels as black
    """
    palette = np.zeros((256, 3), dtype=np.uint8)

    # Each clue class gets color from CLUE_CLASSES
    for class_id, clue_name, color in CLUE_CLASSES:
        palette[class_id] = color

    # look up every pixel at once
    return palette[mask.numpy()]

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_set = VistasClueDataset(args.root, "validation")
    val_set

    # Share of pole + sign pixels in each of the first args.search images
    share = []
    for i in range(args.search):
        _, mask = val_set[i]
        share.append(np.isin(mask.numpy(), FOCUS_CLASSES).mean())

    ids = np.argsort(share)[::-1][:args.num_images]
    print("images:", ids.tolist())



    models = {}
    for name in args.models:
        if name == "mask2former":
            config_path = os.path.join(args.root, "config_v1.2.json")
            model = Mask2FormerClues(config_path=config_path, scale=2).to(device).eval()
        else:
            model_type = name.split("_")[0]
            model = MODELS[model_type](num_classes=NUM_CLASSES).to(device).eval()

            path = os.path.join(args.checkpoints, name + ".pt")

            model.decoder.load_state_dict(torch.load(path, map_location=device))
        models[name] = model


    columns = ["photo", "ground truth"] + list(models)
    fig, axes = plt.subplots(len(ids), len(columns), figsize=(3.2*len(columns), 2.5*len(ids)), squeeze=False)

    mean = torch.tensor(IMAGENET_MEAN).view(3,1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    for row, i in enumerate(ids):
        image, mask = val_set[int(i)]

        # Undo the normalisation, then [3, H, W] -> [H, W, 3] which matplotlib expects
        photo = (image * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        panels = [photo, colorize(mask)]

        with torch.no_grad():
            for model in models.values():
                # image[None] adds a batch dimension: [3, H, W] -> [1, 3, H, W]
                pred = model(image[None].to(device)).argmax(dim=1)[0].cpu()
                # Black out unlabelled pixels like in the ground truth, so only scored pixels are compared
                pred[mask == IGNORE_INDEX] = IGNORE_INDEX
                panels.append(colorize(pred))

        for col, panel in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(panel)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(columns[col], fontsize=10)

    # One coloured box per class at the bottom
    legend = [Patch(facecolor=np.array(color) / 255, edgecolor="black", linewidth=0.5, label=name)
              for _, name, color in CLUE_CLASSES]
    fig.legend(handles=legend, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))   # leave room at the bottom for the legend

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print("saved", args.out)


if __name__ == "__main__":
    main()
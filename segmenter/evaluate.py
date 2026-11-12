"""
Evaluate a trained decoder on the Vistas validation set, per clue class.

Pixel accuracy (printed by train.py) is dominated by big classes like road and
sky. This script reports IoU for every clue class, so we can see whether small
classes like signs and poles are actually found.

Run from the segmenter folder:
    python evaluate.py --checkpoint /fp/projects01/ec674/williaan/checkpoints/linear_seed0.pt
    python evaluate.py --model segmenter --checkpoint .../segmenter_seed0.pt
Quick test on a few images (works on CPU):
    python evaluate.py --checkpoint linear_seed0.pt --limit 8 --num-workers 0
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader, Subset

from clue_classes import CLUE_CLASSES, IGNORE_INDEX
from dataset import VistasClueDataset
from model import LinearSegmenter, MaskSegmenter

NUM_CLASSES = len(CLUE_CLASSES)

# The --model name -> model class. train.py uses the same table, so a name always
# means the same model, and the name is also the start of the checkpoint file name.
MODELS = {"linear": LinearSegmenter, "segmenter": MaskSegmenter}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=os.environ.get("VISTAS_ROOT", "/fp/projects01/ec674/data/vistas"))
    p.add_argument("--checkpoint", required=True)
    # Must match the model the checkpoint was trained with.
    p.add_argument("--model", choices=MODELS, default="linear")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=8)
    # Use only the first N validation images, for quick tests. 0 = all.
    p.add_argument("--limit", type=int, default=0)
    # Must match the image size the checkpoint was trained with (width height).
    p.add_argument("--size", type=int, nargs=2, default=[448, 336])
    return p.parse_args()


def update_table(table, preds, masks):
    """Add one batch of predictions to the confusion table, in place.
    Shared with train.py so the IoU code exists in only one place."""
    # Keep only labelled pixels
    valid = masks != IGNORE_INDEX

    # One number per true, pred pair and then count all the 100 pairs at once
    pairs = masks[valid] * NUM_CLASSES + preds[valid]
    table += torch.bincount(pairs, minlength=NUM_CLASSES ** 2).view(NUM_CLASSES, NUM_CLASSES)


def confusion_matrix(model, loader, device):
    """Return a [NUM_CLASSES, NUM_CLASSES] table where entry [t, p] counts the
    pixels whose true class is t and predicted class is p."""
    model.eval()

    # Start with a zero table
    table = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)

    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)

            scores = model(images)
            preds = scores.argmax(dim=1)

            update_table(table, preds, masks)
    return table


def per_class_iou(table):
    """IoU of each class from the confusion matrix, as a tensor of NUM_CLASSES."""
    table = table.float()
    correct = table.diag()
    predicted = table.sum(dim=0)
    true_c = table.sum(dim=1)

    return correct / (true_c + predicted - correct)


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)

    val_set = VistasClueDataset(args.root, "validation", size=tuple(args.size))
    if args.limit:
        val_set = Subset(val_set, range(args.limit))
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Load decoder
    model = MODELS[args.model](num_classes=NUM_CLASSES).to(device)
    model.decoder.load_state_dict(torch.load(args.checkpoint, map_location=device))

    table = confusion_matrix(model, val_loader, device)
    iou = per_class_iou(table)
    share = table.sum(dim=1).float() / table.sum()

    # printing results
    print(f"{'class':<28} {'IoU':>6} {'pixels':>7}")
    for (class_id, name, _), class_iou, class_share in zip(CLUE_CLASSES, iou, share):
        print(f"{name:<28} {class_iou.item():6.3f} {class_share.item():7.1%}")

    print(f"mean IoU:       {iou.nanmean().item():.4f}")
    print(f"pixel accuracy: {(table.diag().sum() / table.sum()).item():.4f}")

if __name__ == "__main__":
    main()

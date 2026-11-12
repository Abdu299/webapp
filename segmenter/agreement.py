"""
Does our segmenter still work on the week 3 photos, which have no labels?

Mask2Former (0.815 mIoU on Vistas, ours 0.710) is used as the "correct answer": on every
photo both models predict clue classes, and IoU is computed between the two, with
exactly the same code as evaluate.py.

Mask2Former is not perfect either, so the agreement is never 1.0. That is why it is also
measured on Vistas validation photos: if the agreement on OSV-5M or Im2GPS is about the
same as on Vistas, our segmenter works as well there as on Vistas. If a class drops a
lot, our masks for that class are worse on those photos (domain shift).

Run from the segmenter folder:
    python agreement.py vistas --limit 500
    python agreement.py /fp/projects01/ec674/williaan/data/test_sets/osv5m.csv
Quick test on a few photos:
    python agreement.py vistas --limit 4 --num-workers 0
"""

import argparse
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from clue_classes import CLUE_CLASSES, IGNORE_INDEX
from dataset import VistasClueDataset
from evaluate import MODELS, NUM_CLASSES, per_class_iou, update_table
from mask2former_eval import Mask2FormerClues
from predict import load_image


def parse_args():
    p = argparse.ArgumentParser()
    # "vistas" for Vistas validation photos, or a test set CSV from test_sets.py
    p.add_argument("photos")
    p.add_argument("--root", default=os.environ.get("VISTAS_ROOT", "/fp/projects01/ec674/data/vistas"))
    # Same model and size as predict.py, so the check is about the masks class_blur.py uses
    p.add_argument("--checkpoint", default="/fp/projects01/ec674/williaan/checkpoints/segmenter_896_seed0.pt")
    p.add_argument("--model", choices=MODELS, default="segmenter")
    p.add_argument("--size", type=int, nargs=2, default=[896, 672])
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


class PhotoDataset(Dataset):
    """Photos from a test set CSV, resized and normalised like the Vistas photos."""

    def __init__(self, csv_path, size):
        self.paths = pd.read_csv(csv_path)["path"].tolist()
        self.size = size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        image, _ = load_image(self.paths[i], self.size)
        return image


class VistasPhotos(Dataset):
    """Vistas validation photos without their labels, so both test sets are treated the same."""

    def __init__(self, root, size):
        self.vistas = VistasClueDataset(root, "validation", size=size)

    def __len__(self):
        return len(self.vistas)

    def __getitem__(self, i):
        image, _ = self.vistas[i]
        return image


class Mask2FormerWithIgnore(Mask2FormerClues):
    """Mask2Former with an 11th output for the Vistas classes we ignore (camera car,
    unlabelled, ...). Without it, such pixels would get a random clue class, and we
    would count our model as wrong there. Pixels that win this class are left out."""

    def __init__(self, config_path, scale):
        super().__init__(config_path, scale)
        # A Vistas class that belongs to no clue class goes to the extra row
        ignore_row = 1 - self.to_clue.sum(dim=0, keepdim=True)
        self.to_clue = torch.cat([self.to_clue, ignore_row])


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)

    size = tuple(args.size)
    photos = VistasPhotos(args.root, size) if args.photos == "vistas" else PhotoDataset(args.photos, size)
    if args.limit:
        photos = Subset(photos, range(min(args.limit, len(photos))))
    loader = DataLoader(photos, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    ours = MODELS[args.model](num_classes=NUM_CLASSES).to(device).eval()
    ours.decoder.load_state_dict(torch.load(args.checkpoint, map_location=device))

    # Mask2Former was scored at 896x672 in week 2, so scale the photos to that size (x1 at 896x672)
    config_path = os.path.join(args.root, "config_v1.2.json")
    reference = Mask2FormerWithIgnore(config_path, scale=896 / size[0]).to(device).eval()

    # Confusion table as in evaluate.py, with Mask2Former's answer in place of the true label
    table = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
    with torch.no_grad():
        for step, images in enumerate(loader):
            images = images.to(device)
            preds = ours(images).argmax(dim=1)

            answers = reference(images).argmax(dim=1)
            answers[answers == NUM_CLASSES] = IGNORE_INDEX

            update_table(table, preds, answers)
            if step % 50 == 0:
                print(f"{step}/{len(loader)}", flush=True)

    iou = per_class_iou(table)
    share = table.sum(dim=1).float() / table.sum()

    print(f"\n{args.photos}: {len(photos)} photos, our segmenter vs Mask2Former")
    print(f"{'class':<28} {'IoU':>6} {'pixels':>7}")
    for (_, name, _), class_iou, class_share in zip(CLUE_CLASSES, iou, share):
        print(f"{name:<28} {class_iou.item():6.3f} {class_share.item():7.1%}")

    print(f"mean IoU:       {iou.nanmean().item():.4f}")
    print(f"pixel accuracy: {(table.diag().sum() / table.sum()).item():.4f}")


if __name__ == "__main__":
    main()

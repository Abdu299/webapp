"""
Train a decoder on frozen DINOv2 features on Mapillary Vistas: the one-layer
(linear) decoder or the Segmenter mask transformer, with or without class weights.

Run from the segmenter folder:
    python train.py --epochs 5 --seed 0
    python train.py --model segmenter --class-weights --epochs 10 --lr 1e-4
    python train.py --model segmenter --size 672 504 --epochs 10 --lr 1e-4
Quick test on a few images (works on CPU):
    python train.py --epochs 1 --limit 32 --num-workers 0
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from clue_classes import CLUE_CLASSES, IGNORE_INDEX
from dataset import VistasClueDataset
from evaluate import MODELS, NUM_CLASSES, per_class_iou, update_table

# Share of labelled pixels per clue class in class id order
# [road, road_markings, signs, poles, buildings, barriers, vegetation, sky, vehicles, people]
# From the week 1 validation counts; training-set counts would be slightly more correct.
PIXEL_SHARE = [0.210, 0.022, 0.013, 0.018, 0.143, 0.069, 0.174, 0.304, 0.043, 0.004]


def class_weights():
    """1/sqrt(share), scaled so the weights average 1.
    sqrt softens the gap (people ~9x sky instead of ~75x), and mean 1 keeps
    the loss about the same size, so the week 1 learning rate still fits."""
    share = torch.tensor(PIXEL_SHARE)
    weights = 1 / torch.sqrt(share)
    return weights / weights.mean()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=os.environ.get("VISTAS_ROOT", "/fp/projects01/ec674/data/vistas"))
    p.add_argument("--out", default="/fp/projects01/ec674/williaan/checkpoints")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    # Use only the first N images of each split, for quick tests. 0 = all.
    p.add_argument("--limit", type=int, default=0)

    # Value is false unless you write --class-weights in CL
    p.add_argument("--class-weights", action="store_true")
    # Which decoder to train. "linear" is week 1, so old commands still do the same.
    p.add_argument("--model", choices=MODELS, default="linear")
    # Image size (width height) the photos are resized to. Both must be multiples of 14.
    # Bigger means more patches, so thin things like poles cover more than one 14x14 patch.
    p.add_argument("--size", type=int, nargs=2, default=[448, 336])
    return p.parse_args()


def evaluate(model, loader, loss_fn, device):
    """Average loss, pixel accuracy and mean IoU on a dataset, without training."""
    model.eval()

    total_loss = 0.0
    # Confusion table as in evaluate.py: [true class, predicted class] -> pixel count
    table = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)

    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            scores = model(images)
            total_loss += loss_fn(scores, masks).item()

            preds = scores.argmax(dim=1) # Highest-scoring class per pixel
            update_table(table, preds, masks)

    # mIoU is used to pick the best epoch: with class weights the val loss
    # is a different number, so "lowest loss" no longer means "best model".
    pixel_acc = (table.diag().sum() / table.sum()).item()
    miou = per_class_iou(table).nanmean().item()
    return total_loss / len(loader), pixel_acc, miou


def main():
    args = parse_args()

    # Same seed -> same random start of the decoder and same shuffling,
    # so a run can be repeated (week 4 needs 3 runs with different seeds).
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)

    # Data
    train_set = VistasClueDataset(args.root, "training", size=tuple(args.size))
    val_set = VistasClueDataset(args.root, "validation", size=tuple(args.size))

    # Quick-test mode: keep only the first args.limit images.
    if args.limit:
        train_set = Subset(train_set, range(args.limit))
        val_set = Subset(val_set, range(args.limit))

    # Training has shuffle=True so each epoch sees the iages in a new order and the batches arent always the same mix
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = MODELS[args.model](num_classes=len(CLUE_CLASSES)).to(device)
    if args.class_weights:
        loss_fn = nn.CrossEntropyLoss(weight=class_weights().to(device), ignore_index=IGNORE_INDEX)
    else:
        loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        
    optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=args.lr)


    os.makedirs(args.out, exist_ok=True)

    # e.g. linear_seed0, segmenter_weighted_seed0. evaluate.slurm reads the model
    # back from the first part of this name.
    name = f"{args.model}_weighted" if args.class_weights else args.model
    # Add the width when it is not the default (e.g. segmenter_672_seed0), so the
    # week 2 checkpoint names stay the same and are not overwritten.
    if args.size != [448, 336]:
        name = f"{name}_{args.size[0]}"
    checkpoint = os.path.join(args.out, f"{name}_seed{args.seed}.pt")
    best_miou = -1.0

    for epoch in range(args.epochs):
        # Train for one epoch
        model.train()

        running_loss = 0.0
        for step, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            scores = model(images)
            loss = loss_fn(scores, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Print progress every 100 steps, so the Slurm log shows it is alive.
            if step % 100 == 0:
                print(f"epoch {epoch} step {step}/{len(train_loader)} loss {loss.item():.4f}", flush=True)

        train_loss = running_loss / len(train_loader)

        # Check on the validation set
        val_loss, val_acc, val_miou = evaluate(model, val_loader, loss_fn, device)

        print(f"epoch {epoch}: train loss {train_loss:.4f}  val loss {val_loss:.4f}  "
              f"val pixel acc {val_acc:.4f}  val mIoU {val_miou:.4f}", flush=True)

        # Save the best decoder from epoch with best mIoU
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.decoder.state_dict(), checkpoint)

    print("best val mIoU:", best_miou, "saved to", checkpoint, flush=True)


if __name__ == "__main__":
    main()

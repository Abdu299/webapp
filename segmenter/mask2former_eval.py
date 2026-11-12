"""
Reference for the week 2 table
pretrained Mask2Former (Swin-L trained on Mapillary Vistas)
Scored on the 10 clue classes with the same code as our models
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from transformers import Mask2FormerForUniversalSegmentation

from clue_classes import IGNORE_INDEX, build_lookup_table, CLUE_CLASSES
from dataset import VistasClueDataset
from evaluate import NUM_CLASSES, confusion_matrix, per_class_iou

CHECKPOINT = "facebook/mask2former-swin-large-mapillary-vistas-semantic"



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=os.environ.get("VISTAS_ROOT", "/fp/projects01/ec674/data/vistas"))

    # Swin-L is bigger than DINOv2-small, so smaller batches
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)

    # Mask2Former was trained on bigger images so we scale ours
    p.add_argument("--scale", type=int, default=2)
    return p.parse_args()


class Mask2FormerClues(nn.Module):
    """ 
    Wrap Mask2Former so its like our models:
    images in [B, 3, H, W], clue class scores out [B, 10, H, W]
    """

    def __init__(self, config_path, scale):
        super().__init__()
        self.scale = scale
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(CHECKPOINT)

        # Week 1 mapping
        lut = build_lookup_table(config_path=config_path)
        num_m2f = len(self.model.config.id2label) # 65 Mask2Former classes

        # Making a translation table of 10 clue classes and the 65 Vistas classes
        # A 1 means Vistas class belongs to that clue class (e.g. pole and street light -> poles)
        to_clue = torch.zeros(NUM_CLASSES, num_m2f)

        for vistas_id in range(num_m2f):
            clue_id = int(lut[vistas_id])
            if clue_id != IGNORE_INDEX:
                to_clue[clue_id, vistas_id] = 1
        
        self.register_buffer("to_clue", to_clue)

    @torch.no_grad()
    def forward(self, images):
        B, _, H, W = images.shape

        # Scale up
        big = F.interpolate(images, scale_factor=self.scale, mode="bilinear", align_corners=False)

        # Run Mask2Former
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=images.is_cuda):
            out = self.model(pixel_values=big)

        # Combine the Q guesses into one score per Vistas class per pixel
        class_probs = out.class_queries_logits.float().softmax(dim=-1)[..., :-1]
        mask_probs = out.masks_queries_logits.float().sigmoid()
        scores = torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)

        # Shrink back to size of our masks
        scores = F.interpolate(scores, size=(H,W), mode="bilinear", align_corners=False)

        return torch.einsum("kc,bchw->bkhw", self.to_clue, scores)

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)

    # Same validation images as our models
    val_set = VistasClueDataset(args.root, "validation")
    if args.limit:
        val_set = Subset(val_set, range(args.limit))
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Wrapper needs Vistas config for translation table
    config_path = os.path.join(args.root, "config_v1.2.json")
    model = Mask2FormerClues(config_path=config_path, scale=args.scale).to(device)

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
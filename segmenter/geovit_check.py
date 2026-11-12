"""
Sanity check for Abdu's GeoVit before using it: country accuracy on a test set, without blur.
Two inputs are compared:
- whole photo squashed to 224x224: what Abdu trained and tested on (should be near his ~70%)
- 448 centre square: what class_blur.py and heatmaps.py give it (so the masks line up)
A big gap means the centre crop hurts the model, and the clue results should mention it.

Run from the segmenter folder (GPU job, or the login node with --limit):
    python geovit_check.py --csv /fp/projects01/ec674/williaan/data/test_sets/osv5m.csv
"""

import argparse

import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from class_blur import centre_square
from geovit import MEAN, STD, load_geovit, prepare_geovit


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--work-size", type=int, default=448)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def squashed(image, device):
    """Abdu's transform: whole photo to 224x224 (aspect ratio not kept), normalised."""
    x = TF.to_tensor(TF.resize(image, [224, 224])).unsqueeze(0).to(device)
    return (x - MEAN.to(device)) / STD.to(device)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)

    df = pd.read_csv(args.csv, dtype={"name": str}, keep_default_na=False)
    if args.limit:
        df = df.head(args.limit)

    model, label_to_country = load_geovit(device)
    print(f"countries the model knows: {len(label_to_country)}")
    known = df["country"].isin(set(label_to_country.values()))
    print(f"photos in a country the model never saw (always wrong): {(~known).sum()}\n")

    rows = []
    with torch.no_grad():
        for photo in df.itertuples():
            image = Image.open(photo.path).convert("RGB")
            square = centre_square(image, args.work_size, TF.InterpolationMode.BICUBIC)
            inputs = {
                "squashed": squashed(image, device),
                "centre square": prepare_geovit(TF.to_tensor(square).unsqueeze(0).to(device)),
            }
            for name, x in inputs.items():
                probs = F.softmax(model(x), dim=1)[0]
                top5 = [label_to_country[i] for i in probs.topk(5).indices.tolist()]
                rows.append({"input": name, "continent": photo.continent,
                             "top1": top5[0] == photo.country, "top5": photo.country in top5})

    rows = pd.DataFrame(rows)
    print("Country accuracy:")
    print(rows.groupby("input")[["top1", "top5"]].mean().to_string(float_format="{:.1%}".format), "\n")
    print("Top-1 accuracy per continent:")
    print(rows.pivot_table(index="continent", columns="input", values="top1")
              .to_string(float_format="{:.1%}".format))


if __name__ == "__main__":
    main()

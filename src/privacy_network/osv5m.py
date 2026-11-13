"""Make predictions using the baseline model.

Run from the repo root:
    python -m src.privacy_network.osv5m
"""

import sys
import math
import torch
import contextlib
import numpy as np
import pandas as pd
from tqdm import tqdm
import importlib

from pathlib import Path
from PIL import Image
from src.utils.common import (
    print_accuracy,
    print_quality,
    save_output,
    save_example,
    haversine,
    radius_km,
)
from src.image_quality.metrics import load_lpips, image_quality
from src.privacy_network.datasets import photo_loader

model_name = "osv5m"
model_dir = (
    Path(__file__).resolve().parents[3] / model_name
)  # the OSV-5M repo, next to this repo
out_dir = Path("output") / model_name
sys.path.insert(0, str(model_dir))





def load_model():

    try:
        module = importlib.import_module("models.huggingface")
        Geolocalizer = module.Geolocalizer

    except ModuleNotFoundError as error:
        raise RuntimeError(
            "The separate OSV-5M baseline repository "
            "is required to load the baseline model."
        ) from error

    with contextlib.chdir(model_dir):
        return (
            Geolocalizer
            .from_pretrained("osv5m/baseline")
            .eval()
            .requires_grad_(False)
        )

class CellLogits(torch.nn.Module):
    """
    PlaNet and OSV5M baseline model divides earth into cells and classifies photos into them.
    Model picks top cell and uses its coords for predicted coords (with argmax) which has no gradient,
    so we return the scores from before that step to give the optimised blur a gradient.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        """Takes (B,3,H,W) images from model.transform and returns (B, num_cells) cell scores.

        backbone: CLIP ViT image encoder
        mid: MLP on the cls token
        head: dictionary with keys "label" (cell scores) and "gps" (centre of top cell + its offset)
        """
        features = self.model.mid(self.model.backbone({"img": x}))
        return self.model.head(features, None)["label"]  # (B, num_cells)


def cell_centres(model):
    """Centre (lat, lon) in degrees of each of the baseline model's cells, shape (num_cells, 2).
    A cell's centre is the average location of training photos in the cell
    """
    centres = model.head.cell_center.float()  # damn americans

    # the head stores cell centres normalised as lat / 90 and lon / 180
    return centres * torch.tensor([90.0, 180.0], device=centres.device)


def predict(model, x):
    """Predicts location of images.
    x: (B,3,H,W) tensor from model.transform
    Returns (B,2) lat, lon in radians.
    """
    return model(x)


def guess(model, df, device):
    # lat, lon in degrees, NaN for unreadable photos
    guesses = np.full((len(df), 2), np.nan)  # (N, 2)
    for batch in tqdm(photo_loader(df, model.transform), desc="osv5m"):
        rows = batch["row"].tolist()
        with torch.no_grad():
            guesses[rows] = np.degrees(
                predict(model, batch["image"].to(device)).cpu().numpy()
            )  # model gives radians
    return guesses


def cells_near_true_location(centres, lat, lon):
    cell_lat = centres[:, 0].unsqueeze(0)  # (1, num_cells)
    cell_lon = centres[:, 1].unsqueeze(0)  # (1, num_cells)

    photo_lat = lat.unsqueeze(1)  # (B, 1)
    photo_lon = lon.unsqueeze(1)  # (B, 1)

    # distance from every photo to every cell
    km = haversine(cell_lat, cell_lon, photo_lat, photo_lon)  # (B, num_cells)
    return km <= radius_km  # (B, num_cells)


def is_hidden(top_guess, lat, lon):
    km = haversine(lat, lon, top_guess[:, 0], top_guess[:, 1])
    return km > radius_km  # (B,)


def judge(
    model,
    x,  # (B, 3, 224, 224)
    lat,  # (B,)
    lon,  # (B,)
):
    tokens = model.backbone({"img": x})  # (B, 257, 1024)
    features = model.mid(tokens)  # mlp on the class token
    out = model.head(features, None)
    logits = out["label"].float()  # (B, num_cells)
    top_guess = torch.rad2deg(out["gps"].float())  # (B, 2) lat, lon in degrees

    # True for cells within radius_km of the true location
    near = cells_near_true_location(cell_centres(model), lat, lon)  # (B, num_cells)

    # True where the top guess is farther than radius_km
    hidden = is_hidden(top_guess, lat, lon)  # (B,)
    return logits, near, hidden


def locate(model, path, lat=None, lon=None):
    """Guesses where a specific photo was taken.

    Input: lat, lon (actual location in degrees)
    Returns guessed lat and lon in degrees, and distance in km to the true location.
        Distance is None without a true location.
    """
    x = model.transform(Image.open(path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        pred_lat, pred_lon = predict(model, x)[0].tolist()

    # model predicts radians, convert to degrees like the true location
    pred_lat, pred_lon = math.degrees(pred_lat), math.degrees(pred_lon)

    km = None
    if lat is not None and lon is not None:
        km = haversine(lat, lon, pred_lat, pred_lon)

    return pred_lat, pred_lon, km


def evaluate(model, images, out_dir, perturb=None, out_name="baseline"):
    """Predicting each image and saving distance and accuracy.

    out_dir: Output folder of the model, like output/osv5m
    perturb: Optional, blurs images if not None
    out_name: Prefix of the result files
    """
    df = (
        pd.read_csv("privacyData/im2gps.csv")
        .set_index("filename")
        .loc[[path.name for path in images]]
        .reset_index()
    )

    lpips = load_lpips()

    print("Making predictions ... ")
    predictions = []
    quality = []
    with torch.no_grad():
        for path in tqdm(images, desc="photos"):
            x = model.transform(Image.open(path).convert("RGB")).unsqueeze(0)

            # Perturb only when applying blur and save the original to measure image quality
            x_blurred = x
            if perturb is not None:
                x_blurred = perturb(x)

            # Save example image + quality. Send transformation to undo normalization
            save_example(x_blurred, model.transform, path, out_dir, out_name)
            quality.append(image_quality(x, x_blurred, model.transform, lpips))
            predictions.append(predict(model, x_blurred)[0].tolist())

    df[["pred_lat", "pred_lon"]] = predictions
    df[["ssim", "lpips"]] = quality
    save_output(df, out_dir, out_name)


if __name__ == "__main__":
    model = load_model()
    images = sorted(Path("privacyData/im2gps").rglob("*.jpg"))

    evaluate(model, images, out_dir)
    print_accuracy(out_dir / "baseline_accuracy.csv")
    print_quality(out_dir / "baseline_quality.csv")

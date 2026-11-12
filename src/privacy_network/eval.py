import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from src.blur.extremal import BlurPyramid
from src.data_prep.privacy_split import load_split
from src.image_quality.metrics import image_quality, load_lpips
from src.privacy_network.datasets import PhotoDataset, collate
from src.privacy_network.model import Model, judge_module
from src.utils.common import haversine, radius_km, unnormalize


def run_inference(model, df, out_dir, config, device):
    transform = model.model.encoder.transform
    lpips = load_lpips()
    loader = DataLoader(
        PhotoDataset(df, transform),
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        collate_fn=collate,
    )

    z = torch.zeros(1, 3, 224, 224, device=device)
    with torch.no_grad():
        BlurPyramid(z).apply(model(z))  # the first call on the GPU is slow, keep it out of the timing

    quality = np.full((len(df), 4), np.nan)  # NaN for unreadable photos
    
    for batch in tqdm(loader, desc="Blurring"):
        x = batch["image"].to(device)  # (B, 3, 224, 224)
        start = time.time()
        
        with torch.no_grad():
            mask = model(x)  # (B, 1, 224, 224)
            x_blurred = BlurPyramid(x).apply(mask)  # (B, 3, 224, 224)
        if device == "cuda":
            torch.cuda.synchronize() 
            
        seconds = (time.time() - start) / len(x)  # per photo
        x, mask, x_blurred = x.cpu(), mask.cpu(), x_blurred.cpu()
        
        for i, row in enumerate(batch["row"].tolist()):
            name = df["name"].iloc[row]
            to_pil_image(unnormalize(x_blurred[i : i + 1], transform)[0]).save(out_dir / "blurred" / f"{name}.png")
            to_pil_image(mask[i]).save(out_dir / "masks" / f"{name}.png")

            ssim, lpips_score = image_quality(x[i : i + 1], x_blurred[i : i + 1], transform, lpips)
            quality[row] = [mask[i].mean().item(), ssim, lpips_score, seconds]

    inference = df[["name"]].copy()
    inference[["blurred_area", "ssim", "lpips", "seconds"]] = quality
    inference.to_csv(out_dir / "inference.csv", index=False)


def score(df, pngs, judge, judge_model, device):
    guesses = judge_module(judge).guess(judge_model, df.assign(path=pngs), device)  # (N, 2) lat, lon in degrees, NaN if unreadable

    df[["pred_lat", "pred_lon"]] = guesses
    df["km"] = haversine(
        torch.tensor(df["lat"].values),
        torch.tensor(df["lon"].values),
        torch.tensor(guesses[:, 0]),
        torch.tensor(guesses[:, 1]),
    ).numpy()
    df["located"] = df[f"{judge}_unblurred_km"] <= radius_km
    df["hidden"] = df["km"] > radius_km
    return df


if __name__ == "__main__":
    ckpt_path, split = Path(sys.argv[1]), sys.argv[2]
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Load model and config
    model = Model.load_from_checkpoint(ckpt_path, map_location=device, strict=False)
    model = model.to(device).eval()
    config = model.hparams.config

    # Make guess with judge
    if len(sys.argv) > 3:
        judge = sys.argv[3]
    else:
        judge = config["name"]

    # Save folders
    out_dir = ckpt_path.with_suffix("") / split  # like <run_dir>/best/test
    (out_dir / "blurred").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)

    if not (out_dir / "inference.csv").exists():
        run_inference(model, load_split(split), out_dir, config, device)
    df = load_split(split, models=[judge]).merge(pd.read_csv(out_dir / "inference.csv", dtype={"name": str}), on="name")

    
    if judge == config["name"]:
        judge_model = model.model.encoder
    else:
        judge_model = judge_module(judge).load_model().to(device)
        
    pngs = []
    for name in df["name"]:
        pngs.append(str(out_dir / "blurred" / f"{name}.png"))
    df = score(df, pngs, judge, judge_model, device)
    df.to_csv(out_dir / f"{judge}_predictions.csv", index=False)

    # Save summary of run
    # one row with the numbers for this run, read by figures.ipynb
    located = df[df["located"]]
    
    summary = pd.DataFrame([{
        "hidden": located["hidden"].mean(),
        "n_located": len(located),
        "acc_750": (df["km"] <= radius_km).mean(),
        "area": df["blurred_area"].mean(),
        "ssim": df["ssim"].mean(),
        "lpips": df["lpips"].mean(),
        "seconds": df["seconds"].mean(),
    }])
    
    summary.to_csv(out_dir / f"{judge}_summary.csv", index=False)
    print(summary.T.to_string(header=False))
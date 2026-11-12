import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image

from src.blur.extremal import BlurPyramid
from src.blur.optimised import areas
from src.image_quality.metrics import image_quality, load_lpips
from src.privacy_network.osv5m import load_model, predict, out_dir
from src.utils.common import save_example, save_output, unnormalize


def load_heat(heat_dir, name):
    """Heatmap of one photo, resized from 448 to 224."""
    heat = torch.from_numpy(np.load(heat_dir / f"{name}.npy")).float()  # (448, 448)
    return F.interpolate(heat[None, None], size=224, mode="area")  # (1, 1, 224, 224)


def heatmap_blur(model, heat_dir, out_name, device):
    """Blurs each photo where its heatmap is hottest, for every area, and saves output like the other methods."""
    df = pd.read_csv("privacyData/im2gps.csv")
    lpips = load_lpips()

    output = {}  # per area: pred_lat, pred_lon, ssim, lpips, blurred_area
    for a in areas:
        output[a] = []

    with torch.no_grad():
        for filename in tqdm(df["filename"], desc="Photos"):
            path = Path("privacyData/im2gps") / filename
            x = model.transform(Image.open(path).convert("RGB")).unsqueeze(0)
            heat = load_heat(heat_dir, path.stem)
            pyramid = BlurPyramid(x.to(device))
            blurred_dir = out_dir / "blurred" / out_name / path.stem  # every blurred photo, for src/blur/judge.py
            blurred_dir.mkdir(parents=True, exist_ok=True)

            for a in areas:
                q = heat.quantile(1 - a).clamp(min=1e-6)  # above 0 when less than a of the pixels have heat
                mask = (heat / q).clamp(max=1)  # top a pixels fully blurred, the rest in proportion to heat
                x_blurred = pyramid.apply(mask.to(device))
                pred_lat, pred_lon = predict(model, x_blurred)[0].tolist()

                x_blurred = x_blurred.cpu()
                to_pil_image(unnormalize(x_blurred, model.transform)[0]).save(blurred_dir / f"{a}.png")
                ssim, lpips_score = image_quality(x, x_blurred, model.transform, lpips)
                output[a].append([pred_lat, pred_lon, ssim, lpips_score, mask.mean().item()])
                save_example(x_blurred, model.transform, path, out_dir, f"{out_name}_{a}")

    for a in areas:
        df[["pred_lat", "pred_lon", "ssim", "lpips", "blurred_area"]] = output[a]
        save_output(df, out_dir, f"{out_name}_{a}")


if __name__ == "__main__":
    heat_model = "osv5m"
    if len(sys.argv) > 1:
        heat_model = sys.argv[1]
    heat_dir = Path("privacyData/heatmaps") / heat_model / "im2gps"  # from William's segmenter/heatmaps.py, (448, 448) per photo
    out_name = f"heatmap_{heat_model}"

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    model = load_model().to(device)

    heatmap_blur(model, heat_dir, out_name, device)

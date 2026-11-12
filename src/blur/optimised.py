"""Optimised blur (Fong et al.) on the Im2GPS test photos.

Run from the repo root:
    python -m src.blur.optimised [iters] [examples | number of photos]
"""

import sys
import math
import time
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from src.privacy_network.osv5m import (
    load_model,
    CellLogits,
    cell_centres,
    predict,
    out_dir,
)
from src.blur.extremal import extremal_blur
from src.utils.common import save_output, haversine, unnormalize, example_images
from src.image_quality.metrics import load_lpips, image_quality


areas = (0.05, 0.1, 0.2, 0.4, 0.6, 0.8)


def evaluate_optimised(
    model, images, device, out_dir, iters=1600, out_name="optimised"
):
    """Optimised blur of each photo, saving output per area, a* and time per photo.

    output per area are saved like the other methods, with out_name_area as prefix.
    a*: smallest area whose mask reached its target (within 0.02) and moved the guess beyond 750 km.
    Also saves every mask (masks/<photo>.pt) and every blurred photo (blurred/<out_name>/<photo>/<area>.png).
    """
    # coordinates of the given photos, in the same order as images
    df = (
        pd.read_csv("privacyData/im2gps.csv")
        .set_index("filename")
        .loc[[path.name for path in images]]
        .reset_index()
    )
    lpips = load_lpips()
    cell_model = CellLogits(model)
    centres = cell_centres(model)

    output = {
        area: [] for area in areas
    }  # per area: pred_lat, pred_lon, ssim, lpips, blurred_area
    a_stars = []
    seconds = []
    for path, lat, lon in tqdm(
        zip(images, df["lat"], df["lon"]), total=len(images), desc="Photos"
    ):
        x = model.transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)

        start = time.time()
        blurred, _ = extremal_blur(
            cell_model, x, (lat, lon), centres, areas=areas, iters=iters
        )
        seconds.append(time.time() - start)

        # all masks of this photo in one file, shape (areas, 1, H, W), in the order of areas
        (out_dir / "masks").mkdir(parents=True, exist_ok=True)
        torch.save(
            torch.cat([blurred[area][1] for area in areas]).half().cpu(),
            out_dir / "masks" / f"{path.stem}.pt",
        )
        blurred_dir = out_dir / "blurred" / out_name / path.stem
        blurred_dir.mkdir(parents=True, exist_ok=True)

        # for the example photos: original, blurred photos and masks, in a folder per photo
        is_example = path.name in example_images
        example_dir = out_dir / "examples" / out_name / path.stem
        if is_example:
            example_dir.mkdir(parents=True, exist_ok=True)
            to_pil_image(unnormalize(x.cpu(), model.transform)[0]).save(
                example_dir / "original.png"
            )

        a_star = None
        for area in areas:
            x_blurred, mask, _ = blurred[area]
            blurred_area = mask.mean().item()

            # guess with predict, like the other methods
            with torch.no_grad():
                pred_lat, pred_lon = predict(model, x_blurred)[0].tolist()
            km = haversine(lat, lon, math.degrees(pred_lat), math.degrees(pred_lon))

            if a_star is None and abs(blurred_area - area) < 0.02 and km > 750:
                a_star = area

            ssim, lpips_score = image_quality(
                x.cpu(), x_blurred.cpu(), model.transform, lpips
            )
            output[area].append([pred_lat, pred_lon, ssim, lpips_score, blurred_area])
            to_pil_image(unnormalize(x_blurred.cpu(), model.transform)[0]).save(
                blurred_dir / f"{area}.png"
            )

            if is_example:
                to_pil_image(unnormalize(x_blurred.cpu(), model.transform)[0]).save(
                    example_dir / f"blurred_{area}.png"
                )
                to_pil_image(mask[0].cpu()).save(example_dir / f"mask_{area}.png")

        a_stars.append(a_star)

    for area in areas:
        df[["pred_lat", "pred_lon", "ssim", "lpips", "blurred_area"]] = output[area]
        save_output(df, out_dir, f"{out_name}_{area}")

    df["a_star"] = a_stars
    df["seconds"] = seconds
    df[["filename", "lat", "lon", "a_star", "seconds"]].to_csv(
        out_dir / f"{out_name}_a_star.csv", index=False
    )


if __name__ == "__main__":
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 1600  # fewer for testing

    images = sorted(Path("privacyData/im2gps").rglob("*.jpg"))
    if len(sys.argv) > 2 and sys.argv[2] == "examples":
        images = [
            path for path in images if path.name in example_images
        ]  # the six example photos
    elif len(sys.argv) > 2:
        images = images[: int(sys.argv[2])]  # the first photos, for testing

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model().to(device)

    evaluate_optimised(model, images, device, out_dir, iters)


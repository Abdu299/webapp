"""Optimised blur and location guess for a single photo.

Run from the repo root:
    python -m src.blur.single_photo path/to/photo.jpg
"""

import sys
import math
import torch
from pathlib import Path
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from src.privacy_network.osv5m import load_model, CellLogits, cell_centres, predict
from src.blur.extremal import extremal_blur
from src.utils.common import haversine, unnormalize


def guess(model, x):
    """Guessed (lat, lon) in degrees for a (1,3,H,W) tensor from model.transform."""
    with torch.no_grad():
        lat, lon = predict(model, x)[0].tolist()
    return math.degrees(lat), math.degrees(lon)


def protect(model, path, area=0.1, iters=1600, true_coords=None, device="cpu"):
    """Blurs a fraction area of the photo so that the guess moves beyond 750 km.

    true_coords: (lat, lon) in degrees; if None, the model's own guess is the location to hide.
    Returns guess before and after blur (degrees), km moved, blurred photo and mask as PIL images.
    """
    x = model.transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
    before = guess(model, x)
    if true_coords is None:
        true_coords = before

    result, _ = extremal_blur(
        CellLogits(model),
        x,
        true_coords,
        cell_centres(model),
        areas=(area,),
        iters=iters,
    )
    x_blurred, mask, _ = result[area]

    after = guess(model, x_blurred)
    km = haversine(true_coords[0], true_coords[1], after[0], after[1])
    blurred = to_pil_image(unnormalize(x_blurred.cpu(), model.transform)[0])
    return before, after, km, blurred, to_pil_image(mask[0].cpu())


if __name__ == "__main__":
    photo_path = Path(sys.argv[1])
    iters = 1600  # use fewer for testing

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model().to(device)

    before, after, km, blurred, mask = protect(
        model, photo_path, iters=iters, device=device
    )
    print(f"Guess before blur: {before[0]:.2f}, {before[1]:.2f}")
    print(
        f"Guess after blur:  {after[0]:.2f}, {after[1]:.2f} ({km:.0f} km from the hidden location)"
    )

    Path("output").mkdir(exist_ok=True)
    blurred.save(f"output/{photo_path.stem}_blurred.png")
    mask.save(f"output/{photo_path.stem}_mask.png")


"""Full blur.

Run from the repo root:
    python -m src.blur.full
"""

import math
from pathlib import Path
from torchvision.transforms.functional import gaussian_blur, to_pil_image
from tqdm import tqdm
from PIL import Image


from src.privacy_network.osv5m import evaluate, load_model, out_dir
from src.utils.common import print_accuracy, print_quality, unnormalize


def full_blur(x, sigma=20):
    """Gaussian blur of the whole image, see 3.4

    x: image
    sigma: blur strength
    """
    if sigma <= 0:  # cant blur at blur strength 0
        raise ValueError("Sigma value/blur strength must be larger than 0.")

    kernel_size = 2 * math.ceil(3 * sigma) + 1
    return gaussian_blur(
        gaussian_blur(x, [kernel_size, 1], sigma), [1, kernel_size], sigma
    )


def save_blurred(model, images, blurred_dir):
    """Saves every photo with full blur as png in blurred_dir."""
    blurred_dir.mkdir(parents=True, exist_ok=True)

    for path in tqdm(images, desc="Saving blurred photos"):
        x = model.transform(Image.open(path).convert("RGB")).unsqueeze(0)
        to_pil_image(unnormalize(full_blur(x), model.transform)[0]).save(
            blurred_dir / f"{path.stem}.png"
        )


if __name__ == "__main__":
    model = load_model()
    images = sorted(Path("privacyData/im2gps").rglob("*.jpg"))

    evaluate(model, images, out_dir, perturb=full_blur, out_name="full_sigma20")
    print_accuracy(out_dir / "full_sigma20_accuracy.csv")
    print_quality(out_dir / "full_sigma20_quality.csv")

    save_blurred(model, images, out_dir / "blurred" / "full_sigma20")


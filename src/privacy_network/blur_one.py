import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from src.blur.extremal import BlurPyramid
from src.privacy_network.model import Model
from src.utils.common import unnormalize


if __name__ == "__main__":
    ckpt_path, image_path = Path(sys.argv[1]), Path(sys.argv[2])
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model = Model.load_from_checkpoint(ckpt_path, map_location=device, strict=False)
    model = model.to(device).eval()
    transform = model.model.encoder.transform

    x = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)  # (1, 3, 224, 224)
    with torch.no_grad():
        mask = model(x)  # (1, 1, 224, 224)
        x_blurred = BlurPyramid(x).apply(mask)  # (1, 3, 224, 224)

    # Save next to the eval outputs, like <run_dir>/best/single
    out_dir = ckpt_path.with_suffix("") / "single"
    out_dir.mkdir(parents=True, exist_ok=True)
    to_pil_image(unnormalize(x_blurred.cpu(), transform)[0]).save(out_dir / f"{image_path.stem}_blurred.png")
    to_pil_image(mask[0].cpu()).save(out_dir / f"{image_path.stem}_mask.png")
    print(f"Blurred area {mask.mean().item():.2f}, saved to {out_dir}")

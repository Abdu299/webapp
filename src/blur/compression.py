import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from src.data_prep.privacy_split import load_split
from src.privacy_network.eval import score
from src.privacy_network.model import judge_module
from src.privacy_network.osv5m import out_dir


def png(image, path):
    image.save(path)
    return path


def jpeg(image, path):
    path = path.with_suffix(".jpg")
    image.save(path, quality=75)
    return path


def screenshot(image, path):
    image.resize((image.width // 2, image.height // 2), Image.BICUBIC).save(path)  # (112, 112)
    return path


conditions = {"png": png, "jpeg": jpeg, "screenshot": screenshot}


def degrade(pngs, condition, folder):
    paths = []
    for i, path in enumerate(pngs):
        paths.append(condition(Image.open(path).convert("RGB"), folder / f"{i}.png"))
    return paths


if __name__ == "__main__":
    judge = sys.argv[1]
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    df = load_split("im2gps", models=[judge])
    model = judge_module(judge).load_model().to(device)

    methods = {
        "full_sigma20": str(out_dir / "blurred" / "full_sigma20" / "{name}.png"),
        f"heatmap_{judge}_0.1": str(out_dir / "blurred" / f"heatmap_{judge}" / "{name}" / "0.1.png"),
        "network": str(Path("output") / judge / "privacy_network" / f"{judge}_seed0" / "best" / "im2gps" / "blurred" / "{name}.png"),
    }
    if judge == "osv5m":
        methods["optimised_0.1"] = str(out_dir / "blurred" / "optimised" / "{name}" / "0.1.png")

    rows = []
    for method, pattern in methods.items():
        pngs = []
        for name in df["name"]:
            pngs.append(pattern.format(name=name))
        for condition, save in conditions.items():
            with tempfile.TemporaryDirectory() as folder:
                scored = score(df.copy(), degrade(pngs, save, Path(folder)), judge, model, device)
            hidden = scored.loc[scored["located"], "hidden"].mean()
            print(f"{method} {condition}: {hidden:.3f} hidden from {judge} (N = {scored['located'].sum()})")
            rows.append([method, condition, hidden, scored["located"].sum()])

    pd.DataFrame(rows, columns=["method", "condition", "hidden", "n_located"]).to_csv(out_dir / f"compression_im2gps_{judge}.csv", index=False)

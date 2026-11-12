import sys
from pathlib import Path

import torch

from src.blur.full import save_blurred
from src.blur.optimised import areas
from src.data_prep.privacy_split import load_split
from src.privacy_network.eval import score
from src.privacy_network.model import judge_module
from src.privacy_network.osv5m import load_model, out_dir
from src.utils.common import radius_km


if __name__ == "__main__":
    judge, split = sys.argv[1], sys.argv[2]
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    df = load_split(split, models=[judge])

    if split == "im2gps":
        full_dir = out_dir / "blurred" / "full_sigma20"
    else:
        full_dir = out_dir / "blurred" / f"full_sigma20_{split}"
        if not full_dir.exists():
            images = []
            for path in df["path"]:
                images.append(Path(path))
            save_blurred(load_model(), images, full_dir)

    methods = {"full_sigma20": str(full_dir / "{name}.png")}

    if split == "im2gps":
        for a in areas:
            methods[f"optimised_{a}"] = str(out_dir / "blurred" / "optimised" / "{name}" / f"{a}.png")
            methods[f"heatmap_osv5m_{a}"] = str(out_dir / "blurred" / "heatmap_osv5m" / "{name}" / f"{a}.png")
            methods[f"heatmap_geovit_{a}"] = str(out_dir / "blurred" / "heatmap_geovit" / "{name}" / f"{a}.png")

    model = judge_module(judge).load_model().to(device)
    result = df[["name"]].copy()
    result["located"] = df[f"{judge}_unblurred_km"] <= radius_km

    for method, pattern in methods.items():
        pngs = []
        for name in df["name"]:
            pngs.append(pattern.format(name=name))

        scored = score(df.copy(), pngs, judge, model, device)
        result[method] = scored["km"]

        hidden = scored.loc[scored["located"], "hidden"].mean()
        print(f"{method}: {hidden:.3f} hidden from {judge} (N = {result['located'].sum()})")

    result.to_csv(out_dir / f"baselines_{split}_{judge}.csv", index=False)
